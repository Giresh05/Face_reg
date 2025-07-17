from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import os
import mediapipe as mp

app = Flask(__name__)
ENROLL_PATH = "face_embeddings.npz"
PB_MODEL_PATH = "model.pb"  # Your frozen .pb model

# ----------- Load PB Model -----------
def load_graph():
    graph = tf.Graph()
    with tf.io.gfile.GFile(PB_MODEL_PATH, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return graph

graph = load_graph()
sess = tf.compat.v1.Session(graph=graph)

input_tensor = graph.get_tensor_by_name("input:0")          # CHANGE if yours is different
output_tensor = graph.get_tensor_by_name("embeddings:0")    # CHANGE if yours is different

# ----------- Mediapipe for Blink Detection -----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, left_ids, right_ids):
    def eye_aspect_ratio(eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        return (A + B) / (2.0 * C)

    left_eye = np.array([landmarks[i] for i in left_ids])
    right_eye = np.array([landmarks[i] for i in right_ids])
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

def extract_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        return [(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in results.multi_face_landmarks[0].landmark]
    return None

def detect_blink(images):
    ears = []
    for img in images:
        landmarks = extract_landmarks(img)
        if not landmarks:
            continue
        ear = calculate_ear(landmarks, LEFT_EYE_IDX, RIGHT_EYE_IDX)
        ears.append(ear)
    return min(ears) < 0.20 and (max(ears) - min(ears)) > 0.10

# ----------- Embedding Extractor Using PB Model -----------
def preprocess(img):
    face = cv2.resize(img, (160, 160))   # Adjust if your model input size differs
    face = face.astype(np.float32) / 255.0
    return np.expand_dims(face, axis=0)

def get_embedding(img):
    face_input = preprocess(img)
    embedding = sess.run(output_tensor, feed_dict={input_tensor: face_input})
    return embedding[0]

# ----------- Flask Routes -----------
@app.route('/enroll', methods=['POST'])
def enroll():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    embedding = get_embedding(img)
    np.savez_compressed(ENROLL_PATH, embedding=embedding)
    return jsonify({'status': 'Enrolled'})

@app.route('/verify', methods=['POST'])
def verify():
    files = request.files.getlist('images')
    images = [cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR) for file in files]

    if not detect_blink(images):
        return jsonify({'status': 'Failed', 'reason': 'No blink detected'}), 403

    try:
        data = np.load(ENROLL_PATH)
        enrolled = data['embedding']
    except:
        return jsonify({'error': 'No enrolled face found'}), 400

    test_embedding = get_embedding(images[0])  # Pick first image for verification

    # Cosine similarity
    similarity = np.dot(enrolled, test_embedding) / (np.linalg.norm(enrolled) * np.linalg.norm(test_embedding))
    status = 'Success' if similarity > 0.6 else 'Failed'
    return jsonify({'status': status, 'similarity': float(similarity)})

if __name__ == '__main__':
    app.run(debug=True)
