from flask import Flask, request, jsonify
import numpy as np
import cv2
import face_recognition
import os
import mediapipe as mp

app = Flask(__name__)

ENROLL_PATH = "face_embeddings.npz"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def calculate_ear(landmarks, left_ids, right_ids):
    # Use just 1 eye, but you can average both for more robust check
    def eye_aspect_ratio(eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        return (A + B) / (2.0 * C)

    left_eye = np.array([landmarks[i] for i in left_ids])
    right_eye = np.array([landmarks[i] for i in right_ids])
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

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
    # Blink = sudden dip in EAR and rise
    return min(ears) < 0.20 and (max(ears) - min(ears)) > 0.10

@app.route('/enroll', methods=['POST'])
def enroll():
    file = request.files['image']
    img = face_recognition.load_image_file(file)
    encoding = face_recognition.face_encodings(img)
    if not encoding:
        return jsonify({'error': 'No face found'}), 400
    np.savez_compressed(ENROLL_PATH, embedding=encoding[0])
    return jsonify({'status': 'Enrolled successfully'})

@app.route('/verify', methods=['POST'])
def verify():
    files = request.files.getlist('images')
    images = [cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR) for file in files]

    if not detect_blink(images):
        return jsonify({'status': 'Failed', 'reason': 'Liveness check failed'}), 403

    try:
        data = np.load(ENROLL_PATH)
        enrolled_embedding = data['embedding']
    except:
        return jsonify({'error': 'No enrolled face'}), 400

    test_encoding = face_recognition.face_encodings(images[0])
    if not test_encoding:
        return jsonify({'error': 'No face in test image'}), 400

    similarity = np.dot(enrolled_embedding, test_encoding[0]) / (
        np.linalg.norm(enrolled_embedding) * np.linalg.norm(test_encoding[0])
    )

    return jsonify({'status': 'Success' if similarity > 0.6 else 'Failed', 'similarity': float(similarity)})

if __name__ == '__main__':
    app.run(debug=True)
