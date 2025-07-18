import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, jsonify
import logging

# --- Configuration for Render Deployment ---
# The /app/data directory will be a persistent disk on Render
PERSISTENT_DATA_PATH = '/app/data'
EMBEDDINGS_DIR = os.path.join(PERSISTENT_DATA_PATH, 'embeddings')
MODEL_PATH = os.path.join(PERSISTENT_DATA_PATH, 'MobileFaceNet_9925_9680.pb')
UPLOAD_FOLDER = 'uploads' # This can remain temporary

# --- Model & Verification Configuration ---
INPUT_NODE = 'img_inputs:0'
OUTPUT_NODE = 'embeddings:0'
VERIFICATION_THRESHOLD = 1.0  # L2 Distance threshold

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
logging.basicConfig(level=logging.INFO)

# --- Create necessary directories on startup ---
# This ensures the directories exist on the persistent disk
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(EMBEDDINGS_DIR):
    os.makedirs(EMBEDDINGS_DIR)

# --- Load TensorFlow Graph ---
def load_graph(frozen_graph_filename):
    """Loads a frozen TensorFlow model into memory."""
    if not os.path.exists(frozen_graph_filename):
        # Log a critical error if the model file is not found
        app.logger.critical(f"FATAL ERROR: Model file not found at {frozen_graph_filename}")
        # In a real-world scenario, you might want to exit or prevent the app from starting
        return None
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

graph = load_graph(MODEL_PATH)
# Only create a session if the graph was loaded successfully
if graph:
    sess = tf.Session(graph=graph)
    image_tensor = graph.get_tensor_by_name(INPUT_NODE)
    embedding_tensor = graph.get_tensor_by_name(OUTPUT_NODE)
    app.logger.info("✅ TensorFlow model and session loaded.")
else:
    sess = None
    app.logger.error("Could not initialize TensorFlow session because the model failed to load.")


def preprocess_image(image_bytes):
    """Decodes and preprocesses the image for the model."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. It might be corrupted or in an unsupported format.")
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# --- API Endpoints ---

@app.route('/')
def health_check():
    """
    Health check endpoint for Render.
    Render will call this path to ensure the service is alive.
    """
    # Check if the model is loaded as a basic health indicator
    status = "OK" if sess else "Model not loaded"
    return jsonify({'status': status}), 200 if sess else 503

@app.route('/enroll', methods=['POST'])
def enroll_face():
    """
    Enrolls a face by saving its embedding.
    Expects 'image' file and 'user_id' in the form data.
    """
    if not sess:
        return jsonify({'error': 'Server is not ready, model not loaded.'}), 503

    if 'image' not in request.files or 'user_id' not in request.form:
        return jsonify({'error': 'Missing image or user_id'}), 400

    file = request.files['image']
    user_id = request.form['user_id']
    app.logger.info(f"--- Enrolling user: {user_id} ---")
    
    try:
        image_data = preprocess_image(file.read())
        embedding = sess.run(embedding_tensor, feed_dict={image_tensor: image_data})
        
        embedding_path = os.path.join(EMBEDDINGS_DIR, f'{user_id}.npz')
        np.savez_compressed(embedding_path, embedding=embedding.flatten())
        
        app.logger.info(f"✅ User {user_id} enrolled successfully.")
        return jsonify({'message': f'User {user_id} enrolled successfully.'}), 200
    except Exception as e:
        app.logger.error(f"Enrollment error for {user_id}: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/verify', methods=['POST'])
def verify_face():
    """
    Verifies a face against an enrolled embedding.
    Expects 5 'image' files and 'user_id' in the form data.
    """
    if not sess:
        return jsonify({'error': 'Server is not ready, model not loaded.'}), 503

    if 'user_id' not in request.form:
        return jsonify({'error': 'Missing user_id'}), 400
        
    user_id = request.form['user_id']
    files = request.files.getlist('image')
    app.logger.info(f"--- Verifying user: {user_id} with {len(files)} images ---")

    if len(files) != 5:
        return jsonify({'error': 'Please provide exactly 5 images for verification.'}), 400

    enrolled_embedding_path = os.path.join(EMBEDDINGS_DIR, f'{user_id}.npz')
    if not os.path.exists(enrolled_embedding_path):
        return jsonify({'error': 'User not enrolled.'}), 404

    try:
        enrolled_data = np.load(enrolled_embedding_path)
        enrolled_embedding = enrolled_data['embedding']
        app.logger.info(f"Loaded enrolled embedding for {user_id}. Shape: {enrolled_embedding.shape}")

        burst_embeddings = []
        for file in files:
            image_data = preprocess_image(file.read())
            embedding = sess.run(embedding_tensor, feed_dict={image_tensor: image_data})
            burst_embeddings.append(embedding.flatten())

        app.logger.info("Successfully processed burst of 5 images.")

        # --- Basic Liveness Check ---
        is_live = False
        for i in range(len(burst_embeddings) - 1):
            if not np.array_equal(burst_embeddings[i], burst_embeddings[i+1]):
                is_live = True
                break
        
        if not is_live:
             app.logger.warning(f"Liveness check failed for {user_id}.")
             return jsonify({'verified': False, 'message': 'Liveness check failed (images are identical).'}), 200

        # --- Verification Logic ---
        avg_burst_embedding = np.mean(burst_embeddings, axis=0)
        
        distance = np.linalg.norm(enrolled_embedding - avg_burst_embedding)
        
        verified = distance < VERIFICATION_THRESHOLD
        app.logger.info(f"Verification for {user_id}: Distance={distance:.4f}, Verified={verified}")

        return jsonify({
            'verified': bool(verified),
            'distance': float(distance),
            'threshold': VERIFICATION_THRESHOLD,
            'liveness_check': 'Passed'
        }), 200

    except Exception as e:
        app.logger.error(f"Verification error for {user_id}: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # This block is for local development only.
    # Gunicorn will be used in production as defined in render.yaml.
    app.run(host='0.0.0.0', port=5000, debug=True)
