from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)  # allow frontend from another domain to access backend

registered_encoding = None

@app.route('/')
def home():
    return jsonify({"message": "Backend Running!"})

@app.route('/register', methods=['POST'])
def register_face():
    global registered_encoding
    data = request.get_json()
    image_data = data['image']
    image_bytes = base64.b64decode(image_data.split(',')[1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    encodings = face_recognition.face_encodings(img)
    if len(encodings) == 1:
        registered_encoding = encodings[0]
        return jsonify({"message": "✅ Face registered successfully!"})
    else:
        return jsonify({"error": "⚠️ Ensure only ONE face is visible."}), 400

@app.route('/detect', methods=['POST'])
def detect_face():
    global registered_encoding
    if registered_encoding is None:
        return jsonify({"error": "⚠️ Please register your face first."}), 400

    data = request.get_json()
    image_data = data['image']
    image_bytes = base64.b64decode(image_data.split(',')[1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    encodings = face_recognition.face_encodings(img)
    response = {"faces_detected": len(encodings), "confidence": 0, "status": "No Face"}

    if len(encodings) == 0:
        response["status"] = "No Face Detected"
    elif len(encodings) > 1:
        response["status"] = "Multiple Faces Detected"
        response["confidence"] = 40
    else:
        distance = face_recognition.face_distance([registered_encoding], encodings[0])[0]
        confidence = max(0, 100 - distance * 120)
        if distance < 0.45:
            response["status"] = "Authorized Face ✅"
            response["confidence"] = confidence
        else:
            response["status"] = "Unauthorized Face ❌"
            response["confidence"] = 30

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
