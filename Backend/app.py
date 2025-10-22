from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import numpy as np

app = Flask(__name__)

# Try to import face_recognition; if unavailable, use stub functions
try:
    import face_recognition
    FACE_LIB_AVAILABLE = True
except Exception as e:
    print("face_recognition not available locally:", e)
    FACE_LIB_AVAILABLE = False

def load_image_from_base64(data_url):
    # data_url expected 'data:image/jpeg;base64,...'
    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    img_bytes = base64.b64decode(b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

@app.route("/register_face", methods=["POST"])
def register_face():
    if not FACE_LIB_AVAILABLE:
        return jsonify({"status": "error", "message": "face_recognition not available locally"}), 501
    data = request.json
    image_b64 = data.get("image")
    img = load_image_from_base64(image_b64)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        return jsonify({"status": "success", "encoding": encodings[0].tolist()})
    return jsonify({"status": "fail", "message": "No face found"}), 400

@app.route("/detect_face", methods=["POST"])
def detect_face():
    if not FACE_LIB_AVAILABLE:
        return jsonify({"status": "error", "message": "face_recognition not available locally"}), 501
    data = request.json
    image_b64 = data.get("image")
    img = load_image_from_base64(image_b64)
    locations = face_recognition.face_locations(img)
    return jsonify({"face_count": len(locations), "locations": locations})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "face_recognition_installed": FACE_LIB_AVAILABLE})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
