# from fastapi import FastAPI, Request, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# import cv2
# import numpy as np
# from facenet_pytorch import MTCNN
# import torch
# import os
# from datetime import datetime
# import base64

# app = FastAPI()

# # Base directories
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FRONTEND_DIR = os.path.join(BASE_DIR, "../Frontend")
# STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
# TEMPLATES_DIR = os.path.join(FRONTEND_DIR, "templates")
# CAPTURED_DIR = os.path.join(BASE_DIR, "captured")

# # Ensure captured folder exists
# os.makedirs(CAPTURED_DIR, exist_ok=True)

# # Set up templates and static folders
# templates = Jinja2Templates(directory=TEMPLATES_DIR)
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# # Initialize MTCNN
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# mtcnn = MTCNN(keep_all=True, device=device)

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/detect_face")
# async def detect_face(file: UploadFile = File(...)):
#     contents = await file.read()
#     npimg = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#     boxes, _ = mtcnn.detect(frame)
#     face_count = len(boxes) if boxes is not None else 0

#     if boxes is not None:
#         for box in boxes:
#             x1, y1, x2, y2 = [int(b) for b in box]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     confidence = 100 if face_count == 1 else 50 if face_count > 1 else 0

#     if face_count > 0:
#         filename = os.path.join(CAPTURED_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
#         cv2.imwrite(filename, frame)

#     # Encode frame as base64 string
#     _, buffer = cv2.imencode('.jpg', frame)
#     img_str = base64.b64encode(buffer).decode('utf-8')

#     return {"image": img_str, "confidence": confidence}

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import os
from datetime import datetime
import base64
import mediapipe as mp

app = FastAPI()

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../Frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
TEMPLATES_DIR = os.path.join(FRONTEND_DIR, "templates")
CAPTURED_DIR = os.path.join(BASE_DIR, "captured")

# Ensure captured folder exists
os.makedirs(CAPTURED_DIR, exist_ok=True)

# Set up templates and static folders
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize MTCNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect_face")
async def detect_face(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MTCNN face detection
    boxes, _ = mtcnn.detect(frame)
    face_count = len(boxes) if boxes is not None else 0

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop the detected face for MediaPipe
            face_crop = rgb_frame[y1:y2, x1:x2]
            results = face_mesh.process(face_crop)

            # Draw facial landmarks
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for lm in face_landmarks.landmark:
                        lx = int(lm.x * (x2 - x1)) + x1
                        ly = int(lm.y * (y2 - y1)) + y1
                        cv2.circle(frame, (lx, ly), 1, (0, 0, 255), -1)

    # Confidence logic
    confidence = 100 if face_count == 1 else 50 if face_count > 1 else 0

    # Save captured image
    if face_count > 0:
        filename = os.path.join(CAPTURED_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, frame)

    # Encode frame as base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return {"image": img_str, "confidence": confidence}
