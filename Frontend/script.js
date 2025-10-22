// Set this after you deploy to Render. While testing locally, you can set to "http://127.0.0.1:5000"
const BACKEND_URL = "http://127.0.0.1:5000";

const video = document.getElementById("video");
const logEl = document.getElementById("log");

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    log("Camera error: " + err.message);
  }
}

function log(msg){
  logEl.textContent = `${new Date().toLocaleTimeString()} — ${msg}\n` + logEl.textContent;
}

function captureFrame() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return canvas.toDataURL("image/jpeg");
}

async function postImage(endpoint, dataURL) {
  const res = await fetch(`${BACKEND_URL}/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataURL })
  });
  return res.json();
}

document.getElementById("registerBtn").addEventListener("click", async () => {
  log("Capturing for register...");
  const dataURL = captureFrame();
  try {
    const res = await postImage("register_face", dataURL);
    log("Register result: " + JSON.stringify(res));
  } catch (e) {
    log("Register failed: " + e.message);
  }
});

document.getElementById("detectBtn").addEventListener("click", async () => {
  log("Capturing for detect...");
  const dataURL = captureFrame();
  try {
    const res = await postImage("detect_face", dataURL);
    log("Detect result: " + JSON.stringify(res));
  } catch (e) {
    log("Detect failed: " + e.message);
  }
});

initCamera();
