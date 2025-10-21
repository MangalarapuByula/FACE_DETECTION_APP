const BACKEND_URL = "http://localhost:5000"; // change to Render backend URL later

const video = document.getElementById("video");
const statusText = document.getElementById("status");

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

function captureImage() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return canvas.toDataURL("image/jpeg");
}

async function registerFace() {
  const image = captureImage();
  const res = await fetch(`${BACKEND_URL}/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image }),
  });
  const data = await res.json();
  statusText.innerText = data.message || data.error;
}

async function detectFace() {
  const image = captureImage();
  const res = await fetch(`${BACKEND_URL}/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image }),
  });
  const data = await res.json();
  statusText.innerText = `${data.status} (Confidence: ${data.confidence?.toFixed(2)}%)`;
}
