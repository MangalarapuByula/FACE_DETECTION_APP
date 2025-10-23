const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const confidenceText = document.getElementById('confidence');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => { video.srcObject = stream; })
.catch(err => { console.error(err); });

// Send frames to backend every 500ms
setInterval(async () => {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg');
    const blob = await (await fetch(dataUrl)).blob();
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
        const response = await fetch("/detect_face", { method: "POST", body: formData });
        const result = await response.json();

        // Show confidence
        confidenceText.innerText = `Confidence: ${result.confidence}%`;

        // Show updated frame
        const img = new Image();
        img.src = "data:image/jpeg;base64," + result.image;
        img.onload = () => { ctx.drawImage(img, 0, 0, canvas.width, canvas.height); };
    } catch (err) {
        console.error("Error detecting face:", err);
    }
}, 500);

