import eventlet
eventlet.monkey_patch()  # 🔥 CRITICAL: Must be the very first thing in the file

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Initialize WebSockets with eventlet for real-time streaming
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Load model globally
model = YOLO("webapp.pt")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
        
    file = request.files["image"]
    img_bytes = file.read()

    # Convert bytes to opencv image
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Run Inference
    results = model(
        img,
        conf=0.20,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    detections = []
    counts = {"person": 0, "knife": 0, "weapon": 0, "fire": 0}
    speed = {"preprocess": 0, "inference": 0, "postprocess": 0}
    
    # Process Results
    for r in results:
        speed = {
            "preprocess": round(r.speed.get("preprocess", 0), 1),
            "inference": round(r.speed.get("inference", 0), 1),
            "postprocess": round(r.speed.get("postprocess", 0), 1)
        }

        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0])
            ll = cls_name.lower()

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]

            # Map to your categories
            if "person" in ll: counts["person"] += 1
            if "knife" in ll: counts["knife"] += 1
            if "weapon" in ll or "gun" in ll: counts["weapon"] += 1
            if "fire" in ll or "flame" in ll: counts["fire"] += 1

            detections.append({
                "label": cls_name,
                "confidence": round(confidence, 4),
                "bbox": bbox
            })

    # Threat logic
    if counts["knife"] > 0 or counts["weapon"] > 0 or counts["fire"] > 0:
        threat = "DANGER"
    elif counts["person"] > 0:
        threat = "WARNING"
    else:
        threat = "CLEAR"

    payload = {
        "detections": detections,
        "summary": ", ".join([f"{v} {k}" for k, v in counts.items() if v > 0]) or "no detections",
        "threat_level": threat,
        "class_counts": {
            "persons": counts["person"],
            "knives": counts["knife"],
            "weapons": counts["weapon"],
            "fire": counts["fire"]
        },
        "speed": speed,
        "total_objects": len(detections)
    }

    # 🔥 Broadcast to all connected Flutter apps
    socketio.emit('live_detections', payload)

    return jsonify(payload)

if __name__ == "__main__":
    # Use Railway's dynamic port
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, debug=False, host="0.0.0.0", port=port)