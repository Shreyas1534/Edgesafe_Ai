# ─────────────────────────────────────────────────────────────
# CRITICAL: gevent monkey-patching MUST happen before any other
# imports. Without this, Flask-SocketIO's gevent async_mode can't
# cooperatively multiplex the WebSocket handshake/ping-pong with
# incoming /detect requests — under load the socket connection
# from the Flutter app stalls or times out even though the plain
# HTTP /detect endpoint keeps responding.
# ─────────────────────────────────────────────────────────────
import gevent.monkey
gevent.monkey.patch_all()

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64
import threading

app = Flask(__name__)
CORS(app)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    ping_timeout=60,     # tolerate brief stalls instead of dropping the connection
    ping_interval=25,
)

model = YOLO("webapp.pt")
model.to('cpu')

# Serialize inference - Ultralytics model objects aren't guaranteed
# thread/greenlet safe for concurrent forward passes.
inference_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return jsonify({"status": "CORS OK"}), 200

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    with inference_lock:
        results = model(
            img,
            conf=0.45,
            iou=0.50,
            imgsz=480,
            verbose=False
        )
        annotated_frame = results[0].plot()

    # Compress heavily to save bandwidth, then encode to Base64
    _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    detections = []
    counts = {"person": 0, "knife": 0, "weapon": 0, "fire": 0}
    speed = {"preprocess": 0, "inference": 0, "postprocess": 0}

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

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]

            if "person" in ll: counts["person"] += 1
            if "knife" in ll: counts["knife"] += 1
            if "weapon" in ll or "gun" in ll: counts["weapon"] += 1
            if "fire" in ll or "flame" in ll: counts["fire"] += 1

            detections.append({
                "label": cls_name,
                "confidence": round(confidence, 4),
                "bbox": bbox
            })

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
        "total_objects": len(detections),
        "frame": frame_base64
    }

    # 🔥 FLUTTER BROADCAST 🔥
    try:
        socketio.emit('live_detections', payload)
    except Exception as e:
        print(f"Socket emit failed: {e}")

    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    socketio.run(app, debug=False, host="0.0.0.0", port=port)