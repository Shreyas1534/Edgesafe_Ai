from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
model = YOLO("webapp.pt")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    img_bytes = file.read()

    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 🔥 IMPROVED INFERENCE SETTINGS
    results = model(
        img,
        conf=0.20,        # LOWER threshold (more detections)
        iou=0.45,
        imgsz=640,        # better accuracy
        verbose=False
    )

    detections = []
    has_knife = has_weapon = has_fire = has_person = False

    speed_pre = speed_inf = speed_post = 0
    counts = {}

    for r in results:
        speed_pre  = r.speed.get("preprocess", 0)
        speed_inf  = r.speed.get("inference", 0)
        speed_post = r.speed.get("postprocess", 0)

        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [
                round(x1, 1),
                round(y1, 1),
                round(x2 - x1, 1),
                round(y2 - y1, 1)
            ]

            ll = cls_name.lower()

            if "knife" in ll:
                has_knife = True
            if "weapon" in ll or "gun" in ll:
                has_weapon = True
            if "fire" in ll or "flame" in ll:
                has_fire = True
            if "person" in ll or "people" in ll:
                has_person = True

            counts[cls_name] = counts.get(cls_name, 0) + 1

            detections.append({
                "label": cls_name,
                "class_id": cls_id,
                "confidence": round(confidence, 4),
                "bbox": bbox
            })

    # 🔥 DEBUG PRINT (IMPORTANT)
    print("Detections:", detections)

    # 🔥 FALLBACK (for demo safety)
    if len(detections) == 0:
        print("⚠️ No detections → fallback active")

    # Summary
    if detections:
        summary = ", ".join(f"{v} {k}" for k, v in counts.items())
    else:
        summary = "no detections"

    # Counts
    def cnt(keyword):
        return sum(v for k, v in counts.items() if keyword in k.lower())

    persons_count = sum(cnt(kw) for kw in ["person", "people"])
    knives_count  = cnt("knife")
    weapons_count = sum(cnt(kw) for kw in ["weapon", "gun"])
    fire_count    = sum(cnt(kw) for kw in ["fire", "flame"])

    # Threat logic
    if has_knife or has_weapon or has_fire:
        threat = "DANGER"
    elif has_person:
        threat = "WARNING"
    else:
        threat = "CLEAR"

    return jsonify({
        "detections": detections,
        "summary": summary,
        "threat_level": threat,
        "class_counts": {
            "persons": persons_count,
            "knives": knives_count,
            "weapons": weapons_count,
            "fire": fire_count
        },
        "speed": {
            "preprocess": round(speed_pre, 1),
            "inference": round(speed_inf, 1),
            "postprocess": round(speed_post, 1)
        },
        "total_objects": len(detections)
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)