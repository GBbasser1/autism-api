from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)

# تحميل المودلين
object_model = YOLO("yolo11s.pt")         # لاكتشاف الطفل (bounding box)
pose_model = YOLO("yolo11s-pose.pt")      # لاستخراج الحركات والـ keypoints

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        video_url = data.get("video_url")
        if not video_url:
            return jsonify({"error": "Missing video_url"}), 400

        # تحميل الفيديو من الرابط
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(requests.get(video_url).content)

        cap = cv2.VideoCapture(video_path)
        predictions = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # تشغيل object detection أولاً لتحديد الأشخاص
            obj_results = object_model(frame)

            for det in obj_results[0].boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                person_crop = frame[y1:y2, x1:x2]

                # تشغيل pose model على الشخص المحدد فقط
                pose_results = pose_model(person_crop)

                if pose_results and pose_results[0].probs:
                    class_id = pose_results[0].probs.top1
                    class_name = pose_results[0].names[class_id]
                    predictions.append(class_name)

        cap.release()
        os.remove(video_path)

        if not predictions:
            return jsonify({"prediction": "no_detection"})

        final_prediction = max(set(predictions), key=predictions.count)
        return jsonify({"prediction": final_prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    