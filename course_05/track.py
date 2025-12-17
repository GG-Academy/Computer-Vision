import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

# Load YOLOv8 detection model
model = YOLO("../models/yolov8n.pt")

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Open video file
cap = cv2.VideoCapture("videos/plane.mov")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
k = 0

while cap.isOpened():
    k += 1
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.4)[0]

    detections = []

    # Convert YOLO output to SORT format
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]

        detections.append([
            float(x1), float(y1),
            float(x2), float(y2),
            float(conf)
        ])

    detections = np.array(detections)

    # Update tracker
    tracks = tracker.update(detections)

    # Draw tracking results
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if k == frame_count:
        break

cap.release()
cv2.destroyAllWindows()
