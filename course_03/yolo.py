from ultralytics import YOLO

# 1. Load a pretrained YOLO model
# Options: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.
model = YOLO("../models/yolov8n.pt")   # nano model = very fast

# 2. Run inference on a single image
results = model("images/01-airport.jpg")   # replace with your image path

# 3. Print object detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"{cls_name} ({conf:.2f}) at {xyxy}")

# 4. Save annotated results automatically
results[0].save("output/01-airport.jpg")
print("Saved result to output/01-airport.jpg")
