from ultralytics import YOLO
import cv2

# Load segmentation model (pretrained on COCO)
model = YOLO("../models/yolov8n-seg.pt")

# Run segmentation on an image
results = model("images/01-aircraft.jpg")  # Replace with your image path

# Display / save results
results[0].save("output/seg-output.jpg")

# Optional: show masks on screen
annotated = results[0].plot()
cv2.imshow("Segmentation Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
