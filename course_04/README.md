# Computer Vision for IT Applied in Aviation

## Course 4: Image Segmentation

Minimal example showing how to run image segmentation with a pretrained YOLO model: load model, perform inference on an input image, and visualize segmentation masks.

### Object Detection with YOLO

Image segmentation with **YOLO** extends the classic YOLO object-detection framework by predicting not only bounding boxes and class labels, but also *pixel-level masks* for each detected object. Instead of treating segmentation as a separate dense-prediction task, YOLO integrates it directly into its fast, single-stage architecture. During inference, the model outputs the usual object detection components—class probabilities and bounding-box coordinates—along with per-object mask coefficients. These coefficients are combined with a small set of learned prototype masks to produce the final segmentation mask for every instance, allowing YOLO to remain extremely efficient even when producing detailed pixel-accurate outputs.

The provided Python code demonstrates how to use a pretrained **YOLOv8 model** for detecting objects in a single image. The steps are simple: load the YOLO model, run it on your image, and access the detected objects along with their coordinates and confidence scores. The code also automatically saves an annotated image showing **bounding boxes and labels** around each detected object. This hands-on example helps understand the workflow of object detection without the need to train a model from scratch, making it an ideal starting point for learning Computer Vision in aviation applications.

### What the Code Does

#### 1. Downloads YOLOv8-seg automatically if needed

When you run the code `model = YOLO("yolov8n-seg.pt")`, the Ultralytics library *checks if the pretrained YOLOv8 segmentation model is already on your system*. If not, it **downloads it automatically** from the official Ultralytics repository. The file bundles the blueprint + learned knowledge the model uses to perform segmentation.

The file contains everything needed for a YOLOv8 segmentation model to run:
- *Model architecture* – the neural network layers and connections (backbone, neck, segmentation head).
- *Learned weights* – the parameters the model learned during training on large datasets (like *COCO dadaset*).
- *Anchors / model metadata* – configuration details needed for inference.
- *Segmentation head parameters* – the part responsible for generating object masks, not just bounding boxes.
- *Class information* – the list of object categories the model can recognize.

Useful links:
- <a href="https://yolov8.com" target="_blank">Ultralytics YOLOv8</a>
- <a href="https://cocodataset.org/" target="_blank">COCO Dataset</a>

#### 2. Loads the pretrained model

Loads the pretrained **YOLOv8-seg model** into memory so it’s ready for running image segmentation.

You can explore and experiment with other pretrained versions depending on your speed/accuracy needs:
- **YOLOv8n-seg** – nano, very fast, smaller model
- **YOLOv8s-seg** – small, slightly more accurate
- **YOLOv8m-seg** – medium, good balance of speed and precision
- **YOLOv8l/x-seg** – large/extra-large, highest accuracy

#### 3. Runs detection on an image

Applies the **YOLOv8-seg model** to an image to identify objects, their classes, and bounding box locations.

#### 4. Shows annoated image

The model saves a new image (in the `output/` folder). The image shows:
- Colored segmentation masks outlining the exact shape of each detected object.
- Bounding boxes around those objects.
- Class labels and confidence scores.

### Folder Structure

```
course_04/
│
├── segmentation.py     # main script
├── images/                # test images go here
│     ├── 01-aircraft.jpg
|     |
|     |
├── output/
│    └── (annotated images will appear here)
│
└── README.md              # this file
```

### How to Run the Code

**1. Create a Python Virtual Environment**

```
python3 -m venv .venv/
```

**2. Activate the Environment**

```
source ./.venv/bin/activate
```

**3. Install Dependencies**

```
pip install -r requirements.txt
```

**4. Add your Own Images**

Place any *.jpg*, *.jpeg*, or *.png* files inside the *images/* folder.
They do **not** need to be airplane images — you can test anything. Try this with aviation-related images.

**5. Run the Script**

```
python segmentation.py
```

**6. Analyze Output Image**

You should see output showing colored segmentation masks, bounding boxes around objects, and class labels with confidence scores.

### How the Model Makes Predictions

YOLO segmentation works by extending the classic YOLO object detection framework to produce *pixel-level masks* in addition to bounding boxes and class labels. The model first processes the input image through a convolutional backbone to extract features at multiple scales. These features are passed through the detection head, which predicts **object classes**, **bounding box coordinates**, and **mask coefficients** for each potential object. Instead of generating masks from scratch for every object, YOLO uses a small set of learned prototype masks, which are combined with the predicted coefficients to produce the final instance masks. This approach allows the model to remain fast while producing accurate pixel-level segmentation.

During inference, YOLO scans the entire image in a single pass, predicting **multiple objects per grid cell**. For overlapping predictions, Non-Maximum Suppression (NMS) is applied to retain the most confident detection for each object. The final output includes bounding boxes, class labels, confidence scores, and the segmentation masks, which can be overlaid on the original image or used programmatically for further analysis. This makes YOLO segmentation highly efficient and suitable for real-time applications, such as detecting aircraft, vehicles, and runway areas in aviation scenarios.
