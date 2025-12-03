# Computer Vision for IT Applied in Aviation

## Course 3: Object Detection

This project demonstrates how to use a pretrained **YOLO** model to detect and label objects in an image, and save the annotated result.

### Object Detection with YOLO

YOLO (You Only Look Once) is a modern, one-stage object detection model that can identify **multiple objects in an image in a single pass**. Unlike traditional classifiers that only predict what is in an image, YOLO predicts **both the classes and the locations** of objects by outputting bounding boxes along with confidence scores. It divides the image into a grid and, for each cell, predicts whether an object is present, what class it belongs to, and where it is located. This approach allows YOLO to be **extremely fast**, making it suitable for real-time applications such as airport surveillance, drone monitoring, and runway vehicle detection.

The provided Python code demonstrates how to use a pretrained YOLOv8 model for detecting objects in a single image. The steps are simple: load the YOLO model, run it on your image, and access the detected objects along with their coordinates and confidence scores. The code also automatically saves an annotated image showing **bounding boxes and labels** around each detected object. This hands-on example helps beginners understand the workflow of object detection without the need to train a model from scratch, making it an ideal starting point for learning computer vision in aviation applications.

### What the Code Does

#### 1. Downloads YOLOv8 automatically if needed

When you run the code `model = YOLO("yolov8n.pt")`, the Ultralytics library *checks if the pretrained YOLOv8 weights are already on your system**. If not, it **downloads them automatically** from the official Ultralytics repository. This ensures that you can start running object detection immediately **without manually downloading** model files or worrying about paths. The downloaded file contains the pretrained network parameters for YOLOv8 (nano version in this case), which has already been trained on the **COCO dataset**, enabling it to detect 80 common object classes out of the box.

This automatic download is convenient for beginners and ensures reproducibility: everyone running the code gets the same model.

Useful links:
- <a href="https://yolov8.com" target="_blank">Ultralytics YOLOv8</a>
- <a href="https://cocodataset.org/" target="_blank">COCO Dataset</a>

#### 2. Loads a pretrained model

Loads a pretrained **YOLOv8 model** into memory so it’s ready for running object detection on images.

You can explore and experiment with other pretrained versions depending on your speed/accuracy needs:
- **YOLOv8n** – nano, very fast, smaller model
- **YOLOv8s** – small, slightly more accurate
- **YOLOv8m** – medium, good balance of speed and precision
- **YOLOv8l/x** – large/extra-large, highest accuracy

#### 3. Runs detection on an image

Applies the **YOLOv8 model** to an image to identify objects, their classes, and bounding box locations.

#### 4. Prints information

Shows:
- class (e.g., airplane, person, truck)
- confidence
- bounding box coordinates

#### 5. Saves an annotated image with boxes & labels

Saves a copy of the image showing detected objects with bounding boxes and class labels.

### Folder Structure

```
course_03/
│
├── yolo.py     # main script
├── images/                # test images go here
│     ├── 01-airport.jpg
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
They do **not** need to be airplane images — you can test anything.

**5. Run the Script**

```
python yolo.py
```

**6. Evaluate Output**

You should see output like:

```
image 1/1 ../images/01-airport.jpg: 448x640 2 persons, 1 car, 1 airplane, 8 trucks, 70.2ms
Speed: 2.6ms preprocess, 70.2ms inference, 7.3ms postprocess per image at shape (1, 3, 448, 640)
truck (0.69) at [387.01251220703125, 157.00294494628906, 456.51434326171875, 208.3914031982422]
person (0.68) at [156.6243896484375, 209.44784545898438, 169.71044921875, 245.66403198242188]
person (0.64) at [320.897216796875, 170.91207885742188, 330.7296142578125, 199.23379516601562]
truck (0.61) at [83.19467163085938, 208.15090942382812, 170.0054931640625, 264.92010498046875]
truck (0.54) at [20.34429931640625, 174.8273468017578, 169.17889404296875, 244.3300018310547]
airplane (0.54) at [46.07818603515625, 39.00399398803711, 513.6262817382812, 148.16705322265625]
car (0.44) at [82.53626251220703, 145.40512084960938, 112.69051361083984, 158.03414916992188]
truck (0.42) at [158.80563354492188, 143.99786376953125, 178.12020874023438, 159.95361328125]
truck (0.34) at [83.29083251953125, 145.34022521972656, 112.87748718261719, 158.04759216308594]
truck (0.31) at [247.5167694091797, 158.1455078125, 267.12872314453125, 173.91925048828125]
truck (0.28) at [2.3922576904296875, 211.2542724609375, 84.42498779296875, 253.01611328125]
truck (0.27) at [0.05682373046875, 173.8206787109375, 91.48466491699219, 231.12081909179688]
Saved result to output/01-airport.jpg
```

Each image is processed independently.

### How the Model Makes Predictions

YOLOv8 predicts objects by dividing the input image into a **grid of cells**, where each cell is responsible for detecting objects whose centers fall within it. For every cell, the model predicts **bounding box coordinates**, a **confidence score** (how likely an object is present), and class probabilities for each possible object. During inference, these predictions are combined across the image, and **Non-Maximum Suppression (NMS)** is applied to remove duplicate or overlapping boxes, keeping only the most confident ones.

Essentially, the model works in a single pass over the image: it extracts features through a deep convolutional backbone, analyzes multiple locations and scales simultaneously, and outputs both **"what"** each object is and **"where"** it is located. This process allows YOLO to detect **multiple objects of different sizes and classes quickly and efficiently**, making it ideal for real-time applications in aviation, such as monitoring aircraft, vehicles, or drones on runways and in surrounding airspace.
