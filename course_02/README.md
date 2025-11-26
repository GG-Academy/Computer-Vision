# Computer Vision for IT Applied in Aviation

## Course 1: Introduction to Computer Vision

This project demonstrates how to use a pre-trained deep learning model (namely, **MobileNetV2**) to classify images.

### Image Classification with MobileNetV2

The code loads one or more images, prepares them for the model, sends them through the neural network, and prints the top predictions.

It is designed to help you understand how modern computer vision models recognize objects — in this case, aircraft and other aviation-related objects.

### What the Code Does

#### 1. Loads a pre-trained model

We use **MobileNetV2**, a convolutional neural network trained on *ImageNet* (a dataset of 1.2 million images and 1000 categories).

Useful links:
- <a href="https://arxiv.org/abs/1801.04381" target="_blank">Paper on MobileNetV2</a>
- <a href="https://www.image-net.org" target="_blank">ImageNet</a>

#### 2. Reads images from a folder

The script loops through all files ending in *.jpg*, *.jpeg*, or *.png* from a folder named *images*.

#### 3. Resizes each image to the input size expected by the model (224×224)

This ensures the image matches what the network was trained on.

#### 4. Preprocesses the image

These transformations prepare the image for the neural network:
- Converts to a numerical array
- Normalizes pixel values
- Adds a batch dimension

#### 5. Runs the image through **MobileNetV2**

The model outputs a probability distribution across 1000 classes.

#### 6. Prints the top 3 predicted labels

For example:
```
airliner        --> 0.4679
wing            --> 0.2666
warplane        --> 0.1247
```

This means the model is most confident that the image contains an *airliner*.

### Folder Structure

```
course_01/
│
├── classify_images.py     # main script
├── images/                # test images go here
│     ├── airplane_01.jpg
│     ├── cat_01.jpg
│     ├── dog_01.png
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
python classify_images.py
```

**6. Evaluate Output**

You should see output like:

```
--- Processing: plane1.jpg ---
airliner        --> 0.4679
wing            --> 0.2666
warplane        --> 0.1247
```

Each image is processed independently.

### How the Model Makes Predictions

MobileNetV2 was trained on **1,000 categories** including:
- Airliner
- Warplane
- Missile
- Airport terminal
- Ground vehicles
- Many everyday objects (cats, cars, trees, etc.)

Because the model has not been fine-tuned specifically on aircraft datasets, its predictions may not always be perfect.
Still, it gives us a great starting point for:
- understanding neural network pipelines
- experimenting with aviation images
- learning how pre-trained models work

### Tips for Better Results

- Use clear, centered objects
- Avoid heavy cropping
- Test multiple images
- Try letterboxing to preserve aspect ratio

### Next Steps

In the following course, we will fine-tune **MobileNetV2** on a custom aviation dataset. That would allow classifying images into aviation-specific classes like:
- A320
- B737
- Bombardier CRJ
- Ground vehicles
- Runway vs. taxiway scenes
