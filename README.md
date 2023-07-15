# FFA-Lens

FFA-Lens is a web application designed for detecting 25 prevalent lesions in FFA (Fundus Fluorescein Angiography) images related to chronic ocular diseases. It utilizes the YOLOv8 model for accurate and efficient lesion detection. This documentation provides an overview of the application's functionality and usage.

## Features

- Detects 25 types of lesions in FFA images.
- Uses YOLOv8, a state-of-the-art object detection model, for accurate lesion detection.
- Provides configurable parameters such as image size, confidence threshold, and IOU threshold.
- Enables the selection of specific lesion classes for detection.
- Supports image upload and displays the predicted bounding boxes on the uploaded image.
- Allows users to download the image with the predicted bounding boxes.

## Getting Started

To use FFA-Lens, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit application with `streamlit run app.py`.

## Configuration

- Model: The YOLOv8 model is pre-selected and available for use. If other models are needed, please contact us via the email mentioned in the research paper.
- Weights: Choose the weights for the YOLOv8 model. The available weights options are: `yolov8.pt`,`yolov5.pt`,`yolov3.pt`.
- Size Image: Select the desired image size from the provided options.
- Confidence Threshold: Adjust the confidence threshold for object detection.
- IOU Threshold: Set the IOU (Intersection over Union) threshold for non-maximum suppression.
- Classes: Choose specific lesion classes for detection. By default, all classes are selected.
- All Classes: Enable this option to detect lesions from all available classes.

## Usage

1. Launch the application by running `streamlit run app.py`.
2. Interact with the Streamlit application through the sidebar to configure the model, weights, image size, thresholds, and classes.
3. The YOLOv8 model will be loaded based on the selected weights.
4. If desired, upload an FFA image using the provided image upload functionality.
5. The application will process the image using the YOLOv8 model to obtain lesion predictions.
6. The predicted bounding boxes will be drawn on the image, highlighting the detected lesions.
7. The image with the bounding boxes and the corresponding predictions will be displayed.
8. Optionally, click the "Download" button to save the image with the bounding boxes.

## Models Availability

The FFA-Lens application utilizes advanced YOLO models for lesion detection. The following models are available:

- YOLOv8: The YOLOv8 model is the primary model used in the FFA-Lens application. It offers high accuracy and robust performance for detecting lesions in FFA images.

Please note that the YOLOv3 and YOLOv5 models are also compatible with the FFA-Lens application; If you are interested in using the YOLOv3 or YOLOv5 model, please contact us through the mentioned email in the research paper for further details.

For the best results and support, we recommend utilizing the YOLOv8 model provided in this repository.


## Contact

For any inquiries regarding alternative models or further assistance, please contact us via email at [venkat.tummala275@gmail.com](mailto:venkat.tummala275@gmail.com).

