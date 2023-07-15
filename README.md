# FFA-Lens

FFA-Lens is a web application designed for detecting 46 prevalent lesions in FFA (Fundus Fluorescein Angiography) images related to chronic ocular diseases. It utilizes the YOLOv8 model for accurate and efficient lesion detection. This documentation provides an overview of the application's functionality and usage.

## Features

- Detects 46 types of lesions in FFA images.
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
- Weights: Choose the weights for the YOLOv8 model. The available weights options are: `yolov8.pt`.
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

## Contact

For any inquiries regarding alternative models or further assistance, please contact us via email at [research@example.com](mailto:research@example.com).

