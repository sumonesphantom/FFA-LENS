# FFA-Lens: Lesion Detection Web Application

This documentation provides an overview of the FFA-Lens web application, which is designed for detecting lesions in Fluorescein Angiography (FFA) images related to chronic ocular diseases. The application employs YOLO (You Only Look Once) models for accurate lesion detection. The documentation covers the code implementation, functionalities, and the interaction flow of the application.

## Code Overview

The code is divided into two main sections: the user interface implemented using Streamlit and the backend functionality for model loading, prediction, and utility functions.

### User Interface (Streamlit Application)

The Streamlit application provides a user-friendly interface for interacting with FFA-Lens. The key components of the user interface include:

- Sidebar: The sidebar allows the user to select various parameters such as the YOLO model, weights, image size, thresholds, and classes. It provides options for customization and configuration based on user preferences.

- Image Upload Functionality: The application enables users to upload FFA images for lesion detection. The uploaded images are processed by the backend YOLO models for prediction.

- "Download" Button: This button allows users to download the FFA image with bounding boxes drawn around the detected lesions.

### Backend Functionality

The backend functionality consists of functions responsible for YOLO model loading, prediction, and utility functions for color handling and legend generation. The key functions in the backend include:

- Model Loading: The `get_yolo3`, `get_yolo5`, and `get_yolo8` functions load the YOLO models (YOLOv3, YOLOv5, and YOLOv8) based on the selected weights. These models are essential for lesion detection in FFA images.

- Lesion Detection: The `get_preds` function processes the uploaded FFA image using the selected YOLO model and returns the predicted bounding boxes for the detected lesions. It incorporates various parameters such as confidence threshold, IOU threshold, and class filtering to refine the predictions.

- Color Handling and Legend Generation: The `get_colors` function generates a color dictionary based on the selected classes to assign unique colors to each class for visualizing the detected lesions. The `get_legend_color` function retrieves the corresponding color for a specific class for creating a legend.

## Interaction Flow

The interaction flow of the FFA-Lens web application is as follows:

1. The user interacts with the Streamlit application through the user interface.
2. The user selects the YOLO model, weights, image size, thresholds, and classes from the sidebar.
3. Based on the user's selections, the corresponding YOLO model is loaded.
4. If the user uploads an FFA image, it is processed by the YOLO model to obtain predictions for the lesions.
5. The predicted bounding boxes are drawn on the image, indicating the detected lesions.
6. The image with the bounding boxes and the corresponding lesion predictions is displayed to the user.
7. If the user clicks the "Download" button, the image with the drawn bounding boxes is saved to the local system.

Please note that the availability of necessary libraries and modules such as OpenCV (cv2), PyTorch (torch), NumPy (numpy), Streamlit (streamlit), and others is assumed. Additionally, certain functions and modules referenced in the code snippet, such as `config` and `ultralytics`, are not provided, and their detailed implementation is not described here.

## Conclusion

The FFA-Lens web application showcases its potential in detecting various lesions in FFA images related to chronic ocular diseases. By utilizing configurable YOLO models, such as YOLOv5 and YOLOv8, FFA-Lens achieves promising results in lesion detection, exceeding precision and recall values of 0.6. FFA-Lens offers the advantages of early detection, automated lesion identification, improved diagnostic efficiency, and potential integration into real-time clinical settings. Future enhancements could involve expanding the range of detectable lesions and optimizing the performance of YOLO models to further improve the capabilities of FFA-Lens in ocular disease management.
