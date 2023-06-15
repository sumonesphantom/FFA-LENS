from typing import List
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.colors as mcolors
from PIL import Image
from config import CLASSES1
from config import CLASSES2
from ultralytics import YOLO
from pathlib import Path

html = """
<div style = "background-color:black;padding:18px">
<h1 style = "color:green; text-align:center"> Detect Lesion</h1>
</div>
"""
st.set_page_config(
    page_title="YOLO",
)

st.markdown(html, unsafe_allow_html =True)
def attempt_download_yolo(file, repo):
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", '').lower())

@st.cache(max_entries=2)
def get_yolo3(weights):
    # return torch.hub.load("/yolov3","custom",path='{}'.format(weights),source='local',force_reload=True)
    return torch.hub.load("ultralytics/yolov3","custom",path='{}'.format(weights),force_reload=True)

#WongKinYiu
@st.cache(max_entries=2)
def get_yolo5(weights):
    return torch.hub.load('ultralytics/yolov5','custom',path = '{}'.format(weights), force_reload =True)

@st.cache(max_entries=2, allow_output_mutation=True)
def get_yolo8(weights):
    return YOLO(str(weights))
    # return torch.hub.load("ultralytics/ultralytics","custom",path='{}'.format(weights),force_reload=True)


@st.cache(max_entries=10)
def get_preds(img, imgsz):
    if all_classes == False:
        model.conf = conf_thres
        model.classes = classes
        model.iou = iou_thres
        #result = model([img], size=imgsz, conf =conf_thres, iou = iou_thres, max_det = max_det, classes = classes)
        result = model([img], size=imgsz)
    elif all_classes == True and weights!="yolov8.pt"and weights!="yolov7.pt":
        model.conf = conf_thres
        model.classes = None
        model.iou = iou_thres
        #result = model([img], size=imgsz, conf =conf_thres, iou = iou_thres, max_det = max_det, classes = None)
        result = model([img], size=imgsz)
    elif weights=="yolov8.pt":
        if all_classes == False:
            model.classes = classes 
        model.conf = conf_thres
        model.iou = iou_thres
        #result = model([img], size=imgsz, conf =conf_thres, iou = iou_thres, max_det = max_det, classes = None)
        results = model([img])
        boxes = results[0].boxes
        result = boxes.data  # returns one box
        return result.numpy()
    print("hi",result.xyxy[0])
    return result.xyxy[0].numpy()


def get_colors(indexes):
    to_255 = lambda c: int(c*255)
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) 
                                                for name_color in tab_colors]
    base_colors = list(mcolors.BASE_COLORS.values())
    base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
    rgb_colors = tab_colors + base_colors
    rgb_colors = rgb_colors*5

    color_dict = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors):
            color_dict[index] = rgb_colors[i]
        else:
            color_dict[index] = (255,0,0)

    return color_dict

def get_legend_color(class_name):
    index = CLASSES.index(class_name)
    color = rgb_colors[index]
    return 'background-color: rgb({color[0]},{color[1]},{color[2]})'.format(color=color)

st.markdown("""
        <style>
        .format-style{
            font-size:20px;
            font-family:arial;
            color:red;
        }
        </style>
        """,
        unsafe_allow_html= True
    )

st.markdown(
    """
    <style>.
    common-style{
        font-size:18px;
        font-family:arial;
        color:pink;
    }
    </style>
    """,
    unsafe_allow_html= True
)
st.sidebar.markdown(
    '<p class = "format-style"> Parameter </p>',
    unsafe_allow_html= True
)

modelSelect = st.sidebar.selectbox(
    'Model', 
    ('yolov3','yolov5','yolov8'),
    format_func = lambda a: a[:len(a)] 
)


weights = st.sidebar.selectbox(
    'Weights', 
    ('yolov3.pt','yolov5.pt','yolov8.pt'),
    format_func = lambda a: a[:len(a)-3] 
)

if weights == 'yolov3.pt':
    CLASSES = CLASSES1
elif weights == 'yolov5.pt':
    CLASSES = CLASSES1
elif weights == 'yolov8.pt':
    CLASSES = CLASSES1


imgsz = st.sidebar.selectbox(
    'Size Image',
    (416,512,608,896,1024,1280,1408,1536)
)

conf_thres = st.sidebar.slider(
    'Confidence Threshold', 0.00, 1.00, 0.7
)

iou_thres = st.sidebar.slider(
    'IOU Threshold', 0.00,1.00, 0.45
)
# max_det = st.sidebar.selectbox(
#     'Max detection',
#     [i for i in range(1,20)]
# )

classes = st.sidebar.multiselect(
    'Classes',
    [i for i in range(len(CLASSES1))],
    format_func= lambda index: CLASSES1[index]
)

all_classes = st.sidebar.checkbox('All classes', value =True)



with st.spinner('Loading the model...'):
    if(modelSelect==''):
        model = get_yolo5(weights)
    elif(modelSelect=='yolov3'):
        model = get_yolo3(weights)
    elif(modelSelect=='yolov5'):
        model = get_yolo5(weights)
    elif(modelSelect=='yolov8'):
        model = get_yolo8(weights)
    

st.success('Loading '+modelSelect+' model.. Done!')

prediction_mode = st.sidebar.radio(
    "",
    ('Single image','none'),
    index=0)

if all_classes:
    target_class_ids = list(range(len(CLASSES1)))
elif classes:
    target_class_ids = [class_name for class_name in classes]
else:
    target_class_ids = [0]

rgb_colors = get_colors(target_class_ids)

detected_ids = None


if prediction_mode == 'Single image':
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_preds(img,imgsz)

        result_copy = result.copy()
        # if all_classes == True and weights=="yolov8.pt":
        #     result_copy = result_copy[result_copy[:,-1]==target_class_ids]
        # else: 
        result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]

        res =[]
        detected_ids = []
        img_draw = img.copy().astype(np.uint8)
        #img_draw1 = img.copy().astype(np.uint8)
        font = cv2.FONT_HERSHEY_TRIPLEX
        # set some text
        text = "Some text in a box!"
        # get the width and height of the text box
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=0.5, thickness=2)[0]

        for bbox_data in result_copy:
            #print(bbox_data)
            xmin, ymin, xmax, ymax, con, label = bbox_data
            con = round(con,4)
            #print(con)
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
            img_draw = cv2.rectangle(img_draw, 
                                    p0, p1, 
                                    rgb_colors[label], 2) 
            label2 = CLASSES1.index(str(label))
            print(str(label),CLASSES2[label2])
            res.append(str(label)+' '+ CLASSES2[label2])

            box_coords = ((int(xmin)-1, int(ymin)), (int(xmin) + text_width-80, (int(ymin) - 5) - text_height ))
            img_draw = cv2.rectangle(img_draw, box_coords[0], box_coords[1], rgb_colors[label], cv2.FILLED)

            #img_draw1 = cv2.putText(img_draw,str(label)+' '+ CLASSES2[label2], (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb_colors[label], 2)
            img_draw = cv2.putText(img_draw, str(label), (int(xmin), int(ymin) - 5),font, 0.5, (255,255,255), 1)
            #img_draw = cv2.putText(img_draw, ', '+CLASSES2[label2], (int(xmin)+30, int(ymin) - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.4, rgb_colors[label], 2)
            img_draw = cv2.putText(img_draw, ', '+str(con), (int(xmin)+30, int(ymin) - 5),font, 0.5, (255,255,255), 1)
            detected_ids.append(label)
        st.write(res)
        st.image(img_draw, use_column_width=False)
if st.sidebar.button("Download"):
    cv2.imwrite("2.jpg", img_draw)