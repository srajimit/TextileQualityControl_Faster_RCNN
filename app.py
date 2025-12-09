import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

# Load model
num_classes = 1 + 2
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("TextileQualityModel.pth", map_location=torch.device('cpu')))
model.eval()

# Transform
transform = T.Compose([T.ToTensor()])

st.title("Textile Defect Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    img_tensor = transform(image)
    with torch.no_grad():
        prediction = model([img_tensor])
    
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    
    import cv2
    import numpy as np
    img_np = np.array(image)
    
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.8:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(img_np, f"Class {label.item()}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    
    st.image(img_np, caption="Prediction", use_container_width=True)
