import torch
from torch.utils.data import DataLoader
from dataset import TextileDataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix
from split import test_loader

#load the pretrained model
num_classes = 3

TextileModel = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False)

input_features = TextileModel.roi_heads.box_predictor.cls_score.in_features
TextileModel.roi_heads.box_predictor=FastRCNNPredictor(input_features,num_classes)
TextileModel.load_state_dict(torch.load(("TextileQualityModel.pth"),map_location=torch.device('cpu')))
TextileModel.eval()

images_dir = 'Dataset/images'
annotations_dir = 'Dataset/annotations'

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

y_true = []
y_pred = []

if len(test_loader) == 0:
    print("test_loader is empty.")

for imgs, targets in test_loader:
    img = imgs[0]
    gt_boxes = targets[0]["boxes"].numpy()
    gt_labels = targets[0]["labels"].numpy()
    
    with torch.no_grad():
        outputs = TextileModel([img])
    
    pred_boxes = outputs[0]["boxes"].numpy()
    pred_labels = outputs[0]["labels"].numpy()
    pred_scores = outputs[0]["scores"].numpy()
    
    # consider only predictions with score > 0.5
    keep = pred_scores > 0.5
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    
    matched_gt = set()
    for pb, pl in zip(pred_boxes, pred_labels):
        match_found = False
        for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            if i in matched_gt:
                continue
            if iou(pb, gb) >= 0.5 and pl == gl:
                y_true.append(gl)
                y_pred.append(pl)
                matched_gt.add(i)
                match_found = True
                break
        if not match_found:
            y_true.append(0)  # background for unmatched prediction
            y_pred.append(pl)
    
    # For unmatched GT boxes → False Negatives
    for i, gl in enumerate(gt_labels):
        if i not in matched_gt:
            y_true.append(gl)
            y_pred.append(0)  # predicted as background

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:", cm)

