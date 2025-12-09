import torch
from PIL import Image
import os

def parse_annotation(txt_file, img_size):
    """
    txt_file : path to annotation txt
    img_size : (width, height) of image
    """
    boxes = []
    labels = []

    w, h = img_size

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.strip() == "":
            continue
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        # convert from YOLO format to xyxy
        x1 = (x_center - width/2) * w
        y1 = (y_center - height/2) * h
        x2 = (x_center + width/2) * w
        y2 = (y_center + height/2) * h

        boxes.append([x1, y1, x2, y2])
        labels.append(int(class_id)+1)  # +1 if 0=background, or directly class index

    if len(boxes) == 0:
        # No object in image
        boxes = torch.zeros((0,4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    else:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

    target = {"boxes": boxes, "labels": labels}
    return target
