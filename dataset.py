import os
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as T

class TextileDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None, resize=(512, 512)):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.resize = resize
        self.imgs = []
        self.labels = []

        # Scan images and corresponding annotations
        for cls in os.listdir(images_dir):
            cls_path = os.path.join(images_dir, cls)
            for img_file in os.listdir(cls_path):
                if img_file.endswith(".jpg") or img_file.endswith(".png"):
                    self.imgs.append(os.path.join(cls_path, img_file))
                    self.labels.append(os.path.join(
                        annotations_dir, cls, img_file.replace('.jpg','.txt').replace('.png','.txt')
                    ))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label_path = self.labels[idx]

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h_img, w_img = img.shape[:2]

        # Resize if needed
        if self.resize:
            img = cv2.resize(img, self.resize)
            new_h, new_w = self.resize
            scale_x = new_w / w_img
            scale_y = new_h / h_img
        else:
            new_h, new_w = h_img, w_img
            scale_x = scale_y = 1.0

        boxes = []
        labels = []

        # Read YOLO annotations
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    if line.strip() == "":
                        continue
                    cls_id, x_center, y_center, w, h = map(float, line.strip().split())

                    # Convert YOLO normalized coords to pixel coordinates
                    xmin = (x_center - w/2) * w_img
                    ymin = (y_center - h/2) * h_img
                    xmax = (x_center + w/2) * w_img
                    ymax = (y_center + h/2) * h_img

                    # Scale to resized image
                    xmin *= scale_x
                    xmax *= scale_x
                    ymin *= scale_y
                    ymax *= scale_y

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(cls_id)+1)  # defect/stain classes

        # Handle defect-free images
        if len(boxes) == 0:
            # Create a dummy box for background
            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)  # background
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        # Apply transforms
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        return img, target
