import torch
import torchvision
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torchvision.transforms as T

from split import train_loader

print(torchvision.__version__)

transform = T.ToTensor()
#Load the pretrained model
TextileModel = fasterrcnn_resnet50_fpn(pretrained=True)

#classifier with three class
#one for background, one for defect_free, one for stain
num_classes = 3
input_features = TextileModel.roi_heads.box_predictor.cls_score.in_features
TextileModel.roi_heads.box_predictor = FastRCNNPredictor(input_features,num_classes)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

TextileModel.to(device)

#optimizers
params = [p for p in TextileModel.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params,lr=0.005,momentum=0.9,weight_decay=0.0005)

num_epochs = 3
for epoch in range(num_epochs):
    TextileModel.train()
    epoch_loss = 0
    batch_count = 0
    for imgs, targets in train_loader:
        batch_count += 1
        # Move tensors to device
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = TextileModel(imgs, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    print(f"Epoch {epoch+1}, Average Loss: {epoch_loss / batch_count:.4f}")


torch.save(TextileModel.state_dict(),"TextileQualityModel.pth")
print("Model Saved for identifying defects in Textiles")






