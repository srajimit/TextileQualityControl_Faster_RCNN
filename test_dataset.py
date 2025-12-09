# test_dataset.py
from dataset import TextileDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

# To specify images and annotations
images_dir = "Dataset/images"
annotations_dir = "Dataset/annotations"

# create dataset object
dataset = TextileDataset(images_dir, annotations_dir, transforms=transforms.ToTensor())

print("Total samples:", len(dataset))

# check for a sample
img, target = dataset[0]
print("Image shape:", img.shape)
print("Boxes:", target["boxes"])
print("Labels:", target["labels"])

# visualize a sample
def show_sample(index=0):
    img, target = dataset[index]
    img = img.permute(1, 2, 0).numpy()
    boxes = target["boxes"]
    labels = target["labels"]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"Class {label.item()}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

show_sample(0)

# check whether data can be loaded
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
for imgs, targets in loader:
    print("Batch size:", len(imgs))
    print("First target keys:", targets[0].keys())
    break
