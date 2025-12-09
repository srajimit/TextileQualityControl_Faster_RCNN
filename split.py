from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from dataset import TextileDataset


images_dir = 'Dataset/images'
annotations_dir = 'Dataset/annotations'

dataset = TextileDataset(images_dir,annotations_dir)

train_indices, test_indices = train_test_split(list(range(len(dataset))),test_size =0.2, random_state=42)

train_dataset = Subset(dataset,train_indices)
test_dataset = Subset(dataset,test_indices)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)
test_loader = DataLoader(test_dataset,
                         batch_size = 2,
                         shuffle = False,
                         collate_fn=collate_fn)

print("Train and Test set split completed")