import os

from PIL import Image
from torch.utils.data import Dataset

import torch
import torchvision.transforms as transforms


# Create a dataset class for COCO dataset

class FeaturesCOCODataset(Dataset):
    
    def __init__(self, coco_path, split_name='train', transform=None):
        
        self.coco_path = coco_path
        
        self.coco_dataset = os.listdir(self.coco_path)
     
        self.transform = transform
        
        self.PIL_transform = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, index):
        
        img = os.path.join(self.coco_path, self.coco_dataset[index])
        
        img = Image.open(img)
        # img = self.PIL_transform(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
    def __len__(self):
        return len(self.coco_dataset)
       