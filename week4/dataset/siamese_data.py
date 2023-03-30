import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class SiameseMITDataset(Dataset):
    def __init__(self, data_dir, split_name='train', transform=None):
        self.data_dir = os.path.join(data_dir, split_name)
        self.transform = transform
        self.dataset = ImageFolder(root=self.data_dir, transform=None)
        self.samples = []
        for i in range(len(self.dataset)):
            for j in range(i+1, len(self.dataset)):
                item1 = self.dataset[i]
                item2 = self.dataset[j]
                self.samples.append((item1, item2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item1, item2 = self.samples[idx]
        path1, target1 = item1[0], item1[1]
        path2, target2 = item2[0], item2[1]

        sample1 = item1[0]
        sample2 = item2[0]

        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return sample1, sample2, torch.tensor(target1 == target2, dtype=torch.float32)