import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class SiameseMITDataset(Dataset):
    def __init__(self, data_dir, split_name='train', transform=None):
        self.data_dir = os.path.join(data_dir, split_name)
        self.transform = transform
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = []
        for target in self.classes:
            d = os.path.join(self.data_dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname1 in sorted(fnames):
                    path1 = os.path.join(root, fname1)
                    item1 = (path1, self.class_to_idx[target])
                    for fname2 in sorted(fnames):
                        path2 = os.path.join(root, fname2)
                        item2 = (path2, self.class_to_idx[target])
                        self.samples.append((item1, item2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item1, item2 = self.samples[idx]
        path1, target1 = item1
        path2, target2 = item2

        sample1 = Image.open(path1).convert('RGB')
        sample2 = Image.open(path2).convert('RGB')

        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return sample1, sample2, torch.tensor(target1 == target2, dtype=torch.float32)
