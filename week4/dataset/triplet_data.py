import os
import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class TripletMITDataset(Dataset):
    def __init__(self, data_dir='/ghome/group03/mcv/m3/datasets/MIT_small_train_1', split_name='train', transform=None):
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
                for i in range(len(fnames)):
                    anchor_path = os.path.join(root, fnames[i])
                    anchor_target = self.class_to_idx[target]
                    pos_path = anchor_path
                    pos_target = anchor_target
                    j = random.randint(0, len(fnames) - 1)
                    while j == i:
                        j = random.randint(0, len(fnames) - 1)
                    neg_path = os.path.join(root, fnames[j])
                    neg_target = self.class_to_idx[target]
                    item = (anchor_path, pos_path, neg_path, anchor_target, pos_target, neg_target)
                    self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor_path, pos_path, neg_path, anchor_target, pos_target, neg_target = self.samples[index]
        anchor = Image.open(anchor_path).convert('RGB')
        pos = Image.open(pos_path).convert('RGB')
        neg = Image.open(neg_path).convert('RGB')
        if self.transform is not None:
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)
        return anchor, pos, neg, anchor_target, pos_target, neg_target
