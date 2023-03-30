import os
import random

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
                for i, fname in enumerate(sorted(fnames)):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target], i)
                    self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor_path, anchor_target, anchor_idx = self.samples[index]

        # choose a positive example from the same class as the anchor
        positive_candidates = [s for s in self.samples if s[1] == anchor_target and s[2] != anchor_idx]
        positive_path, _, _ = random.choice(positive_candidates)

        # choose a negative example from a different class than the anchor
        negative_candidates = [s for s in self.samples if s[1] != anchor_target]
        negative_path, _, _ = random.choice(negative_candidates)

        anchor_sample = Image.open(anchor_path).convert('RGB')
        positive_sample = Image.open(positive_path).convert('RGB')
        negative_sample = Image.open(negative_path).convert('RGB')

        if self.transform is not None:
            anchor_sample = self.transform(anchor_sample)
            positive_sample = self.transform(positive_sample)
            negative_sample = self.transform(negative_sample)

        return anchor_sample, positive_sample, negative_sample
