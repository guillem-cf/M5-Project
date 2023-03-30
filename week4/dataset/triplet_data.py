import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class TripletMITDataset(Dataset):
    def __init__(self, data_dir='/ghome/group03/mcv/m3/datasets/MIT_small_train_1', split_name='train', transform=None):
        self.data_dir = os.path.join(data_dir, split_name)
        self.transform = transform
        self.dataset = ImageFolder(root=self.data_dir, transform=None)
        self.samples = []
        for i in range(len(self.dataset)):
            path, target = self.dataset.imgs[i]
            self.samples.append((path, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor_path, anchor_target = self.samples[index]

        # choose a positive example from the same class as the anchor
        positive_candidates = [s for s in self.samples if s[1] == anchor_target and s[0] != anchor_path]
        positive_path, _ = random.choice(positive_candidates)

        # choose a negative example from a different class than the anchor
        negative_candidates = [s for s in self.samples if s[1] != anchor_target]
        negative_path, _ = random.choice(negative_candidates)

        anchor_sample = Image.open(anchor_path).convert('RGB')
        positive_sample = Image.open(positive_path).convert('RGB')
        negative_sample = Image.open(negative_path).convert('RGB')

        if self.transform is not None:
            anchor_sample = self.transform(anchor_sample)
            positive_sample = self.transform(positive_sample)
            negative_sample = self.transform(negative_sample)

        return anchor_sample, positive_sample, negative_sample
