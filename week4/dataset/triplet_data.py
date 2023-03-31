import os

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class TripletMITDataset(Dataset):

    def __init__(self, data_dir='/ghome/group03/mcv/m3/datasets/MIT_small_train_1', split_name='train', transform=None):
        self.data_dir = os.path.join(data_dir, split_name)
        self.dataset = ImageFolder(data_dir, transform=transform)
        self.n_samples = len(self.dataset)
        self.split_name = split_name
        self.transform = transform

        if self.split_name == 'train':
            self.train_labels = self.dataset.targets
            self.train_data = self.dataset.samples
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.dataset.targets
            self.test_data = self.dataset.samples
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.asarray(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - {self.test_labels[i]})
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.split_name == 'train':
            anchor, label1 = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - {label1}))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            positive = self.train_data[positive_index]
            negative = self.train_data[negative_index]
        else:
            anchor = self.test_data[self.test_triplets[index][0]]
            positive = self.test_data[self.test_triplets[index][1]]
            negative = self.test_data[self.test_triplets[index][2]]

        anchor = Image.open(anchor[0])
        positive = Image.open(positive[0])
        negative = Image.open(negative[0])

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative

    def __len__(self):
        return self.n_samples
