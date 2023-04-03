import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize
import numpy as np
from PIL import Image

import numpy as np


# class SiameseMITDataset(Dataset):
#     def __init__(self, data_dir='/ghome/group03/mcv/m3/datasets/MIT_small_train_1', split_name='train', transform=None):
#         self.data_dir = os.path.join(data_dir, split_name)
#         self.dataset = ImageFolder(data_dir, transform=transform)
#         self.n_samples = len(self.dataset)
#         self.split_name = split_name
#         self.transform = transform

#         if self.split_name == 'train':
#             self.labels = self.dataset.targets
#             self.data = self.dataset.samples
#             self.label_to_indices = {label: np.where(np.asarray(self.labels) == label)[0] for label in
#                                      np.unique(self.labels)}
#         else:
#             self.test_labels = self.dataset.targets
#             self.test_data = self.dataset.samples
#             self.labels_set = set(self.test_labels)
#             self.label_to_indices = {label: np.where(np.asarray(self.test_labels) == label)[0]
#                                      for label in self.labels_set}

#             random_state = np.random.RandomState(2023)

#             # Generate fixed pairs for testing
#             positive_pairs = [[i, random_state.choice(self.label_to_indices[self.test_labels[i]]), 1]
#                               for i in range(0, len(self.test_data), 2)]

#             negative_pairs = [[i, random_state.choice(
#                 self.label_to_indices[np.random.choice(list(self.labels_set - {self.test_labels[i]}))]), 0]
#                               for i in range(1, len(self.test_data), 2)]

#             self.test_pairs = positive_pairs + negative_pairs

#     def __getitem__(self, index):
#         if self.split_name == 'train':
#             target = np.random.randint(0, 2)  # 0 or 1 (negative or positive)
#             img1, label1 = self.data[index], self.labels[index]

#             if target == 1:
#                 siamese_index = index
#                 while siamese_index == index:
#                     siamese_index = np.random.choice(self.label_to_indices[label1])
                
#             else:
#                 siamese_label = np.random.choice(list(self.label_to_indices.keys() - {label1}))
#                 siamese_index = np.random.choice(self.label_to_indices[siamese_label])
                

#             img2 = self.data[siamese_index]

#         else:
#             img1 = self.test_data[self.test_pairs[index][0]]
#             img2 = self.test_data[self.test_pairs[index][1]]
#             target = self.test_pairs[index][2]

#         img1 = Image.open(img1[0])
#         img2 = Image.open(img2[0])

#         if self.transform is not None:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)

#         return img1, img2, target

#     def __len__(self):
#         return self.n_samples




class SiameseMITDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mit_dataset, split_name='train'):
        self.mit_dataset = mit_dataset

        self.train = split_name == 'train'
     
            
        self.transform = self.mit_dataset.transform

        if self.train:
            self.train_labels = self.mit_dataset.targets
            self.train_data = self.mit_dataset.samples
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mit_dataset.targets
            self.test_data = self.mit_dataset.samples
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.asarray(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i]]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.open(img1[0])
        img2 = Image.open(img2[0])
    
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mit_dataset)
