import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import cv2

class TripletMITDataset(Dataset):
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

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.open(img1[0])
        img2 = Image.open(img2[0])
        img3 = Image.open(img3[0])

        # img1 = Image.fromarray(img1.numpy(), mode='L')
        # img2 = Image.fromarray(img2.numpy(), mode='L')
        # img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mit_dataset)


class TripletCOCODataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, coco_dataset, obj_img_dict, dataset_path, split_name='train', transform=None):

        self.coco = coco_dataset.coco
        self.obj_img_dict = obj_img_dict[split_name]
        self.transform = transform
        self.dataset_path = dataset_path

        # Get ID's of all images
        self.imgs_ids = self.coco.getImgIds()

        # Create a list of all image IDs that contain at least one object from obj_img_dict
        self.obj_img_ids = []
        for obj in self.obj_img_dict:
            cat_ids = self.coco.getCatIds(catNms=[obj])
            img_ids = self.coco.getImgIds(catIds=cat_ids)
            self.obj_img_ids.extend(img_ids)
        self.obj_img_ids = list(set(self.obj_img_ids))

        # # Create a list of all image IDs that do not contain any object from obj_img_dict
        # self.non_obj_img_ids = list(set(self.imgs_ids) - set(self.obj_img_ids))

    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def resize_bounding_boxes(self, boxes, image_size, target_size):
        """
        Resize bounding boxes based on the size of the image.

        Args:
            boxes (list): List of bounding box coordinates [x1, y1, x2, y2].
            image_size (tuple): Original size of the image (width, height).
            target_size (tuple): Target size of the image (width, height).

        Returns:
            list: List of resized bounding box coordinates [x1, y1, x2, x2].
        """
        resized_boxes = []
        ratio_x = target_size[0] / image_size[0]
        ratio_y = target_size[1] / image_size[1]
        for box in boxes:
            x = box[0] * ratio_x
            y = box[1] * ratio_y
            width = box[2] * ratio_x
            height = box[3] * ratio_y
            resized_boxes.append([x, y, x + width, y + height])
        return resized_boxes

    def __getitem__(self, index):
        # Choose anchor image
        while True:
            # Choose anchor image
            anchor_img_id = self.obj_img_ids[index % len(self.obj_img_ids)]
            anchor_img = self.coco.loadImgs(anchor_img_id)[0]
            anchor_ann_ids = self.coco.getAnnIds(imgIds=anchor_img_id)  # Get the id of the instances
            anchor_anns = self.coco.loadAnns(anchor_ann_ids)
            anchor_cat_ids = list(set([ann['category_id'] for ann in anchor_anns]))
            anchor_cat_ids_str = [str(cat) for cat in anchor_cat_ids]

            if not anchor_cat_ids_str:
                index += 1
                continue
            else:
                break
            
        rand_cat = random.choice(anchor_cat_ids_str)
        # Choose positive image that contains at least one object from the same class as the anchor
        positive_img_id = anchor_img_id
        while positive_img_id == anchor_img_id:
            possible_positive_imgs = self.intersection(self.obj_img_dict[rand_cat], self.obj_img_ids)
            if possible_positive_imgs == []:
                continue
            positive_img_id = random.choice(possible_positive_imgs)

        positive_img = self.coco.loadImgs(positive_img_id)[0]
        positive_ann_ids = self.coco.getAnnIds(imgIds=positive_img_id)  # Get the id of the instances
        positive_anns = self.coco.loadAnns(positive_ann_ids)
        positive_cat_ids = list(set([ann['category_id'] for ann in positive_anns]))

        # negative_img_id = anchor_img_id
        # # Choose negative image that does not contain any object from the same class as the anchor
        # non_possible_negative_img = []
        # for cat in anchor_cat_ids_str:
        #     non_possible_negative_img += self.obj_img_dict[cat]
        # non_possible_negative_img = self.intersection(non_possible_negative_img, self.obj_img_ids)
        # while negative_img_id == anchor_img_id:
        #     negative_img_id = random.choice([item for item in self.obj_img_ids if item not in non_possible_negative_img])

        # negative_img = self.coco.loadImgs(negative_img_id)[0]

        # Choose negative image that does not contain any object from the same class as the anchor
        negative_img_id = random.choice([item for item in self.obj_img_ids if item != anchor_img_id and all(
            cat_id not in self.obj_img_dict for cat_id in anchor_cat_ids)])
        
        negative_img = self.coco.loadImgs(negative_img_id)[0]
        negative_ann_ids = self.coco.getAnnIds(imgIds=negative_img_id)  # Get the id of the instances
        negative_anns = self.coco.loadAnns(negative_ann_ids)
        negative_cat_ids = list(set([ann['category_id'] for ann in negative_anns]))

        # Load anchor, positive, and negative images
        anchor_img_path = os.path.join(self.dataset_path, anchor_img['file_name'])
        positive_img_path = os.path.join(self.dataset_path, positive_img['file_name'])
        negative_img_path = os.path.join(self.dataset_path, negative_img['file_name'])

        anchor_img = Image.open(anchor_img_path).convert('RGB')
        positive_img = Image.open(positive_img_path).convert('RGB')
        negative_img = Image.open(negative_img_path).convert('RGB')

        # Get bounding box coordinates and class labels for anchor and positive images
        anchor_boxes = []
        anchor_labels = []
        positive_boxes = []
        positive_labels = []
        negative_boxes = []
        negative_labels = []

        for ann in anchor_anns:
            anchor_boxes.append(ann['bbox'])
            anchor_labels.append(ann['category_id'])

        for ann in positive_anns:
            # if ann['category_id'] in anchor_cat_ids:
            positive_boxes.append(ann['bbox'])
            positive_labels.append(ann['category_id'])
            
        for ann in negative_anns:
            negative_boxes.append(ann['bbox'])
            negative_labels.append(ann['category_id'])

        # # AssertionError: Expected target boxes to be a tensor of shape [N, 4], got torch.Size([0]).
        # negative_boxes = torch.zeros((0, 4), dtype=torch.float32)
        # # Expected target boxes to be a tensor of shape [N, 4], got torch.Size([0])
        # negative_labels = torch.zeros((1, 1), dtype=torch.int64)

        target_size = [256, 256]
        anchor_boxes = torch.Tensor(self.resize_bounding_boxes(anchor_boxes, anchor_img.size, target_size))
        positive_boxes = torch.Tensor(self.resize_bounding_boxes(positive_boxes, positive_img.size, target_size))
        negative_boxes = torch.Tensor(self.resize_bounding_boxes(negative_boxes, negative_img.size, target_size))
        

        # Apply transformations to images, if provided
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        # Draw bounding boxes on images
        # anchor_img_bbox = anchor_img.clone().numpy()
        # positive_img_bbox = positive_img.clone().numpy()
        # negative_img_bbox = negative_img.clone().numpy()
        
        # cv2.rectangle(anchor_img_bbox, (anchor_boxes[0][0], anchor_boxes[0][1]), (anchor_boxes[0][2], anchor_boxes[0][3]), (0, 255, 0), 2)
        # cv2.rectangle(positive_img_bbox, (positive_boxes[0][0], positive_boxes[0][1]), (positive_boxes[0][2], positive_boxes[0][3]), (0, 255, 0), 2)
        # cv2.rectangle(negative_img_bbox, (negative_boxes[0][0], negative_boxes[0][1]), (negative_boxes[0][2], negative_boxes[0][3]), (0, 255, 0), 2)
        
        

        target = [anchor_boxes, torch.LongTensor(anchor_labels), positive_boxes, torch.LongTensor(positive_labels),
                  negative_boxes, torch.LongTensor(negative_labels)]
        

        return (anchor_img, positive_img, negative_img), target

    def __len__(self):
        return len(self.obj_img_ids)