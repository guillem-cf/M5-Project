import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import cv2


class TripletCOCODataset(Dataset):
    def __init__(self, coco_dataset, obj_img_dict, dataset_path, split_name='train', transform=None):
        self.coco = coco_dataset.coco
        self.obj_img_dict = obj_img_dict[split_name]
        self.transform = transform
        self.dataset_path = dataset_path

        self.imgs_ids = self.coco.getImgIds()

        self.obj_img_ids = []
        for obj in self.obj_img_dict:
            cat_ids = self.coco.getCatIds(catNms=[obj])
            img_ids = self.coco.getImgIds(catIds=cat_ids)
            self.obj_img_ids.extend(img_ids)
        self.obj_img_ids = list(set(self.obj_img_ids))

        self.intersect_dict = {}
        self.negative_candidates = {}
        for cat in self.obj_img_dict:
            self.intersect_dict[cat] = list(set(self.obj_img_dict[cat]) & set(self.obj_img_ids))
        
        for img_id in self.obj_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            cat_ids = list(set([ann['category_id'] for ann in anns]))
            negative_candidates = [item for item in self.obj_img_ids if item != img_id and all(cat_id not in self.obj_img_dict for cat_id in cat_ids)]
            self.negative_candidates[img_id] = negative_candidates

    def resize_bounding_boxes(self, boxes, image_size, target_size):
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
        while True:
            anchor_img_id = self.obj_img_ids[index % len(self.obj_img_ids)]
            anchor_img = self.coco.loadImgs(anchor_img_id)[0]
            anchor_ann_ids = self.coco.getAnnIds(imgIds=anchor_img_id)
            anchor_anns = self.coco.loadAnns(anchor_ann_ids)
            anchor_cat_ids = list(set([ann['category_id'] for ann in anchor_anns]))
            anchor_cat_ids_str = [str(cat) for cat in anchor_cat_ids]

            if not anchor_cat_ids_str:
                index += 1
                continue
            else:
                break

        rand_cat = random.choice(anchor_cat_ids_str)
        possible_positive_imgs = self.intersect_dict[rand_cat]
        positive_img_id = random.choice(possible_positive_imgs)
        positive_img = self.coco.loadImgs(positive_img_id)[0]
        positive_ann_ids = self.coco.getAnnIds(imgIds=positive_img_id)
        positive_anns = self.coco.loadAnns(positive_ann_ids)

        negative_anns = []
        while negative_anns == []:
            negative_img_id = random.choice(self.negative_candidates[anchor_img_id])
            negative_img = self.coco.loadImgs(negative_img_id)[0]
            negative_ann_ids = self.coco.getAnnIds(imgIds=negative_img_id)
            negative_anns = self.coco.loadAnns(negative_ann_ids)
        
        anchor_img_path = os.path.join(self.dataset_path, anchor_img['file_name'])
        positive_img_path = os.path.join(self.dataset_path, positive_img['file_name'])
        negative_img_path = os.path.join(self.dataset_path, negative_img['file_name'])

        anchor_img = Image.open(anchor_img_path).convert('RGB')
        positive_img = Image.open(positive_img_path).convert('RGB')
        negative_img = Image.open(negative_img_path).convert('RGB')

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
            positive_boxes.append(ann['bbox'])
            positive_labels.append(ann['category_id'])

        for ann in negative_anns:
            negative_boxes.append(ann['bbox'])
            negative_labels.append(ann['category_id'])

        target_size = [800, 800]
        anchor_boxes = torch.Tensor(self.resize_bounding_boxes(anchor_boxes, anchor_img.size, target_size))
        positive_boxes = torch.Tensor(self.resize_bounding_boxes(positive_boxes, positive_img.size, target_size))
        negative_boxes = torch.Tensor(self.resize_bounding_boxes(negative_boxes, negative_img.size, target_size))

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        target = [anchor_boxes, torch.LongTensor(anchor_labels), positive_boxes, torch.LongTensor(positive_labels),
                negative_boxes, torch.LongTensor(negative_labels)]

        return (anchor_img, positive_img, negative_img), target

    def __len__(self):
        return len(self.obj_img_ids)