import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import cv2
from tqdm import tqdm

import json
import random
import sys


from pycocotools.coco import COCO



class TripletIm2Text(Dataset):
    def __init__(self, ann_file, img_dir, transform=None, preprocessing='FastText'):
        self.img_dir = img_dir
        self.transform = transform
        self.preprocessing = preprocessing

        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images = self.annotations['images']
        self.annotations_an = self.annotations['annotations']

        # Create a dictionary with the image id as key and the annotation index
        # Each image can have multiple annotations
        self.img2ann = {}
        for i in range(len(self.annotations_an)):
            img_id = self.annotations_an[i]['image_id']
            if img_id not in self.img2ann:
                self.img2ann[img_id] = [i]
            else:
                self.img2ann[img_id].append(i)       
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.img_dir + '/' + self.images[index]['file_name']
        img_id = self.images[index]['id']
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # Choose randomly one captions for the image
        idx_pos = random.choice(self.img2ann[img_id])
        assert self.annotations_an[idx_pos]['image_id'] == img_id
        positive_caption_id = self.annotations_an[idx_pos]['id']
        positive_caption = self.annotations_an[idx_pos]['caption']
        
        # Choose randomly one caption that is not the same as the positive caption
        negative_caption_id = positive_caption_id
        while negative_caption_id == positive_caption_id:
            neg_ann_idx = random.choice(range(len(self.annotations_an)))
            neg_ann = self.annotations_an[neg_ann_idx]
            if neg_ann_idx in self.img2ann[img_id]:
                continue
            negative_caption_id = neg_ann['id'] 
            
        negative_caption = neg_ann['caption']
    
        
        return (image, positive_caption, negative_caption), []
    
    
    
class TripletText2Im(Dataset):
    def __init__(self, ann_file, img_dir, transform=None, preprocessing='FastText'):
        self.img_dir = img_dir
        self.transform = transform
        self.preprocessing = preprocessing

        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images = self.annotations['images']
        self.annotations_an = self.annotations['annotations']
 
                
        # Create a dictionary with the image id as key and the annotation index
        # Each image can have multiple annotations
        self.img2ann = {}
        for i in range(len(self.annotations_an)):
            img_id = self.annotations_an[i]['image_id']
            if img_id not in self.img2ann:
                self.img2ann[img_id] = [self.annotations[i]['caption_id']]
            else:
                self.img2ann[img_id].append(self.annotations[i]['caption_id']) 
           
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        caption_id = self.annotations_an[index]['id']
        caption = self.annotations_an[index]['caption']
        
        # Positive image
        pos_img_id = self.annotations_an[index]['image_id']
        img_path = self.img_dir + '/' + self.images[pos_img_id]['file_name']
        positive_image = Image.open(img_path).convert('RGB')
        
        # Negative image
        neg_img_id = pos_img_id
        while neg_img_id == pos_img_id:
            neg_img = random.choice(self.images)
            neg_img_id = neg_img['id']
            if caption_id in self.img2ann[neg_img_id]:
                continue
    
        neg_img_path = self.img_dir + '/' + self.images[neg_img_id]['file_name']
        negative_image = Image.open(neg_img_path).convert('RGB')
        
        if self.transform is not None:
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)
        
        return (caption, positive_image, negative_image), []
    
    
    
    
 
    
class TripletCOCORetrieval(Dataset):
    """
    Dataset for retrieval
    """

    def __init__(self, databaseImagesFolder, obj_img_dict, 
                 transform, split_name, allLabels  = None):
        
        self.labelDatabase = obj_img_dict[split_name]
        self.transform = transform
        self.databaseImagesFolder = databaseImagesFolder + '/'
        self.databaseImages = os.listdir(databaseImagesFolder)
        
        # Obtain labels
        self.objs = {}
        
        # Get objects per image
        for obj in self.labelDatabase.keys():
            for image in self.labelDatabase[obj]:
                if image in self.objs.keys():
                    self.objs[image].append(obj)
                else:
                    self.objs[image] = [obj]
        
        # Remove images that do not have any object
        aux = 0
        while aux < len(self.databaseImages):
            image1 = self.databaseImages[aux]
            image1Num = int(image1[:-4].split("_")[2])
            
            if not(image1Num in self.objs.keys()):
                del self.databaseImages[aux]
            else:
                aux += 1
        
        if not(allLabels is None):
            # Get every object in the image
            coco=COCO(allLabels)
            
            # Obtain labels
            self.objs = {}
            for image in self.databaseImages:
                imageId = int(image[:-4].split('_')[-1])
                ann_ids = coco.getAnnIds(imgIds=[imageId])
                anns = coco.loadAnns(ann_ids)
                annId = []
                for ann in anns:
                    annId.append(str(ann["category_id"]))
                if len(annId)>0:
                    self.objs[imageId]=annId

    def __getitem__(self, index):
        # Get image
        img1name = self.databaseImages[index]
        img1 = cv2.imread(self.databaseImagesFolder + img1name)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # Transform
        img1 = Image.fromarray(img1)
        if self.transform is not None:
            img1 = self.transform(img1)
        return img1, []

    def getObjs(self, index):
        # Get image name
        img1name = self.databaseImages[index]

        # Get objs
        img1value = int(img1name[:-4].split("_")[2])
        
        img1objs = self.objs[img1value]
        
        return img1objs
        
    def __len__(self):
        return len(self.databaseImages)

class CocoDatasetWeek5(Dataset):
    def __init__(self, ann_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = self.annotations['images']
        self.annotations = self.annotations['annotations']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.img_dir + '/' + self.images[index]['file_name']
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        annotation = self.annotations[index]
        
        return image, annotation