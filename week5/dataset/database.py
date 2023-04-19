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



class ImageDatabase(Dataset):
    def __init__(self, ann_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images = self.annotations['images']
        self.annotations_an = self.annotations['annotations']
        
        # Create a dictionary with the image id as key and the annotation caption id
        # Each image can have multiple captions id
        self.img2ann = {}
        for i in range(len(self.annotations_an)):
            img_id = self.annotations_an[i]['image_id']
            if img_id not in self.img2ann:
                self.img2ann[img_id] = [self.annotations_an[i]['id']]
            else:
                self.img2ann[img_id].append(self.annotations_an[i]['id'])       
        
    def __len__(self):
        return len(self.images)
    
    def getCaptions(self, index):
        img_id = self.images[index]['id']
        
        return self.img2ann[img_id]

    def __getitem__(self, index):
        img_path = self.img_dir + '/' + self.images[index]['file_name']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        # img_id = self.images[index]['id']

        return image, []
    
    
    
class TextDatabase(Dataset):
    def __init__(self, ann_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images = self.annotations['images']
        self.annotations_an = self.annotations['annotations']
 
                
        # Create a dictionary with the caption id as key and the images id that have this caption
        # Each image can have multiple annotations
        self.capt2img = {}
        for i in range(len(self.annotations_an)):
            caption_id = self.annotations_an[i]['id']
            if caption_id not in self.capt2img:
                self.capt2img[caption_id] = [self.annotations_an[i]['image_id']]
            else:
                self.capt2img[caption_id].append(self.annotations_an[i]['image_id'])
           
        
    def __len__(self):
        return len(self.images)
    
    def getImages(self, index):
        caption_id = self.annotations_an[index]['id']
        
        return self.capt2img[caption_id]
    
    def getCaptionId(self, index):
        return self.annotations_an[index]['id']
    

    def __getitem__(self, index):
        
        caption = self.annotations_an[index]['caption']
        caption_id = self.annotations_an[index]['id']
        
        return caption, []