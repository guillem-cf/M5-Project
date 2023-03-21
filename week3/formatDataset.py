import os
import random

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

def get_ooc_dicts(subset, pretrained = False):
    images = "/ghome/group03/mcv/datasets/out_of_context/"

    if subset == "train":
        sequences_id = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015"]
        
    elif subset == "val":
        sequences_id = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015"]

    elif subset == "val_subset":
        sequences_id = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015"]

    dataset_dicts = []
    idx = 1

    for seq_id in tqdm(sequences_id):
            
        record = {}

        filename = os.path.join(images, "im" + str(seq_id) + ".jpg")

        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        dataset_dicts.append(record)

        idx += 1

    return dataset_dicts
