import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.boxes import BoxMode
import cv2
import pycocotools.mask as mask_utils
import numpy as np

classes = ['car', 'pedestrian']


def line_to_object(line):
    line = line.replace("\n", "").split(" ")

    #Each line of an annotation txt file is structured like this (where rle means run-length encoding from COCO): time_frame id class_id img_height img_width rle

    time_frame, obj_id, class_id, img_height, img_width, rle = int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]), line[5]

    #obj_instance_id = obj_id % 1000

    mask = mask_utils.decode(rle)
    y, x = np.where(mask == 1)
    bbox = [int(np.min(x)), int(np.min(y)), int(np.max(x) - np.min(x)), int(np.max(y) - np.min(y))]

    return {
        "category_id": class_id,
        "bbox": bbox,
        "bbox_mode": BoxMode.XYWH_ABS,
    }

def get_kitti_dicts():

    anotations_dir = "/home/group01/mcv/datasets/KITTI-MOTS/instances_txt/"

    txt_file = os.path.join(root, f"{seq}_kitti.txt")

    with open(txt_file, "r") as f:
        lines = f.readlines()

    dataset_dicts = []
    idx = 0

    for idx, line in enumerate(lines):
        annotation_filename = 
        record = {}
        
        filename = os.path.join(root, "data_object_image_2/training/image_2", annotation_filename.replace("txt", "png"))
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      

        annotation_filepath = os.path.join(root, "training/label_2", annotation_filename)

        with open(annotation_filepath, "r") as f:
            objects = f.readlines()

        objs = []
        for obj_line in objects:
            obj = line_to_object(obj_line)
            if obj is not None:
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    

    return dataset_dicts


def register_kitti_dataset():
    subset= "train"
    DatasetCatalog.register(f"kitti_{subset}", lambda subset=subset: get_kitti_dicts())
    print(f"Successfully registered 'kitti_{subset}'!")
    MetadataCatalog.get(f"kitti_{subset}").thing_classes = classes



