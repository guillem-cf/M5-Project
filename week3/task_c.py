import random
import os

import cv2
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

setup_logger()

# import some common libraries
import argparse

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances

# import some common detectron2 utilities
from detectron2 import model_zoo

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pycocotools.mask as mask_utils

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task C')
    parser.add_argument('--network', type=str, default='mask_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model')
    args = parser.parse_args()

    # Register dataset
    """
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign', 'parking meter',
                'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    # for subset in ["train", "val", "val_subset"]:
    """

    classes =  {0: u'__background__',
                1: u'person',
                2: u'bicycle',
                3: u'car',
                4: u'motorcycle',
                5: u'airplane',
                6: u'bus',
                7: u'train',
                8: u'truck',
                9: u'boat',
                10: u'traffic light',
                11: u'fire hydrant',
                12: u'stop sign',
                13: u'parking meter',
                14: u'bench',
                15: u'bird',
                16: u'cat',
                17: u'dog',
                18: u'horse',
                19: u'sheep',
                20: u'cow',
                21: u'elephant',
                22: u'bear',
                23: u'zebra',
                24: u'giraffe',
                25: u'backpack',
                26: u'umbrella',
                27: u'handbag',
                28: u'tie',
                29: u'suitcase',
                30: u'frisbee',
                31: u'skis',
                32: u'snowboard',
                33: u'sports ball',
                34: u'kite',
                35: u'baseball bat',
                36: u'baseball glove',
                37: u'skateboard',
                38: u'surfboard',
                39: u'tennis racket',
                40: u'bottle',
                41: u'wine glass',
                42: u'cup',
                43: u'fork',
                44: u'knife',
                45: u'spoon',
                46: u'bowl',
                47: u'banana',
                48: u'apple',
                49: u'sandwich',
                50: u'orange',
                51: u'broccoli',
                52: u'carrot',
                53: u'hot dog',
                54: u'pizza',
                55: u'donut',
                56: u'cake',
                57: u'chair',
                58: u'couch',
                59: u'potted plant',
                60: u'bed',
                61: u'dining table',
                62: u'toilet',
                63: u'tv',
                64: u'laptop',
                65: u'mouse',
                66: u'remote',
                67: u'keyboard',
                68: u'cell phone',
                69: u'microwave',
                70: u'oven',
                71: u'toaster',
                72: u'sink',
                73: u'refrigerator',
                74: u'book',
                75: u'clock',
                76: u'vase',
                77: u'scissors',
                78: u'teddy bear',
                79: u'hair drier',
                80: u'toothbrush'}
 
    """# Register ms coco dataset
    for d in ["train", "val"]:
        DatasetCatalog.register("coco_" + d, lambda d=d: get_coco_dicts(d))
        MetadataCatalog.get("coco_" + d).set(thing_classes=classes)
    """
    
    # Register ms coco dataset
    register_coco_instances("coco2017_test", {}, "/ghome/group03/annotations/image_info_test2017.json", "/ghome/group03/test2017")
    register_coco_instances("coco2017_val", {}, "/ghome/group03/annotations/instances_val2017.json", "/ghome/group03/val2017")


    # Config
    cfg = get_cfg()

    if args.network == 'faster_RCNN':
        output_path = '/ghome/group03/results_w3_m5/Task_c/faster_RCNN/'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    elif args.network == 'mask_RCNN':
        output_path = '/ghome/group03/results_w3_m5/Task_c/mask_RCNN/'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    # Create output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.DATASETS.TEST = ("coco2017_val",)


    # Predictor
    predictor = DefaultPredictor(cfg)

    dataset_dicts_val = DatasetCatalog.get('coco2017_val') #get_coco_dicts('val', pretrained=True)
    dataset_dicts_test = DatasetCatalog.get('coco2017_test')


    image_count = 0
    for d in random.sample(dataset_dicts_val, len(dataset_dicts_val)):
        im = cv2.imread(d["file_name"])
        # write the image at the output path
        path = output_path + d["file_name"].split('/')[-1].split('.')[0] + '_original.jpg'
        cv2.imwrite(path, im)
        outputs = predictor(im)

        # Write the image with the predictions
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        path = output_path + d["file_name"].split('/')[-1].split('.')[0] + '_pred.jpg'
        cv2.imwrite(path, out.get_image()[:, :, ::-1])


        # Get the list of the objects in the image represented by their class
        classes_im = outputs["instances"].to("cpu").pred_classes.tolist()

        # Choose randomly one of the objects of this class and segment it using the ground truth mask
        object_chosen = False
        counter = 0
        while not object_chosen:
            if counter == 10:
                break
            class_index = random.choice(classes_im)
            try:
                selected_obj = d['annotations'][classes_im.index(class_index)]
                object_chosen = True
            except:
                counter += 1
                continue
                
        if counter == 10:
            continue

        # Get the mask of the object
        rles = mask_utils.frPyObjects(selected_obj['segmentation'], im.shape[0], im.shape[1])
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)

        a = np.where(mask != False)
        try:
            im2 = im[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
            mask2 = mask[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
        except:
            print("Empty mask")
            continue

        im2c = np.where(np.repeat(np.expand_dims(mask2, axis=2), 3, axis=-1), im2, 0)

        if im2c.shape[0] >= im.shape[0] or im2c.shape[1] >= im.shape[1]:
            continue

        
        
        for i in range(4):
            im3 = np.copy(im)
            p0, p1 = np.random.uniform(low=0, high=im.shape[0]-im2c.shape[0], size=(1)).astype(int)[0], np.random.uniform(low=0, high=im.shape[1]-im2c.shape[1], size=(1)).astype(int)[0]
            im3[p0:p0+im2c.shape[0], p1:p1+im2c.shape[1]] = im2c

            im3 = np.where(im3==0, im, im3)

            # Write the modified image with the predictions
            outputs3 = predictor(im3)
            v = Visualizer(im3[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs3["instances"].to("cpu"))
            path = output_path + d["file_name"].split('/')[-1].split('.png')[0] + f'_{i}.jpg'
            cv2.imwrite(path, out.get_image()[:, :, ::-1])

        print("Processed image: " + d["file_name"].split('/')[-1] + "saved on " + output_path)

        image_count = image_count + 1

        if image_count == 100:
            break

    print("Done")