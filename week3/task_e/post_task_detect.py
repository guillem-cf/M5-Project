import os.path
import random
from os import listdir
from os.path import isfile

import cv2
import torch
import wandb
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from torch import nn, optim
from torchvision.transforms import transforms
import torchvision.models as models
from torch.autograd import Variable
import wandb

# wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb.init(project="resnet_50_style_transfer")

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
from PIL import Image

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task E')
    parser.add_argument('--network', type=str, default='mask_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model')
    args = parser.parse_args()

    # Register dataset
    """classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign', 'parking meter',
                'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']"""

    classes = {0: u'__background__',
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

    # Register ms coco dataset val
    register_coco_instances("MSCOCO_val", {}, "../../dataset/COCO/annotations/instances_val2017.json",
                            "../../dataset/COCO/val2017")

    # Register ms coco dataset test
    register_coco_instances("MSCOCO_test", {}, "../../dataset/COCO/annotations/image_info_test2017.json",
                            "../../dataset/COCO/test2017")

    # Config
    cfg = get_cfg()
    current_path = os.getcwd()
    if args.network == 'faster_RCNN':
        output_path = os.path.join(current_path, 'Results/Task_e/faster_RCNN/')
        # get if the path exists, if not create it
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    elif args.network == 'mask_RCNN':
        output_path = os.path.join(current_path, 'Results/Task_e/mask_RCNN/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.DATASETS.TEST = ("MSCOCO_test",)
    cfg.MODEL.DEVICE = 'cpu'
    # Predictor
    predictor = DefaultPredictor(cfg)

    """
    # Evaluator
    evaluator = COCOEvaluator("MSCOCO_val", cfg, False, output_dir=output_path)

    # Evaluate the model
    val_loader = build_detection_test_loader(cfg, "MSCOCO_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    
    """

    # dataset_dicts = get_ooc_dicts('val', pretrained=True)
    # dataset_dicts = DatasetCatalog.get("MSCOCO_test")

    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()

    output_path = os.path.join(current_path, '../Results/Task_e/style_transfer/post_detection')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    read_path = os.path.join(current_path, '../Results/Task_e/style_transfer/')
    # read images from read_path
    image_list = [Image.open(os.path.join(read_path, filename))
                  for filename in os.listdir(read_path)
                  if os.path.isfile(os.path.join(read_path, filename)) and filename.endswith('.png')]

    for image in image_list:
        img_1 = image
        # TODO: POSSIBILITAT DE FER SEGMENTATION O DIRECTAMENT LA IMATGE
        # PREDICCIO DE LA IMATGE
        # img_1 to numpy array
        img_1 = np.array(img_1)

        outputs_img1 = predictor(img_1)
        v = Visualizer(img_1[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        try:
            output_img1 = v.draw_instance_predictions(outputs_img1["instances"].to("cpu")[0])
        except:
            print("No objects detected")
            continue

        # get only segmentation and put pixels outside the mask to white

        image_name = os.path.splitext(image.filename)[0].split('/')[-1]

        cv2.imwrite(output_path + "final_detect_"+image_name + ".png", output_img1.get_image()[:, :, ::-1])


