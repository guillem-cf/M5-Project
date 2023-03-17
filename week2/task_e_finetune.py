# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer

from formatDataset import get_kitti_mots_dicts



if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser(description='Task E: Finetuning')
    parser.add_argument('--network', type=str, default='faster_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    train_dicts = get_kitti_mots_dicts("train")
    val_dicts = get_kitti_mots_dicts("val")

    cfg = get_cfg()

    if args.network == 'faster_RCNN':
        output_path = './Results/Task_e/faster_RCNN'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("kitti_mots_train",)
        cfg.DATASETS.TEST = ("kitti_mots_val",)
    
 




