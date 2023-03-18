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

from formatDataset import get_kitti_dicts, register_kitti_dataset


if __name__ == '__main__':
 
    # args parser
    parser = argparse.ArgumentParser(description='Task C: Inference')
    parser.add_argument('--network', type=str, default='faster_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    args = parser.parse_args()


    dataset_dicts = get_kitti_dicts("val")
    kitti_metadata = register_kitti_dataset("val")

    cfg = get_cfg()

    if args.network == 'faster_RCNN':
        output_path = './Results/Task_d/faster_RCNN'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        
    elif args.network == 'mask_RCNN':
        output_path = './Results/Task_d/mask_RCNN'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.DATASETS.TEST = ("kitti_val", )
    predictor = DefaultPredictor(cfg)
    
    evaluator = COCOEvaluator("kitti_val", cfg, False, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, "kitti_val")

    print(inference_on_dataset(predictor.model, val_loader, evaluator))