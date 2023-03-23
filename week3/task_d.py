import random

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

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task D')
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

    # Register ms coco dataset val
    register_coco_instances("MSCOCO_val",{},"/ghome/group03/annotations/instances_val2017.json", "/ghome/group03/val2017")
    

    # Register ms coco dataset test
    register_coco_instances("MSCOCO_test",{},"/ghome/group03/annotations/image_info_test2017.json", "/ghome/group03/test2017")
    


    # Config
    cfg = get_cfg()

    if args.network == 'faster_RCNN':
        output_path = 'Results/Task_d/faster_RCNN/'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    elif args.network == 'mask_RCNN':
        output_path = 'Results/Task_d/mask_RCNN/'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")


    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.DATASETS.TEST = ("MSCOCO_test",)


    # Predictor
    predictor = DefaultPredictor(cfg)

    # Evaluator
    evaluator = COCOEvaluator("MSCOCO_val", cfg, False, output_dir=output_path)

    # Evaluate the model
    val_loader = build_detection_test_loader(cfg, "MSCOCO_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


    #dataset_dicts = get_ooc_dicts('val', pretrained=True)
    dataset_dicts = DatasetCatalog.get("MSCOCO_test")

    
    for d in random.sample(dataset_dicts, 20):
        im = cv2.imread(d["file_name"])
        # write the image at the output path
        cv2.imwrite(output_path + d["file_name"].split('/')[-1], im)
        outputs = predictor(im)

        
        classes_im = outputs["instances"].to("cpu").pred_classes.tolist()
        print(classes_im)

        # outputs are ordered by score. We take the detection with highest score
        mask = outputs["instances"].to("cpu").pred_masks[0]
        a = np.where(mask != False)
        
        # Duplicate the image and put the pixels outside the bounding box to black
        im2b = np.copy(im) 
        im2b[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1] = 0

        # Duplicate the image and put the pixels outside the mask to black
        im2c = np.copy(im)
        im2c[np.where(mask == False)] = 0

        # Duplicate the image and put the pixels outside the mask to black and add random noise to the pixels outside the bounding box
        im2d = np.copy(im)
        im2d[np.where(mask == False)] = 0
        im2d[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1] = np.random.uniform(low=0, high=255, size=(np.max(a[0])-np.min(a[0])+1, np.max(a[1])-np.min(a[1])+1, 3))

        #compute the outputs for each of the 4 images and save them
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu")[0]) #take the highest score detection
        cv2.imwrite(output_path + "pred_"+d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        outputs2b = predictor(im2b)
        v = Visualizer(im2b[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs2b["instances"].to("cpu"))
        cv2.imwrite(output_path + "bBB_"+d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        outputs2c = predictor(im2c)
        v = Visualizer(im2c[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs2c["instances"].to("cpu"))
        cv2.imwrite(output_path + "bM_" + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        
        outputs2d = predictor(im2d)
        v = Visualizer(im2d[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs2c["instances"].to("cpu"))
        cv2.imwrite(output_path +"_noise_" + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])


        print("Processed image: " + d["file_name"].split('/')[-1])


