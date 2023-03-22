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
from formatDataset import *

# import some common detectron2 utilities
from detectron2 import model_zoo

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task B')
    parser.add_argument('--network', type=str, default='mask_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model')
    args = parser.parse_args()

    # Register dataset
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign', 'parking meter',
                'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    for subset in ["train", "val", "val_subset"]:
        DatasetCatalog.register(f"ooc_{subset}", lambda subset=subset: get_ooc_dicts(subset, pretrained=True))
        MetadataCatalog.get(f"ooc_{subset}").set(thing_classes=classes)



    # Config
    cfg = get_cfg()

    if args.network == 'faster_RCNN':
        output_path = 'Results/Task_b/faster_RCNN/'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    elif args.network == 'mask_RCNN':
        output_path = 'Results/Task_b/mask_RCNN/'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")


    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.DATASETS.TEST = ("ooc_val",)


    # Predictor
    predictor = DefaultPredictor(cfg)

    # Evaluator
    evaluator = COCOEvaluator("ooc_val", cfg, False, output_dir=output_path)

    # Evaluate the model
    val_loader = build_detection_test_loader(cfg, "ooc_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))




    # --------------------------------- INFERENCE --------------------------------- #


    co_ocurrence_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    dataset_dicts = get_ooc_dicts('val', pretrained=True)
    for d in random.sample(dataset_dicts, len(dataset_dicts)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite(output_path + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        print("Processed image: " + d["file_name"].split('/')[-1])

        # For each predicted object in the image:
        for object_class1 in outputs["instances"].to("cpu").pred_classes.tolist():
            # We look for all the other objects that also appear in that image:
            for object_class2 in outputs["instances"].to("cpu").pred_classes.tolist():
                if object_class1 != object_class2:
                    co_ocurrence_matrix[object_class1][object_class2] = co_ocurrence_matrix[object_class1][object_class2] + 1



    mask = np.nonzero(np.sum(co_ocurrence_matrix, axis=0) > 35)[0]
    co_ocurrence_matrix = co_ocurrence_matrix[np.ix_(mask, mask)]

    fig, ax = plt.subplots(figsize=(15,15), dpi=200)
    ax.set_title("Co-ocurrence matrix for the COCO objects \n(top " + str(len(mask)) + " co-ocurrences)", size=20)
    heatmap = sns.heatmap(co_ocurrence_matrix, annot=False, linewidth=.5, xticklabels=np.array(classes)[mask], yticklabels=np.array(classes)[mask], ax=ax)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=15)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize=15)
    fig.savefig("task_b_heatmap.png")
