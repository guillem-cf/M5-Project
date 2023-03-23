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
from ooc_Dataset import *

# import some common detectron2 utilities
from detectron2 import model_zoo

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task C')
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
    # for subset in ["train", "val", "val_subset"]:
    #     DatasetCatalog.register(f"ooc_{subset}", lambda subset=subset: get_ooc_dicts(subset, pretrained=True))
    #     MetadataCatalog.get(f"ooc_{subset}").set(thing_classes=classes)

    # Register ms coco dataset
    for d in ["train", "val"]:
        DatasetCatalog.register("coco_" + d, lambda d=d: get_coco_dicts(d))
        MetadataCatalog.get("coco_" + d).set(thing_classes=classes)


    # Config
    cfg = get_cfg()

    if args.network == 'faster_RCNN':
        output_path = 'Results/Task_c/faster_RCNN/'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    elif args.network == 'mask_RCNN':
        output_path = 'Results/Task_c/mask_RCNN/'
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


    co_ocurrence_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    dataset_dicts = get_ooc_dicts('val', pretrained=True)

    # CONCURRENCE ALREADY COMPUTED IN TASK B
    # for d in random.sample(dataset_dicts, len(dataset_dicts)):
    #     im = cv2.imread(d["file_name"])
    #     outputs = predictor(im)

    #     print("Processed image: ", d["file_name"])

    #     # For each predicted object in the image:
    #     for object_class1 in outputs["instances"].to("cpu").pred_classes.tolist():
    #         # We look for all the other objects that also appear in that image:
    #         for object_class2 in outputs["instances"].to("cpu").pred_classes.tolist():
    #             if object_class1 != object_class2:
    #                 co_ocurrence_matrix[object_class1][object_class2] = co_ocurrence_matrix[object_class1][object_class2] + 1

    
    # mask = np.nonzero(np.sum(co_ocurrence_matrix, axis=0) > 35)[0]
    # co_ocurrence_matrix = co_ocurrence_matrix[np.ix_(mask, mask)]

    # fig, ax = plt.subplots(figsize=(15,15), dpi=200)
    # ax.set_title("Co-ocurrence matrix for the COCO objects \n(top " + str(len(mask)) + " co-ocurrences)", size=20)
    # heatmap = sns.heatmap(co_ocurrence_matrix, annot=False, linewidth=.5, xticklabels=np.array(classes)[mask], yticklabels=np.array(classes)[mask], ax=ax)
    # heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=15)
    # heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize=15)
    # fig.savefig("task_c_heatmap.png")


    for d in random.sample(dataset_dicts, len(dataset_dicts)):
        im = cv2.imread(d["file_name"])
        # write the image at the output path
        cv2.imwrite(output_path + d["file_name"].split('/')[-1], im)
        outputs = predictor(im)

        if classes.index('person') not in outputs["instances"].to("cpu").pred_classes.tolist():
            continue

        # We look for an object that is not a person
        classes_im = outputs["instances"].to("cpu").pred_classes.tolist()


        # Choose randomly one of the objects of this class that is not a person and segment it
        class_index = random.choice(classes_im)

        mask = outputs["instances"].to("cpu").pred_masks[classes_im.index(class_index)]
        a = np.where(mask != False)
        
        # Duplicate the image and crop it to the bounding box of the object
        im2b = im[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
        maskb = mask[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]

        im2c = np.where(np.repeat(np.expand_dims(maskb, axis=2), 3, axis=-1), im2b, 0)

        if im2c.shape[0] >= im.shape[0] or im2c.shape[1] >= im.shape[1]:
            continue

        im3 = np.copy(im)
        p0, p1 = np.random.uniform(low=0, high=im.shape[0]-im2c.shape[0], size=(1)).astype(int)[0], np.random.uniform(low=0, high=im.shape[1]-im2c.shape[1], size=(1)).astype(int)[0]
        im3[p0:p0+im2c.shape[0], p1:p1+im2c.shape[1]] = im2c

        im3 = np.where(im3==0, im, im3)

        outputs3 = predictor(im3)
        v = Visualizer(im3[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs3["instances"].to("cpu"))

        cv2.imwrite(output_path + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        print("Processed image: " + d["file_name"].split('/')[-1])



        # for d2 in random.sample(dataset_dicts, len(dataset_dicts)):
        #     im2 = cv2.imread(d2["file_name"])
        #     outputs2 = predictor(im2)

        #     classes2 = outputs2["instances"].to("cpu").pred_classes.tolist()

        #     if classes.index('car') not in classes2 and classes.index('skis') not in classes2 and classes.index('dining table') not in classes2:
        #         continue
        #     else:

        #         if classes.index('car') in classes2:
        #             class_index = classes.index('car')
                
        #         if classes.index('skis') in classes2:
        #             class_index = classes.index('skis')
                    
        #         if classes.index('dining table') in classes2:
        #             class_index = classes.index('dining table')

        #         mask = outputs2["instances"].to("cpu").pred_masks[classes2.index(class_index)]
                
        #         a = np.where(mask != False)
        #         im2b = im2[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
        #         maskb = mask[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]

        #         im2c = np.where(np.repeat(np.expand_dims(maskb, axis=2), 3, axis=-1), im2b, 0)

        #         if im2c.shape[0] >= im.shape[0] or im2c.shape[1] >= im.shape[1]:
        #             continue

        #         im3 = np.copy(im)
        #         p0, p1 = np.random.uniform(low=0, high=im.shape[0]-im2c.shape[0], size=(1)).astype(int)[0], np.random.uniform(low=0, high=im.shape[1]-im2c.shape[1], size=(1)).astype(int)[0]
        #         im3[p0:p0+im2c.shape[0], p1:p1+im2c.shape[1]] = im2c

        #         im3 = np.where(im3==0, im, im3)

        #         outputs3 = predictor(im3)
        #         v = Visualizer(im3[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        #         out = v.draw_instance_predictions(outputs3["instances"].to("cpu"))

        #         cv2.imwrite(output_path + d["file_name"].split('/')[-1], out.get_image()[:, :, ::-1])

        #         print("Processed image: " + d["file_name"].split('/')[-1])

        #         break