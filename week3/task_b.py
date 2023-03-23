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

import pycocotools.mask as mask_utils

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task B')
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
    
    """
    for subset in ["train", "val", "val_subset"]:
        DatasetCatalog.register(f"coco2017_{subset}", lambda subset=subset: get_coco_dicts(subset, pretrained=True))
        MetadataCatalog.get(f"coco2017_{subset}").set(thing_classes=list(classes.values()))
    """
    register_coco_instances("coco2017_test", {}, "/ghome/group03/annotations/image_info_test2017.json", "/ghome/group03/test2017")
    register_coco_instances("coco2017_val", {}, "/ghome/group03/annotations/instances_val2017.json", "/ghome/group03/val2017")


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
    cfg.DATASETS.TEST = ("coco2017_test",)


    # Predictor
    predictor = DefaultPredictor(cfg)


    # --------------------------------- INFERENCE --------------------------------- #


    co_ocurrence_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    dataset_dicts_val = DatasetCatalog.get('coco2017_val') #get_coco_dicts('val', pretrained=True)
    dataset_dicts_test = DatasetCatalog.get('coco2017_test')

    for d in random.sample(dataset_dicts_val, len(dataset_dicts_val)):
        im = cv2.imread(d["file_name"])
        """
        outputs = predictor(im)
        print("Processed image: " + d["file_name"].split('/')[-1])

        # For each predicted object in the image:
        for object_class1 in outputs["instances"].to("cpu").pred_classes.tolist():
        """

        for obj1 in d['annotations']:

            object_class1 = obj1['category_id']

            if object_class1 != 0: # Avoid background

                # We look for all the other objects that also appear in that image:
                for obj2 in d['annotations']:

                    object_class2 = obj2['category_id']

                    if object_class2 != 0: # Avoid background

                        if object_class1 != object_class2:
                            co_ocurrence_matrix[object_class1][object_class2] = co_ocurrence_matrix[object_class1][object_class2] + 1



    co_ocurrence_matrix = np.where(co_ocurrence_matrix==0, np.inf, co_ocurrence_matrix)
    co_ocurrence_matrix = 1/(co_ocurrence_matrix + 1e-7)
    mask = np.nonzero(np.sum(co_ocurrence_matrix, axis=0) > 15)[0]
    co_ocurrence_matrix = co_ocurrence_matrix[np.ix_(mask, mask)]

    top_classes_names = np.array(list(classes.values()))[mask].tolist()
    top_classes_ids = np.array(list(classes.keys()))[mask].tolist()

    fig, ax = plt.subplots(figsize=(15,15), dpi=200)
    ax.set_title("(1/Co-ocurrence) matrix values for the COCO objects \n(top " + str(len(mask)) + " (1/co-ocurrences))", size=20)
    heatmap = sns.heatmap(co_ocurrence_matrix, annot=False, linewidth=.5, xticklabels=top_classes_names, yticklabels=top_classes_names, ax=ax)
    heatmap.set_yticklabels(labels=heatmap.get_yticklabels(), rotation=0, fontsize=15)
    heatmap.set_xticklabels(labels=heatmap.get_xticklabels(), rotation=45, fontsize=15)
    fig.savefig("task_b_heatmap.png")


    image_count = 0
    for d in random.sample(dataset_dicts_val, len(dataset_dicts_val)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        is_class = False
        for obj1 in d['annotations']:

            object_class1 = obj1['category_id']

            if object_class1 in top_classes_ids:
                is_class = True
                break

        if not is_class:
            continue

        # We look for a second image
        for d2 in random.sample(dataset_dicts_val, len(dataset_dicts_val)):

            if d["file_name"] == d2["file_name"]:
                continue

            im2 = cv2.imread(d2["file_name"])
            #outputs2 = predictor(im2)

            is_class2 = False
            class_pos = 0
            selected_obj2 = None
            for o_i, obj2 in enumerate(d2['annotations']):

                if 'segmentation' not in obj2:
                    continue

                if len(obj2['segmentation']) == 0:
                    continue

                object_class2 = obj2['category_id']

                if object_class2 in top_classes_ids:
                    is_class2 = object_class2
                    class_pos = o_i
                    selected_obj2 = obj2
                    break

            if is_class2 == False:
                continue

            class_index = is_class2

            rles = mask_utils.frPyObjects(selected_obj2['segmentation'], im2.shape[1], im2.shape[0])
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle).astype(bool)

            #mask = outputs2["instances"].to("cpu").pred_masks[class_pos]
            
            a = np.where(mask != False)
            try:
                im2b = im2[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
                maskb = mask[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
            except:
                print('Empty mask')
                continue

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

            image_count = image_count + 1

            break
    
        if image_count == 5000:
            break
