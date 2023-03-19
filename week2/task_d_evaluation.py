# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import argparse

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from formatDataset import get_kitti_dicts, register_kitti_dataset

# import some common detectron2 utilities
from detectron2 import model_zoo

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser(description='Task C: Inference')
    parser.add_argument('--network', type=str, default='faster_RCNN', help='Network to use: faster_RCNN or mask_RCNN')
    args = parser.parse_args()

    dataset_dicts = get_kitti_dicts("val")
    # kitti_metadata = register_kitti_dataset("val")
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign', 'parking meter',
                'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    for subset in ["train", "val", "val_subset"]:
        DatasetCatalog.register(f"kitti_{subset}", lambda subset=subset: get_kitti_dicts(subset, pretrained=True))
        print(f"Successfully registered 'kitti_{subset}'!")
        MetadataCatalog.get(f"kitti_{subset}").set(thing_classes=classes)

    kitty_metadata = MetadataCatalog.get("kitti_train")

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
    cfg.DATASETS.TEST = ("kitti_val",)
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("kitti_val", cfg, False, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, "kitti_val")

    print(inference_on_dataset(predictor.model, val_loader, evaluator))