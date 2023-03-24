import random

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

wandb.init(project="resnet_50_style_transfer")

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


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """

    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()

    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram


img_size = 512


def load_image(input_image, maxsize=400):
    if max(input_image.size) > maxsize:
        size = maxsize
    else:
        size = max(input_image.size)

    preprocess = transforms.Compose([
        transforms.Resize(size),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    return input_batch


def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image, model):
    x = image
    features = {}
    # -_-
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    i = 0

    for name, layer in model.layer1._modules.items():
        x = layer(x)
        i += 1
        features[f"bneck_{name}_{i}"] = x

    for name, layer in model.layer2._modules.items():
        x = layer(x)
        i += 1
        features[f"bneck_{name}_{i}"] = x

    for name, layer in model.layer3._modules.items():
        x = layer(x)
        i += 1
        features[f"bneck_{name}_{i}"] = x

    for name, layer in model.layer4._modules.items():
        x = layer(x)
        i += 1
        features[f"bneck_{name}_{i}"] = x

    return features


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
    register_coco_instances("MSCOCO_val", {}, "../../dataset/annotations/instances_val2017.json",
                            "/ghome/group03/val2017")

    # Register ms coco dataset test
    register_coco_instances("MSCOCO_test", {}, "/ghome/group03/annotations/image_info_test2017.json",
                            "/ghome/group03/test2017")

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

    """
    # Evaluator
    evaluator = COCOEvaluator("MSCOCO_val", cfg, False, output_dir=output_path)

    # Evaluate the model
    val_loader = build_detection_test_loader(cfg, "MSCOCO_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    
    """

    # dataset_dicts = get_ooc_dicts('val', pretrained=True)
    dataset_dicts = DatasetCatalog.get("MSCOCO_test")

    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()

    for d in random.sample(dataset_dicts, 20):
        img_2 = random.sample(dataset_dicts, 1)
        img_1 = cv2.imread(d["file_name"])
        img_2 = cv2.imread(d["file_name"])

        # TODO: POSSIBILITAT DE FER SEGMENTATION O DIRECTAMENT LA IMATGE
        # PREDICCIO DE LA IMATGE
        cv2.imwrite(output_path + d["file_name"].split('/')[-1], img_1)
        outputs_img1 = predictor(img_1)

        # IMATGE 2 ORIGINAL
        cv2.imwrite(output_path + d["file_name"].split('/')[-1], img_2)

        # load images, ordered as [style_image, content_image]

        content = load_image(outputs_img1)
        style = load_image(img_2)

        content_features = get_features(content, resnet50)
        style_features = get_features(style, resnet50)

        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

        style_weights = {
            'bneck_0_1': 19,
            'bneck_1_2': 0,
            'bneck_2_3': 7,
            'bneck_0_4': 0,
            'bneck_1_5': 4,
            'bneck_2_6': 3,
            'bneck_3_7': 2,
            'bneck_0_8': 1,
            'bneck_1_9': 1,
            'bneck_2_10': 1,
            'bneck_3_11': 0.5,
            'bneck_4_12': 0.1,
            'bneck_5_13': 0.5,
            'bneck_0_14': 0.01,
            'bneck_1_15': 0.005,
            'bneck_2_16': 0.001
        }

        content_weight = 1  # alpha
        style_weight = 1e7  # beta
        target = content.clone().requires_grad_(True).to(device)

        # for displaying the target image, intermittently
        show_every = 200

        # iteration hyperparameters
        optimizer = optim.Adam([target], lr=0.003)
        steps = 20000  # decide how many iterations to update your image (5000)

        for ii in range(1, steps + 1):

            # get the features from your target image
            target_features = get_features(target, resnet50)

            # the content loss
            content_loss = torch.mean((target_features['bneck_4_12'] - content_features['bneck_4_12']) ** 2)

            # the style loss
            # initialize the style loss to 0
            style_loss = 0
            # then add to it for each layer's gram matrix loss
            for layer in style_weights:
                # get the "target" style representation for the layer
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                # get the "style" style representation
                style_gram = style_grams[layer]
                # the style loss for one layer, weighted appropriately
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)

            # calculate the *total* loss
            total_loss = content_weight * content_loss + style_weight * style_loss

            wandb.log({"total_loss": total_loss, "content_loss": content_loss, "style_loss": style_loss})

            # update your target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # display intermediate images and print the loss
            if ii % show_every == 0:
                print('Total loss: ', total_loss.item())
                plt.imshow(im_convert(target))
                plt.savefig("RES/img-" + str(ii) + ".jpg")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        # content and style ims side-by-side
        ax1.set_axis_off()
        ax1.imshow(im_convert(content))
        ax2.set_axis_off()
        ax2.imshow(im_convert(style))
        ax3.set_axis_off()
        ax3.imshow(im_convert(target))
        fig.savefig(output_path + str(id(target)) + ".jpg")
