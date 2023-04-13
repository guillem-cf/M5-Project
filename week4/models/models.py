import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import resnet18, resnet50
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import boxes as box_ops


class EmbeddingNet(nn.Module):
    def __init__(self, weights, resnet_type='resnet50'):
        super(EmbeddingNet, self).__init__()

        if resnet_type == 'resnet50':
            self.resnet = resnet50(weights=weights)
            self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
            # print dimensionality of the last layer                          
            self.fc = nn.Sequential(nn.Linear(2048, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, 2)
                                    )
        elif resnet_type == 'resnet18':
            self.resnet = resnet18(weights=weights)
            self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

            self.fc = nn.Sequential(nn.Linear(512, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, 2)
                                    )

    def forward(self, x):
        output = self.resnet(x).squeeze()
        # output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


def is_target_empty(target):
    if target is None:
        return True

    if all(len(t['boxes']) == 0 and len(t['labels']) == 0 for t in target):
        return True

    return False


# No updates in faster RCNN
# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
class ObjectEmbeddingNet(nn.Module):
    def __init__(self, weights, num_classes):
        super(ObjectEmbeddingNet, self).__init__()

        self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(weights=weights)

        num_features_in = self.faster_rcnn.roi_heads.box_head.fc7.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(num_features_in, 91)

        self.fc = nn.Sequential(nn.Sequential(nn.Linear(1024, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 2)
                                              ))

        self.features = []
        def hook(module, input, output):
            self.features.append(output)

        self.faster_rcnn.roi_heads.box_head.fc7.register_forward_hook(hook)

    def extract_roi_features(self, images, targets=None):
        images, targets = self.faster_rcnn.transform(images, targets)
        features = self.faster_rcnn.backbone(images.tensors)
        proposals, proposal_losses = self.faster_rcnn.rpn(images, features, targets)
        detections, detector_losses = self.faster_rcnn.roi_heads(features, proposals, images.image_sizes, targets)
        return detections, detector_losses

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided")

        if self.training:
            n = self.faster_rcnn.backbone(images)
            # change format to calculate triplet loss
            n = list(n.values())

        else:
            detections, losses = self.extract_roi_features(images, targets)

        embeddings = []
        for image_detections in detections:
            indices = box_ops.batched_nms(
                image_detections['boxes'],
                image_detections['scores'],
                image_detections['labels'],
                self.threshold
            )
            selected_features = image_detections['roi_features'][indices]
            embeddings.append(torch.mean(selected_features, dim=0))

        return torch.stack(embeddings)

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet_fasterRCNN(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet_fasterRCNN, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3, target1, target2, target3):
        output1 = self.embedding_net(x1, target1)
        output2 = self.embedding_net(x2, target2)
        output3 = self.embedding_net(x3, target3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class BasicEmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2))

    def forward(self, x):
        output = self.convnet(x)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ObjectEmbeddingNet_v2(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(ObjectEmbeddingNet, self).__init__()
        self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.faster_rcnn.roi_heads.box_predictor.cls_score = nn.Identity()
        self.faster_rcnn.roi_heads.box_predictor.bbox_pred = nn.Identity()
        self.feature_extractor = nn.Sequential(
            nn.Linear(1024, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, images):
        features = []
        for image in images:
            detections = self.faster_rcnn(image)
            object_features = []
            for box in detections[0]['boxes']:
                object_image = image[:, box[1]:box[3], box[0]:box[2]]
                object_feature = self.faster_rcnn.backbone(object_image)
                object_features.append(object_feature)
            if len(object_features) > 0:
                object_features = torch.cat(object_features)
                object_features = self.feature_extractor(object_features)
                features.append(torch.mean(object_features, dim=0))
            else:
                features.append(torch.zeros(self.feature_dim))
        features = torch.stack(features)
        return features
