import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import resnet18, resnet50
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


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


# No updates in faster RCNN
# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
class ObjectEmbeddingNet(nn.Module):
    def __init__(self, weights, num_classes):
        super(ObjectEmbeddingNet, self).__init__()

        self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        # in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        # self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Remove the softmax layer
        self.faster_rcnn.roi_heads.box_predictor.cls_score = nn.Identity()

        self.fc = nn.Sequential(nn.Sequential(nn.Linear(2048, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 2)
                                              ))

    def forward(self, x, target):
        # detections = self.faster_rcnn(x, target)
        # object_embeddings = []

        images = list(x)
        features = self.faster_rcnn.backbone(x)  # Extract features from backbone
        proposals, _ = self.faster_rcnn.rpn(images, features)  # ARREGLAR AIXO
        detections, _ = self.faster_rcnn.roi_heads(features, proposals, images.image_sizes)  # ARREGLAR AIXO

        object_embeddings = []
        for detection in detections:
            object_features = detection['features']
            object_embeddings.append(self.fc(object_features))

        object_embeddings = torch.stack(object_embeddings).mean(dim=0)

        return object_embeddings

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