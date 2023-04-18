import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import resnet18, resnet50
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import boxes as box_ops


import cv2


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


class EmbeddingNetImage(nn.Module):
    def __init__(self, weights, resnet_type='V1', with_fc = True):
        super(EmbeddingNetImage, self).__init__()
        self.with_fc = with_fc

        if resnet_type == 'V1':
            # Load the Faster R-CNN model with ResNet-50 backbone
            self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        elif resnet_type == 'V2':
            self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

        self.backbone = nn.Sequential(*list(self.faster_rcnn.backbone.children())[:-1])

        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
            print(name, param.requires_grad)

        # Replace the box predictor with a custom Fast R-CNN predictor
        in_features = self.faster_rcnn.roi_heads.box_head.fc7.in_features


        # Define the fully connected layers for embedding
        self.fc = nn.Sequential(nn.Sequential(nn.Linear(in_features, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 2)
                                              ))

    def forward(self, x):
        output = self.faster_rcnn(x)
        
        if self.with_fc:
            output = self.fc(output) # [images, 2]
        
        return output



class TripletNetIm2Text(nn.Module):
    def __init__(self, embedding_net_image, embedding_net_text):
        super(TripletNetIm2Text, self).__init__()
        self.embedding_net_image = embedding_net_image
        self.embedding_net_text = embedding_net_text

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net_image(x1)
        output2 = self.embedding_net_text(x2)
        output3 = self.embedding_net_text(x3)
        return output1, output2, output3

    def get_embedding_image(self, x):
        return self.embedding_net_image(x)
    
    def get_embedding_text(self, x):
        return self.embedding_net_text(x)
    
class TripletNetText2Img(nn.Module):
    def __init__(self, embedding_net_image, embedding_net_text):
        super(TripletNetText2Img, self).__init__()
        self.embedding_net_image = embedding_net_image
        self.embedding_net_text = embedding_net_text

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net_text(x1)
        output2 = self.embedding_net_image(x2)
        output3 = self.embedding_net_image(x3)
        return output1, output2, output3

    def get_embedding_image(self, x):
        return self.embedding_net_image(x)
    
    def get_embedding_text(self, x):
        return self.embedding_net_text(x)