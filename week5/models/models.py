import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import resnet18, resnet50
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import boxes as box_ops
import fasttext


import cv2


class EmbeddingNetImage(nn.Module):
    def __init__(self, weights, resnet_type='V1', dim_out_fc = 'as_image'):   # dim_out_fc = 'as_image' or 'as_text'
        super(EmbeddingNetImage, self).__init__()
        
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
        in_features = self.faster_rcnn.roi_heads.box_head.fc7.in_features  # 2048

        if dim_out_fc == 'as_image':
            # Define the fully connected layers for embedding
            self.fc = nn.Linear(in_features, in_features)
        elif dim_out_fc == 'as_text':
            self.fc = nn.Linear(in_features, 300)
            

    def forward(self, x):
        output = self.faster_rcnn(x)
        output = self.fc(output) 
        
        return output



class EmbeddingNetText(nn.Module):
    def __init__(self, weights, type='FastText', dim_out_fc = 'as_image'):  # type = 'FastText' or 'BERT'
        super(EmbeddingNetText, self).__init__()
        
        # TODO
        self.model = fasttext.load_model(weights)

        if dim_out_fc == 'as_image':
            self.fc = nn.Linear(300, 2048)
        elif dim_out_fc == 'as_text':
            self.fc = nn.Linear(300, 300)

    def forward(self, x):
        output = self.model[x]
        output = self.fc(output) 
        
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