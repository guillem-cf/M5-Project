import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import resnet18, resnet50
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList

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
        

        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # # Remove the softmax layer
        # self.faster_rcnn.roi_heads.box_predictor.cls_score = nn.Identity()

        self.fc = nn.Sequential(nn.Sequential(nn.Linear(1024, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 2)
                                              ))

    def forward(self, x, target):
        # detections = self.faster_rcnn(x, target)
        # object_embeddings = []
        # images, target = self.faster_rcnn.transform(x, target)
        # images = ImageList(x, [(x.size(2), x.size(3))] * x.size(0))
        # and list is not empty
        
        # Get the features of the detections in the last softmax layer
        
        images, _ = self.faster_rcnn.transform(x)
        features = self.faster_rcnn.backbone(images.tensors)
        proposals, _ = self.faster_rcnn.rpn(images, features, targets=target)  
        roi_features = self.faster_rcnn.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = self.faster_rcnn.roi_heads.box_head(roi_features)
        
        class_logits = self.faster_rcnn.roi_heads.box_predictor.cls_score(box_features)
        score = torch.nn.functional.softmax(class_logits, -1)
        # PROBLEMA AQUÍ, TROBEM CONFIDENCESMOLTBAIXES
        confidences, _ = torch.max(score, dim=1)
        confidences_split = torch.split(confidences, confidences.shape[0]//x.shape[0], dim=0)
        
        box_features_2dim = self.fc(box_features)
        box_features_2dim = torch.split(box_features_2dim, box_features_2dim.shape[0]//x.shape[0], dim=0)
      
        
        # Construct a tensor with dimensions (batch_size, num_detections, 2)
        for i in range(x.shape[0]):
            object_embeddings = []
            for j in range(box_features_2dim[i].shape[0]):
                if confidences_split[i][j] > 0.5:
                    object_embeddings.append(box_features_2dim[i][j])
            object_embeddings = torch.stack(object_embeddings)
            object_embeddings = object_embeddings.unsqueeze(0)
            if i == 0:
                object_embeddings_all = object_embeddings
            else:
                object_embeddings_all = torch.cat((object_embeddings_all, object_embeddings), dim=0)
        
        # Gloabal average pooling to get (batch_size, 2) tensor
        image_embeddings = torch.mean(object_embeddings_all, dim=1)
        
        
        

        
        # detections, _ = self.faster_rcnn.roi_heads(features, proposals, images.image_sizes, targets=target)
        
        # if detections == []:
        #     # Return a zero tensor if no proposals are generated
        #     return torch.zeros(x.size(0), 2), None
        
        # scores = detections[0]['scores']  # Get the scores of the detections
        # idx = torch.argmax(scores)  # Get the index of the detection with highest score
        # fc7_output = self.faster_rcnn.roi_heads.box_head(detections[0]['features'][[idx]])  # Get the features of the highest scoring detection and pass through the box head
        # fc7_output = fc7_output.view(fc7_output.size(0), -1)  # Flatten the features
        # fc7_output = self.fc(fc7_output)  # Pass through the fc layers to get the object embeddings
   
   
        # detections, _ = self.faster_rcnn.roi_heads(features, proposals, images.image_sizes, targets=target)
        # outputs = self.faster_rcnn(x, target)
        # feature_vectors = self.faster_rcnn.roi_heads.box_head.fc7(outputs[0]['boxes'])
     
        # object_embeddings = []
        # for box in fc7_output:
        #     # object_features = detection['features']
        #     object_embeddings.append(self.fc(fc7_output))
        
        # if not object_embeddings:
        #     # zero_tensor = torch.zeros(x.size(0), 2)
        #     # zero_tensor.requires_grad = True  # Set requires_grad=True for the zero tensor
        #     # return zero_tensor
        #     object_embeddings = 1
  
        # object_embeddings = torch.stack(fc7_output).mean(dim=0)
        
        
        return image_embeddings

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