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


# No updates in faster RCNN
# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
class ObjectEmbeddingNet(nn.Module):
    def __init__(self, weights, num_classes):
        super(ObjectEmbeddingNet, self).__init__()

        # Load the Faster R-CNN model with ResNet-50 backbone
        self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(weights=weights)

        # Replace the box predictor with a custom Fast R-CNN predictor
        in_features = self.faster_rcnn.roi_heads.box_head.fc7.in_features
        # self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 91)   # El nostre dataset te 80 classes, no 91. Per aixo passavem num_classes

        # Define the fully connected layers for embedding
        self.fc = nn.Sequential(nn.Sequential(nn.Linear(in_features, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 2)
                                              ))

        self.proposals = []
        def hook_proposals(module, input, output):
            self.proposals.append(input)
        self.features = []
        def hook_features(module, input, output):
            self.features.append(output)
        self.scores = []
        def hook_scores(module, input, output):
            self.scores.append(output)

        
        layer_to_hook_proposals = 'roi_heads.box_roi_pool'
        layer_to_hook_features = 'roi_heads.box_head.fc7'
        layer_to_hook_scores = 'roi_heads.box_predictor.cls_score'
        for name, layer in self.faster_rcnn.named_modules():
            if name == layer_to_hook_proposals:
                layer.register_forward_hook(hook_proposals)
            if name == layer_to_hook_features:
                layer.register_forward_hook(hook_features)
            if name == layer_to_hook_scores:
                layer.register_forward_hook(hook_scores)
        
        
        # # Define a hook to extract the object features from the Fast R-CNN head
        # self.features = []
        # def hook(module, input, output):
        #     self.features.append(output)

        # self.faster_rcnn.roi_heads.box_head.fc7.register_forward_hook(hook)

    def extract_roi_features(self, images, targets=None):
        # Apply the transform to the input images
        images, targets = self.faster_rcnn.transform(images, targets)   # Aqui suposo que es fa el resize de les bounding boxes del GT, si no surten be els resultats es una cosa que podriem revisar
        
        # Pass the images through the Faster R-CNN backbone and RPN
        features = self.faster_rcnn.backbone(images.tensors)
        proposals, proposal_losses = self.faster_rcnn.rpn(images, features, targets)
        
        # RoI pooling
        object_features = self.faster_rcnn.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        
        # Pass the features through the Fast R-CNN head
        bbox_features = self.faster_rcnn.roi_heads.box_head(object_features)
        
        # Scores and bounding boxes
        scores = self.faster_rcnn.roi_heads.box_predictor(bbox_features)
        
        # # Pass the features and proposals through the Fast R-CNN head
        # detections, detector_losses = self.faster_rcnn.roi_heads(features, proposals, images.image_sizes, targets)
        
        return bbox_features, scores

    def forward(self, x, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided") 
        
        # Write the detections to the images to check the results
        im_1 = x[0].permute(1, 2, 0).cpu().numpy()
        # im_2 = x[1].permute(1, 2, 0).cpu().numpy()
        
        # use cv2 to draw the bounding boxes
        im_1 = cv2.cvtColor(im_1, cv2.COLOR_RGB2BGR)
        # im_2 = cv2.cvtColor(im_2, cv2.COLOR_RGB2BGR)
        
        # im_1_boxes = detections[0]['boxes'].detach().cpu().numpy()
        im_1_boxes_gt = targets[0]['boxes'].detach().cpu().numpy()
        # im_2_boxes = detections[1]['boxes'].detach().cpu().numpy()
        # im_2_boxes_gt = targets[1]['boxes'].detach().cpu().numpy()
        
        for box_gt in list(im_1_boxes_gt):
        #     cv2.rectangle(im_1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.rectangle(im_1, (int(box_gt[0]), int(box_gt[1])), (int(box_gt[2]), int(box_gt[3])), (255, 0, 0), 2)
            
        # for box_gt in list(im_2_boxes_gt):
        #     # cv2.rectangle(im_2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        #     cv2.rectangle(im_2, (int(box_gt[0]), int(box_gt[1])), (int(box_gt[2]), int(box_gt[3])), (255, 0, 0), 2) 

        # images,targets = self.faster_rcnn.transform(images,targets)
        # outputs = self.faster_rcnn(images, targets) [4000, 1024]
        # bbox_features, scores = self.extract_roi_features(images, targets)
        # # Calculate the most probable class (ignoring the background class) and their scores
        #         max_scores, max_classes = torch.max(scores[:, 1:], dim=1)
        #
        #         # Filter RoIs based on the confidence threshold
        #         mask = max_scores > confidence_threshold
        #         filtered_bbox_features = bbox_features[mask]

        # Pass the filtered features through the fully connected layers
        embeddings = self.fc(filtered_bbox_features)

        # Filter the corresponding scores and classes
        # filtered_scores = max_scores[mask]
        # filtered_classes = max_classes[mask] + 1  # Add 1 to the indices to account for the background class
        
        targets = {}
        targets['boxes'] = torch.zeros((0,4)).to(x.device)
        targets['labels'] = torch.zeros((0), dtype = torch.int64).to(x.device)
        targets['image_id'] = torch.zeros((0), dtype = torch.int64).to(x.device)
        
        targets = [targets]*x.shape[0]
        
        output = self.faster_rcnn(x, targets)
        
        proposals = self.proposals[0]
        self.proposals = []
        scores = self.scores[0]
        self.scores = []
        features = self.features[0]
        self.features = []
        
        
        
        
        
    
        # if self.training:
        #     # Apply the transform to the input images
        #     images, targets = self.faster_rcnn.transform(x, targets)   # Aqui suposo que es fa el resize de les bounding boxes del GT, si no surten be els resultats es una cosa que podriem revisar
            
        #     # Pass the images through the Faster R-CNN backbone and RPN
        #     features = self.faster_rcnn.backbone(images.tensors)
        #     proposals, proposal_losses = self.faster_rcnn.rpn(images, features, targets)
            
        #     # RoI pooling
        #     object_features = self.faster_rcnn.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
            
        #     # Pass the features through the Fast R-CNN head
        #     bbox_features = self.faster_rcnn.roi_heads.box_head(object_features)
            
        #     # Pass the features through the fully connected layers [4000,2] # Problem here, not always dimensionality 4000!!
        #     bbox_features = self.fc(bbox_features)
            
        #     # Split the features for each image in the batch and construct a tensor with dim [batch_size, num_objects, 2]
        #     bbox_features_split = torch.split(bbox_features, bbox_features.shape[0]//x.shape[0], dim=0)
        #     # bbox_features_split = [bbox_features_split[i].squeeze() for i in range(len(bbox_features_split))]
        #     bbox_features_split = torch.mean(torch.stack(bbox_features_split, dim=0), dim=1)
            
        #     # If some bbox_featurs_split is nan or inf, replace it with 0
        #     if torch.isnan(bbox_features_split).any():
        #         bbox_features_split[bbox_features_split == torch.nan] = 0
                
        #     return bbox_features_split
        
        # else:
        #     output = self.faster_rcnn(x)
            
            # Extract the features for the selected objects
            

        # embeddings = []
        # for image_detections in detections:
        #     # indices = box_ops.batched_nms(
        #     #     image_detections['boxes'],
        #     #     image_detections['scores'],
        #     #     image_detections['labels'],
        #     #     self.threshold
        #     # )
        #     # selected_features = image_detections['roi_features'][indices]
        #     # embeddings.append(torch.mean(selected_features, dim=0))
            
        #     # Compute the features for the selected objects
        #     object_features = 1

        # embeddings = torch.stack(embeddings)
        # embeddings = self.fc(embeddings)
        
        # # TODO: PASSAR LA FULLY CONNECTED PER TENIR ELS EMBEDDINGS DE CADA IMATGE DE DIMENSIO 2

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