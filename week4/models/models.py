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
    def __init__(self, weights, resnet_type='V1', weighted = True):
        super(ObjectEmbeddingNet, self).__init__()
        
        self.weighted = weighted

        if resnet_type == 'V1':
            # Load the Faster R-CNN model with ResNet-50 backbone
            self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        elif resnet_type == 'V2':
            self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

        # Replace the box predictor with a custom Fast R-CNN predictor
        in_features = self.faster_rcnn.roi_heads.box_head.fc7.in_features


        # Define the fully connected layers for embedding
        self.fc = nn.Sequential(nn.Sequential(nn.Linear(in_features, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 2)
                                              ))

        self.features = []
        def hook_features(module, input, output):
            self.features.append(output)
        self.scores = []
        def hook_scores(module, input, output):
            self.scores.append(output)
        

        layer_to_hook_features = 'roi_heads.box_head.fc7'
        layer_to_hook_scores = 'roi_heads.box_predictor.cls_score'
        for name, layer in self.faster_rcnn.named_modules():
            if name == layer_to_hook_features:
                layer.register_forward_hook(hook_features)
            if name == layer_to_hook_scores:
                layer.register_forward_hook(hook_scores)
   

    def forward(self, x, targets=None):
        targets = {}
        targets['boxes'] = torch.zeros((0,4)).to(x.device)
        targets['labels'] = torch.zeros((0), dtype = torch.int64).to(x.device)
        targets['image_id'] = torch.zeros((0), dtype = torch.int64).to(x.device)
        
        targets = [targets]*x.shape[0]
        
        output = self.faster_rcnn(x, targets)
        
        
        scores = self.scores[0] # 512 boxes per images and 91 scores per box
        self.scores = []
        features = self.features[0] # 512 boxes per images and 1024 features per box
        self.features = []
        
        # Softmax over the scores and get the maximum
        scores_max = nn.functional.softmax(scores, dim=1) # 512 boxes per images and 91 scores per box
        scores_max = torch.max(scores, dim=1)[0] # 512 boxes per images and 1 score per box (the maximum score)
        
        
        if features.shape[0] != 512 * x.shape[0]:  # box_batch_size_per_image = 512  
            print('Number of boxes is not 512')
            # List with the number of boxes per image
            bbox_per_image = self.faster_rcnn.roi_heads.bboxes_per_image
            
            # Split the dim=0 of the features and scores tensors according to the number of boxes per image
            features_img = []
            features_split = []
            scores_split = []
            accumulated_boxes = 0
            for num_boxes in bbox_per_image:
                features_split = features[accumulated_boxes:accumulated_boxes+num_boxes]
                
                if self.weighted:
                    # Obtain weights for the features
                    scores_split = scores_max[accumulated_boxes:accumulated_boxes+num_boxes]
                    scores_split = nn.functional.softmax(scores_split, dim=0).unsqueeze(1)
                    
                    # Weighted features with scores
                    features_weight = torch.mul(features_split, scores_split)
                    
                    features_img.append(torch.sum(features_weight, dim=0))
                else:
                    # Non weighted features
                    features_img.append(torch.mean(features_split, dim=0))
                
                accumulated_boxes += num_boxes
                
                
                
            features_img = torch.stack(features_img, dim=0)
                 
        else:
            # print('Number of boxes is 512')
            features_split = torch.stack(torch.split(features, features.shape[0]//x.shape[0], dim=0),dim=0)
            
            if self.weighted:
                # Obtain weights for the features
                scores_split = torch.stack(torch.split(scores_max, scores_max.shape[0]//x.shape[0], dim=0),dim=0)
                scores_split = nn.functional.softmax(scores_split, dim=1).unsqueeze(2)
                
                # Weighted features with scores
                features_weight = torch.mul(features_split, scores_split)
                
                features_img = torch.sum(features_weight, dim=1)
            else:
                # Non weighted features
                features_img = torch.mean(features_split, dim=1)
            
        features_img = self.fc(features_img) # [images, 2]
        
        return features_img
    
    
    def get_embedding(self, x):
        return self.forward(x)
    
    # def multihead_attention(self, features, scores):
    # # Project the features tensor to a lower-dimensional space
    #     query = self.query_layer(features)  # shape: [B, 512, d]
    #     key = self.key_layer(features)  # shape: [B, 512, d]
    #     value = self.value_layer(features)  # shape: [B, 512, d]
        
    #     # Compute the attention scores using the queries, keys, and values
    #     attn_output, _ = self.attn_layer(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), 
    #                                       key_padding_mask=None, need_weights=False)
    #     # Transpose the output back to [B, 512, d]
    #     attn_output = attn_output.transpose(0, 1)  # shape: [B, 512, d]
        
    #     # Apply a final linear projection layer
    #     features_weighted = self.out_layer(attn_output)  # shape: [B, 512, d]
        
    #     # Compute the attention scores using the class scores
    #     attn_scores = torch.softmax(scores, dim=2)  # shape: [B, 512, 91]
        
    #     # Weight the feature vectors using the attention scores
    #     features_weighted = torch.sum(features_weighted * attn_scores.unsqueeze(2), dim=1)  # shape: [B, d]
        
    #     return features_weighted


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


# class TripletNet_fasterRCNN(nn.Module):
#     def __init__(self, embedding_net):
#         super(TripletNet_fasterRCNN, self).__init__()
#         self.embedding_net = embedding_net

#     # def forward(self, x1, x2, x3, target1, target2, target3):
#         # output1 = self.embedding_net(x1, target1)
#         # output2 = self.embedding_net(x2, target2)
#         # output3 = self.embedding_net(x3, target3)
#         # return output1, output2, output3
#     def forward(self, x1,x2,x3):
#         output1 = self.embedding_net(x1)
#         output2 = self.embedding_net(x2)
#         output3 = self.embedding_net(x3)
#         return output1, output2, output3

#     def get_embedding(self, x):
#         return self.embedding_net(x)


# class BasicEmbeddingNet(nn.Module):
#     def __init__(self):
#         super(EmbeddingNet, self).__init__()

#         self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
#                                      nn.MaxPool2d(2, stride=2),
#                                      nn.Conv2d(32, 64, 5), nn.PReLU(),
#                                      nn.MaxPool2d(2, stride=2))

#         self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
#                                 nn.PReLU(),
#                                 nn.Linear(256, 256),
#                                 nn.PReLU(),
#                                 nn.Linear(256, 2))

#     def forward(self, x):
#         output = self.convnet(x)
#         output = self.fc(output)
#         return output

#     def get_embedding(self, x):
#         return self.forward(x)


# class ObjectEmbeddingNet_v2(nn.Module):
#     def __init__(self, num_classes, feature_dim):
#         super(ObjectEmbeddingNet, self).__init__()
#         self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#         self.faster_rcnn.roi_heads.box_predictor.cls_score = nn.Identity()
#         self.faster_rcnn.roi_heads.box_predictor.bbox_pred = nn.Identity()
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(1024, feature_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim, feature_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim, feature_dim),
#         )

#     def forward(self, images):
#         features = []
#         for image in images:
#             detections = self.faster_rcnn(image)
#             object_features = []
#             for box in detections[0]['boxes']:
#                 object_image = image[:, box[1]:box[3], box[0]:box[2]]
#                 object_feature = self.faster_rcnn.backbone(object_image)
#                 object_features.append(object_feature)
#             if len(object_features) > 0:
#                 object_features = torch.cat(object_features)
#                 object_features = self.feature_extractor(object_features)
#                 features.append(torch.mean(object_features, dim=0))
#             else:
#                 features.append(torch.zeros(self.feature_dim))
#         features = torch.stack(features)
#         return features

class ObjectEmbeddingNetV2(nn.Module):
    
    def __init__(self, weights = None, resnet_type = None,  weighted = False, with_fc = True):
        """
        Loads the pretrained Resnet50 in COCO (from pretrained FasterRCNN)

        Returns
        -------
        None.

        """
        super().__init__()
        
        # Mode
        self.weighted = weighted
        self.with_fc = with_fc
        
        # Create the model
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size = 240, max_size = 320)
        
        self.features = []
        self.scores = []
        self.proposals = []
        def save_features(mod, inp, outp):
            self.features.append(outp)
        def save_scores(mod, inp, outp):
            self.scores.append(outp)
        # def save_proposals(mod, inp, outp):
        #     self.proposals.append(inp)
        
        # you can also hook layers inside the roi_heads
        layerFeatures = 'roi_heads.box_head.fc7'
        layerClasses = "roi_heads.box_predictor.cls_score"
        # layerProposals = "roi_heads.box_roi_pool"#­"rpn.head.cls_logits"
        for name, layer in model.named_modules():
            if name == layerFeatures:
                layer.register_forward_hook(save_features)
            
            if name == layerClasses:
                layer.register_forward_hook(save_scores)
                
            # if name == layerProposals:
            #     layer.register_forward_hook(save_proposals)
    
        self.model = model
        
        # Replace the box predictor with a custom Fast R-CNN predictor
        in_features = self.model.roi_heads.box_head.fc7.in_features

        if self.with_fc:
            # Define the fully connected layers for embedding
            self.fc = nn.Sequential(nn.Sequential(nn.Linear(in_features, 256),
                                                nn.PReLU(),
                                                nn.Linear(256, 256),
                                                nn.PReLU(),
                                                nn.Linear(256, 2)
                                                ))
        
    def forward(self,x):
        
        target = {}
        target["boxes"] = torch.zeros((0,4)).to("cuda")
        target["labels"] = torch.zeros((0), dtype = torch.int64).to("cuda")
        target["image_id"] = torch.zeros((0), dtype = torch.int64).to("cuda")
        
        targets = [target]*x.shape[0]
        
        features = self.model(x, targets)
        
        
        # Obtain stored data
        proposalsScores = self.scores[0]
        del self.scores[0]
        # proposals = self.proposals[0]
        # del self.proposals[0]
        proposalsFeatures = self.features[0]
        del self.features[0]
        
        # Proposals per image        
        bbox_per_image = proposalsFeatures.shape[0]//x.shape[0]
        features_per_image = [bbox_per_image]*x.shape[0]
        
        # Split proposals into images
        proposalsFeatures = proposalsFeatures.split(features_per_image, 0)
        
        
        # Softmax per BBox proposal
        proposalsScores = torch.nn.functional.softmax(proposalsScores, dim = 1)
        # Get max value (confidence) per proposal
        proposalsScores = torch.max(proposalsScores, dim = 1)[0]
        # Split into images
        proposalsScores = proposalsScores.split(features_per_image, 0)
        # Obtain weights from confidences
        proposalsScores = [torch.nn.functional.softmax(a, dim = 0).unsqueeze(1) for a in proposalsScores]
        
        if self.weighted:
            # Use weighted sum of proposals
            features = [proposalsFeatures[i]*proposalsScores[i] for i in range(len(proposalsScores))]
            features = [torch.sum(f, dim = 0).unsqueeze(0) for f in features]
        else:
            # Use mean of all proposals
            features = proposalsFeatures
            features = [torch.mean(f, dim = 0).unsqueeze(0) for f in features]
        
        # Stack all images features
        features = torch.vstack(features)
        
        if self.with_fc:
            # Pass features through fully connected layers
            features = self.fc(features)
        
        return features
    