import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

from torchvision import models
from torchvision.models import resnet18, resnet50


# class EmbeddingNet(nn.Module):
#     def __init__(self, in_features_resnet):
#         super(EmbeddingNet, self).__init__()

#         # NETWORK FROM SLIDES
#         self.convnet = nn.Sequential(
#                                      nn.Conv2d(3, 32, 5), nn.PReLU(),
#                                      nn.MaxPool2d(2, stride=2),
#                                      nn.Conv2d(32, 64, 5), nn.PReLU(),
#                                      nn.MaxPool2d(2, stride=2))

#         self.fc = nn.Sequential(nn.Linear(in_features_resnet, 256),
#                                 nn.PReLU(),
#                                 nn.Linear(256, 256),
#                                 nn.PReLU(),
#                                 nn.Linear(256, 2)
#                                 )

#     def forward(self, x):
#         x = self.convnet(x)
#         x = self.fc(x)
#         return x

#     def forward_once(self, x):
#         return self.forward(x)


# class SiameseResNet(nn.Module):
#     def __init__(self, weights=None):
#         super(SiameseResNet, self).__init__()
#         self.resnet = resnet50(weights=weights)
#         # remove last layer
#         self.class_net = EmbeddingNet(self.resnet.fc.in_features)

#         self.resnet.fc = self.class_net

#     def forward_once(self, x):
#         x = self.resnet(x)
#         return x

#     def forward(self, x1, x2):
#         x1 = self.forward_once(x1)
#         x2 = self.forward_once(x2)
#         return x1, x2


class EmbeddingNet(nn.Module):
    def __init__(self, weights):
        super(EmbeddingNet, self).__init__()
        self.resnet = resnet50(weights=weights)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        # print dimensionality of the last layer
                                
        self.fc = nn.Sequential(nn.Linear(2048, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )
                                
                                           
        # self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
        #                 nn.MaxPool2d(2, stride=2),
        #                 nn.Conv2d(32, 64, 5), nn.PReLU(),
        #                 nn.MaxPool2d(2, stride=2))
        
        # self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
        #                 nn.PReLU(),
        #                 nn.Linear(256, 256),
        #                 nn.PReLU(),
        #                 nn.Linear(256, 2))
        
    def forward(self, x):
        output = self.resnet(x).squeeze()
        # output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
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



class TripletResNet(nn.Module):
    def __init__(self, weights=None):
        super(TripletResNet, self).__init__()
        self.resnet = resnet50(weights=weights)
        # remove last layer
        self.resnet.fc = Sequential(*list(self.resnet.fc.children())[:-1])
        self.class_net = EmbeddingNet()

    def forward_once(self, x):
        x = self.resnet(x)
        x = self.class_net(x)
        return x

    def forward(self, x1, x2, x3):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        x3 = self.forward_once(x3)
        return x1, x2, x3


