import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

from torchvision import models
from torchvision.models import resnet18, resnet50



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

