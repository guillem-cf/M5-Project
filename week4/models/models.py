import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision.models import resnet18


class ClassNet(nn.Module):

    # NETWORK FROM SLIDES
    def __init__(self):
        super(ClassNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(10816, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward_once(self, x):
        return self.forward(x)


class SiameseResNet(nn.Module):
    def __init__(self, weights):
        super(SiameseResNet, self).__init__()
        self.resnet = resnet18(weights=weights)
        self.class_net = ClassNet()

    def forward_once(self, x):
        return self.class_net(x)

    def forward(self, x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        return x1, x2


class TripletResNet(nn.Module):
    def __init__(self, class_net):
        super(TripletResNet, self).__init__()
        self.class_net = class_net

    def forward_once(self, x):
        return self.class_net(x)

    def forward(self, x1, x2, x3):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        x3 = self.forward_once(x3)
        return x1, x2, x3
