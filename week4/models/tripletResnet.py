import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class TripletResNet(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.resnet = resnet50(weights=weights)
        self.fc = nn.Linear(self.resnet.fc.in_features, 8)

    def forward_once(self, x):
        x = self.resnet(x)
        return x

    def forward(self, x1, x2, x3):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        out3 = self.forward_once(x3)
        return out1, out2, out3
