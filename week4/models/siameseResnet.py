import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18


class SiameseResNet(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.resnet = resnet18(weights=weights)

    def forward_once(self, x):
        x = self.resnet(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2
