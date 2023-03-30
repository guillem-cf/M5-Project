import torch.nn as nn
from torchvision.models import resnet18


class SiameseResNet(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.resnet = resnet18(weights=weights)

        self.fc = nn.Sequential(nn.Linear(1000, 512), nn.PReLU(), nn.Linear(1024, 256), nn.PReLU(), nn.Linear(256, 2))

    def forward_once(self, x):
        x = self.resnet(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2
