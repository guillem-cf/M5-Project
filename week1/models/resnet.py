import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = nn.functional.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.block1 = ResNetBlock(32, 32)
        self.block2 = ResNetBlock(32, 32)
        self.block3 = ResNetBlock(32, 32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 8)
        self.init_layers()

    def init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.pool(out)
        # out = torch.flatten(out, start_dim=1)
        # change with average pooling
        out = nn.AdaptiveAvgPool2d((1, 1))(out).squeeze()
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    model = ResNet()
    print(model)

    x = torch.randn(1, 3, 32, 32)
    print(model(x).shape)
