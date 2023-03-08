import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet_Convblock(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, filters, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding='same')
        self.batch_norm = nn.BatchNorm2d(filters)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batch_norm(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = ResNet_Convblock(3, 32)
        self.conv_block2 = ResNet_Convblock(32, 32)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 8)

        self.dropout6 = nn.Dropout(0.6)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv_block1(x)
        x1 = x.clone()

        x = self.conv_block2(x)
        x2 = x.clone()
        x = x1 + x2

        x = self.conv_block2(x)
        x3 = x.clone()
        x = x1 + x2 + x3

        x = self.conv_block2(x)
        x4 = x.clone()
        x = x1 + x2 + x3 + x4

        x = nn.AdaptiveAvgPool2d((1, 1))(x)

        x = F.relu(self.fc1(x))
        x = self.dropout6(x)

        x = F.relu(self.fc2(x))
        x = self.dropout5(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        x = F.softmax(x)

        return x
