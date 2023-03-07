from torch import nn

from week1.models.base_model import BaseModel
from week1.models.components.cnn_block import cbrblock
from week1.models.components.residual_block import conv_block
from week1.models.components.simple_dense_net import SimpleDenseNet


class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        nChannels = [1, 16, 160, 320, 640]

        self.input_block = cbrblock(nChannels[0], nChannels[1])

        # Module with alternating components employing input scaling
        self.block1 = conv_block(nChannels[1], nChannels[2], 1)
        self.block2 = conv_block(nChannels[2], nChannels[2], 0)
        self.pool1 = nn.MaxPool2d(2)
        self.block3 = conv_block(nChannels[2], nChannels[3], 1)
        self.block4 = conv_block(nChannels[3], nChannels[3], 0)
        self.pool2 = nn.MaxPool2d(2)
        self.block5 = conv_block(nChannels[3], nChannels[4], 1)
        self.block6 = conv_block(nChannels[4], nChannels[4], 0)

        # Global average pooling
        self.pool = nn.AvgPool2d(7)

        # Feature flattening followed by linear layer
        self.flat = nn.Flatten()
        self.fc = nn.Linear(nChannels[4], num_classes)

    def forward(self, x):
        out = self.input_block(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool(out)
        out = self.flat(out)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    model = Model(784, 10)
    print(model)
