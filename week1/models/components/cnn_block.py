from torch import nn


class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=(1, 1),
                      padding='same', bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
            )

    def forward(self, x):
        out = self.cbr(x)
        return out
