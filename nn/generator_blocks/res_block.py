from torch import nn
from .activation_glu import GLU


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            GLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out
