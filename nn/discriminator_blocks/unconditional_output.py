import torch
from torch import nn


class UnconditionalOutput(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.final_conv = nn.Conv2d(self.channels, 1, kernel_size=4, stride=4)

    def forward(self, x):
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        x = x.view(-1)
        return x
