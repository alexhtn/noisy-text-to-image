import torch
from torch import nn


class GetImage(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=(3, 3), stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x
