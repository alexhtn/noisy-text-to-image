import torch
from torch import nn
from .common import block3x3_leak_relu


class ConditionalOutput(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.channels = channels
        self.condition_dim = condition_dim

        self.joint_conv = block3x3_leak_relu(self.channels + self.condition_dim, self.channels)
        self.final_conv = nn.Conv2d(self.channels, 1, kernel_size=4, stride=4)

    def forward(self, x, condition_features):
        # conditioning output
        condition_features = condition_features.view(-1, self.condition_dim, 1, 1)
        condition_features = condition_features.repeat(1, 1, 4, 4)

        # state size (channels + condition_dim) x 4 x 4
        x = torch.cat((x, condition_features), 1)

        # state size channels x 4 x 4
        x = self.joint_conv(x)

        x = self.final_conv(x)
        x = torch.sigmoid(x)
        x = x.view(-1)

        return x
