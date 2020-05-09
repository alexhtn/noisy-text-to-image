import torch
from torch import nn
from .activation_glu import GLU
from .up_block import up_block


class InitialStage(nn.Module):
    def __init__(self, condition_dim, noise_dim, out_channels):
        super().__init__()
        self.in_channels = condition_dim + noise_dim
        self.out_channels = out_channels

        self.fc = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels * 16 * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.out_channels * 16 * 4 * 4 * 2),
            GLU())

        self.upsample1 = up_block(self.out_channels * 16, self.out_channels * 8)
        self.upsample2 = up_block(self.out_channels * 8, self.out_channels * 4)
        self.upsample3 = up_block(self.out_channels * 4, self.out_channels * 2)
        self.upsample4 = up_block(self.out_channels * 2, self.out_channels)

    def forward(self, conditional_features, noise):
        x = torch.cat((conditional_features, noise), 1)
        x = self.fc(x)
        x = x.view(-1, self.out_channels * 16, 4, 4)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)

        return x
