from torch import nn
from .discriminator_blocks.common import encode_image_by_16times
from .discriminator_blocks.conditional_output import ConditionalOutput
from .discriminator_blocks.unconditional_output import UnconditionalOutput


class Discriminator64(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.block16 = encode_image_by_16times(channels)

        self.conditional_output = ConditionalOutput(channels * 8, condition_dim)
        self.unconditional_output = UnconditionalOutput(channels * 8)

    def forward(self, x):
        x = self.block16(x)
        return x
