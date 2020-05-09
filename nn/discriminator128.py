from torch import nn
from .discriminator_blocks.common import encode_image_by_16times, down_block, block3x3_leak_relu
from .discriminator_blocks.conditional_output import ConditionalOutput
from .discriminator_blocks.unconditional_output import UnconditionalOutput


class Discriminator128(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.block16 = encode_image_by_16times(channels)
        self.block32_1 = down_block(channels * 8, channels * 16)
        self.block32_2 = block3x3_leak_relu(channels * 16, channels * 8)

        self.conditional_output = ConditionalOutput(channels * 8, condition_dim)
        self.unconditional_output = UnconditionalOutput(channels * 8)

    def forward(self, x):
        x = self.block16(x)
        x = self.block32_1(x)
        x = self.block32_2(x)
        return x
