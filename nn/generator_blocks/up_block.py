from torch import nn
from .activation_glu import GLU


def up_block(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes * 2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block
