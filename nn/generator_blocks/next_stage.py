import torch
from torch import nn
from ..attention.layer import GlobalAttentionLayer
from .up_block import up_block
from .res_block import ResBlock


class NextStage(nn.Module):
    def __init__(self, out_channels, word_dim):
        super().__init__()

        self.attention = GlobalAttentionLayer(out_channels, word_dim)
        self.residual = nn.Sequential(
            ResBlock(out_channels * 2),
            ResBlock(out_channels * 2)
        )
        self.upsample = up_block(out_channels * 2, out_channels)

    def forward(self, previous_feature_map, words_emb):
        c_code, att = self.attention(previous_feature_map, words_emb)
        h_c_code = torch.cat((previous_feature_map, c_code), 1)
        out_code = self.residual(h_c_code)
        out_code = self.upsample(out_code)

        return out_code, att
