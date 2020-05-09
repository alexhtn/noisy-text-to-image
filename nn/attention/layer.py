import torch
from torch import nn


class GlobalAttentionLayer(nn.Module):
    def __init__(self, idf, word_dim):
        super().__init__()
        self.conv_context = nn.Conv2d(word_dim, idf, kernel_size=1, stride=1,
                                      padding=0, bias=False)

    def forward(self, x, context):
        """
            input: batch x idf x ih x iw (query_length=ihxiw)
            context: batch x cdf x source_length
        """
        ih, iw = x.size(2), x.size(3)
        query_length = ih * iw
        batch_size, source_length = context.size(0), context.size(2)

        # --> batch x query_length x idf
        target = x.view(batch_size, -1, query_length)
        target_t = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x source_length --> batch x cdf x source_length x 1
        source_t = context.unsqueeze(3).contiguous()
        # --> batch x idf x source_length
        source_t = self.conv_context(source_t).squeeze(3).contiguous()

        # Get attention
        # (batch x query_length x idf)(batch x idf x source_length)
        # -->batch x query_length x source_length
        attn = torch.bmm(target_t, source_t)
        # --> batch*query_length x source_length
        attn = attn.view(batch_size*query_length, source_length)

        attn = torch.softmax(attn, dim=1)  # Eq. (2)
        # --> batch x query_length x source_length
        attn = attn.view(batch_size, query_length, source_length)
        # --> batch x source_length x query_length
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x source_length)(batch x source_length x query_length)
        # --> batch x idf x query_length
        weighted_context = torch.bmm(source_t, attn)
        weighted_context = weighted_context.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weighted_context, attn
