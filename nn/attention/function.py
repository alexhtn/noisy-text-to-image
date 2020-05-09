"""
Global attention takes a matrix and a query metrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \\ |   |      /
              .....
          \\   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.

References:
https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
http://www.aclweb.org/anthology/D15-1166
"""

import torch


def attention_function(query, context, gamma1):
    """
    query: batch x ndf x query_length
    context: batch x ndf x ih x iw (source_length=ihxiw)
    mask: batch_size x source_length
    """
    batch_size, query_length = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    source_length = ih * iw

    # --> batch x source_length x ndf
    context = context.view(batch_size, -1, source_length)
    context_t = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x source_length x ndf)(batch x ndf x query_length)
    # -->batch x source_length x query_length
    attn = torch.bmm(context_t, query)  # Eq. (7) in AttnGAN paper
    # --> batch*source_length x query_length
    attn = attn.view(batch_size*source_length, query_length)
    attn = torch.softmax(attn, dim=1)  # Eq. (8)

    # --> batch x source_length x query_length
    attn = attn.view(batch_size, source_length, query_length)
    # --> batch*query_length x source_length
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*query_length, source_length)
    #  Eq. (9)
    attn = attn * gamma1
    attn = torch.softmax(attn, dim=1)
    attn = attn.view(batch_size, query_length, source_length)
    # --> batch x source_length x query_length
    attn_t = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x source_length)(batch x source_length x query_length)
    # --> batch x ndf x query_length
    weighted_context = torch.bmm(context, attn_t)

    return weighted_context, attn.view(batch_size, -1, ih, iw)
