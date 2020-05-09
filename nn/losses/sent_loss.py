import torch
from torch import nn


def sent_loss(global_features, conditional_features, gamma3=10):
    labels = torch.arange(global_features.size(0), dtype=torch.long, device=global_features.device)

    # --> seq_len x batch_size x nef
    if global_features.dim() == 2:
        global_features = global_features.unsqueeze(0)
        conditional_features = conditional_features.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(global_features, 2, dim=2, keepdim=True)
    conditional_features_norm = torch.norm(conditional_features, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(global_features, conditional_features.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, conditional_features_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=1e-8) * gamma3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0, 1)

    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)

    return loss0, loss1
