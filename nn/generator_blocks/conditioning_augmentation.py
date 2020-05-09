import torch
from torch import nn
from .activation_glu import GLU


class ConditioningAugmentation(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)

    def __init__(self, text_embedding_dim, condition_dim):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.condition_dim = condition_dim
        self.fc = nn.Linear(self.text_embedding_dim, self.condition_dim * 4, bias=True)
        self.activation = GLU()

    def encode(self, text_embedding):
        x = self.activation(self.fc(text_embedding))
        mu = x[:, :self.condition_dim]
        logvar = x[:, self.condition_dim:]
        return mu, logvar

    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar
