import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, text_dim, noise_dim, out_dim, word_dim, rdc_text_dim=1000, h_dim=4096) -> None:
        super().__init__()

        self.out_dim = out_dim
        self.word_dim = word_dim

        self.rdc_text = nn.Linear(text_dim, rdc_text_dim)
        self.main = nn.Sequential(nn.Linear(noise_dim + rdc_text_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, out_dim),
                                  nn.Tanh())

    def forward(self, text, noise):
        x = self.rdc_text(text)
        x = torch.cat([noise, x], 1)
        x = self.main(x)
        words_emb = x.view(-1, self.out_dim // self.word_dim, self.word_dim)
        words_emb = words_emb.permute(0, 2, 1)
        return x, words_emb
