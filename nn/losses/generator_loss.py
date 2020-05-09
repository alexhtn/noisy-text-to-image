from torch import nn
from .words_loss import words_loss
from .sent_loss import sent_loss


class GeneratorLoss(nn.Module):
    def __init__(self, discriminators, image_encoder, real_labels, smooth_lambda=5.0) -> None:
        super().__init__()

        self.discriminators = discriminators
        self.image_encoder = image_encoder
        self.real_labels = real_labels
        self.smooth_lambda = smooth_lambda

        self.bce = nn.BCELoss()

    def forward(self, fake_images, words_embs, conditional_features):
        # Ranking loss
        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        global_features, region_features = self.image_encoder(fake_images[-1])
        w_loss0, w_loss1, _ = words_loss(region_features, words_embs)
        w_loss = (w_loss0 + w_loss1) * self.smooth_lambda

        s_loss0, s_loss1 = sent_loss(global_features, conditional_features)
        s_loss = (s_loss0 + s_loss1) * self.smooth_lambda

        total_loss = w_loss + s_loss

        # Real/fake loss
        for i in range(len(self.discriminators)):
            fake_features = self.discriminators[i](fake_images[i])
            fake_cond_loss = self.bce(self.discriminators[i].conditional_output(fake_features, conditional_features),
                                      self.real_labels)

            fake_uncond_loss = self.bce(self.discriminators[i].unconditional_output(fake_features),
                                        self.real_labels)

            total_loss += fake_cond_loss + fake_uncond_loss

        return total_loss
