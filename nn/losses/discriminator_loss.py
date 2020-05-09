from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(self, discriminator, real_labels, fake_labels, wrong_labels) -> None:
        super().__init__()

        self.discriminator = discriminator
        self.real_labels = real_labels
        self.fake_labels = fake_labels
        self.wrong_labels = wrong_labels

        self.bce = nn.BCELoss()

    def forward(self, real_features, fake_features, conditional_features):
        real_cond_loss = self.bce(self.discriminator.conditional_output(real_features, conditional_features),
                                  self.real_labels)
        fake_cond_loss = self.bce(self.discriminator.conditional_output(fake_features, conditional_features),
                                  self.fake_labels)

        real_uncond_loss = self.bce(self.discriminator.unconditional_output(real_features),
                                    self.real_labels)
        fake_uncond_loss = self.bce(self.discriminator.unconditional_output(fake_features),
                                    self.fake_labels)

        wrong_cond_loss = self.bce(self.discriminator.conditional_output(real_features[:-1], conditional_features[1:]),
                                   self.wrong_labels)

        loss = ((real_uncond_loss + real_cond_loss) / 2. +
                (fake_uncond_loss + fake_cond_loss + wrong_cond_loss) / 3.)

        return loss
