from torch import nn
from .generator_blocks.initial_stage import InitialStage
from .generator_blocks.next_stage import NextStage
from .generator_blocks.get_image import GetImage


class Generator(nn.Module):
    def __init__(self, condition_dim, word_dim, noise_dim):
        super().__init__()
        stage_out_channels = 32

        self.block1 = InitialStage(condition_dim, noise_dim, stage_out_channels)
        self.get_image1 = GetImage(stage_out_channels)

        self.block2 = NextStage(stage_out_channels, word_dim)
        self.get_image2 = GetImage(stage_out_channels)

        self.block3 = NextStage(stage_out_channels, word_dim)
        self.get_image3 = GetImage(stage_out_channels)

    def forward(self, conditional_features, noise, words_emb):
        fake_images = []

        # Block 1
        feature_map1 = self.block1(conditional_features, noise)
        fake_img1 = self.get_image1(feature_map1)
        fake_images.append(fake_img1)

        # Block 2
        feature_map2, _ = self.block2(feature_map1, words_emb)
        fake_img2 = self.get_image2(feature_map2)
        fake_images.append(fake_img2)

        # # Block 3
        feature_map3, _ = self.block3(feature_map2, words_emb)
        fake_img3 = self.get_image3(feature_map3)
        fake_images.append(fake_img3)

        return fake_images
