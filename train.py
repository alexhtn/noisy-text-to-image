import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from nn.losses.discriminator_loss import DiscriminatorLoss
from nn.losses.generator_loss import GeneratorLoss
from nn.text_encoder import TextEncoder
from nn.generator import Generator
from nn.discriminator64 import Discriminator64
from nn.discriminator128 import Discriminator128
from nn.discriminator256 import Discriminator256
from nn.image_encoder.image_encoder import ImageEncoder
from dataset import BirdsDataset
from utils import get_save_directory

device = torch.device('cuda:0')
dataset_root = '/datasets/CUB_200_2011'
batch_size = 16
text_dim = 7551
text_noise_dim = 100
condition_dim = 3584
word_dim = 512
image_noise_dim = 128
results_size = 16


def main():
    text_encoder = TextEncoder(text_dim=text_dim, noise_dim=text_noise_dim, out_dim=condition_dim, word_dim=word_dim)
    text_encoder.load_state_dict(torch.load(os.path.join(get_save_directory(), 'text-encoder.pth'))['state_dict_G'])
    text_encoder.to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

    generator = Generator(condition_dim=condition_dim, noise_dim=image_noise_dim, word_dim=word_dim)
    generator.to(device)
    generator.train()

    discriminators = []
    discriminator = Discriminator64(channels=16, condition_dim=condition_dim)
    discriminator.to(device)
    discriminator.train()
    discriminators.append(discriminator)
    discriminator = Discriminator128(channels=16, condition_dim=condition_dim)
    discriminator.to(device)
    discriminator.train()
    discriminators.append(discriminator)
    discriminator = Discriminator256(channels=16, condition_dim=condition_dim)
    discriminator.to(device)
    discriminator.train()
    discriminators.append(discriminator)

    image_encoder = ImageEncoder(condition_dim=condition_dim, word_dim=word_dim)
    image_encoder.load_state_dict(torch.load(os.path.join(get_save_directory(), 'image-encoder-best.pth')))
    image_encoder.to(device)
    image_encoder.eval()
    for param in image_encoder.parameters():
        param.requires_grad = False

    # dataset
    image_transform = transforms.Compose([
        transforms.Resize(int(256 * 76 / 64)),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip()])
    train_dataset = BirdsDataset(dataset_root, resolutions=(64, 128, 256), transform=image_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    real_labels = torch.ones([batch_size], device=device)
    fake_labels = torch.zeros([batch_size], device=device)
    wrong_labels = fake_labels[1:]

    discriminator_criterions = [DiscriminatorLoss(discriminator, real_labels, fake_labels, wrong_labels)
                                for discriminator in discriminators]

    generator_criterion = GeneratorLoss(discriminators, image_encoder, real_labels)

    discriminator_optimizers = [Adam(discriminator.parameters()) for discriminator in discriminators]

    generator_optimizer = Adam(generator.parameters())

    # fixed inputs for results
    output_dir = os.path.join(get_save_directory(), datetime.now().strftime('%Y-%m-%d-%H-%M'))
    os.makedirs(output_dir, exist_ok=True)
    fixed_text_noise = torch.stack([torch.randn([text_noise_dim], device=device) for _ in range(results_size)])
    fixed_image_noise = torch.stack([torch.randn([image_noise_dim], device=device) for _ in range(results_size)])
    fixed_text_feature = []
    fixed_real_images = []
    for i in range(results_size):
        _, _, img, text_feature, _ = train_dataset.__getitem__(i * 500)
        fixed_text_feature.append(text_feature)
        fixed_real_images.append(img)
    fixed_text_feature = torch.stack(fixed_text_feature).to(device)
    fixed_real_images = torch.stack(fixed_real_images)
    fixed_real_images = (fixed_real_images + 1) / 2 * 255
    fixed_real_images = make_grid(fixed_real_images, nrow=4)
    fixed_real_images = Image.fromarray(fixed_real_images.permute(1, 2, 0).numpy().astype(np.uint8))
    fixed_real_images.save(os.path.join(output_dir, 'real.jpg'))

    with torch.no_grad():
        fixed_conditional_features, fixed_words_embs = text_encoder(fixed_text_feature, fixed_text_noise)

    # training
    start_iteration = 0
    total_d_loss = []
    total_g_loss = []
    dataloader_iterator = iter(train_dataloader)
    for iteration in range(start_iteration, 1_000_000):

        try:
            data = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_dataloader)
            data = next(dataloader_iterator)

        img64, img128, img256, text_feature, class_id = [x.to(device) for x in data]

        real_images = [img64, img128, img256]

        text_noise = torch.stack([torch.randn([text_noise_dim], device=device) for _ in range(batch_size)])
        image_noise = torch.stack([torch.randn([image_noise_dim], device=device) for _ in range(batch_size)])

        #######################################################
        # Generate fake images
        conditional_features, words_embs = text_encoder(text_feature, text_noise)
        fake_images = generator(conditional_features, image_noise, words_embs)

        #######################################################
        # Update D network
        batch_d_loss = 0
        for i in range(3):
            real_features = discriminators[i](real_images[i])
            fake_features = discriminators[i](fake_images[i].detach())

            discriminator_loss = discriminator_criterions[i](real_features, fake_features, conditional_features)
            discriminator_optimizers[i].zero_grad()
            discriminator_loss.backward()
            discriminator_optimizers[i].step()
            batch_d_loss += discriminator_loss.item()

        #######################################################
        # Update G network: maximize log(D(G(z)))
        generator_optimizer.zero_grad()
        generator_loss = generator_criterion(fake_images, words_embs, conditional_features)
        generator_loss.backward()
        generator_optimizer.step()

        #######################################################
        # Totals
        total_d_loss.append(batch_d_loss)
        total_g_loss.append(generator_loss.item())

        if iteration % 100 == 0:
            total_d_loss = np.mean(total_d_loss)
            total_g_loss = np.mean(total_g_loss)

            print(f'Iteration {iteration}: '
                  f'd_loss {total_d_loss:5.2f} | '
                  f'g_loss {total_g_loss:5.2f}  ')

            total_d_loss = []
            total_g_loss = []

            with torch.no_grad():
                generator.eval()
                generated_images_all_sizes = generator(fixed_conditional_features, fixed_image_noise, fixed_words_embs)
                for i, generated_images in enumerate(generated_images_all_sizes):
                    generated_images = (generated_images + 1) / 2 * 255
                    generated_images = make_grid(generated_images, nrow=4)
                    generated_images = Image.fromarray(generated_images.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    generated_images.save(os.path.join(output_dir, f'gen-{iteration:06d}-{i}.jpg'))

                generator.train()


if __name__ == '__main__':
    main()
