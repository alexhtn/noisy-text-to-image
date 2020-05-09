import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from nn.losses.words_loss import words_loss
from nn.losses.sent_loss import sent_loss
from nn.text_encoder import TextEncoder
from nn.image_encoder.image_encoder import ImageEncoder
from dataset import BirdsDataset
from utils import get_save_directory, save_best

device = torch.device('cuda:0')
dataset_root = '/datasets/CUB_200_2011'
batch_size = 48
text_dim = 7551
text_noise_dim = 100
condition_dim = 3584
word_dim = 512
image_noise_dim = 128


def main():
    text_encoder = TextEncoder(text_dim=text_dim, noise_dim=text_noise_dim, out_dim=condition_dim, word_dim=word_dim)
    text_encoder.load_state_dict(torch.load(os.path.join(get_save_directory(), 'text-encoder.pth'))['state_dict_G'])
    text_encoder.to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

    image_encoder = ImageEncoder(condition_dim=condition_dim, word_dim=word_dim)
    image_encoder.to(device)
    image_encoder.train()

    image_transform = transforms.Compose([
        transforms.Resize(int(256 * 76 / 64)),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip()])
    train_dataset = BirdsDataset(dataset_root, resolutions=[256], transform=image_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    optimizer = Adam(image_encoder.parameters())

    total_s_loss0 = []
    total_s_loss1 = []
    total_w_loss0 = []
    total_w_loss1 = []
    start_iteration = 0
    dataloader_iterator = iter(train_dataloader)
    for iteration in range(start_iteration, 1_000_000):

        try:
            data = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_dataloader)
            data = next(dataloader_iterator)

        img, text_feature, class_id = [x.to(device) for x in data]

        text_noise = torch.stack([torch.randn([text_noise_dim], device=device) for _ in range(batch_size)])

        conditional_features, words_embs = text_encoder(text_feature, text_noise)

        global_features, region_features = image_encoder(img)
        w_loss0, w_loss1, _ = words_loss(region_features, words_embs)
        w_loss = (w_loss0 + w_loss1)

        s_loss0, s_loss1 = sent_loss(global_features, conditional_features)
        s_loss = (s_loss0 + s_loss1)

        loss = w_loss + s_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_s_loss0.append(s_loss0.item())
        total_s_loss1.append(s_loss1.item())
        total_w_loss0.append(w_loss0.item())
        total_w_loss1.append(w_loss1.item())

        if iteration % 100 == 0:
            total_s_loss0 = np.mean(total_s_loss0)
            total_s_loss1 = np.mean(total_s_loss1)
            total_w_loss0 = np.mean(total_w_loss0)
            total_w_loss1 = np.mean(total_w_loss1)

            print(f'Iteration {iteration}: '
                  f's_loss {total_s_loss0:5.2f} {total_s_loss1:5.2f} | '
                  f'w_loss {total_w_loss0:5.2f} {total_w_loss1:5.2f} ')

            total_s_loss0 = []
            total_s_loss1 = []
            total_w_loss0 = []
            total_w_loss1 = []

            save_best(image_encoder, prefix='image-encoder')


if __name__ == '__main__':
    main()
