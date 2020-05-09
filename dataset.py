import os
import numpy as np
from collections import namedtuple

import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms


DatasetItem = namedtuple('DatasetItem', ['path', 'class_id', 'bbox'])


class BirdsDataset(Dataset):
    def __init__(self, dataset_root, resolutions, transform=None):

        self.dataset_root = dataset_root
        self.resolutions = resolutions
        self.transform = transform

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load text features
        self.text_features = loadmat(os.path.join(dataset_root, 'zsl-data', 'CUB_Porter_7551D_TFIDF_new.mat'))
        self.text_features = torch.from_numpy(self.text_features['PredicateMatrix'].astype(np.float32))

        # load bboxes
        bboxes = {}
        with open(os.path.join(dataset_root, 'bounding_boxes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split(' ')
                image_id = line[0]
                bbox = [int(float(x)) for x in line[1:]]
                bboxes[image_id] = bbox

        # create items list
        image_id_to_class_id = {}
        with open(os.path.join(dataset_root, 'image_class_labels.txt'), 'r') as f:
            for line in f.readlines():
                image_id, class_id = line.replace('\n', '').split(' ')
                class_id = int(class_id) - 1
                image_id_to_class_id[image_id] = class_id

        self.items = []
        with open(os.path.join(dataset_root, 'images.txt'), 'r') as f:
            for line in f.readlines():
                image_id, path = line.replace('\n', '').split(' ')
                self.items.append(DatasetItem(path, image_id_to_class_id[image_id], bboxes[image_id]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        return_item = self.get_images(item)

        class_id = item.class_id
        text_feature = self.text_features[class_id]

        return_item.append(text_feature)
        return_item.append(class_id)

        return return_item

    def get_images(self, item):
        img = Image.open(os.path.join(self.dataset_root, 'images', item.path)).convert('RGB')
        # img = img.resize((self.resolution, self.resolution), Image.BILINEAR)

        bbox = item.bbox
        width, height = img.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

        if self.transform is not None:
            img = self.transform(img)

        imgs = []
        for resolution in self.resolutions:
            resized_img = transforms.Resize(size=(resolution, resolution))(img)
            resized_img = self.normalize(resized_img)
            imgs.append(resized_img)

        return imgs
