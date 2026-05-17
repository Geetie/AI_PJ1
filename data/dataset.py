import json
import os
import random
from glob import glob

import numpy as np
import torch as t
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import config, data_dir
from data.transform import transform_with_bbox, transform_test_img, resize_keep_aspect, normalize_tensor
from utils.misc import PadToSquare


class DigitsDataset(Dataset):
    def __init__(self, mode='train', aug=True, input_size=(224, 224)):
        super(DigitsDataset, self).__init__()
        self.aug = aug
        self.mode = mode
        self.input_size = input_size
        if mode == 'test':
            self.imgs = sorted(glob(data_dir['test_data'] + '*.png'))
            self.labels = None
        else:
            labels = json.load(open(data_dir['%s_label' % mode], 'r'))
            imgs = sorted(glob(data_dir['%s_data' % mode] + '*.png'))
            self.imgs = [(img, labels[os.path.split(img)[-1]]) for img in imgs
                         if os.path.split(img)[-1] in labels]

    def __getitem__(self, idx):
        if self.mode != 'test':
            img, label = self.imgs[idx]
        else:
            img = self.imgs[idx]
            label = None
        img = Image.open(img).convert('RGB')

        if self.mode != 'test':
            padded_label = label['label'][:config.num_heads] + \
                           (config.num_heads - len(label['label'])) * [10]
            n_digits = len(label['label'])
            bboxes = np.zeros((config.num_heads, 4), dtype=np.float32)
            for i in range(min(n_digits, config.num_heads)):
                bboxes[i] = [label['left'][i], label['top'][i],
                             label['width'][i], label['height'][i]]
            img, bboxes = transform_with_bbox(img, bboxes, self.input_size, self.aug)
            bbox_mask = np.zeros(config.num_heads, dtype=np.float32)
            bbox_mask[:min(n_digits, config.num_heads)] = 1.0
            return img, t.tensor(padded_label).long(), t.tensor(bboxes).float(), t.tensor(bbox_mask).float()
        else:
            img = transform_test_img(img, self.input_size, config.keep_aspect_ratio)
            return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)


class CTCDataset(Dataset):
    def __init__(self, mode='train', aug=True, input_size=(224, 224)):
        super(CTCDataset, self).__init__()
        self.aug = aug
        self.mode = mode
        self.input_size = input_size
        if mode == 'test':
            self.imgs = sorted(glob(data_dir['test_data'] + '*.png'))
            self.labels = None
        else:
            labels = json.load(open(data_dir['%s_label' % mode], 'r'))
            imgs = sorted(glob(data_dir['%s_data' % mode] + '*.png'))
            self.imgs = [(img, labels[os.path.split(img)[-1]]) for img in imgs
                         if os.path.split(img)[-1] in labels]

    def _transform_img(self, img):
        if config.keep_aspect_ratio:
            if self.aug:
                img, _ = resize_keep_aspect(img, None, config.resize_size)
                i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                    img, (self.input_size[0], self.input_size[1]))
                img = transforms.functional.crop(img, i, j, h_crop, w_crop)
                img = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)(img)
                img = transforms.RandomGrayscale(0.1)(img)
                if random.random() < 0.3:
                    img = transforms.RandomPerspective(distortion_scale=0.2, fill=127)(img)
            else:
                img, _ = resize_keep_aspect(img, None, self.input_size[0])
                i = max(0, (img.size[1] - self.input_size[0]) // 2)
                j = max(0, (img.size[0] - self.input_size[1]) // 2)
                img = transforms.functional.crop(img, i, j, self.input_size[0], self.input_size[1])
        else:
            if self.aug:
                img = PadToSquare(fill=(127, 127, 127))(img)
                img = transforms.functional.resize(img, (config.resize_size, config.resize_size))
                i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                    img, (self.input_size[0], self.input_size[1]))
                img = transforms.functional.crop(img, i, j, h_crop, w_crop)
                img = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)(img)
                img = transforms.RandomGrayscale(0.1)(img)
                if random.random() < 0.3:
                    img = transforms.RandomPerspective(distortion_scale=0.2, fill=127)(img)
            else:
                img = PadToSquare(fill=(127, 127, 127))(img)
                img = transforms.functional.resize(img, (self.input_size[0], self.input_size[1]))
        img = normalize_tensor(img)
        if self.aug:
            img = transforms.RandomErasing(p=config.erase_prob, scale=(0.02, 0.1), ratio=(0.3, 3.3))(img)
        return img

    def __getitem__(self, idx):
        if self.mode != 'test':
            img, label = self.imgs[idx]
        else:
            img = self.imgs[idx]
            label = None
        img = Image.open(img).convert('RGB')

        if self.mode != 'test':
            digits = label['label']
            return self._transform_img(img), t.tensor(digits, dtype=t.long), len(digits)
        else:
            return self._transform_img(img), self.imgs[idx]

    def __len__(self):
        return len(self.imgs)


def ctc_collate_fn(batch):
    imgs, labels, lengths = zip(*batch)
    imgs = t.stack(imgs, 0)
    label_concat = t.cat(labels, 0)
    return imgs, label_concat, t.tensor(lengths, dtype=t.long)


def ctc_test_collate_fn(batch):
    imgs, names = zip(*batch)
    imgs = t.stack(imgs, 0)
    return imgs, list(names)
