import random

import numpy as np
from torchvision import transforms

from config import config
from utils.misc import PadToSquare


def resize_keep_aspect(img, bboxes, target_size):
    w, h = img.size
    if w <= h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    new_h = max(new_h, target_size)
    new_w = max(new_w, target_size)
    img = transforms.functional.resize(img, (new_h, new_w))
    scale_x = new_w / w
    scale_y = new_h / h
    if bboxes is not None and len(bboxes) > 0:
        bboxes[:, 0] *= scale_x
        bboxes[:, 1] *= scale_y
        bboxes[:, 2] *= scale_x
        bboxes[:, 3] *= scale_y
    return img, bboxes


def apply_augmentation(img):
    img = transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)(img)
    img = transforms.RandomGrayscale(0.1)(img)
    if config.aug_rotation_degrees > 0:
        img = transforms.RandomRotation(config.aug_rotation_degrees, fill=127)(img)
    if config.aug_blur_prob > 0 and random.random() < config.aug_blur_prob:
        img = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))(img)
    if random.random() < 0.3:
        try:
            img = transforms.RandomPerspective(distortion_scale=0.3, fill=127)(img)
        except RuntimeError:
            pass
    if random.random() < 0.2:
        img = transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)(img)
    if random.random() < 0.15:
        img = transforms.RandomAutocontrast(p=0.5)(img)
    return img


def normalize_tensor(img):
    img = transforms.functional.to_tensor(img)
    img = transforms.functional.normalize(img,
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img


def transform_with_bbox(img, bboxes, input_size, aug=True):
    w, h = img.size

    if config.keep_aspect_ratio:
        if aug:
            img, bboxes = resize_keep_aspect(img, bboxes, config.resize_size)
            i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                img, (input_size[0], input_size[1]))
            img = transforms.functional.crop(img, i, j, h_crop, w_crop)
            if bboxes is not None and len(bboxes) > 0:
                bboxes[:, 0] -= j
                bboxes[:, 1] -= i
                bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w_crop)
                bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h_crop)
                bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w_crop - bboxes[:, 0])
                bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h_crop - bboxes[:, 1])
            img = apply_augmentation(img)
        else:
            img, bboxes = resize_keep_aspect(img, bboxes, input_size[0])
            i = max(0, (img.size[1] - input_size[0]) // 2)
            j = max(0, (img.size[0] - input_size[1]) // 2)
            img = transforms.functional.crop(img, i, j, input_size[0], input_size[1])
            if bboxes is not None and len(bboxes) > 0:
                bboxes[:, 0] -= j
                bboxes[:, 1] -= i
                bboxes[:, 0] = np.clip(bboxes[:, 0], 0, input_size[1])
                bboxes[:, 1] = np.clip(bboxes[:, 1], 0, input_size[0])
                bboxes[:, 2] = np.clip(bboxes[:, 2], 0, input_size[1] - bboxes[:, 0])
                bboxes[:, 3] = np.clip(bboxes[:, 3], 0, input_size[0] - bboxes[:, 1])
    else:
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        img = transforms.functional.pad(img,
            (pad_left, pad_top, max_dim - w - pad_left, max_dim - h - pad_top),
            fill=(127, 127, 127))
        if bboxes is not None and len(bboxes) > 0:
            bboxes[:, 0] += pad_left
            bboxes[:, 1] += pad_top

        if aug:
            img = transforms.functional.resize(img, (config.resize_size, config.resize_size))
            scale = config.resize_size / max_dim
            if bboxes is not None and len(bboxes) > 0:
                bboxes *= scale
            i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                img, (input_size[0], input_size[1]))
            img = transforms.functional.crop(img, i, j, h_crop, w_crop)
            if bboxes is not None and len(bboxes) > 0:
                bboxes[:, 0] -= j
                bboxes[:, 1] -= i
            img = apply_augmentation(img)
        else:
            img = transforms.functional.resize(img, (input_size[0], input_size[1]))
            scale = input_size[0] / max_dim
            if bboxes is not None and len(bboxes) > 0:
                bboxes *= scale

    img = normalize_tensor(img)

    if aug:
        img = transforms.RandomErasing(p=config.erase_prob, scale=(0.02, 0.1), ratio=(0.3, 3.3))(img)

    if bboxes is not None and len(bboxes) > 0:
        bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2] / 2) / input_size[1]
        bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3] / 2) / input_size[0]
        bboxes[:, 2] = bboxes[:, 2] / input_size[1]
        bboxes[:, 3] = bboxes[:, 3] / input_size[0]
        bboxes = np.clip(bboxes, 0, 1)

    return img, bboxes


def transform_test_img(img, input_size, keep_aspect_ratio):
    if keep_aspect_ratio:
        w, h = img.size
        if w <= h:
            new_w = input_size[1]
            new_h = max(int(h * input_size[1] / w), input_size[0])
        else:
            new_h = input_size[0]
            new_w = max(int(w * input_size[0] / h), input_size[1])
        img = transforms.functional.resize(img, (new_h, new_w))
        i = max(0, (new_h - input_size[0]) // 2)
        j = max(0, (new_w - input_size[1]) // 2)
        img = transforms.functional.crop(img, i, j, input_size[0], input_size[1])
        img = normalize_tensor(img)
    else:
        trans = transforms.Compose([
            PadToSquare(fill=(127, 127, 127)),
            transforms.Resize((input_size[0], input_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = trans(img)
    return img
