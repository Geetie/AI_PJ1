import pandas as pd
import os
import requests
import zipfile
from glob import glob
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision.models.resnet import resnet101, ResNet101_Weights
import random
from torch.optim import SGD
from torch.amp import autocast, GradScaler
import copy
import multiprocessing
import shutil
import gc

try:
    from torchvision.ops import roi_align
except ImportError:
    roi_align = None

# 统一使用config.py的配置
from config import config, set_seed, BASE_DIR, SCRIPT_DIR, NUM_WORKERS, data_dir, NUM_HEADS

# 设置随机种子
set_seed(42)

if t.cuda.is_available():
    t.set_float32_matmul_precision('high')
    print('✅ TF32 matmul precision enabled')

# 设置torch hub目录
t.hub.set_dir(os.path.join(SCRIPT_DIR, 'torch_hub'))


class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        padding = (pad_left, pad_top, max_dim - w - pad_left, max_dim - h - pad_top)
        return transforms.functional.pad(img, padding, fill=self.fill)


# 从config.py导入data_dir等配置
from config import data_dir

# 打印数据集信息
train_list = glob(data_dir['train_data'] + '*.png')
test_list = glob(data_dir['test_data'] + '*.png')
val_list = glob(data_dir['val_data'] + '*.png')
print('train image counts: %d' % len(train_list))
print('val image counts: %d' % len(val_list))
print('test image counts: %d' % len(test_list))


def look_train_json():
    with open(data_dir['train_label'], 'r', encoding='utf-8') as f:
        content = f.read()
    content = json.loads(content)
    print(content['000000.png'])


def look_submit():
    df = pd.read_csv(data_dir['submit_file'], sep=',')
    print(df.head(5))


def img_size_summary(sample_ratio=0.1, use_cache=True, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'img_size_cache.npy')

    if use_cache and os.path.exists(cache_file):
        sizes = np.load(cache_file)
        print(f'✅ Loaded cached image sizes from {cache_file}')
    else:
        all_imgs = glob(data_dir['train_data'] + '*.png')

        if sample_ratio < 1.0:
            total_count = len(all_imgs)
            sample_size = int(total_count * sample_ratio)
            all_imgs = random.sample(all_imgs, sample_size)
            print(f'📊 Sampling {sample_size}/{total_count} images...')

        sizes = []
        for img_path in tqdm(all_imgs, desc='Analyzing image sizes'):
            img = Image.open(img_path)
            sizes.append(img.size)
        sizes = np.array(sizes)

        if use_cache:
            np.save(cache_file, sizes)
            print(f'💾 Cached image sizes to {cache_file}')

    plt.figure(figsize=(10, 8))
    plt.scatter(sizes[:, 0], sizes[:, 1], alpha=0.6, s=10)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('image width-height summary')
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f'📈 Image size statistics:')
    print(f'   - Min width: {sizes[:, 0].min()}, Max width: {sizes[:, 0].max()}')
    print(f'   - Min height: {sizes[:, 1].min()}, Max height: {sizes[:, 1].max()}')
    print(f'   - Mean width: {sizes[:, 0].mean():.1f}, Mean height: {sizes[:, 1].mean():.1f}')


def bbox_summary(sample_ratio=0.1, use_cache=True, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'bbox_cache.npy')

    if use_cache and os.path.exists(cache_file):
        bboxes = np.load(cache_file)
        print(f'✅ Loaded cached bbox data from {cache_file}')
    else:
        marks = json.loads(open(data_dir['train_label'], 'r').read())

        if sample_ratio < 1.0:
            total_count = len(marks)
            sample_keys = random.sample(list(marks.keys()),
                                       int(total_count * sample_ratio))
            marks = {k: marks[k] for k in sample_keys}
            print(f'📊 Sampling {len(marks)}/{total_count} labels...')

        bboxes = []
        for mark in tqdm(marks.values(), desc='Analyzing bboxes'):
            for i in range(len(mark['label'])):
                bboxes.append([mark['left'][i], mark['top'][i],
                              mark['width'][i], mark['height'][i]])
        bboxes = np.array(bboxes)

        if use_cache:
            np.save(cache_file, bboxes)
            print(f'💾 Cached bbox data to {cache_file}')

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(bboxes[:, 2], bboxes[:, 3], alpha=0.6, s=10)
    ax.set_title('bbox width-height summary')
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    ax.grid(True, alpha=0.3)
    plt.show()

    print(f'📈 Bbox size statistics:')
    print(f'   - Min width: {bboxes[:, 2].min()}, Max width: {bboxes[:, 2].max()}')
    print(f'   - Min height: {bboxes[:, 3].min()}, Max height: {bboxes[:, 3].max()}')
    print(f'   - Mean width: {bboxes[:, 2].mean():.1f}, Mean height: {bboxes[:, 3].mean():.1f}')


def label_summary():
    marks = json.load(open(data_dir['train_label'], 'r'))
    dicts = {}
    for img, mark in marks.items():
        if len(mark['label']) not in dicts:
            dicts[len(mark['label'])] = 0
        dicts[len(mark['label'])] += 1
    dicts = sorted(dicts.items(), key=lambda x: x[0])
    for k, v in dicts:
        print('%d个数字的图片数目: %d' % (k, v))


# 配置已经从config.py统一导入，无需再定义内部Config类


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.fc(self.pool(x).view(B, C))
        return x * w.view(B, C, 1, 1)


class FPNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, replace_stride_with_dilation=[False, False, True])
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # 改进：增加layer1的特征融合，提供更细粒度的空间信息
        self.l1_reduce = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.l2_reduce = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.l3_reduce = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.l4_reduce = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Smooth卷积用于减少上采样的棋盘效应
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.smooth_p1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 改进：融合P1/P2/P3/P4四个尺度的特征
        # P1: 128 channels, P2: 256, P3: 256, P4: 256 -> total 896
        self.fuse = nn.Sequential(
            nn.Conv2d(896, config.multiscale_feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.multiscale_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(config.multiscale_feat_dim)
        self.use_checkpoint = True

    def _forward_early(self, x):
        """
        改进：返回c1/c2/c3以支持更精细的多尺度融合
        """
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        return c1, c2, c3

    def forward(self, x):
        if self.training and self.use_checkpoint:
            c1, c2, c3 = t.utils.checkpoint.checkpoint(self._forward_early, x, use_reentrant=False)
        else:
            c1, c2, c3 = self._forward_early(x)
        c4 = self.layer4(c3)
        
        # Top-down pathway with bilinear interpolation (smoother than nearest)
        p4 = self.l4_reduce(c4)
        p3 = self.l3_reduce(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.smooth_p3(p3)
        p2 = self.l2_reduce(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.smooth_p2(p2)
        # 新增：融合P1层特征
        p1 = self.l1_reduce(c1) + F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.smooth_p1(p1)
        
        # 上采样所有特征到P1的分辨率进行融合
        p2_up = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)
        
        # 融合四个尺度的特征
        return self.se(self.fuse(t.cat([p1, p2_up, p3_up, p4_up], dim=1)))


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

    def _resize_keep_aspect(self, img, bboxes, target_size):
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
            bboxes[:, 0] *= scale_x  # left
            bboxes[:, 1] *= scale_y  # top
            bboxes[:, 2] *= scale_x  # width
            bboxes[:, 3] *= scale_y  # height
        return img, bboxes

    def _transform_with_bbox(self, img, bboxes):
        w, h = img.size

        if config.keep_aspect_ratio:
            if self.aug:
                img, bboxes = self._resize_keep_aspect(img, bboxes, config.resize_size)
                i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                    img, (self.input_size[0], self.input_size[1]))
                img = transforms.functional.crop(img, i, j, h_crop, w_crop)
                if bboxes is not None and len(bboxes) > 0:
                    bboxes[:, 0] -= j  # left
                    bboxes[:, 1] -= i  # top
                    # 修复：确保bbox在crop后的图像范围内
                    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w_crop)
                    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h_crop)
                    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w_crop - bboxes[:, 0])
                    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h_crop - bboxes[:, 1])
                img = transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)(img)  # 增强色彩抖动
                img = transforms.RandomGrayscale(0.1)(img)
                if config.aug_rotation_degrees > 0:
                    img = transforms.RandomRotation(config.aug_rotation_degrees, fill=127)(img)
                if config.aug_blur_prob > 0 and t.rand(1).item() < config.aug_blur_prob:
                    img = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))(img)
                if t.rand(1).item() < 0.3:
                    img = transforms.RandomPerspective(distortion_scale=0.3, fill=127)(img)
                if t.rand(1).item() < 0.2:
                    img = transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)(img)
                if t.rand(1).item() < 0.15:
                    img = transforms.RandomAutocontrast(p=0.5)(img)
            else:
                img, bboxes = self._resize_keep_aspect(img, bboxes, self.input_size[0])
                i = max(0, (img.size[1] - self.input_size[0]) // 2)
                j = max(0, (img.size[0] - self.input_size[1]) // 2)
                img = transforms.functional.crop(img, i, j, self.input_size[0], self.input_size[1])
                if bboxes is not None and len(bboxes) > 0:
                    bboxes[:, 0] -= j  # left
                    bboxes[:, 1] -= i  # top
                    # 修复：确保bbox在crop后的图像范围内
                    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, self.input_size[1])
                    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, self.input_size[0])
                    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, self.input_size[1] - bboxes[:, 0])
                    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, self.input_size[0] - bboxes[:, 1])
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

            if self.aug:
                img = transforms.functional.resize(img, (config.resize_size, config.resize_size))
                scale = config.resize_size / max_dim
                if bboxes is not None and len(bboxes) > 0:
                    bboxes *= scale
                i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                    img, (self.input_size[0], self.input_size[1]))
                img = transforms.functional.crop(img, i, j, h_crop, w_crop)
                if bboxes is not None and len(bboxes) > 0:
                    bboxes[:, 0] -= j
                    bboxes[:, 1] -= i
                img = transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)(img)  # 增强色彩抖动
                img = transforms.RandomGrayscale(0.1)(img)
                if config.aug_rotation_degrees > 0:
                    img = transforms.RandomRotation(config.aug_rotation_degrees, fill=127)(img)
                if config.aug_blur_prob > 0 and t.rand(1).item() < config.aug_blur_prob:
                    img = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))(img)
                if t.rand(1).item() < 0.3:
                    img = transforms.RandomPerspective(distortion_scale=0.3, fill=127)(img)
                if t.rand(1).item() < 0.2:
                    img = transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)(img)
                if t.rand(1).item() < 0.15:
                    img = transforms.RandomAutocontrast(p=0.5)(img)
            else:
                img = transforms.functional.resize(img, (self.input_size[0], self.input_size[1]))
                scale = self.input_size[0] / max_dim
                if bboxes is not None and len(bboxes) > 0:
                    bboxes *= scale

        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img,
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.aug:
            img = transforms.RandomErasing(p=config.erase_prob, scale=(0.02, 0.1), ratio=(0.3, 3.3))(img)

        if bboxes is not None and len(bboxes) > 0:
            bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2] / 2) / self.input_size[1]
            bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3] / 2) / self.input_size[0]
            bboxes[:, 2] = bboxes[:, 2] / self.input_size[1]
            bboxes[:, 3] = bboxes[:, 3] / self.input_size[0]
            bboxes = np.clip(bboxes, 0, 1)

        return img, bboxes

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
            img, bboxes = self._transform_with_bbox(img, bboxes)
            bbox_mask = np.zeros(config.num_heads, dtype=np.float32)
            bbox_mask[:min(n_digits, config.num_heads)] = 1.0
            return img, t.tensor(padded_label).long(), t.tensor(bboxes).float(), t.tensor(bbox_mask).float()
        else:
            if config.keep_aspect_ratio:
                w, h = img.size
                if w <= h:
                    new_w = self.input_size[1]
                    new_h = max(int(h * self.input_size[1] / w), self.input_size[0])
                else:
                    new_h = self.input_size[0]
                    new_w = max(int(w * self.input_size[0] / h), self.input_size[1])
                img = transforms.functional.resize(img, (new_h, new_w))
                i = max(0, (new_h - self.input_size[0]) // 2)
                j = max(0, (new_w - self.input_size[1]) // 2)
                img = transforms.functional.crop(img, i, j, self.input_size[0], self.input_size[1])
                img = transforms.functional.to_tensor(img)
                img = transforms.functional.normalize(img,
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                trans = transforms.Compose([
                    PadToSquare(fill=(127, 127, 127)),
                    transforms.Resize((self.input_size[0], self.input_size[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                img = trans(img)
            return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)


class PositionAwareAttentionHead(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, head_idx):
        super().__init__()
        self.head_idx = head_idx
        S = config.feat_spatial_size
        # 改进：使用可学习的位置编码而非固定高斯先验
        self.pos_embed = nn.Parameter(t.randn(1, config.pos_embed_channels, S, S) * 0.02)
        self.head_embed = nn.Parameter(t.randn(1, config.pos_embed_channels, 1, 1) * 0.02)
        self.num_attn_channels = config.num_attn_channels
        # 增加深度可分离卷积的层数以提升表达能力
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels + config.pos_embed_channels * 2, in_channels + config.pos_embed_channels * 2, 3,
                      padding=1, groups=in_channels + config.pos_embed_channels * 2, bias=False),
            nn.BatchNorm2d(in_channels + config.pos_embed_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels + config.pos_embed_channels * 2, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_attn_channels, 1),
        )
        self.attn_pool = nn.AdaptiveAvgPool2d(4)
        pool_dim = in_channels * 4 * 4
        self.feat_proj = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.cls_layer = nn.Linear(hidden_dim, num_classes)
        self.bbox_pool = nn.AdaptiveAvgPool2d(4)
        self.bbox_head = nn.Sequential(
            nn.Linear(in_channels * 4 * 4, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4),
            nn.Sigmoid()
        )

    def forward(self, x, return_attn=False):
        B, C, H, W = x.shape
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        pos = pos.expand(B, -1, -1, -1)
        head = self.head_embed.expand(B, -1, H, W)
        x_input = t.cat([x, pos, head], dim=1)
        attn_raw = self.attention_conv(x_input)
        if self.num_attn_channels == 1:
            attn_weights = F.softmax(attn_raw.view(B, -1), dim=1).view(B, 1, H, W)
        else:
            attn_per_ch = F.softmax(attn_raw.view(B, self.num_attn_channels, -1), dim=2)
            attn_per_ch = attn_per_ch.view(B, self.num_attn_channels, H, W)
            peak_conf = attn_per_ch.amax(dim=(2, 3))
            best_k = peak_conf.argmax(dim=1)
            attn_weights = attn_per_ch[t.arange(B, device=x.device), best_k].unsqueeze(1)
        weighted_feat = x * attn_weights
        pooled = self.attn_pool(weighted_feat).flatten(1)
        hidden = self.feat_proj(pooled)
        cls_out = self.cls_layer(hidden)
        bbox_feat = self.bbox_pool(weighted_feat).flatten(1)
        bbox_out = self.bbox_head(bbox_feat)
        if return_attn:
            return cls_out, bbox_out, hidden, attn_weights
        return cls_out, bbox_out, hidden


def generate_gaussian_attn_target(bbox_target, bbox_mask, feat_h, feat_w, device='cpu'):
    B, N, _ = bbox_target.shape
    grid_y = t.arange(feat_h, device=device, dtype=t.float32).view(1, 1, feat_h, 1) / feat_h
    grid_x = t.arange(feat_w, device=device, dtype=t.float32).view(1, 1, 1, feat_w) / feat_w
    cx = bbox_target[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    cy = bbox_target[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    bw = bbox_target[:, :, 2].clamp(min=0.02).unsqueeze(-1).unsqueeze(-1)
    bh = bbox_target[:, :, 3].clamp(min=0.02).unsqueeze(-1).unsqueeze(-1)
    sigma_x = bw / 2.5
    sigma_y = bh / 2.5
    gauss = t.exp(-((grid_x - cx) ** 2 / (2 * sigma_x ** 2) +
                    (grid_y - cy) ** 2 / (2 * sigma_y ** 2)))
    mask_expand = bbox_mask.unsqueeze(-1).unsqueeze(-1)
    gauss = gauss * mask_expand
    s = gauss.sum(dim=(2, 3), keepdim=True).clamp(min=1e-8)
    gauss = gauss / s
    return gauss


class AttentionSupervisionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_maps, bbox_target, bbox_mask):
        """
        修复：修正KL散度方向
        KL(P||Q) = sum(P * log(P/Q))，要求P是真实分布（target），Q是预测分布（pred）
        PyTorch的F.kl_div(input, target)计算的是 KL(target||input)
        因此应该传入: F.kl_div(pred.log(), gt)
        """
        if attn_maps is None or len(attn_maps) == 0:
            return t.tensor(0.0, device=bbox_target.device, requires_grad=True)
        B, N, _ = bbox_target.shape
        H, W = attn_maps[0].shape[2], attn_maps[0].shape[3]
        gt_gauss = generate_gaussian_attn_target(bbox_target, bbox_mask, H, W, device=attn_maps[0].device)
        loss = t.tensor(0.0, device=bbox_target.device)
        count = 0
        for h in range(min(len(attn_maps), N)):
            mask = bbox_mask[:, h] > 0
            if mask.sum() == 0:
                continue
            attn = attn_maps[h].squeeze(1)
            pred_h = attn[mask]
            gt_h = gt_gauss[mask, h]
            # 修复：正确的KL散度方向 KL(gt||pred) = sum(gt * log(gt/pred))
            # PyTorch F.kl_div(pred.log(), gt) 计算的就是这个
            loss = loss + F.kl_div(pred_h.clamp(min=1e-8).log(), gt_h, reduction='batchmean')
            count += 1
        if count == 0:
            return t.tensor(0.0, device=bbox_target.device, requires_grad=True)
        return loss / count


class HeadInteractionLayer(nn.Module):
    def __init__(self, feat_dim, num_heads, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        # 改进：使用正弦位置编码 + 可学习嵌入，加入几何先验
        # 正弦编码提供固定的位置信息，可学习部分适应任务
        self.num_heads = num_heads
        self.feat_dim = feat_dim
        
        # 正弦位置编码（固定，提供从左到右的几何先验）
        pos_encoding = self._create_sinusoidal_encoding(num_heads, feat_dim)
        self.register_buffer('sinusoidal_pos', pos_encoding)  # [1, N, D]
        
        # 可学习的位置偏移（适应特定任务）
        self.learnable_pos = nn.Parameter(t.randn(1, num_heads, feat_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=nhead,
            dim_feedforward=feat_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _create_sinusoidal_encoding(self, num_heads, feat_dim):
        """
        创建正弦位置编码，提供从左到右的几何先验
        """
        pos = t.arange(num_heads).float().unsqueeze(1)  # [N, 1]
        div_term = t.exp(t.arange(0, feat_dim, 2).float() * (-t.log(t.tensor(10000.0)) / feat_dim))
        
        encoding = t.zeros(1, num_heads, feat_dim)
        encoding[0, :, 0::2] = t.sin(pos * div_term)
        if feat_dim % 2 == 0:
            encoding[0, :, 1::2] = t.cos(pos * div_term)
        else:
            encoding[0, :, 1::2] = t.cos(pos * div_term[:-1])
        
        return encoding

    def forward(self, head_features):
        B = head_features[0].shape[0]
        stacked = t.stack(head_features, dim=1)  # [B, N, D]
        # 结合正弦编码和可学习编码，打破permutation invariance
        pos_embed = self.sinusoidal_pos + self.learnable_pos
        stacked = stacked + pos_embed.expand(B, -1, -1)
        encoded = self.encoder(stacked)
        return [encoded[:, i, :] for i in range(self.num_heads)]


class CrossHeadCommLayer(nn.Module):
    def __init__(self, feat_dim, num_heads, pos_channels):
        super().__init__()
        self.num_heads = num_heads
        self.pos_proj = nn.Conv2d(pos_channels, 32, 1, bias=False)
        self.comm_conv = nn.Sequential(
            nn.Conv2d(feat_dim + 32 * num_heads, feat_dim, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat, pos_embeds):
        B, C, H, W = feat.shape
        all_pos = []
        for pe in pos_embeds:
            p = F.interpolate(pe, size=(H, W), mode='bilinear', align_corners=False)
            p = self.pos_proj(p.expand(B, -1, -1, -1))
            all_pos.append(p)
        x = t.cat([feat] + all_pos, dim=1)
        return feat + self.comm_conv(x)


class DigitsResnet101(nn.Module):
    def __init__(self, class_num=11, num_heads=6):
        super(DigitsResnet101, self).__init__()
        self.num_heads = num_heads
        self.roi_gt_prob = config.roi_gt_prob
        self.backbone = FPNBackbone()
        self.pre_head_comm = CrossHeadCommLayer(config.multiscale_feat_dim, num_heads, config.pos_embed_channels)
        self.heads = nn.ModuleList([
            PositionAwareAttentionHead(config.multiscale_feat_dim, config.fc_hidden, class_num, head_idx=i)
            for i in range(num_heads)
        ])
        self.has_roi = config.roi_refine and roi_align is not None
        if self.has_roi:
            # 使用小型CNN处理ROI的空间特征，而非简单全局池化
            self.roi_cnn = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(config.multiscale_feat_dim, config.roi_feat_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(config.roi_feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(config.roi_feat_dim, config.roi_feat_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(config.roi_feat_dim),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(config.roi_feat_dim, config.roi_feat_dim),
                    nn.BatchNorm1d(config.roi_feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(config.dropout),
                ) for _ in range(num_heads)
            ])
            self.roi_cls_heads = nn.ModuleList([
                nn.Linear(config.roi_feat_dim, class_num) for _ in range(num_heads)
            ])
        self.head_interaction = HeadInteractionLayer(
            config.fc_hidden, num_heads,
            num_layers=config.head_interaction_layers,
            nhead=min(config.transformer_heads, num_heads),
            dropout=config.dropout
        )
        self.head_fc = nn.ModuleList([
            nn.Linear(config.fc_hidden, class_num) for _ in range(num_heads)
        ])

    def _extract_roi_feat(self, feat, bbox_pred, head_idx):
        """
        修复：正确处理bbox坐标归一化
        bbox_pred是sigmoid输出的[0,1]范围，表示(cx, cy, w, h)相对于图像尺寸的比例
        ROI Align需要绝对像素坐标[x1, y1, x2, y2]
        """
        B, C, H, W = feat.shape
        # bbox_pred格式: [cx, cy, w, h]，值域[0,1]
        cx, cy, bw, bh = bbox_pred[:, 0], bbox_pred[:, 1], bbox_pred[:, 2], bbox_pred[:, 3]
        
        # 转换为绝对像素坐标
        # cx, cy是中心点相对于特征图尺寸的比例
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H
        
        # 确保坐标在有效范围内
        x1 = x1.clamp(min=0, max=W-1)
        y1 = y1.clamp(min=0, max=H-1)
        x2 = x2.clamp(min=0, max=W-1)
        y2 = y2.clamp(min=0, max=H-1)
        
        batch_idx = t.arange(B, device=feat.device).float()
        boxes = t.stack([batch_idx, x1, y1, x2, y2], dim=1)
        
        # 提取7x7的ROI特征，保留空间信息
        roi_feat = roi_align(feat, boxes, output_size=7, spatial_scale=1.0)
        
        # 通过小型CNN处理空间特征，而非简单的mean pooling
        roi_processed = self.roi_cnn[head_idx](roi_feat)
        return self.roi_cls_heads[head_idx](roi_processed)

    def _apply_roi_refine(self, feat, cls_outs, bbox_outs, gt_bboxes=None):
        """
        修复：改进Teacher Forcing策略，确保训练-推理一致性
        """
        if not self.has_roi:
            return cls_outs
        
        # 训练时：根据roi_gt_prob决定是否使用GT bbox
        # 推理时：始终使用预测的bbox
        use_gt = False
        if self.training and gt_bboxes is not None and config.roi_teacher_forcing:
            use_gt = t.rand(1).item() < self.roi_gt_prob
        
        roi_cls = tuple(
            self._extract_roi_feat(feat, gt_bboxes[:, h, :] if use_gt else bbox_outs[h], h)
            for h in range(self.num_heads)
        )
        
        if not self.training:
            # 推理阶段：使用门控机制融合原始分类和ROI分类
            refined = []
            for h in range(self.num_heads):
                # 计算"非空字符"的概率作为gate
                p_no_digit = F.softmax(cls_outs[h].detach(), dim=1)[:, 10:11]
                gated_roi = roi_cls[h] * (1 - p_no_digit)
                refined.append(cls_outs[h] + gated_roi)
            return tuple(refined)
        else:
            # 训练阶段：直接相加（因为可能使用GT bbox）
            return tuple(cls_outs[h] + roi_cls[h] for h in range(self.num_heads))

    def set_roi_gt_prob(self, prob):
        self.roi_gt_prob = prob

    def forward(self, img, gt_bboxes=None):
        feat = self.backbone(img)
        feat = self.pre_head_comm(feat, [h.pos_embed for h in self.heads])
        results = [head(feat) for head in self.heads]
        bbox_outs = tuple(r[1] for r in results)
        head_feats = [r[2] for r in results]
        interacted = self.head_interaction(head_feats)
        cls_outs = tuple(self.head_fc[h](interacted[h]) for h in range(self.num_heads))
        cls_outs = self._apply_roi_refine(feat, cls_outs, bbox_outs, gt_bboxes)
        return cls_outs, bbox_outs

    def forward_with_attn(self, img, gt_bboxes=None):
        feat = self.backbone(img)
        feat = self.pre_head_comm(feat, [h.pos_embed for h in self.heads])
        cls_outs, bbox_outs, attn_maps = [], [], []
        head_feats = []
        for head in self.heads:
            cls_out, bbox_out, hidden, attn = head(feat, return_attn=True)
            cls_outs.append(cls_out)
            bbox_outs.append(bbox_out)
            head_feats.append(hidden)
            attn_maps.append(attn)
        bbox_tuple = tuple(bbox_outs)
        interacted = self.head_interaction(head_feats)
        cls_list = tuple(self.head_fc[h](interacted[h]) for h in range(self.num_heads))
        cls_list = self._apply_roi_refine(feat, cls_list, bbox_tuple, gt_bboxes)
        return cls_list, bbox_tuple, attn_maps

    def forward_with_probs(self, img):
        feat = self.backbone(img)
        feat = self.pre_head_comm(feat, [h.pos_embed for h in self.heads])
        results = [head(feat) for head in self.heads]
        bbox_outs = tuple(r[1] for r in results)
        head_feats = [r[2] for r in results]
        interacted = self.head_interaction(head_feats)
        cls_outs = tuple(self.head_fc[h](interacted[h]) for h in range(self.num_heads))
        cls_outs = self._apply_roi_refine(feat, cls_outs, bbox_outs)
        return tuple(F.softmax(c, dim=1) for c in cls_outs)


class TransformerDigitsModel(nn.Module):
    def __init__(self, class_num=11, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.backbone = FPNBackbone()
        feat_dim = config.multiscale_feat_dim
        self.query_embed = nn.Parameter(t.randn(num_heads, feat_dim) * 0.02)
        self.pos_2d = nn.Parameter(t.randn(1, feat_dim, 20, 20) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=config.transformer_heads,
            dim_feedforward=feat_dim * 4, dropout=config.dropout,
            batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.transformer_layers)
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(feat_dim // 2, class_num)
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 4, 4),
            nn.Sigmoid()
        )

    def _prepare_memory(self, feat):
        B, C, H, W = feat.shape
        pos = F.interpolate(self.pos_2d, size=(H, W), mode='bilinear', align_corners=False)
        memory = (feat + pos).flatten(2).permute(0, 2, 1)
        return memory, H, W

    def _decode_with_attn(self, queries, memory, tgt_mask, H_feat, W_feat):
        attn_maps = []
        x = queries
        for layer in self.decoder.layers:
            x_norm = layer.norm1(x)
            x_sa, _ = layer.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask, need_weights=False)
            x = x + layer.dropout1(x_sa)
            x_norm2 = layer.norm2(x)
            x_ca, cross_attn_w = layer.multihead_attn(x_norm2, memory, memory, need_weights=True)
            x = x + layer.dropout2(x_ca)
            if cross_attn_w is not None:
                B, nq, L = cross_attn_w.shape
                attn_2d = cross_attn_w.view(B, nq, H_feat, W_feat)
                head_attns = [attn_2d[:, h, :, :] for h in range(self.num_heads)]
                attn_maps.append(head_attns)
            x_norm3 = layer.norm3(x)
            x_ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_norm3))))
            x = x + layer.dropout3(x_ff)
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        avg_attns = []
        if attn_maps:
            for h in range(self.num_heads):
                stacked = t.stack([layer_attns[h] for layer_attns in attn_maps], dim=0)
                avg_attns.append(stacked.mean(dim=0).unsqueeze(1))
        return x, avg_attns

    def forward(self, img, gt_bboxes=None):
        feat = self.backbone(img)
        memory, H, W = self._prepare_memory(feat)
        B = feat.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        # 修复：移除causal mask，使用None允许所有head互相可见（并行预测任务不需要autoregressive）
        decoded = self.decoder(queries, memory, tgt_mask=None)
        cls_outs = tuple(self.cls_head(decoded[:, i, :]) for i in range(self.num_heads))
        bbox_outs = tuple(self.bbox_head(decoded[:, i, :]) for i in range(self.num_heads))
        return cls_outs, bbox_outs

    def forward_with_attn(self, img, gt_bboxes=None):
        feat = self.backbone(img)
        memory, H, W = self._prepare_memory(feat)
        B = feat.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        # 修复：移除causal mask以支持双向attention
        decoded, attn_maps = self._decode_with_attn(queries, memory, tgt_mask=None, H_feat=H, W_feat=W)
        cls_outs = tuple(self.cls_head(decoded[:, i, :]) for i in range(self.num_heads))
        bbox_outs = tuple(self.bbox_head(decoded[:, i, :]) for i in range(self.num_heads))
        return cls_outs, bbox_outs, attn_maps if attn_maps else None

    def forward_with_probs(self, img):
        cls_outs, _ = self.forward(img)
        return tuple(F.softmax(c, dim=1) for c in cls_outs)


def create_model(model_type=None):
    if model_type is None:
        model_type = config.model_type
    if model_type == 'transformer':
        return TransformerDigitsModel(config.class_num, config.num_heads)
    return DigitsResnet101(config.class_num, config.num_heads)


def attention_diversity_loss(attn_maps):
    if attn_maps is None or len(attn_maps) < 2:
        return t.tensor(0.0, device='cuda' if t.cuda.is_available() else 'cpu', requires_grad=True)
    n = len(attn_maps)
    loss = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ai = attn_maps[i].flatten(2)
            aj = attn_maps[j].flatten(2)
            loss = loss + (ai * aj).sum(dim=2).mean()
    return loss / (n * (n - 1) / 2)


def spatial_ordering_loss(attn_maps, bbox_preds=None, bbox_mask=None):
    """
    改进的空间排序损失：同时考虑attention maps和bbox预测的几何一致性
    """
    if attn_maps is None or len(attn_maps) < 2:
        return t.tensor(0.0, device='cuda' if t.cuda.is_available() else 'cpu', requires_grad=True)
    
    loss = t.tensor(0.0, device=attn_maps[0].device)
    
    # 原有的attention-based ordering loss
    for i in range(len(attn_maps) - 1):
        ai = attn_maps[i].flatten(2)
        aj = attn_maps[i + 1].flatten(2)
        H, W = attn_maps[i].shape[2], attn_maps[i].shape[3]
        grid_x = t.arange(W, device=ai.device).float()
        grid_x = grid_x.unsqueeze(0).expand(H, W).reshape(1, 1, -1)
        ci = (ai * grid_x).sum(dim=2) / (ai.sum(dim=2) + 1e-8)
        cj = (aj * grid_x).sum(dim=2) / (aj.sum(dim=2) + 1e-8)
        violation = F.relu(ci - cj)
        loss = loss + violation.mean()
    
    # 新增：bbox-based ordering loss（如果提供了bbox预测）
    if bbox_preds is not None and bbox_mask is not None:
        for i in range(len(bbox_preds) - 1):
            # 获取第i个和第i+1个head的bbox中心x坐标
            cx_i = bbox_preds[i][:, 0]  # [B]
            cx_j = bbox_preds[i+1][:, 0]  # [B]
            
            # 只有当两个head都有有效字符时才施加约束
            mask_both_valid = (bbox_mask[:, i] > 0) & (bbox_mask[:, i+1] > 0)
            if mask_both_valid.sum() > 0:
                # 要求cx_i < cx_j（从左到右）
                violation = F.relu(cx_i[mask_both_valid] - cx_j[mask_both_valid])
                loss = loss + violation.mean() * 0.5  # 权重稍低
    
    return loss


class LabelSmoothEntropy(nn.Module):
    def __init__(self, smooth=0.1, class_weights=None, size_average='mean'):
        super(LabelSmoothEntropy, self).__init__()
        self.size_average = size_average
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, preds, targets):
        lb_pos, lb_neg = 1 - self.smooth, self.smooth / (preds.shape[1] - 1)
        smoothed_lb = t.zeros_like(preds).fill_(lb_neg).scatter_(1, targets[:, None], lb_pos)
        log_soft = F.log_softmax(preds, dim=1)
        if self.class_weights is not None:
            loss = -log_soft * smoothed_lb * self.class_weights[None, :]
        else:
            loss = -log_soft * smoothed_lb
        loss = loss.sum(1)
        if self.size_average == 'mean':
            return loss.mean()
        elif self.size_average == 'sum':
            return loss.sum()
        elif self.size_average == 'none':
            return loss
        else:
            raise NotImplementedError


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with t.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


def mixup_data(x, y, alpha=0.2):
    """Mixup已禁用，保留函数以防兼容性问题"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = t.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, bbox_target, bbox_mask, alpha=1.0):
    """
    CutMix数据增强：更适合目标检测/多任务学习场景
    返回混合后的图像、标签、bbox和mask
    """
    batch_size = x.size(0)
    index = t.randperm(batch_size).to(x.device)
    
    # 生成随机裁剪区域
    lam = t.distributions.Beta(alpha, alpha).sample().item()
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = int(t.randint(0, W, (1,)).item())
    cy = int(t.randint(0, H, (1,)).item())
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 执行CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # 计算实际的lambda（基于裁剪区域面积）
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    # 标签和bbox也按照lam混合
    y_a, y_b = y, y[index]
    bbox_a, bbox_b = bbox_target, bbox_target[index]
    mask_a, mask_b = bbox_mask, bbox_mask[index]
    
    return mixed_x, y_a, y_b, bbox_a, bbox_b, mask_a, mask_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer:
    def __init__(self, val=True, model_type=None):
        self.device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        print(f'Using device: {self.device}')
        self.use_amp = self.device.type == 'cuda'
        self.model_type = model_type or config.model_type
        
        self._base_seed = 42
        self._train_generator = None
        
        self.train_set = DigitsDataset(mode='train', aug=True,
                                       input_size=(config.input_height, config.input_width))
        # 使用初始种子创建生成器（对应 epoch 0）
        self._train_generator = self._make_epoch_generator(self._base_seed, epoch=0)
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=True,
                                       persistent_workers=NUM_WORKERS > 0,
                                       drop_last=True,
                                       prefetch_factor=2,
                                       generator=self._train_generator)
        if val:
            self.val_set = DigitsDataset(mode='val', aug=False,
                                         input_size=(config.input_height, config.input_width))
            self.val_loader = DataLoader(self.val_set, batch_size=config.batch_size,
                                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                                         persistent_workers=NUM_WORKERS > 0,
                                         prefetch_factor=2)
        else:
            self.val_loader = None

        self.model = create_model(self.model_type).to(self.device)
        self.ema = ModelEMA(self.model, decay=config.ema_decay)
        if config.use_torch_compile and t.cuda.is_available():
            self.model = t.compile(self.model, mode="reduce-overhead")
            print('✅ torch.compile enabled')

        self.attn_supervision = AttentionSupervisionLoss()

        class_weights = self._compute_class_weights()
        
        self.head_criteria = nn.ModuleList()
        for h in range(config.num_heads):
            self.head_criteria.append(LabelSmoothEntropy(smooth=config.smooth, class_weights=class_weights, size_average='none'))

        backbone_params = list(self.model.backbone.parameters())
        other_params = [p for n, p in self.model.named_parameters() if not n.startswith('backbone.')]
        self.optimizer = SGD([
            {'params': backbone_params, 'lr': config.lr * config.backbone_lr_factor},
            {'params': other_params, 'lr': config.lr},
        ], momentum=config.momentum, weight_decay=config.weights_decay, nesterov=True)

        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01,
                                    total_iters=config.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epoches - config.warmup_epochs,
                                             eta_min=1e-6)
        self.lr_scheduler = SequentialLR(self.optimizer,
                                         schedulers=[warmup_scheduler, cosine_scheduler],
                                         milestones=[config.warmup_epochs])

        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.best_acc = 0
        self.best_checkpoint_path = ''
        self.train_log = []
        self.patience_counter = 0
        self.early_stop_triggered = False
        if config.pretrained is not None:
            self.load_model(config.pretrained, save_opt=False)
            if self.val_loader is not None:
                acc = self.eval()
            self.best_acc = acc
            print('Load model from %s, Eval Acc: %.2f' % (config.pretrained, acc * 100))
            print('Warning: Optimizer and scheduler NOT restored. Using new config.')
    
    def _make_epoch_generator(self, base_seed, epoch):
        """创建指定epoch的随机数生成器
        
        确保每个epoch使用唯一的种子，保证数据加载顺序的随机性。
        种子计算公式: base_seed + epoch * 1000，确保不同epoch的生成器产生不同的随机序列。
        
        Args:
            base_seed: 基础种子值
            epoch: 当前epoch编号
        
        Returns:
            torch.Generator: 配置好种子的随机数生成器
        """
        epoch_seed = base_seed + epoch * 1000
        generator = t.Generator()
        generator.manual_seed(epoch_seed)
        return generator
    
    def _cleanup_dataloader(self, loader):
        if loader is not None:
            if hasattr(loader, '_iterator'):
                try:
                    del loader._iterator
                except Exception:
                    pass
            gc.collect()
    
    def _pre_epoch_hook(self, epoch):
        # 使用统一的种子设置函数，确保全局种子和数据加载器种子同步
        epoch_seed = self._base_seed + epoch * 1000
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        t.manual_seed(epoch_seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(epoch_seed)
        
        if hasattr(self.model, 'set_roi_gt_prob'):
            import math
            if epoch < config.warmup_epochs:
                self.model.set_roi_gt_prob(1.0)
            else:
                decay_end = int(config.epoches * 0.8)
                if epoch >= decay_end:
                    self.model.set_roi_gt_prob(0.0)
                else:
                    progress = (epoch - config.warmup_epochs) / max(decay_end - config.warmup_epochs, 1)
                    self.model.set_roi_gt_prob(0.5 * (1 + math.cos(math.pi * progress)))
        
        self._cleanup_dataloader(self.train_loader)
        # 使用与全局种子同步的生成器种子，确保数据加载顺序在每个epoch都不同
        self._train_generator = self._make_epoch_generator(self._base_seed, epoch=epoch)
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=True,
                                       persistent_workers=False,
                                       drop_last=True,
                                       prefetch_factor=2,
                                       generator=self._train_generator)
        print(f'[EPOCH-PRE] epoch={epoch+1} seed={epoch_seed}')
        print(f'[EPOCH-PRE] DataLoader generator seed={epoch_seed}, ensuring unique random sequence')

    def _compute_class_weights(self):
        """
        改进：动态计算类别权重，基于训练集统计
        避免硬编码空字符（class 10）的权重
        """
        # 统计训练集中每个类别的出现次数
        class_counts = t.zeros(config.class_num)
        
        for img, label, bbox_target, bbox_mask in self.train_loader:
            # label形状: [B, num_heads]
            for h in range(config.num_heads):
                labels_h = label[:, h]
                for cls_idx in range(config.class_num):
                    class_counts[cls_idx] += (labels_h == cls_idx).sum().item()
        
        # 计算权重：inverse frequency
        # 添加平滑项避免除零
        class_weights = 1.0 / (class_counts + 1e-6)
        
        # 归一化使权重总和为class_num
        class_weights = class_weights * config.class_num / class_weights.sum()
        
        # 移动到设备
        class_weights = class_weights.to(self.device)
        
        print(f'✅ Computed class weights: {class_weights.cpu().numpy()}')
        print(f'   Class 10 (empty) weight: {class_weights[10].item():.3f}')
        
        return class_weights

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            if self.early_stop_triggered:
                print(f'\n🛑 Early stopping triggered at epoch {epoch + 1}')
                print(f'   Best val acc: {self.best_acc * 100:.2f}%')
                print(f'   Best checkpoint: {self.best_checkpoint_path}')
                break
            
            self._pre_epoch_hook(epoch)
            
            train_acc = self.train_epoch(epoch)
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1} LR: {current_lr:.6f}')

            if (epoch + 1) % config.eval_interval == 0:
                print('Start Evaluation')
                acc = 0.0
                if self.val_loader is not None:
                    acc = self.eval()
                self.train_log.append({
                    'epoch': epoch + 1,
                    'train_acc': train_acc,
                    'val_acc': acc * 100,
                    'lr': current_lr
                })
                if acc > self.best_acc:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    save_path = os.path.join(config.checkpoints,
                                             'epoch-resnet101-%d-acc-%.2f.pth' % (epoch + 1, acc * 100))
                    self.save_model(save_path, save_opt=True)
                    print('%s saved successfully...' % save_path)
                    self.best_acc = acc
                    self.best_checkpoint_path = save_path
                    # 重置patience计数器
                    self.patience_counter = 0
                    print(f'✅ New best model! Patience counter reset to 0')
                else:
                    # 验证集准确率没有提升，增加patience计数
                    self.patience_counter += 1
                    print(f'⚠️  No improvement. Patience: {self.patience_counter}/{config.early_stopping_patience}')
                    
                    # 检查是否达到patience限制
                    if self.patience_counter >= config.early_stopping_patience:
                        self.early_stop_triggered = True
                        print(f'\n🛑 Early stopping will be triggered after this epoch')

    def train_epoch(self, epoch):
        total_loss = 0
        corrects = 0
        total = 0
        tbar = tqdm(self.train_loader)
        self.model.train()

        for i, (img, label, bbox_target, bbox_mask) in enumerate(tbar):
            img = img.to(self.device)
            label = label.to(self.device)
            bbox_target = bbox_target.to(self.device)
            bbox_mask = bbox_mask.to(self.device)
            
            # 应用CutMix数据增强（替代Mixup）
            if config.cutmix_prob > 0 and random.random() < config.cutmix_prob and config.cutmix_alpha > 0:
                img, label_a, label_b, bbox_a, bbox_b, mask_a, mask_b, lam = cutmix_data(
                    img, label, bbox_target, bbox_mask, alpha=config.cutmix_alpha)
                use_cutmix = True
            else:
                label_a = label
                label_b = label
                bbox_a = bbox_target
                bbox_b = bbox_target
                mask_a = bbox_mask
                mask_b = bbox_mask
                lam = 1.0
                use_cutmix = False
            
            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                pred, pred_bboxes, attn_maps = self.model.forward_with_attn(img, gt_bboxes=bbox_target)
                
                # 动态掩码：根据bbox_mask计算每个样本的真实长度
                true_lengths = bbox_mask.sum(dim=1).long()  # [B]
                
                # 分类损失：只对有效的head计算损失
                cls_loss = t.tensor(0.0, device=self.device, requires_grad=True)
                for h in range(config.num_heads):
                    # 创建mask：只有当该head位置有真实字符时才计算损失
                    valid_mask = (true_lengths > h).float()  # [B], 1 if head h is valid
                    if valid_mask.sum() > 0:
                        if use_cutmix:
                            # CutMix情况：对两个标签分别计算损失并加权
                            head_loss_a = self.head_criteria[h](pred[h], label_a[:, h])
                            head_loss_b = self.head_criteria[h](pred[h], label_b[:, h])
                            head_loss = lam * head_loss_a + (1 - lam) * head_loss_b
                        else:
                            head_loss = self.head_criteria[h](pred[h], label[:, h])
                        # 只累加有效样本的损失
                        cls_loss = cls_loss + (head_loss * valid_mask).sum() / valid_mask.sum()
                
                if use_cutmix:
                    div_loss = t.tensor(0.0, device=self.device, requires_grad=True)
                    attn_sup_loss = t.tensor(0.0, device=self.device, requires_grad=True)
                    ord_loss = t.tensor(0.0, device=self.device, requires_grad=True)
                    bbox_loss = t.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    div_loss = attention_diversity_loss(attn_maps)
                    # 改进：传入bbox预测和mask以增强几何一致性约束
                    ord_loss = spatial_ordering_loss(attn_maps, bbox_preds=pred_bboxes, bbox_mask=bbox_mask)
                    attn_sup_loss = self.attn_supervision(attn_maps, bbox_target, bbox_mask)
                    bbox_loss = t.tensor(0.0, device=self.device, requires_grad=True)
                    valid_bbox_sum = (bbox_target * bbox_mask.unsqueeze(-1)).sum(dim=1)
                    valid_bbox_count = bbox_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    mean_bbox = valid_bbox_sum / valid_bbox_count
                    for h in range(config.num_heads):
                        mask = bbox_mask[:, h]
                        if mask.sum() > 0:
                            bbox_loss_h = F.smooth_l1_loss(
                                pred_bboxes[h][mask > 0], bbox_target[:, h, :][mask > 0])
                            bbox_loss = bbox_loss + bbox_loss_h
                        else:
                            empty_mask = (mask == 0)
                            if empty_mask.sum() > 0:
                                bbox_loss = bbox_loss + F.smooth_l1_loss(
                                    pred_bboxes[h][empty_mask], mean_bbox[empty_mask].detach()) * 0.3
                
                # 动态调整loss权重，前期侧重分类，后期增强几何约束
                epoch_ratio = (epoch + 1) / config.epoches
                dynamic_ordering_weight = config.ordering_loss_weight * min(1.0, epoch_ratio * 2)
                dynamic_attn_weight = config.attn_supervision_weight * min(1.0, epoch_ratio * 1.5)
                
                loss = (config.cls_loss_weight * cls_loss + config.bbox_loss_weight * bbox_loss
                        + config.attn_diversity_weight * div_loss
                        + dynamic_ordering_weight * ord_loss
                        + dynamic_attn_weight * attn_sup_loss)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)
            total_loss += loss.item()

            total += img.size(0)
            # 修复：使用字符级准确率而非joint accuracy
            if config.use_char_level_acc:
                # 只计算有效字符位置的准确率
                for h in range(config.num_heads):
                    valid_mask = (true_lengths > h).float()
                    if valid_mask.sum() > 0:
                        corrects += ((pred[h].argmax(1) == label[:, h]) * valid_mask).sum().item()
                # 计算总的有效字符数
                total_chars = true_lengths.sum().item()
                tbar.set_description(
                    'Epoch %d, loss: %.3f, char_acc: %.3f' % (epoch + 1, total_loss / (i + 1), corrects * 100 / max(total_chars, 1)))
            else:
                # 原有的joint accuracy（所有head都正确）
                temp = t.stack([pred[h].argmax(1) == label[:, h] for h in range(config.num_heads)], dim=1)
                corrects += t.all(temp, dim=1).sum().item()
                tbar.set_description(
                    'Epoch %d, loss: %.3f, joint_acc: %.3f' % (epoch + 1, total_loss / (i + 1), corrects * 100 / max(total, 1)))
        
        # 返回字符级准确率或joint accuracy
        if config.use_char_level_acc:
            total_chars = sum([int((bbox_mask > 0).sum(dim=1)[i].item()) for i in range(len(bbox_mask))])
            return corrects * 100 / max(total_chars, 1)
        else:
            return corrects * 100 / max(total, 1)

    def eval(self):
        if self.ema is not None:
            model = self.ema.ema
        else:
            model = self.model
        model.eval()
        corrects = 0
        total = 0
        total_chars = 0
        with t.no_grad():
            tbar = tqdm(self.val_loader)
            for i, (img, label, bbox_target, bbox_mask) in enumerate(tbar):
                img = img.to(self.device)
                label = label.to(self.device)
                bbox_mask = bbox_mask.to(self.device)
                pred_cls, _ = model(img)
                
                if config.use_char_level_acc:
                    # 字符级准确率：只计算有效字符位置
                    true_lengths = bbox_mask.sum(dim=1).long()
                    for h in range(config.num_heads):
                        valid_mask = (true_lengths > h).float()
                        if valid_mask.sum() > 0:
                            corrects += ((pred_cls[h].argmax(1) == label[:, h]) * valid_mask).sum().item()
                    total_chars += true_lengths.sum().item()
                    tbar.set_description('Val Char Acc: %.2f' % (corrects * 100 / max(total_chars, 1)))
                else:
                    # Joint accuracy
                    temp = t.stack([pred_cls[h].argmax(1) == label[:, h] for h in range(config.num_heads)], dim=1)
                    corrects += t.all(temp, dim=1).sum().item()
                    total += img.size(0)
                    tbar.set_description('Val Joint Acc: %.2f' % (corrects * 100 / total))
                
                del img, label, pred_cls
        t.cuda.empty_cache()
        self.model.train()
        
        if config.use_char_level_acc:
            return corrects / max(total_chars, 1)
        else:
            return corrects / total

    def eval_detailed(self):
        if self.ema is not None:
            model = self.ema.ema
        else:
            model = self.model
        model.eval()
        head_corrects = [0] * config.num_heads
        head_totals = [0] * config.num_heads
        corrects = 0
        total = 0
        total_chars = 0
        with t.no_grad():
            for img, label, bbox_target, bbox_mask in tqdm(self.val_loader, desc='Detailed Eval'):
                img = img.to(self.device)
                label = label.to(self.device)
                bbox_mask = bbox_mask.to(self.device)
                pred_cls, _ = model(img)
                
                true_lengths = bbox_mask.sum(dim=1).long()
                
                for h in range(config.num_heads):
                    valid_mask = (true_lengths > h).float()
                    head_corrects[h] += ((pred_cls[h].argmax(1) == label[:, h]) * valid_mask).sum().item()
                    head_totals[h] += valid_mask.sum().item()
                
                if config.use_char_level_acc:
                    for h in range(config.num_heads):
                        valid_mask = (true_lengths > h).float()
                        if valid_mask.sum() > 0:
                            corrects += ((pred_cls[h].argmax(1) == label[:, h]) * valid_mask).sum().item()
                    total_chars += true_lengths.sum().item()
                else:
                    temp = t.stack([pred_cls[h].argmax(1) == label[:, h] for h in range(config.num_heads)], dim=1)
                    corrects += t.all(temp, dim=1).sum().item()
                    total += img.size(0)
                
                del img, label, pred_cls
        
        for h in range(config.num_heads):
            if head_totals[h] > 0:
                print(f'  Head {h+1} Char Acc: {head_corrects[h] / head_totals[h] * 100:.2f}% ({head_corrects[h]}/{head_totals[h]})')
        
        if config.use_char_level_acc:
            print(f'  Overall Char Acc: {corrects / max(total_chars, 1) * 100:.2f}%')
        else:
            print(f'  Joint Acc: {corrects / total * 100:.2f}%')
        
        t.cuda.empty_cache()
        self.model.train()
        
        if config.use_char_level_acc:
            return corrects / max(total_chars, 1)
        else:
            return corrects / total

    def eval_tta(self):
        if self.ema is not None:
            model = self.ema.ema
        else:
            model = self.model
        model.eval()

        all_probs = [t.zeros(len(self.val_set), config.class_num, device='cpu') for _ in range(config.num_heads)]
        all_labels = t.zeros(len(self.val_set), config.num_heads, dtype=t.long, device='cpu')

        for tta_size in config.tta_sizes:
            val_set_tta = DigitsDataset(mode='val', aug=False,
                                        input_size=(tta_size, tta_size))
            val_loader_tta = DataLoader(val_set_tta, batch_size=config.batch_size,
                                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                                        persistent_workers=False)
            sample_idx = 0
            with t.no_grad():
                for img, label, _, _ in tqdm(val_loader_tta, desc=f'TTA size={tta_size}'):
                    img = img.to(self.device)
                    probs = model.forward_with_probs(img)
                    bs = img.size(0)
                    for h in range(config.num_heads):
                        all_probs[h][sample_idx:sample_idx + bs] += probs[h].cpu()
                    if tta_size == config.tta_sizes[0]:
                        all_labels[sample_idx:sample_idx + bs] = label.cpu()
                    sample_idx += bs
                    del img, probs
                t.cuda.empty_cache()

        pred_heads = t.stack([all_probs[h].argmax(1) for h in range(config.num_heads)], dim=1)
        corrects = t.all(pred_heads == all_labels, dim=1).sum().item()
        acc = corrects / len(self.val_set)
        print(f'TTA Val Acc: {acc * 100:.2f}')
        self.model.train()
        return acc

    def save_model(self, save_path, save_opt=False, save_config=False):
        if self.ema is not None:
            dicts = {'model': self.ema.ema.state_dict()}
        else:
            dicts = {'model': self.model.state_dict()}
        dicts['model_type'] = self.model_type
        if save_opt:
            dicts['opt'] = self.optimizer.state_dict()
            dicts['lr_scheduler'] = self.lr_scheduler.state_dict()
            dicts['scaler'] = self.scaler.state_dict()
        if save_config:
            dicts['config'] = {s: config.__getattribute__(s) for s in dir(config) if not s.startswith('_')}
        dicts['train_log'] = self.train_log
        dicts['best_acc'] = self.best_acc
        dicts['best_checkpoint_path'] = self.best_checkpoint_path
        t.save(dicts, save_path)

    def load_model(self, load_path, skip_load_weights=False, save_opt=False, save_config=False):
        dicts = t.load(load_path, map_location=self.device, weights_only=False)
        if not skip_load_weights:
            self.model.load_state_dict(dicts['model'])
            if self.ema is not None:
                self.ema.ema.load_state_dict(dicts['model'])
        if save_opt:
            self.optimizer.load_state_dict(dicts['opt'])
            if 'lr_scheduler' in dicts:
                self.lr_scheduler.load_state_dict(dicts['lr_scheduler'])
            if 'scaler' in dicts:
                self.scaler.load_state_dict(dicts['scaler'])
            if 'train_log' in dicts and len(dicts['train_log']) > 0:
                config.start_epoch = dicts['train_log'][-1]['epoch']
                print(f'Resuming from epoch {config.start_epoch + 1}')
            if 'best_acc' in dicts:
                self.best_acc = dicts['best_acc']
            if 'best_checkpoint_path' in dicts:
                self.best_checkpoint_path = dicts['best_checkpoint_path']
        else:
            if 'best_acc' in dicts:
                self.best_acc = dicts['best_acc']
        if save_config:
            for k, v in dicts['config'].items():
                config.__setattr__(k, v)


def parse2class(prediction):
    char_list = [str(i) for i in range(10)] + ['']
    preds = [p.argmax(1) for p in prediction]
    batch_size = preds[0].shape[0]
    res = []
    for b in range(batch_size):
        chars = []
        for h in range(len(preds)):
            idx = preds[h][b].item()
            if idx == 10:
                break
            chars.append(char_list[idx])
        res.append(''.join(chars))
    return res


def parse2class_from_probs(all_probs):
    char_list = [str(i) for i in range(10)] + ['']
    res = []
    for b in range(len(all_probs[0])):
        chars = []
        for h in range(len(all_probs)):
            idx = all_probs[h][b].argmax().item()
            if idx == 10:
                break
            chars.append(char_list[idx])
        res.append(''.join(chars))
    return res


def write2csv(results, csv_path):
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    df['file_name'] = df['file_name'].apply(lambda x: os.path.basename(x))
    df.to_csv(csv_path, sep=',', index=None)
    print('Results saved to %s' % csv_path)


def predicts(model_path, csv_path, use_tta=True, model_type=None):
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    mt = model_type or config.model_type
    res_net = create_model(mt).to(device)
    ckpt = t.load(model_path, map_location=device, weights_only=False)
    res_net.load_state_dict(ckpt['model'])
    if 'model_type' in ckpt:
        mt = ckpt['model_type']
    print('Load model from %s successfully' % model_path)
    res_net.eval()

    if use_tta:
        test_set_for_count = DigitsDataset(mode='test', aug=False,
                                           input_size=(config.input_height, config.input_width))
        num_test = len(test_set_for_count)
        all_probs = [t.zeros(num_test, config.class_num) for _ in range(config.num_heads)]
        all_names = []

        for tta_size in config.tta_sizes:
            test_set_tta = DigitsDataset(mode='test', aug=False,
                                         input_size=(tta_size, tta_size))
            test_loader_tta = DataLoader(test_set_tta, batch_size=config.batch_size,
                                         shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                                         drop_last=False, persistent_workers=False)
            sample_idx = 0
            with t.no_grad():
                for img, img_names in tqdm(test_loader_tta, desc=f'Test TTA size={tta_size}'):
                    img = img.to(device)
                    probs = res_net.forward_with_probs(img)
                    bs = img.size(0)
                    for h in range(config.num_heads):
                        all_probs[h][sample_idx:sample_idx + bs] += probs[h].cpu()
                    if tta_size == config.tta_sizes[0]:
                        all_names.extend(img_names)
                    sample_idx += bs

        results = [[name, code] for name, code in zip(all_names, parse2class_from_probs(all_probs))]
    else:
        test_loader = DataLoader(DigitsDataset(mode='test', aug=False,
                                               input_size=(config.input_height, config.input_width)),
                                 batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True, drop_last=False, persistent_workers=NUM_WORKERS > 0)
        results = []
        with t.no_grad():
            for img, img_names in tqdm(test_loader):
                img = img.to(device)
                pred_cls, _ = res_net(img)
                results += [[name, code] for name, code in zip(img_names, parse2class(pred_cls))]

    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results


def ensemble_predict(model_paths, csv_path, model_type=None):
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    mt = model_type or config.model_type
    models = []
    for mp in model_paths:
        m = create_model(mt).to(device)
        ckpt = t.load(mp, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model'])
        m.eval()
        models.append(m)
        print(f'Loaded model: {mp}')

    test_set_for_count = DigitsDataset(mode='test', aug=False,
                                       input_size=(config.input_height, config.input_width))
    num_test = len(test_set_for_count)
    all_probs = [t.zeros(num_test, config.class_num) for _ in range(config.num_heads)]
    all_names = []

    for tta_size in config.tta_sizes:
        test_set_tta = DigitsDataset(mode='test', aug=False,
                                     input_size=(tta_size, tta_size))
        test_loader_tta = DataLoader(test_set_tta, batch_size=config.batch_size,
                                     shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                                     drop_last=False, persistent_workers=False)
        sample_idx = 0
        with t.no_grad():
            for img, img_names in tqdm(test_loader_tta, desc=f'Ensemble TTA size={tta_size}'):
                img = img.to(device)
                bs = img.size(0)
                for model in models:
                    probs = model.forward_with_probs(img)
                    for h in range(config.num_heads):
                        all_probs[h][sample_idx:sample_idx + bs] += probs[h].cpu()
                if tta_size == config.tta_sizes[0]:
                    all_names.extend(img_names)
                sample_idx += bs

    results = [[name, code] for name, code in zip(all_names, parse2class_from_probs(all_probs))]
    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = glob(os.path.join(checkpoint_dir, '*.pth'))
    if not checkpoint_files:
        return None

    latest = max(checkpoint_files, key=lambda x: os.path.getmtime(x))
    return latest


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

    def _resize_keep_aspect(self, img, target_size):
        w, h = img.size
        if w <= h:
            new_w = target_size
            new_h = max(int(h * target_size / w), target_size)
        else:
            new_h = target_size
            new_w = max(int(w * target_size / h), target_size)
        return transforms.functional.resize(img, (new_h, new_w))

    def _transform_img(self, img):
        if config.keep_aspect_ratio:
            if self.aug:
                img = self._resize_keep_aspect(img, config.resize_size)
                i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                    img, (self.input_size[0], self.input_size[1]))
                img = transforms.functional.crop(img, i, j, h_crop, w_crop)
                img = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)(img)
                img = transforms.RandomGrayscale(0.1)(img)
                # 添加透视变换
                if random.random() < 0.3:
                    img = transforms.RandomPerspective(distortion_scale=0.2, fill=127)(img)
            else:
                img = self._resize_keep_aspect(img, self.input_size[0])
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
                # 添加透视变换
                if random.random() < 0.3:
                    img = transforms.RandomPerspective(distortion_scale=0.2, fill=127)(img)
            else:
                img = PadToSquare(fill=(127, 127, 127))(img)
                img = transforms.functional.resize(img, (self.input_size[0], self.input_size[1]))
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img,
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


class CTCModel(nn.Module):
    def __init__(self, num_classes=11):
        super(CTCModel, self).__init__()
        backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, replace_stride_with_dilation=[False, False, True])
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # 改进：增加layer1特征融合，使用bilinear上采样
        self.l1_reduce = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.l2_reduce = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.l3_reduce = nn.Sequential(
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.l4_reduce = nn.Sequential(
            nn.Conv2d(2048, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Smooth卷积
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.smooth_p1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 融合P1/P2/P3/P4: 64+128+128+128 = 448
        self.fuse = nn.Sequential(
            nn.Conv2d(448, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm = nn.LSTM(256, 256, num_layers=2, bidirectional=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        self.use_checkpoint = True

    def _forward_backbone(self, x):
        """
        改进：返回c1/c2/c3以支持更精细的多尺度融合
        """
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        return c1, c2, c3

    def forward(self, x):
        if self.training and self.use_checkpoint:
            c1, c2, c3 = t.utils.checkpoint.checkpoint(self._forward_backbone, x, use_reentrant=False)
        else:
            c1, c2, c3 = self._forward_backbone(x)
        c4 = self.layer4(c3)
        
        # Top-down pathway with bilinear interpolation
        p4 = self.l4_reduce(c4)
        p3 = self.l3_reduce(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.l2_reduce(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.smooth_p2(p2)
        
        # 新增：融合P1层特征
        p1 = self.l1_reduce(c1) + F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.smooth_p1(p1)
        
        # 上采样所有特征到P1的分辨率进行融合
        p2_up = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)
        
        feat = self.fuse(t.cat([p1, p2_up, p3_up, p4_up], dim=1))
        feat = self.height_pool(feat)
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C, W).permute(2, 0, 1)
        feat, _ = self.lstm(feat)
        logit = self.fc(feat)
        return F.log_softmax(logit, dim=2)


class CTCTrainer:
    def __init__(self, val=True):
        self.device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        print(f'CTC Model - Using device: {self.device}')
        self.use_amp = self.device.type == 'cuda'
        self.train_set = CTCDataset(mode='train', aug=True,
                                    input_size=(config.input_height, config.input_width))
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=True,
                                       persistent_workers=NUM_WORKERS > 0,
                                       drop_last=True, prefetch_factor=2,
                                       collate_fn=ctc_collate_fn)
        if val:
            self.val_set = CTCDataset(mode='val', aug=False,
                                      input_size=(config.input_height, config.input_width))
            self.val_loader = DataLoader(self.val_set, batch_size=config.batch_size,
                                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                                         persistent_workers=NUM_WORKERS > 0,
                                         prefetch_factor=2, collate_fn=ctc_collate_fn)
        else:
            self.val_loader = None

        self.model = CTCModel(num_classes=config.class_num).to(self.device)
        self.ema = ModelEMA(self.model, decay=config.ema_decay)
        self.criterion = nn.CTCLoss(blank=10, zero_infinity=True)

        backbone_prefixes = ['stem', 'layer1', 'layer2', 'layer3', 'layer4']
        backbone_params = [p for n, p in self.model.named_parameters()
                           if any(n.startswith(px) for px in backbone_prefixes)]
        other_params = [p for n, p in self.model.named_parameters()
                        if not any(n.startswith(px) for px in backbone_prefixes)]
        self.optimizer = SGD([
            {'params': backbone_params, 'lr': config.lr * config.backbone_lr_factor},
            {'params': other_params, 'lr': config.lr},
        ], momentum=config.momentum, weight_decay=config.weights_decay, nesterov=True)

        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01,
                                    total_iters=config.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epoches - config.warmup_epochs,
                                             eta_min=1e-6)
        self.lr_scheduler = SequentialLR(self.optimizer,
                                         schedulers=[warmup_scheduler, cosine_scheduler],
                                         milestones=[config.warmup_epochs])

        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.best_acc = 0
        self.best_checkpoint_path = ''
        self.train_log = []
        # Early stopping 机制
        self.patience_counter = 0
        self.early_stop_triggered = False
        if config.pretrained is not None:
            self.load_model(config.pretrained, save_opt=False)

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            # 检查是否触发early stopping
            if self.early_stop_triggered:
                print(f'\n🛑 Early stopping triggered at epoch {epoch + 1}')
                print(f'   Best val acc: {self.best_acc * 100:.2f}%')
                print(f'   Best checkpoint: {self.best_checkpoint_path}')
                break
            
            train_acc = self.train_epoch(epoch)
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1} LR: {current_lr:.6f}')

            if (epoch + 1) % config.eval_interval == 0:
                print('Start CTC Evaluation')
                acc = 0.0
                if self.val_loader is not None:
                    acc = self.eval()
                self.train_log.append({
                    'epoch': epoch + 1,
                    'train_acc': train_acc,
                    'val_acc': acc * 100,
                    'lr': current_lr
                })
                if acc > self.best_acc:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    save_path = os.path.join(config.checkpoints,
                                             'epoch-ctc-%d-acc-%.2f.pth' % (epoch + 1, acc * 100))
                    self.save_model(save_path, save_opt=True)
                    print('%s saved successfully...' % save_path)
                    self.best_acc = acc
                    self.best_checkpoint_path = save_path
                    # 重置patience计数器
                    self.patience_counter = 0
                    print(f'✅ New best model! Patience counter reset to 0')
                else:
                    # 验证集准确率没有提升，增加patience计数
                    self.patience_counter += 1
                    print(f'⚠️  No improvement. Patience: {self.patience_counter}/{config.early_stopping_patience}')
                    
                    # 检查是否达到patience限制
                    if self.patience_counter >= config.early_stopping_patience:
                        self.early_stop_triggered = True
                        print(f'\n🛑 Early stopping will be triggered after this epoch')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        corrects = 0
        total = 0
        tbar = tqdm(self.train_loader)

        for i, (img, label_concat, lengths) in enumerate(tbar):
            img = img.to(self.device)
            label_concat = label_concat.to(self.device)
            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                log_probs = self.model(img)
                T = log_probs.size(0)
                B = log_probs.size(1)
                input_lengths = t.full((B,), T, dtype=t.long)
                target_lengths = lengths.to(self.device)
                loss = self.criterion(log_probs, label_concat, input_lengths, target_lengths)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)
            total_loss += loss.item()

            if i % config.train_eval_interval == 0:
                pred_strs = ctc_greedy_decode(log_probs.detach())
                offset = 0
                for b in range(B):
                    gt = label_concat[offset:offset + lengths[b]].tolist()
                    offset += lengths[b]
                    if pred_strs[b] == gt:
                        corrects += 1
                    total += 1

            tbar.set_description(
                'Epoch %d, loss: %.3f, acc: %.3f' % (epoch + 1, total_loss / (i + 1), corrects * 100 / max(total, 1)))

        return corrects * 100 / max(total, 1)

    def eval(self):
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()
        corrects = 0
        total = 0
        with t.no_grad():
            tbar = tqdm(self.val_loader)
            for img, label_concat, lengths in tbar:
                img = img.to(self.device)
                log_probs = model(img)
                pred_strs = ctc_beam_decode(log_probs.cpu())
                offset = 0
                for b in range(len(lengths)):
                    gt = label_concat[offset:offset + lengths[b]].tolist()
                    offset += lengths[b]
                    if pred_strs[b] == gt:
                        corrects += 1
                    total += 1
                tbar.set_description('CTC Val Acc: %.2f' % (corrects * 100 / max(total, 1)))
                del img, log_probs
        t.cuda.empty_cache()
        self.model.train()
        return corrects / max(total, 1)

    def save_model(self, save_path, save_opt=False, save_config=False):
        dicts = {'model': self.ema.ema.state_dict() if self.ema else self.model.state_dict(),
                 'model_type': 'ctc'}
        if save_opt:
            dicts['opt'] = self.optimizer.state_dict()
            dicts['lr_scheduler'] = self.lr_scheduler.state_dict()
            dicts['scaler'] = self.scaler.state_dict()
        if save_config:
            dicts['config'] = {s: config.__getattribute__(s) for s in dir(config) if not s.startswith('_')}
        dicts['train_log'] = self.train_log
        dicts['best_acc'] = self.best_acc
        dicts['best_checkpoint_path'] = self.best_checkpoint_path
        t.save(dicts, save_path)

    def load_model(self, load_path, save_opt=False, save_config=False):
        dicts = t.load(load_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(dicts['model'])
        if self.ema is not None:
            self.ema.ema.load_state_dict(dicts['model'])
        if save_opt:
            self.optimizer.load_state_dict(dicts['opt'])
            if 'lr_scheduler' in dicts:
                self.lr_scheduler.load_state_dict(dicts['lr_scheduler'])
            if 'scaler' in dicts:
                self.scaler.load_state_dict(dicts['scaler'])
            if 'train_log' in dicts and len(dicts['train_log']) > 0:
                config.start_epoch = dicts['train_log'][-1]['epoch']
            if 'best_acc' in dicts:
                self.best_acc = dicts['best_acc']
            if 'best_checkpoint_path' in dicts:
                self.best_checkpoint_path = dicts['best_checkpoint_path']
        else:
            if 'best_acc' in dicts:
                self.best_acc = dicts['best_acc']
        if save_config:
            for k, v in dicts['config'].items():
                config.__setattr__(k, v)


def ctc_greedy_decode(log_probs):
    log_probs = log_probs.permute(1, 0, 2)
    results = []
    for b in range(log_probs.size(0)):
        pred = log_probs[b].argmax(dim=1).tolist()
        decoded = []
        prev = -1
        for p in pred:
            if p != prev and p != 10:
                decoded.append(p)
            prev = p
        results.append(decoded)
    return results


def ctc_beam_decode(log_probs, beam_width=10):
    log_probs = log_probs.permute(1, 0, 2)
    B, T, C = log_probs.shape
    blank = 10
    results = []
    for b in range(B):
        beams = [([], blank, 0.0)]
        for t in range(T):
            new_beams = {}
            topk_vals, topk_idx = log_probs[b, t].topk(min(beam_width, C))
            for k in range(topk_idx.size(0)):
                c = topk_idx[k].item()
                log_p = topk_vals[k].item()
                for seq, last, score in beams:
                    new_seq = seq.copy()
                    if c == blank:
                        pass
                    elif c == last:
                        pass
                    else:
                        new_seq.append(c)
                    key = (tuple(new_seq), c)
                    new_score = score + log_p
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score
            beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
            beams = [([x for x in k[0]], k[1], v) for k, v in beams]
        if beams:
            results.append(beams[0][0])
        else:
            results.append([])
    return results


def ctc_predict(model_path, csv_path, use_tta=False):
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    model = CTCModel(num_classes=config.class_num).to(device)
    model.load_state_dict(t.load(model_path, map_location=device, weights_only=False)['model'])
    print('Load CTC model from %s successfully' % model_path)
    model.eval()
    char_list = [str(i) for i in range(10)] + ['']

    test_loader = DataLoader(CTCDataset(mode='test', aug=False,
                                        input_size=(config.input_height, config.input_width)),
                             batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True, drop_last=False, persistent_workers=NUM_WORKERS > 0,
                             collate_fn=ctc_test_collate_fn)
    results = []
    with t.no_grad():
        for img, img_names in tqdm(test_loader, desc='CTC Predict'):
            img = img.to(device)
            log_probs = model(img)
            pred_strs = ctc_beam_decode(log_probs.cpu())
            for b in range(len(img_names)):
                code = ''.join(char_list[d] for d in pred_strs[b])
                results.append([img_names[b], code])

    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results


def cross_model_ensemble(multihead_path, ctc_path, csv_path, model_type=None):
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    mt = model_type or config.model_type
    mh_model = create_model(mt).to(device)
    mh_model.load_state_dict(t.load(multihead_path, map_location=device, weights_only=False)['model'])
    mh_model.eval()
    print(f'Loaded multihead model: {multihead_path}')

    ctc_model = CTCModel(num_classes=config.class_num).to(device)
    ctc_model.load_state_dict(t.load(ctc_path, map_location=device, weights_only=False)['model'])
    ctc_model.eval()
    print(f'Loaded CTC model: {ctc_path}')

    test_set = CTCDataset(mode='test', aug=False,
                          input_size=(config.input_height, config.input_width))
    test_loader = DataLoader(test_set, batch_size=config.batch_size,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                             drop_last=False, persistent_workers=False,
                             collate_fn=ctc_test_collate_fn)

    char_list = [str(i) for i in range(10)] + ['']
    results = []

    with t.no_grad():
        for img, img_names in tqdm(test_loader, desc='Cross-model Ensemble'):
            img = img.to(device)
            bs = img.size(0)

            mh_probs = mh_model.forward_with_probs(img)
            mh_preds = [p.argmax(1) for p in mh_probs]
            mh_confidences = [p.max(1)[0] for p in mh_probs]

            ctc_log_probs = ctc_model(img)
            ctc_preds = ctc_beam_decode(ctc_log_probs.cpu())
            ctc_probs = ctc_log_probs.exp().cpu()
            ctc_confidences = []
            for b in range(bs):
                conf = []
                prev = -1
                for t_step in range(ctc_probs.size(0)):
                    p = ctc_probs[t_step, b].argmax().item()
                    if p != prev and p != 10:
                        conf.append(ctc_probs[t_step, b, p].item())
                    prev = p
                ctc_confidences.append(conf)

            for b in range(bs):
                mh_code = ''.join(char_list[mh_preds[h][b].item()] for h in range(config.num_heads))
                ctc_code = ''.join(char_list[d] for d in ctc_preds[b])

                if mh_code == ctc_code:
                    final_code = mh_code
                else:
                    mh_digits = []
                    for h in range(config.num_heads):
                        d = mh_preds[h][b].item()
                        if d == 10:
                            break
                        mh_digits.append((str(d), mh_confidences[h][b].item()))

                    ctc_digits = [(char_list[d], ctc_confidences[b][i] if i < len(ctc_confidences[b]) else 0.0)
                                  for i, d in enumerate(ctc_preds[b])]

                    max_len = max(len(mh_digits), len(ctc_digits))
                    final_chars = []
                    for pos in range(max_len):
                        mh_c, mh_conf = mh_digits[pos] if pos < len(mh_digits) else ('', 0.0)
                        ctc_c, ctc_conf = ctc_digits[pos] if pos < len(ctc_digits) else ('', 0.0)
                        if mh_c == ctc_c:
                            final_chars.append(mh_c)
                        elif mh_conf >= ctc_conf:
                            final_chars.append(mh_c)
                        else:
                            final_chars.append(ctc_c)
                    final_code = ''.join(final_chars)

                results.append([img_names[b], final_code])

    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results


if __name__ == '__main__':
    latest_checkpoint = find_latest_checkpoint(config.checkpoints)
    if latest_checkpoint:
        print(f'Found latest checkpoint: {latest_checkpoint}')
        config.pretrained = latest_checkpoint
    else:
        print('No checkpoint found, starting from scratch.')

    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else config.model_type

    if model_type == 'ctc':
        print('\n========== Training CTC Model ==========')
        ctc_trainer = CTCTrainer()
        ctc_trainer.train()

        print('\n=== CTC Evaluation ===')
        ctc_acc = ctc_trainer.eval()
        print(f'CTC Best Val Acc: {ctc_trainer.best_acc * 100:.2f}')

        ctc_predict(ctc_trainer.best_checkpoint_path,
                    os.path.join(BASE_DIR, 'result_ctc.csv'), use_tta=False)
    elif model_type == 'transformer':
        print('\n========== Training Transformer Model ==========')
        trainer = Trainer(model_type='transformer')
        trainer.train()

        print('\n=== Standard Evaluation ===')
        val_acc = trainer.eval()
        print(f'Best Val Acc: {trainer.best_acc * 100:.2f}')

        print('\n=== Detailed Evaluation ===')
        trainer.eval_detailed()

        print('\n=== TTA Evaluation ===')
        tta_acc = trainer.eval_tta()
        if tta_acc > trainer.best_acc:
            print(f'TTA improved acc: {tta_acc * 100:.2f} > {trainer.best_acc * 100:.2f}')

        predicts(trainer.best_checkpoint_path, os.path.join(BASE_DIR, 'result.csv'), use_tta=True, model_type='transformer')
    else:
        print('\n========== Training FPN Multi-Head Model ==========')
        trainer = Trainer(model_type='fpn_multihead')
        trainer.train()

        print('\n=== Standard Evaluation ===')
        val_acc = trainer.eval()
        print(f'Best Val Acc: {trainer.best_acc * 100:.2f}')

        print('\n=== Detailed Evaluation ===')
        trainer.eval_detailed()

        print('\n=== TTA Evaluation ===')
        tta_acc = trainer.eval_tta()
        if tta_acc > trainer.best_acc:
            print(f'TTA improved acc: {tta_acc * 100:.2f} > {trainer.best_acc * 100:.2f}')

        predicts(trainer.best_checkpoint_path, os.path.join(BASE_DIR, 'result.csv'), use_tta=True, model_type='fpn_multihead')
