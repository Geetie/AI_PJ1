import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101, ResNet101_Weights
from config import config


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
        self.fuse = nn.Sequential(
            nn.Conv2d(896, config.multiscale_feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.multiscale_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(config.multiscale_feat_dim)
        self.use_checkpoint = True

    def _forward_early(self, x):
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
        p4 = self.l4_reduce(c4)
        p3 = self.l3_reduce(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.smooth_p3(p3)
        p2 = self.l2_reduce(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.smooth_p2(p2)
        p1 = self.l1_reduce(c1) + F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.smooth_p1(p1)
        p2_up = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)
        return self.se(self.fuse(t.cat([p1, p2_up, p3_up, p4_up], dim=1)))
