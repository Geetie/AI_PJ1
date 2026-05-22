"""Backbone模块 - FPN特征提取器"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101, ResNet101_Weights

# 延迟导入config避免循环依赖
_config = None


def _get_config():
    """懒加载配置"""
    global _config
    if _config is None:
        from config import config as cfg
        _config = cfg
    return _config


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.fc(self.pool(x).view(B, C))
        return x * w.view(B, C, 1, 1)


class FPNBackbone(nn.Module):
    """
    FPN (Feature Pyramid Network) Backbone
    基于ResNet101的多尺度特征融合网络
    """
    def __init__(self):
        super().__init__()
        config = _get_config()
        p1_ch = config.backbone_p1_channels
        
        backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, 
                            replace_stride_with_dilation=[False, False, True])
        self._replace_relu_with_leaky(backbone)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        self.l1_reduce = nn.Sequential(
            nn.Conv2d(256, p1_ch, 1, bias=False),
            nn.BatchNorm2d(p1_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.l2_reduce = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.l2_to_p1 = nn.Sequential(
            nn.Conv2d(256, p1_ch, 1, bias=False),
            nn.BatchNorm2d(p1_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.l3_reduce = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.l4_reduce = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
        )
        
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.smooth_p1 = nn.Sequential(
            nn.Conv2d(p1_ch, p1_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(p1_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        
        fuse_in = p1_ch + 256 + 256 + 256
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, config.multiscale_feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.multiscale_feat_dim),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.se = SEBlock(config.multiscale_feat_dim)
        self.use_checkpoint = True
        self._init_weights()
        self._reset_batch_norm_stats()

    def _replace_relu_with_leaky(self, module, negative_slope=0.01):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                inplace = child.inplace
                setattr(module, name, nn.LeakyReLU(negative_slope, inplace=inplace))
            else:
                self._replace_relu_with_leaky(child, negative_slope)

    def _reset_batch_norm_stats(self):
        fpn_prefixes = (
            'l1_reduce', 'l2_reduce', 'l2_to_p1', 'l3_reduce', 'l4_reduce',
            'smooth_p3', 'smooth_p2', 'smooth_p1', 'fuse',
        )
        for name, m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                is_fpn = any(name.startswith(prefix) for prefix in fpn_prefixes)
                if is_fpn:
                    m.running_mean.fill_(0)
                    m.running_var.fill_(1)
                    m.num_batches_tracked.zero_()

    def _init_weights(self):
        """初始化FPN相关层的权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

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
        config = _get_config()
        use_ckpt = self.training and self.use_checkpoint
        use_ckpt = use_ckpt and not config.use_torch_compile
        if use_ckpt and config.use_bf16 and not config.gradient_checkpoint_with_bf16:
            use_ckpt = False
        
        if use_ckpt:
            c1, c2, c3 = t.utils.checkpoint.checkpoint(self._forward_early, x, use_reentrant=False)
        else:
            c1, c2, c3 = self._forward_early(x)
        c4 = self.layer4(c3)
        
        p4 = self.l4_reduce(c4)
        p3 = self.l3_reduce(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.smooth_p3(p3)
        p2 = self.l2_reduce(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.smooth_p2(p2)
        p1 = self.l1_reduce(c1) + self.l2_to_p1(F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False))
        p1 = self.smooth_p1(p1)
        
        p2_up = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)
        
        fused = t.cat([p1, p2_up, p3_up, p4_up], dim=1)
        fused = self.fuse(fused)
        return self.se(fused)
