import torch as t
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import FPNBackbone
from config import config

try:
    from torchvision.ops import roi_align
except ImportError:
    roi_align = None


class PositionAwareAttentionHead(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, head_idx):
        super().__init__()
        self.head_idx = head_idx
        S = config.feat_spatial_size
        self.pos_embed = nn.Parameter(t.randn(1, config.pos_embed_channels, S, S) * 0.1)
        self.head_embed = nn.Parameter(t.randn(1, config.pos_embed_channels, 1, 1) * 0.1)
        self.num_attn_channels = config.num_attn_channels
        self.attn_temperature = config.soft_attn_temperature
        self.norm_input = nn.BatchNorm2d(in_channels + config.pos_embed_channels * 2)
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels + config.pos_embed_channels * 2, in_channels + config.pos_embed_channels * 2, 3,
                      padding=1, groups=in_channels + config.pos_embed_channels * 2, bias=False),
            nn.BatchNorm2d(in_channels + config.pos_embed_channels * 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(in_channels + config.pos_embed_channels * 2, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(128, self.num_attn_channels, 1),
        )
        self.attn_pool = nn.AdaptiveAvgPool2d(4)
        pool_dim = in_channels * 4 * 4
        self.feat_proj = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(config.dropout),
        )
        self.cls_layer = nn.Linear(hidden_dim, num_classes)
        self.bbox_pool = nn.AdaptiveAvgPool2d(4)
        self.bbox_head = nn.Sequential(
            nn.Linear(in_channels * 4 * 4, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden_dim // 2, 4),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == m.in_channels and m.groups > 1:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif m.groups > 1:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_attn=False):
        B, C, H, W = x.shape
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        pos = pos.expand(B, -1, -1, -1)
        head = self.head_embed.expand(B, -1, H, W)
        x_input = t.cat([x, pos, head], dim=1)
        x_input = self.norm_input(x_input)
        attn_raw = self.attention_conv(x_input)
        
        if self.num_attn_channels == 1:
            attn_weights = F.softmax(attn_raw.view(B, -1), dim=1).view(B, 1, H, W)
        else:
            attn_per_ch = F.softmax(attn_raw.view(B, self.num_attn_channels, -1), dim=2)
            attn_per_ch = attn_per_ch.view(B, self.num_attn_channels, H, W)
            peak_conf = attn_per_ch.amax(dim=(2, 3))
            soft_weights = F.softmax(peak_conf.float() / self.attn_temperature, dim=1).to(attn_per_ch.dtype)
            attn_weights = (soft_weights.unsqueeze(-1).unsqueeze(-1) * attn_per_ch).sum(dim=1, keepdim=True)
        
        # 使用除法归一化，attn_weights已经经过softmax，不需要再次softmax
        attn_weights_safe = attn_weights + 1e-8
        attn_weights_norm = attn_weights_safe / (attn_weights_safe.sum(dim=(2, 3), keepdim=True))
        
        weighted_feat = x * attn_weights_norm
        pooled = self.attn_pool(weighted_feat).flatten(1)
        hidden = self.feat_proj(pooled)
        cls_out = self.cls_layer(hidden)
        bbox_feat = self.bbox_pool(weighted_feat).flatten(1)
        bbox_out = self.bbox_head(bbox_feat)
        if return_attn:
            return cls_out, bbox_out, hidden, attn_weights_norm
        return cls_out, bbox_out, hidden


class CrossHeadCommLayer(nn.Module):
    def __init__(self, feat_dim, num_heads, pos_channels):
        super().__init__()
        self.num_heads = num_heads
        self.pos_proj = nn.Sequential(
            nn.Conv2d(pos_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.comm_conv = nn.Sequential(
            nn.Conv2d(feat_dim + 32 * num_heads, feat_dim, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat, pos_embeds):
        B, C, H, W = feat.shape
        all_pos = []
        for pe in pos_embeds:
            p = F.interpolate(pe, size=(H, W), mode='bilinear', align_corners=False)
            p = self.pos_proj(p.expand(B, -1, -1, -1))
            all_pos.append(p)
        x = t.cat([feat] + all_pos, dim=1)
        return feat + self.comm_conv(x)


class HeadInteractionLayer(nn.Module):
    def __init__(self, feat_dim, num_heads, num_layers=2, nhead=4, dropout=None):
        super().__init__()
        if dropout is None:
            dropout = config.dropout
        self.num_heads = num_heads
        self.feat_dim = feat_dim
        self.pos_scale = 1.0
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(dropout)
        )
        
        pos_encoding = self._create_sinusoidal_encoding(num_heads, feat_dim)
        self.register_buffer('sinusoidal_pos', pos_encoding)
        self.learnable_pos = nn.Parameter(t.randn(1, num_heads, feat_dim) * 0.02)

    def _create_sinusoidal_encoding(self, num_heads, feat_dim):
        pos = t.arange(num_heads).float().unsqueeze(1)
        div_term = t.exp(t.arange(0, feat_dim, 2).float() * (-t.log(t.tensor(10000.0)) / feat_dim))
        encoding = t.zeros(1, num_heads, feat_dim)
        encoding[0, :, 0::2] = t.sin(pos * div_term)
        if feat_dim % 2 == 0:
            encoding[0, :, 1::2] = t.cos(pos * div_term)
        else:
            encoding[0, :, 1::2] = t.cos(pos * div_term[:-1])
        return encoding / (feat_dim ** 0.5)

    def forward(self, head_features):
        B = head_features[0].shape[0]
        x = t.stack(head_features, dim=1)
        pos_embed = self.sinusoidal_pos + self.learnable_pos
        x = x + pos_embed.expand(B, -1, -1)
        
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return [x[:, i, :] for i in range(self.num_heads)]


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
            self.roi_cnn = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(config.multiscale_feat_dim, config.roi_feat_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(config.roi_feat_dim),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Conv2d(config.roi_feat_dim, config.roi_feat_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(config.roi_feat_dim),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(config.roi_feat_dim, config.roi_feat_dim),
                    nn.BatchNorm1d(config.roi_feat_dim),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Dropout(config.dropout),
                ) for _ in range(num_heads)
            ])
            self.roi_cls_heads = nn.ModuleList([
                nn.Linear(config.roi_feat_dim, class_num) for _ in range(num_heads)
            ])
        self._init_extra_weights()
        self.head_interaction = HeadInteractionLayer(
            config.fc_hidden, num_heads,
            num_layers=config.head_interaction_layers,
            nhead=min(config.transformer_heads, num_heads),
            dropout=config.dropout
        )
        self.head_fc = nn.ModuleList([
            nn.Linear(config.fc_hidden, class_num) for _ in range(num_heads)
        ])
        self.length_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.multiscale_feat_dim, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(64, num_heads + 1),
        )
        self._bn_protection = None

    def _init_extra_weights(self):
        """额外的权重初始化，确保关键层的梯度流动"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == config.class_num:
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif 'roi_cls' in str(m) or 'length_head' in str(m):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        self._init_roi_weights()

    def _init_roi_weights(self):
        """专门优化ROI分支的权重初始化，确保梯度流动"""
        if not self.has_roi:
            return
        
        for i in range(self.num_heads):
            for m in self.roi_cnn[i].modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    @t.compiler.disable
    def _extract_roi_feat(self, feat, bbox_pred, head_idx):
        B, C, H, W = feat.shape
        cx, cy, bw, bh = bbox_pred[:, 0], bbox_pred[:, 1], bbox_pred[:, 2], bbox_pred[:, 3]
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H
        x1 = x1.clamp(min=0, max=W-1)
        y1 = y1.clamp(min=0, max=H-1)
        x2 = x2.clamp(min=0, max=W-1)
        y2 = y2.clamp(min=0, max=H-1)
        batch_idx = t.arange(B, device=feat.device).float()
        boxes = t.stack([batch_idx, x1, y1, x2, y2], dim=1)
        try:
            roi_feat = roi_align(feat, boxes, output_size=7, spatial_scale=1.0)
        except RuntimeError:
            self.has_roi = False
            return t.zeros(B, config.class_num, device=feat.device)
        if self.training and config.use_gradient_checkpoint and not config.use_torch_compile:
            roi_processed = t.utils.checkpoint.checkpoint(self.roi_cnn[head_idx], roi_feat, use_reentrant=True)
        else:
            roi_processed = self.roi_cnn[head_idx](roi_feat)
        return self.roi_cls_heads[head_idx](roi_processed)

    @t.compiler.disable
    def _apply_roi_refine(self, feat, cls_outs, bbox_outs, gt_bboxes=None):
        if not self.has_roi:
            return cls_outs
        use_gt = False
        if self.training and gt_bboxes is not None and config.roi_teacher_forcing:
            use_gt = t.rand(1).item() < self.roi_gt_prob
        roi_cls = tuple(
            self._extract_roi_feat(feat, gt_bboxes[:, h, :] if use_gt else bbox_outs[h], h)
            for h in range(self.num_heads)
        )
        if not self.training:
            refined = []
            for h in range(self.num_heads):
                p_no_digit = F.softmax(cls_outs[h].detach().float(), dim=1)[:, 10:11].to(cls_outs[h].dtype)
                gated_roi = roi_cls[h] * (1 - p_no_digit)
                refined.append(cls_outs[h] + gated_roi)
            return tuple(refined)
        return tuple(cls_outs[h] + roi_cls[h] for h in range(self.num_heads))

    def set_roi_gt_prob(self, prob):
        self.roi_gt_prob = prob

    def setup_bn_protection(self, warmup_epochs=10, freeze_epochs=0,
                            bn_grad_max_norm=1.0, auto_fix=True):
        from utils.bn_protection import install_bn_protection
        self._bn_protection = install_bn_protection(
            self, warmup_epochs=warmup_epochs, freeze_epochs=freeze_epochs,
            bn_grad_max_norm=bn_grad_max_norm, auto_fix=auto_fix)
        return self._bn_protection

    def forward(self, img, gt_bboxes=None):
        feat = self.backbone(img)
        length_logits = self.length_head(feat)
        feat = self.pre_head_comm(feat, [h.pos_embed for h in self.heads])
        results = []
        use_ckpt = self.training and config.use_gradient_checkpoint and not config.use_torch_compile
        if config.use_bf16 and not config.gradient_checkpoint_with_bf16:
            use_ckpt = False
        if use_ckpt:
            for head in self.heads:
                def forward_fn(x, return_attn=False):
                    return head(x, return_attn=return_attn)
                results.append(t.utils.checkpoint.checkpoint(forward_fn, feat, False, use_reentrant=True))
        else:
            for head in self.heads:
                results.append(head(feat))
        head_cls_outs = tuple(r[0] for r in results)
        bbox_outs = tuple(r[1] for r in results)
        head_feats = [r[2] for r in results]
        interacted = self.head_interaction(head_feats)
        fc_outs = tuple(self.head_fc[h](interacted[h]) for h in range(self.num_heads))
        cls_outs = tuple(fc_outs[h] + head_cls_outs[h] for h in range(self.num_heads))
        cls_outs = self._apply_roi_refine(feat, cls_outs, bbox_outs, gt_bboxes)
        return cls_outs, bbox_outs, length_logits

    def forward_with_attn(self, img, gt_bboxes=None):
        feat = self.backbone(img)
        length_logits = self.length_head(feat)
        feat = self.pre_head_comm(feat, [h.pos_embed for h in self.heads])
        head_cls_outs, bbox_outs, attn_maps = [], [], []
        head_feats = []
        for head in self.heads:
            cls_out, bbox_out, hidden, attn = head(feat, return_attn=True)
            head_cls_outs.append(cls_out)
            bbox_outs.append(bbox_out)
            head_feats.append(hidden)
            attn_maps.append(attn)
        bbox_tuple = tuple(bbox_outs)
        interacted = self.head_interaction(head_feats)
        fc_outs = tuple(self.head_fc[h](interacted[h]) for h in range(self.num_heads))
        cls_list = tuple(fc_outs[h] + head_cls_outs[h] for h in range(self.num_heads))
        cls_list = self._apply_roi_refine(feat, cls_list, bbox_tuple, gt_bboxes)
        return cls_list, bbox_tuple, attn_maps, tuple(head_cls_outs), length_logits

    def forward_with_probs(self, img):
        feat = self.backbone(img)
        length_logits = self.length_head(feat)
        feat = self.pre_head_comm(feat, [h.pos_embed for h in self.heads])
        results = [head(feat) for head in self.heads]
        head_cls_outs = tuple(r[0] for r in results)
        bbox_outs = tuple(r[1] for r in results)
        head_feats = [r[2] for r in results]
        interacted = self.head_interaction(head_feats)
        fc_outs = tuple(self.head_fc[h](interacted[h]) for h in range(self.num_heads))
        cls_outs = tuple(fc_outs[h] + head_cls_outs[h] for h in range(self.num_heads))
        cls_outs = self._apply_roi_refine(feat, cls_outs, bbox_outs)
        probs = [F.softmax(c.float(), dim=1).to(c.dtype) for c in cls_outs]
        pred_length = length_logits.argmax(dim=1)
        for h in range(self.num_heads):
            mask = (pred_length <= h).unsqueeze(1).expand_as(probs[h])
            probs[h] = probs[h].masked_fill(mask, 0.0)
            probs[h][:, 10] = probs[h][:, 10].masked_fill(mask[:, 10], 1.0)
        return tuple(probs)
