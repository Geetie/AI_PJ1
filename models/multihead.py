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
        self.pos_embed = nn.Parameter(t.randn(1, config.pos_embed_channels, S, S) * 0.02)
        self.head_embed = nn.Parameter(t.randn(1, config.pos_embed_channels, 1, 1) * 0.02)
        self.num_attn_channels = config.num_attn_channels
        self.attn_temperature = config.soft_attn_temperature
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
            soft_weights = F.softmax(peak_conf / self.attn_temperature, dim=1)
            attn_weights = (soft_weights.unsqueeze(-1).unsqueeze(-1) * attn_per_ch).sum(dim=1, keepdim=True)
        weighted_feat = x * attn_weights
        pooled = self.attn_pool(weighted_feat).flatten(1)
        hidden = self.feat_proj(pooled)
        cls_out = self.cls_layer(hidden)
        bbox_feat = self.bbox_pool(weighted_feat).flatten(1)
        bbox_out = self.bbox_head(bbox_feat)
        if return_attn:
            return cls_out, bbox_out, hidden, attn_weights
        return cls_out, bbox_out, hidden


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


class HeadInteractionLayer(nn.Module):
    def __init__(self, feat_dim, num_heads, num_layers=2, nhead=4, dropout=None):
        super().__init__()
        if dropout is None:
            dropout = config.dropout
        self.num_heads = num_heads
        self.feat_dim = feat_dim
        pos_encoding = self._create_sinusoidal_encoding(num_heads, feat_dim)
        self.register_buffer('sinusoidal_pos', pos_encoding)
        self.learnable_pos = nn.Parameter(t.randn(1, num_heads, feat_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=nhead,
            dim_feedforward=feat_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _create_sinusoidal_encoding(self, num_heads, feat_dim):
        pos = t.arange(num_heads).float().unsqueeze(1)
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
        stacked = t.stack(head_features, dim=1)
        pos_embed = self.sinusoidal_pos + self.learnable_pos
        stacked = stacked + pos_embed.expand(B, -1, -1)
        encoded = self.encoder(stacked)
        return [encoded[:, i, :] for i in range(self.num_heads)]


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
        self.length_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.multiscale_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(64, num_heads + 1),
        )

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
            roi_processed = t.utils.checkpoint.checkpoint(self.roi_cnn[head_idx], roi_feat, use_reentrant=False)
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
                p_no_digit = F.softmax(cls_outs[h].detach(), dim=1)[:, 10:11]
                gated_roi = roi_cls[h] * (1 - p_no_digit)
                refined.append(cls_outs[h] + gated_roi)
            return tuple(refined)
        return tuple(cls_outs[h] + roi_cls[h] for h in range(self.num_heads))

    def set_roi_gt_prob(self, prob):
        self.roi_gt_prob = prob

    def forward(self, img, gt_bboxes=None):
        feat = self.backbone(img)
        length_logits = self.length_head(feat)
        feat = self.pre_head_comm(feat, [h.pos_embed for h in self.heads])
        results = []
        use_ckpt = self.training and config.use_gradient_checkpoint and not config.use_torch_compile
        for head in self.heads:
            if use_ckpt:
                results.append(t.utils.checkpoint.checkpoint(head, feat, False, use_reentrant=False))
            else:
                results.append(head(feat))
        bbox_outs = tuple(r[1] for r in results)
        head_feats = [r[2] for r in results]
        interacted = self.head_interaction(head_feats)
        cls_outs = tuple(self.head_fc[h](interacted[h]) for h in range(self.num_heads))
        cls_outs = self._apply_roi_refine(feat, cls_outs, bbox_outs, gt_bboxes)
        return cls_outs, bbox_outs, length_logits

    def forward_with_attn(self, img, gt_bboxes=None):
        feat = self.backbone(img)
        length_logits = self.length_head(feat)
        feat = self.pre_head_comm(feat, [h.pos_embed for h in self.heads])
        head_cls_outs, bbox_outs, attn_maps = [], [], []
        head_feats = []
        use_ckpt = self.training and config.use_gradient_checkpoint and not config.use_torch_compile
        for head in self.heads:
            if use_ckpt:
                cls_out, bbox_out, hidden, attn = t.utils.checkpoint.checkpoint(
                    lambda h, f: h(f, return_attn=True), 
                    head, feat, use_reentrant=False
                )
            else:
                cls_out, bbox_out, hidden, attn = head(feat, return_attn=True)
            head_cls_outs.append(cls_out)
            bbox_outs.append(bbox_out)
            head_feats.append(hidden)
            attn_maps.append(attn)
        bbox_tuple = tuple(bbox_outs)
        interacted = self.head_interaction(head_feats)
        cls_list = tuple(self.head_fc[h](interacted[h]) for h in range(self.num_heads))
        cls_list = self._apply_roi_refine(feat, cls_list, bbox_tuple, gt_bboxes)
        return cls_list, bbox_tuple, attn_maps, tuple(head_cls_outs), length_logits

    def forward_with_probs(self, img):
        feat = self.backbone(img)
        length_logits = self.length_head(feat)
        feat = self.pre_head_comm(feat, [h.pos_embed for h in self.heads])
        results = [head(feat) for head in self.heads]
        bbox_outs = tuple(r[1] for r in results)
        head_feats = [r[2] for r in results]
        interacted = self.head_interaction(head_feats)
        cls_outs = tuple(self.head_fc[h](interacted[h]) for h in range(self.num_heads))
        cls_outs = self._apply_roi_refine(feat, cls_outs, bbox_outs)
        probs = tuple(F.softmax(c, dim=1) for c in cls_outs)
        pred_length = length_logits.argmax(dim=1)
        for h in range(self.num_heads):
            mask = (pred_length <= h).unsqueeze(1).expand_as(probs[h])
            probs[h] = probs[h].masked_fill(mask, 0.0)
            probs[h][:, 10] = probs[h][:, 10].masked_fill(mask[:, 10], 1.0)
        return probs
