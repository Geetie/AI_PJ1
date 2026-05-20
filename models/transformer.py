import torch as t
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import FPNBackbone
from config import config


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
        self.length_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(64, num_heads + 1),
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
        length_logits = self.length_head(feat)
        memory, H, W = self._prepare_memory(feat)
        B = feat.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, memory, tgt_mask=None)
        cls_outs = tuple(self.cls_head(decoded[:, i, :]) for i in range(self.num_heads))
        bbox_outs = tuple(self.bbox_head(decoded[:, i, :]) for i in range(self.num_heads))
        return cls_outs, bbox_outs, length_logits

    def forward_with_attn(self, img, gt_bboxes=None):
        feat = self.backbone(img)
        length_logits = self.length_head(feat)
        memory, H, W = self._prepare_memory(feat)
        B = feat.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        decoded, attn_maps = self._decode_with_attn(queries, memory, tgt_mask=None, H_feat=H, W_feat=W)
        cls_outs = tuple(self.cls_head(decoded[:, i, :]) for i in range(self.num_heads))
        bbox_outs = tuple(self.bbox_head(decoded[:, i, :]) for i in range(self.num_heads))
        return cls_outs, bbox_outs, attn_maps if attn_maps else None, (), length_logits

    def forward_with_probs(self, img):
        cls_outs, _, length_logits = self.forward(img)
        probs = tuple(F.softmax(c, dim=1) for c in cls_outs)
        pred_length = length_logits.argmax(dim=1)
        for h in range(self.num_heads):
            mask = (pred_length <= h).unsqueeze(1).expand_as(probs[h])
            probs[h] = probs[h].masked_fill(mask, 0.0)
            probs[h][:, 10] = probs[h][:, 10].masked_fill(mask[:, 10], 1.0)
        return probs
