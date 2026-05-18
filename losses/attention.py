import torch as t
import torch.nn as nn
import torch.nn.functional as F


def compute_single_gaussian_kl(attn_single, cx, cy, bw, bh, h, w):
    """高效计算单张 attention map 与单张高斯 target 的 KL 散度，避免大张量"""
    grid_y = t.arange(h, device=attn_single.device, dtype=t.float32).view(1, h, 1) / h
    grid_x = t.arange(w, device=attn_single.device, dtype=t.float32).view(1, 1, w) / w
    sigma_x = bw / 2.5
    sigma_y = bh / 2.5
    gauss = t.exp(-((grid_x - cx.view(-1, 1, 1)) ** 2 / (2 * sigma_x.view(-1, 1, 1) ** 2) +
                    (grid_y - cy.view(-1, 1, 1)) ** 2 / (2 * sigma_y.view(-1, 1, 1) ** 2)))
    s = gauss.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    gauss = gauss / s
    return F.kl_div(attn_single.clamp(min=1e-8).log(), gauss, reduction='batchmean')


class AttentionSupervisionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_maps, bbox_target, bbox_mask):
        if attn_maps is None or len(attn_maps) == 0:
            return t.tensor(0.0, device=bbox_target.device, requires_grad=True)
        B, N, _ = bbox_target.shape
        H, W = attn_maps[0].shape[2], attn_maps[0].shape[3]
        loss = t.tensor(0.0, device=bbox_target.device)
        count = 0
        for h in range(min(len(attn_maps), N)):
            mask = bbox_mask[:, h] > 0
            if mask.sum() == 0:
                continue
            attn = attn_maps[h].squeeze(1)
            pred_h = attn[mask]
            cx = bbox_target[mask, h, 0]
            cy = bbox_target[mask, h, 1]
            bw = bbox_target[mask, h, 2].clamp(min=0.02)
            bh = bbox_target[mask, h, 3].clamp(min=0.02)
            # 逐样本计算，避免 [B,N,H,W] 大张量
            kl_h = compute_single_gaussian_kl(pred_h, cx, cy, bw, bh, H, W)
            loss = loss + kl_h
            count += 1
        if count == 0:
            return t.tensor(0.0, device=bbox_target.device, requires_grad=True)
        return loss / count


def attention_diversity_loss(attn_maps):
    if attn_maps is None or len(attn_maps) < 2:
        return t.tensor(0.0, device=attn_maps[0].device if attn_maps is not None and len(attn_maps) > 0 else t.device('cpu'), requires_grad=True)
    n = len(attn_maps)
    loss = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ai = attn_maps[i].flatten(2)
            aj = attn_maps[j].flatten(2)
            loss = loss + (ai * aj).sum(dim=2).mean()
    return loss / (n * (n - 1) / 2)


def spatial_ordering_loss(attn_maps, bbox_preds=None, bbox_mask=None):
    if attn_maps is None or len(attn_maps) < 2:
        return t.tensor(0.0, device=attn_maps[0].device if attn_maps is not None and len(attn_maps) > 0 else t.device('cpu'), requires_grad=True)

    loss = t.tensor(0.0, device=attn_maps[0].device)

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

    if bbox_preds is not None and bbox_mask is not None:
        for i in range(len(bbox_preds) - 1):
            cx_i = bbox_preds[i][:, 0]
            cx_j = bbox_preds[i + 1][:, 0]

            mask_both_valid = (bbox_mask[:, i] > 0) & (bbox_mask[:, i + 1] > 0)
            if mask_both_valid.sum() > 0:
                violation = F.relu(cx_i[mask_both_valid] - cx_j[mask_both_valid])
                loss = loss + violation.mean() * 0.5

    return loss
