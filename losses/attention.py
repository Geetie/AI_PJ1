import torch as t
import torch.nn as nn
import torch.nn.functional as F


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
            loss = loss + F.kl_div(pred_h.clamp(min=1e-8).log(), gt_h, reduction='batchmean')
            count += 1
        if count == 0:
            return t.tensor(0.0, device=bbox_target.device, requires_grad=True)
        return loss / count


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
    if attn_maps is None or len(attn_maps) < 2:
        return t.tensor(0.0, device='cuda' if t.cuda.is_available() else 'cpu', requires_grad=True)

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
