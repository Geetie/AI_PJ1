import torch as t
import torch.nn as nn
import torch.nn.functional as F


def compute_single_gaussian_kl(attn_single, cx, cy, bw, bh, h, w):
    eps = 1e-8
    grid_y = t.arange(h, device=attn_single.device, dtype=t.float32).view(1, h, 1) / max(h, 1)
    grid_x = t.arange(w, device=attn_single.device, dtype=t.float32).view(1, 1, w) / max(w, 1)
    sigma_x = (bw / 2.5).clamp(min=0.05)
    sigma_y = (bh / 2.5).clamp(min=0.05)
    sq_x = (grid_x - cx.view(-1, 1, 1)) ** 2 / (2 * sigma_x.view(-1, 1, 1) ** 2 + eps)
    sq_y = (grid_y - cy.view(-1, 1, 1)) ** 2 / (2 * sigma_y.view(-1, 1, 1) ** 2 + eps)
    sq_sum = (sq_x + sq_y).clamp(max=30.0)
    gauss = t.exp(-sq_sum)
    gauss_sum = gauss.sum(dim=(1, 2), keepdim=True).clamp(min=eps)
    gauss = gauss / gauss_sum
    
    pred_sum = attn_single.sum(dim=(1, 2), keepdim=True).clamp(min=eps)
    pred_normalized = attn_single / pred_sum
    
    kl = F.kl_div(
        t.log(pred_normalized.clamp(min=eps, max=1.0)),
        gauss,
        reduction='batchmean'
    )
    if t.isnan(kl).any() or t.isinf(kl).any():
        kl = F.kl_div(
            t.log(pred_normalized.clamp(min=eps, max=0.99)),
            gauss.clamp(min=eps, max=0.99),
            reduction='mean'
        )
    
    return kl.clamp(max=10.0)


class AttentionSupervisionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_maps, bbox_target, bbox_mask):
        if attn_maps is None or len(attn_maps) == 0:
            return t.tensor(0.0, device=bbox_target.device)
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
            kl_h = compute_single_gaussian_kl(pred_h, cx, cy, bw, bh, H, W)
            loss = loss + kl_h
            count += 1
        if count == 0:
            return t.tensor(0.0, device=bbox_target.device)
        return loss / count


def attention_diversity_loss(attn_maps):
    eps = 1e-8
    if attn_maps is None or len(attn_maps) < 2:
        dev = attn_maps[0].device if attn_maps is not None and len(attn_maps) > 0 else t.device('cpu')
        return t.tensor(0.0, device=dev)
    n = len(attn_maps)
    loss = t.tensor(0.0, device=attn_maps[0].device)
    for i in range(n):
        for j in range(i + 1, n):
            ai = attn_maps[i].flatten(2).clamp(min=eps)
            aj = attn_maps[j].flatten(2).clamp(min=eps)
            loss = loss + (ai * aj).sum(dim=2).mean()
    return (loss / (n * (n - 1) / 2)).clamp(max=10.0)


def spatial_ordering_loss(attn_maps, bbox_preds=None, bbox_mask=None):
    eps = 1e-8
    if attn_maps is None or len(attn_maps) < 2:
        return t.tensor(0.0, device=attn_maps[0].device if attn_maps is not None and len(attn_maps) > 0 else t.device('cpu'))

    loss = t.tensor(0.0, device=attn_maps[0].device)
    pair_count = 0

    for i in range(len(attn_maps) - 1):
        if bbox_mask is not None:
            mask_both = (bbox_mask[:, i] > 0) & (bbox_mask[:, i + 1] > 0)
            if mask_both.sum() == 0:
                continue
        ai = attn_maps[i].flatten(2).clamp(min=eps)
        aj = attn_maps[i + 1].flatten(2).clamp(min=eps)
        H, W = attn_maps[i].shape[2], attn_maps[i].shape[3]
        grid_x = t.arange(W, device=ai.device).float() / max(W - 1, 1)
        grid_x = grid_x.unsqueeze(0).expand(H, W).reshape(1, 1, -1)
        ci = (ai * grid_x).sum(dim=2) / (ai.sum(dim=2) + eps)
        cj = (aj * grid_x).sum(dim=2) / (aj.sum(dim=2) + eps)
        violation = F.relu(ci - cj)
        if bbox_mask is not None:
            violation = violation[mask_both]
        loss = loss + violation.mean().clamp(max=10.0)
        pair_count += 1

    if bbox_preds is not None and bbox_mask is not None:
        for i in range(len(bbox_preds) - 1):
            cx_i = bbox_preds[i][:, 0]
            cx_j = bbox_preds[i + 1][:, 0]

            mask_both_valid = (bbox_mask[:, i] > 0) & (bbox_mask[:, i + 1] > 0)
            if mask_both_valid.sum() > 0:
                violation = F.relu(cx_i[mask_both_valid] - cx_j[mask_both_valid])
                loss = loss + violation.mean().clamp(max=10.0) * 0.5
                pair_count += 1

    if pair_count > 0:
        loss = loss / pair_count
    return loss.clamp(max=10.0)
