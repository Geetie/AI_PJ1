import numpy as np
import torch as t


def cutmix_data(x, y, bbox_target, bbox_mask, alpha=1.0):
    batch_size = x.size(0)
    index = t.randperm(batch_size).to(x.device)
    lam = np.random.beta(alpha, alpha)
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    bbox_a, bbox_b = bbox_target, bbox_target[index]
    mask_a, mask_b = bbox_mask, bbox_mask[index]
    return mixed_x, y_a, y_b, bbox_a, bbox_b, mask_a, mask_b, lam
