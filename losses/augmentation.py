import torch as t
from torch.distributions import Beta


def cutmix_data(x, y, bbox_target, bbox_mask, alpha=1.0):
    batch_size = x.size(0)
    index = t.randperm(batch_size).to(x.device)
    lam = Beta(t.tensor(alpha, dtype=t.float32), t.tensor(alpha, dtype=t.float32)).sample().item()
    W, H = x.size(3), x.size(2)
    cut_rat = (1. - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = t.randint(0, W, (1,)).item()
    cy = t.randint(0, H, (1,)).item()
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    bbox_a, bbox_b = bbox_target, bbox_target[index]
    mask_a, mask_b = bbox_mask, bbox_mask[index]
    return mixed_x, y_a, y_b, bbox_a, bbox_b, mask_a, mask_b, lam
