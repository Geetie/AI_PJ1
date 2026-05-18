import os
from glob import glob

import pandas as pd
from torchvision import transforms

from config import config


class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        padding = (pad_left, pad_top, max_dim - w - pad_left, max_dim - h - pad_top)
        return transforms.functional.pad(img, padding, fill=self.fill)


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoint_files = glob(os.path.join(checkpoint_dir, '*.pth'))
    if not checkpoint_files:
        return None
    best_files = [f for f in checkpoint_files if os.path.basename(f).startswith('best-')]
    if best_files:
        return max(best_files, key=lambda x: os.path.getmtime(x))
    return max(checkpoint_files, key=lambda x: os.path.getmtime(x))


def write2csv(results, csv_path):
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    df['file_name'] = df['file_name'].apply(lambda x: os.path.basename(x))
    df.to_csv(csv_path, sep=',', index=None)
    print('Results saved to %s' % csv_path)
