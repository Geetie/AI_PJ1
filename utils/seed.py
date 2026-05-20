import random
import numpy as np
import torch as t


def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    if hasattr(t.backends, 'cudnn'):
        t.backends.cudnn.benchmark = not deterministic
        t.backends.cudnn.deterministic = deterministic


def make_epoch_generator(base_seed=42, epoch=0):
    generator = t.Generator()
    generator.manual_seed(base_seed + epoch)
    return generator
