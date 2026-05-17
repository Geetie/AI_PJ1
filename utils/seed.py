import random
import numpy as np
import torch as t


def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.benchmark = not deterministic
    t.backends.cudnn.deterministic = deterministic
