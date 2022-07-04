import gzip
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_open_func(path):
    openfunc = gzip.open if path.endswith(".gz") else open
    return openfunc


def count_lines(path):
    with set_open_func(path)(path, "rt") as fp:
        for i, _ in enumerate(fp):
            pass

    return i
