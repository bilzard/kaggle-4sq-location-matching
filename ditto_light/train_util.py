import gzip
import itertools
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
    return i + 1


def as_chunks(iterable, num_chunks):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, num_chunks))
        if not chunk:
            return
        yield chunk


def worker_init_fn(worker_id):
    worker_seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
