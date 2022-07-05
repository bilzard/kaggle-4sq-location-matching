import numpy as np


class MemMapSequentialWriter:
    """sequencial writer"""

    def __init__(self, filename, dtype, shape):
        self._memmap = np.memmap(filename, dtype=dtype, mode="w+", shape=shape)
        self.head = 0

    def write(self, obj):
        """write a single object to memmap"""
        self._memmap[self.head] = obj
        self.head += 1

    def write_all(self, objs):
        """write multiple objects to memmap"""
        n_objs = len(objs)
        self._memmap[self.head : self.head + n_objs] = objs
        self.head += n_objs

    def flush(self):
        self._memmap.flush()
