import importlib
import itertools


def import_by_name(module_path, keyword):
    target = getattr(importlib.import_module(module_path), keyword)
    return target


def as_chunks(iterable, num_chunks):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, num_chunks))
        if not chunk:
            return
        yield chunk
