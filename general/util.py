import importlib


def import_by_name(module_path, keyword):
    target = getattr(importlib.import_module(module_path), keyword)
    return target
