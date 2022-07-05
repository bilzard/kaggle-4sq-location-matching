import importlib

from humanize import naturalsize


def import_by_name(module_path, keyword):
    target = getattr(importlib.import_module(module_path), keyword)
    return target


def show_memory_usage(df):
    total_usage = 0

    print("Memory usage:")
    for col in df.columns:
        usage = df[col].memory_usage(deep=True)
        total_usage += usage
        print(f"  - {col}: {naturalsize(usage)}")

    print(f"Total: {naturalsize(total_usage)}")
