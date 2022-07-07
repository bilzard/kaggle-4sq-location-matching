import pandas as pd
from humanize import naturalsize


def show_memory_usage(df):
    """show memory usage of a dataframe in human-readable output."""
    total_usage = 0

    print("Memory usage:")
    for col in df.columns:
        usage = df[col].memory_usage(deep=True)
        total_usage += usage
        print(f"  - {col}: {naturalsize(usage)}")

    print(f"Total: {naturalsize(total_usage)}")


def sort_by_categorical(df, col):
    """
    Sort input dataframe by col.
    The order is based on value_counts.

    This process is necessary for efficient access to embeddings backed on memmap during k-nearest neighbor search.
    ---
    [side effects]
        - The output dataframe doesn't preserve the index of the input dataframe.
    """
    df = df.copy()
    vc = df.value_counts(col)
    vc_df = pd.DataFrame({"count": vc}).reset_index()
    cat = pd.CategoricalDtype(vc_df[col], ordered=True)
    df[col] = pd.Categorical(df[col].astype(cat))
    df.sort_values(col, inplace=True)
    df = df.reset_index(drop=True)

    return df


def save_to_chunks(df, prefix, chunk_size=1024 * 256):
    total_len = len(df)
    for i in range(total_len // chunk_size):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        chunk.to_parquet(f"{prefix}.{i}.parquet", index=False)
    if total_len - chunk_size * i > 0:
        chunk = df.iloc[(i + 1) * chunk_size :]
        chunk.to_parquet(f"{prefix}.{i + 1}.parquet", index=False)
