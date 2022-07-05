import pandas as pd


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
