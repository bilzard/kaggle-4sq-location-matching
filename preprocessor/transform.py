import unicodedata


def compose(transforms):
    def transform_func(df):
        for transform in transforms:
            df = transform(df)
        return df

    return transform_func


def drop_na(df, cols):
    print(f"drop NaN {cols}:")
    df = df.dropna(subset=cols)
    return df


def normalize(
    df, cols, lower=False, uc_format="NFKC", remove_non_word=False, norm_blank=True
):
    print(f"Normalize: [{cols}]")
    df = df.copy()

    total_affected = 0
    for col in cols:
        ser = df[col]
        if lower:
            ser = ser.str.lower()

        if uc_format is not None:
            ser = ser.apply(lambda x: unicodedata.normalize(uc_format, x))

        if remove_non_word:
            ser = ser.str.replace("\W", " ", regex=True)

        if norm_blank:
            ser = ser.str.replace("\s+", " ", regex=True)
            ser = ser.str.replace("^ $", " ", regex=True)

        filtr = df[col] != ser
        n_affected = len(ser[filtr])
        total_affected += n_affected
        print(f"  - #affected in `{col}`: {n_affected:,}")
        df[f"{col}"] = ser

    print(f"  - Total: {total_affected:,}")
    return df


def normalize_phone(df):
    print(f"Normalize phone:")
    df = df.copy()
    df["phone"] = df["phone"].str.replace("[-() ]", "", regex=True)

    return df


def normalize_zip(df):
    print(f"Normalize zip:")
    df = df.copy()
    df["zip"] = df["zip"].str.replace("[-() ]", "", regex=True)

    return df


def fill_na(df, cols):
    df = df.copy()
    df[cols] = df[cols].fillna("")
    return df


def drop_duplicated(df, cols):
    print(f"drop duplicated {cols}:")
    print(f"  #locations before: {len(df):,}")
    df = df.drop_duplicates(cols)
    print(f"  #locations after: {len(df):,}")
    return df


def drop_single_poi(df):
    print("drop single POI:")
    poi_df = df.groupby("point_of_interest").agg(count=("id", "count")).reset_index()
    print(f"  #POIs before: {len(poi_df):,}")
    poi_df = poi_df.query("count > 1")
    print(f"  #POIs after: {len(poi_df):,}")
    pois = set(poi_df["point_of_interest"].to_numpy())
    print(f"  #locations before: {len(df):,}")
    df = df.query("point_of_interest in @pois")
    print(f"  #locations after: {len(df):,}")
    return df


def filter_spam_v1(df):
    df = df.copy()
    print("Spam filter:")
    spam_filter = df["address"].str.contains(
        "↘|▊|高仿|精仿|免费|男士|https?://|<iframe|阿玛尼", na=False, regex=True
    )
    num_spams = len(df.loc[spam_filter, "address"])
    print(f"  - #Spams filtered: {num_spams}")
    df.loc[spam_filter, "address"] = ""
    return df


def fill_blank(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = df[col].str.replace("^$", "-", regex=True)
    return df
