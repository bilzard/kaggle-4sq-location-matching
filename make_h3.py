import argparse
import os

from functools import partial
from multiprocessing import Pool, cpu_count

import h3
from humanize import naturalsize
import pandas as pd

from tqdm import tqdm


def show_memory_usage(df):
    total_usage = 0

    print("Memory usage:")
    for col in df.columns:
        usage = df[col].memory_usage(deep=True)
        total_usage += usage
        print(f"  - {col}: {naturalsize(usage)}")

    print(f"Total: {naturalsize(total_usage)}")


def geo_to_h3_res(data, resolution):
    lat, lon = data
    return h3.geo_to_h3(lat, lon, resolution=resolution)


def make_h3(hp):
    h3_col = f"h3_res{hp.resolution}"
    train = pd.read_csv(hp.input_path)
    lats, lons = train["latitude"].to_numpy(), train["longitude"].to_numpy()

    h3_df = pd.DataFrame({"id": train["id"]})
    n_thread = hp.num_workers * 2
    with Pool(n_thread) as p:
        converter = partial(geo_to_h3_res, resolution=hp.resolution)
        h3_df[h3_col] = list(p.imap(converter, tqdm(zip(lats, lons), total=len(lats))))

    show_memory_usage(h3_df)
    h3_df.to_csv(
        os.path.join(hp.output_path, f"h3_res{hp.resolution}.csv.gz"),
        compression="gzip",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--ouput_ptah", type=str, default="./")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=None)
    hp = parser.parse_args()

    if hp.num_workers is None:
        hp.num_workers = cpu_count()

    make_h3(hp)
