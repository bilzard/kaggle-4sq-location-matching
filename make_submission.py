import argparse
import os.path as osp

import pandas as pd
from tqdm import tqdm


def make_submission(hp):
    dfs = []

    reader = pd.read_csv(hp.input_path, compression="gzip", chunksize=hp.chunk_size)
    for chunk in tqdm(reader):
        chunk = chunk[chunk["prob"] > hp.threshold]
        dfs.append(chunk)

    submission = pd.concat(dfs)
    submission = (
        submission.groupby("id1")
        .agg(matches=("id2", lambda x: " ".join(x)))
        .reset_index()
    )
    submission.rename({"id1": "id"}, axis=1, inplace=True)
    submission.to_csv(osp.join(hp.output_path, "submission.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--chunk_size", type=int, default=1024 * 256)
    parser.add_argument("--threshold", type=int, default=0.5)

    hp = parser.parse_args()

    make_submission(hp)
