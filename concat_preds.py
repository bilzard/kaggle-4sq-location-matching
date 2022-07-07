import argparse
import os.path as osp

import pandas as pd
from tqdm import tqdm
from general.tabular import ChunkedCsvWriter


def make_submission(hp):
    writer = ChunkedCsvWriter(osp.join(hp.output_path, "union_preds.csv.gz"))

    readers = [
        pd.read_csv(preds_path, compression="gzip", chunksize=hp.chunk_size)
        for preds_path in hp.preds_paths
    ]
    for preds_list in tqdm(zip(*readers)):
        for preds in preds_list:
            preds["preds"] = preds["preds"].apply(lambda x: set(eval(x)))

        union_preds = (
            pd.concat(preds_list, axis=0)
            .groupby("id")
            .agg(preds=("preds", lambda x: list(set.union(*x))))
        ).reset_index()
        writer.write(union_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_paths", action="append", required=True)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--chunk_size", type=int, default=1024 * 256)

    hp = parser.parse_args()

    make_submission(hp)
