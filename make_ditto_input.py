import argparse

import os.path as osp

import pandas as pd

from tqdm import tqdm

from general.util import import_by_name
from general.profile import SimpleTimer


def make_ditto_output(hp):
    Preprocessor = import_by_name(f"preprocessor.{hp.preprocessor}", "Preprocessor")
    cfg = import_by_name(f"config.{hp.config}", "cfg")

    timer = SimpleTimer()
    preprocessor = Preprocessor(hp, cfg)

    timer.start("loading input dataframe")
    input_df = pd.read_csv(hp.input_path)
    timer.endshow()

    timer.start("preprocess")
    input_df = preprocessor.run(input_df)
    timer.endshow()

    print("Preprocessed dataframe:")
    print(input_df.head())

    tqdm.pandas()
    print("Computing ditto text columns...")
    input_df["text"] = input_df.progress_apply(
        lambda x: " ".join(
            [f"COL {col} VAL {x[col]}" for col in cfg.text_cols]
            + [f"COL {col} VAL {x[col]:3f}" for col in cfg.num_cols]
        ),
        axis=1,
    )
    text = input_df[["id", "text"]]

    del input_df

    first_time = True
    pbar = tqdm(total=len(text))
    for preds in pd.read_csv(
        hp.preds_path, usecols=["id", "preds"], chunksize=1024 * 64
    ):
        preds["preds"] = preds["preds"].apply(lambda x: eval(x))
        pairs = preds.explode("preds")
        pairs = pairs.rename(columns={"id": "id1", "preds": "id2"})

        result = pairs.merge(
            text.rename(columns={"id": "id1", "text": "left"}), on="id1", how="left"
        ).merge(
            text.rename(columns={"id": "id2", "text": "right"}), on="id2", how="left"
        )
        result["matched"] = 0

        result.to_csv(
            osp.join(hp.output_path, "ditto.csv.gz"),
            compression="gzip",
            index=False,
            mode="w" if first_time else "a",
            header=first_time,
        )
        pbar.update(len(preds))
        if first_time:
            print("First Few lines of result")
            print(result.head())
            first_time = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("preds_path", type=str)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--preprocessor", type=str, default="ppm_v0")
    hp = parser.parse_args()

    make_ditto_output(hp)
