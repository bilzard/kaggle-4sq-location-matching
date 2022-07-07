import argparse

import os.path as osp

import dask.dataframe as dd
import pandas as pd

from dask.diagnostics import ProgressBar

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

    input_df = dd.from_pandas(input_df, chunksize=1024 * 256)
    input_df["text"] = input_df.apply(
        lambda x: " ".join(
            [f"COL {col} VAL {x[col]}" for col in cfg.text_cols]
            + [f"COL {col} VAL {x[col]:3f}" for col in cfg.num_cols]
        ),
        axis=1,
        meta=(None, "str"),
    )
    print("Computing ditto text columns...")
    with ProgressBar():
        text = input_df[["id", "text"]].compute()

    del input_df

    print(f"{hp.preds_prefix}")
    preds = dd.read_parquet(f"{hp.preds_prefix}.*.parquet")[["id", "preds"]]
    pairs = preds.explode("preds")
    pairs = pairs.rename(columns={"id": "id1", "preds": "id2"})
    pairs = pairs.query("id1 != id2")

    result = pairs.merge(
        text.rename(columns={"id": "id1", "text": "left"}), on="id1", how="left"
    ).merge(text.rename(columns={"id": "id2", "text": "right"}), on="id2", how="left")
    result["matched"] = 0

    print("First few lines of final output:")
    print(result.head())

    print("Saving output to file...")
    with ProgressBar():
        result.to_parquet(
            osp.join(hp.output_path, "ditto"),
            name_function=lambda x: f"ditto.{x}.parquet",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("preds_prefix", type=str)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--preprocessor", type=str, default="ppm_v0")
    hp = parser.parse_args()

    make_ditto_output(hp)
