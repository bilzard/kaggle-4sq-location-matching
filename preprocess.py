import argparse
import os

import pandas as pd

from general.util import import_by_name
from general.tabular import sort_by_categorical, show_memory_usage


def preprocess(hp):
    Preprocessor = import_by_name(f"preprocessor.{hp.preprocessor}", "Preprocessor")
    cfg = import_by_name(f"config.{hp.config}", "cfg")

    input_df = pd.read_csv(f"{hp.input_path}")
    h3_df = pd.read_csv(hp.h3_path)

    pre_processor = Preprocessor(hp, cfg)
    target_df = pre_processor.run(input_df)

    target_df = target_df.merge(h3_df, on="id")
    target_df = sort_by_categorical(target_df, cfg.h3_col)

    show_memory_usage(target_df)
    target_df.to_csv(
        os.path.join(hp.output_path, "preprocessed.csv.gz"),
        compression="gzip",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("h3_path", type=str)
    parser.add_argument("--preprocessor", type=str, default="pp_v0")
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--output_path", type=str, default="./output")
    hp = parser.parse_args()

    preprocess(hp)
