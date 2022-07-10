import argparse
import os

import pandas as pd

from general.util import import_by_name, show_memory_usage


def preprocess(hp):
    Preprocessor = import_by_name(f"preprocess.{hp.preprocessor}", "Processor")

    input_df = pd.read_csv(f"{hp.input_path}")
    h3_df = pd.read_csv(hp.h3_path)

    pre_processor = Preprocessor()
    target_df = pre_processor.run(input_df)

    target_df = target_df.merge(h3_df, on="id")
    target_df.to_csv(
        os.path.join(hp.output_path, "preprocessed.csv.gz"),
        compression="gzip",
        index=False,
    )
    show_memory_usage(target_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("h3_path", type=str)
    parser.add_argument("--preprocessor", type=str, default="pp_v0")
    parser.add_argument("--output_path", type=str, default="./output")
    hp = parser.parse_args()

    preprocess(hp)
