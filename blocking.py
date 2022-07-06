import argparse
import os.path as osp

import numpy as np
import pandas as pd

from general.tabular import show_memory_usage
from general.util import import_by_name
from blocker.util import (
    create_gpos_cartesian,
    create_gpos_haversine,
    subtract_mean,
    normalize_L2,
)
from blocker.blocker import do_blocking


def load_data(hp, cfg):
    usecols = ["id", cfg.h3_col]
    if hp.evaluate:
        usecols += ["point_of_interest"]

    input_df = pd.read_csv(f"{hp.input_path}", compression="gzip", usecols=usecols)
    input_df[cfg.h3_col] = input_df[cfg.h3_col].astype("category")
    if hp.evaluate:
        input_df["point_of_interest"] = input_df["point_of_interest"].astype("category")

    print(f"Memory usage of input dataframe:")
    show_memory_usage(input_df)

    return input_df


def load_embeddings_list(hp, cfg, input_df):
    embeddings_list = []
    if hp.blocker_type in {"combination", "text"}:
        embeddings_list += [
            np.memmap(
                osp.join(hp.embeddings_path, f"embeddings_{col}.mmp"),
                mode="r",
                dtype="float32",
                shape=(
                    len(input_df),
                    384,
                ),  # TODO: preserve shape and dtype in metadata
            )
            for col in cfg.text_embedding_cols
        ]
    if hp.blocker_type == "combination":
        embeddings_list += [create_gpos_cartesian(input_df)]
    if hp.blocker_type == "location":
        embeddings_list += [create_gpos_haversine(input_df)]
    return embeddings_list


def transform_embeddings_list(embeddings_list, input_size, cfg):
    embeddings_list_norm = [
        np.memmap(
            osp.join(hp.output_path, f"embeddings_{col}_norm.mmp"),
            mode="w+",
            dtype="float32",
            shape=(input_size, 384),  # TODO: preserve shape and dtype in metadata
        )
        for col in cfg.text_embedding_cols
    ]

    for embeddings, embeddings_norm in zip(embeddings_list, embeddings_list_norm):
        subtract_mean(embeddings, embeddings_norm)
        normalize_L2(embeddings_norm, embeddings_norm)
        embeddings_norm.flags.writeable = False

    return embeddings_list_norm


def block(hp):
    cfg = import_by_name(f"config.{hp.config}", "cfg")

    input_df = load_data(hp, cfg)

    # subtract mean embedding & normalize(switched by option)
    embeddings_list = load_embeddings_list(hp.blocker_type, input_df)
    if hp.blocker_type in {"combination", "text"}:
        embeddings_list[: len(cfg.text_embedding_cols)] = transform_embeddings_list(
            embeddings_list[: len(cfg.text_embedding_cols)], len(input_df), cfg
        )

    # knn search grouped by H3 location ID
    do_blocking(
        input_df,
        embeddings_list,
        cfg.blocker_weights[hp.blocker_type],
        hp,
        cfg,
        k_neighbor=hp.k_neighbor,
        normalize=hp.normalize,
        evaluate=hp.evaluate,
        monitor=hp.monitor,
        debug_iter=hp.debug_iter,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument(
        "--blocker_type",
        type=str,
        choices={"text", "location", "combination"},
        default="combination",
    )
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--embeddings_path", type=str, default="./embeddings")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--k_neighbor", type=int, default=25)
    parser.add_argument("--evaluate", dest="evaluate", action="store_true")
    parser.add_argument("--monitor", dest="monitor", action="store_true")
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--debug_iter", type=int, default=0)
    hp = parser.parse_args()

    block(hp)
