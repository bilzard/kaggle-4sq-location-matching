import argparse
import os.path as osp

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from general.util import import_by_name, as_chunks
from general.array import MemMapSequentialWriter


def calc_embeddings(hp):
    cfg = import_by_name(f"config.{hp.config}", "cfg")
    assert hp.column in cfg.text_columns, f"column should be text column"

    model = SentenceTransformer(hp.model)
    n_dims = model.get_sentence_embedding_dimension()
    n_items = len(
        pd.read_csv(
            f"{hp.input_path}",
            compression="gzip",
            usecols=[hp.column],
        )
    )
    reader = pd.read_csv(
        hp.input_path,
        compression="gzip",
        usecols=[hp.column],
        chunksize=hp.chunk_size,
    )
    writer = MemMapSequentialWriter(
        osp.join(hp.output_path, f"embeddings_{hp.column}.mmp"),
        dtype="float32",
        shape=(n_items, n_dims),
    )

    for chunk in reader:
        sentences = chunk[hp.column].to_numpy()
        embeddings_chunk = model.encode(
            sentences, show_progress_bar=True, batch_size=hp.batch_size
        )
        writer.write_all(embeddings_chunk)

    writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--column", type=str, default="name")
    parser.add_argument("--batch_size", type=str, default=256)
    parser.add_argument("--chunk_size", type=str, default=1024 * 256)
    parser.add_argument("--config", type=str, default="base")
    hp = parser.parse_args()

    calc_embeddings(hp)
