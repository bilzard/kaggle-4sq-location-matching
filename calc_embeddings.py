import argparse
import os.path as osp

import pandas as pd
from sentence_transformers import SentenceTransformer

from general.util import import_by_name
from general.array import MemMapSequentialWriter
from general.profile import SimpleTimer


def calc_embeddings(hp):
    cfg = import_by_name(f"config.{hp.config}", "cfg")
    assert hp.column in cfg.text_cols, f"column should be text column"
    timer = SimpleTimer()

    model = SentenceTransformer(hp.model)
    n_dims = model.get_sentence_embedding_dimension()
    input_df = pd.read_csv(
        f"{hp.input_path}",
        compression="gzip",
        usecols=[hp.column],
        keep_default_na=False,
    )
    n_items = len(input_df)
    if hp.column == "categories":
        timer.start("Replacing cammas to the special token `[SEP]`")
        input_df[hp.column] = input_df[hp.column].str.replace(", ", " [SEP] ")
        timer.endshow()

        print("Rows which contains special token:")
        filter_special_token = input_df[hp.column].str.contains("[SEP]", regex=False)
        print(input_df[filter_special_token].head())

    timer.start("Saving updated dataframe")
    input_df.to_csv(
        osp.join(hp.output_path, "tmp.csv.gz"), compression="gzip", index=False
    )
    timer.endshow()
    del input_df

    reader = pd.read_csv(
        osp.join(hp.output_path, "tmp.csv.gz"),
        compression="gzip",
        usecols=[hp.column],
        chunksize=hp.chunk_size,
        keep_default_na=False,
    )
    writer = MemMapSequentialWriter(
        osp.join(hp.output_path, f"embeddings_{hp.column}.mmp"),
        dtype="float32",
        shape=(n_items, n_dims),
    )

    total_chunks = n_items // hp.chunk_size + 1
    for i, chunk in enumerate(reader):
        print("=" * 38 + f" chunk {i + 1}/{total_chunks} " + "=" * 38)
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
    parser.add_argument(
        "--column", type=str, choices={"name", "categories"}, default="name"
    )
    parser.add_argument("--batch_size", type=str, default=256)
    parser.add_argument("--chunk_size", type=str, default=1024 * 256)
    parser.add_argument("--config", type=str, default="base")
    hp = parser.parse_args()

    calc_embeddings(hp)
