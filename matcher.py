import argparse
import os
import time

from multiprocessing import cpu_count

import jsonlines
import torch
import wandb

from torch.utils import data
from tqdm.auto import tqdm
from scipy.special import softmax
from sklearn.metrics import jaccard_score

from ditto_light.ditto import evaluate, DittoModel
from ditto_light.exceptions import ModelNotFoundError
from ditto_light.dataset import DittoDataset
from ditto_light.train_util import (
    seed_everything,
    set_open_func,
    count_lines,
    as_chunks,
)


def make_task_name(path):
    """
    "/data/hoge/hoge_hoge.tsv.gz" -> hoge_hoge
    """
    return path.split("/")[-1].split(".")[0]


def make_run_tag(hp):
    run_tag = f"{hp.task}_{hp.lm}_id{hp.seed}"
    run_tag = run_tag.replace("/", "_")
    return run_tag


def classify(sentences, model, pbar, batch_size=256, lm="distilbert", max_len=256):
    """Apply the MRPC model.

    Args:
        sentences (list of str): the sequence pairs
        model (MultiTaskNet): the model in pytorch
        max_len (int, optional): the max sequence length

    Returns:
        list of float: the scores of the pairs
    """
    dataset = DittoDataset(sentences, max_len=max_len, lm=lm)
    iterator = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=hp.num_workers,
        collate_fn=DittoDataset.pad,
    )

    # prediction
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, _ = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            pbar.update(x.shape[0])

    return all_probs


def predict(
    input_path,
    output_path,
    model,
    run_tag,
    chunk_size=1024 * 128,
    lm="distilbert",
    max_len=256,
    batch_size=256,
):
    """Run the model over the input file containing the candidate entry pairs

    Args:
        input_path (str): the input file path
        output_path (str): the output file path
        config (Dictionary): task configuration
        model (DittoModel): the model for prediction
        batch_size (int): the batch size
        max_len (int, optional): the max sequence length

    Returns:
        None
    """
    start_time = time.time()
    total_inputs = count_lines(input_path)
    with set_open_func(input_path)(input_path, "rt") as reader, jsonlines.open(
        output_path, mode="w"
    ) as writer:
        pbar = tqdm(total=total_inputs)
        for chunks in as_chunks(reader, chunk_size):
            probs = classify(
                chunks,
                model,
                pbar,
                batch_size=batch_size,
                lm=lm,
                max_len=max_len,
            )
            writer.write_all(probs)

    run_time = time.time() - start_time
    os.system("echo %s %f >> log.txt" % (run_tag, run_time))


def tune_threshold(model, hp):
    """Tune the prediction threshold for a given model on a validation set"""

    # summarize the sequences up to the max sequence length
    seed_everything(hp.seed)

    # load dev sets
    valid_dataset = DittoDataset(hp.val_path, max_len=hp.max_len, lm=hp.lm)
    valid_iter = data.DataLoader(
        dataset=valid_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=hp.num_workers,
        collate_fn=DittoDataset.pad,
    )

    result = evaluate(model, valid_iter, threshold=None)
    threshold = result["threshold"]

    seed_everything(hp.seed)
    predict(
        hp.val_path,
        "tmp.jsonl",
        model,
        hp.run_tag,
        max_len=hp.max_len,
        lm=hp.lm,
        batch_size=hp.batch_size,
    )

    predicts = []
    with jsonlines.open("tmp.jsonl", mode="r") as reader:
        for line in reader:
            predicts.append(int(float(line) > threshold))
    os.system("rm tmp.jsonl")

    labels = []
    with set_open_func(hp.val_path)(hp.val_path, "rt") as fp:
        for line in fp:
            labels.append(int(line.split("\t")[-1]))

    # Check if actual score
    pred_score = jaccard_score(labels, predicts)
    print(", ".join(f"val/{key}={val:.5f}" for key, val in result.items()))
    print(f"eval score: {result['iou']:.5f}, pred scpre: {pred_score:.5f}")

    return threshold


def load_model(checkpoint_path, lm, use_gpu):
    """Load a model for a specific task.

    Args:
        checkpoint_path (str): the path of the model checkpoint file
        lm (str): the language model
        use_gpu (boolean): whether to use gpu

    Returns:
        Dictionary: the task config
        MultiTaskNet: the model
    """
    # load models
    if not os.path.exists(checkpoint_path):
        raise ModelNotFoundError(checkpoint_path)

    if use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    model = DittoModel(device=device, lm=lm)

    saved_state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(saved_state["model"])
    model = model.to(device)

    return model


def match(hp):
    model = load_model(hp.checkpoint_path, hp.lm, hp.use_gpu)

    # load the models
    seed_everything(hp.seed)

    # tune threshold
    if hp.val_path is not None:
        tune_threshold(model, hp)

    # run prediction
    predict(
        hp.input_path,
        hp.output_path,
        model,
        hp.run_tag,
        max_len=hp.max_len,
        lm=hp.lm,
        batch_size=hp.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_path", type=str, default="output/result.jsonl")
    parser.add_argument("--lm", type=str, default="distilbert")
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=1024 * 512)
    parser.add_argument("--monitor", dest="monitor", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    hp = parser.parse_args()

    hp.task = make_task_name(hp.input_path)
    hp.run_tag = make_run_tag(hp)
    if hp.num_workers is None:
        hp.num_workers = cpu_count()

    if hp.monitor:
        with wandb.init(project="4sq-matcher", name=hp.run_tag, config=vars(hp)):
            match(hp)
    else:
        match(hp)
