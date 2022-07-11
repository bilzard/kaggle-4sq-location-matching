import argparse
import json
import os

from multiprocessing import cpu_count

import torch
import wandb

from torch.utils import data

from ditto_light.ditto import evaluate, DittoModel
from ditto_light.exceptions import ModelNotFoundError
from ditto_light.dataset import DittoDataset
from ditto_light.train_util import seed_everything, worker_init_fn


def make_task_name(path):
    """
    "/data/hoge/hoge_hoge.tsv.gz" -> hoge_hoge
    """
    return path.split("/")[-1].split(".")[0]


def make_run_tag(hp):
    run_tag = f"{hp.task}_{hp.lm}_id{hp.seed}"
    run_tag = run_tag.replace("/", "_")
    return run_tag


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
    models = [load_model(path, hp.lm, hp.use_gpu) for path in hp.checkpoint_paths]

    seed_everything(hp.seed)

    dataset = DittoDataset(hp.input_path, max_len=hp.max_len, lm=hp.lm)
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=hp.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=DittoDataset.pad,
    )
    result = evaluate(models, loader, threshold=hp.threshold)

    print(", ".join(f"val/{key}={val:.5f}" for key, val in result.items()))
    with open(os.path.join(hp.output_path, f"{hp.run_tag}.json"), "w") as fp:
        json.dump(result, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--checkpoint_paths", action="append", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--lm", type=str, default="distilbert")
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=1024 * 512)
    parser.add_argument("--monitor", dest="monitor", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--task", type=str, default=None)
    hp = parser.parse_args()

    if hp.task is None:
        hp.task = make_task_name(hp.input_path)
    hp.run_tag = make_run_tag(hp)
    if hp.num_workers is None:
        hp.num_workers = cpu_count()

    if hp.monitor:
        with wandb.init(project="4sq-matcher", name=hp.run_tag, config=vars(hp)):
            match(hp)
    else:
        match(hp)
