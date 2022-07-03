import argparse
import sys
import wandb

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.ditto import train
from train_util.seed import seed_everything


def make_task_name(path):
    """
    "/data/hoge/hoge_hoge.tsv.gz" -> hoge_hoge
    """
    return path.split("/")[-1].split(".")[0]


def make_run_tag(hp):
    run_tag = f"{hp.task}_{hp.lm}_ep{hp.n_epochs}_id{hp.run_id}"
    run_tag = run_tag.replace("/", "_")
    return run_tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str)
    parser.add_argument("val_path", type=str)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default="distilbert")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)

    hp = parser.parse_args()

    # set seeds
    seed_everything(hp.run_id)

    # only a single task for baseline
    hp.task = make_task_name(hp.train_path)

    # create the tag of the run
    hp.run_tag = make_run_tag(hp)

    # load train/dev/test sets
    train_dataset = DittoDataset(
        hp.train_path, lm=hp.lm, max_len=hp.max_len, size=hp.size, da=hp.da
    )
    valid_dataset = DittoDataset(hp.val_path, lm=hp.lm)

    # train and evaluate the model
    with wandb.init(project="4sq", name=hp.run_tag, config=vars(hp)):
        train(train_dataset, valid_dataset, hp)
