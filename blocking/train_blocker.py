import gzip
import os
import argparse
import math
import wandb

from sentence_transformers.readers import InputExample
from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from torch.utils.data import DataLoader


class Reader:
    def get_examples(self, fn):
        fopen_func = gzip.open if fn.endswith(".gz") else open
        with fopen_func(fn, "rt") as fp:
            return self.process_file(fp)

    def process_file(self, fp):
        raise NotImplementedError()


class TrainReader(Reader):
    """A simple reader class for the matching datasets."""

    def __init__(self):
        self.guid = 0

    def process_file(self, fp):
        examples = []
        for line in fp:
            sent1, sent2, label = line.strip().split("\t")
            examples.append(
                InputExample(guid=self.guid, texts=[sent1, sent2], label=int(label))
            )
            self.guid += 1
        return examples


class EvalReader(Reader):
    """A simple reader class for the matching datasets for eval."""

    def process_file(self, fp):
        sentences1 = []
        sentences2 = []
        scores = []
        for line in fp:
            sent1, sent2, label = line.strip().split("\t")
            sentences1.append(sent1)
            sentences2.append(sent2)
            scores.append(int(label))
        return sentences1, sentences2, scores


def train(hp):
    """Train the advanced blocking model
    Store the trained model in hp.model_fn.

    Args:
        hp (Namespace): the hyperparameters

    Returns:
        None
    """
    # define model
    model_names = {
        "distilbert": "distilbert-base-uncased",
        "bert": "bert-base-uncased",
        "albert": "albert-base-v2",
        "stsb-xlm-r-multilingual": "sentence-transformers/stsb-xlm-r-multilingual",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "all-distilroberta-v1": "sentence-transformers/all-distilroberta-v1",
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    }

    word_embedding_model = models.Transformer(
        model_names[hp.lm], max_seq_length=hp.max_seq_length
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # load the training and validation data
    reader = TrainReader()
    train_examples = reader.get_examples(hp.train_fn)
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=hp.batch_size)
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=2,
    )

    dev_reader = EvalReader()
    sentences1, sentences2, scores = dev_reader.get_examples(hp.valid_fn)
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1, sentences2, scores, batch_size=hp.batch_size, show_progress_bar=True
    )

    warmup_steps = math.ceil(
        len(train_dataloader) * hp.n_epochs / hp.batch_size * 0.1
    )  # 10% of train data for warm-up

    if os.path.exists(hp.model_fn):
        import shutil

        shutil.rmtree(hp.model_fn)

    def eval_callback(score, epoch, steps):
        print(f"epoch {epoch} ({steps} steps): val/score={score}")
        wandb.log({"epoch": epoch, "steps": steps, "val/score": score})

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=hp.n_epochs,
        evaluation_steps=hp.evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=hp.model_fn,
        use_amp=hp.fp16,
        callback=eval_callback,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_fn", type=str, default="../data/er_magellan/Structured/Beer/train.txt"
    )
    parser.add_argument(
        "--valid_fn", type=str, default="../data/er_magellan/Structured/Beer/valid.txt"
    )
    parser.add_argument("--model_fn", type=str, default="model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default="distilbert")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--evaluation_steps", type=int, default=0)

    hp = parser.parse_args()

    # create the tag of the run
    run_tag = "lm=%s" % (hp.lm)
    run_tag = run_tag.replace("/", "_")

    with wandb.init(project="4sq-blocker", name=run_tag, config=vars(hp)):
        train(hp)
