import math
import os

import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics
import wandb

from torch.utils import data
from torch.optim import AdamW
from transformers import AutoModel, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

# map lm name to huggingface's pre-trained model names
lm_mp = {
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "xlmr-base": "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
    "xlmr-xl": "facebook/xlm-roberta-xl",
    "xlmr-xxl": "facebook/xlm-roberta-xxl",
    "multilingual-bert-cased": "bert-base-multilingual-cased",
}


class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device="cuda", lm="xlmr-base", alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device)  # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device)  # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size]  # (batch_size, emb_size)
            enc2 = enc[batch_size:]  # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc)  # .squeeze() # .sigmoid()


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    iou = None
    if threshold is None:
        # search best threshold
        threshold = 0.5
        iou = 0.0

        for thr in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > thr else 0 for p in all_probs]
            best_iou = metrics.jaccard_score(all_y, pred, average="binary")
            if best_iou > iou:
                iou = best_iou
                threshold = thr

    pred = [1 if p > threshold else 0 for p in all_probs]
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        all_y, pred, average="binary"
    )
    if iou is None:
        iou = metrics.jaccard_score(all_y, pred, average="binary")
    return {
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "iou": iou,
        "threshold": threshold,
    }


def train_step(train_iter, model, optimizer, scheduler, hp, monitor_step=10):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=hp.fp16)

    # criterion = nn.MSELoss()
    mean_loss = 0
    num_iter = len(train_iter)
    pbar = tqdm(train_iter)
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()

        with autocast(enabled=hp.fp16):
            if len(batch) == 2:
                x, y = batch
                prediction = model(x)
            else:
                x1, x2, y = batch
                prediction = model(x1, x2)

            loss = criterion(prediction, y.to(model.device))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if i % monitor_step == 0:
            pbar.set_description(f"Train loss (running): {loss.item():.4f}")
            wandb.log({"train/loss": loss.item()})
        mean_loss += loss.item() / num_iter
        del loss
    return mean_loss


def train(trainset, validset, testset, hp):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(
        dataset=trainset,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=padder,
    )
    valid_iter = data.DataLoader(
        dataset=validset,
        batch_size=hp.batch_size * 16,
        shuffle=False,
        num_workers=0,
        collate_fn=padder,
    )
    test_iter = data.DataLoader(
        dataset=testset,
        batch_size=hp.batch_size * 16,
        shuffle=False,
        num_workers=0,
        collate_fn=padder,
    )

    # initialize model, optimizer, and LR scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DittoModel(device=device, lm=hp.lm, alpha_aug=hp.alpha_aug)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    # 10% of train data for warm-up
    warmup_steps = math.ceil(len(train_iter) * hp.n_epochs / hp.batch_size * 0.1)

    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps
    )

    best_dev_score = best_test_score = 0.0
    for epoch in range(1, hp.n_epochs + 1):
        # train
        model.train()
        loss = train_step(train_iter, model, optimizer, scheduler, hp)

        # eval
        model.eval()
        dev_result = evaluate(model, valid_iter)
        test_result = evaluate(model, test_iter, threshold=dev_result["threshold"])

        if dev_result["iou"] > best_dev_score:
            best_dev_score = dev_result["iou"]
            best_test_score = test_result["iou"]
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, "model.pt")
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                }
                torch.save(ckpt, ckpt_path)

        # logging
        scalars = {
            **{f"val/{k}": v for k, v in dev_result.items() if k != "threshold"},
            **{f"test/{k}": v for k, v in test_result.items() if k != "threshold"},
            "train/loss_epoch": loss,
        }
        print(
            f"[epoch {epoch}] test/best_score: {best_test_score:.3f}, "
            + ", ".join([f"{k}={v:.3f}" for k, v in scalars.items()])
        )
        wandb.log(dict(**scalars, epoch=epoch))
