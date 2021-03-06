import gzip

import torch

from torch.utils import data
from transformers import AutoTokenizer

from config.model_alias import model_alias


def get_tokenizer(lm):
    if lm in model_alias:
        return AutoTokenizer.from_pretrained(model_alias[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self, path, max_len=256, size=None, lm="xlmr-base", da=None):
        self.tokenizer = get_tokenizer(lm)
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        def _handle_data(iterable):
            pairs, labels = [], []
            for line in iterable:
                s1, s2, label = line.strip().split("\t")
                pairs.append((s1, s2))
                labels.append(int(label))
            return pairs, labels

        if isinstance(path, list):
            self.pairs, self.labels = _handle_data(path)
        elif isinstance(path, str):
            fopen_func = gzip.open if path.endswith(".gz") else open
            with fopen_func(path, "rt") as fp:
                self.pairs, self.labels = _handle_data(fp)
        else:
            raise RuntimeError("path's type should be list or string")

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.da = da
        self.augmenter = None

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # left + right
        x = self.tokenizer.encode(
            text=left, text_pair=right, max_length=self.max_len, truncation=True
        )

        # augment if da is set
        if self.da is not None:
            combined = self.augmenter.augment_sent(left + " [SEP] " + right, self.da)
            left, right = combined.split(" [SEP] ")
            x_aug = self.tokenizer.encode(
                text=left, text_pair=right, max_length=self.max_len, truncation=True
            )
            return x, x_aug, self.labels[idx]
        else:
            return x, self.labels[idx]

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = max([len(x) for x in x1 + x2])
            x1 = [xi + [0] * (maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0] * (maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), torch.LongTensor(x2), torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0] * (maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), torch.LongTensor(y)
