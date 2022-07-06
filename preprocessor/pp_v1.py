from functools import partial
from preprocessor.transform import (
    fill_na,
    filter_spam_v1,
    normalize,
    fill_blank,
    compose,
)


class Preprocessor:
    def __init__(self, hp, cfg):
        self.transforms = compose(
            [
                partial(fill_na, cols=cfg.text_cols),
                filter_spam_v1,
                partial(
                    normalize,
                    cols=cfg.text_cols,
                    lower=False,
                    uc_format=None,
                    remove_non_word=False,
                    norm_blank=True,
                ),
            ]
        )

    def run(self, df):
        df = self.transforms(df)
        return df
