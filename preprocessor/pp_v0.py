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
            ]
        )

    def run(self, df):
        df = self.transforms(df)
        return df
