from functools import partial
from preprocessor.transform import (
    fill_na,
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
        print("Proportion of filled rows:")
        print((1 - (df.isna().sum(axis=0) / len(df))).reset_index())
        return df
