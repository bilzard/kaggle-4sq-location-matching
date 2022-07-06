from types import SimpleNamespace

cfg = SimpleNamespace()

cfg.text_cols = [
    "name",
    "address",
    "city",
    "state",
    "zip",
    "country",
    "url",
    "phone",
    "categories",
]
cfg.num_cols = ["latitude", "longitude"]
cfg.h3_resolution = 1
cfg.h3_col = f"h3_res{cfg.h3_resolution}"
cfg.text_embedding_cols = ["name", "categories"]
cfg.k_neighbor = 25
cfg.blocker_weights = {
    "text": [1, 0.341558957832999],
    "location": [1],
    "combination": [1, 0.4509794940496579, 861.3927932470813],
}
