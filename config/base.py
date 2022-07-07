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
    "combination": [1, 0.4521, 789.4274],
    "location": [1],
    "text": [1, 0.3415],
}
cfg.blocker_thresholds = {
    "combination": 3.162277660168379e-06,
    "location": 0.00031622776601683794,
    "text": 1.0,
}
