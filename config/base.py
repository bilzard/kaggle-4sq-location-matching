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
