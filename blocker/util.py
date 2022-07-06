import faiss
import h3
import numpy as np
import pandas as pd

from numpy import deg2rad, cos, sin
from sklearn.neighbors import KNeighborsRegressor


def create_gpos_haversine(df):
    lat = df["latitude"].apply(deg2rad).to_numpy()
    lon = df["longitude"].apply(deg2rad).to_numpy()
    gpos = np.stack([lat, lon], axis=-1)
    return gpos


def create_gpos_cartesian(df):
    lat = df["latitude"].apply(deg2rad).to_numpy(dtype=np.float32)
    lon = df["longitude"].apply(deg2rad).to_numpy(dtype=np.float32)
    x = cos(lat) * cos(lon)
    y = cos(lat) * sin(lon)
    z = sin(lat)
    gpos = np.stack([x, y, z], axis=-1)
    return gpos


def subtract_mean(src, dst):
    dst[:] -= src.mean(axis=0, keepdims=True)[:]


def normalize_L2(src, dst):
    dst[:] /= ((src**2).sum(axis=1, keepdims=True) ** 0.5)[:]


def concat_embeddings(embeddings_list, weights):
    assert len(embeddings_list) == len(
        weights
    ), f"{len(embeddings_list)}, {len(weights)}"
    return np.concatenate(
        [w * ebd for ebd, w in zip(embeddings_list, weights)], axis=-1
    )


class L2NeighborSearcher:
    def __init__(self, k_neighbor, normalize):
        self.k_neighbor = k_neighbor
        self.index = None
        self.normalize = normalize

    def fit(self, X):
        dims = X.shape[-1]
        X = np.ascontiguousarray(X)

        if self.normalize:
            faiss.normalize_L2(X)
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        flat_config
        index = faiss.GpuIndexFlatL2(res, dims, flat_config)
        index.add(X)
        self.index = index

    def search(self, Xq):
        Xq = np.ascontiguousarray(Xq)
        if self.normalize:
            faiss.normalize_L2(Xq)
        D, I = self.index.search(Xq, self.k_neighbor)
        return I, D


class HaversineNeighborSearcher:
    def __init__(self, k_neighbor):
        self.k_neighbor = k_neighbor
        self.knn = KNeighborsRegressor(
            n_neighbors=k_neighbor, metric="haversine", n_jobs=-1
        )

    def fit(self, X):
        self.knn.fit(X, np.arange(len(X)))

    def search(self, Xq):
        dists, idxs = self.knn.kneighbors(Xq, return_distance=True)
        return idxs, dists


class GeneralPredictor:
    def __init__(self, k_neighbor, normalize):
        self.k_neighbor = k_neighbor
        self.normalize = normalize

    def predict(self, search, query, idx2id):
        assert query.shape[-1] == search.shape[-1]
        assert query.shape[0] <= search.shape[0], f"{query.shape[0]} {search.shape[0]}"

        knn = L2NeighborSearcher(self.k_neighbor, self.normalize)
        knn.fit(search)

        I, D = knn.search(query)
        preds = I.tolist()
        distances = D.tolist()
        preds = [[idx2id[p] for p in pred] for pred in preds]
        return preds, distances


class LocationPredictor:
    def __init__(self, k_neighbor):
        self.k_neighbor = k_neighbor

    def predict(self, search, query, idx2id):
        assert query.shape[-1] == search.shape[-1]
        assert query.shape[0] <= search.shape[0], f"{query.shape[0]} {search.shape[0]}"

        knn = HaversineNeighborSearcher(self.k_neighbor)
        knn.fit(search)

        I, D = knn.search(query)
        preds = I.tolist()
        distances = D.tolist()
        preds = [[idx2id[p] for p in pred] for pred in preds]
        return preds, distances


def create_gts(df, col="id"):
    gt_df = (
        df.groupby("point_of_interest").agg(gt=(col, lambda x: list(x))).reset_index()
    )
    df = pd.merge(df, gt_df, on="point_of_interest", how="left")
    gts = df["gt"].to_numpy()
    return gts


def evaluate(preds, gts, eps=1e-15):
    preds_ = [set(pred) for pred in preds]
    gts_ = [set(gt) for gt in gts]
    intersections = [pred.intersection(gt) for pred, gt in zip(preds_, gts_)]
    unions = [pred.union(gt) for pred, gt in zip(preds_, gts_)]
    ious = [
        len(intersec) / len(union) for intersec, union in zip(intersections, unions)
    ]
    recalls = [len(intersec) / len(gt) for intersec, gt in zip(intersections, gts_)]
    precisions = [
        len(intersec) / len(pred) for intersec, pred in zip(intersections, preds_)
    ]
    f1 = np.mean([2 * r * p / (r + p + eps) for r, p in zip(recalls, precisions)])

    result = {
        "IoU": np.mean(ious),
        "Recall": np.mean(recalls),
        "Precision": np.mean(precisions),
        "F1": f1,
    }

    return result


class Logger:
    def __init__(self, is_debug=True):
        self.is_debug = is_debug

    def set_debug(self, is_debug):
        self.is_debug = is_debug

    def log(self, text):
        if self.is_debug:
            print(text)


def get_search_points(origin):
    s1, s2 = h3.k_ring_distances(origin, 1)
    return list(s1.union(s2))


def set_exponential_monitor(logger, n):
    if n < 10:
        is_monitor = True
    elif n < 100:
        is_monitor = n % 10 == 0
    elif n < 1000:
        is_monitor = n % 100 == 0
    else:
        is_monitor = n % 1000 == 0

    logger.set_debug(is_monitor)


def simple_eval(recalls, weights, n_total):
    return np.sum([r * w for r, w in zip(recalls, weights)]) / np.sum(weights)
