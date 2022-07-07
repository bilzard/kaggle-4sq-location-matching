import os.path as osp
from tqdm import tqdm

from blocker.util import *
from general.profile import SimpleTimer
from general.tabular import save_to_chunks


def prune_by_threshold(preds, distances, threshold):
    preds = np.array(preds)
    distances = np.array(distances)
    # keep distance only less than threshold
    for i in range(len(preds)):
        min_dist = distances[i].min()
        do_keep = (distances[i] < threshold) | (distances[i] == min_dist)
        preds[i] = preds[i][do_keep]
        distances[i] = distances[i][do_keep]
    return preds, distances


def do_blocking(
    df,
    embeddings_list,
    weight_list,
    hp,
    cfg,
):
    grouped = df.groupby(cfg.h3_col)
    h3_to_count = df.value_counts(cfg.h3_col)
    origin_to_search_point = {
        origin: get_search_points(origin) for origin in h3_to_count.keys()
    }
    point_set = set(df[cfg.h3_col].unique().tolist())
    logger = Logger()
    timer = SimpleTimer()

    recalls = []
    weights = []
    h3_ids = []
    preds_list = []
    ids_list = []
    distances_list = []
    score = 0.0
    weight_processed = 0.0

    pbar = tqdm(total=len(df))
    n = 0
    for i, (origin, query_df) in enumerate(grouped):
        if hp.debug_iter > 0:
            if i > hp.debug_iter:
                continue
        n += 1
        if hp.monitor:
            set_exponential_monitor(logger, n)
        else:
            logger.set_debug(False)

        logger.log("=" * 36 + f" ROUND {i + 1} " + "=" * 36)
        logger.log(f"Constructing search & query of {origin}:")

        query_df = query_df.reset_index()
        search_points = origin_to_search_point[origin]
        search_points = [pt for pt in search_points if pt in point_set]
        search_df = pd.concat([grouped.get_group(pt) for pt in search_points])
        search_df = search_df.reset_index()
        query_set = set(query_df[cfg.h3_col].unique().tolist())
        query_idx = search_df.query(f"{cfg.h3_col} in @query_set").index

        logger.log(f"  - #points in query: {len(query_df)}")
        logger.log(f"  - #points in search space: {len(search_df)}")

        k = min(len(search_df), hp.k_neighbor)
        if len(search_df) > 1:
            logger.log("Calculate Embeddings:")
            global_search_index = search_df["index"].to_numpy()

            embeddings_local = [
                embds[global_search_index].copy() for embds in embeddings_list
            ]

            embeddings_concat = concat_embeddings(embeddings_local, weight_list)
            search = embeddings_concat
            query = search[query_idx]

            if hp.blocker_type in {"text", "combination"}:
                predictor = GeneralPredictor(k_neighbor=k, normalize=hp.normalize)
            elif hp.blocker_type == "location":
                predictor = LocationPredictor(k_neighbor=k)
            else:
                raise NotImplementedError(hp.blocker_type)

            idx2id = search_df["id"].to_dict()
            preds, distances = predictor.predict(search, query, idx2id)

            # prune by threshold
            preds, distances = prune_by_threshold(
                preds, distances, cfg.blocker_thresholds[hp.blocker_type]
            )

        else:
            logger.log("Skip calculating Embeddings:")
            preds = search_df["id"].to_numpy().reshape(1, -1).tolist()
            distances = [[0.0]]

        logger.log(f"  - len(pred[0]): {len(preds[0])}")
        logger.log("Evaluation without post process:")

        weight = len(query_df)
        pbar.update(weight)
        weight_processed += weight
        if hp.evaluate:
            gts = create_gts(query_df)
            result = evaluate(preds, gts, eps=1e-15)
            logger.log(
                ", ".join([f"{key}: {value:.4f}" for key, value in result.items()])
            )

            recall = result["Recall"]
            score += weight * recall
            logger.log(f"score (running mean): {score / weight_processed:5f}")
            recalls.append(recall)

        weights.append(weight)
        h3_ids.append(origin)
        preds_list.append(preds)
        ids_list.append(query_df["id"].to_numpy())
        distances_list.append(distances)

    print("=" * 80)
    if hp.monitor:
        if hp.evaluate:
            score = simple_eval(recalls, weights, len(df))
            print(f"approximate score: {score:.4f}")
        print(f"processed: {sum(weights) / len(df):.4f}")

    timer.start("flatten result")
    preds_flat, ids_flat, distances_flat = [], [], []
    for ids, preds, distances in zip(ids_list, preds_list, distances_list):
        preds_flat.extend(preds)
        ids_flat.extend(ids)
        distances_flat.extend(distances)
    timer.endshow()

    timer.start("making & saving preds")
    preds_df = pd.DataFrame(
        {"id": ids_flat, "preds": preds_flat, "distances": distances_flat}
    )
    preds_df.to_csv(
        osp.join(hp.output_path, f"preds_{hp.blocker_type}.csv.gz"),
        compression="gzip",
        index=False,
    )
    timer.endshow()

    timer.start("making & saving stat")
    stat_df = pd.DataFrame({cfg.h3_col: h3_ids, "weight": weights})
    if hp.evaluate:
        stat_df["recall"] = recalls
    stat_df.to_csv(
        osp.join(hp.output_path, f"stat_{hp.blocker_type}.csv.gz"),
        compression="gzip",
        index=False,
    )
    timer.endshow()
