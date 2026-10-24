"""Единые матрицы, кандидаты и метрики для всех моделей."""

import numpy as np
import pandas as pd
from scipy import sparse

from models.custom_target_als import graded_preference


def index_events(events, metadata):
    users = np.sort(events.uid.unique())
    items = np.sort(metadata.item_id.unique())
    frame = events.copy()
    frame["u"] = pd.Categorical(frame.uid, categories=users).codes.astype(np.int32)
    frame["i"] = pd.Categorical(frame.item_id, categories=items).codes.astype(np.int32)
    meta = metadata.set_index("item_id").loc[items].reset_index()
    return frame, meta, users, items


def _binary_matrix(rows, cols, n_users, n_items):
    result = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items), dtype=np.float32,
    )
    result.data.fill(1.0)
    return result


def training_matrices(events, cutoff, n_users, n_items):
    past = events.loc[events.timestamp < cutoff]
    known = _binary_matrix(past.u.to_numpy(), past.i.to_numpy(), n_users, n_items)
    listens = past.loc[(past.event_type == "listen") & (past.played_ratio_pct > 50)]
    grouped = listens.groupby(["u", "i"], observed=True).agg(
        n_long=("item_id", "size"), max_pct=("played_ratio_pct", "max")
    ).reset_index()
    u = grouped.u.to_numpy(dtype=np.int32)
    i = grouped.i.to_numpy(dtype=np.int32)
    binary = _binary_matrix(u, i, n_users, n_items)

    reactions = past.loc[past.event_type.isin(["like", "unlike", "dislike", "undislike"]),
                         ["u", "i", "event_type", "timestamp"]]
    if len(reactions):
        states = reactions.groupby(["u", "i", "event_type"], observed=True).timestamp.max()
        states = states.unstack("event_type").reindex(
            columns=["like", "unlike", "dislike", "undislike"]
        )
        grouped = grouped.join(states, on=["u", "i"])
        liked = (grouped["like"].fillna(-1) > grouped["unlike"].fillna(-1)).to_numpy()
        disliked = (grouped["dislike"].fillna(-1) > grouped["undislike"].fillna(-1)).to_numpy()
    else:
        liked = np.zeros(len(grouped), dtype=bool)
        disliked = np.zeros(len(grouped), dtype=bool)
    target = graded_preference(grouped.max_pct.to_numpy(),
                               grouped.n_long.to_numpy(), liked, disliked)
    custom = sparse.csr_matrix((target, (u, i)), shape=(n_users, n_items))
    custom.eliminate_zeros()
    popularity = np.asarray(binary.sum(axis=0)).ravel().astype(np.float32)
    return {"binary": binary, "custom": custom, "known": known,
            "popularity": popularity, "positive_pairs": len(grouped)}


def future_truth(events, start, finish, known):
    future = events.loc[(events.timestamp >= start) & (events.timestamp < finish)
                        & (events.event_type == "listen")
                        & (events.played_ratio_pct > 50), ["u", "i"]].drop_duplicates()
    pairs = future.to_numpy(dtype=np.int32)
    if len(pairs):
        novel = np.asarray(known[pairs[:, 0], pairs[:, 1]]).ravel() == 0
        pairs = pairs[novel]
    truth = {}
    for u, i in pairs:
        truth.setdefault(int(u), set()).add(int(i))
    return truth


def rank_and_evaluate(scores, matrices, truth, k=10):
    scores = np.asarray(scores, dtype=np.float32).copy()
    n_users, n_items = scores.shape
    if not truth:
        raise ValueError("Нет будущих новых положительных треков для оценки")
    cold = np.diff(matrices["binary"].indptr) == 0
    scores[cold] = matrices["popularity"]
    known_u, known_i = matrices["known"].nonzero()
    scores[known_u, known_i] = -np.inf
    users = np.array(sorted(truth), dtype=np.int32)
    selected = scores[users]
    top = np.argpartition(-selected, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(selected, top, axis=1)
    top = np.take_along_axis(top, np.argsort(-top_scores, axis=1), axis=1)
    recalls, ndcgs = [], []
    discounts = 1 / np.log2(np.arange(2, k + 2))
    for row, u in zip(top, users):
        hits = np.fromiter((int(i in truth[int(u)]) for i in row), dtype=np.int8)
        recalls.append(hits.sum() / len(truth[int(u)]))
        ndcgs.append((hits * discounts).sum() / discounts[:min(k, len(truth[int(u)]))].sum())
    return {
        "NDCG@10": float(np.mean(ndcgs)),
        "Recall@10": float(np.mean(recalls)),
        "Coverage@10": float(np.unique(top).size / n_items),
        "users": len(users), "cold_users": int(cold[users].sum()),
        "new_positive_pairs": int(sum(map(len, truth.values()))),
    }
