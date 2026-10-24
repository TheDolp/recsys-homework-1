"""Небольшая общая обвязка поиска и оценки, не содержащая готовых моделей."""

from time import perf_counter

import pandas as pd

from evaluation import rank_and_evaluate
from models.item_knn import ItemKNN
from models.implicit_als import ImplicitALS
from models.ease import EASE
from models.fm import FactorizationMachine
from models.custom_target_als import CustomTargetALS


def fit_and_predict(name, config, matrices, metadata):
    start = perf_counter()
    if name == "ItemKNN":
        model = ItemKNN(**config).fit(matrices["binary"])
        scores = model.predict(matrices["binary"])
    elif name == "iALS":
        model = ImplicitALS(**config).fit(matrices["binary"])
        scores = model.predict()
    elif name == "EASE":
        model = EASE(**config).fit(matrices["binary"])
        scores = model.predict(matrices["binary"])
    elif name == "FM":
        model = FactorizationMachine(**config).fit(
            matrices["binary"], matrices["known"], metadata
        )
        scores = model.predict()
    elif name == "CustomALS":
        model = CustomTargetALS(**config).fit(
            matrices["custom"], matrices["binary"]
        )
        scores = model.predict()
    else:
        raise ValueError(name)
    return scores, perf_counter() - start, model


def search(name, configs, matrices, metadata, truth):
    rows = []
    for config in configs:
        scores, seconds, model = fit_and_predict(name, config, matrices, metadata)
        metrics = rank_and_evaluate(scores, matrices, truth)
        row = {"модель": name, "параметры": str(config), "секунд": round(seconds, 2), **metrics}
        if name == "FM":
            row["последний_BPR_loss"] = model.loss_history[-1]
        rows.append(row)
    return pd.DataFrame(rows)


def chosen_config(table, configs):
    return configs[int(table["NDCG@10"].to_numpy().argmax())]
