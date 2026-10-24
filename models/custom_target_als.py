"""Пятая модель: тот же решатель ALS с новой числовой целью вместо 0/1."""

import numpy as np

from .implicit_als import ImplicitALS


def graded_preference(max_percent, n_long, liked, disliked):
    """Целевой сигнал для пар с хотя бы одним Listen+ до временной границы."""
    completion = (np.minimum(max_percent, 100) - 50) / 50
    repeated = np.minimum(n_long, 3) / 3
    return np.clip(0.4 + 0.3 * completion + 0.2 * repeated
                   + 0.1 * liked - 0.5 * disliked, 0.0, 1.0).astype(np.float32)


class CustomTargetALS(ImplicitALS):
    def fit(self, graded_target, positive_observations):
        return super().fit(graded_target, observed=positive_observations)
