import numpy as np


class ImplicitALS:
    """Weighted ALS; target может быть бинарным или авторским непрерывным."""

    def __init__(self, factors=24, alpha=20.0, regularization=10.0,
                 iterations=4, seed=42):
        self.factors = factors
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed

    def _update(self, fixed, observed, target):
        result = np.empty((observed.shape[0], self.factors), dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            gram = fixed.T @ fixed
        eye = np.eye(self.factors, dtype=np.float64)
        for row in range(observed.shape[0]):
            ids = observed.indices[observed.indptr[row]:observed.indptr[row + 1]]
            if not len(ids):
                result[row] = 0
                continue
            vectors = fixed[ids]
            values = target[row, ids].toarray().ravel().astype(np.float64)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                matrix = gram + self.alpha * (vectors.T @ vectors) + self.regularization * eye
                right = (1 + self.alpha) * (values @ vectors)
            result[row] = np.linalg.solve(matrix, right)
        if not np.isfinite(result).all():
            raise FloatingPointError("ALS получил нечисловые факторы")
        return result

    def fit(self, target, observed=None):
        observed = target if observed is None else observed
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, 0.1, (target.shape[1], self.factors)).astype(np.float64)
        target_t = target.T.tocsr()
        observed_t = observed.T.tocsr()
        for _ in range(self.iterations):
            self.users = self._update(self.items, observed, target)
            self.items = self._update(self.users, observed_t, target_t)
        return self

    def predict(self):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scores = self.users @ self.items.T
        if not np.isfinite(scores).all():
            raise FloatingPointError("ALS получил нечисловые оценки")
        return scores
