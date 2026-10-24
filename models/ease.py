import numpy as np
from scipy.linalg import cho_factor, cho_solve


class EASE:
    def __init__(self, regularization=300.0):
        self.regularization = regularization

    def fit(self, interactions):
        gram = (interactions.T @ interactions).toarray().astype(np.float64)
        gram.flat[::len(gram) + 1] += self.regularization
        factor = cho_factor(gram, overwrite_a=True, check_finite=False)
        inverse = cho_solve(factor, np.eye(len(gram)), check_finite=False)
        self.weights = (-inverse / np.diag(inverse)[None, :]).astype(np.float32)
        np.fill_diagonal(self.weights, 0)
        return self

    def predict(self, interactions):
        return np.asarray(interactions @ self.weights, dtype=np.float32)
