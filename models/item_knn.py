import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbours=100, shrinkage=20):
        self.neighbours = neighbours
        self.shrinkage = shrinkage

    def fit(self, interactions):
        counts = np.asarray(interactions.sum(axis=0)).ravel()
        co = (interactions.T @ interactions).toarray().astype(np.float32)
        denominator = np.sqrt(np.outer(counts, counts)) + self.shrinkage
        similarity = np.divide(co, denominator, out=np.zeros_like(co), where=denominator > 0)
        np.fill_diagonal(similarity, 0)
        n_items = similarity.shape[0]
        rows, cols, values = [], [], []
        for source in range(n_items):
            row = similarity[source]
            k = min(self.neighbours, n_items - 1)
            best = np.argpartition(row, -k)[-k:]
            best = best[row[best] > 0]
            rows.extend([source] * len(best))
            cols.extend(best.tolist())
            values.extend(row[best].tolist())
        self.similarity = sparse.csr_matrix(
            (values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32
        )
        return self

    def predict(self, interactions):
        return (interactions @ self.similarity).toarray()
