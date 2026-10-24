"""FM второго порядка с попарным BPR-лоссом, реализованная на PyTorch."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


class _FMNetwork(nn.Module):
    def __init__(self, n_users, n_items, artist_ids, album_ids, factors, use_metadata):
        super().__init__()
        self.use_metadata = use_metadata
        self.register_buffer("artist_ids", torch.as_tensor(artist_ids, dtype=torch.long))
        self.register_buffer("album_ids", torch.as_tensor(album_ids, dtype=torch.long))
        dimensions = [n_users, n_items, int(max(artist_ids)) + 1, int(max(album_ids)) + 1]
        self.linear = nn.ModuleList([nn.Embedding(size, 1) for size in dimensions])
        self.latent = nn.ModuleList([nn.Embedding(size, factors) for size in dimensions])
        self.intercept = nn.Parameter(torch.zeros(()))
        for table in self.linear:
            nn.init.zeros_(table.weight)
        for table in self.latent:
            nn.init.normal_(table.weight, std=0.05)

    def forward(self, users, items):
        fields = [users, items]
        if self.use_metadata:
            fields.extend([self.artist_ids[items], self.album_ids[items]])
        vectors = torch.stack([self.latent[j](field) for j, field in enumerate(fields)], dim=1)
        linear = sum(self.linear[j](field).squeeze(-1) for j, field in enumerate(fields))
        interactions = 0.5 * (vectors.sum(dim=1).square() - vectors.square().sum(dim=1))
        return self.intercept + linear + interactions.sum(dim=1)

    @torch.no_grad()
    def score_all(self):
        user_vectors = self.latent[0].weight
        item_vectors = self.latent[1].weight
        item_bias = self.linear[1].weight.squeeze(-1)
        if self.use_metadata:
            artist_vectors = self.latent[2](self.artist_ids)
            album_vectors = self.latent[3](self.album_ids)
            item_parts = torch.stack([item_vectors, artist_vectors, album_vectors], dim=1)
            item_sum = item_parts.sum(dim=1)
            item_bias = item_bias + self.linear[2](self.artist_ids).squeeze(-1)
            item_bias = item_bias + self.linear[3](self.album_ids).squeeze(-1)
            item_bias = item_bias + 0.5 * (
                item_sum.square() - item_parts.square().sum(dim=1)
            ).sum(dim=1)
        else:
            item_sum = item_vectors
        score = user_vectors @ item_sum.T + item_bias[None, :]
        score += self.linear[0].weight + self.intercept
        return score.cpu().numpy().astype(np.float32)


class FactorizationMachine:
    def __init__(self, factors=16, learning_rate=0.01, regularization=1e-5,
                 epochs=3, batch_size=8192, seed=42, use_metadata=True):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.use_metadata = use_metadata

    def fit(self, positives, known, metadata):
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        rng = np.random.default_rng(self.seed)
        artists = pd.factorize(metadata.artist_id.fillna(-1), sort=True)[0]
        albums = pd.factorize(metadata.album_id.fillna(-1), sort=True)[0]
        self.net = _FMNetwork(positives.shape[0], positives.shape[1],
                              artists, albums, self.factors, self.use_metadata)
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.regularization)
        pos_users, pos_items = positives.nonzero()
        self.loss_history = []
        for _ in range(self.epochs):
            order = rng.permutation(len(pos_users))
            total_loss = 0.0
            for batch in np.array_split(order, max(1, int(np.ceil(len(order) / self.batch_size)))):
                users = pos_users[batch]
                items = pos_items[batch]
                negatives = rng.integers(0, positives.shape[1], size=len(batch))
                invalid = np.asarray(known[users, negatives]).ravel() > 0
                while invalid.any():
                    negatives[invalid] = rng.integers(0, positives.shape[1], size=invalid.sum())
                    invalid = np.asarray(known[users, negatives]).ravel() > 0
                u = torch.as_tensor(users, dtype=torch.long)
                positive = torch.as_tensor(items, dtype=torch.long)
                negative = torch.as_tensor(negatives, dtype=torch.long)
                optimizer.zero_grad()
                loss = F.softplus(self.net(u, negative) - self.net(u, positive)).mean()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch)
            self.loss_history.append(total_loss / len(order))
        return self

    def predict(self):
        return self.net.score_all()
