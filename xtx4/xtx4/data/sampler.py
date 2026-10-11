"""Multi-scale batch sampler: one network size ``S`` per batch.

The dataset renders each sample directly at ``S`` (fix D3), so the size must be
chosen *before* loading -- by the batch sampler. Each yielded batch is a list of
``(sample_idx, S)`` keys sharing one ``S`` drawn with ``size_probs``.

Use with ``DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, ...)``.
Everything is a pure function of ``(seed, epoch)`` -> identical in every worker
and on resume after ``set_epoch``.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class MultiScaleBatchSampler(Sampler[list[tuple[int, int]]]):
    """Yields lists of ``(idx, S)`` with one ``S`` per batch.

    Args:
        num_samples: dataset length.
        batch_size: samples per batch.
        sizes: candidate network sizes.
        size_probs: probabilities per size (normalised; ``None`` -> uniform).
        seed: base seed; the epoch is mixed in via :meth:`set_epoch`.
        drop_last: drop the final incomplete batch.
        weights: optional per-sample weights. ``None`` -> a permutation (every
            index exactly once per epoch); otherwise ``num_samples`` draws with
            replacement proportional to the weights (e.g. ×2 for small faces /
            crowded images).
        shuffle: ``False`` -> sequential order (evaluation); sizes still drawn
            per batch (pass a single size for deterministic evaluation).
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        sizes: Sequence[int],
        size_probs: Sequence[float] | None = None,
        seed: int = 0,
        drop_last: bool = True,
        *,
        weights: Sequence[float] | np.ndarray | None = None,
        shuffle: bool = True,
    ) -> None:
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive")
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.sizes = tuple(int(s) for s in sizes)
        if not self.sizes:
            raise ValueError("sizes must not be empty")
        probs = np.ones(len(self.sizes)) if size_probs is None else np.asarray(size_probs, dtype=np.float64)
        if probs.shape != (len(self.sizes),) or (probs < 0).any() or probs.sum() <= 0:
            raise ValueError(f"size_probs {size_probs} do not match sizes {self.sizes}")
        self.size_probs = probs / probs.sum()
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.weights: np.ndarray | None = None
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (self.num_samples,) or (w < 0).any() or w.sum() <= 0:
                raise ValueError("weights must be a non-negative vector of length num_samples")
            self.weights = w / w.sum()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _order(self, rng: np.random.Generator) -> np.ndarray:
        if not self.shuffle:
            return np.arange(self.num_samples)
        if self.weights is None:
            return rng.permutation(self.num_samples)
        return rng.choice(self.num_samples, size=self.num_samples, replace=True, p=self.weights)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        order = self._order(rng)
        n_batches = len(self)
        batch_sizes = rng.choice(len(self.sizes), size=n_batches, p=self.size_probs)
        for b in range(n_batches):
            chunk = order[b * self.batch_size : (b + 1) * self.batch_size]
            size = self.sizes[int(batch_sizes[b])]
            yield [(int(i), size) for i in chunk]

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size
