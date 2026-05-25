#
#  Copyright 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._doc import DOC_KEY0


def _permutation(
    num_cases: int,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    shuffle: bool = True,
) -> Array:
    n = int(num_cases)
    if n <= 0:
        raise ValueError("num_cases must be positive.")
    indices = jnp.arange(n, dtype=jnp.int32)
    if not bool(shuffle):
        return indices
    return jr.permutation(key, indices)


def train_test_split_indices(
    num_cases: int,
    /,
    *,
    test_fraction: float = 0.2,
    key: Key[Array, ""] = DOC_KEY0,
    shuffle: bool = True,
) -> tuple[Array, Array]:
    """Return `(train_indices, test_indices)` for finite empirical cases."""
    n = int(num_cases)
    if n < 2:
        raise ValueError("num_cases must be at least 2 for a train/test split.")
    fraction = float(test_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1.")

    perm = _permutation(n, key=key, shuffle=shuffle)
    n_test = int(round(float(n) * fraction))
    n_test = min(max(n_test, 1), n - 1)
    return perm[n_test:], perm[:n_test]


def kfold_indices(
    num_cases: int,
    num_folds: int,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    shuffle: bool = True,
) -> tuple[tuple[Array, Array], ...]:
    """Return `(train_indices, validation_indices)` pairs for K-fold splits."""
    n = int(num_cases)
    k = int(num_folds)
    if n < 2:
        raise ValueError("num_cases must be at least 2 for K-fold splits.")
    if k < 2:
        raise ValueError("num_folds must be at least 2.")
    if k > n:
        raise ValueError("num_folds cannot exceed num_cases.")

    perm = _permutation(n, key=key, shuffle=shuffle)
    base = n // k
    remainder = n % k
    folds: list[tuple[Array, Array]] = []
    offset = 0
    for fold in range(k):
        size = base + (1 if fold < remainder else 0)
        start = offset
        stop = offset + size
        validation = perm[start:stop]
        train = jnp.concatenate((perm[:start], perm[stop:]), axis=0)
        folds.append((train, validation))
        offset = stop
    return tuple(folds)


__all__ = [
    "kfold_indices",
    "train_test_split_indices",
]
