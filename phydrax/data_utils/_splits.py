#
#  Copyright 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import jax.random as jr
import numpy as np
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


def train_calibration_test_split_indices(
    num_cases: int,
    /,
    *,
    calibration_fraction: float = 0.2,
    test_fraction: float = 0.2,
    key: Key[Array, ""] = DOC_KEY0,
    shuffle: bool = True,
) -> tuple[Array, Array, Array]:
    """Return disjoint non-empty train, calibration, and test case indices."""
    n = int(num_cases)
    if n < 3:
        raise ValueError("num_cases must be at least 3 for a three-way split.")
    calibration = float(calibration_fraction)
    test = float(test_fraction)
    if calibration <= 0.0 or test <= 0.0 or calibration + test >= 1.0:
        raise ValueError(
            "calibration_fraction and test_fraction must be positive and sum to less than one."
        )
    n_calibration = max(1, int(round(n * calibration)))
    n_test = max(1, int(round(n * test)))
    while n_calibration + n_test > n - 1:
        if n_test >= n_calibration and n_test > 1:
            n_test -= 1
        elif n_calibration > 1:
            n_calibration -= 1
        else:
            raise ValueError("Fractions leave no case for the training split.")
    permutation = _permutation(n, key=key, shuffle=shuffle)
    calibration_indices = permutation[:n_calibration]
    test_indices = permutation[n_calibration : n_calibration + n_test]
    train_indices = permutation[n_calibration + n_test :]
    return train_indices, calibration_indices, test_indices


def grouped_train_validation_test_split_indices(
    components: Sequence[Sequence[int]],
    num_cases: int,
    /,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split whole case components into non-empty train, validation, and test sets."""
    n = int(num_cases)
    if n < 3:
        raise ValueError("num_cases must be at least 3 for a three-way split.")
    train = float(train_fraction)
    validation = float(validation_fraction)
    if not 0.0 < train < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one.")
    if not 0.0 < validation < 1.0 or train + validation >= 1.0:
        raise ValueError("validation_fraction must leave a non-empty test fraction.")
    groups = [tuple(int(index) for index in component) for component in components]
    if len(groups) < 3 or any(not component for component in groups):
        raise ValueError(
            "A grouped train/validation/test split requires at least three non-empty groups."
        )
    flattened = tuple(index for component in groups for index in component)
    if len(flattened) != n or set(flattened) != set(range(n)):
        raise ValueError("components must partition every case index exactly once.")
    if shuffle:
        permutation = np.random.default_rng(int(seed)).permutation(len(groups))
        groups = [groups[int(index)] for index in permutation]
    cumulative = np.cumsum([len(component) for component in groups])
    train_goal = float(n) * train
    validation_goal = float(n) * (train + validation)
    train_cut = min(
        range(1, len(groups) - 1),
        key=lambda cut: abs(float(cumulative[cut - 1]) - train_goal),
    )
    validation_cut = min(
        range(train_cut + 1, len(groups)),
        key=lambda cut: abs(float(cumulative[cut - 1]) - validation_goal),
    )
    train_indices = tuple(
        index for component in groups[:train_cut] for index in component
    )
    validation_indices = tuple(
        index for component in groups[train_cut:validation_cut] for index in component
    )
    test_indices = tuple(
        index for component in groups[validation_cut:] for index in component
    )
    return train_indices, validation_indices, test_indices


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
    "grouped_train_validation_test_split_indices",
    "kfold_indices",
    "train_calibration_test_split_indices",
    "train_test_split_indices",
]
