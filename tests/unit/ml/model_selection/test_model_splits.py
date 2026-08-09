#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.ml import MLBatch
from phydrax.ml.model_selection import (
    BlockSplitPlan,
    GroupKFoldPlan,
    KFoldPlan,
    NestedSplitPlan,
    RollingWindowSplitPlan,
    StratifiedKFoldPlan,
    TimeSeriesSplitPlan,
)


def _batch(num_samples=12, *, targets=None, groups=None):
    features = jnp.arange(float(num_samples)).reshape(num_samples, 1)
    if targets is None:
        targets = jnp.arange(float(num_samples))
    return MLBatch(features, targets, groups=groups)


def _assert_disjoint(folds):
    for fold in folds:
        assert fold.train_indices.ndim == 1
        assert fold.validation_indices.ndim == 1
        assert not bool(jnp.any(jnp.isin(fold.train_indices, fold.validation_indices)))


def test_kfold_is_keyed_deterministic_and_partitions_the_sample_axis():
    batch = _batch(12)
    plan = KFoldPlan(4, shuffle=True)

    first = plan.split(batch, key=jr.key(7))
    repeated = plan.split(batch, key=jr.key(7))
    different = plan.split(batch, key=jr.key(8))

    _assert_disjoint(first.folds)
    validation = jnp.concatenate(tuple(fold.validation_indices for fold in first.folds))
    assert jnp.array_equal(jnp.sort(validation), jnp.arange(12))
    assert all(
        jnp.array_equal(left.validation_indices, right.validation_indices)
        for left, right in zip(first.folds, repeated.folds, strict=True)
    )
    assert any(
        not jnp.array_equal(left.validation_indices, right.validation_indices)
        for left, right in zip(first.folds, different.folds, strict=True)
    )
    with pytest.raises(ValueError, match="explicit JAX key"):
        plan.split(batch, key=None)


def test_stratified_folds_balance_classes_without_overlap():
    labels = jnp.asarray([0, 1] * 6, dtype=jnp.int32)
    split = StratifiedKFoldPlan(3, shuffle=True).split(
        _batch(12, targets=labels), key=jr.key(11)
    )

    _assert_disjoint(split.folds)
    for fold in split.folds:
        held_labels = labels[fold.validation_indices]
        assert int(jnp.sum(held_labels == 0)) == 2
        assert int(jnp.sum(held_labels == 1)) == 2


def test_group_folds_never_split_one_group_between_train_and_validation():
    groups = jnp.repeat(jnp.arange(6, dtype=jnp.int32), 2)
    split = GroupKFoldPlan(3, shuffle=True).split(
        _batch(12, groups=groups), key=jr.key(12)
    )

    _assert_disjoint(split.folds)
    held_groups = []
    for fold in split.folds:
        train_groups = jnp.unique(groups[fold.train_indices])
        validation_groups = jnp.unique(groups[fold.validation_indices])
        assert not bool(jnp.any(jnp.isin(train_groups, validation_groups)))
        held_groups.append(validation_groups)
    assert jnp.array_equal(jnp.sort(jnp.concatenate(tuple(held_groups))), jnp.arange(6))


def test_time_block_and_rolling_windows_respect_order_and_purged_gaps():
    batch = _batch(15)
    time_split = TimeSeriesSplitPlan(3, validation_size=2, min_train_size=6, gap=1).split(
        batch, key=jr.key(1)
    )
    block_split = BlockSplitPlan(3, gap=1).split(batch, key=jr.key(2))
    rolling_split = RollingWindowSplitPlan(5, 2, step=3, gap=1, max_folds=3).split(
        batch, key=jr.key(3)
    )

    _assert_disjoint(time_split.folds)
    _assert_disjoint(block_split.folds)
    _assert_disjoint(rolling_split.folds)
    for fold in time_split.folds:
        assert int(jnp.max(fold.train_indices)) + 1 < int(
            jnp.min(fold.validation_indices)
        )
    for fold in block_split.folds:
        validation_start = int(jnp.min(fold.validation_indices))
        validation_stop = int(jnp.max(fold.validation_indices))
        assert bool(jnp.all(jnp.diff(fold.validation_indices) == 1))
        assert not bool(
            jnp.any(
                (fold.train_indices >= validation_start - 1)
                & (fold.train_indices <= validation_stop + 1)
            )
        )
    assert jnp.array_equal(
        rolling_split.folds[0].train_indices, jnp.arange(5, dtype=jnp.int32)
    )
    assert jnp.array_equal(
        rolling_split.folds[0].validation_indices,
        jnp.asarray([6, 7], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        rolling_split.folds[1].train_indices,
        jnp.arange(3, 8, dtype=jnp.int32),
    )


def test_nested_inner_folds_are_confined_to_each_outer_training_partition():
    batch = _batch(12)
    nested = NestedSplitPlan(
        KFoldPlan(3, shuffle=True), KFoldPlan(2, shuffle=True)
    ).split(batch, key=jr.key(19))

    for nested_fold in nested.folds:
        outer = nested_fold.outer_fold
        inner = nested_fold.inner_split
        assert jnp.array_equal(
            jnp.sort(inner.sample_indices), jnp.sort(outer.train_indices)
        )
        for fold in inner.folds:
            assert bool(jnp.all(jnp.isin(fold.train_indices, outer.train_indices)))
            assert bool(jnp.all(jnp.isin(fold.validation_indices, outer.train_indices)))
            assert not bool(
                jnp.any(jnp.isin(fold.train_indices, outer.validation_indices))
            )
            assert not bool(
                jnp.any(jnp.isin(fold.validation_indices, outer.validation_indices))
            )
