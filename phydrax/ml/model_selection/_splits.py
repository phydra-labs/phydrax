#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from ..._strict import StrictModule
from ...data_utils._splits import kfold_indices
from .._batch import MLBatch
from .._contracts import GradientContract


_SPLIT_GRADIENT_CONTRACT = GradientContract(
    prediction_inputs="none",
    prediction_parameters="none",
    fit_features="none",
    fit_targets="none",
    fit_weights="none",
    fit_hyperparameters="none",
    fit_mode="stopped",
    nondifferentiable_outputs=("train_indices", "validation_indices"),
    conditions=("Split membership is a discrete, stopped choice.",),
)


def _require_key(key: Any, /) -> Any:
    if key is None:
        raise ValueError("An explicit JAX key is required.")
    jr.key_data(key)
    return key


def _sample_complement(num_samples: int, validation: Array, /) -> Array:
    keep = jnp.ones((num_samples,), dtype=bool).at[validation].set(False)
    return jnp.nonzero(keep)[0].astype(jnp.int32)


def _validate_fold_indices(
    num_samples: int, train_indices: Any, validation_indices: Any, /
) -> tuple[Array, Array]:
    train = jnp.asarray(train_indices, dtype=jnp.int32)
    validation = jnp.asarray(validation_indices, dtype=jnp.int32)
    if train.ndim != 1 or validation.ndim != 1:
        raise ValueError("Fold indices must be one-dimensional.")
    if train.size == 0 or validation.size == 0:
        raise ValueError("Every fold requires non-empty training and validation sets.")
    if bool(jnp.any((train < 0) | (train >= num_samples))) or bool(
        jnp.any((validation < 0) | (validation >= num_samples))
    ):
        raise ValueError("Fold indices lie outside the batch sample axis.")
    if int(jnp.unique(train).size) != int(train.size) or int(
        jnp.unique(validation).size
    ) != int(validation.size):
        raise ValueError("Fold indices must not contain duplicates.")
    if bool(jnp.any(jnp.isin(train, validation))):
        raise ValueError("Training and validation indices must be disjoint.")
    return train, validation


class FoldRecord(StrictModule):
    """One immutable, discrete training/validation split."""

    train_indices: Array
    validation_indices: Array
    fold_id: int = eqx.field(static=True)

    def __init__(
        self,
        train_indices: Any,
        validation_indices: Any,
        /,
        *,
        fold_id: int,
        num_samples: int,
    ):
        train, validation = _validate_fold_indices(
            int(num_samples), train_indices, validation_indices
        )
        self.train_indices = train
        self.validation_indices = validation
        self.fold_id = int(fold_id)


class SplitPlanResult(StrictModule):
    """Materialized folds and the sample universe from which they were made."""

    folds: tuple[FoldRecord, ...]
    sample_indices: Array
    key: Any
    gradient_contract: GradientContract
    method: str = eqx.field(static=True)

    def __init__(
        self,
        folds: tuple[FoldRecord, ...],
        /,
        *,
        sample_indices: Any,
        key: Any,
        method: str,
    ):
        if not folds:
            raise ValueError("A split result requires at least one fold.")
        if any(not isinstance(fold, FoldRecord) for fold in folds):
            raise TypeError("folds must contain only FoldRecord objects.")
        samples = jnp.asarray(sample_indices, dtype=jnp.int32)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("sample_indices must be a non-empty one-dimensional array.")
        if bool(jnp.any(samples < 0)):
            raise ValueError("sample_indices must be non-negative.")
        if int(jnp.unique(samples).size) != int(samples.size):
            raise ValueError("sample_indices must not contain duplicates.")
        for fold in folds:
            if bool(jnp.any(~jnp.isin(fold.train_indices, samples))) or bool(
                jnp.any(~jnp.isin(fold.validation_indices, samples))
            ):
                raise ValueError("Every fold index must belong to sample_indices.")
        self.folds = tuple(folds)
        self.sample_indices = samples
        self.key = _require_key(key)
        self.gradient_contract = _SPLIT_GRADIENT_CONTRACT
        self.method = str(method)


def _validate_split_result_for_batch(
    split_result: SplitPlanResult, batch: MLBatch, /
) -> None:
    if not isinstance(split_result, SplitPlanResult):
        raise TypeError("split_result must be a SplitPlanResult.")
    if bool(jnp.any(split_result.sample_indices >= batch.sample_count)):
        raise ValueError("split_result sample_indices exceed the batch sample axis.")
    for fold in split_result.folds:
        if bool(jnp.any(fold.train_indices >= batch.sample_count)) or bool(
            jnp.any(fold.validation_indices >= batch.sample_count)
        ):
            raise ValueError("A fold index exceeds the batch sample axis.")


class AbstractSplitPlan(StrictModule):
    """Immutable recipe for discrete sample-axis folds."""

    @abstractmethod
    def split(self, batch: MLBatch, /, *, key: Any) -> SplitPlanResult:
        raise NotImplementedError


def _fold_result(
    pairs: tuple[tuple[Array, Array], ...],
    batch: MLBatch,
    key: Any,
    method: str,
    /,
) -> SplitPlanResult:
    folds = tuple(
        FoldRecord(train, validation, fold_id=i, num_samples=batch.sample_count)
        for i, (train, validation) in enumerate(pairs)
    )
    return SplitPlanResult(
        folds,
        sample_indices=jnp.arange(batch.sample_count, dtype=jnp.int32),
        key=key,
        method=method,
    )


class KFoldPlan(AbstractSplitPlan):
    """Ordinary K-fold splitting backed by :mod:`phydrax.data_utils` primitives."""

    num_folds: int = eqx.field(static=True)
    shuffle: bool = eqx.field(static=True)

    def __init__(self, num_folds: int = 5, /, *, shuffle: bool = True):
        if int(num_folds) < 2:
            raise ValueError("num_folds must be at least 2.")
        self.num_folds = int(num_folds)
        self.shuffle = bool(shuffle)

    def split(self, batch: MLBatch, /, *, key: Any) -> SplitPlanResult:
        key = _require_key(key)
        pairs = kfold_indices(
            batch.sample_count,
            self.num_folds,
            key=key,
            shuffle=self.shuffle,
        )
        return _fold_result(pairs, batch, key, "kfold")


def _shared_labels(batch: MLBatch, /) -> Array:
    targets = batch.require_targets()
    if batch.target_shape == (1,):
        targets = jnp.squeeze(targets, axis=-1)
    elif batch.target_shape != ():
        raise ValueError(
            "Stratified folds require one scalar class label per case/sample."
        )
    if not (
        jnp.issubdtype(targets.dtype, jnp.number)
        or jnp.issubdtype(targets.dtype, jnp.bool_)
    ) or jnp.issubdtype(targets.dtype, jnp.complexfloating):
        raise TypeError("Stratification labels must be real numeric or boolean values.")
    if not bool(jnp.all(jnp.isfinite(targets))):
        raise ValueError("Stratification labels must be finite.")
    labels = jnp.reshape(targets, (-1, batch.sample_count))
    reference = labels[0]
    if not bool(jnp.all(labels == reference)):
        raise ValueError(
            "Case-dependent labels cannot define one shared sample-axis stratification."
        )
    if batch.target_mask is not None and not bool(jnp.all(batch.target_mask)):
        raise ValueError("Stratified folds require every class label to be valid.")
    return reference


class StratifiedKFoldPlan(AbstractSplitPlan):
    """K folds with deterministic round-robin allocation within each class."""

    num_folds: int = eqx.field(static=True)
    shuffle: bool = eqx.field(static=True)

    def __init__(self, num_folds: int = 5, /, *, shuffle: bool = True):
        if int(num_folds) < 2:
            raise ValueError("num_folds must be at least 2.")
        self.num_folds = int(num_folds)
        self.shuffle = bool(shuffle)

    def split(self, batch: MLBatch, /, *, key: Any) -> SplitPlanResult:
        key = _require_key(key)
        n = batch.sample_count
        if self.num_folds > n:
            raise ValueError("num_folds cannot exceed the batch sample count.")
        labels = _shared_labels(batch)
        classes = jnp.unique(labels)
        validation_parts: list[list[Array]] = [list() for _ in range(self.num_folds)]
        offset = 0
        for class_id in range(int(classes.size)):
            indices = jnp.nonzero(labels == classes[class_id])[0].astype(jnp.int32)
            if self.shuffle:
                indices = jr.permutation(jr.fold_in(key, class_id), indices)
            count = int(indices.size)
            for local_index in range(count):
                fold_id = (offset + local_index) % self.num_folds
                validation_parts[fold_id].append(indices[local_index : local_index + 1])
            offset = (offset + count) % self.num_folds
        pairs: list[tuple[Array, Array]] = []
        for fold_id, parts in enumerate(validation_parts):
            if not parts:
                raise ValueError("Stratification produced an empty validation fold.")
            validation = jnp.concatenate(tuple(parts), axis=0)
            if self.shuffle:
                validation = jr.permutation(
                    jr.fold_in(key, self.num_folds + fold_id), validation
                )
            pairs.append((_sample_complement(n, validation), validation))
        return _fold_result(tuple(pairs), batch, key, "stratified_kfold")


def _shared_groups(batch: MLBatch, /) -> Array:
    if batch.groups is None:
        raise ValueError("Group folds require batch.groups.")
    groups = jnp.reshape(batch.groups, (-1, batch.sample_count))
    reference = groups[0]
    if not bool(jnp.all(groups == reference)):
        raise ValueError(
            "Case-dependent groups cannot define one shared sample-axis group split."
        )
    return reference


class GroupKFoldPlan(AbstractSplitPlan):
    """Group-exclusive folds, greedily balanced by validation sample count."""

    num_folds: int = eqx.field(static=True)
    shuffle: bool = eqx.field(static=True)

    def __init__(self, num_folds: int = 5, /, *, shuffle: bool = False):
        if int(num_folds) < 2:
            raise ValueError("num_folds must be at least 2.")
        self.num_folds = int(num_folds)
        self.shuffle = bool(shuffle)

    def split(self, batch: MLBatch, /, *, key: Any) -> SplitPlanResult:
        key = _require_key(key)
        groups = _shared_groups(batch)
        unique_groups = jnp.unique(groups)
        if int(unique_groups.size) < self.num_folds:
            raise ValueError("num_folds cannot exceed the number of distinct groups.")
        if self.shuffle:
            unique_groups = jr.permutation(key, unique_groups)
        counts = jnp.stack([jnp.sum(groups == group) for group in unique_groups])
        order = jnp.argsort(-counts, stable=True)
        fold_groups: list[list[Array]] = [list() for _ in range(self.num_folds)]
        fold_sizes = [0] * self.num_folds
        for position in range(int(order.size)):
            group_position = int(order[position])
            fold_id = min(range(self.num_folds), key=lambda i: (fold_sizes[i], i))
            fold_groups[fold_id].append(
                unique_groups[group_position : group_position + 1]
            )
            fold_sizes[fold_id] += int(counts[group_position])
        pairs: list[tuple[Array, Array]] = []
        for assigned in fold_groups:
            held_groups = jnp.concatenate(tuple(assigned), axis=0)
            validation = jnp.nonzero(jnp.isin(groups, held_groups))[0].astype(jnp.int32)
            pairs.append((_sample_complement(batch.sample_count, validation), validation))
        return _fold_result(tuple(pairs), batch, key, "group_kfold")


class TimeSeriesSplitPlan(AbstractSplitPlan):
    """Expanding-window time splits with a purged gap before validation."""

    num_folds: int = eqx.field(static=True)
    validation_size: int | None = eqx.field(static=True)
    min_train_size: int | None = eqx.field(static=True)
    gap: int = eqx.field(static=True)

    def __init__(
        self,
        num_folds: int = 5,
        /,
        *,
        validation_size: int | None = None,
        min_train_size: int | None = None,
        gap: int = 0,
    ):
        if int(num_folds) < 1:
            raise ValueError("num_folds must be positive.")
        if validation_size is not None and int(validation_size) < 1:
            raise ValueError("validation_size must be positive when provided.")
        if min_train_size is not None and int(min_train_size) < 1:
            raise ValueError("min_train_size must be positive when provided.")
        if int(gap) < 0:
            raise ValueError("gap must be non-negative.")
        self.num_folds = int(num_folds)
        self.validation_size = None if validation_size is None else int(validation_size)
        self.min_train_size = None if min_train_size is None else int(min_train_size)
        self.gap = int(gap)

    def split(self, batch: MLBatch, /, *, key: Any) -> SplitPlanResult:
        key = _require_key(key)
        n = batch.sample_count
        validation_size = (
            max(1, n // (self.num_folds + 1))
            if self.validation_size is None
            else self.validation_size
        )
        min_train = (
            n - self.num_folds * validation_size - self.gap
            if self.min_train_size is None
            else self.min_train_size
        )
        if min_train < 1:
            raise ValueError("The requested time splits leave no training samples.")
        required = min_train + self.gap + self.num_folds * validation_size
        if required > n:
            raise ValueError("The requested time split windows exceed the sample axis.")
        pairs: list[tuple[Array, Array]] = []
        for fold_id in range(self.num_folds):
            train_stop = min_train + fold_id * validation_size
            validation_start = train_stop + self.gap
            validation_stop = validation_start + validation_size
            train = jnp.arange(train_stop, dtype=jnp.int32)
            validation = jnp.arange(validation_start, validation_stop, dtype=jnp.int32)
            pairs.append((train, validation))
        return _fold_result(tuple(pairs), batch, key, "time_series")


class BlockSplitPlan(AbstractSplitPlan):
    """Contiguous validation blocks with symmetric purging from training folds."""

    num_folds: int = eqx.field(static=True)
    gap: int = eqx.field(static=True)

    def __init__(self, num_folds: int = 5, /, *, gap: int = 0):
        if int(num_folds) < 2:
            raise ValueError("num_folds must be at least 2.")
        if int(gap) < 0:
            raise ValueError("gap must be non-negative.")
        self.num_folds = int(num_folds)
        self.gap = int(gap)

    def split(self, batch: MLBatch, /, *, key: Any) -> SplitPlanResult:
        key = _require_key(key)
        base_pairs = kfold_indices(
            batch.sample_count, self.num_folds, key=key, shuffle=False
        )
        pairs: list[tuple[Array, Array]] = []
        all_indices = jnp.arange(batch.sample_count, dtype=jnp.int32)
        for _, validation in base_pairs:
            start = int(validation[0])
            stop = int(validation[-1]) + 1
            keep = (all_indices < max(0, start - self.gap)) | (
                all_indices >= min(batch.sample_count, stop + self.gap)
            )
            train = jnp.nonzero(keep)[0].astype(jnp.int32)
            if train.size == 0:
                raise ValueError("gap removes every training sample from a block fold.")
            pairs.append((train, validation))
        return _fold_result(tuple(pairs), batch, key, "blocked")


class RollingWindowSplitPlan(AbstractSplitPlan):
    """Fixed-width rolling training windows followed by held-out windows."""

    train_size: int = eqx.field(static=True)
    validation_size: int = eqx.field(static=True)
    step: int = eqx.field(static=True)
    gap: int = eqx.field(static=True)
    max_folds: int | None = eqx.field(static=True)

    def __init__(
        self,
        train_size: int,
        validation_size: int,
        /,
        *,
        step: int | None = None,
        gap: int = 0,
        max_folds: int | None = None,
    ):
        if int(train_size) < 1 or int(validation_size) < 1:
            raise ValueError("train_size and validation_size must be positive.")
        step_ = int(validation_size) if step is None else int(step)
        if step_ < 1:
            raise ValueError("step must be positive.")
        if int(gap) < 0:
            raise ValueError("gap must be non-negative.")
        if max_folds is not None and int(max_folds) < 1:
            raise ValueError("max_folds must be positive when provided.")
        self.train_size = int(train_size)
        self.validation_size = int(validation_size)
        self.step = step_
        self.gap = int(gap)
        self.max_folds = None if max_folds is None else int(max_folds)

    def split(self, batch: MLBatch, /, *, key: Any) -> SplitPlanResult:
        key = _require_key(key)
        pairs: list[tuple[Array, Array]] = []
        start = 0
        while True:
            train_stop = start + self.train_size
            validation_start = train_stop + self.gap
            validation_stop = validation_start + self.validation_size
            if validation_stop > batch.sample_count:
                break
            pairs.append(
                (
                    jnp.arange(start, train_stop, dtype=jnp.int32),
                    jnp.arange(validation_start, validation_stop, dtype=jnp.int32),
                )
            )
            if self.max_folds is not None and len(pairs) >= self.max_folds:
                break
            start += self.step
        if not pairs:
            raise ValueError("The rolling windows do not fit on the batch sample axis.")
        return _fold_result(tuple(pairs), batch, key, "rolling_window")


class NestedFoldRecord(StrictModule):
    """One outer fold and inner folds expressed in original-batch indices."""

    outer_fold: FoldRecord
    inner_split: SplitPlanResult

    def __init__(self, outer_fold: FoldRecord, inner_split: SplitPlanResult, /):
        if not isinstance(outer_fold, FoldRecord):
            raise TypeError("outer_fold must be a FoldRecord.")
        if not isinstance(inner_split, SplitPlanResult):
            raise TypeError("inner_split must be a SplitPlanResult.")
        if bool(
            jnp.any(jnp.isin(inner_split.sample_indices, outer_fold.validation_indices))
        ):
            raise ValueError("Inner folds must exclude the outer validation samples.")
        self.outer_fold = outer_fold
        self.inner_split = inner_split


class NestedSplitResult(StrictModule):
    """Materialized outer folds and their leakage-safe inner folds."""

    folds: tuple[NestedFoldRecord, ...]
    key: Any
    gradient_contract: GradientContract
    method: str = eqx.field(static=True)

    def __init__(self, folds: tuple[NestedFoldRecord, ...], /, *, key: Any):
        if not folds:
            raise ValueError("A nested split requires at least one outer fold.")
        if any(not isinstance(fold, NestedFoldRecord) for fold in folds):
            raise TypeError("folds must contain only NestedFoldRecord objects.")
        self.folds = tuple(folds)
        self.key = _require_key(key)
        self.gradient_contract = _SPLIT_GRADIENT_CONTRACT
        self.method = "nested"


class NestedSplitPlan(StrictModule):
    """Compose arbitrary outer and inner plans without exposing outer holdouts."""

    outer_plan: AbstractSplitPlan
    inner_plan: AbstractSplitPlan

    def __init__(self, outer_plan: AbstractSplitPlan, inner_plan: AbstractSplitPlan, /):
        if not isinstance(outer_plan, AbstractSplitPlan) or not isinstance(
            inner_plan, AbstractSplitPlan
        ):
            raise TypeError("outer_plan and inner_plan must be split plans.")
        self.outer_plan = outer_plan
        self.inner_plan = inner_plan

    def split(self, batch: MLBatch, /, *, key: Any) -> NestedSplitResult:
        key = _require_key(key)
        outer = self.outer_plan.split(batch, key=jr.fold_in(key, 0))
        nested: list[NestedFoldRecord] = []
        for outer_position, outer_fold in enumerate(outer.folds):
            outer_train = batch.take_samples(outer_fold.train_indices)
            local_inner = self.inner_plan.split(
                outer_train, key=jr.fold_in(key, outer_position + 1)
            )
            global_folds = tuple(
                FoldRecord(
                    jnp.take(outer_fold.train_indices, fold.train_indices),
                    jnp.take(outer_fold.train_indices, fold.validation_indices),
                    fold_id=fold.fold_id,
                    num_samples=batch.sample_count,
                )
                for fold in local_inner.folds
            )
            global_inner = SplitPlanResult(
                global_folds,
                sample_indices=outer_fold.train_indices,
                key=local_inner.key,
                method=f"nested_{local_inner.method}",
            )
            nested.append(NestedFoldRecord(outer_fold, global_inner))
        return NestedSplitResult(tuple(nested), key=key)


__all__ = [
    "AbstractSplitPlan",
    "BlockSplitPlan",
    "FoldRecord",
    "GroupKFoldPlan",
    "KFoldPlan",
    "NestedFoldRecord",
    "NestedSplitPlan",
    "NestedSplitResult",
    "RollingWindowSplitPlan",
    "SplitPlanResult",
    "StratifiedKFoldPlan",
    "TimeSeriesSplitPlan",
]
