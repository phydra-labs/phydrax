#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..data import (
    FunctionSamples,
    OperatorAxis,
    OperatorBatch,
    OperatorCaseProvenance,
    OperatorFieldBatch,
    OperatorOutputSpec,
    OperatorTargetBatch,
    slice_operator_batch,
    stack_operator_batches,
)


@dataclass(frozen=True)
class OperatorSplitPolicy:
    """Random grouped or chronological leakage-safe split policy."""

    group_by: tuple[str, ...] | Literal["all"] = "all"
    order_by: str | None = None
    seed: int = 0

    def __post_init__(self):
        if self.group_by != "all":
            keys = tuple(str(key) for key in self.group_by)
            if len(set(keys)) != len(keys) or any(not key for key in keys):
                raise ValueError("group_by keys must be unique and non-empty.")
            object.__setattr__(self, "group_by", keys)
        order = None if self.order_by is None else str(self.order_by)
        if order == "":
            raise ValueError("order_by must be non-empty when provided.")
        object.__setattr__(self, "order_by", order)
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class OperatorDataset:
    """Case-indexed operator inputs and supervised query targets."""

    batch: OperatorBatch
    targets: OperatorTargetBatch

    provenance: tuple[OperatorCaseProvenance, ...] | None = None

    def __post_init__(self):
        if len(self.batch.case_shape) != 1:
            raise ValueError("OperatorDataset requires exactly one case axis.")
        if not isinstance(self.targets, OperatorTargetBatch):
            raise TypeError("OperatorDataset targets must be an OperatorTargetBatch.")
        self.targets.validate(self.batch)

        provenance = (
            tuple(
                OperatorCaseProvenance(f"case:{index}")
                for index in range(self.batch.case_shape[0])
            )
            if self.provenance is None
            else tuple(self.provenance)
        )
        if len(provenance) != self.batch.case_shape[0]:
            raise ValueError("Dataset provenance must contain one record per case.")
        if any(not isinstance(record, OperatorCaseProvenance) for record in provenance):
            raise TypeError("Dataset provenance entries must be OperatorCaseProvenance.")
        case_ids = tuple(record.case_id for record in provenance)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Dataset provenance case IDs must be unique.")
        object.__setattr__(self, "provenance", provenance)

    @property
    def size(self) -> int:
        return self.batch.case_shape[0]

    def take(self, indices: Any, /) -> "OperatorDataset":
        index = jnp.asarray(indices, dtype=jnp.int32)
        if index.ndim != 1:
            raise ValueError("Dataset indices must be one-dimensional.")
        assert self.provenance is not None
        return OperatorDataset(
            slice_operator_batch(self.batch, index, axis=0),
            self.targets.take(index, axis=0),
            tuple(self.provenance[int(position)] for position in np.asarray(index)),
        )


@dataclass(frozen=True)
class OperatorDatasetSplit:
    """Deterministic train, validation, and test partitions."""

    train: OperatorDataset
    validation: OperatorDataset
    test: OperatorDataset
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    policy: OperatorSplitPolicy
    group_keys: tuple[str, ...]

    @property
    def seed(self) -> int:
        return self.policy.seed


def _provenance_components(
    provenance: tuple[OperatorCaseProvenance, ...],
    keys: tuple[str, ...],
    /,
) -> tuple[tuple[int, ...], ...]:
    parent = list(range(len(provenance)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for key in keys:
        owners: dict[str, int] = {}
        for index, record in enumerate(provenance):
            if key not in record.identities:
                continue
            value = record.identities[key]
            if value in owners:
                union(owners[value], index)
            else:
                owners[value] = index
    components: dict[int, list[int]] = {}
    for index in range(len(provenance)):
        components.setdefault(root(index), []).append(index)
    return tuple(
        tuple(component)
        for _, component in sorted(
            components.items(),
            key=lambda item: min(item[1]),
        )
    )


def _split_component_boundaries(
    components: Sequence[Sequence[int]],
    size: int,
    train_fraction: float,
    validation_fraction: float,
    /,
) -> tuple[int, int]:
    if len(components) < 3:
        raise ValueError(
            "A provenance-safe train/validation/test split requires at least "
            "three independent groups."
        )
    cumulative = np.cumsum([len(component) for component in components])
    train_goal = float(size) * train_fraction
    validation_goal = float(size) * (train_fraction + validation_fraction)
    train_cut = min(
        range(1, len(components) - 1),
        key=lambda cut: abs(float(cumulative[cut - 1]) - train_goal),
    )
    validation_cut = min(
        range(train_cut + 1, len(components)),
        key=lambda cut: abs(float(cumulative[cut - 1]) - validation_goal),
    )
    return train_cut, validation_cut


def split_operator_dataset(
    dataset: OperatorDataset,
    /,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    policy: OperatorSplitPolicy | None = None,
) -> OperatorDatasetSplit:
    """Split whole provenance components without identity or temporal leakage."""
    train = float(train_fraction)
    validation = float(validation_fraction)
    if not 0.0 < train < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one.")
    if not 0.0 <= validation < 1.0 or train + validation >= 1.0:
        raise ValueError("validation_fraction must leave a non-empty test fraction.")
    if dataset.size < 3:
        raise ValueError("A train/validation/test split requires at least three cases.")
    resolved = OperatorSplitPolicy() if policy is None else policy
    if not isinstance(resolved, OperatorSplitPolicy):
        raise TypeError("policy must be an OperatorSplitPolicy.")
    assert dataset.provenance is not None
    if resolved.group_by == "all":
        group_keys = tuple(
            sorted({key for record in dataset.provenance for key in record.identities})
        )
    else:
        group_keys = resolved.group_by
        missing = {
            key
            for key in group_keys
            if any(key not in record.identities for record in dataset.provenance)
        }
        if missing:
            raise ValueError(
                f"Every case must define requested provenance identities {tuple(sorted(missing))}."
            )
    components = list(_provenance_components(dataset.provenance, group_keys))
    if resolved.order_by is None:
        permutation = np.random.default_rng(resolved.seed).permutation(len(components))
        components = [components[int(index)] for index in permutation]
    else:
        order_name = resolved.order_by
        if any(order_name not in record.order for record in dataset.provenance):
            raise ValueError(
                f"Every case must define provenance order coordinate {order_name!r}."
            )
        intervals = [
            (
                min(dataset.provenance[index].order[order_name] for index in component),
                max(dataset.provenance[index].order[order_name] for index in component),
                component,
            )
            for component in components
        ]
        intervals.sort(key=lambda item: (item[0], item[1]))
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError(
                "Grouped provenance intervals overlap on the requested order axis; "
                "a leakage-free chronological split is impossible."
            )
        components = [item[2] for item in intervals]
    train_cut, validation_cut = _split_component_boundaries(
        components,
        dataset.size,
        train,
        validation,
    )
    train_indices = tuple(
        index for component in components[:train_cut] for index in component
    )
    validation_indices = tuple(
        index for component in components[train_cut:validation_cut] for index in component
    )
    test_indices = tuple(
        index for component in components[validation_cut:] for index in component
    )
    return OperatorDatasetSplit(
        train=dataset.take(train_indices),
        validation=dataset.take(validation_indices),
        test=dataset.take(test_indices),
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        policy=resolved,
        group_keys=group_keys,
    )


def operator_dataset_from_arrays(
    inputs: Mapping[str, Any],
    targets: Mapping[str, Any],
    /,
    *,
    source_axes: Mapping[str, Sequence[OperatorAxis]],
    query_axes: Sequence[OperatorAxis] = (),
    query_coordinates: Array | None = None,
    query_quadrature_weights: Array | None = None,
    query_mask: Array | None = None,
    target_queries: Mapping[str, str] | None = None,
    target_specs: Mapping[str, OperatorOutputSpec] | None = None,
    provenance: Sequence[OperatorCaseProvenance] | None = None,
    case_axis: str = "case",
) -> OperatorDataset:
    """Adapt dense case-first arrays to the canonical operator protocol."""
    if not inputs:
        raise ValueError("inputs must not be empty.")
    if tuple(inputs) != tuple(source_axes):
        raise ValueError("source_axes must define each input in the same order.")
    samples = {
        name: FunctionSamples(values=jnp.asarray(value), axes=tuple(source_axes[name]))
        for name, value in inputs.items()
    }
    sizes = {int(jnp.asarray(value).shape[0]) for value in inputs.values()}
    if len(sizes) != 1:
        raise ValueError("All operator inputs must have the same case count.")
    size = sizes.pop()
    query = FunctionSamples(
        values=None,
        axes=tuple(query_axes),
        coordinates=query_coordinates,
        quadrature_weights=query_quadrature_weights,
        mask=query_mask,
    )
    batch = OperatorBatch(
        inputs=samples,
        queries={"query": query},
        case_axes=(str(case_axis),),
        case_shape=(size,),
    )
    target_batch = OperatorTargetBatch.from_arrays(
        {name: jnp.asarray(value) for name, value in targets.items()},
        batch,
        query_names=target_queries,
        specs=target_specs,
    )
    return OperatorDataset(
        batch,
        target_batch,
        None if provenance is None else tuple(provenance),
    )


def _pad_target(target: Array, current: int, size: int, case_ndim: int) -> Array:
    if current == size:
        return target
    padding = [(0, 0)] * target.ndim
    padding[case_ndim] = (0, size - current)
    return jnp.pad(target, padding)


def operator_dataset_from_cases(
    batches: Sequence[OperatorBatch],
    targets: Sequence[OperatorTargetBatch],
    /,
    *,
    case_axis: str = "case",
    provenance: Sequence[OperatorCaseProvenance] | None = None,
) -> OperatorDataset:
    """Collate variable-cardinality point-cloud cases with mask-safe padding."""
    batch_tuple = tuple(batches)
    target_tuple = tuple(targets)
    if not batch_tuple or len(batch_tuple) != len(target_tuple):
        raise ValueError("batches and targets must be non-empty and have equal length.")
    first_shape = batch_tuple[0].case_shape
    if any(batch.case_shape != first_shape for batch in batch_tuple[1:]):
        raise ValueError("Every case batch must have the same existing case shape.")
    for batch, target in zip(batch_tuple, target_tuple, strict=True):
        target.validate(batch)
    target_names = tuple(target_tuple[0].fields)
    if any(set(target.fields) != set(target_names) for target in target_tuple[1:]):
        raise ValueError("Every case must define the same target fields.")
    stacked_batch = stack_operator_batches(batch_tuple, case_axis=case_axis)
    stacked_fields: dict[str, OperatorFieldBatch] = {}
    for name in target_names:
        fields = tuple(target.field(name) for target in target_tuple)
        first_field = fields[0]
        if any(field.query_name != first_field.query_name for field in fields[1:]):
            raise ValueError(f"Target field {name!r} must use one query branch.")
        if any(
            field.spec.channels != first_field.spec.channels
            or field.spec.component_names != first_field.spec.component_names
            for field in fields[1:]
        ):
            raise ValueError(f"Target field {name!r} must use one output contract.")
        sample_shapes = tuple(
            batch.query(field.query_name).sample_shape
            for batch, field in zip(batch_tuple, fields, strict=True)
        )
        if all(shape == sample_shapes[0] for shape in sample_shapes):
            values = jnp.stack(tuple(field.values for field in fields), axis=0)
        elif all(len(shape) == 1 for shape in sample_shapes):
            maximum = max(shape[0] for shape in sample_shapes)
            values = jnp.stack(
                tuple(
                    _pad_target(field.values, shape[0], maximum, len(first_shape))
                    for field, shape in zip(fields, sample_shapes, strict=True)
                ),
                axis=0,
            )
        else:
            raise ValueError(
                f"Target field {name!r} has incompatible query sample shapes."
            )
        stacked_fields[name] = OperatorFieldBatch(
            values,
            query_name=first_field.query_name,
            spec=first_field.spec,
        )
    stacked_targets = OperatorTargetBatch(
        stacked_fields,
        case_axes=stacked_batch.case_axes,
        case_shape=stacked_batch.case_shape,
    )
    return OperatorDataset(
        stacked_batch,
        stacked_targets,
        None if provenance is None else tuple(provenance),
    )


__all__ = [
    "OperatorDataset",
    "OperatorDatasetSplit",
    "OperatorSplitPolicy",
    "operator_dataset_from_arrays",
    "operator_dataset_from_cases",
    "split_operator_dataset",
]
