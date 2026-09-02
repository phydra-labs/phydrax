#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native whole-case and named-query operator coreset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ....coresets import (
    CoresetSelection,
    kernel_herd,
    KernelHerding,
    moment_recombine,
    MomentRecombination,
)
from ..data import (
    FunctionSamples,
    OperatorBatch,
    OperatorFieldBatch,
    OperatorTargetBatch,
)
from ._dataset import OperatorDataset


OperatorCoresetMethod = MomentRecombination | KernelHerding


@dataclass(frozen=True)
class OperatorCaseCoreset:
    """Atomic selected cases with canonical dataset weights and provenance."""

    dataset: OperatorDataset
    selection: CoresetSelection
    source_case_ids: tuple[str, ...]
    objective: str


@dataclass(frozen=True)
class OperatorQueryCoreset:
    """Named query compression with aligned geometry, targets, and physical mass."""

    dataset: OperatorDataset
    query_name: str
    selection: CoresetSelection | tuple[CoresetSelection, ...]
    source_physical_mass: Array
    objective: str


def compress_operator_cases(
    dataset: OperatorDataset,
    method: OperatorCoresetMethod,
    /,
    *,
    features: ArrayLike,
    log_weights: ArrayLike | None = None,
    mask: ArrayLike | None = None,
) -> OperatorCaseCoreset:
    """Select whole cases; no input/query/target/provenance branch is separated."""
    if not isinstance(dataset, OperatorDataset):
        raise TypeError("dataset must be an OperatorDataset.")
    feature_values = jnp.asarray(features)
    if feature_values.ndim != 2 or feature_values.shape[0] != dataset.size:
        raise ValueError("features must have shape (case, feature).")
    source_log_weights = (
        dataset.case_log_weights
        if log_weights is None
        else jnp.asarray(log_weights, dtype=float)
    )
    source_mask = dataset.case_mask if mask is None else jnp.asarray(mask, dtype=bool)
    selection = _select(
        feature_values,
        method,
        log_weights=source_log_weights,
        mask=source_mask,
    )
    selected = dataset.take(selection.indices)
    selected = OperatorDataset(
        selected.batch,
        selected.targets,
        selected.provenance,
        case_log_weights=selection.log_weights,
        case_mask=selection.mask,
    )
    assert dataset.provenance is not None
    source_case_ids = tuple(
        dataset.provenance[int(index)].case_id
        for index in np.asarray(jax.device_get(selection.indices))
    )
    return OperatorCaseCoreset(
        dataset=selected,
        selection=selection,
        source_case_ids=source_case_ids,
        objective=(
            "supplied-case-feature-moments"
            if isinstance(method, MomentRecombination)
            else "declared-kernel-MMD"
        ),
    )


def compress_operator_queries(
    dataset: OperatorDataset,
    query_name: str,
    method: OperatorCoresetMethod,
    /,
    *,
    features: ArrayLike,
    output_geometry: Literal["point-cloud"] | None = None,
) -> OperatorQueryCoreset:
    """Compress one named query without flattening cases or other query branches."""
    if not isinstance(dataset, OperatorDataset):
        raise TypeError("dataset must be an OperatorDataset.")
    if query_name not in dataset.batch.queries:
        raise KeyError(f"Unknown operator query {query_name!r}.")
    query = dataset.batch.query(query_name)
    if query.topology is not None:
        raise ValueError("Topology-bearing queries cannot be compressed as point clouds.")
    coordinates, source_weights, source_mask = _point_cloud_geometry(
        query,
        case_shape=dataset.batch.case_shape,
        output_geometry=output_geometry,
    )
    feature_values = jnp.asarray(features)
    case_count = dataset.size
    point_count = int(coordinates.shape[-2])
    if feature_values.shape[:1] == (point_count,) and feature_values.ndim == 2:
        shared = True
        selection = _select(
            feature_values,
            method,
            log_weights=jnp.log(
                jnp.where(source_weights[0] > 0.0, source_weights[0], 1.0)
            ),
            mask=source_mask[0] & (source_weights[0] > 0.0),
        )
        selections: tuple[CoresetSelection, ...] | None = None
        indices = jnp.broadcast_to(selection.indices, (case_count, selection.capacity))
        selected_mask = jnp.broadcast_to(selection.mask, indices.shape)
        normalized = jnp.broadcast_to(selection.weights, indices.shape)
    elif feature_values.ndim == 3 and feature_values.shape[:2] == (
        case_count,
        point_count,
    ):
        shared = False
        result = tuple(
            _select(
                feature_values[case],
                method,
                log_weights=jnp.log(
                    jnp.where(source_weights[case] > 0.0, source_weights[case], 1.0)
                ),
                mask=source_mask[case] & (source_weights[case] > 0.0),
            )
            for case in range(case_count)
        )
        selections = result
        selection = result[0]
        if any(item.capacity != selection.capacity for item in result[1:]):
            raise ValueError("Per-case query selectors must have one fixed capacity.")
        indices = jnp.stack(tuple(item.indices for item in result))
        selected_mask = jnp.stack(tuple(item.mask for item in result))
        normalized = jnp.stack(tuple(item.weights for item in result))
    else:
        raise ValueError(
            "features must have shape (point, feature) for shared geometry or "
            "(case, point, feature) for per-case point clouds."
        )
    physical_mass = jnp.sum(source_weights, axis=1)
    selected_coordinates = jnp.take_along_axis(
        coordinates,
        indices[..., None],
        axis=1,
    )
    selected_weights = normalized * physical_mass[:, None]
    selected_query = FunctionSamples(
        values=_gather_optional_query_values(
            query.values, indices, case_count=case_count
        ),
        coordinates=(
            selected_coordinates[0]
            if shared and query.geometry_case_shape == ()
            else selected_coordinates
        ),
        quadrature_weights=(selected_weights[0] if shared else selected_weights),
        mask=(selected_mask[0] if shared else selected_mask),
        support_id=f"coreset:{query.support_id}",
        measure_id=query.measure_id,
    )
    queries = dict(dataset.batch.queries)
    queries[query_name] = selected_query
    batch = OperatorBatch(
        inputs=dataset.batch.inputs,
        queries=queries,
        case_axes=dataset.batch.case_axes,
        case_shape=dataset.batch.case_shape,
    )
    target_fields = {}
    for name, field in dataset.targets.fields.items():
        if field.query_name == query_name:
            values = _gather_target_values(
                field.values,
                indices,
                case_count=case_count,
            )
            target_fields[name] = OperatorFieldBatch(
                values,
                query_name=field.query_name,
                spec=field.spec,
            )
        else:
            target_fields[name] = field
    targets = OperatorTargetBatch(
        target_fields,
        case_axes=dataset.targets.case_axes,
        case_shape=dataset.targets.case_shape,
    )
    compressed = OperatorDataset(
        batch,
        targets,
        dataset.provenance,
        case_log_weights=dataset.case_log_weights,
        case_mask=dataset.case_mask,
    )
    return OperatorQueryCoreset(
        dataset=compressed,
        query_name=query_name,
        selection=selection if shared else selections,
        source_physical_mass=physical_mass,
        objective=(
            "supplied-query-feature-moments"
            if isinstance(method, MomentRecombination)
            else "declared-query-kernel-MMD"
        ),
    )


def _select(
    features: Array,
    method: OperatorCoresetMethod,
    /,
    *,
    log_weights: Array,
    mask: Array,
) -> CoresetSelection:
    if isinstance(method, MomentRecombination):
        return moment_recombine(
            features,
            method,
            log_weights=log_weights,
            mask=mask,
        )
    if isinstance(method, KernelHerding):
        return kernel_herd(
            features,
            method,
            log_weights=log_weights,
            mask=mask,
        )
    raise TypeError("method must be MomentRecombination or KernelHerding.")


def _point_cloud_geometry(
    query: FunctionSamples,
    /,
    *,
    case_shape: tuple[int, ...],
    output_geometry: Literal["point-cloud"] | None,
) -> tuple[Array, Array, Array]:
    case_count = case_shape[0]
    if query.axes:
        if output_geometry != "point-cloud":
            raise ValueError(
                "Tensor-product queries require output_geometry='point-cloud' before "
                "compression because the tensor axes are not preserved."
            )
        mesh = jnp.meshgrid(*(axis.nodes for axis in query.axes), indexing="ij")
        shared_coordinates = jnp.stack(
            tuple(coordinate.reshape((-1,)) for coordinate in mesh), axis=1
        )
        coordinates = jnp.broadcast_to(
            shared_coordinates, (case_count,) + shared_coordinates.shape
        )
    elif query.coordinates is not None:
        coordinates = jnp.broadcast_to(
            query.coordinates,
            (case_count,) + query.coordinates.shape[-2:],
        )
    else:
        raise ValueError("Named query has no compressible sample geometry.")
    weights = query.weights(case_shape=case_shape)
    mask = query.mask_array(case_shape=case_shape)
    return coordinates, weights.reshape((case_count, -1)), mask.reshape((case_count, -1))


def _gather_optional_query_values(
    values: Array | None,
    indices: Array,
    /,
    *,
    case_count: int,
) -> Array | None:
    if values is None:
        return None
    array = jnp.asarray(values)
    if array.shape[0] != case_count:
        raise ValueError("Per-case query values must align with dataset cases.")
    trailing = (1,) * (array.ndim - 2)
    expanded = indices.reshape(indices.shape + trailing)
    expanded = jnp.broadcast_to(expanded, indices.shape + array.shape[2:])
    return jnp.take_along_axis(array, expanded, axis=1)


def _gather_target_values(
    values: Array,
    indices: Array,
    /,
    *,
    case_count: int,
) -> Array:
    array = jnp.asarray(values)
    if array.shape[0] != case_count:
        raise ValueError("Operator targets must align with dataset cases.")
    trailing = (1,) * (array.ndim - 2)
    expanded = indices.reshape(indices.shape + trailing)
    expanded = jnp.broadcast_to(expanded, indices.shape + array.shape[2:])
    return jnp.take_along_axis(array, expanded, axis=1)


__all__ = [
    "OperatorCaseCoreset",
    "OperatorCoresetMethod",
    "OperatorQueryCoreset",
    "compress_operator_cases",
    "compress_operator_queries",
]
