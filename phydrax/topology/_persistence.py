#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._coefficients import PrimeField
from ._complex import CellComplexPair, CellSubcomplex, compact_boundary, CompactCellLayout
from ._diagram import PackedPersistenceDiagram, PersistenceDiagram
from ._filtration import CellFiltration
from ._reduction import (
    field_columns,
    FieldVector,
    reduce_columns,
    verify_boundary_composition,
)
from ._resources import (
    TopologyReductionEvidence,
    TopologyResourceError,
    TopologyResourcePolicy,
)


PersistenceRepresentativeKind: TypeAlias = Literal["none", "cycles"]


class PersistenceRepresentatives(StrictModule, NonTrainableState):
    """Sparse-storage birth-cycle representatives aligned with persistence pairs."""

    cell_order_indices: Array
    pair_indices: Array
    coefficients: Array
    pair_count: int = eqx.field(static=True)
    field: PrimeField
    source_id: str = eqx.field(static=True)
    representatives_id: str = eqx.field(static=True)

    def __init__(
        self,
        vectors: Sequence[FieldVector],
        field: PrimeField,
        /,
        *,
        source_id: str,
    ):
        cells = []
        pairs = []
        coefficients = []
        for pair, vector in enumerate(vectors):
            for cell, coefficient in sorted(vector.items()):
                normalized = field.normalize(coefficient)
                if normalized:
                    cells.append(int(cell))
                    pairs.append(pair)
                    coefficients.append(normalized)
        cell_array = np.asarray(cells, dtype=np.int32)
        pair_array = np.asarray(pairs, dtype=np.int32)
        coefficient_array = np.asarray(coefficients, dtype=np.int64)
        self.cell_order_indices = jnp.asarray(cell_array)
        self.pair_indices = jnp.asarray(pair_array)
        self.coefficients = jnp.asarray(coefficient_array)
        self.pair_count = len(tuple(vectors))
        self.field = field
        self.source_id = str(source_id)
        self.representatives_id = canonical_fingerprint(
            {
                "kind": "persistence-representatives",
                "source": self.source_id,
                "field": field.field_id,
                "cells": array_tree_fingerprint(cell_array),
                "pairs": array_tree_fingerprint(pair_array),
                "coefficients": array_tree_fingerprint(coefficient_array),
            }
        )


class PersistencePairing(StrictModule, NonTrainableState):
    """Deterministic cell-level reduction pairing for one exact filtration."""

    degrees: Array
    birth_order_indices: Array
    death_order_indices: Array
    birth_global_indices: Array
    death_global_indices: Array
    birth_entity_ids: Array
    death_entity_ids: Array
    has_finite_death: Array
    order_degrees: Array
    order_ambient_indices: Array
    order_global_indices: Array
    reference_canonical_order_values: Array
    representatives: PersistenceRepresentatives | None
    pair_count: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    field: PrimeField
    pairing_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        degrees: ArrayLike,
        birth_order_indices: ArrayLike,
        death_order_indices: ArrayLike,
        birth_global_indices: ArrayLike,
        death_global_indices: ArrayLike,
        birth_entity_ids: ArrayLike,
        death_entity_ids: ArrayLike,
        has_finite_death: ArrayLike,
        order_degrees: ArrayLike,
        order_ambient_indices: ArrayLike,
        order_global_indices: ArrayLike,
        reference_canonical_order_values: ArrayLike,
        source_id: str,
        topology_id: str,
        layout_id: str,
        field: PrimeField,
        representatives: PersistenceRepresentatives | None = None,
    ):
        pair_arrays = tuple(
            np.asarray(value)
            for value in (
                degrees,
                birth_order_indices,
                death_order_indices,
                birth_global_indices,
                death_global_indices,
                birth_entity_ids,
                death_entity_ids,
                has_finite_death,
            )
        )
        if any(value.ndim != 1 for value in pair_arrays):
            raise ValueError("Persistence pairing arrays must be rank-1.")
        if len({value.shape for value in pair_arrays}) != 1:
            raise ValueError("Persistence pairing arrays must have identical shapes.")
        order_arrays = tuple(
            np.asarray(value)
            for value in (
                order_degrees,
                order_ambient_indices,
                order_global_indices,
                reference_canonical_order_values,
            )
        )
        if any(value.ndim != 1 for value in order_arrays):
            raise ValueError("Persistence order arrays must be rank-1.")
        if len({value.shape for value in order_arrays}) != 1:
            raise ValueError("Persistence order arrays must have identical shapes.")
        if (
            representatives is not None
            and representatives.pair_count != pair_arrays[0].size
        ):
            raise ValueError("Persistence representatives must align with every pair.")
        self.degrees = jnp.asarray(pair_arrays[0], dtype=jnp.int32)
        self.birth_order_indices = jnp.asarray(pair_arrays[1], dtype=jnp.int32)
        self.death_order_indices = jnp.asarray(pair_arrays[2], dtype=jnp.int32)
        self.birth_global_indices = jnp.asarray(pair_arrays[3], dtype=jnp.int32)
        self.death_global_indices = jnp.asarray(pair_arrays[4], dtype=jnp.int32)
        self.birth_entity_ids = jnp.asarray(pair_arrays[5], dtype=jnp.int64)
        self.death_entity_ids = jnp.asarray(pair_arrays[6], dtype=jnp.int64)
        self.has_finite_death = jnp.asarray(pair_arrays[7], dtype=bool)
        self.order_degrees = jnp.asarray(order_arrays[0], dtype=jnp.int32)
        self.order_ambient_indices = jnp.asarray(order_arrays[1], dtype=jnp.int32)
        self.order_global_indices = jnp.asarray(order_arrays[2], dtype=jnp.int32)
        self.reference_canonical_order_values = jnp.asarray(order_arrays[3])
        self.representatives = representatives
        self.pair_count = int(pair_arrays[0].size)
        self.source_id = str(source_id)
        self.topology_id = str(topology_id)
        self.layout_id = str(layout_id)
        self.field = field
        self.pairing_id = canonical_fingerprint(
            {
                "kind": "persistence-pairing",
                "source": self.source_id,
                "topology": self.topology_id,
                "layout": self.layout_id,
                "field": field.field_id,
                "degrees": array_tree_fingerprint(pair_arrays[0]),
                "births": array_tree_fingerprint(pair_arrays[1]),
                "deaths": array_tree_fingerprint(pair_arrays[2]),
                "finite": array_tree_fingerprint(pair_arrays[7]),
                "order": array_tree_fingerprint(order_arrays[2]),
                "representatives": (
                    None
                    if representatives is None
                    else representatives.representatives_id
                ),
            }
        )


class PersistenceResult(StrictModule, NonTrainableState):
    """Exact persistence reduction and backend-neutral interval views."""

    pairing: PersistencePairing
    compact_values: Array
    canonical_compact_values: Array
    evidence: TopologyReductionEvidence
    filtration_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        pairing: PersistencePairing,
        compact_values: ArrayLike,
        canonical_compact_values: ArrayLike,
        evidence: TopologyReductionEvidence,
        /,
        *,
        filtration_id: str,
    ):
        values = jnp.asarray(compact_values)
        canonical = jnp.asarray(canonical_compact_values)
        if values.ndim != 1 or canonical.shape != values.shape:
            raise ValueError("Persistence compact values must be matching vectors.")
        self.pairing = pairing
        self.compact_values = values
        self.canonical_compact_values = canonical
        self.evidence = evidence
        self.filtration_id = str(filtration_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "persistence-result",
                "filtration": self.filtration_id,
                "pairing": pairing.pairing_id,
                "evidence": evidence.evidence_id,
            }
        )

    def diagram(self, /, *, include_zero_length: bool = False) -> PersistenceDiagram:
        pairing = self.pairing
        birth_global = np.asarray(pairing.birth_global_indices, dtype=np.int32)
        death_global = np.asarray(pairing.death_global_indices, dtype=np.int32)
        finite = np.asarray(pairing.has_finite_death, dtype=bool)
        compact = np.asarray(self.compact_values)
        canonical = np.asarray(self.canonical_compact_values)
        births = compact[birth_global]
        safe_death = np.where(finite, death_global, 0)
        deaths = np.where(finite, compact[safe_death], np.zeros((), dtype=compact.dtype))
        canonical_births = canonical[birth_global]
        canonical_deaths = canonical[safe_death]
        keep = ~finite | (canonical_deaths > canonical_births)
        if include_zero_length:
            keep = np.ones_like(keep)
        indices = np.flatnonzero(keep).astype(np.int32)
        return PersistenceDiagram(
            np.asarray(pairing.degrees)[keep],
            births[keep],
            deaths[keep],
            np.asarray(pairing.birth_entity_ids)[keep],
            np.asarray(pairing.death_entity_ids)[keep],
            finite[keep],
            indices,
            source_id=self.result_id,
        )

    def pack(
        self,
        capacity: int,
        /,
        *,
        include_zero_length: bool = False,
    ) -> PackedPersistenceDiagram:
        return PackedPersistenceDiagram(
            self.diagram(include_zero_length=include_zero_length),
            capacity,
        )


class FrozenPersistenceEvaluation(StrictModule):
    """JAX endpoint values and validity for one frozen exact pairing."""

    degrees: Array
    birth_values: Array
    death_values: Array
    has_finite_death: Array
    ordering_valid: Array
    ordering_margin: Array

    def __init__(
        self,
        degrees: Array,
        birth_values: Array,
        death_values: Array,
        has_finite_death: Array,
        ordering_valid: Array,
        ordering_margin: Array,
        /,
    ):
        self.degrees = jnp.asarray(degrees)
        self.birth_values = jnp.asarray(birth_values)
        self.death_values = jnp.asarray(death_values)
        self.has_finite_death = jnp.asarray(has_finite_death)
        self.ordering_valid = jnp.asarray(ordering_valid)
        self.ordering_margin = jnp.asarray(ordering_margin)


class FrozenPersistencePairing(StrictModule, NonTrainableState):
    """Locally differentiable endpoint evaluation for one frozen reduction order."""

    pairing: PersistencePairing
    layout: CompactCellLayout
    direction: str = eqx.field(static=True)
    frozen_id: str = eqx.field(static=True)

    def __init__(
        self,
        result: PersistenceResult,
        layout: CompactCellLayout,
        /,
        *,
        direction: str,
    ):
        if not isinstance(result, PersistenceResult):
            raise TypeError("Frozen pairing requires a PersistenceResult.")
        if not isinstance(layout, CompactCellLayout):
            raise TypeError("layout must be a CompactCellLayout.")
        if result.pairing.layout_id != layout.layout_id:
            raise ValueError(
                "Frozen pairing layout does not match the persistence result."
            )
        if direction not in ("sublevel", "superlevel"):
            raise ValueError("Unknown frozen filtration direction.")
        self.pairing = result.pairing
        self.layout = layout
        self.direction = direction
        self.frozen_id = canonical_fingerprint(
            {
                "kind": "frozen-persistence-pairing",
                "pairing": result.pairing.pairing_id,
                "layout": layout.layout_id,
                "direction": direction,
            }
        )

    def evaluate(self, values: Sequence[ArrayLike], /) -> FrozenPersistenceEvaluation:
        supplied = tuple(jnp.asarray(value) for value in values)
        if len(supplied) != len(self.layout.counts):
            raise ValueError("One dynamic value array is required per cell degree.")
        batch_shape = supplied[0].shape[:-1]
        compact_parts = []
        for degree, (value, compact) in enumerate(
            zip(supplied, self.layout.compact_to_ambient, strict=True)
        ):
            if (
                value.ndim == 0
                or value.shape[:-1] != batch_shape
                or int(value.shape[-1]) != int(self.layout.masks[degree].shape[0])
            ):
                raise ValueError("Dynamic filtration values do not match the topology.")
            compact_parts.append(value[..., compact])
        compact_values = jnp.concatenate(tuple(compact_parts), axis=-1)
        canonical = compact_values if self.direction == "sublevel" else -compact_values
        order = self.pairing.order_global_indices
        ordered = canonical[..., order]
        finite_values = jnp.all(jnp.isfinite(ordered), axis=-1)
        if ordered.shape[-1] <= 1:
            ordering_valid = finite_values
            margin = jnp.full(batch_shape, jnp.inf, dtype=ordered.dtype)
        else:
            differences = ordered[..., 1:] - ordered[..., :-1]
            ordering_valid = finite_values & jnp.all(differences >= 0, axis=-1)
            positive = jnp.where(differences > 0, differences, jnp.inf)
            margin = jnp.min(positive, axis=-1)
        birth_global = self.pairing.birth_global_indices
        finite = self.pairing.has_finite_death
        death_global = jnp.where(finite, self.pairing.death_global_indices, 0)
        births = compact_values[..., birth_global]
        deaths = jnp.where(finite, compact_values[..., death_global], 0)
        return FrozenPersistenceEvaluation(
            self.pairing.degrees,
            births,
            deaths,
            finite,
            ordering_valid,
            margin,
        )


def _analysis_layout(
    filtration: CellFiltration,
    relative_to: CellSubcomplex | None,
    /,
) -> tuple[CellSubcomplex | CellComplexPair, CompactCellLayout, str]:
    if relative_to is None:
        return (
            filtration.complex,
            filtration.complex.layout,
            filtration.complex.subcomplex_id,
        )
    pair = CellComplexPair(filtration.complex, relative_to)
    return pair, pair.quotient_layout, pair.pair_id


def _ordered_cells(
    filtration: CellFiltration,
    layout: CompactCellLayout,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = tuple(np.cumsum((0,) + layout.counts[:-1], dtype=np.int64).tolist())
    records = []
    canonical_values = []
    original_values = []
    for degree, (ambient, entities, values) in enumerate(
        zip(
            layout.compact_to_ambient,
            filtration.complex.topology.entity_sets,
            filtration.values,
            strict=True,
        )
    ):
        ambient_host = np.asarray(ambient, dtype=np.int32)
        entity_ids = np.asarray(entities.entity_ids)[ambient_host]
        degree_values = np.asarray(values)[ambient_host]
        degree_canonical = (
            degree_values if filtration.direction == "sublevel" else -degree_values
        )
        original_values.extend(degree_values.tolist())
        canonical_values.extend(degree_canonical.tolist())
        for compact_index, (ambient_index, entity_id, canonical) in enumerate(
            zip(ambient_host, entity_ids, degree_canonical, strict=True)
        ):
            records.append(
                (
                    float(canonical),
                    degree,
                    int(entity_id),
                    int(ambient_index),
                    offsets[degree] + compact_index,
                )
            )
    records.sort(key=lambda value: value[:4])
    return (
        np.asarray([value[1] for value in records], dtype=np.int32),
        np.asarray([value[3] for value in records], dtype=np.int32),
        np.asarray([value[4] for value in records], dtype=np.int32),
        np.asarray(original_values),
        np.asarray(canonical_values),
    )


def _filtered_columns(
    analysis: CellSubcomplex | CellComplexPair,
    layout: CompactCellLayout,
    order_global: np.ndarray,
    field: PrimeField,
    policy: TopologyResourcePolicy,
    /,
) -> tuple[list[FieldVector], int]:
    if layout.cell_count > policy.max_cells:
        raise TopologyResourceError("Filtered complex exceeds max_cells.")
    boundaries = tuple(
        compact_boundary(analysis, degree) for degree in range(layout.max_degree + 1)
    )
    nonzeros = sum(value.nonzero_count for value in boundaries)
    if nonzeros > policy.max_boundary_nonzeros:
        raise TopologyResourceError("Filtered complex exceeds max_boundary_nonzeros.")
    degree_columns = tuple(field_columns(value, field) for value in boundaries)
    for lower, upper in zip(degree_columns[:-1], degree_columns[1:], strict=True):
        verify_boundary_composition(lower, upper, field, policy)
    offsets = tuple(np.cumsum((0,) + layout.counts[:-1], dtype=np.int64).tolist())
    global_columns: list[FieldVector] = [dict() for _ in range(layout.cell_count)]
    for degree in range(1, layout.max_degree + 1):
        for local_column, column in enumerate(degree_columns[degree]):
            global_column = offsets[degree] + local_column
            global_columns[global_column] = {
                offsets[degree - 1] + row: value for row, value in column.items()
            }
    inverse_order = np.empty((layout.cell_count,), dtype=np.int32)
    inverse_order[order_global] = np.arange(layout.cell_count, dtype=np.int32)
    ordered_columns = []
    for global_index in order_global:
        column = {
            int(inverse_order[row]): value
            for row, value in global_columns[int(global_index)].items()
        }
        current_position = int(inverse_order[int(global_index)])
        if any(row >= current_position for row in column):
            raise ValueError("Filtered boundary contains a face after its coface.")
        ordered_columns.append(column)
    return ordered_columns, nonzeros


def _entity_ids_in_order(
    filtration: CellFiltration,
    order_degrees: np.ndarray,
    order_ambient: np.ndarray,
    /,
) -> np.ndarray:
    return np.asarray(
        [
            int(
                np.asarray(
                    filtration.complex.topology.entity_sets[int(degree)].entity_ids
                )[int(ambient)]
            )
            for degree, ambient in zip(order_degrees, order_ambient, strict=True)
        ],
        dtype=np.int64,
    )


def _verify_representative(
    vector: FieldVector,
    columns: list[FieldVector],
    field: PrimeField,
    /,
) -> None:
    image: FieldVector = {}
    for column_index, coefficient in vector.items():
        for row, value in columns[column_index].items():
            current = field.add(
                image.get(row, 0),
                field.multiply(coefficient, value),
            )
            if current:
                image[row] = current
            elif row in image:
                del image[row]
    if image:
        raise RuntimeError("Persistence birth representative is not a cycle.")


def compute_persistence(
    filtration: CellFiltration,
    /,
    *,
    coefficients: PrimeField,
    relative_to: CellSubcomplex | None = None,
    max_degree: int | None = None,
    representatives: PersistenceRepresentativeKind = "none",
    resources: TopologyResourcePolicy | None = None,
) -> PersistenceResult:
    """Compute exact ordinary or induced-relative persistent homology on the host."""
    if not isinstance(filtration, CellFiltration):
        raise TypeError("compute_persistence requires a CellFiltration.")
    if not isinstance(coefficients, PrimeField):
        raise TypeError("Persistent homology requires an explicit PrimeField.")
    if representatives not in ("none", "cycles"):
        raise ValueError("Persistence representatives must be 'none' or 'cycles'.")
    if relative_to is not None and not isinstance(relative_to, CellSubcomplex):
        raise TypeError("relative_to must be a CellSubcomplex or None.")
    policy = TopologyResourcePolicy() if resources is None else resources
    if not isinstance(policy, TopologyResourcePolicy):
        raise TypeError("resources must be a TopologyResourcePolicy.")
    analysis, layout, source_id = _analysis_layout(filtration, relative_to)
    degree_limit = layout.max_degree if max_degree is None else int(max_degree)
    if degree_limit < 0 or degree_limit > layout.max_degree:
        raise ValueError(f"max_degree must lie in [0, {layout.max_degree}].")
    order_degrees, order_ambient, order_global, compact_values, canonical_values = (
        _ordered_cells(filtration, layout)
    )
    columns, boundary_nonzeros = _filtered_columns(
        analysis,
        layout,
        order_global,
        coefficients,
        policy,
    )
    reduction = reduce_columns(
        columns,
        coefficients,
        policy,
        track_transformations=representatives == "cycles",
    )
    entity_ids = _entity_ids_in_order(filtration, order_degrees, order_ambient)
    creator_indices = [
        index
        for index in reduction.zero_columns
        if int(order_degrees[index]) <= degree_limit
    ]
    records = []
    transforms = []
    for creator in creator_indices:
        death = reduction.pivot_to_column.get(creator, -1)
        finite = death >= 0
        death_entity = int(entity_ids[death]) if finite else 0
        records.append(
            (
                int(order_degrees[creator]),
                creator,
                death,
                int(order_global[creator]),
                int(order_global[death]) if finite else 0,
                int(entity_ids[creator]),
                death_entity,
                finite,
            )
        )
        if reduction.transformations is not None:
            transform = reduction.transformations[creator]
            _verify_representative(transform, columns, coefficients)
            transforms.append(transform)
    records.sort(
        key=lambda value: (
            value[0],
            float(canonical_values[value[3]]),
            0 if value[7] else 1,
            float(canonical_values[value[4]]) if value[7] else np.inf,
            value[5],
        )
    )
    if transforms:
        transform_by_creator = {
            creator: transform
            for creator, transform in zip(creator_indices, transforms, strict=True)
        }
        sorted_transforms = [transform_by_creator[value[1]] for value in records]
        representative_values = PersistenceRepresentatives(
            sorted_transforms,
            coefficients,
            source_id=source_id,
        )
    else:
        representative_values = None
    pairing = PersistencePairing(
        degrees=np.asarray([value[0] for value in records], dtype=np.int32),
        birth_order_indices=np.asarray([value[1] for value in records], dtype=np.int32),
        death_order_indices=np.asarray([value[2] for value in records], dtype=np.int32),
        birth_global_indices=np.asarray([value[3] for value in records], dtype=np.int32),
        death_global_indices=np.asarray([value[4] for value in records], dtype=np.int32),
        birth_entity_ids=np.asarray([value[5] for value in records], dtype=np.int64),
        death_entity_ids=np.asarray([value[6] for value in records], dtype=np.int64),
        has_finite_death=np.asarray([value[7] for value in records], dtype=bool),
        order_degrees=order_degrees,
        order_ambient_indices=order_ambient,
        order_global_indices=order_global,
        reference_canonical_order_values=canonical_values[order_global],
        source_id=source_id,
        topology_id=filtration.complex.topology.topology_id,
        layout_id=layout.layout_id,
        field=coefficients,
        representatives=representative_values,
    )
    evidence = TopologyReductionEvidence(
        "exact-filtered-column-reduction",
        coefficients.field_id,
        {
            "cells": layout.cell_count,
            "boundary_nonzeros": boundary_nonzeros,
            "pairs": len(records),
            "operations": reduction.stats.operations,
            "peak_reduction_entries": reduction.stats.peak_entries,
            "representative_entries": reduction.stats.representative_entries,
        },
    )
    return PersistenceResult(
        pairing,
        compact_values,
        canonical_values,
        evidence,
        filtration_id=filtration.filtration_id,
    )


def freeze_persistence_pairing(
    result: PersistenceResult,
    filtration: CellFiltration,
    /,
    *,
    relative_to: CellSubcomplex | None = None,
) -> FrozenPersistencePairing:
    """Bind an exact persistence pairing to its complete compact structure."""
    if not isinstance(result, PersistenceResult) or not isinstance(
        filtration, CellFiltration
    ):
        raise TypeError("Freezing requires a PersistenceResult and CellFiltration.")
    if result.filtration_id != filtration.filtration_id:
        raise ValueError("Persistence result belongs to a different filtration.")
    _, layout, _ = _analysis_layout(filtration, relative_to)
    return FrozenPersistencePairing(
        result,
        layout,
        direction=filtration.direction,
    )


__all__ = [
    "FrozenPersistenceEvaluation",
    "FrozenPersistencePairing",
    "PersistencePairing",
    "PersistenceRepresentativeKind",
    "PersistenceRepresentatives",
    "PersistenceResult",
    "compute_persistence",
    "freeze_persistence_pairing",
]
