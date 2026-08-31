#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellComplexTopology
from ..sparse import EdgeRelation
from ._complex import CellSubcomplex, CellVertexSupport


FiltrationDirection: TypeAlias = Literal["sublevel", "superlevel"]


class CellFiltration(StrictModule, NonTrainableState):
    """Validated scalar filtration of one compact algebraic cell subcomplex."""

    complex: CellSubcomplex
    values: tuple[Array, ...]
    order_degrees: Array
    order_ambient_indices: Array
    order_compact_indices: Array
    order_global_indices: Array
    canonical_order_values: Array
    degree_offsets: tuple[int, ...] = eqx.field(static=True)
    direction: FiltrationDirection = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    filtration_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: CellSubcomplex,
        values: Sequence[ArrayLike],
        /,
        *,
        direction: FiltrationDirection = "sublevel",
        source_id: str,
    ):
        if not isinstance(complex, CellSubcomplex):
            raise TypeError("Cell filtrations require a CellSubcomplex.")
        if direction not in ("sublevel", "superlevel"):
            raise ValueError("Filtration direction must be 'sublevel' or 'superlevel'.")
        source = str(source_id)
        if not source:
            raise ValueError("Filtration source_id must be non-empty.")
        supplied = tuple(values)
        if len(supplied) != len(complex.topology.entity_sets):
            raise ValueError("One filtration-value array is required per cell degree.")
        normalized = []
        canonical = []
        for degree, (value, entity_set, mask) in enumerate(
            zip(
                supplied,
                complex.topology.entity_sets,
                complex.masks,
                strict=True,
            )
        ):
            array = np.asarray(value)
            if array.shape != (entity_set.count,):
                raise ValueError(
                    f"Degree-{degree} filtration values must have shape "
                    f"{(entity_set.count,)}."
                )
            selected = np.asarray(mask, dtype=bool)
            if np.any(~np.isfinite(array[selected])):
                raise ValueError("Selected filtration values must be finite.")
            stored = np.where(selected, array, np.zeros((), dtype=array.dtype))
            normalized.append(jnp.asarray(stored))
            canonical.append(stored if direction == "sublevel" else -stored)
        for incidence in complex.topology.incidences:
            valid = np.asarray(incidence.relation.valid, dtype=bool)
            lower = np.asarray(incidence.relation.source_indices)[valid]
            upper = np.asarray(incidence.relation.target_indices)[valid]
            selected = np.asarray(complex.masks[incidence.degree])[upper]
            if np.any(
                canonical[incidence.degree - 1][lower][selected]
                > canonical[incidence.degree][upper][selected]
            ):
                raise ValueError("Filtration values violate face monotonicity.")
        offsets = tuple(
            np.cumsum((0,) + complex.layout.counts[:-1], dtype=np.int64).tolist()
        )
        records = []
        for degree, (entity_set, compact, canonical_values) in enumerate(
            zip(
                complex.topology.entity_sets,
                complex.layout.compact_to_ambient,
                canonical,
                strict=True,
            )
        ):
            ambient = np.asarray(compact, dtype=np.int32)
            entity_ids = np.asarray(entity_set.entity_ids)[ambient]
            for compact_index, (ambient_index, entity_id) in enumerate(
                zip(ambient, entity_ids, strict=True)
            ):
                records.append(
                    (
                        float(canonical_values[int(ambient_index)]),
                        degree,
                        int(entity_id),
                        int(ambient_index),
                        compact_index,
                        offsets[degree] + compact_index,
                    )
                )
        records.sort(key=lambda value: value[:4])
        self.complex = complex
        self.values = tuple(normalized)
        self.order_degrees = jnp.asarray(
            np.asarray([value[1] for value in records], dtype=np.int32)
        )
        self.order_ambient_indices = jnp.asarray(
            np.asarray([value[3] for value in records], dtype=np.int32)
        )
        self.order_compact_indices = jnp.asarray(
            np.asarray([value[4] for value in records], dtype=np.int32)
        )
        self.order_global_indices = jnp.asarray(
            np.asarray([value[5] for value in records], dtype=np.int32)
        )
        self.canonical_order_values = jnp.asarray(
            np.asarray([value[0] for value in records])
        )
        self.degree_offsets = offsets
        self.direction = direction
        self.source_id = source
        self.filtration_id = canonical_fingerprint(
            {
                "kind": "cell-filtration",
                "complex": complex.subcomplex_id,
                "direction": direction,
                "source": source,
                "values": [array_tree_fingerprint(value) for value in normalized],
                "order": array_tree_fingerprint(self.order_global_indices),
            }
        )

    @property
    def cell_count(self) -> int:
        return self.complex.layout.cell_count

    @property
    def max_degree(self) -> int:
        return self.complex.max_degree

    def compact_values(self, /) -> Array:
        return jnp.concatenate(
            tuple(
                value[compact]
                for value, compact in zip(
                    self.values,
                    self.complex.layout.compact_to_ambient,
                    strict=True,
                )
            )
        )

    def canonical_compact_values(self, /) -> Array:
        values = self.compact_values()
        return values if self.direction == "sublevel" else -values


class PreparedVertexFiltration(StrictModule, NonTrainableState):
    """Fixed-structure JAX evaluation of lower- or upper-star cell values."""

    complex: CellSubcomplex
    support: CellVertexSupport
    direction: FiltrationDirection = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: CellSubcomplex,
        support: CellVertexSupport,
        /,
        *,
        direction: FiltrationDirection,
    ):
        if not isinstance(complex, CellSubcomplex):
            raise TypeError("Prepared vertex filtrations require a CellSubcomplex.")
        if not isinstance(support, CellVertexSupport):
            raise TypeError("support must be a CellVertexSupport.")
        if support.topology_id != complex.topology.topology_id:
            raise ValueError("Vertex support belongs to a different topology.")
        if direction not in ("sublevel", "superlevel"):
            raise ValueError("Unknown vertex-filtration direction.")
        self.complex = complex
        self.support = support
        self.direction = direction
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-vertex-filtration",
                "complex": complex.subcomplex_id,
                "support": support.support_id,
                "direction": direction,
            }
        )

    def cell_values(self, vertex_values: ArrayLike, /) -> tuple[Array, ...]:
        values = jnp.asarray(vertex_values)
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            raise TypeError(
                "Prepared vertex filtration values must use an inexact dtype."
            )
        vertex_count = self.complex.topology.entity_sets[0].count
        if values.ndim == 0 or int(values.shape[-1]) != vertex_count:
            raise ValueError(
                "Vertex filtration values require trailing topology vertex count."
            )
        flat = values.reshape((-1, vertex_count))
        outputs = []
        for degree, (relation, entity_set, mask) in enumerate(
            zip(
                self.support.relations,
                self.complex.topology.entity_sets,
                self.complex.masks,
                strict=True,
            )
        ):
            if degree == 0:
                result = flat
            else:
                valid = relation.valid
                source = jnp.where(valid, relation.source_indices, 0)
                target = jnp.where(valid, relation.target_indices, 0)

                def reduce_one(
                    row,
                    *,
                    source_=source,
                    target_=target,
                    valid_=valid,
                    entity_count=entity_set.count,
                    direction=self.direction,
                ):
                    gathered = row[source_]
                    if direction == "sublevel":
                        gathered = jnp.where(valid_, gathered, -jnp.inf)
                        initial = jnp.full((entity_count,), -jnp.inf, dtype=row.dtype)
                        return initial.at[target_].max(gathered)
                    gathered = jnp.where(valid_, gathered, jnp.inf)
                    initial = jnp.full((entity_count,), jnp.inf, dtype=row.dtype)
                    return initial.at[target_].min(gathered)

                result = jax.vmap(reduce_one)(flat)
            result = jnp.where(jnp.asarray(mask)[None, :], result, 0)
            outputs.append(result.reshape(values.shape[:-1] + (entity_set.count,)))
        return tuple(outputs)

    def snapshot(
        self,
        vertex_values: ArrayLike,
        /,
        *,
        source_id: str,
    ) -> CellFiltration:
        values = self.cell_values(vertex_values)
        if jnp.asarray(vertex_values).ndim != 1:
            raise ValueError("Exact filtration snapshots require one unbatched field.")
        return CellFiltration(
            self.complex,
            values,
            direction=self.direction,
            source_id=source_id,
        )


def cell_vertex_support(
    topology: CellComplexTopology,
    vertices_by_degree: Sequence[ArrayLike],
    /,
) -> CellVertexSupport:
    """Build explicit padded cell-to-vertex support relations."""
    supplied = tuple(vertices_by_degree)
    if len(supplied) != len(topology.entity_sets):
        raise ValueError("One cell-vertex array is required per cell degree.")
    relations = []
    vertex_count = topology.entity_sets[0].count
    for degree, (value, entity_set) in enumerate(
        zip(supplied, topology.entity_sets, strict=True)
    ):
        indices = np.asarray(value)
        if indices.ndim != 2 or indices.shape[0] != entity_set.count:
            raise ValueError(
                f"Degree-{degree} cell vertices require shape (cell_count, width)."
            )
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Cell-vertex support indices must use integer dtype.")
        valid = indices >= 0
        if np.any(valid & (indices >= vertex_count)):
            raise ValueError("Cell-vertex support index exceeds the vertex count.")
        source = np.where(valid, indices, 0).reshape((-1,)).astype(np.int32)
        target = np.repeat(
            np.arange(entity_set.count, dtype=np.int32),
            indices.shape[1],
        )
        relations.append(
            EdgeRelation(
                source,
                target,
                source_size=vertex_count,
                target_size=entity_set.count,
                valid=valid.reshape((-1,)),
            )
        )
    return CellVertexSupport(topology, relations)


def lower_star_filtration(
    complex: CellSubcomplex,
    support: CellVertexSupport,
    vertex_values: ArrayLike,
    /,
    *,
    source_id: str,
) -> CellFiltration:
    return PreparedVertexFiltration(
        complex,
        support,
        direction="sublevel",
    ).snapshot(vertex_values, source_id=source_id)


def upper_star_filtration(
    complex: CellSubcomplex,
    support: CellVertexSupport,
    vertex_values: ArrayLike,
    /,
    *,
    source_id: str,
) -> CellFiltration:
    return PreparedVertexFiltration(
        complex,
        support,
        direction="superlevel",
    ).snapshot(vertex_values, source_id=source_id)


__all__ = [
    "CellFiltration",
    "FiltrationDirection",
    "PreparedVertexFiltration",
    "cell_vertex_support",
    "lower_star_filtration",
    "upper_star_filtration",
]
