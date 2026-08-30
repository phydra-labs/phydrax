#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellComplexTopology
from ..sparse import EdgeRelation


class CompactCellLayout(StrictModule, NonTrainableState):
    """Compact coordinates for a masked view of one fixed-capacity cell complex."""

    masks: tuple[Array, ...]
    ambient_to_compact: tuple[Array, ...]
    compact_to_ambient: tuple[Array, ...]
    entity_ids: tuple[Array, ...]
    counts: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CellComplexTopology,
        masks: Sequence[ArrayLike],
        /,
        *,
        selection_id: str | None = None,
    ):
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("Compact layouts require a CellComplexTopology.")
        selected = tuple(masks)
        if len(selected) != len(topology.entity_sets):
            raise ValueError("One compact-layout mask is required per cell degree.")
        normalized_masks = []
        ambient_to_compact = []
        compact_to_ambient = []
        entity_ids = []
        counts = []
        for degree, (value, entities) in enumerate(
            zip(selected, topology.entity_sets, strict=True)
        ):
            mask = np.asarray(value, dtype=bool)
            active = np.asarray(entities.active_mask, dtype=bool)
            if mask.shape != (entities.count,):
                raise ValueError(
                    f"Degree-{degree} mask must have shape {(entities.count,)}."
                )
            if np.any(mask & ~active):
                raise ValueError("Compact selections cannot include inactive cells.")
            compact = np.flatnonzero(mask).astype(np.int32)
            inverse = np.full((entities.count,), -1, dtype=np.int32)
            inverse[compact] = np.arange(compact.size, dtype=np.int32)
            normalized_masks.append(jnp.asarray(mask))
            ambient_to_compact.append(jnp.asarray(inverse))
            compact_to_ambient.append(jnp.asarray(compact))
            entity_ids.append(jnp.asarray(np.asarray(entities.entity_ids)[compact]))
            counts.append(int(compact.size))
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "cell-selection",
                    "topology": topology.topology_id,
                    "masks": [
                        array_tree_fingerprint(value) for value in normalized_masks
                    ],
                }
            )
            if selection_id is None
            else str(selection_id)
        )
        if not identifier:
            raise ValueError("selection_id must be non-empty.")
        self.masks = tuple(normalized_masks)
        self.ambient_to_compact = tuple(ambient_to_compact)
        self.compact_to_ambient = tuple(compact_to_ambient)
        self.entity_ids = tuple(entity_ids)
        self.counts = tuple(counts)
        self.topology_id = topology.topology_id
        self.selection_id = identifier
        self.layout_id = canonical_fingerprint(
            {
                "kind": "compact-cell-layout",
                "topology": topology.topology_id,
                "selection": identifier,
                "counts": counts,
                "ambient": [
                    array_tree_fingerprint(value) for value in compact_to_ambient
                ],
            }
        )

    @property
    def max_degree(self) -> int:
        return len(self.counts) - 1

    @property
    def cell_count(self) -> int:
        return sum(self.counts)


class CellSubcomplex(StrictModule, NonTrainableState):
    """Validated algebraic coordinate subcomplex of a canonical cell complex."""

    topology: CellComplexTopology
    layout: CompactCellLayout
    subcomplex_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CellComplexTopology,
        masks: Sequence[ArrayLike],
        /,
        *,
        subcomplex_id: str | None = None,
    ):
        layout = CompactCellLayout(topology, masks)
        self._validate_closed(topology, layout)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "cell-subcomplex",
                    "topology": topology.topology_id,
                    "layout": layout.layout_id,
                }
            )
            if subcomplex_id is None
            else str(subcomplex_id)
        )
        if not identifier:
            raise ValueError("subcomplex_id must be non-empty.")
        self.topology = topology
        self.layout = CompactCellLayout(
            topology,
            layout.masks,
            selection_id=identifier,
        )
        self.subcomplex_id = identifier

    @staticmethod
    def _validate_closed(
        topology: CellComplexTopology,
        layout: CompactCellLayout,
        /,
    ) -> None:
        for incidence in topology.incidences:
            valid = np.asarray(incidence.relation.valid, dtype=bool)
            lower = np.asarray(incidence.relation.source_indices)[valid]
            upper = np.asarray(incidence.relation.target_indices)[valid]
            lower_selected = np.asarray(layout.masks[incidence.degree - 1])[lower]
            upper_selected = np.asarray(layout.masks[incidence.degree])[upper]
            if np.any(upper_selected & ~lower_selected):
                raise ValueError(
                    "Selected cells do not form a subcomplex: a selected cell has "
                    "an excluded boundary cell."
                )

    @classmethod
    def full(cls, topology: CellComplexTopology, /) -> "CellSubcomplex":
        return cls(
            topology,
            tuple(entity_set.active_mask for entity_set in topology.entity_sets),
        )

    @classmethod
    def from_subsets(
        cls,
        topology: CellComplexTopology,
        name: str,
        /,
    ) -> "CellSubcomplex":
        subset_name = str(name)
        if not subset_name:
            raise ValueError("Subset name must be non-empty.")
        return cls(
            topology,
            tuple(
                entity_set.subset(subset_name).mask for entity_set in topology.entity_sets
            ),
        )

    @property
    def masks(self) -> tuple[Array, ...]:
        return self.layout.masks

    @property
    def max_degree(self) -> int:
        return self.layout.max_degree


class CellComplexPair(StrictModule, NonTrainableState):
    """An ambient subcomplex and one included relative subcomplex."""

    ambient: CellSubcomplex
    relative: CellSubcomplex
    quotient_layout: CompactCellLayout
    pair_id: str = eqx.field(static=True)

    def __init__(self, ambient: CellSubcomplex, relative: CellSubcomplex, /):
        if not isinstance(ambient, CellSubcomplex) or not isinstance(
            relative, CellSubcomplex
        ):
            raise TypeError("Cell-complex pairs require two CellSubcomplex values.")
        if ambient.topology.topology_id != relative.topology.topology_id:
            raise ValueError("Cell-complex pairs must share one exact topology.")
        quotient_masks = []
        for ambient_mask, relative_mask in zip(
            ambient.masks, relative.masks, strict=True
        ):
            ambient_host = np.asarray(ambient_mask, dtype=bool)
            relative_host = np.asarray(relative_mask, dtype=bool)
            if np.any(relative_host & ~ambient_host):
                raise ValueError("The relative subcomplex must be contained in ambient.")
            quotient_masks.append(ambient_host & ~relative_host)
        identifier = canonical_fingerprint(
            {
                "kind": "cell-complex-pair",
                "ambient": ambient.subcomplex_id,
                "relative": relative.subcomplex_id,
            }
        )
        self.ambient = ambient
        self.relative = relative
        self.quotient_layout = CompactCellLayout(
            ambient.topology,
            quotient_masks,
            selection_id=f"{identifier}:quotient",
        )
        self.pair_id = identifier

    @property
    def topology(self) -> CellComplexTopology:
        return self.ambient.topology

    @property
    def max_degree(self) -> int:
        return self.ambient.max_degree


class CompactBoundary(StrictModule, NonTrainableState):
    """Exact integer COO boundary in compact chain coordinates."""

    row_indices: Array
    column_indices: Array
    coefficients: Array
    degree: int = eqx.field(static=True)
    row_count: int = eqx.field(static=True)
    column_count: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        row_count: int,
        column_count: int,
        row_indices: ArrayLike,
        column_indices: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        source_id: str,
    ):
        degree_ = int(degree)
        rows = np.asarray(row_indices)
        columns = np.asarray(column_indices)
        values = np.asarray(coefficients)
        if degree_ < 0:
            raise ValueError("Boundary degree must be non-negative.")
        if rows.ndim != 1 or columns.ndim != 1 or values.ndim != 1:
            raise ValueError("Compact boundary arrays must be rank-1.")
        if rows.shape != columns.shape or rows.shape != values.shape:
            raise ValueError("Compact boundary arrays must have identical shapes.")
        if not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(
            columns.dtype, np.integer
        ):
            raise TypeError("Compact boundary indices must use integer dtypes.")
        rounded = np.rint(values)
        if np.any(values != rounded):
            raise ValueError("Compact boundary coefficients must be exact integers.")
        rows = rows.astype(np.int32, copy=False)
        columns = columns.astype(np.int32, copy=False)
        values = rounded.astype(np.int64, copy=False)
        if np.any(rows < 0) or np.any(rows >= int(row_count)):
            raise ValueError("Compact boundary row index is out of range.")
        if np.any(columns < 0) or np.any(columns >= int(column_count)):
            raise ValueError("Compact boundary column index is out of range.")
        source = str(source_id)
        if not source:
            raise ValueError("Compact boundary source_id must be non-empty.")
        self.row_indices = jnp.asarray(rows)
        self.column_indices = jnp.asarray(columns)
        self.coefficients = jnp.asarray(values)
        self.degree = degree_
        self.row_count = int(row_count)
        self.column_count = int(column_count)
        self.source_id = source
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "compact-boundary",
                "degree": degree_,
                "shape": [int(row_count), int(column_count)],
                "source": source,
                "rows": array_tree_fingerprint(rows),
                "columns": array_tree_fingerprint(columns),
                "coefficients": array_tree_fingerprint(values),
            }
        )

    @property
    def nonzero_count(self) -> int:
        return int(self.coefficients.shape[0])


class CellVertexSupport(StrictModule, NonTrainableState):
    """Explicit vertex support for every cell of one canonical topology."""

    topology_id: str = eqx.field(static=True)
    relations: tuple[EdgeRelation, ...]
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CellComplexTopology,
        relations: Sequence[EdgeRelation],
        /,
    ):
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("Cell vertex support requires a CellComplexTopology.")
        values = tuple(relations)
        if len(values) != len(topology.entity_sets):
            raise ValueError("One vertex-support relation is required per cell degree.")
        vertex_count = topology.entity_sets[0].count
        supports: list[list[set[int]]] = []
        for degree, (relation, entity_set) in enumerate(
            zip(values, topology.entity_sets, strict=True)
        ):
            if not isinstance(relation, EdgeRelation):
                raise TypeError("Vertex supports must contain EdgeRelation values.")
            if (
                relation.source_size != vertex_count
                or relation.target_size != entity_set.count
            ):
                raise ValueError(
                    "Vertex-support relation sizes do not match the topology."
                )
            valid = np.asarray(relation.valid, dtype=bool)
            vertices = np.asarray(relation.source_indices)[valid]
            cells = np.asarray(relation.target_indices)[valid]
            degree_support = [set() for _ in range(entity_set.count)]
            for vertex, cell in zip(vertices, cells, strict=True):
                degree_support[int(cell)].add(int(vertex))
            active = np.asarray(entity_set.active_mask, dtype=bool)
            if any(
                active[index] and not support
                for index, support in enumerate(degree_support)
            ):
                raise ValueError("Every active cell requires non-empty vertex support.")
            if degree == 0 and any(
                active[index] and support != {index}
                for index, support in enumerate(degree_support)
            ):
                raise ValueError("Active vertices must support themselves exactly.")
            supports.append(degree_support)
        for incidence in topology.incidences:
            valid = np.asarray(incidence.relation.valid, dtype=bool)
            lower = np.asarray(incidence.relation.source_indices)[valid]
            upper = np.asarray(incidence.relation.target_indices)[valid]
            for lower_index, upper_index in zip(lower, upper, strict=True):
                if not supports[incidence.degree - 1][int(lower_index)].issubset(
                    supports[incidence.degree][int(upper_index)]
                ):
                    raise ValueError(
                        "The vertex support of a boundary cell must be contained in "
                        "the support of its coface."
                    )
        self.topology_id = topology.topology_id
        self.relations = values
        self.support_id = canonical_fingerprint(
            {
                "kind": "cell-vertex-support",
                "topology": topology.topology_id,
                "relations": [
                    {
                        "source": array_tree_fingerprint(value.source_indices),
                        "target": array_tree_fingerprint(value.target_indices),
                        "valid": array_tree_fingerprint(value.valid),
                    }
                    for value in values
                ],
            }
        )


def compact_boundary(
    complex_or_pair: CellSubcomplex | CellComplexPair,
    degree: int,
    /,
) -> CompactBoundary:
    """Extract one exact compact boundary, including relative quotient semantics."""
    degree_ = int(degree)
    if not isinstance(complex_or_pair, (CellSubcomplex, CellComplexPair)):
        raise TypeError("compact_boundary requires a CellSubcomplex or CellComplexPair.")
    layout = (
        complex_or_pair.layout
        if isinstance(complex_or_pair, CellSubcomplex)
        else complex_or_pair.quotient_layout
    )
    source_id = (
        complex_or_pair.subcomplex_id
        if isinstance(complex_or_pair, CellSubcomplex)
        else complex_or_pair.pair_id
    )
    if degree_ < 0 or degree_ > layout.max_degree:
        raise ValueError(f"Boundary degree must lie in [0, {layout.max_degree}].")
    if degree_ == 0:
        return CompactBoundary(
            0,
            0,
            layout.counts[0],
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
            source_id=source_id,
        )
    incidence = complex_or_pair.topology.incidences[degree_ - 1]
    valid = np.asarray(incidence.relation.valid, dtype=bool)
    ambient_rows = np.asarray(incidence.relation.source_indices)[valid]
    ambient_columns = np.asarray(incidence.relation.target_indices)[valid]
    values = np.asarray(incidence.signs)[valid]
    row_map = np.asarray(layout.ambient_to_compact[degree_ - 1])
    column_map = np.asarray(layout.ambient_to_compact[degree_])
    compact_rows = row_map[ambient_rows]
    compact_columns = column_map[ambient_columns]
    selected = (compact_rows >= 0) & (compact_columns >= 0)
    accumulator: dict[tuple[int, int], int] = {}
    for row, column, value in zip(
        compact_rows[selected], compact_columns[selected], values[selected], strict=True
    ):
        rounded = int(np.rint(value))
        if float(value) != rounded:
            raise ValueError("Incidence coefficients must be exact integers.")
        key = (int(row), int(column))
        accumulator[key] = accumulator.get(key, 0) + rounded
    entries = tuple(
        (row, column, value)
        for (row, column), value in sorted(accumulator.items())
        if value != 0
    )
    rows = np.asarray([entry[0] for entry in entries], dtype=np.int32)
    columns = np.asarray([entry[1] for entry in entries], dtype=np.int32)
    coefficients = np.asarray([entry[2] for entry in entries], dtype=np.int64)
    return CompactBoundary(
        degree_,
        layout.counts[degree_ - 1],
        layout.counts[degree_],
        rows,
        columns,
        coefficients,
        source_id=source_id,
    )


__all__ = [
    "CellComplexPair",
    "CellSubcomplex",
    "CellVertexSupport",
    "CompactBoundary",
    "CompactCellLayout",
    "compact_boundary",
]
