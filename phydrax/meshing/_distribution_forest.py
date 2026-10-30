#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Stable-ID extraction and validation for adaptive simplex bisection forests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_array_hex_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexState,
)


_CELL_FIELDS = (
    "active",
    "blocks",
    "bisection_vertex_ids",
    "cell_classes",
    "cells",
    "children_ids",
    "facet_classes",
    "generations",
    "parent_ids",
    "parent_ordinals",
    "refine_rejected",
    "retired",
    "tags",
    "tuples",
    "coarsen_marked",
)
_VERTEX_FIELDS = (
    "active",
    "coordinates",
    "levels",
    "parents",
    "protected",
    "removal",
)


def _array(value: ArrayLike, name: str, /) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.hasobject or not result.dtype.isnative:
        raise ValueError(f"{name} must be a native numeric array.")
    return result


def _fingerprint_arrays(
    kind: str,
    ids: np.ndarray,
    fields: tuple[tuple[str, np.ndarray], ...],
    /,
    *,
    dimension: int,
    ambient_dimension: int,
    next_vertex_id: int,
    next_cell_id: int,
) -> str:
    return canonical_fingerprint(
        {
            "kind": kind,
            "dimension": dimension,
            "ambient_dimension": ambient_dimension,
            "ids": canonical_array_hex_fingerprint(ids),
            "fields": tuple(
                (name, canonical_array_hex_fingerprint(value)) for name, value in fields
            ),
            "next_vertex_id": next_vertex_id,
            "next_cell_id": next_cell_id,
        }
    )


def _validate_ids(value: np.ndarray, name: str, /) -> None:
    if (
        value.ndim != 1
        or value.dtype.kind not in "iu"
        or (value.size and np.any(value < 0))
        or (value.size and np.any(np.diff(value.astype(np.int64)) <= 0))
    ):
        raise ValueError(f"{name} must be a strictly increasing stable-ID vector.")


def _field_bank(
    fields: Mapping[str, ArrayLike], names: tuple[str, ...], row_count: int, name: str, /
) -> tuple[tuple[str, np.ndarray], ...]:
    if set(fields) != set(names):
        raise ValueError(f"{name} fields do not match the stable forest schema.")
    bank = tuple((field, _array(fields[field], f"{name}.{field}")) for field in names)
    for field, value in bank:
        if value.ndim == 0 or value.shape[0] != row_count:
            raise ValueError(f"{name}.{field} rows do not match their stable IDs.")
    return bank


def _row_for_ids(ids: np.ndarray, queries: np.ndarray, name: str, /) -> np.ndarray:
    positions = np.searchsorted(ids, queries)
    safe = np.minimum(positions, max(ids.size - 1, 0))
    if ids.size == 0:
        if np.any(queries >= 0):
            raise ValueError(f"{name} refers to a missing stable ID.")
        return np.full(queries.shape, -1, dtype=np.int64)
    found = (queries < 0) | ((positions < ids.size) & (ids[safe] == queries))
    if not np.all(found):
        raise ValueError(f"{name} refers to a missing stable ID.")
    return np.where(queries >= 0, safe, -1).astype(np.int64)


def _validate_links(
    cell_ids: np.ndarray,
    cell_fields: tuple[tuple[str, np.ndarray], ...],
    vertex_ids: np.ndarray,
    vertex_fields: tuple[tuple[str, np.ndarray], ...],
    /,
    *,
    dimension: int,
) -> None:
    cells = dict(cell_fields)
    vertices = dict(vertex_fields)
    if cells["cells"].shape != (cell_ids.size, dimension + 1):
        raise ValueError("Bisection forest cell rows have the wrong simplex width.")
    if cells["tuples"].shape != (cell_ids.size, dimension + 1):
        raise ValueError("Bisection forest tuple rows have the wrong simplex width.")
    if cells["children_ids"].shape != (cell_ids.size, 2):
        raise ValueError("Bisection forest child rows must have width two.")
    if cells["facet_classes"].shape != (cell_ids.size, dimension + 1):
        raise ValueError("Bisection forest facet classes have the wrong width.")
    if vertices["coordinates"].ndim != 2:
        raise ValueError("Bisection forest vertex coordinates must be a matrix.")
    if vertices["parents"].shape != (vertex_ids.size, 2):
        raise ValueError("Bisection forest vertex-parent rows must have width two.")
    if not np.all(np.isfinite(vertices["coordinates"])):
        raise ValueError("Bisection forest coordinates must be finite.")
    for field in ("cells", "tuples"):
        if np.any(cells[field] < 0):
            raise ValueError(f"Cell {field} contain a negative vertex ID.")
        _row_for_ids(vertex_ids, cells[field], f"cell {field}")
    if np.any(cells["parent_ids"] < -1) or np.any(cells["children_ids"] < -1):
        raise ValueError("Bisection forest uses an invalid negative cell ID.")
    if np.any(vertices["parents"] < -1):
        raise ValueError("Bisection forest uses an invalid negative vertex ID.")
    _row_for_ids(vertex_ids, cells["bisection_vertex_ids"], "bisection vertex")
    parent_ids = cells["parent_ids"]
    parent_ordinals = cells["parent_ordinals"]
    children = cells["children_ids"]
    bisection = cells["bisection_vertex_ids"]
    _row_for_ids(cell_ids, parent_ids, "cell parent")
    _row_for_ids(cell_ids, children, "cell child")
    roots = parent_ids < 0
    if np.any(parent_ordinals[roots] != -1) or np.any(
        (parent_ordinals[~roots] < 0) | (parent_ordinals[~roots] > 1)
    ):
        raise ValueError("Bisection forest parent ordinals are malformed.")
    if np.any((children[:, 0] < 0) != (children[:, 1] < 0)):
        raise ValueError("A bisection record must have zero or two child IDs.")
    if np.any(bisection < -1) or np.any((children[:, 0] >= 0) != (bisection >= 0)):
        raise ValueError("Bisection forest midpoint records disagree with child records.")
    child_position = _row_for_ids(cell_ids, children, "cell child")
    for parent_row in np.flatnonzero(children[:, 0] >= 0):
        for ordinal in range(2):
            child_row = int(child_position[parent_row, ordinal])
            if (
                parent_ids[child_row] != cell_ids[parent_row]
                or parent_ordinals[child_row] != ordinal
            ):
                raise ValueError("Bisection forest parent and child links disagree.")
        if bisection[parent_row] < 0:
            raise ValueError("A split cell has no stable bisection vertex.")
    if np.any(cells["active"] & cells["retired"]):
        raise ValueError("An active bisection cell cannot be retired.")
    vertex_parents = vertices["parents"]
    no_parents = np.all(vertex_parents < 0, axis=1)
    complete_parents = np.all(vertex_parents >= 0, axis=1)
    if not np.all(no_parents | complete_parents):
        raise ValueError("A bisection vertex has an incomplete parent pair.")
    _row_for_ids(vertex_ids, vertex_parents, "vertex parent")
    parent_rows = _row_for_ids(cell_ids, parent_ids, "cell parent")
    for start in np.flatnonzero(~roots):
        current = int(start)
        visited: set[int] = set()
        while parent_ids[current] >= 0:
            if current in visited:
                raise ValueError("Bisection forest contains a parent cycle.")
            visited.add(current)
            current = int(parent_rows[current])
        if current in visited:
            raise ValueError("Bisection forest contains a parent cycle.")


def _validate_protected_edges(edges: np.ndarray, vertex_ids: np.ndarray, /) -> None:
    _validate_ids(vertex_ids, "vertex_ids")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Protected edge IDs must be a two-column matrix.")
    if np.any(edges < 0) or np.any(edges[:, 0] >= edges[:, 1]):
        raise ValueError("Protected edge IDs must be canonical increasing pairs.")
    _row_for_ids(vertex_ids, edges, "protected edge")
    if edges.shape[0] > 1:
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        if not np.array_equal(order, np.arange(edges.shape[0])) or np.any(
            np.all(edges[1:] == edges[:-1], axis=1)
        ):
            raise ValueError("Protected edge IDs must be sorted and unique.")


@final
class SimplexForestRecords(StrictModule, NonTrainableState):
    """Canonical ID-keyed cell and vertex history, independent of owner placement."""

    cell_ids: Array
    cell_owner_ranks: Array
    cell_fields: tuple[tuple[str, Array], ...]
    vertex_ids: Array
    vertex_owner_ranks: Array
    vertex_fields: tuple[tuple[str, Array], ...]
    protected_edge_ids: Array
    dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    next_vertex_id: int = eqx.field(static=True)
    next_cell_id: int = eqx.field(static=True)
    forest_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_ids: ArrayLike,
        cell_owner_ranks: ArrayLike,
        cell_fields: Mapping[str, ArrayLike],
        vertex_ids: ArrayLike,
        vertex_owner_ranks: ArrayLike,
        vertex_fields: Mapping[str, ArrayLike],
        protected_edge_ids: ArrayLike,
        dimension: int,
        ambient_dimension: int,
        next_vertex_id: int,
        next_cell_id: int,
        /,
    ) -> None:
        cells = _array(cell_ids, "cell_ids").astype(np.int64, copy=False)
        vertices = _array(vertex_ids, "vertex_ids").astype(np.int64, copy=False)
        _validate_ids(cells, "cell_ids")
        _validate_ids(vertices, "vertex_ids")
        cell_owner = _array(cell_owner_ranks, "cell_owner_ranks").astype(
            np.int32, copy=False
        )
        vertex_owner = _array(vertex_owner_ranks, "vertex_owner_ranks").astype(
            np.int32, copy=False
        )
        if (
            cell_owner.shape != cells.shape
            or vertex_owner.shape != vertices.shape
            or np.any(cell_owner < 0)
            or np.any(vertex_owner < 0)
            or isinstance(dimension, (bool, np.bool_))
            or not isinstance(dimension, (int, np.integer))
            or int(dimension) not in (2, 3)
            or isinstance(ambient_dimension, (bool, np.bool_))
            or not isinstance(ambient_dimension, (int, np.integer))
            or int(ambient_dimension) < int(dimension)
            or isinstance(next_vertex_id, (bool, np.bool_))
            or not isinstance(next_vertex_id, (int, np.integer))
            or isinstance(next_cell_id, (bool, np.bool_))
            or not isinstance(next_cell_id, (int, np.integer))
            or int(next_vertex_id) <= (int(vertices[-1]) if vertices.size else -1)
            or int(next_cell_id) <= (int(cells[-1]) if cells.size else -1)
        ):
            raise ValueError(
                "Bisection forest owner, dimension, or cursor data is malformed."
            )
        cell_bank = _field_bank(cell_fields, _CELL_FIELDS, cells.size, "cell")
        vertex_bank = _field_bank(vertex_fields, _VERTEX_FIELDS, vertices.size, "vertex")
        protected_edges = _array(protected_edge_ids, "protected_edge_ids").astype(
            np.int64, copy=False
        )
        _validate_protected_edges(protected_edges, vertices)
        _validate_links(cells, cell_bank, vertices, vertex_bank, dimension=int(dimension))
        fingerprint = canonical_fingerprint(
            {
                "cell_forest": _fingerprint_arrays(
                    "simplex-bisection-cell-forest",
                    cells,
                    cell_bank,
                    dimension=int(dimension),
                    ambient_dimension=int(ambient_dimension),
                    next_vertex_id=int(next_vertex_id),
                    next_cell_id=int(next_cell_id),
                ),
                "vertex_forest": _fingerprint_arrays(
                    "simplex-bisection-vertex-forest",
                    vertices,
                    vertex_bank,
                    dimension=int(dimension),
                    ambient_dimension=int(ambient_dimension),
                    next_vertex_id=int(next_vertex_id),
                    next_cell_id=int(next_cell_id),
                ),
                "protected_edge_ids": canonical_array_hex_fingerprint(protected_edges),
            }
        )
        self.cell_ids = jnp.asarray(cells, dtype=jnp.int64)
        self.cell_owner_ranks = jnp.asarray(cell_owner, dtype=jnp.int32)
        self.cell_fields = tuple((name, jnp.asarray(value)) for name, value in cell_bank)
        self.vertex_ids = jnp.asarray(vertices, dtype=jnp.int64)
        self.vertex_owner_ranks = jnp.asarray(vertex_owner, dtype=jnp.int32)
        self.vertex_fields = tuple(
            (name, jnp.asarray(value)) for name, value in vertex_bank
        )
        self.protected_edge_ids = jnp.asarray(protected_edges, dtype=jnp.int64)
        self.dimension = int(dimension)
        self.ambient_dimension = int(ambient_dimension)
        self.next_vertex_id = int(next_vertex_id)
        self.next_cell_id = int(next_cell_id)
        self.forest_id = fingerprint

    def cell_field_map(self) -> dict[str, Array]:
        """Return named ID-aligned cell records."""
        return dict(self.cell_fields)

    def vertex_field_map(self) -> dict[str, Array]:
        """Return named ID-aligned vertex records."""
        return dict(self.vertex_fields)


def validate_simplex_forest(records: SimplexForestRecords, /) -> None:
    """Recheck every stable-ID reference before transfer or restart."""
    if not isinstance(records, SimplexForestRecords):
        raise TypeError("records must be SimplexForestRecords.")
    _validate_links(
        np.asarray(records.cell_ids, dtype=np.int64),
        tuple((name, np.asarray(value)) for name, value in records.cell_fields),
        np.asarray(records.vertex_ids, dtype=np.int64),
        tuple((name, np.asarray(value)) for name, value in records.vertex_fields),
        dimension=records.dimension,
    )
    _validate_protected_edges(
        np.asarray(records.protected_edge_ids, dtype=np.int64),
        np.asarray(records.vertex_ids, dtype=np.int64),
    )


def _owner_rows(
    values: ArrayLike | None,
    ids: np.ndarray,
    slots: np.ndarray,
    capacity: int,
    name: str,
    /,
) -> np.ndarray:
    if values is None:
        return np.zeros(ids.shape, dtype=np.int32)
    raw = _array(values, name)
    if raw.dtype.kind not in "iu" or raw.ndim != 1:
        raise ValueError(f"{name} must be an integer rank vector.")
    if raw.shape == (capacity,):
        raw = raw[slots]
    if raw.shape != ids.shape:
        raise ValueError(f"{name} must align with allocated stable IDs.")
    owners = raw.astype(np.int32, copy=False)
    if np.any(owners < 0):
        raise ValueError(f"{name} cannot contain a negative owner rank.")
    return owners


def _stable_references(
    references: np.ndarray, ids: np.ndarray, name: str, /
) -> np.ndarray:
    result = np.full(references.shape, -1, dtype=np.int64)
    valid = references >= 0
    if np.any(references[valid] >= ids.size):
        raise ValueError(f"{name} refers outside its allocated slot bank.")
    if np.any(ids[references[valid]] < 0):
        raise ValueError(f"{name} refers to an unallocated slot.")
    result[valid] = ids[references[valid]]
    return result


def extract_simplex_forest(
    state: AdaptiveSimplexState,
    /,
    *,
    cell_owner_ranks: ArrayLike | None = None,
    vertex_owner_ranks: ArrayLike | None = None,
) -> SimplexForestRecords:
    """Extract allocated records and convert every slot reference to a stable ID."""
    if not isinstance(state, AdaptiveSimplexState):
        raise TypeError("state must be AdaptiveSimplexState.")
    host = jax.device_get(state)
    mesh = host.mesh
    all_cell_ids = np.asarray(mesh.cell_ids, dtype=np.int64)
    all_vertex_ids = np.asarray(mesh.vertex_ids, dtype=np.int64)
    cell_slots = np.flatnonzero(all_cell_ids >= 0)
    vertex_slots = np.flatnonzero(all_vertex_ids >= 0)
    cell_ids = all_cell_ids[cell_slots]
    vertex_ids = all_vertex_ids[vertex_slots]
    if np.any(np.diff(cell_ids) <= 0) or np.any(np.diff(vertex_ids) <= 0):
        raise ValueError("Adaptive simplex records must be ordered by stable ID.")
    parent_codes = np.asarray(host.parents, dtype=np.int64)[cell_slots]
    parent_slots = np.where(parent_codes >= 0, parent_codes // 2, -1)
    parent_ids = _stable_references(parent_slots, all_cell_ids, "cell parent")
    parent_ordinals = np.where(parent_codes >= 0, parent_codes % 2, -1).astype(np.int32)
    children = _stable_references(
        np.asarray(host.children, dtype=np.int64)[cell_slots],
        all_cell_ids,
        "cell child",
    )
    cells = _stable_references(
        np.asarray(mesh.cells, dtype=np.int64)[cell_slots], all_vertex_ids, "cell vertex"
    )
    tuples = _stable_references(
        np.asarray(host.tuples, dtype=np.int64)[cell_slots],
        all_vertex_ids,
        "cell tuple",
    )
    bisection_vertices = _stable_references(
        np.asarray(host.bisection_vertices, dtype=np.int64)[cell_slots],
        all_vertex_ids,
        "bisection vertex",
    )
    vertex_parents = _stable_references(
        np.asarray(host.vertex_parents, dtype=np.int64)[vertex_slots],
        all_vertex_ids,
        "vertex parent",
    )
    cell_owners = _owner_rows(
        cell_owner_ranks, cell_ids, cell_slots, all_cell_ids.size, "cell_owner_ranks"
    )
    vertex_owners = _owner_rows(
        vertex_owner_ranks,
        vertex_ids,
        vertex_slots,
        all_vertex_ids.size,
        "vertex_owner_ranks",
    )
    vertex_capacity = all_vertex_ids.size
    protected_codes = np.asarray(host.protected_codes, dtype=np.int64)
    protected_slot_pairs = np.stack(
        (protected_codes // vertex_capacity, protected_codes % vertex_capacity), axis=1
    )
    protected_slot_pairs = protected_slot_pairs[protected_codes != np.iinfo(np.int64).max]
    protected_pairs = _stable_references(
        protected_slot_pairs,
        all_vertex_ids,
        "protected edge endpoint",
    )
    protected_pairs = np.unique(np.sort(protected_pairs, axis=1), axis=0)
    cell_fields = {
        "active": np.asarray(mesh.cell_active, dtype=np.bool_)[cell_slots],
        "blocks": np.asarray(host.blocks, dtype=np.int32)[cell_slots],
        "bisection_vertex_ids": bisection_vertices,
        "cell_classes": np.asarray(host.cell_classes, dtype=np.int32)[cell_slots],
        "cells": cells,
        "children_ids": children,
        "facet_classes": np.asarray(host.facet_classes, dtype=np.int32)[cell_slots],
        "generations": np.asarray(host.generations, dtype=np.int32)[cell_slots],
        "parent_ids": parent_ids,
        "parent_ordinals": parent_ordinals,
        "refine_rejected": np.asarray(host.refine_rejected, dtype=np.bool_)[cell_slots],
        "retired": np.asarray(host.retired, dtype=np.bool_)[cell_slots],
        "tags": np.asarray(host.tags, dtype=np.int32)[cell_slots],
        "tuples": tuples,
        "coarsen_marked": np.asarray(host.coarsen_marked, dtype=np.bool_)[cell_slots],
    }
    vertex_fields = {
        "active": np.asarray(mesh.vertex_active, dtype=np.bool_)[vertex_slots],
        "coordinates": np.asarray(mesh.coordinates)[vertex_slots],
        "levels": np.asarray(host.vertex_levels, dtype=np.int32)[vertex_slots],
        "parents": vertex_parents,
        "protected": np.asarray(host.vertex_protected, dtype=np.bool_)[vertex_slots],
        "removal": np.asarray(host.vertex_removal, dtype=np.int32)[vertex_slots],
    }
    dimension = mesh.dimension
    next_vertex_id, next_cell_id = np.asarray(host.cursors, dtype=np.int64)[2:4]
    records = SimplexForestRecords(
        cell_ids,
        cell_owners,
        cell_fields,
        vertex_ids,
        vertex_owners,
        vertex_fields,
        protected_pairs,
        dimension,
        mesh.ambient_dimension,
        int(next_vertex_id),
        int(next_cell_id),
    )
    return records


def reconstruct_adaptive_simplex_state(
    records: SimplexForestRecords,
    template: AdaptiveSimplexState,
    layout: AdaptiveSimplexLayout,
    /,
) -> AdaptiveSimplexState:
    """Rebuild a device state from stable IDs while retaining epoch clocks."""
    if not isinstance(records, SimplexForestRecords):
        raise TypeError("records must be SimplexForestRecords.")
    if not isinstance(template, AdaptiveSimplexState):
        raise TypeError("template must be AdaptiveSimplexState.")
    if not isinstance(layout, AdaptiveSimplexLayout):
        raise TypeError("layout must be AdaptiveSimplexLayout.")
    if (
        records.dimension != layout.dimension
        or records.ambient_dimension != layout.ambient_dimension
        or template.mesh.dimension != layout.dimension
        or template.mesh.ambient_dimension != layout.ambient_dimension
        or np.dtype(dict(records.vertex_fields)["coordinates"].dtype)
        != layout.coordinate_dtype
        or records.cell_ids.shape[0] > layout.cell_capacity
        or records.vertex_ids.shape[0] > layout.vertex_capacity
    ):
        raise ValueError("Stable forest does not fit the requested simplex layout.")
    cell_ids = np.asarray(records.cell_ids, dtype=np.int64)
    vertex_ids = np.asarray(records.vertex_ids, dtype=np.int64)
    cell_fields = {name: np.asarray(value) for name, value in records.cell_fields}
    vertex_fields = {name: np.asarray(value) for name, value in records.vertex_fields}
    cells = _row_for_ids(vertex_ids, cell_fields["cells"], "cell vertex")
    tuples = _row_for_ids(vertex_ids, cell_fields["tuples"], "cell tuple")
    vertex_parents = _row_for_ids(vertex_ids, vertex_fields["parents"], "vertex parent")
    parent_rows = _row_for_ids(cell_ids, cell_fields["parent_ids"], "cell parent")
    parents = np.where(
        cell_fields["parent_ids"] >= 0,
        2 * parent_rows + cell_fields["parent_ordinals"],
        -1,
    )
    children = _row_for_ids(cell_ids, cell_fields["children_ids"], "cell child")
    bisection_vertices = _row_for_ids(
        vertex_ids, cell_fields["bisection_vertex_ids"], "bisection vertex"
    )
    protected_ids = np.asarray(records.protected_edge_ids, dtype=np.int64)
    local_protected = protected_ids[np.all(np.isin(protected_ids, vertex_ids), axis=1)]
    protected_edges = _row_for_ids(vertex_ids, local_protected, "protected edge")
    if protected_edges.shape[0] > layout.protected_edge_capacity:
        raise ValueError("Stable forest protected edges exceed the simplex layout.")

    def padded(
        values: np.ndarray,
        capacity: int,
        fill: int | bool,
        dtype: np.dtype,
        /,
    ) -> np.ndarray:
        result = np.full((capacity, *values.shape[1:]), fill, dtype=dtype)
        result[: values.shape[0]] = values
        return result

    state = adaptive_simplex_state(
        layout,
        coordinates=vertex_fields["coordinates"],
        vertex_ids=vertex_ids,
        vertex_active=vertex_fields["active"],
        vertex_parents=vertex_parents,
        vertex_protected=vertex_fields["protected"],
        cells=cells,
        tuples=tuples,
        tags=cell_fields["tags"],
        blocks=cell_fields["blocks"],
        generations=cell_fields["generations"],
        parents=parents,
        children=children,
        bisection_vertices=bisection_vertices,
        cell_ids=cell_ids,
        cell_active=cell_fields["active"],
        cell_classes=cell_fields["cell_classes"],
        facet_classes=cell_fields["facet_classes"],
        protected_edges=protected_edges,
        next_vertex_id=records.next_vertex_id,
        next_cell_id=records.next_cell_id,
    )
    return eqx.tree_at(
        lambda value: (
            value.retired,
            value.vertex_levels,
            value.vertex_removal,
            value.refine_rejected,
            value.coarsen_marked,
            value.clocks,
            value.counters,
        ),
        state,
        (
            jnp.asarray(
                padded(
                    cell_fields["retired"],
                    layout.cell_capacity,
                    False,
                    np.dtype(np.bool_),
                )
            ),
            jnp.asarray(
                padded(
                    vertex_fields["levels"],
                    layout.vertex_capacity,
                    0,
                    np.dtype(np.int32),
                )
            ),
            jnp.asarray(
                padded(
                    vertex_fields["removal"],
                    layout.vertex_capacity,
                    -1,
                    np.dtype(np.int32),
                )
            ),
            jnp.asarray(
                padded(
                    cell_fields["refine_rejected"],
                    layout.cell_capacity,
                    False,
                    np.dtype(np.bool_),
                )
            ),
            jnp.asarray(
                padded(
                    cell_fields["coarsen_marked"],
                    layout.cell_capacity,
                    False,
                    np.dtype(np.bool_),
                )
            ),
            template.clocks,
            template.counters,
        ),
    )


__all__ = [
    "SimplexForestRecords",
    "extract_simplex_forest",
    "reconstruct_adaptive_simplex_state",
    "validate_simplex_forest",
]
