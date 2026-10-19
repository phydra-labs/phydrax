from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.discretization._topology_rows import has_duplicate_rows

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._reference_cell import (
    reference_cell_topology,
    REFERENCE_TOPOLOGIES,
    ReferenceCellTopology,
)


@runtime_checkable
class CellGeometryElement(Protocol):
    """Common layout metadata, including nonpolynomial vertex descriptors.

    Scalar tabulation and source order belong to validated concrete coordinate
    elements, not to every layout descriptor. Source order alone does not
    establish polynomiality; certificates inspect the represented source expressions.
    """

    @property
    def cell_kind(self) -> str: ...

    @property
    def conformity(self) -> str: ...

    @property
    def local_dof_count(self) -> int: ...

    @property
    def element_id(self) -> str: ...


def _admit_cell_kind(kind: str, /) -> ReferenceCellTopology | None:
    if (
        kind not in REFERENCE_TOPOLOGIES
        and kind not in ("polygon", "polyhedron")
        and not kind.startswith(("simplex:", "tensor:"))
    ):
        raise ValueError(
            f"CellBlock.cell_kind must name a reference cell, simplex:N, tensor:N, polygon, or polyhedron; got {kind!r}."
        )
    return None if kind in ("polygon", "polyhedron") else reference_cell_topology(kind)


def _admit_cell_rows(
    kind: str, reference: ReferenceCellTopology | None, vertices: ArrayLike, /
) -> tuple[np.ndarray, int]:
    """Own reference-cell arity and nonempty rank-two connectivity admission."""
    cells = np.asarray(vertices, dtype=np.int32)
    if reference is None:
        arity = cells.shape[1] if cells.ndim == 2 else -1
        minimum_arity = 4 if kind == "polyhedron" else 3
    else:
        arity = len(reference.vertices)
        minimum_arity = 2 if reference.dimension == 1 else 3
    if (
        cells.ndim != 2
        or cells.shape[0] == 0
        or cells.shape[1] != arity
        or arity < minimum_arity
        or (kind == "polygon" and arity < 5)
    ):
        raise ValueError(
            f"CellBlock.vertices must match reference arity {arity} for {kind} with nonempty rank-two rows; got {cells.shape}."
        )
    return cells, minimum_arity


def _admit_active_cell_vertices(
    kind: str, cells: np.ndarray, vertex_valid: ArrayLike | None, minimum_arity: int, /
) -> np.ndarray:
    """Validate fixed-capacity padding and active vertex-set uniqueness."""
    valid = (
        np.ones_like(cells, dtype=np.bool_)
        if vertex_valid is None
        else np.asarray(vertex_valid, dtype=np.bool_)
    )
    if valid.shape != cells.shape:
        raise ValueError(
            f"CellBlock.vertex_valid must match cell vertex storage; got {valid.shape}."
        )
    if kind != "polyhedron" and not np.all(valid):
        raise ValueError(
            f"CellBlock.vertex_valid must contain padding only for polyhedron blocks; got {kind!r}."
        )
    if np.any(np.sum(valid, axis=1) < minimum_arity):
        raise ValueError(
            f"CellBlock.vertices must contain at least {minimum_arity} active vertices per {kind} cell; got {cells.shape}."
        )
    if np.any(cells[valid] < 0):
        raise ValueError(
            f"CellBlock.vertices must contain nonnegative active vertex indices; got {cells.shape}."
        )
    # Padding sorts first as -1, so equal sorted rows are equal active sets.
    active_rows = np.sort(np.where(valid, cells, -1), axis=1)
    if np.any((active_rows[:, 1:] == active_rows[:, :-1]) & (active_rows[:, 1:] >= 0)):
        raise ValueError(
            f"CellBlock.vertices must reference distinct active vertices per cell; got {cells.shape}."
        )
    if has_duplicate_rows(active_rows):
        raise ValueError(
            f"CellBlock.vertices must contain no duplicate cells; got {cells.shape}."
        )
    return valid


def _admit_cell_global_ids(
    cell_count: int, global_ids: ArrayLike | None, /
) -> np.ndarray:
    """Admit one unique stable identity per cell without changing numeric representation."""
    ids = (
        np.arange(cell_count, dtype=np.int64)
        if global_ids is None
        else np.asarray(global_ids, dtype=np.int64)
    )
    if ids.shape != (cell_count,):
        raise ValueError(
            f"CellBlock.global_ids must have one entry per cell; got {ids.shape}."
        )
    if np.any(ids < 0) or np.unique(ids).size != ids.size:
        raise ValueError(
            f"CellBlock.global_ids must be unique nonnegative integers; got {ids.shape}."
        )
    return ids


class CellBlock(StrictModule, NonTrainableState):
    """One homogeneous, ordered block of top-dimensional cells."""

    name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    vertices: Array
    vertex_valid: Array
    global_ids: Array
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        cell_kind: str,
        vertices: ArrayLike,
        /,
        *,
        vertex_valid: ArrayLike | None = None,
        global_ids: ArrayLike | None = None,
    ) -> None:
        block_name = str(name)
        kind = str(cell_kind)
        if not block_name:
            raise ValueError(f"CellBlock.name must be nonempty; got {block_name!r}.")
        reference = _admit_cell_kind(kind)
        cells, minimum_arity = _admit_cell_rows(kind, reference, vertices)
        valid = _admit_active_cell_vertices(kind, cells, vertex_valid, minimum_arity)
        ids = _admit_cell_global_ids(cells.shape[0], global_ids)
        self.name = block_name
        self.cell_kind = kind
        self.vertices, self.vertex_valid, self.global_ids = jax.device_put(
            (cells, valid, ids)
        )
        self.block_id = canonical_fingerprint(
            {
                "kind": "cell-block",
                "name": block_name,
                "cell_kind": kind,
                "vertices": array_tree_fingerprint(cells),
                "vertex_valid": array_tree_fingerprint(valid),
                "global_ids": array_tree_fingerprint(ids),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: CellBlock, /) -> CellBlock:
        """Revalidate stored cell rows, activity masks, and global identities."""
        return cls(
            value.name,
            value.cell_kind,
            value.vertices,
            vertex_valid=value.vertex_valid,
            global_ids=value.global_ids,
        )

    @property
    def cell_count(self) -> int:
        return self.vertices.shape[0]

    @property
    def arity(self) -> int:
        return (
            self.vertices.shape[1]
            if self.cell_kind in ("polygon", "polyhedron")
            else len(reference_cell_topology(self.cell_kind).vertices)
        )

    @property
    def topological_dimension(self) -> int:
        if self.cell_kind == "polygon":
            return 2
        if self.cell_kind == "polyhedron":
            return 3
        return reference_cell_topology(self.cell_kind).dimension


class PolyhedralBlock(StrictModule, NonTrainableState):
    """One fixed-width block of arbitrary polyhedra grouped by vertex count."""

    name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    vertices: Array
    vertex_valid: Array
    global_ids: Array
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        vertices: ArrayLike,
        /,
        *,
        global_ids: ArrayLike | None = None,
    ) -> None:
        block_name = str(name)
        cells = np.asarray(vertices, dtype=np.int32)
        if not block_name:
            raise ValueError(
                f"PolyhedralBlock.name must be nonempty; got {block_name!r}."
            )
        if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] < 4:
            raise ValueError(
                f"PolyhedralBlock.vertices must have shape (cells > 0, arity >= 4); got {cells.shape}."
            )
        if np.any(cells < 0):
            raise ValueError(
                f"PolyhedralBlock.vertices must contain nonnegative vertex indices; got {cells.shape}."
            )
        ordered = np.sort(cells, axis=1)
        if np.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError(
                f"PolyhedralBlock.vertices must reference distinct vertices per cell; got {cells.shape}."
            )
        if has_duplicate_rows(ordered):
            raise ValueError(
                f"PolyhedralBlock.vertices must contain no duplicate cells; got {cells.shape}."
            )
        ids = (
            np.arange(cells.shape[0], dtype=np.int64)
            if global_ids is None
            else np.asarray(global_ids, dtype=np.int64)
        )
        if ids.shape != (cells.shape[0],):
            raise ValueError(
                f"PolyhedralBlock.global_ids must match the cell count; got {ids.shape}."
            )
        if np.any(ids < 0) or np.unique(ids).size != ids.size:
            raise ValueError(
                f"PolyhedralBlock.global_ids must be unique nonnegative integers; got {ids.shape}."
            )
        valid = np.ones_like(cells, dtype=np.bool_)
        self.name = block_name
        self.cell_kind = "polyhedron"
        self.vertices, self.vertex_valid, self.global_ids = jax.device_put(
            (cells, valid, ids)
        )
        self.block_id = canonical_fingerprint(
            {
                "kind": "polyhedral-block",
                "name": block_name,
                "vertices": array_tree_fingerprint(cells),
                "global_ids": array_tree_fingerprint(ids),
            }
        )

    @property
    def cell_count(self) -> int:
        return self.vertices.shape[0]

    @property
    def arity(self) -> int:
        return self.vertices.shape[1]

    @property
    def topological_dimension(self) -> int:
        return 3
