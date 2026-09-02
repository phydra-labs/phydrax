#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._cell_mesh import CellMesh


class CellLocationStatus(IntEnum):
    LOCATED = 0
    OUTSIDE = 1
    DEGENERATE_CELL = 2
    NONFINITE = 3
    RESOURCE_EXCEEDED = 4


class CellLocationResult(StrictModule):
    cell_ids: Array
    barycentric: Array
    inside: Array
    used_fallback: Array
    candidate_count: Array
    status: Array
    successful: Array
    locator_id: str = eqx.field(static=True)


class SegmentLocationResult(StrictModule):
    start: CellLocationResult
    end: CellLocationResult
    crossed: Array
    exited: Array
    successful: Array
    locator_id: str = eqx.field(static=True)


class PreparedSimplicialCellLocator(StrictModule, NonTrainableState):
    """Fixed-shape affine triangle/tetrahedron ownership locator."""

    mesh: CellMesh
    cells: Array
    origins: Array
    inverse_jacobians: Array
    tolerance: float = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    locator_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        /,
        *,
        tolerance: float = 1.0e-10,
        maximum_cells: int = 100_000,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind not in (
            "triangle",
            "tetrahedron",
        ):
            raise ValueError(
                "Simplicial locator requires one triangle or tetrahedron block."
            )
        block = mesh.blocks[0]
        cells = np.asarray(block.vertices, dtype=np.int32)
        if cells.shape[0] > int(maximum_cells):
            raise ValueError("Simplicial locator exceeds maximum_cells.")
        coordinates = np.asarray(mesh.coordinates, dtype=float)
        vertices = coordinates[cells]
        origins = vertices[:, 0]
        jacobians = np.swapaxes(vertices[:, 1:] - origins[:, None, :], 1, 2)
        determinants = np.linalg.det(jacobians)
        if np.any(~np.isfinite(determinants)) or np.any(
            np.abs(determinants) <= float(tolerance)
        ):
            raise ValueError("Simplicial mesh contains degenerate cells.")
        inverse = np.linalg.solve(
            jacobians,
            np.eye(jacobians.shape[-1], dtype=jacobians.dtype),
        )
        self.mesh = mesh
        self.cells = jnp.asarray(cells)
        self.origins = jnp.asarray(origins)
        self.inverse_jacobians = jnp.asarray(inverse)
        self.tolerance = float(tolerance)
        self.maximum_cells = int(maximum_cells)
        self.locator_id = canonical_fingerprint(
            {
                "kind": "prepared-simplicial-cell-locator",
                "mesh": mesh.mesh_id,
                "tolerance": float(tolerance),
                "maximum_cells": int(maximum_cells),
            }
        )

    @property
    def dimension(self) -> int:
        return int(self.origins.shape[1])

    @property
    def cell_count(self) -> int:
        return int(self.cells.shape[0])

    def locate(self, points: ArrayLike, /) -> CellLocationResult:
        values = jnp.asarray(points, dtype=self.origins.dtype)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("Locator points must have shape (count, mesh dimension).")
        relative = values[:, None, :] - self.origins[None, :, :]
        reduced = contract("cij,pcj->pci", self.inverse_jacobians, relative)
        first = 1.0 - jnp.sum(reduced, axis=-1, keepdims=True)
        barycentric_all = jnp.concatenate((first, reduced), axis=-1)
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        contained = jnp.all(barycentric_all >= -self.tolerance, axis=-1) & jnp.all(
            barycentric_all <= 1.0 + self.tolerance, axis=-1
        )
        contained = contained & finite[:, None]
        any_cell = jnp.any(contained, axis=1)
        cell = jnp.argmax(contained.astype(jnp.int32), axis=1).astype(jnp.int32)
        safe = jnp.maximum(cell, 0)
        barycentric = barycentric_all[jnp.arange(values.shape[0]), safe]
        cell_ids = jnp.where(any_cell, cell, -1)
        status = jnp.where(
            finite,
            jnp.where(
                any_cell,
                int(CellLocationStatus.LOCATED),
                int(CellLocationStatus.OUTSIDE),
            ),
            int(CellLocationStatus.NONFINITE),
        ).astype(jnp.int32)
        return CellLocationResult(
            cell_ids,
            jnp.where(any_cell[:, None], barycentric, 0.0),
            any_cell,
            jnp.ones_like(any_cell),
            jnp.sum(contained, axis=1, dtype=jnp.int32),
            status,
            any_cell & finite,
            self.locator_id,
        )

    def locate_segment(
        self, start: ArrayLike, end: ArrayLike, /
    ) -> SegmentLocationResult:
        left = self.locate(start)
        right = self.locate(end)
        crossed = left.inside & right.inside & (left.cell_ids != right.cell_ids)
        exited = left.inside & ~right.inside
        successful = left.successful & (right.successful | exited)
        return SegmentLocationResult(
            left, right, crossed, exited, successful, self.locator_id
        )


__all__ = [
    "CellLocationResult",
    "CellLocationStatus",
    "PreparedSimplicialCellLocator",
    "SegmentLocationResult",
]
