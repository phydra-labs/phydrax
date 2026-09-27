#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sampled corner-Jacobian acceptance shared by every fixed-topology motion owner.

Finite-element mesh motion, unstructured finite-volume ALE, and variable-patch
ALE accept or reject a coordinate proposal through :class:`MotionValidityPlan`.
Corner frames sample the Jacobian determinant of every standard cell kind at
its vertices: for simplices and bilinear quadrilaterals this decides validity
exactly, for trilinear and rational cells it is a sampled necessary condition.
The traceable check runs inside compiled steps; the host-epoch proof owner is
:func:`certify_cell_geometry_validity`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntFlag
from functools import cache
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ._reference_cell import reference_cell_topology


_CORNER_DIMENSIONS = {
    "interval": 1,
    "triangle": 2,
    "quadrilateral": 2,
    "tetrahedron": 3,
    "hexahedron": 3,
    "prism": 3,
    "pyramid": 3,
}
_SIMPLEX_KINDS = frozenset(("interval", "triangle", "tetrahedron"))


@cache
def reference_corner_frames(cell_kind: str, /) -> tuple[np.ndarray, np.ndarray]:
    """Corner simplices ``(vertex, neighbors)`` positively oriented on the reference.

    Every vertex of a standard cell carries one frame spanned by its incident
    edges; the pyramid apex has four base neighbors and carries every cyclically
    consecutive triple of its base loop. Neighbor columns are swapped where
    needed so each frame has a positive reference determinant.
    """

    match cell_kind:
        case "interval":
            vertices = np.zeros((1,), dtype=np.int64)
            neighbors = np.ones((1, 1), dtype=np.int64)
        case "triangle" | "quadrilateral":
            arity = 3 if cell_kind == "triangle" else 4
            vertices = np.arange(arity, dtype=np.int64)
            neighbors = np.stack(((vertices + 1) % arity, (vertices - 1) % arity), axis=1)
        case "tetrahedron" | "hexahedron" | "prism" | "pyramid":
            topology = reference_cell_topology(cell_kind)
            reference = np.asarray(topology.vertices, dtype=np.float64)
            adjacency = {vertex: [] for vertex in range(reference.shape[0])}
            for start, stop in topology.entities[1]:
                adjacency[int(start)].append(int(stop))
                adjacency[int(stop)].append(int(start))
            rows = []
            for vertex, adjacent in adjacency.items():
                if len(adjacent) == 3:
                    rows.append((vertex, *adjacent))
                else:
                    cycle = sorted(adjacent)
                    rows.extend(
                        (vertex, *(cycle[(start + offset) % 4] for offset in range(3)))
                        for start in range(4)
                    )
            rows_ = np.asarray(rows, dtype=np.int64)
            frames = reference[rows_[:, 1:]] - reference[rows_[:, :1]]
            negative = np.linalg.det(np.swapaxes(frames, -1, -2)) < 0.0
            rows_[negative, 2:4] = rows_[negative, 3:1:-1]
            vertices = rows_[:, 0]
            neighbors = rows_[:, 1:]
        case _:
            raise ValueError(f"No corner Jacobian frames for cell kind {cell_kind!r}.")
    vertices.setflags(write=False)
    neighbors.setflags(write=False)
    return vertices, neighbors


def motion_corner_routes(cell_kind: str, /) -> tuple[np.ndarray, np.ndarray]:
    """Corner frames that decide motion validity and corner-weighted energies.

    Simplices carry one affine Jacobian, so their first frame represents the
    cell; every other kind keeps each vertex frame.
    """

    vertices, neighbors = reference_corner_frames(cell_kind)
    if cell_kind in _SIMPLEX_KINDS:
        return vertices[:1], neighbors[:1]
    return vertices, neighbors


def motion_cell_dimension(cell_kind: str, /) -> int:
    dimension = _CORNER_DIMENSIONS.get(cell_kind)
    if dimension is None:
        raise ValueError(f"Fixed-topology motion does not support {cell_kind!r} cells.")
    return dimension


def corner_frame_matrices(
    coordinates: Array, vertices: Array, neighbors: Array, /
) -> Array:
    """Frames with columns ``x[neighbor] - x[vertex]`` shaped (..., dim, dim)."""

    return jnp.swapaxes(
        coordinates[neighbors] - coordinates[vertices][..., None, :], -1, -2
    )


class MotionValidityStatus(IntFlag):
    """Reasons a fixed-topology coordinate proposal is rejected."""

    VALID = 0
    NONFINITE_COORDINATES = 1
    EXCESSIVE_DISPLACEMENT = 2
    JACOBIAN_TOO_SMALL = 4
    ORIENTATION_CHANGED = 8


class MotionValidityPolicy(StrictModule, NonTrainableState):
    """Acceptance thresholds for sampled corner Jacobians and displacement.

    ``minimum_absolute_jacobian`` bounds ``|det A|`` and
    ``minimum_relative_jacobian`` bounds the signed ratio against the reference
    corner determinant; zero floors accept every orientation-preserving proposal.
    ``maximum_displacement_fraction`` bounds the largest vertex displacement in
    units of the shortest reference edge; ``None`` disables the bound.
    """

    minimum_absolute_jacobian: float = eqx.field(static=True)
    minimum_relative_jacobian: float = eqx.field(static=True)
    maximum_displacement_fraction: float | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_absolute_jacobian: float = 1.0e-10,
        minimum_relative_jacobian: float = 0.05,
        maximum_displacement_fraction: float | None = 0.5,
    ) -> None:
        absolute = float(minimum_absolute_jacobian)
        relative = float(minimum_relative_jacobian)
        for name, value in (
            ("minimum_absolute_jacobian", absolute),
            ("minimum_relative_jacobian", relative),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        displacement = (
            None
            if maximum_displacement_fraction is None
            else float(maximum_displacement_fraction)
        )
        if displacement is not None and (
            not math.isfinite(displacement) or displacement <= 0.0
        ):
            raise ValueError("maximum_displacement_fraction must be positive or None.")
        self.minimum_absolute_jacobian = absolute
        self.minimum_relative_jacobian = relative
        self.maximum_displacement_fraction = displacement
        self.policy_id = canonical_fingerprint(
            {
                "kind": "motion-validity-policy",
                "minimum_absolute_jacobian": absolute,
                "minimum_relative_jacobian": relative,
                "maximum_displacement_fraction": displacement,
            }
        )


class MotionValidityEvidence(StrictModule):
    """Traceable sampled-validity evidence of one coordinate proposal.

    ``minimum_relative_jacobian`` is signed, so an inverted corner reports a
    negative ratio.
    """

    finite: Array
    orientation_preserved: Array
    minimum_absolute_jacobian: Array
    minimum_relative_jacobian: Array
    maximum_displacement_ratio: Array
    status: Array

    def __init__(
        self,
        *,
        finite: Any,
        orientation_preserved: Any,
        minimum_absolute_jacobian: Any,
        minimum_relative_jacobian: Any,
        maximum_displacement_ratio: Any,
        status: Any,
    ) -> None:
        self.finite = jnp.asarray(finite, dtype=jnp.bool_).reshape(())
        self.orientation_preserved = jnp.asarray(
            orientation_preserved, dtype=jnp.bool_
        ).reshape(())
        self.minimum_absolute_jacobian = jnp.asarray(minimum_absolute_jacobian).reshape(
            ()
        )
        self.minimum_relative_jacobian = jnp.asarray(minimum_relative_jacobian).reshape(
            ()
        )
        self.maximum_displacement_ratio = jnp.asarray(maximum_displacement_ratio).reshape(
            ()
        )
        self.status = jnp.asarray(status, dtype=jnp.int32).reshape(())

    @property
    def valid(self) -> Array:
        return self.status == int(MotionValidityStatus.VALID)


class MotionValidityPlan(StrictModule, NonTrainableState):
    """Prepared corner routes and reference determinants of one cell layout.

    ``cell_blocks`` lists ``(cell_kind, cells)`` pairs in the canonical vertex
    order of :func:`reference_cell_topology`; ``active_cells`` optionally masks
    padding cells per block. Cells report in block order through
    :meth:`cell_minimum_determinants`.
    """

    reference_coordinates: Array
    corner_vertices: Array
    corner_neighbors: Array
    corner_cells: Array
    corner_active: Array
    reference_determinants: Array
    vertex_active: Array
    length_scale: Array
    determinant_plan: SmallLinearSolvePlan
    policy: MotionValidityPolicy
    cell_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_coordinates: ArrayLike,
        cell_blocks: Sequence[tuple[str, ArrayLike]],
        /,
        *,
        policy: MotionValidityPolicy | None = None,
        active_cells: Sequence[ArrayLike] | None = None,
    ) -> None:
        policy_ = MotionValidityPolicy() if policy is None else policy
        if not isinstance(policy_, MotionValidityPolicy):
            raise TypeError("policy must be MotionValidityPolicy or None.")
        reference = np.asarray(reference_coordinates, dtype=np.float64)
        if (
            reference.ndim != 2
            or reference.shape[0] == 0
            or reference.shape[1] not in (1, 2, 3)
        ):
            raise ValueError("Reference coordinates must have shape (points, 1-3).")
        if not np.all(np.isfinite(reference)):
            raise ValueError("Reference coordinates must be finite.")
        blocks = tuple(cell_blocks)
        if not blocks:
            raise ValueError("Motion validity requires at least one cell block.")
        masks = (
            tuple(None for _ in blocks) if active_cells is None else tuple(active_cells)
        )
        if len(masks) != len(blocks):
            raise ValueError("active_cells must provide one mask per cell block.")
        dimension = reference.shape[1]
        vertex_rows = []
        neighbor_rows = []
        cell_rows = []
        active_rows = []
        offset = 0
        for (kind, cells), mask in zip(blocks, masks, strict=True):
            if motion_cell_dimension(str(kind)) != dimension:
                raise ValueError(
                    "Motion validity requires full-dimensional cells of the ambient "
                    "dimension."
                )
            table_vertices, table_neighbors = motion_corner_routes(str(kind))
            cells_ = np.asarray(cells, dtype=np.int64)
            arity = len(reference_cell_topology(str(kind)).vertices)
            if cells_.ndim != 2 or cells_.shape[1] != arity:
                raise ValueError(f"{kind} cells must have shape (cells, {arity}).")
            if np.any(cells_ < 0) or np.any(cells_ >= reference.shape[0]):
                raise ValueError("Cell vertices must index the reference coordinates.")
            active = (
                np.ones((cells_.shape[0],), dtype=np.bool_)
                if mask is None
                else np.asarray(mask, dtype=np.bool_)
            )
            if active.shape != (cells_.shape[0],):
                raise ValueError("Each active-cell mask must have one entry per cell.")
            corners = table_vertices.size
            vertex_rows.append(cells_[:, table_vertices].reshape(-1))
            neighbor_rows.append(cells_[:, table_neighbors].reshape(-1, dimension))
            cell_rows.append(np.repeat(np.arange(cells_.shape[0]) + offset, corners))
            active_rows.append(np.repeat(active, corners))
            offset += cells_.shape[0]
        corner_vertices = np.concatenate(vertex_rows)
        corner_neighbors = np.concatenate(neighbor_rows)
        corner_active = np.concatenate(active_rows)
        determinant_plan = SmallLinearSolvePlan(dimension)
        frames = np.asarray(
            corner_frame_matrices(
                jnp.asarray(reference),
                jnp.asarray(corner_vertices),
                jnp.asarray(corner_neighbors),
            )
        )
        determinants = np.asarray(
            determinant_small_linear(determinant_plan, jnp.asarray(frames))
        )
        if np.any(~np.isfinite(determinants[corner_active])) or np.any(
            determinants[corner_active] == 0.0
        ):
            raise ValueError("Reference geometry has a singular corner Jacobian.")
        edge_lengths = np.linalg.norm(frames, axis=-2)[corner_active]
        length_scale = float(np.min(edge_lengths)) if edge_lengths.size else 1.0
        vertex_active = np.zeros((reference.shape[0],), dtype=np.bool_)
        vertex_active[corner_vertices[corner_active]] = True
        vertex_active[corner_neighbors[corner_active].reshape(-1)] = True
        self.reference_coordinates = jnp.asarray(reference)
        self.corner_vertices = jnp.asarray(corner_vertices, dtype=jnp.int32)
        self.corner_neighbors = jnp.asarray(corner_neighbors, dtype=jnp.int32)
        self.corner_cells = jnp.asarray(np.concatenate(cell_rows), dtype=jnp.int32)
        self.corner_active = jnp.asarray(corner_active)
        self.reference_determinants = jnp.asarray(determinants)
        self.vertex_active = jnp.asarray(vertex_active)
        self.length_scale = jnp.asarray(length_scale, dtype=jnp.float64)
        self.determinant_plan = determinant_plan
        self.policy = policy_
        self.cell_count = offset
        self.plan_id = canonical_fingerprint(
            {
                "kind": "motion-validity-plan",
                "reference": array_tree_fingerprint(reference),
                "blocks": [str(kind) for kind, _ in blocks],
                "corner_vertices": array_tree_fingerprint(corner_vertices),
                "corner_neighbors": array_tree_fingerprint(corner_neighbors),
                "corner_active": array_tree_fingerprint(corner_active),
                "policy": policy_.policy_id,
            }
        )

    def corner_determinants(self, coordinates: ArrayLike, /) -> Array:
        """Signed corner Jacobian determinants of ``coordinates``."""

        points = jnp.asarray(coordinates)
        if points.shape != self.reference_coordinates.shape:
            raise ValueError("Coordinates must match the reference coordinate shape.")
        return determinant_small_linear(
            self.determinant_plan,
            corner_frame_matrices(points, self.corner_vertices, self.corner_neighbors),
        )

    def cell_minimum_determinants(self, coordinates: ArrayLike, /) -> Array:
        """Smallest signed corner determinant of every cell (inactive cells: +inf)."""

        determinants = jnp.where(
            self.corner_active, self.corner_determinants(coordinates), jnp.inf
        )
        return (
            jnp.full((self.cell_count,), jnp.inf, dtype=determinants.dtype)
            .at[self.corner_cells]
            .min(determinants)
        )

    def evaluate(self, coordinates: ArrayLike, /) -> MotionValidityEvidence:
        """Traceable acceptance evidence for one coordinate proposal."""

        points = jnp.asarray(coordinates)
        determinants = self.corner_determinants(points)
        active = self.corner_active
        finite = jnp.all(
            jnp.where(self.vertex_active[:, None], jnp.isfinite(points), True)
        ) & jnp.all(~active | jnp.isfinite(determinants))
        relative = determinants / self.reference_determinants
        orientation_preserved = jnp.all(~active | (relative > 0.0))
        minimum_absolute = jnp.min(jnp.where(active, jnp.abs(determinants), jnp.inf))
        minimum_relative = jnp.min(jnp.where(active, relative, jnp.inf))
        displacement = points - self.reference_coordinates
        norms = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        maximum_ratio = (
            jnp.max(jnp.where(self.vertex_active, norms, 0.0)) / self.length_scale
        )
        policy = self.policy
        displacement_limit = (
            jnp.inf
            if policy.maximum_displacement_fraction is None
            else policy.maximum_displacement_fraction
        )
        flags = (
            (~finite, MotionValidityStatus.NONFINITE_COORDINATES),
            (
                ~(maximum_ratio <= displacement_limit),
                MotionValidityStatus.EXCESSIVE_DISPLACEMENT,
            ),
            (
                ~(minimum_absolute >= policy.minimum_absolute_jacobian)
                | ~(minimum_relative >= policy.minimum_relative_jacobian),
                MotionValidityStatus.JACOBIAN_TOO_SMALL,
            ),
            (~orientation_preserved, MotionValidityStatus.ORIENTATION_CHANGED),
        )
        status = jnp.asarray(int(MotionValidityStatus.VALID), dtype=jnp.int32)
        for failed, flag in flags:
            status = status | jnp.where(failed, int(flag), 0).astype(jnp.int32)
        return MotionValidityEvidence(
            finite=finite,
            orientation_preserved=orientation_preserved,
            minimum_absolute_jacobian=minimum_absolute,
            minimum_relative_jacobian=minimum_relative,
            maximum_displacement_ratio=maximum_ratio,
            status=status,
        )


__all__ = [
    "MotionValidityEvidence",
    "MotionValidityPlan",
    "MotionValidityPolicy",
    "MotionValidityStatus",
]
