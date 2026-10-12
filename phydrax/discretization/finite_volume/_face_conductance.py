#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from typing import assert_never

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from ...typing import checked, parse
from .._tensor_support import PreparedTensorGrid
from ._diffusion import _face_center_distances, FaceCoefficientPlan, FaceInterpolationKind
from ._precision import FiniteVolumePrecisionPolicy
from ._structured import _face_measure


class StructuredFaceConductancePlan(StrictModule, NonTrainableState):
    """Canonical active-cell edges with conductance ``k_face * A / d``.

    Coefficients are conductivity integrated over omitted dimensions; in two
    dimensions they are sheet conductance (conductivity times thickness).
    Full tensors are incompatible with this two-point flux representation.
    """

    grid: PreparedTensorGrid
    relation: EdgeRelation
    active_index: Array
    face_indices: Array
    geometry: Array
    interpolation: FaceCoefficientPlan
    precision: FiniteVolumePrecisionPolicy
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        grid: PreparedTensorGrid,
        active: ArrayLike,
        /,
        *,
        interpolation: FaceInterpolationKind = "harmonic",
        precision: FiniteVolumePrecisionPolicy | None = None,
    ) -> None:
        from ...topology import grid_adjacency_relation

        kind = parse(interpolation, FaceInterpolationKind, "interpolation")
        match kind:
            case "arithmetic" | "harmonic":
                pass
            case "upwind" | "callable":
                raise ValueError(
                    "Conductances require arithmetic or harmonic interpolation."
                )
            case _:
                assert_never(kind)
        mask = np.asarray(active)
        if mask.dtype.kind != "b" or mask.shape != grid.shape:
            raise ValueError("active must be a Boolean mask with the grid shape.")
        periodic = tuple(axis.periodic for axis in grid.structured_axes)
        if any(wraps and size < 3 for size, wraps in zip(grid.shape, periodic)):
            raise ValueError("Periodic conductance axes require at least three cells.")
        precision_ = FiniteVolumePrecisionPolicy() if precision is None else precision
        topology = grid_adjacency_relation(grid.shape, periodic)
        lower = np.asarray(topology.source_indices)
        upper = np.asarray(topology.target_indices)
        keep = mask.ravel()[lower] & mask.ravel()[upper]
        lower, upper = lower[keep], upper[keep]
        active_count = int(np.count_nonzero(mask))
        active_index = np.full(grid.shape, -1, dtype=np.int32)
        active_index[mask] = np.arange(active_count, dtype=np.int32)
        left = np.stack(np.unravel_index(lower, grid.shape), axis=-1)
        right = np.stack(np.unravel_index(upper, grid.shape), axis=-1)
        differs = left != right
        if np.any(np.sum(differs, axis=1) != 1):
            raise ValueError("Grid adjacency must differ along exactly one axis.")
        edge_axes = np.argmax(differs, axis=1)
        face_indices = np.empty(lower.size, dtype=np.int32)
        geometry_parts: list[Array] = []
        offset = 0
        for axis, name in enumerate(grid.axis_names):
            layout = grid.faces(name)
            selected = edge_axes == axis
            coordinates = right[selected].copy()
            wrap = periodic[axis] & (right[selected, axis] - left[selected, axis] > 1)
            coordinates[wrap, axis] = 0
            face_indices[selected] = offset + np.ravel_multi_index(
                tuple(coordinates.T), layout.shape
            )
            distance_shape = [1] * len(grid.shape)
            distance_shape[axis] = layout.shape[axis]
            distances = _face_center_distances(grid, axis).reshape(distance_shape)
            geometry_parts.append((_face_measure(grid, axis) / distances).ravel())
            offset += np.prod(layout.shape).item()
        self.grid = grid
        self.relation = EdgeRelation(
            active_index.ravel()[lower],
            active_index.ravel()[upper],
            source_size=active_count,
            target_size=active_count,
        )
        self.active_index = jnp.asarray(active_index, dtype=jnp.int32)
        self.face_indices = jnp.asarray(face_indices, dtype=jnp.int32)
        self.geometry = precision_.flux(
            jnp.concatenate(geometry_parts)[self.face_indices]
        )
        self.interpolation = FaceCoefficientPlan(grid, kind=kind)
        self.precision = precision_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "structured-face-conductance",
                "grid": grid.prepared_id,
                "active": array_tree_fingerprint(mask),
                "interpolation": kind,
                "precision": precision_.policy_id,
            }
        )

    def conductances(self, coefficient: ArrayLike, /) -> Array:
        """Return differentiable conductances in canonical relation route order."""
        value = jnp.asarray(coefficient)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise ValueError("Conductivity must be real.")
        dimension = len(self.grid.shape)
        if value.shape == ():
            value = jnp.broadcast_to(value, self.grid.shape)
        diagonal = value.shape == self.grid.shape + (dimension,)
        if value.shape != self.grid.shape and not diagonal:
            raise ValueError(
                "Coefficient must be scalar, isotropic cells, or per-axis diagonal cells; full tensors are refused."
            )
        value = self.precision.flux(value)
        faces = tuple(
            self.interpolation.interpolate(
                value[..., axis] if diagonal else value, name
            ).ravel()
            for axis, name in enumerate(self.grid.axis_names)
        )
        return self.precision.output(
            jnp.concatenate(faces)[self.face_indices] * self.geometry
        )

    def node_indices(self, cells: ArrayLike, /) -> np.ndarray:
        """Map host multi-indices to compact ids, refusing inactive cells."""
        indices = np.asarray(cells)
        if (
            indices.dtype.kind not in "iu"
            or indices.ndim != 2
            or indices.shape[1] != len(self.grid.shape)
        ):
            raise ValueError("cells must be an integer (m, dimension) matrix.")
        if np.any(indices < 0) or np.any(indices >= np.asarray(self.grid.shape)):
            raise ValueError("Cell indices are out of range.")
        nodes = np.asarray(self.active_index)[tuple(indices.T)]
        if np.any(nodes < 0):
            raise ValueError("Cell indices include inactive cells.")
        return np.asarray(nodes, dtype=np.int32)


__all__ = ["StructuredFaceConductancePlan"]
