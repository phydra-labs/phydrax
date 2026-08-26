#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._boundary import FiniteVolumeBoundarySet, PrescribedNormalFluxBoundary
from ._high_resolution import (
    CharacteristicReconstructionPlan,
    HighResolutionReconstructionPlan,
    NonuniformWENOReconstructionPlan,
)
from ._mapped import MappedFiniteVolumeDiscretization
from ._reconstruction import AbstractFaceReconstructionPlan
from ._structured import FiniteVolumeDiscretization
from ._weno import WENOReconstructionPlan


def reconstruction_ghost_width(reconstruction: Any, /) -> int:
    if isinstance(reconstruction, AbstractFaceReconstructionPlan):
        return reconstruction.ghost_width
    if isinstance(reconstruction, CharacteristicReconstructionPlan):
        return reconstruction.reconstruction.radius
    if isinstance(reconstruction, HighResolutionReconstructionPlan):
        return reconstruction.radius
    if isinstance(reconstruction, NonuniformWENOReconstructionPlan):
        return 3
    if isinstance(reconstruction, WENOReconstructionPlan):
        return (reconstruction.order + 1) // 2
    raise TypeError("Unsupported reconstruction for finite-volume halo planning.")


class FiniteVolumeHaloPlan(StrictModule, NonTrainableState):
    """Symbolic physical/periodic halo requirements for one FV method."""

    discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization
    reconstruction: Any
    boundaries: FiniteVolumeBoundarySet
    depth: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        reconstruction: Any,
        boundaries: FiniteVolumeBoundarySet,
        /,
    ):
        if not isinstance(
            discretization,
            (FiniteVolumeDiscretization, MappedFiniteVolumeDiscretization),
        ):
            raise TypeError("Halo planning requires finite-volume geometry.")
        if not isinstance(boundaries, FiniteVolumeBoundarySet):
            raise TypeError("boundaries must be a FiniteVolumeBoundarySet.")
        if boundaries.axis_names != discretization.grid.axis_names:
            raise ValueError("Halo boundary axes must match finite-volume geometry.")
        depth = reconstruction_ghost_width(reconstruction)
        for axis, count in enumerate(discretization.cell_shape):
            if count < 2 * depth:
                raise ValueError(
                    f"Axis {discretization.grid.axis_names[axis]!r} has {count} cells, "
                    f"but reconstruction halo depth {depth} requires at least {2 * depth}."
                )
            periodic = discretization.grid.structured_axes[axis].periodic
            pair = boundaries.pairs[axis]
            if periodic != (pair is None):
                raise ValueError("Grid periodicity and halo boundary ownership disagree.")
        self.discretization = discretization
        self.reconstruction = reconstruction
        self.boundaries = boundaries
        self.depth = depth
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-volume-halo-plan",
                "discretization": discretization.prepared_id,
                "reconstruction": reconstruction.plan_id,
                "boundaries": boundaries.boundary_set_id,
                "depth": depth,
            }
        )

    def prepare(self, /) -> "PreparedFiniteVolumeHaloPlan":
        return PreparedFiniteVolumeHaloPlan(self)


class FiniteVolumeGhostedAxis(StrictModule):
    """One axis-local ghosted state and its physical axis coordinates."""

    values: Array
    physical_centers: Array
    axis_coordinates: Array
    axis: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    interior_slice: slice = eqx.field(static=True)


class PreparedFiniteVolumeHaloPlan(StrictModule, NonTrainableState):
    """Executable physical/periodic face halo authority."""

    plan: FiniteVolumeHaloPlan
    depth_by_axis: tuple[int, ...] = eqx.field(static=True)
    interior_slices: tuple[slice, ...] = eqx.field(static=True)
    needs_edge_halos: bool = eqx.field(static=True)
    needs_vertex_halos: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: FiniteVolumeHaloPlan, /):
        if not isinstance(plan, FiniteVolumeHaloPlan):
            raise TypeError("plan must be a FiniteVolumeHaloPlan.")
        dimension = len(plan.discretization.cell_shape)
        depths = (plan.depth,) * dimension
        self.plan = plan
        self.depth_by_axis = depths
        self.interior_slices = tuple(
            slice(depth, depth + count)
            for depth, count in zip(depths, plan.discretization.cell_shape, strict=True)
        )
        self.needs_edge_halos = dimension > 1 and plan.depth > 1
        self.needs_vertex_halos = dimension == 3 and plan.depth > 1
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-volume-halo",
                "plan": plan.plan_id,
                "depths": list(depths),
                "needs_edges": self.needs_edge_halos,
                "needs_vertices": self.needs_vertex_halos,
            }
        )

    def boundary_states(
        self,
        system: Any,
        time: Array,
        state: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array | None, Array | None]:
        axis_ = int(axis)
        discretization = self.plan.discretization
        structured_axis = discretization.grid.structured_axes[axis_]
        if structured_axis.periodic:
            return None, None
        pair = self.plan.boundaries.pairs[axis_]
        if pair is None:
            raise ValueError("Bounded halo axis is missing a boundary pair.")
        lower_interior = jnp.take(state, 0, axis=axis_)
        upper_interior = jnp.take(state, state.shape[axis_] - 1, axis=axis_)
        lower_coordinates = jnp.take(discretization.face_centers[axis_], 0, axis=axis_)
        upper_coordinates = jnp.take(
            discretization.face_centers[axis_],
            discretization.face_layouts[axis_].shape[axis_] - 1,
            axis=axis_,
        )
        lower = (
            lower_interior
            if isinstance(pair.lower, PrescribedNormalFluxBoundary)
            else pair.lower.exterior_state(
                system,
                time,
                lower_interior,
                lower_coordinates,
                discretization.outward_normal(axis_, "lower"),
                axis_,
                args,
            )
        )
        upper = (
            upper_interior
            if isinstance(pair.upper, PrescribedNormalFluxBoundary)
            else pair.upper.exterior_state(
                system,
                time,
                upper_interior,
                upper_coordinates,
                discretization.outward_normal(axis_, "upper"),
                axis_,
                args,
            )
        )
        return lower, upper

    def materialize_axis(
        self,
        system: Any,
        time: Array,
        state: ArrayLike,
        axis: int,
        args: Any = None,
        /,
    ) -> FiniteVolumeGhostedAxis:
        values = jnp.asarray(state)
        axis_ = int(axis)
        depth = self.depth_by_axis[axis_]
        moved = jnp.moveaxis(values, axis_, 0)
        center_values = jnp.moveaxis(self.plan.discretization.cell_centers, axis_, 0)
        structured_axis = self.plan.discretization.grid.structured_axes[axis_]
        centers = structured_axis.interval_centers
        if structured_axis.periodic:
            extent = structured_axis.bounds[1] - structured_axis.bounds[0]
            lower = moved[-depth:]
            upper = moved[:depth]
            lower_coordinates = centers[-depth:] - extent
            upper_coordinates = centers[:depth] + extent
            lower_physical = center_values[-depth:]
            upper_physical = center_values[:depth]
        else:
            pair = self.plan.boundaries.pairs[axis_]
            if pair is None:
                raise ValueError("Bounded halo axis is missing a boundary pair.")
            lower_face = jnp.take(
                self.plan.discretization.face_centers[axis_], 0, axis=axis_
            )
            upper_face = jnp.take(
                self.plan.discretization.face_centers[axis_],
                self.plan.discretization.face_layouts[axis_].shape[axis_] - 1,
                axis=axis_,
            )
            lower_layers = []
            upper_layers = []
            lower_center_layers = []
            upper_center_layers = []
            for layer in range(depth):
                lower_index = min(layer, moved.shape[0] - 1)
                upper_index = max(0, moved.shape[0] - 1 - layer)
                lower_interior = moved[lower_index]
                upper_interior = moved[upper_index]
                lower_layers.append(
                    lower_interior
                    if isinstance(pair.lower, PrescribedNormalFluxBoundary)
                    else pair.lower.exterior_state(
                        system,
                        time,
                        lower_interior,
                        lower_face,
                        self.plan.discretization.outward_normal(axis_, "lower"),
                        axis_,
                        args,
                    )
                )
                upper_layers.append(
                    upper_interior
                    if isinstance(pair.upper, PrescribedNormalFluxBoundary)
                    else pair.upper.exterior_state(
                        system,
                        time,
                        upper_interior,
                        upper_face,
                        self.plan.discretization.outward_normal(axis_, "upper"),
                        axis_,
                        args,
                    )
                )
                lower_center_layers.append(2.0 * lower_face - center_values[lower_index])
                upper_center_layers.append(2.0 * upper_face - center_values[upper_index])
            lower = jnp.stack(tuple(reversed(lower_layers)), axis=0)
            upper = jnp.stack(tuple(upper_layers), axis=0)
            lower_physical = jnp.stack(tuple(reversed(lower_center_layers)), axis=0)
            upper_physical = jnp.stack(tuple(upper_center_layers), axis=0)
            lower_coordinates = (2.0 * structured_axis.bounds[0] - centers[:depth])[::-1]
            upper_coordinates = (2.0 * structured_axis.bounds[1] - centers[-depth:])[::-1]
        ghosted = jnp.concatenate((lower, moved, upper), axis=0)
        coordinates = jnp.concatenate((lower_coordinates, centers, upper_coordinates))
        physical_centers = jnp.concatenate(
            (lower_physical, center_values, upper_physical), axis=0
        )
        return FiniteVolumeGhostedAxis(
            values=jnp.moveaxis(ghosted, 0, axis_),
            physical_centers=jnp.moveaxis(physical_centers, 0, axis_),
            axis_coordinates=coordinates,
            axis=axis_,
            depth=depth,
            interior_slice=slice(depth, depth + moved.shape[0]),
        )

    def ghosted_axis(
        self,
        system: Any,
        time: Array,
        state: ArrayLike,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        return self.materialize_axis(system, time, state, axis, args).values


__all__ = [
    "FiniteVolumeGhostedAxis",
    "FiniteVolumeHaloPlan",
    "PreparedFiniteVolumeHaloPlan",
    "reconstruction_ghost_width",
]
