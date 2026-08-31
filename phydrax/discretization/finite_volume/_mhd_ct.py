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
from .._structured_cochain import StructuredCochainBridge
from ._dynamics import PreparedFiniteVolumeDynamics
from ._mhd_boundary import ConstrainedMHDBoundarySet
from ._mhd_reconstruction import MHDPrimitiveReconstructionPlan
from ._riemann import AbstractNumericalFluxPlan
from ._uct import AbstractUCTElectromotivePlan, HLLUCTElectromotivePlan


class ConstrainedMagneticStateLayout(StrictModule, NonTrainableState):
    """Dimension-dependent ownership of ideal-MHD magnetic components."""

    dimension: int = eqx.field(static=True)
    reduced_component_indices: tuple[int, ...] = eqx.field(static=True)
    cochain_magnetic_indices: tuple[int, ...] = eqx.field(static=True)
    cell_magnetic_indices: tuple[int, ...] = eqx.field(static=True)
    magnetic_degree: int = eqx.field(static=True)
    electromotive_degree: int | None = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /):
        dimension_ = int(dimension)
        if dimension_ not in (1, 2, 3):
            raise ValueError(
                "Constrained magnetic layout dimension must be one, two, or three."
            )
        cochain = tuple(range(5, 5 + dimension_))
        cell_owned = tuple(range(5 + dimension_, 8))
        reduced = (0, 1, 2, 3, 4, *cell_owned)
        self.dimension = dimension_
        self.reduced_component_indices = reduced
        self.cochain_magnetic_indices = cochain
        self.cell_magnetic_indices = cell_owned
        self.magnetic_degree = dimension_ - 1
        self.electromotive_degree = dimension_ - 2 if dimension_ >= 2 else None
        self.layout_id = canonical_fingerprint(
            {
                "kind": "constrained-magnetic-state-layout",
                "dimension": dimension_,
                "reduced_components": list(reduced),
                "cochain_magnetic_components": list(cochain),
                "cell_magnetic_components": list(cell_owned),
            }
        )

    @property
    def reduced_component_count(self) -> int:
        return len(self.reduced_component_indices)

    def reduce_full_state(self, full_state: ArrayLike, /) -> Array:
        full = jnp.asarray(full_state)
        if full.shape[-1] != 8:
            raise ValueError("Ideal-MHD full state must have eight components.")
        return full[..., jnp.asarray(self.reduced_component_indices)]

    def expand_reduced_state(
        self,
        reduced_state: ArrayLike,
        cochain_cell_field: ArrayLike,
        /,
    ) -> Array:
        reduced = jnp.asarray(reduced_state)
        cochain_field = jnp.asarray(cochain_cell_field)
        if reduced.shape[-1] != self.reduced_component_count:
            raise ValueError("Reduced MHD component count does not match its layout.")
        if cochain_field.shape != reduced.shape[:-1] + (self.dimension,):
            raise ValueError("Cochain-owned cell magnetic field shape is invalid.")
        full = jnp.zeros(reduced.shape[:-1] + (8,), dtype=reduced.dtype)
        full = full.at[..., jnp.asarray(self.reduced_component_indices)].set(reduced)
        return full.at[..., jnp.asarray(self.cochain_magnetic_indices)].set(cochain_field)


class MHDCTRateResult(StrictModule):
    cell_rate: Array
    magnetic_rate: Array
    edge_electromotive_circulation: Array
    normal_fluxes: tuple[Array, ...]
    integrated_normal_fluxes: tuple[Array, ...]
    signal_speeds: tuple[Array, ...]
    stable_step: Array
    fallback_activated: Array
    uct_consistency_defect: Array
    uct_maximum_dissipation: Array


class UpwindConstrainedTransportPlan(StrictModule, NonTrainableState):
    """Periodic Cartesian flux-CT in one, two, or three physical dimensions."""

    dynamics: PreparedFiniteVolumeDynamics
    bridge: StructuredCochainBridge
    interface_solver: AbstractNumericalFluxPlan
    layout: ConstrainedMagneticStateLayout
    reconstruction: MHDPrimitiveReconstructionPlan
    electromotive_plan: AbstractUCTElectromotivePlan
    boundary_set: ConstrainedMHDBoundarySet | None
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedFiniteVolumeDynamics,
        bridge: StructuredCochainBridge,
        /,
        *,
        interface_solver: AbstractNumericalFluxPlan | None = None,
        reconstruction: MHDPrimitiveReconstructionPlan | None = None,
        electromotive_plan: AbstractUCTElectromotivePlan | None = None,
        boundary_set: ConstrainedMHDBoundarySet | None = None,
    ):
        if not isinstance(dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError("dynamics must be PreparedFiniteVolumeDynamics.")
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be StructuredCochainBridge.")
        if tuple(dynamics.system.component_names) != (
            "density",
            "momentum_x",
            "momentum_y",
            "momentum_z",
            "total_energy",
            "magnetic_x",
            "magnetic_y",
            "magnetic_z",
        ):
            raise TypeError("Constrained transport requires canonical ideal MHD.")
        dimension = dynamics.system.dimension
        if dynamics.discretization.grid.shape != bridge.grid.shape:
            raise ValueError("Finite-volume and cochain grids must match.")
        if bridge.dimension != dimension:
            raise ValueError("MHD system and cochain dimensions must match.")
        solver = (
            dynamics.method.interface_solver
            if interface_solver is None
            else interface_solver
        )
        if not isinstance(solver, AbstractNumericalFluxPlan):
            raise TypeError("interface_solver must implement AbstractNumericalFluxPlan.")
        layout = ConstrainedMagneticStateLayout(dimension)
        reconstruction_ = (
            MHDPrimitiveReconstructionPlan() if reconstruction is None else reconstruction
        )
        electromotive_ = (
            HLLUCTElectromotivePlan()
            if electromotive_plan is None
            else electromotive_plan
        )
        if not isinstance(reconstruction_, MHDPrimitiveReconstructionPlan):
            raise TypeError("reconstruction must be MHDPrimitiveReconstructionPlan.")
        if not isinstance(electromotive_, AbstractUCTElectromotivePlan):
            raise TypeError("electromotive_plan must implement the UCT contract.")
        if boundary_set is not None and not isinstance(
            boundary_set, ConstrainedMHDBoundarySet
        ):
            raise TypeError("boundary_set must be ConstrainedMHDBoundarySet or None.")
        if (
            any(not axis.periodic for axis in bridge.grid.structured_axes)
            and boundary_set is None
        ):
            raise ValueError("Nonperiodic constrained transport requires MHD boundaries.")
        if boundary_set is not None and boundary_set.axis_names != tuple(
            bridge.grid.axis_names
        ):
            raise ValueError("MHD boundary axes do not match the constrained grid.")
        self.dynamics = dynamics
        self.bridge = bridge
        self.interface_solver = solver
        self.layout = layout
        self.reconstruction = reconstruction_
        self.electromotive_plan = electromotive_
        self.boundary_set = boundary_set
        self.cell_shape = tuple(int(value) for value in bridge.grid.shape)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "upwind-constrained-transport",
                "dynamics": dynamics.dynamics_id,
                "bridge": bridge.bridge_id,
                "interface_solver": solver.flux_id,
                "layout": layout.layout_id,
                "reconstruction": reconstruction_.reconstruction_id,
                "electromotive": electromotive_.electromotive_id,
                "boundaries": (
                    None if boundary_set is None else boundary_set.boundary_set_id
                ),
            }
        )

    def validate_reduced_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        expected = self.cell_shape + (self.layout.reduced_component_count,)
        if value.shape != expected:
            raise ValueError(f"Reduced MHD state must have shape {expected}.")
        return value

    def validate_magnetic_flux(self, magnetic_flux: ArrayLike, /) -> Array:
        value = jnp.asarray(magnetic_flux)
        expected = self.bridge.cochain.cell_counts[self.layout.magnetic_degree]
        if value.shape != (expected,):
            raise ValueError(f"Magnetic flux must have shape ({expected},).")
        return value

    def cochain_cell_magnetic_field(self, magnetic_flux: ArrayLike, /) -> Array:
        faces = self.bridge.unpack_normal_flux(self.validate_magnetic_flux(magnetic_flux))
        centered = tuple(
            0.5 * (face + jnp.roll(face, 1, axis=axis))
            if self.bridge.grid.structured_axes[axis].periodic
            else 0.5
            * (
                jnp.take(face, jnp.arange(face.shape[axis] - 1), axis=axis)
                + jnp.take(face, jnp.arange(1, face.shape[axis]), axis=axis)
            )
            for axis, face in enumerate(faces)
        )
        return jnp.stack(centered, axis=-1)

    def full_state(self, reduced: ArrayLike, magnetic_flux: ArrayLike, /) -> Array:
        cell = self.validate_reduced_state(reduced)
        magnetic = self.cochain_cell_magnetic_field(magnetic_flux)
        return self.layout.expand_reduced_state(cell, magnetic)

    def magnetic_constraint(self, magnetic_flux: ArrayLike, /) -> Array:
        return self.bridge.exterior_derivative(
            self.layout.magnetic_degree,
            self.validate_magnetic_flux(magnetic_flux),
        )

    def _face_states(
        self,
        full_state: Array,
        normal_fields: tuple[Array, ...],
        axis: int,
        time: Array,
        args: Any,
        /,
    ) -> tuple[Array, Array]:
        if self.bridge.grid.structured_axes[axis].periodic:
            return self.reconstruction.reconstruct(
                self.dynamics.system,
                full_state,
                normal_fields[axis],
                axis,
            )
        if self.boundary_set is None:
            raise RuntimeError("Bounded MHD traces require a prepared boundary set.")
        lower_interior = jnp.take(full_state, 0, axis=axis)
        upper_interior = jnp.take(full_state, full_state.shape[axis] - 1, axis=axis)
        lower_normal = jnp.take(normal_fields[axis], 0, axis=axis)
        upper_normal = jnp.take(
            normal_fields[axis], normal_fields[axis].shape[axis] - 1, axis=axis
        )
        lower = self.boundary_set.boundary(axis, "lower").trace(
            self.dynamics.system,
            lower_interior,
            lower_normal,
            axis,
            "lower",
            time,
            args,
        )
        upper = self.boundary_set.boundary(axis, "upper").trace(
            self.dynamics.system,
            upper_interior,
            upper_normal,
            axis,
            "upper",
            time,
            args,
        )
        return self.reconstruction.reconstruct(
            self.dynamics.system,
            full_state,
            normal_fields[axis],
            axis,
            lower_exterior=lower.exterior_state,
            upper_exterior=upper.exterior_state,
        )

    def _average_to_edges(self, value: Array, axis: int, /) -> Array:
        if self.bridge.grid.structured_axes[axis].periodic:
            return 0.5 * (value + jnp.roll(value, -1, axis=axis))
        lower = jnp.take(value, jnp.asarray([0]), axis=axis)
        upper = jnp.take(value, jnp.asarray([value.shape[axis] - 1]), axis=axis)
        interior = 0.5 * (
            jnp.take(value, jnp.arange(value.shape[axis] - 1), axis=axis)
            + jnp.take(value, jnp.arange(1, value.shape[axis]), axis=axis)
        )
        return jnp.concatenate((lower, interior, upper), axis=axis)

    def _bounded_electromotive_components(
        self,
        face_fluxes: tuple[Array, ...],
        /,
    ) -> tuple[Array, ...]:
        if self.layout.dimension == 1:
            return ()
        if self.layout.dimension == 2:
            flux_x, flux_y = face_fluxes
            return (
                0.5
                * (
                    -self._average_to_edges(flux_x[..., 6], 1)
                    + self._average_to_edges(flux_y[..., 5], 0)
                ),
            )
        flux_x, flux_y, flux_z = face_fluxes
        ex = 0.5 * (
            -self._average_to_edges(flux_y[..., 7], 2)
            + self._average_to_edges(flux_z[..., 6], 1)
        )
        ey = 0.5 * (
            self._average_to_edges(flux_x[..., 7], 2)
            - self._average_to_edges(flux_z[..., 5], 0)
        )
        ez = 0.5 * (
            -self._average_to_edges(flux_x[..., 6], 1)
            + self._average_to_edges(flux_y[..., 5], 0)
        )
        return ex, ey, ez

    def rate(
        self,
        time: Array,
        reduced_state: ArrayLike,
        magnetic_flux: ArrayLike,
        args: Any = None,
        /,
        *,
        cfl: float = 0.4,
    ) -> MHDCTRateResult:
        cell = self.validate_reduced_state(reduced_state)
        magnetic = self.validate_magnetic_flux(magnetic_flux)
        full = self.full_state(cell, magnetic)
        normal_fields = self.bridge.unpack_normal_flux(magnetic)
        fluxes = []
        speeds = []
        fallbacks = []
        for axis in range(self.layout.dimension):
            left, right = self._face_states(full, normal_fields, axis, time, args)
            result = self.interface_solver.face_flux(
                self.dynamics.system, left, right, axis, args
            )
            fluxes.append(result.normal_flux)
            speeds.append(result.max_speed)
            fallbacks.append(result.fallback_activated)
        flux_tuple = tuple(fluxes)
        speed_tuple = tuple(speeds)
        residual = jnp.zeros_like(full)
        inverse_dt = jnp.zeros(self.cell_shape, dtype=full.dtype)
        volumes = self.dynamics.discretization.cell_volumes.astype(full.dtype)
        integrated_fluxes = []
        measures = self.dynamics.discretization.face_measures
        for axis, (flux, speed, measure) in enumerate(
            zip(flux_tuple, speed_tuple, measures, strict=True)
        ):
            integrated = flux * measure[..., None]
            integrated_fluxes.append(integrated)
            if self.bridge.grid.structured_axes[axis].periodic:
                residual = (
                    residual
                    - (integrated - jnp.roll(integrated, 1, axis=axis))
                    / volumes[..., None]
                )
                inverse_dt = inverse_dt + speed * measure / volumes
            else:
                lower_indices = jnp.arange(integrated.shape[axis] - 1)
                upper_indices = jnp.arange(1, integrated.shape[axis])
                lower_flux = jnp.take(integrated, lower_indices, axis=axis)
                upper_flux = jnp.take(integrated, upper_indices, axis=axis)
                residual = residual - (upper_flux - lower_flux) / volumes[..., None]
                face_rate = speed * measure
                lower_rate = jnp.take(face_rate, lower_indices, axis=axis)
                upper_rate = jnp.take(face_rate, upper_indices, axis=axis)
                inverse_dt = inverse_dt + jnp.maximum(lower_rate, upper_rate) / volumes
        residual = residual.at[
            ..., jnp.asarray(self.layout.cochain_magnetic_indices)
        ].set(0.0)
        bounded = any(not axis.periodic for axis in self.bridge.grid.structured_axes)
        if bounded:
            electromotive_components = self._bounded_electromotive_components(flux_tuple)
            uct_defect = jnp.asarray(0.0, dtype=full.dtype)
            uct_dissipation = jnp.asarray(0.0, dtype=full.dtype)
        else:
            electromotive = self.electromotive_plan.electromotive(
                full,
                flux_tuple,
                speed_tuple,
                self.layout.dimension,
            )
            electromotive_components = electromotive.components
            uct_defect = electromotive.one_dimensional_consistency_defect
            uct_dissipation = electromotive.maximum_dissipation
        if self.layout.dimension == 1:
            edge_circulation = jnp.zeros((0,), dtype=full.dtype)
            magnetic_rate = jnp.zeros_like(magnetic)
        else:
            edge_circulation = self.bridge.pack_electromotive(electromotive_components)
            magnetic_rate = -self.bridge.exterior_derivative(
                int(self.layout.electromotive_degree), edge_circulation
            )
        stable = jnp.asarray(float(cfl), dtype=full.dtype) / jnp.max(inverse_dt)
        fallback = jnp.any(jnp.stack(fallbacks, axis=0))
        return MHDCTRateResult(
            cell_rate=residual[..., jnp.asarray(self.layout.reduced_component_indices)],
            magnetic_rate=magnetic_rate,
            edge_electromotive_circulation=edge_circulation,
            normal_fluxes=flux_tuple,
            integrated_normal_fluxes=tuple(integrated_fluxes),
            signal_speeds=speed_tuple,
            stable_step=stable,
            fallback_activated=fallback,
            uct_consistency_defect=uct_defect,
            uct_maximum_dissipation=uct_dissipation,
        )


__all__ = [
    "ConstrainedMagneticStateLayout",
    "MHDCTRateResult",
    "UpwindConstrainedTransportPlan",
]
