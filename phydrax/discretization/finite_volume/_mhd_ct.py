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
from ._riemann import AbstractNumericalFluxPlan


class MHDCTRateResult(StrictModule):
    cell_rate: Array
    magnetic_rate: Array
    edge_electromotive_circulation: Array
    normal_fluxes: tuple[Array, Array, Array]
    signal_speeds: tuple[Array, Array, Array]
    stable_step: Array
    fallback_activated: Array


class UpwindConstrainedTransportPlan(StrictModule, NonTrainableState):
    """Periodic Cartesian flux-CT from upwind Riemann face fluxes."""

    dynamics: PreparedFiniteVolumeDynamics
    bridge: StructuredCochainBridge
    interface_solver: AbstractNumericalFluxPlan
    cell_shape: tuple[int, int, int] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedFiniteVolumeDynamics,
        bridge: StructuredCochainBridge,
        /,
    ):
        if not isinstance(dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError("dynamics must be PreparedFiniteVolumeDynamics.")
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be StructuredCochainBridge.")
        if bridge.dimension != 3 or tuple(dynamics.discretization.cell_shape) != tuple(
            bridge.grid.shape
        ):
            raise ValueError("Constrained transport requires one shared 3D tensor grid.")
        if bridge.grid.prepared_id != dynamics.discretization.grid.prepared_id:
            raise ValueError("Finite volume and cochain bridge must share grid identity.")
        if any(not axis.periodic for axis in bridge.grid.structured_axes):
            raise ValueError("Initial constrained-transport support is fully periodic.")
        expected = (
            "density",
            "momentum_x",
            "momentum_y",
            "momentum_z",
            "total_energy",
            "magnetic_x",
            "magnetic_y",
            "magnetic_z",
        )
        if tuple(dynamics.system.component_names) != expected:
            raise TypeError("Constrained transport requires canonical ideal-MHD layout.")
        solver = dynamics.method.interface_solver
        if not isinstance(solver, AbstractNumericalFluxPlan):
            raise TypeError("Constrained transport requires a numerical-flux solver.")
        self.dynamics = dynamics
        self.bridge = bridge
        self.interface_solver = solver
        self.cell_shape = tuple(int(value) for value in bridge.grid.shape)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "upwind-flux-constrained-transport",
                "dynamics": dynamics.dynamics_id,
                "bridge": bridge.bridge_id,
                "interface": solver.flux_id,
            }
        )

    def validate_reduced_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        expected = self.cell_shape + (5,)
        if value.shape != expected:
            raise ValueError(f"Reduced MHD cell state must have shape {expected}.")
        return value

    def validate_magnetic_flux(self, magnetic_flux: ArrayLike, /) -> Array:
        value = jnp.asarray(magnetic_flux)
        expected = self.bridge.cochain.cell_counts[2]
        if value.shape != (expected,):
            raise ValueError(f"Magnetic flux cochain must have shape ({expected},).")
        return value

    def cell_magnetic_field(self, magnetic_flux: ArrayLike, /) -> Array:
        bx_face, by_face, bz_face = self.bridge.unpack_face_flux(
            self.validate_magnetic_flux(magnetic_flux)
        )
        cell = tuple(
            0.5 * (component + jnp.roll(component, -1, axis=axis))
            for axis, component in enumerate((bx_face, by_face, bz_face))
        )
        return jnp.stack(cell, axis=-1)

    def full_state(self, reduced: ArrayLike, magnetic_flux: ArrayLike, /) -> Array:
        cell = self.validate_reduced_state(reduced)
        magnetic = self.cell_magnetic_field(magnetic_flux)
        return jnp.concatenate((cell, magnetic), axis=-1)

    def magnetic_constraint(self, magnetic_flux: ArrayLike, /) -> Array:
        return self.bridge.exterior_derivative(
            2, self.validate_magnetic_flux(magnetic_flux)
        )

    def _face_states(
        self, full_state: Array, magnetic_flux: Array, axis: int, /
    ) -> tuple[Array, Array]:
        left = full_state
        right = jnp.roll(full_state, -1, axis=axis)
        primitive_left = self.dynamics.system.conserved_to_primitive(left)
        primitive_right = self.dynamics.system.conserved_to_primitive(right)
        normal_face = self.bridge.unpack_face_flux(magnetic_flux)[axis]
        primitive_left = primitive_left.at[..., 5 + axis].set(normal_face)
        primitive_right = primitive_right.at[..., 5 + axis].set(normal_face)
        return (
            self.dynamics.system.primitive_to_conserved(primitive_left),
            self.dynamics.system.primitive_to_conserved(primitive_right),
        )

    def _edge_electromotive(
        self, fluxes: tuple[Array, Array, Array], /
    ) -> tuple[Array, Array, Array]:
        flux_x, flux_y, flux_z = fluxes
        ex = 0.25 * (
            -flux_y[..., 7]
            - jnp.roll(flux_y[..., 7], -1, axis=2)
            + flux_z[..., 6]
            + jnp.roll(flux_z[..., 6], -1, axis=1)
        )
        ey = 0.25 * (
            flux_x[..., 7]
            + jnp.roll(flux_x[..., 7], -1, axis=2)
            - flux_z[..., 5]
            - jnp.roll(flux_z[..., 5], -1, axis=0)
        )
        ez = 0.25 * (
            -flux_x[..., 6]
            - jnp.roll(flux_x[..., 6], -1, axis=1)
            + flux_y[..., 5]
            + jnp.roll(flux_y[..., 5], -1, axis=0)
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
        del time
        reduced = self.validate_reduced_state(reduced_state)
        magnetic = self.validate_magnetic_flux(magnetic_flux)
        full = self.full_state(reduced, magnetic)
        fluxes = []
        speeds = []
        fallbacks = []
        for axis in range(3):
            left, right = self._face_states(full, magnetic, axis)
            solved = self.interface_solver.face_flux(
                self.dynamics.system, left, right, axis, args
            )
            fluxes.append(solved.normal_flux)
            speeds.append(solved.max_speed)
            fallbacks.append(solved.fallback_activated)
        flux_tuple = (fluxes[0], fluxes[1], fluxes[2])
        speed_tuple = (speeds[0], speeds[1], speeds[2])
        residual = jnp.zeros_like(full)
        inverse_dt = jnp.zeros(self.cell_shape, dtype=full.dtype)
        volumes = self.dynamics.discretization.cell_volumes.astype(full.dtype)
        for axis, (flux, speed, measure) in enumerate(
            zip(
                flux_tuple,
                speed_tuple,
                self.dynamics.discretization.face_measures,
                strict=True,
            )
        ):
            integrated = flux * measure[..., None]
            residual = (
                residual
                - (integrated - jnp.roll(integrated, 1, axis=axis)) / volumes[..., None]
            )
            inverse_dt = inverse_dt + speed * measure / volumes
        residual = residual.at[..., 5:8].set(0.0)
        edge_components = self._edge_electromotive(flux_tuple)
        edge_circulation = self.bridge.pack_edge_circulation(edge_components)
        magnetic_rate = -self.bridge.exterior_derivative(1, edge_circulation)
        stable = jnp.asarray(float(cfl), dtype=full.dtype) / jnp.max(inverse_dt)
        fallback = jnp.any(jnp.stack(fallbacks, axis=0))
        return MHDCTRateResult(
            cell_rate=residual[..., :5],
            magnetic_rate=magnetic_rate,
            edge_electromotive_circulation=edge_circulation,
            normal_fluxes=flux_tuple,
            signal_speeds=speed_tuple,
            stable_step=stable,
            fallback_activated=fallback,
        )


__all__ = ["MHDCTRateResult", "UpwindConstrainedTransportPlan"]
