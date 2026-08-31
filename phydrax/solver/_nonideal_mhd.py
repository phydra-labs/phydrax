#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._constrained_mhd import ConstrainedMHDState


class AnisotropicThermalTransportDiagnostics(StrictModule):
    heat_flux: Array
    energy_change: Array
    stable_step: Array
    successful: Array


class AnisotropicThermalTransportPlan(StrictModule, NonTrainableState):
    parallel_conductivity: float = eqx.field(static=True)
    perpendicular_conductivity: float = eqx.field(static=True)
    cfl: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        parallel_conductivity: float,
        /,
        *,
        perpendicular_conductivity: float = 0.0,
        cfl: float = 0.25,
    ):
        parallel = float(parallel_conductivity)
        perpendicular = float(perpendicular_conductivity)
        cfl_ = float(cfl)
        if (
            parallel < 0.0
            or perpendicular < 0.0
            or perpendicular > parallel
            or not 0.0 < cfl_ <= 1.0
        ):
            raise ValueError("Anisotropic thermal transport controls are invalid.")
        self.parallel_conductivity = parallel
        self.perpendicular_conductivity = perpendicular
        self.cfl = cfl_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "anisotropic-thermal-transport",
                "parallel_conductivity": parallel,
                "perpendicular_conductivity": perpendicular,
                "cfl": cfl_,
            }
        )

    def advance(
        self,
        temperature: Array,
        material_energy: Array,
        magnetic_field: Array,
        step_size: ArrayLike,
        spacings: tuple[float, ...],
        /,
    ) -> tuple[Array, AnisotropicThermalTransportDiagnostics]:
        dimension = temperature.ndim
        if (
            magnetic_field.shape != temperature.shape + (dimension,)
            or material_energy.shape != temperature.shape
            or len(spacings) != dimension
        ):
            raise ValueError("Anisotropic thermal transport arrays are inconsistent.")
        norm = jnp.sqrt(jnp.sum(magnetic_field**2, axis=-1))
        direction = magnetic_field / jnp.maximum(norm[..., None], 1e-12)
        gradient = jnp.stack(
            tuple(
                (
                    jnp.roll(temperature, -1, axis=axis)
                    - jnp.roll(temperature, 1, axis=axis)
                )
                / (2.0 * spacing)
                for axis, spacing in enumerate(spacings)
            ),
            axis=-1,
        )
        parallel_gradient = jnp.sum(direction * gradient, axis=-1)
        flux = (
            -self.perpendicular_conductivity * gradient
            - (self.parallel_conductivity - self.perpendicular_conductivity)
            * parallel_gradient[..., None]
            * direction
        )
        divergence = sum(
            (
                jnp.roll(flux[..., axis], -1, axis=axis)
                - jnp.roll(flux[..., axis], 1, axis=axis)
            )
            / (2.0 * spacing)
            for axis, spacing in enumerate(spacings)
        )
        step = jnp.asarray(step_size, dtype=temperature.dtype)
        maximum = max(self.parallel_conductivity, self.perpendicular_conductivity)
        stable = jnp.asarray(
            jnp.inf if maximum == 0.0 else self.cfl * min(spacings) ** 2 / maximum,
            dtype=temperature.dtype,
        )
        change = -step * divergence
        candidate = material_energy + change
        successful = (
            jnp.isfinite(step)
            & (step > 0.0)
            & (step <= stable)
            & jnp.all(jnp.isfinite(candidate))
            & jnp.all(candidate > 0.0)
        )
        accepted = jnp.where(successful, candidate, material_energy)
        return accepted, AnisotropicThermalTransportDiagnostics(
            heat_flux=flux,
            energy_change=accepted - material_energy,
            stable_step=stable,
            successful=successful,
        )


class NonIdealMHDDiagnostics(StrictModule):
    edge_electromotive: Array
    magnetic_energy_before: Array
    magnetic_energy_after: Array
    material_heating: Array
    magnetic_constraint_change: Array
    stable_step: Array
    successful: Array


class NonIdealMHDPlan(StrictModule, NonTrainableState):
    """Compatible explicit resistive, Hall, and ambipolar magnetic update."""

    spatial: object
    resistivity: float = eqx.field(static=True)
    hall_coefficient: float = eqx.field(static=True)
    ambipolar_coefficient: float = eqx.field(static=True)
    cfl: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial,
        /,
        *,
        resistivity: float = 0.0,
        hall_coefficient: float = 0.0,
        ambipolar_coefficient: float = 0.0,
        cfl: float = 0.25,
    ):
        from ..discretization.finite_volume import UpwindConstrainedTransportPlan

        if not isinstance(spatial, UpwindConstrainedTransportPlan):
            raise TypeError("spatial must be UpwindConstrainedTransportPlan.")
        coefficients = tuple(
            float(value)
            for value in (resistivity, hall_coefficient, ambipolar_coefficient)
        )
        cfl_ = float(cfl)
        if (
            spatial.layout.dimension != 3
            or any(not np.isfinite(value) or value < 0.0 for value in coefficients)
            or not np.isfinite(cfl_)
            or not 0.0 < cfl_ <= 1.0
        ):
            raise ValueError(
                "Non-ideal MHD currently requires valid three-dimensional coefficients."
            )
        self.spatial = spatial
        self.resistivity, self.hall_coefficient, self.ambipolar_coefficient = coefficients
        self.cfl = cfl_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compatible-nonideal-mhd",
                "spatial": spatial.plan_id,
                "resistivity": coefficients[0],
                "hall_coefficient": coefficients[1],
                "ambipolar_coefficient": coefficients[2],
                "cfl": cfl_,
            }
        )

    def _cell_current(self, magnetic_flux: Array, /) -> tuple[Array, Array]:
        bridge = self.spatial.bridge
        degree = self.spatial.layout.magnetic_degree
        current_cochain = bridge.codifferential(degree, magnetic_flux)
        components = bridge.unpack_edge_circulation(current_cochain)
        centered = tuple(
            0.25
            * (
                component
                + jnp.roll(component, 1, axis=(axis + 1) % 3)
                + jnp.roll(component, 1, axis=(axis + 2) % 3)
                + jnp.roll(
                    jnp.roll(component, 1, axis=(axis + 1) % 3),
                    1,
                    axis=(axis + 2) % 3,
                )
            )
            for axis, component in enumerate(components)
        )
        return current_cochain, jnp.stack(centered, axis=-1)

    def advance(
        self,
        state: ConstrainedMHDState,
        end_time: ArrayLike,
        /,
    ) -> tuple[ConstrainedMHDState, NonIdealMHDDiagnostics]:
        magnetic = self.spatial.validate_magnetic_flux(state.magnetic_flux)
        end = jnp.asarray(end_time, dtype=state.time.dtype).reshape(())
        step = end - state.time
        current_cochain, current = self._cell_current(magnetic)
        magnetic_cell = self.spatial.cochain_cell_magnetic_field(magnetic)
        resistive = self.resistivity * current
        hall = self.hall_coefficient * jnp.cross(current, magnetic_cell)
        ambipolar = self.ambipolar_coefficient * jnp.cross(
            jnp.cross(current, magnetic_cell), magnetic_cell
        )
        electric_cell = resistive + hall - ambipolar
        edge_components = tuple(
            0.25
            * (
                electric_cell[..., axis]
                + jnp.roll(electric_cell[..., axis], -1, axis=(axis + 1) % 3)
                + jnp.roll(electric_cell[..., axis], -1, axis=(axis + 2) % 3)
                + jnp.roll(
                    jnp.roll(electric_cell[..., axis], -1, axis=(axis + 1) % 3),
                    -1,
                    axis=(axis + 2) % 3,
                )
            )
            for axis in range(3)
        )
        edge_electromotive = self.spatial.bridge.pack_edge_circulation(edge_components)
        magnetic_rate = -self.spatial.bridge.exterior_derivative(1, edge_electromotive)
        maximum = max(
            self.resistivity,
            self.hall_coefficient,
            self.ambipolar_coefficient,
        )
        spacing = min(
            float(np.min(axis.interval_widths))
            for axis in self.spatial.bridge.grid.structured_axes
        )
        stable = jnp.asarray(
            jnp.inf if maximum == 0.0 else self.cfl * spacing**2 / maximum,
            dtype=step.dtype,
        )
        candidate_magnetic = magnetic + step * magnetic_rate
        hodge_before = self.spatial.bridge.cochain.apply_hodge(2, magnetic)
        hodge_after = self.spatial.bridge.cochain.apply_hodge(2, candidate_magnetic)
        magnetic_energy_before = 0.5 * jnp.vdot(magnetic, hodge_before).real
        magnetic_energy_after = 0.5 * jnp.vdot(candidate_magnetic, hodge_after).real
        heating_total = jnp.maximum(
            magnetic_energy_before - magnetic_energy_after,
            0.0,
        )
        volumes = self.spatial.dynamics.discretization.cell_volumes
        heating_density = heating_total / jnp.sum(volumes)
        cell_state = state.cell_state.at[..., 4].add(heating_density)
        constraint_before = self.spatial.magnetic_constraint(magnetic)
        constraint_after = self.spatial.magnetic_constraint(candidate_magnetic)
        constraint_change = jnp.max(
            jnp.abs(constraint_after - constraint_before), initial=0.0
        )
        successful = (
            jnp.isfinite(step)
            & (step > 0.0)
            & (step <= stable)
            & jnp.all(jnp.isfinite(candidate_magnetic))
            & jnp.all(
                self.spatial.dynamics.system.admissible(
                    self.spatial.full_state(cell_state, candidate_magnetic)
                )
            )
            & (constraint_change <= 1e-10)
        )
        accepted = ConstrainedMHDState(
            jnp.where(successful, cell_state, state.cell_state),
            jnp.where(successful, candidate_magnetic, state.magnetic_flux),
            jnp.where(successful, end, state.time),
            jnp.where(successful, step, state.step_size),
            state.accepted_step + successful.astype(jnp.int32),
            jnp.where(successful, 0, 3).astype(jnp.int32),
        )
        diagnostics = NonIdealMHDDiagnostics(
            edge_electromotive=edge_electromotive,
            magnetic_energy_before=magnetic_energy_before,
            magnetic_energy_after=magnetic_energy_after,
            material_heating=heating_total,
            magnetic_constraint_change=constraint_change,
            stable_step=stable,
            successful=successful,
        )
        del current_cochain
        return accepted, diagnostics


__all__ = [
    "AnisotropicThermalTransportDiagnostics",
    "AnisotropicThermalTransportPlan",
    "NonIdealMHDDiagnostics",
    "NonIdealMHDPlan",
]
