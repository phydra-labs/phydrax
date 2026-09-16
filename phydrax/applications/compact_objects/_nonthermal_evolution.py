#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class NonthermalLorentzGrid(StrictModule, NonTrainableState):
    edges: Array
    centers: Array
    widths: Array
    grid_id: str = eqx.field(static=True)

    def __init__(self, edges: ArrayLike, /) -> None:
        values = np.asarray(edges, dtype=float)
        if (
            values.ndim != 1
            or values.size < 3
            or np.any(~np.isfinite(values))
            or values[0] < 1.0
            or np.any(np.diff(values) <= 0.0)
        ):
            raise ValueError(
                "Lorentz-factor edges must be finite and increasing from one."
            )
        centers = np.sqrt(values[:-1] * values[1:])
        widths = np.diff(values)
        self.edges = jnp.asarray(values)
        self.centers = jnp.asarray(centers)
        self.widths = jnp.asarray(widths)
        self.grid_id = canonical_fingerprint(
            {
                "kind": "nonthermal-lorentz-grid",
                "edges": array_tree_fingerprint(values),
            }
        )

    @property
    def bin_count(self) -> int:
        return int(self.centers.size)


class NonthermalElectronState(StrictModule):
    bin_number_density: Array
    time: Array
    accepted_steps: Array


class NonthermalElectronLedger(StrictModule):
    number_before: Array
    number_after: Array
    number_injected: Array
    number_thermalized: Array
    number_escaped: Array
    number_defect: Array
    energy_before: Array
    energy_after: Array
    injected_energy: Array
    thermal_energy_exchange: Array
    radiation_energy_exchange: Array
    adiabatic_energy_exchange: Array
    energy_defect: Array
    minimum_population: Array
    finite: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class NonthermalElectronResult(StrictModule):
    candidate: NonthermalElectronState
    state: NonthermalElectronState
    ledger: NonthermalElectronLedger
    accepted: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class NonthermalElectronEvolutionPlan(StrictModule, NonTrainableState):
    """Positive finite-volume evolution on a bounded Lorentz-factor grid."""

    scale: RelativityScaleContract
    grid: NonthermalLorentzGrid
    particle_rest_energy: float = eqx.field(static=True)
    injection_slope: float = eqx.field(static=True)
    injection_minimum: float = eqx.field(static=True)
    injection_maximum: float = eqx.field(static=True)
    synchrotron_coefficient: float = eqx.field(static=True)
    inverse_compton_coefficient: float = eqx.field(static=True)
    bremsstrahlung_coefficient: float = eqx.field(static=True)
    coulomb_coefficient: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        grid: NonthermalLorentzGrid,
        /,
        *,
        particle_rest_energy: float,
        injection_slope: float,
        injection_minimum: float,
        injection_maximum: float,
        synchrotron_coefficient: float = 0.0,
        inverse_compton_coefficient: float = 0.0,
        bremsstrahlung_coefficient: float = 0.0,
        coulomb_coefficient: float = 0.0,
        energy_tolerance: float = 1.0e-8,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        if not isinstance(grid, NonthermalLorentzGrid):
            raise TypeError("grid must be NonthermalLorentzGrid.")
        values = tuple(
            float(value)
            for value in (
                particle_rest_energy,
                injection_slope,
                injection_minimum,
                injection_maximum,
                synchrotron_coefficient,
                inverse_compton_coefficient,
                bremsstrahlung_coefficient,
                coulomb_coefficient,
                energy_tolerance,
            )
        )
        if (
            not np.isfinite(values[0])
            or values[0] <= 0.0
            or not np.isfinite(values[1])
            or values[1] <= 1.0
            or not grid.edges[0] <= values[2] < values[3] <= grid.edges[-1]
            or any(not np.isfinite(value) or value < 0.0 for value in values[4:8])
            or not np.isfinite(values[8])
            or values[8] <= 0.0
        ):
            raise ValueError("Nonthermal electron evolution controls are invalid.")
        self.scale = scale
        self.grid = grid
        (
            self.particle_rest_energy,
            self.injection_slope,
            self.injection_minimum,
            self.injection_maximum,
            self.synchrotron_coefficient,
            self.inverse_compton_coefficient,
            self.bremsstrahlung_coefficient,
            self.coulomb_coefficient,
            self.energy_tolerance,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nonthermal-electron-evolution",
                "scale": scale.scale_id,
                "grid": grid.grid_id,
                "particle_rest_energy": values[0],
                "injection": values[1:4],
                "cooling_coefficients": values[4:8],
                "energy_tolerance": values[8],
            }
        )

    def initialize(
        self,
        bin_number_density: ArrayLike,
        /,
        *,
        time: ArrayLike = 0.0,
    ) -> NonthermalElectronState:
        density = jnp.asarray(bin_number_density)
        if density.shape[-1:] != (self.grid.bin_count,):
            raise ValueError("Nonthermal population must match the Lorentz grid.")
        density = eqx.error_if(
            density,
            jnp.any(~jnp.isfinite(density) | (density < 0.0)),
            "Initial nonthermal electron population is invalid.",
        )
        return NonthermalElectronState(
            density,
            jnp.asarray(time, dtype=density.dtype).reshape(()),
            jnp.zeros((), dtype=jnp.int32),
        )

    def _moments(self, density: Array, /) -> tuple[Array, Array]:
        gamma = self.grid.centers.astype(density.dtype)
        number = jnp.sum(density, axis=-1)
        energy = self.particle_rest_energy * jnp.sum(density * (gamma - 1.0), axis=-1)
        return number, energy

    def _injection(self, energy: Array, dtype, /) -> tuple[Array, Array]:
        gamma = self.grid.centers.astype(dtype)
        mask = (gamma >= self.injection_minimum) & (gamma <= self.injection_maximum)
        shape = gamma ** (-self.injection_slope) * mask
        energy_weight = self.particle_rest_energy * (gamma - 1.0)
        normalization = jnp.sum(shape * energy_weight)
        distribution = (
            energy[..., None] * shape / jnp.maximum(normalization, jnp.finfo(dtype).tiny)
        )
        return distribution, jnp.sum(distribution, axis=-1)

    def advance(
        self,
        state: NonthermalElectronState,
        step_size: ArrayLike,
        /,
        *,
        expansion_rate: ArrayLike,
        magnetic_squared: ArrayLike,
        radiation_energy_density: ArrayLike,
        radiation_temperature: ArrayLike,
        thermal_electron_density: ArrayLike,
        dissipative_heating: ArrayLike = 0.0,
        injection_fraction: ArrayLike = 0.0,
    ) -> NonthermalElectronResult:
        if not isinstance(state, NonthermalElectronState):
            raise TypeError("state must be NonthermalElectronState.")
        density = state.bin_number_density
        leading = density.shape[:-1]
        (
            step,
            expansion,
            magnetic,
            radiation,
            radiation_temperature_,
            thermal_density,
            heating,
            fraction,
        ) = jnp.broadcast_arrays(
            jnp.asarray(step_size),
            jnp.asarray(expansion_rate),
            jnp.asarray(magnetic_squared),
            jnp.asarray(radiation_energy_density),
            jnp.asarray(radiation_temperature),
            jnp.asarray(thermal_electron_density),
            jnp.asarray(dissipative_heating),
            jnp.asarray(injection_fraction),
        )
        for value in (
            step,
            expansion,
            magnetic,
            radiation,
            radiation_temperature_,
            thermal_density,
            heating,
            fraction,
        ):
            if value.shape != leading:
                raise ValueError(
                    "Nonthermal process fields must match the population batch."
                )
        dtype = density.dtype
        gamma = self.grid.centers.astype(dtype)
        theta_r = (
            jnp.asarray(float(self.scale.boltzmann_constant), dtype)
            * radiation_temperature_
            / jnp.asarray(self.particle_rest_energy, dtype)
        )
        kinetic_factor = gamma**2 - 1.0
        adiabatic_rate = -(expansion[..., None] / 3.0) * (gamma - 1.0 / gamma)
        synchrotron_rate = (
            -self.synchrotron_coefficient * magnetic[..., None] * kinetic_factor
        )
        klein_nishina = (1.0 + 4.0 * theta_r[..., None] * gamma) ** (-1.5)
        inverse_compton_rate = (
            -self.inverse_compton_coefficient
            * radiation[..., None]
            * kinetic_factor
            * klein_nishina
        )
        bremsstrahlung_rate = (
            -self.bremsstrahlung_coefficient
            * thermal_density[..., None]
            * gamma
            * (jnp.log(gamma) + 0.36)
        )
        coulomb_rate = (
            -self.coulomb_coefficient * thermal_density[..., None] * jnp.ones_like(gamma)
        )
        total_rate = (
            adiabatic_rate
            + synchrotron_rate
            + inverse_compton_rate
            + bremsstrahlung_rate
            + coulomb_rate
        )
        edge_rate = jnp.concatenate(
            (
                total_rate[..., :1],
                0.5 * (total_rate[..., :-1] + total_rate[..., 1:]),
                total_rate[..., -1:],
            ),
            axis=-1,
        )
        left = jnp.concatenate((jnp.zeros_like(density[..., :1]), density), axis=-1)
        right = jnp.concatenate((density, jnp.zeros_like(density[..., -1:])), axis=-1)
        upwind = jnp.where(edge_rate >= 0.0, left, right)
        gamma_flux = edge_rate * upwind
        transport_change = -step[..., None] * (gamma_flux[..., 1:] - gamma_flux[..., :-1])
        injected_energy = jnp.maximum(heating * fraction, 0.0)
        injection, number_injected = self._injection(injected_energy, dtype)
        candidate_density = density + transport_change + injection
        number_before, energy_before = self._moments(density)
        number_after, energy_after = self._moments(candidate_density)
        lower_number = step * jnp.maximum(-gamma_flux[..., 0], 0.0)
        upper_number = step * jnp.maximum(gamma_flux[..., -1], 0.0)
        lower_energy = (
            lower_number * self.particle_rest_energy * (self.grid.edges[0] - 1.0)
        )
        upper_energy = (
            upper_number * self.particle_rest_energy * (self.grid.edges[-1] - 1.0)
        )
        predicted_total = self.particle_rest_energy * jnp.sum(
            step[..., None] * density * total_rate, axis=-1
        )
        predicted_adiabatic = self.particle_rest_energy * jnp.sum(
            step[..., None] * density * adiabatic_rate, axis=-1
        )
        predicted_coulomb = self.particle_rest_energy * jnp.sum(
            step[..., None] * density * coulomb_rate, axis=-1
        )
        predicted_radiative = predicted_total - predicted_adiabatic - predicted_coulomb
        energy_change = energy_after - energy_before
        actual_process_energy = (
            energy_change - injected_energy + lower_energy + upper_energy
        )
        correction = jnp.where(
            jnp.abs(predicted_total) > jnp.finfo(dtype).tiny,
            actual_process_energy / predicted_total,
            0.0,
        )
        adiabatic_energy = correction * predicted_adiabatic
        coulomb_energy = correction * predicted_coulomb
        radiative_energy = correction * predicted_radiative
        expected_energy = (
            injected_energy
            + adiabatic_energy
            + coulomb_energy
            + radiative_energy
            - lower_energy
            - upper_energy
        )
        energy_defect = energy_change - expected_energy
        number_defect = (
            number_after - number_before - number_injected + lower_number + upper_number
        )
        finite = (
            jnp.all(jnp.isfinite(candidate_density), axis=-1)
            & jnp.isfinite(energy_defect)
            & jnp.isfinite(number_defect)
        )
        physical = (
            finite
            & (step >= 0.0)
            & (magnetic >= 0.0)
            & (radiation >= 0.0)
            & (thermal_density >= 0.0)
            & (fraction >= 0.0)
            & (fraction <= 1.0)
            & jnp.all(candidate_density >= 0.0, axis=-1)
        )
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(energy_change), jnp.abs(expected_energy)), 1.0
        )
        number_scale = jnp.maximum(number_after + number_before + number_injected, 1.0)
        balanced = (jnp.abs(energy_defect) <= self.energy_tolerance * scale) & (
            jnp.abs(number_defect) <= self.energy_tolerance * number_scale
        )
        qualified_local = physical & balanced
        accepted = jnp.all(qualified_local)
        candidate = NonthermalElectronState(
            candidate_density,
            state.time + jnp.max(step),
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
        )
        accepted_state = NonthermalElectronState(
            jnp.where(accepted, candidate_density, density),
            jnp.where(accepted, candidate.time, state.time),
            state.accepted_steps + accepted.astype(jnp.int32),
        )
        ledger = NonthermalElectronLedger(
            number_before,
            number_after,
            number_injected,
            lower_number,
            upper_number,
            number_defect,
            energy_before,
            energy_after,
            injected_energy,
            coulomb_energy + lower_energy + upper_energy,
            radiative_energy,
            adiabatic_energy,
            energy_defect,
            jnp.min(candidate_density, axis=-1),
            jnp.all(finite),
            accepted,
            self.plan_id,
        )
        return NonthermalElectronResult(
            candidate,
            accepted_state,
            ledger,
            accepted,
            jnp.all(finite),
            jnp.all(physical),
            accepted,
            accepted & jnp.all(candidate_density > 0.0),
        )


__all__ = [
    "NonthermalElectronEvolutionPlan",
    "NonthermalElectronLedger",
    "NonthermalElectronResult",
    "NonthermalElectronState",
    "NonthermalLorentzGrid",
]
