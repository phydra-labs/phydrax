#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._closure import ScientificArtifactEnvelope
from ._coupled import ComovingEulerState


PRIMORDIAL_SPECIES = ("HI", "HII", "HeI", "HeII", "HeIII", "electron")
PRIMORDIAL_PROCESSES = (
    "H_collisional_ionization",
    "H_recombination",
    "HeI_collisional_ionization",
    "HeII_recombination",
    "HeII_collisional_ionization",
    "HeIII_recombination",
    "HI_photoionization",
    "HeI_photoionization",
    "HeII_photoionization",
    "cooling",
    "photoheating",
)


def _solve_small_dense(matrix: Array, right_hand_side: Array, /) -> Array:
    """Solve one fixed 4x4 system by traced Gaussian elimination."""
    state = jnp.concatenate((matrix, right_hand_side[:, None]), axis=1)
    for pivot in range(4):
        row = pivot + jnp.argmax(jnp.abs(state[pivot:, pivot]))
        pivot_row = state[row]
        selected_row = state[pivot]
        state = state.at[pivot].set(pivot_row).at[row].set(selected_row)
        denominator = state[pivot, pivot]
        denominator = jnp.where(
            jnp.abs(denominator) > jnp.finfo(state.dtype).tiny,
            denominator,
            jnp.asarray(jnp.nan, dtype=state.dtype),
        )
        state = state.at[pivot].set(state[pivot] / denominator)
        factors = state[:, pivot].at[pivot].set(0.0)
        state = state - factors[:, None] * state[pivot][None, :]
    return state[:, -1]


class PrimordialSpeciesState(StrictModule):
    number_densities: Array
    internal_energy: Array
    scale_factor: Array

    def __init__(
        self,
        number_densities: ArrayLike,
        internal_energy: ArrayLike,
        scale_factor: ArrayLike,
        /,
    ):
        densities = jnp.asarray(number_densities)
        energy = jnp.asarray(internal_energy, dtype=densities.dtype)
        scale = jnp.asarray(scale_factor, dtype=densities.dtype)
        if densities.ndim < 1 or densities.shape[-1] != len(PRIMORDIAL_SPECIES):
            raise ValueError("Primordial species state must end in six named species.")
        if energy.shape != densities.shape[:-1] or scale.shape != ():
            raise ValueError("Primordial energy/scale shapes do not match species state.")
        electron = densities[..., 1] + densities[..., 3] + 2.0 * densities[..., 4]
        densities = densities.at[..., 5].set(electron)
        densities = eqx.error_if(
            densities,
            jnp.any(~jnp.isfinite(densities))
            | jnp.any(densities < 0.0)
            | jnp.any(~jnp.isfinite(energy))
            | jnp.any(energy <= 0.0)
            | ~jnp.isfinite(scale)
            | (scale <= 0.0),
            "Primordial species state must be finite, non-negative, and energetic.",
        )
        self.number_densities = densities
        self.internal_energy = energy
        self.scale_factor = scale


class PrimordialRateTable(StrictModule, NonTrainableState):
    temperatures: Array
    scale_factors: Array
    rates: Array
    artifact: ScientificArtifactEnvelope
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperatures: ArrayLike,
        scale_factors: ArrayLike,
        rates: ArrayLike,
        artifact: ScientificArtifactEnvelope,
        /,
    ):
        temperature = jax.lax.stop_gradient(jnp.asarray(temperatures))
        scale = jax.lax.stop_gradient(jnp.asarray(scale_factors, dtype=temperature.dtype))
        values = jax.lax.stop_gradient(jnp.asarray(rates, dtype=temperature.dtype))
        expected = (len(PRIMORDIAL_PROCESSES), scale.size, temperature.size)
        if (
            temperature.ndim != 1
            or scale.ndim != 1
            or temperature.size < 2
            or scale.size < 2
            or values.shape != expected
        ):
            raise ValueError(f"Primordial rate table must have shape {expected}.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(temperature))
            | jnp.any(temperature <= 0.0)
            | jnp.any(jnp.diff(temperature) <= 0.0)
            | jnp.any(~jnp.isfinite(scale))
            | jnp.any(scale <= 0.0)
            | jnp.any(jnp.diff(scale) <= 0.0)
            | jnp.any(~jnp.isfinite(values))
            | jnp.any(values < 0.0),
            "Primordial rate table must be finite, non-negative, and increasing in coordinates.",
        )
        self.temperatures = temperature
        self.scale_factors = scale
        self.rates = values
        self.artifact = artifact
        self.table_id = canonical_fingerprint(
            {
                "kind": "primordial-rate-table",
                "artifact": artifact.artifact_id,
                "arrays": array_tree_fingerprint((temperature, scale, values)),
                "processes": list(PRIMORDIAL_PROCESSES),
            }
        )

    def evaluate(self, temperature: Array, scale_factor: Array, /) -> Array:
        query_temperature = jnp.asarray(temperature, dtype=self.temperatures.dtype)
        query_scale = jnp.asarray(scale_factor, dtype=self.scale_factors.dtype)
        query_temperature = eqx.error_if(
            query_temperature,
            jnp.any(query_temperature < self.temperatures[0])
            | jnp.any(query_temperature > self.temperatures[-1]),
            "Microphysics temperature is outside the rate table.",
        )
        query_scale = eqx.error_if(
            query_scale,
            (query_scale < self.scale_factors[0])
            | (query_scale > self.scale_factors[-1]),
            "Microphysics scale factor is outside the rate table.",
        )
        flat_temperature = query_temperature.reshape((-1,))
        at_each_scale = jax.vmap(
            lambda process: jax.vmap(
                lambda row: jnp.interp(flat_temperature, self.temperatures, row)
            )(process)
        )(self.rates)
        values = jax.vmap(
            lambda process: jax.vmap(
                lambda column: jnp.interp(query_scale, self.scale_factors, column),
                in_axes=1,
                out_axes=0,
            )(process)
        )(at_each_scale)
        return values.reshape((len(PRIMORDIAL_PROCESSES),) + query_temperature.shape)


class PrimordialMicrophysicsLedger(StrictModule):
    hydrogen_nuclei_defect: Array
    helium_nuclei_defect: Array
    charge_defect: Array
    energy_change: Array
    maximum_residual: Array
    iterations: Array
    positive: Array
    converged: Array
    successful: Array


class PrimordialMicrophysicsResult(StrictModule):
    state: PrimordialSpeciesState
    ledger: PrimordialMicrophysicsLedger
    successful: Array


class PrimordialMicrophysicsPlan(StrictModule, NonTrainableState):
    rate_table: PrimordialRateTable
    adiabatic_index: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rate_table: PrimordialRateTable,
        /,
        *,
        adiabatic_index: float = 5.0 / 3.0,
        boltzmann_constant: float = 1.0,
        maximum_iterations: int = 16,
        tolerance: float = 1.0e-8,
    ):
        gamma = float(adiabatic_index)
        boltzmann = float(boltzmann_constant)
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        if (
            not isinstance(rate_table, PrimordialRateTable)
            or not np.isfinite(gamma)
            or gamma <= 1.0
            or not np.isfinite(boltzmann)
            or boltzmann <= 0.0
            or iterations <= 0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError("Primordial microphysics policy is invalid.")
        self.rate_table = rate_table
        self.adiabatic_index = gamma
        self.boltzmann_constant = boltzmann
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "primordial-h-he-microphysics",
                "rate_table": rate_table.table_id,
                "adiabatic_index": gamma,
                "boltzmann_constant": boltzmann,
                "maximum_iterations": iterations,
                "tolerance": tolerance_,
            }
        )

    def _reduced(self, state: PrimordialSpeciesState, /) -> Array:
        densities = state.number_densities
        return jnp.concatenate(
            (
                densities[..., 1:2],
                densities[..., 3:5],
                state.internal_energy[..., None],
            ),
            axis=-1,
        )

    def _full(
        self,
        reduced: Array,
        hydrogen_total: Array,
        helium_total: Array,
        scale_factor: Array,
    ) -> PrimordialSpeciesState:
        hii, heii, heiii, energy = tuple(reduced[..., index] for index in range(4))
        hi = hydrogen_total - hii
        hei = helium_total - heii - heiii
        electron = hii + heii + 2.0 * heiii
        densities = jnp.stack((hi, hii, hei, heii, heiii, electron), axis=-1)
        return PrimordialSpeciesState(densities, energy, scale_factor)

    def _rates(
        self,
        reduced: Array,
        hydrogen_total: Array,
        helium_total: Array,
        scale_factor: Array,
    ) -> Array:
        hii, heii, heiii, energy = tuple(reduced[..., index] for index in range(4))
        hi = hydrogen_total - hii
        hei = helium_total - heii - heiii
        electron = hii + heii + 2.0 * heiii
        particles = hydrogen_total + helium_total + electron
        temperature = (
            (self.adiabatic_index - 1.0)
            * energy
            / jnp.maximum(
                particles * self.boltzmann_constant, jnp.finfo(energy.dtype).tiny
            )
        )
        coefficients = self.rate_table.evaluate(temperature, scale_factor)
        h_ion = coefficients[0] * hi * electron + coefficients[6] * hi
        h_rec = coefficients[1] * hii * electron
        he1_ion = coefficients[2] * hei * electron + coefficients[7] * hei
        he2_rec = coefficients[3] * heii * electron
        he2_ion = coefficients[4] * heii * electron + coefficients[8] * heii
        he3_rec = coefficients[5] * heiii * electron
        dhii = h_ion - h_rec
        dheii = he1_ion - he2_rec - he2_ion + he3_rec
        dheiii = he2_ion - he3_rec
        cooling = coefficients[9] * electron * (hydrogen_total + helium_total)
        heating = coefficients[10] * (hi + hei + heii)
        denergy = heating - cooling
        return jnp.stack((dhii, dheii, dheiii, denergy), axis=-1)

    def advance(
        self,
        state: PrimordialSpeciesState,
        delta_time: ArrayLike,
        /,
    ) -> PrimordialMicrophysicsResult:
        step = jnp.asarray(delta_time, dtype=state.internal_energy.dtype)
        if step.shape != ():
            raise ValueError("Microphysics delta_time must be scalar.")
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Microphysics delta_time must be finite and positive.",
        )
        initial = self._reduced(state)
        hydrogen_total = jnp.sum(state.number_densities[..., :2], axis=-1)
        helium_total = jnp.sum(state.number_densities[..., 2:5], axis=-1)

        def residual(value):
            return (
                value
                - initial
                - step
                * self._rates(value, hydrogen_total, helium_total, state.scale_factor)
            )

        def iteration(_, carry):
            value, converged = carry
            residual_value = residual(value)
            flat_shape = value.shape[:-1]
            flat_value = value.reshape((-1, 4))
            flat_residual = residual_value.reshape((-1, 4))

            def solve_cell(cell_value, cell_residual, h_total, he_total):
                def cell_function(candidate):
                    return (
                        candidate
                        - cell_value
                        - step
                        * self._rates(
                            candidate,
                            h_total,
                            he_total,
                            state.scale_factor,
                        )
                    )

                jacobian = jax.jacfwd(cell_function)(cell_value)
                update = _solve_small_dense(jacobian, cell_residual)
                return cell_value - update

            candidate = jax.vmap(solve_cell)(
                flat_value,
                flat_residual,
                hydrogen_total.reshape((-1,)),
                helium_total.reshape((-1,)),
            ).reshape(flat_shape + (4,))
            norm = jnp.max(jnp.abs(residual(candidate)))
            now_converged = norm <= self.tolerance
            return jnp.where(converged, value, candidate), converged | now_converged

        value, converged = jax.lax.fori_loop(
            0,
            self.maximum_iterations,
            iteration,
            (initial, jnp.asarray(False)),
        )
        result_state = self._full(value, hydrogen_total, helium_total, state.scale_factor)
        final_residual = jnp.max(jnp.abs(residual(value)))
        positive = jnp.all(result_state.number_densities >= 0.0) & jnp.all(
            result_state.internal_energy > 0.0
        )
        successful = converged & positive & jnp.isfinite(final_residual)
        accepted = PrimordialSpeciesState(
            jnp.where(successful, result_state.number_densities, state.number_densities),
            jnp.where(successful, result_state.internal_energy, state.internal_energy),
            state.scale_factor,
        )
        final_charge = accepted.number_densities[..., 5]
        ledger = PrimordialMicrophysicsLedger(
            jnp.max(
                jnp.abs(
                    jnp.sum(accepted.number_densities[..., :2], axis=-1) - hydrogen_total
                )
            ),
            jnp.max(
                jnp.abs(
                    jnp.sum(accepted.number_densities[..., 2:5], axis=-1) - helium_total
                )
            ),
            jnp.max(
                jnp.abs(
                    final_charge
                    - (
                        accepted.number_densities[..., 1]
                        + accepted.number_densities[..., 3]
                        + 2.0 * accepted.number_densities[..., 4]
                    )
                )
            ),
            accepted.internal_energy - state.internal_energy,
            final_residual,
            jnp.asarray(self.maximum_iterations, dtype=jnp.int32),
            positive,
            converged,
            successful,
        )
        return PrimordialMicrophysicsResult(accepted, ledger, successful)

    def apply_to_gas(
        self,
        gas: ComovingEulerState,
        species: PrimordialSpeciesState,
        delta_time: ArrayLike,
        /,
    ) -> tuple[ComovingEulerState, PrimordialMicrophysicsResult]:
        if gas.scale_factor != species.scale_factor:
            raise ValueError("Gas and microphysics scale factors disagree.")
        result = self.advance(species, delta_time)
        density = gas.cell_average[..., 0]
        momentum = gas.cell_average[..., 1:-1]
        kinetic = jnp.sum(momentum**2, axis=-1) / (2.0 * density)
        candidate = gas.cell_average.at[..., -1].set(
            result.state.internal_energy + kinetic
        )
        accepted = ComovingEulerState(
            jnp.where(result.successful, candidate, gas.cell_average),
            gas.scale_factor,
        )
        return accepted, result


__all__ = [
    "PRIMORDIAL_PROCESSES",
    "PRIMORDIAL_SPECIES",
    "PrimordialMicrophysicsLedger",
    "PrimordialMicrophysicsPlan",
    "PrimordialMicrophysicsResult",
    "PrimordialRateTable",
    "PrimordialSpeciesState",
]
