#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._chemical_rates import ChemicalRateRuntime
from ..equations._ionized_gas import (
    IonizedMultitemperatureEulerSystem,
    IonizedMultitemperatureNavierStokesSystem,
)
from ..equations._plasma_chemistry import PreparedPlasmaMechanism
from ..linalg._dense_inverse import dense_inverse


class ThermochemicalSourceEvidence(StrictModule):
    residual_norm: Array
    element_defect: Array
    charge_defect: Array
    energy_defect: Array
    minimum_species_density: Array
    minimum_mode_energy: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ThermochemicalSourceResult(StrictModule):
    candidate: Array
    accepted: Array
    source_increment: Array
    evidence: ThermochemicalSourceEvidence


class FixedWorkThermochemicalSourcePlan(StrictModule, NonTrainableState):
    """Fixed-work backward-Euler plasma source solve with physical line search."""

    mechanism: PreparedPlasmaMechanism
    substeps: int = eqx.field(static=True)
    newton_iterations: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    minimum_line_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mechanism: PreparedPlasmaMechanism,
        /,
        *,
        substeps: int = 4,
        newton_iterations: int = 10,
        residual_tolerance: float = 1.0e-9,
        minimum_line_fraction: float = 1.0e-8,
    ):
        substeps_ = int(substeps)
        iterations = int(newton_iterations)
        tolerance = float(residual_tolerance)
        line = float(minimum_line_fraction)
        if (
            not isinstance(mechanism, PreparedPlasmaMechanism)
            or substeps_ <= 0
            or iterations <= 0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or not np.isfinite(line)
            or not 0.0 < line <= 1.0
        ):
            raise ValueError("Thermochemical source controls are invalid.")
        self.mechanism = mechanism
        self.substeps = substeps_
        self.newton_iterations = iterations
        self.residual_tolerance = tolerance
        self.minimum_line_fraction = line
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-work-thermochemical-source",
                "mechanism": mechanism.mechanism_id,
                "substeps": substeps_,
                "newton_iterations": iterations,
                "residual_tolerance": tolerance,
                "minimum_line_fraction": line,
            }
        )

    def _check_system(self, system: Any, /) -> None:
        if not isinstance(
            system,
            (
                IonizedMultitemperatureEulerSystem,
                IonizedMultitemperatureNavierStokesSystem,
            ),
        ):
            raise TypeError("Plasma source solve requires an ionized gas system.")
        if (
            system.thermodynamics.schema.schema_id
            != self.mechanism.mechanism.schema.schema_id
            or system.mode_count != self.mechanism.mode_count
        ):
            raise ValueError("Source mechanism and ionized gas state do not match.")

    def _one_cell(
        self,
        system: Any,
        incoming: Array,
        step: Array,
        runtime: ChemicalRateRuntime,
        /,
    ) -> tuple[Array, Array]:
        species_count = system.species_count
        mode_count = system.mode_count
        molar_masses = system.thermodynamics.schema.molar_masses.astype(incoming.dtype)

        def assemble(base, unknown):
            value = base.at[:species_count].set(unknown[:species_count])
            return value.at[system.mode_slice].set(unknown[species_count:])

        def source(unknown, base):
            state = assemble(base, unknown)
            recovered = system.recover_thermodynamics(state)
            concentrations = state[:species_count] / molar_masses
            evaluation = self.mechanism.evaluate(
                concentrations,
                recovered.base_recovery.state.heavy_temperature,
                recovered.base_recovery.state.mode_temperatures,
                recovered.electron_temperature,
                recovered.total_pressure,
                runtime=runtime,
            )
            species_rate = evaluation.species_amount_rate * molar_masses
            return jnp.concatenate((species_rate, evaluation.mode_energy_rate))

        def one_substep(_, state):
            initial_unknown = jnp.concatenate(
                (state[:species_count], state[system.mode_slice])
            )

            def residual(unknown):
                return unknown - initial_unknown - step * source(unknown, state)

            def newton_body(_, unknown):
                value = residual(unknown)
                jacobian = jax.jacfwd(residual)(unknown)
                direction = -contract(
                    "ij,j->i",
                    dense_inverse(jacobian),
                    value,
                    backend="jax",
                )
                admissible_bound = jnp.min(
                    jnp.where(
                        direction < 0.0,
                        -0.9
                        * unknown
                        / jnp.minimum(direction, -jnp.finfo(direction.dtype).tiny),
                        1.0,
                    )
                )
                fraction = jnp.clip(admissible_bound, self.minimum_line_fraction, 1.0)
                return unknown + fraction * direction

            unknown = jax.lax.fori_loop(
                0,
                self.newton_iterations,
                newton_body,
                initial_unknown,
            )
            candidate = assemble(state, unknown)
            return candidate

        state = jax.lax.fori_loop(0, self.substeps, one_substep, incoming)
        final_unknown = jnp.concatenate((state[:species_count], state[system.mode_slice]))
        initial_unknown = jnp.concatenate(
            (incoming[:species_count], incoming[system.mode_slice])
        )
        substep = step
        residual = (
            final_unknown
            - initial_unknown
            - self.substeps * substep * source(final_unknown, incoming)
        )
        return state, jnp.max(jnp.abs(residual))

    def advance(
        self,
        system: Any,
        state: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        runtime: ChemicalRateRuntime | None = None,
    ) -> ThermochemicalSourceResult:
        self._check_system(system)
        incoming = jnp.asarray(state)
        step = jnp.asarray(step_size, dtype=incoming.dtype)
        if incoming.ndim < 1 or incoming.shape[-1] != system.component_count:
            raise ValueError("Thermochemical source state shape is invalid.")
        runtime_ = ChemicalRateRuntime() if runtime is None else runtime
        flat = incoming.reshape((-1, system.component_count))
        candidate_flat, residual_flat = jax.vmap(
            lambda cell: self._one_cell(system, cell, step / self.substeps, runtime_)
        )(flat)
        candidate = candidate_flat.reshape(incoming.shape)
        residual_norm = residual_flat.reshape(incoming.shape[:-1])
        species_before = incoming[..., : system.species_count]
        species_after = candidate[..., : system.species_count]
        schema = system.thermodynamics.schema
        amount_before = species_before / schema.molar_masses.astype(incoming.dtype)
        amount_after = species_after / schema.molar_masses.astype(incoming.dtype)
        element_defect = schema.element_amount(amount_after) - schema.element_amount(
            amount_before
        )
        charge_defect = schema.charge_amount(amount_after) - schema.charge_amount(
            amount_before
        )
        energy_defect = (
            candidate[..., system.energy_index] - incoming[..., system.energy_index]
        )
        scale = jnp.maximum(jnp.max(jnp.abs(candidate), axis=-1), 1.0)
        finite = jnp.all(jnp.isfinite(candidate), axis=-1) & jnp.isfinite(residual_norm)
        successful_cells = (
            finite
            & (residual_norm <= self.residual_tolerance * scale)
            & system.admissible(candidate)
            & jnp.all(species_after >= 0.0, axis=-1)
            & jnp.all(candidate[..., system.mode_slice] >= 0.0, axis=-1)
            & jnp.all(jnp.abs(element_defect) <= 1.0e-9, axis=-1)
            & (jnp.abs(charge_defect) <= 1.0e-9)
            & (jnp.abs(energy_defect) <= 1.0e-12 * scale)
        )
        successful = jnp.all(successful_cells)
        accepted = jnp.where(successful, candidate, incoming)
        evidence = ThermochemicalSourceEvidence(
            jnp.max(residual_norm),
            element_defect,
            charge_defect,
            energy_defect,
            jnp.min(species_after),
            jnp.min(candidate[..., system.mode_slice]),
            jnp.all(finite),
            successful,
            self.plan_id,
        )
        return ThermochemicalSourceResult(
            candidate,
            accepted,
            accepted - incoming,
            evidence,
        )


__all__ = [
    "FixedWorkThermochemicalSourcePlan",
    "ThermochemicalSourceEvidence",
    "ThermochemicalSourceResult",
]
