#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._chemical_mechanism import PreparedChemicalMechanism
from ...equations._chemical_rates import ChemicalRateRuntime
from ...equations._nonequilibrium_gas import (
    TwoTemperatureMixtureEulerSystem,
    TwoTemperatureMixtureNavierStokesSystem,
)
from ...solver._balance_law import (
    AbstractBalanceLawProcessPlan,
    AbstractPreparedBalanceLawProcess,
    BalanceLawProcessAdvance,
    BalanceLawProcessState,
)
from ...solver._balance_law_transport import (
    AbstractPreparedBalanceLawTransport,
    BalanceLawSourceView,
)


class LandauTellerRelaxationEvaluation(StrictModule):
    equilibrium_mode_energy: Array
    mode_energy_source: Array
    relaxation_times: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class LandauTellerRelaxationPlan(StrictModule, NonTrainableState):
    """Fixed relaxation times for explicit thermal-mode energy pools."""

    relaxation_times: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, relaxation_times: ArrayLike, /):
        times = jnp.asarray(relaxation_times)
        host = np.asarray(times)
        if (
            times.ndim != 1
            or times.size == 0
            or np.any(~np.isfinite(host))
            or np.any(host <= 0.0)
        ):
            raise ValueError("Relaxation times must be finite and positive.")
        self.relaxation_times = times
        self.plan_id = canonical_fingerprint(
            {
                "kind": "landau-teller-relaxation",
                "times": array_tree_fingerprint(times),
            }
        )

    def evaluate(
        self,
        system: TwoTemperatureMixtureEulerSystem
        | TwoTemperatureMixtureNavierStokesSystem,
        state: ArrayLike,
        /,
    ) -> LandauTellerRelaxationEvaluation:
        if not isinstance(
            system,
            (
                TwoTemperatureMixtureEulerSystem,
                TwoTemperatureMixtureNavierStokesSystem,
            ),
        ):
            raise TypeError("Landau-Teller relaxation requires a two-temperature system.")
        if self.relaxation_times.shape != (system.mode_count,):
            raise ValueError("Relaxation times must match the system mode count.")
        value = jnp.asarray(state)
        recovered = system.recover_thermodynamics(value)
        species_density = value[..., : system.species_count]
        equilibrium_temperatures = jnp.broadcast_to(
            recovered.state.heavy_temperature[..., None],
            value.shape[:-1] + (system.mode_count,),
        )
        equilibrium = system.thermodynamics.modes.evaluate(
            species_density, equilibrium_temperatures
        )
        current = value[..., system.mode_slice]
        times = self.relaxation_times.astype(value.dtype)
        source = (equilibrium.energy_densities - current) / times
        finite = (
            recovered.finite & equilibrium.finite & jnp.all(jnp.isfinite(source), axis=-1)
        )
        successful = finite & recovered.successful & equilibrium.successful
        return LandauTellerRelaxationEvaluation(
            equilibrium.energy_densities,
            source,
            times,
            finite,
            successful,
            self.plan_id,
        )


class ThermochemicalNonequilibriumDiagnostics(StrictModule):
    species_before: Array
    species_after: Array
    mode_energy_before: Array
    mode_energy_after: Array
    invariant_defect: Array
    total_energy_change: Array
    relaxation_residual: Array
    finite: Array
    successful: Array


class ThermochemicalNonequilibriumProcessPlan(AbstractBalanceLawProcessPlan):
    """Joint fixed-work chemistry and mode relaxation with one energy ledger."""

    relaxation: LandauTellerRelaxationPlan
    mechanism: PreparedChemicalMechanism | None
    subcycles: int = eqx.field(static=True)
    nonlinear_iterations: int = eqx.field(static=True)
    chemistry_temperature: Literal["heavy", "geometric-mean"] = eqx.field(static=True)

    def __init__(
        self,
        relaxation: LandauTellerRelaxationPlan,
        mechanism: PreparedChemicalMechanism | None = None,
        /,
        *,
        subcycles: int = 8,
        nonlinear_iterations: int = 8,
        chemistry_temperature: Literal["heavy", "geometric-mean"] = "geometric-mean",
    ):
        subcycles_ = int(subcycles)
        iterations = int(nonlinear_iterations)
        if (
            not isinstance(relaxation, LandauTellerRelaxationPlan)
            or (
                mechanism is not None
                and not isinstance(mechanism, PreparedChemicalMechanism)
            )
            or subcycles_ <= 0
            or iterations <= 0
            or chemistry_temperature not in ("heavy", "geometric-mean")
        ):
            raise ValueError("Nonequilibrium process inputs are invalid.")
        self.relaxation = relaxation
        self.mechanism = mechanism
        self.subcycles = subcycles_
        self.nonlinear_iterations = iterations
        self.chemistry_temperature = chemistry_temperature
        self.process_id = canonical_fingerprint(
            {
                "kind": "thermochemical-nonequilibrium-process",
                "relaxation": relaxation.plan_id,
                "mechanism": None if mechanism is None else mechanism.mechanism_id,
                "subcycles": subcycles_,
                "nonlinear_iterations": iterations,
                "chemistry_temperature": chemistry_temperature,
            }
        )

    def prepare(
        self, transport: AbstractPreparedBalanceLawTransport, /
    ) -> PreparedThermochemicalNonequilibriumProcess:
        return PreparedThermochemicalNonequilibriumProcess(self, transport)


class PreparedThermochemicalNonequilibriumProcess(AbstractPreparedBalanceLawProcess):
    plan: ThermochemicalNonequilibriumProcessPlan
    transport: AbstractPreparedBalanceLawTransport
    species_indices: tuple[int, ...] = eqx.field(static=True)
    mode_indices: tuple[int, ...] = eqx.field(static=True)
    energy_index: int = eqx.field(static=True)

    def __init__(
        self,
        plan: ThermochemicalNonequilibriumProcessPlan,
        transport: AbstractPreparedBalanceLawTransport,
        /,
    ):
        system = transport.dynamics.system
        if not isinstance(
            system,
            (
                TwoTemperatureMixtureEulerSystem,
                TwoTemperatureMixtureNavierStokesSystem,
            ),
        ):
            raise TypeError("Nonequilibrium process requires a two-temperature system.")
        if plan.relaxation.relaxation_times.shape != (system.mode_count,):
            raise ValueError("Relaxation plan and gas mode counts differ.")
        if plan.mechanism is not None and (
            plan.mechanism.schema.schema_id != system.thermodynamics.schema.schema_id
            or plan.mechanism.thermodynamics.thermodynamics_id
            != system.thermodynamics.heavy_thermodynamics.thermodynamics_id
        ):
            raise ValueError("Chemistry and heavy-particle thermodynamics must match.")
        self.plan = plan
        self.transport = transport
        self.species_indices = tuple(range(system.species_count))
        self.mode_indices = tuple(range(system.mode_slice.start, system.mode_slice.stop))
        self.energy_index = system.energy_index
        self.process_id = canonical_fingerprint(
            {
                "kind": "prepared-thermochemical-nonequilibrium-process",
                "plan": plan.process_id,
                "transport": transport.transport_id,
            }
        )
        self.requires_realization = False
        self.realization_name = None
        self.differentiability = "branchwise-fixed-work"
        self.modified_components = tuple(
            system.component_names[index]
            for index in (*self.species_indices, *self.mode_indices)
        )

    @property
    def system(self):
        return self.transport.dynamics.system

    def initialize(
        self, source_view: BalanceLawSourceView, args: Any = None, /
    ) -> BalanceLawProcessState:
        del source_view, args
        return BalanceLawProcessState.empty(self.process_id)

    def _chemistry_rate(
        self, state: Array, runtime: ChemicalRateRuntime | None, /
    ) -> tuple[Array, Array]:
        mechanism = self.plan.mechanism
        if mechanism is None:
            return (
                jnp.zeros(
                    state.shape[:-1] + (self.system.species_count,), dtype=state.dtype
                ),
                jnp.ones(state.shape[:-1], dtype=bool),
            )
        recovered = self.system.recover_thermodynamics(state)
        heavy = recovered.state.heavy_temperature
        mode_mean = jnp.mean(recovered.state.mode_temperatures, axis=-1)
        temperature = (
            heavy
            if self.plan.chemistry_temperature == "heavy"
            else jnp.sqrt(heavy * mode_mean)
        )
        species_density = state[..., : self.system.species_count]
        concentration = species_density / mechanism.schema.molar_masses.astype(
            state.dtype
        )
        rates = mechanism.evaluate(
            concentration,
            temperature,
            recovered.state.pressure,
            runtime=runtime,
        )
        mass_rate = rates.species_amount_rate * mechanism.schema.molar_masses.astype(
            state.dtype
        )
        return mass_rate, recovered.successful & rates.successful

    def step_limit(
        self,
        time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        args: Any = None,
        /,
    ) -> Array:
        del time, process_state
        runtime = args if isinstance(args, ChemicalRateRuntime) else None
        rate, successful = self._chemistry_rate(cell_average, runtime)
        species = cell_average[..., : self.system.species_count]
        limit = jnp.min(
            jnp.where(rate < 0.0, 0.25 * species / jnp.maximum(-rate, 1.0e-30), jnp.inf)
        )
        return jnp.where(jnp.all(successful), limit, 0.0)

    def advance(
        self,
        start_time: Array,
        end_time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        realization: Any = None,
        args: Any = None,
        /,
    ) -> BalanceLawProcessAdvance:
        del realization
        incoming = jnp.asarray(cell_average)
        species_before = incoming[..., self.species_indices]
        modes_before = incoming[..., self.mode_indices]
        total_energy_before = incoming[..., self.energy_index]
        step = (end_time - start_time) / self.plan.subcycles
        runtime = args if isinstance(args, ChemicalRateRuntime) else None

        def subcycle(_, carry):
            candidate, chemistry_success = carry
            mass_rate, rate_success = self._chemistry_rate(candidate, runtime)
            candidate = candidate.at[..., self.species_indices].add(step * mass_rate)
            mode_start = candidate[..., self.mode_indices]

            def relaxation_iteration(_, current):
                relaxation = self.plan.relaxation.evaluate(self.system, current)
                fraction = -jnp.expm1(
                    -step / relaxation.relaxation_times.astype(current.dtype)
                )
                mode_energy = mode_start + fraction * (
                    relaxation.equilibrium_mode_energy - mode_start
                )
                return current.at[..., self.mode_indices].set(mode_energy)

            candidate = jax.lax.fori_loop(
                0,
                self.plan.nonlinear_iterations,
                relaxation_iteration,
                candidate,
            )
            return candidate, chemistry_success & jnp.all(rate_success)

        candidate, chemistry_success = jax.lax.fori_loop(
            0,
            self.plan.subcycles,
            subcycle,
            (incoming, jnp.asarray(True)),
        )
        final_relaxation = self.plan.relaxation.evaluate(self.system, candidate)
        relaxation_residual = jnp.max(
            jnp.abs(final_relaxation.mode_energy_source), axis=-1
        )
        species_after = candidate[..., self.species_indices]
        schema = self.system.thermodynamics.schema
        amount_before = species_before / schema.molar_masses.astype(incoming.dtype)
        amount_after = species_after / schema.molar_masses.astype(incoming.dtype)
        element_defect = schema.element_amount(amount_after) - schema.element_amount(
            amount_before
        )
        charge_defect = schema.charge_amount(amount_after) - schema.charge_amount(
            amount_before
        )
        invariant_defect = jnp.concatenate(
            (element_defect, charge_defect[..., None]), axis=-1
        )
        finite = (
            jnp.all(jnp.isfinite(candidate), axis=-1)
            & jnp.all(jnp.isfinite(invariant_defect), axis=-1)
            & jnp.isfinite(relaxation_residual)
        )
        local_success = (
            finite
            & chemistry_success
            & final_relaxation.successful
            & self.system.admissible(candidate)
            & jnp.all(species_after >= 0.0, axis=-1)
            & jnp.all(jnp.abs(invariant_defect) <= 1.0e-9, axis=-1)
        )
        successful = jnp.all(local_success)
        accepted = jnp.where(successful, candidate, incoming)
        diagnostics = ThermochemicalNonequilibriumDiagnostics(
            species_before,
            accepted[..., self.species_indices],
            modes_before,
            accepted[..., self.mode_indices],
            invariant_defect,
            accepted[..., self.energy_index] - total_energy_before,
            relaxation_residual,
            jnp.all(finite),
            successful,
        )
        return BalanceLawProcessAdvance(
            accepted,
            process_state,
            successful,
            accepted - incoming,
            diagnostics,
        )


__all__ = [
    "LandauTellerRelaxationEvaluation",
    "LandauTellerRelaxationPlan",
    "PreparedThermochemicalNonequilibriumProcess",
    "ThermochemicalNonequilibriumDiagnostics",
    "ThermochemicalNonequilibriumProcessPlan",
]
