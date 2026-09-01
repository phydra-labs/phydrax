#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..equations._chemical_mechanism import PreparedChemicalMechanism
from ..equations._chemical_rates import ChemicalRateRuntime
from ..equations._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ..equations._hyperbolic_systems import MultispeciesEulerSystem
from ._balance_law import (
    AbstractBalanceLawProcessPlan,
    AbstractPreparedBalanceLawProcess,
    BalanceLawProcessAdvance,
    BalanceLawProcessState,
)
from ._balance_law_transport import (
    AbstractPreparedBalanceLawTransport,
    BalanceLawSourceView,
)


class ThermochemistryDiagnostics(StrictModule):
    species_before: Array
    species_after: Array
    invariant_defect: Array
    energy_change: Array
    successful: Array


class ThermochemistryProcessPlan(AbstractBalanceLawProcessPlan):
    mechanism: PreparedChemicalMechanism
    subcycles: int = eqx.field(static=True)
    safety_fraction: float = eqx.field(static=True)

    def __init__(
        self,
        mechanism: PreparedChemicalMechanism,
        /,
        *,
        subcycles: int = 8,
        safety_fraction: float = 0.25,
    ):
        if not isinstance(mechanism, PreparedChemicalMechanism):
            raise TypeError("mechanism must be PreparedChemicalMechanism.")
        count = int(subcycles)
        fraction = float(safety_fraction)
        if count <= 0 or not 0.0 < fraction <= 1.0:
            raise ValueError("Thermochemistry integration controls are invalid.")
        self.mechanism = mechanism
        self.subcycles = count
        self.safety_fraction = fraction
        self.process_id = canonical_fingerprint(
            {
                "kind": "thermochemistry-process",
                "mechanism": mechanism.mechanism_id,
                "subcycles": count,
                "safety_fraction": fraction,
            }
        )

    def prepare(
        self, transport: AbstractPreparedBalanceLawTransport, /
    ) -> PreparedThermochemistryProcess:
        return PreparedThermochemistryProcess(self, transport)


class PreparedThermochemistryProcess(AbstractPreparedBalanceLawProcess):
    plan: ThermochemistryProcessPlan
    transport: AbstractPreparedBalanceLawTransport
    species_indices: tuple[int, ...] = eqx.field(static=True)
    energy_index: int = eqx.field(static=True)

    def __init__(
        self,
        plan: ThermochemistryProcessPlan,
        transport: AbstractPreparedBalanceLawTransport,
        /,
    ):
        if not isinstance(transport.dynamics.system, MultispeciesEulerSystem):
            raise TypeError("Continuum thermochemistry requires MultispeciesEulerSystem.")
        system = transport.dynamics.system
        if system.species_count != plan.mechanism.schema.species_count:
            raise ValueError("Chemical mechanism species do not match transport system.")
        self.plan = plan
        self.transport = transport
        self.species_indices = tuple(range(system.species_count))
        self.energy_index = len(system.component_names) - 1
        self.process_id = canonical_fingerprint(
            {
                "kind": "prepared-thermochemistry-process",
                "plan": plan.process_id,
                "transport": transport.transport_id,
            }
        )
        self.requires_realization = False
        self.realization_name = None
        self.differentiability = "branchwise-explicit-subcycled"
        self.modified_components = tuple(
            system.component_names[index] for index in self.species_indices
        ) + ("total_energy",)

    def initialize(
        self, source_view: BalanceLawSourceView, args: Any = None, /
    ) -> BalanceLawProcessState:
        del source_view, args
        return BalanceLawProcessState.empty(self.process_id)

    def step_limit(
        self,
        time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        args: Any = None,
        /,
    ) -> Array:
        del time, process_state
        evaluation = self._evaluate(cell_average, args)
        return self.plan.safety_fraction * jnp.min(evaluation.explicit_step_restriction)

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
        energy_before = incoming[..., self.energy_index]
        substep = (end_time - start_time) / self.plan.subcycles

        def body(_, candidate):
            evaluation = self._evaluate(candidate, args)
            mass_rate = (
                evaluation.species_amount_rate * self.plan.mechanism.schema.molar_masses
            )
            updated = candidate.at[..., self.species_indices].add(substep * mass_rate)
            return updated.at[..., self.energy_index].add(
                substep * evaluation.molar_energy_rate
            )

        candidate = jax.lax.fori_loop(
            0,
            self.plan.subcycles,
            body,
            incoming,
        )
        species_after = candidate[..., self.species_indices]
        amount_before = species_before / self.plan.mechanism.schema.molar_masses
        amount_after = species_after / self.plan.mechanism.schema.molar_masses
        element_before = contract(
            "...s,es->...e",
            amount_before,
            self.plan.mechanism.schema.element_composition,
        )
        element_after = contract(
            "...s,es->...e",
            amount_after,
            self.plan.mechanism.schema.element_composition,
        )
        charge_before = contract(
            "...s,s->...",
            amount_before,
            self.plan.mechanism.schema.charges,
        )
        charge_after = contract(
            "...s,s->...",
            amount_after,
            self.plan.mechanism.schema.charges,
        )
        invariant_defect = jnp.concatenate(
            ((element_after - element_before), (charge_after - charge_before)[..., None]),
            axis=-1,
        )
        final_evaluation = self._evaluate(candidate, args)
        successful = (
            final_evaluation.successful
            & jnp.all(species_after >= 0.0)
            & jnp.all(jnp.isfinite(candidate))
            & jnp.all(jnp.abs(invariant_defect) <= 1.0e-10)
            & jnp.all(self.transport.dynamics.system.admissible(candidate))
        )
        accepted = jnp.where(successful, candidate, incoming)
        diagnostics = ThermochemistryDiagnostics(
            species_before=species_before,
            species_after=accepted[..., self.species_indices],
            invariant_defect=invariant_defect,
            energy_change=accepted[..., self.energy_index] - energy_before,
            successful=successful,
        )
        return BalanceLawProcessAdvance(
            cell_average=accepted,
            process_state=process_state,
            successful=successful,
            source_change=accepted - incoming,
            diagnostics=diagnostics,
        )

    def _evaluate(self, state, args):
        system = self.transport.dynamics.system
        mass_density = state[..., self.species_indices]
        concentration = mass_density / self.plan.mechanism.schema.molar_masses
        pressure = system.pressure(state)
        temperature = pressure / (
            jnp.maximum(jnp.sum(concentration, axis=-1), jnp.finfo(state.dtype).tiny)
            * UNIVERSAL_GAS_CONSTANT
        )
        runtime = args if isinstance(args, ChemicalRateRuntime) else None
        return self.plan.mechanism.evaluate(
            concentration,
            temperature,
            pressure,
            runtime=runtime,
        )


__all__ = [
    "PreparedThermochemistryProcess",
    "ThermochemistryDiagnostics",
    "ThermochemistryProcessPlan",
]
