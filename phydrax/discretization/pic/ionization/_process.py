#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Field and electron-impact ionization as population-stage PIC processes.

Both create electrons at the ionized ion's position, allocated through the
run's allocation route (`allocate_particles`), with a compensating charge
transition, so the deposited charge is preserved pointwise.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from ...._fingerprint import canonical_fingerprint
from ...._trainable import NonTrainableState
from ....typing import checked, PRNGKey
from ...particle import ParticlePopulationPlan
from .._charge_state import PICSpeciesPlan, PICSpeciesState
from .._process import (
    AbstractPICProcess,
    PICProcessContext,
    PICProcessLedger,
    PICProcessResult,
    PICProcessStage,
    PICProcessStatePartition,
    RadiationOwnership,
)
from ._field import FieldIonizationPlan
from ._impact import ElectronImpactIonizationPlan
from ._types import PICIonizationResult


def _species_pair(ions: int, electrons: int, /) -> tuple[int, int]:
    ion_index, electron_index = int(ions), int(electrons)
    if ion_index == electron_index:
        raise ValueError("Ionization requires distinct ion and electron species.")
    return ion_index, electron_index


def _require_key(context: PICProcessContext, /) -> PRNGKey:
    if context.key is None:
        raise ValueError("Stochastic PIC ionization requires a derived process key.")
    return context.key


def _ionized_species(
    context: PICProcessContext,
    ions: int,
    electrons: int,
    result: PICIonizationResult,
    process_id: str,
    /,
) -> PICProcessResult:
    ion_state = context.species[ions]
    values = list(context.species)
    values[ions] = PICSpeciesState(
        result.ion_particles, ion_state.population, result.ion_charge
    )
    values[electrons] = PICSpeciesState(
        result.electron_particles, result.electron_population, result.electron_charge
    )
    return PICProcessResult(
        tuple(values),
        PICProcessLedger(
            result.event_count,
            result.charge_defect,
            result.momentum_defect,
            result.energy_defect,
            result.successful,
            process_id,
        ),
    )


def _local_events(maximum_events: int, parts: int, limit: int, /) -> int:
    """One device's share ``⌈maximum_events / parts⌉`` of a step's event budget.

    The share must fit ``limit``, the slot blocks' request capacity.
    """
    if parts <= 0:
        raise ValueError("parts must be positive.")
    events = -(-maximum_events // parts)
    if events > limit:
        raise ValueError(
            "The per-device ionization event share exceeds the species' slot blocks."
        )
    return events


class _AbstractIonizationProcess(AbstractPICProcess):
    """Stateless distribution shell of the ionization processes."""

    def bank_plans(self) -> tuple[ParticlePopulationPlan, ...]:
        """Ionization owns no particle bank."""
        return ()

    def partition_state(self, state: None, /) -> PICProcessStatePartition:
        """Ionization is stateless: an empty partition."""
        if state is not None:
            raise TypeError("Ionization processes carry no state.")
        return PICProcessStatePartition((), (), (), (), ())

    def assemble_state(self, partition: PICProcessStatePartition, /) -> None:
        """Ionization is stateless."""
        if (
            partition.companions
            or partition.banks
            or partition.totals
            or partition.shared
        ):
            raise ValueError("Ionization processes carry no state.")

    def combine(
        self, ledger: PICProcessLedger, evidence: None, /
    ) -> tuple[PICProcessLedger, None]:
        """The run's ledger from per-device rows.

        Events, the signed charge source and the ionization energy add; the
        momentum defect (a nonnegative norm) takes the largest device value;
        success requires every device.
        """
        if evidence is not None:
            raise TypeError("Ionization processes report no evidence.")
        return PICProcessLedger(
            jnp.sum(ledger.event_count, axis=0, dtype=ledger.event_count.dtype),
            jnp.sum(ledger.charge_defect, axis=0, dtype=ledger.charge_defect.dtype),
            jnp.max(ledger.momentum_defect, axis=0),
            jnp.sum(ledger.energy_defect, axis=0, dtype=ledger.energy_defect.dtype),
            jnp.all(ledger.successful, axis=0),
            self.process_id,
        ), None


class FieldIonizationProcess(_AbstractIonizationProcess, NonTrainableState):
    """`FieldIonizationPlan` driven by the push-stage electric field on the ions."""

    plan: FieldIonizationPlan
    ions: int = eqx.field(static=True)
    electrons: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    stage: PICProcessStage = eqx.field(static=True)
    stochastic: bool = eqx.field(static=True)
    radiation_ownership: RadiationOwnership | None = eqx.field(static=True)
    species_indices: tuple[int, ...] = eqx.field(static=True)

    @checked
    def __init__(self, plan: FieldIonizationPlan, ions: int, electrons: int, /) -> None:
        ion_index, electron_index = _species_pair(ions, electrons)
        self.plan = plan
        self.ions = ion_index
        self.electrons = electron_index
        self.stage = "population"
        self.stochastic = True
        self.radiation_ownership = None
        self.species_indices = (ion_index, electron_index)
        self.process_id = canonical_fingerprint(
            {
                "kind": "pic-field-ionization-process",
                "plan": plan.plan_id,
                "ions": ion_index,
                "electrons": electron_index,
            }
        )

    def apply(
        self,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        key = _require_key(context)
        ion_plan, electron_plan = species[self.ions], species[self.electrons]
        ions, electrons = context.species[self.ions], context.species[self.electrons]
        result = self.plan.apply(
            ion_plan.charge_model,
            ions.population,
            ions.charge,
            ions.particles,
            context.electric[self.ions],
            electron_plan.charge_model,
            electron_plan.population,
            electrons.population,
            electrons.charge,
            electrons.particles,
            key,
            context.step_size,
            context.step_index,
            allocator=context.allocator,
        )
        return _ionized_species(
            context, self.ions, self.electrons, result, self.process_id
        )

    def localize(
        self, species: tuple[PICSpeciesPlan, ...], parts: int, /
    ) -> FieldIonizationProcess:
        """The process on one device's slot blocks.

        Each device ionizes at most its share ``⌈maximum_events / parts⌉`` of
        the step's event budget, which must fit the ion block and the electron
        block's allocation capacity. The local view keeps the run's process
        identity, so its random stream is the run's.
        """
        plan = self.plan
        events = _local_events(
            plan.maximum_events,
            parts,
            min(
                species[self.ions].capacity,
                species[self.electrons].population.allocation_capacity,
            ),
        )
        local = FieldIonizationPlan(
            plan.rate_coefficient,
            field_power=plan.field_power,
            ionization_energy=plan.ionization_energy,
            maximum_probability=plan.maximum_probability,
            maximum_events=events,
        )
        return eqx.tree_at(lambda process: process.plan, self, local)


def _random_active_slots(key: PRNGKey, active: Array, count: int, /) -> Array:
    """Up to ``count`` distinct active slots in random order; ``-1`` pads."""
    scores = jr.uniform(key, active.shape, dtype=jnp.float64)
    order = jnp.argsort(jnp.where(active, scores, jnp.inf))
    size = min(count, active.shape[0])
    selected = order[:size].astype(jnp.int32)
    selected = jnp.where(active[selected], selected, -1)
    return jnp.pad(selected, (0, count - size), constant_values=-1)


class ImpactIonizationProcess(_AbstractIonizationProcess, NonTrainableState):
    """`ElectronImpactIonizationPlan` over randomly paired active ions/electrons.

    Each step pairs up to ``maximum_events`` distinct active ions with distinct
    active electrons in independent random order drawn from the process key;
    unpaired lanes are ``-1`` and inert.
    """

    plan: ElectronImpactIonizationPlan
    ions: int = eqx.field(static=True)
    electrons: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    stage: PICProcessStage = eqx.field(static=True)
    stochastic: bool = eqx.field(static=True)
    radiation_ownership: RadiationOwnership | None = eqx.field(static=True)
    species_indices: tuple[int, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: ElectronImpactIonizationPlan, ions: int, electrons: int, /
    ) -> None:
        ion_index, electron_index = _species_pair(ions, electrons)
        self.plan = plan
        self.ions = ion_index
        self.electrons = electron_index
        self.stage = "population"
        self.stochastic = True
        self.radiation_ownership = None
        self.species_indices = (ion_index, electron_index)
        self.process_id = canonical_fingerprint(
            {
                "kind": "pic-impact-ionization-process",
                "plan": plan.plan_id,
                "ions": ion_index,
                "electrons": electron_index,
            }
        )

    def apply(
        self,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        key = _require_key(context)
        ion_plan, electron_plan = species[self.ions], species[self.electrons]
        ions, electrons = context.species[self.ions], context.species[self.electrons]
        ion_key, electron_key, event_key = jr.split(key, 3)
        events = self.plan.maximum_events
        result = self.plan.apply(
            ion_plan.charge_model,
            ions.population,
            ions.charge,
            ions.particles,
            electron_plan.charge_model,
            electron_plan.population,
            electrons.population,
            electrons.charge,
            electrons.particles,
            _random_active_slots(ion_key, ions.population.active, events),
            _random_active_slots(electron_key, electrons.population.active, events),
            event_key,
            context.step_size,
            context.step_index,
            allocator=context.allocator,
        )
        return _ionized_species(
            context, self.ions, self.electrons, result, self.process_id
        )

    def localize(
        self, species: tuple[PICSpeciesPlan, ...], parts: int, /
    ) -> ImpactIonizationProcess:
        """The process on one device's slot blocks.

        Each device pairs at most its share ``⌈maximum_events / parts⌉`` of the
        step's pair budget among its own ions and electrons, which must fit the
        electron block's allocation capacity. The local view keeps the run's
        process identity, so its random stream is the run's.
        """
        plan = self.plan
        events = _local_events(
            plan.maximum_events,
            parts,
            species[self.electrons].population.allocation_capacity,
        )
        local = ElectronImpactIonizationPlan(
            plan.energy_grid,
            plan.cross_section,
            ionization_energy=plan.ionization_energy,
            rate_scale=plan.rate_scale,
            maximum_probability=plan.maximum_probability,
            maximum_events=events,
        )
        return eqx.tree_at(lambda process: process.plan, self, local)


__all__ = ["FieldIonizationProcess", "ImpactIonizationProcess"]
