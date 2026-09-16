#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ._anti_trapping import AntiTrappingCurrentEvaluation, AntiTrappingCurrentPlan
from ._coupling_graph import PhaseFieldCouplingGraph
from ._electrostatic import (
    ElectrochemicalCouplingEvaluation,
    ElectrochemicalCouplingPlan,
    ElectrostaticCouplingEvaluation,
    PhaseElectrostaticCouplingPlan,
)
from ._flow import ModelHCouplingEvaluation, ModelHCouplingPlan
from ._mechanics import PhaseMechanicalEvaluation, PhaseMechanicalModel
from ._multiphysics_ledger import (
    ConservationChannelBalance,
    CoupledPhaseFieldLedger,
    EnergyChannelBalance,
    InternalEnergyExchange,
)
from ._nucleation import (
    NucleationClockState,
    NucleationEventPlan,
    NucleationProposal,
    NucleationTransaction,
)
from ._thermal import (
    NonisothermalSolidificationPlan,
    NonisothermalSolidificationState,
    ThermalSolidificationEvidence,
)


class CoupledMultiphysicsState(StrictModule):
    thermal: NonisothermalSolidificationState
    nucleation: NucleationClockState
    mechanical_energy: Array
    kinetic_energy: Array
    electrostatic_energy: Array
    component_inventory: Array
    thermal_reservoir: Array
    accepted_steps: Array


class CoupledMultiphysicsStepInputs(StrictModule):
    phase_logits: Array
    chemical_potential: Array
    phase_rate: Array
    phase_value: Array
    phase_gradient: Array
    scalar_chemical_potential: Array
    scalar_chemical_gradient: Array
    displacement_gradient: Array
    velocity: Array
    velocity_gradient: Array
    electric_potential: Array
    potential_gradient: Array
    displacement_divergence: Array
    free_charge: Array
    concentrations: Array
    component_chemical_potentials: Array
    component_chemical_gradients: Array
    nucleation_driving_force: Array
    nucleation_temperature: Array
    heat_input: Array
    entropy_flux: Array
    entropy_production: Array
    mechanical_work: Array
    flow_work: Array
    electrical_work: Array


class CoupledMultiphysicsEvidence(StrictModule):
    thermal: ThermalSolidificationEvidence
    anti_trapping: AntiTrappingCurrentEvaluation
    mechanics: PhaseMechanicalEvaluation
    flow: ModelHCouplingEvaluation
    electrostatic: ElectrostaticCouplingEvaluation
    electrochemical: ElectrochemicalCouplingEvaluation
    nucleation_proposal: NucleationProposal
    nucleation_transaction: NucleationTransaction
    ledger: CoupledPhaseFieldLedger
    finite: Array
    successful: Array


class CoupledMultiphysicsStepResult(StrictModule):
    candidate_state: CoupledMultiphysicsState
    accepted_state: CoupledMultiphysicsState
    successful: Array
    evidence: CoupledMultiphysicsEvidence


class CoupledMultiphysicsPlan(StrictModule, NonTrainableState):
    graph: PhaseFieldCouplingGraph
    thermal: NonisothermalSolidificationPlan
    anti_trapping: AntiTrappingCurrentPlan
    nucleation: NucleationEventPlan
    mechanics: PhaseMechanicalModel
    flow: ModelHCouplingPlan
    electrostatic: PhaseElectrostaticCouplingPlan
    electrochemical: ElectrochemicalCouplingPlan
    energy_tolerance: float = eqx.field(static=True)
    exchange_tolerance: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    entropy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        graph: PhaseFieldCouplingGraph,
        thermal: NonisothermalSolidificationPlan,
        anti_trapping: AntiTrappingCurrentPlan,
        nucleation: NucleationEventPlan,
        mechanics: PhaseMechanicalModel,
        flow: ModelHCouplingPlan,
        electrostatic: PhaseElectrostaticCouplingPlan,
        electrochemical: ElectrochemicalCouplingPlan,
        /,
        *,
        energy_tolerance: float = 1.0e-8,
        exchange_tolerance: float = 1.0e-8,
        conservation_tolerance: float = 1.0e-8,
        entropy_tolerance: float = 1.0e-8,
    ):
        required = (
            (graph, PhaseFieldCouplingGraph),
            (thermal, NonisothermalSolidificationPlan),
            (anti_trapping, AntiTrappingCurrentPlan),
            (nucleation, NucleationEventPlan),
            (mechanics, PhaseMechanicalModel),
            (flow, ModelHCouplingPlan),
            (electrostatic, PhaseElectrostaticCouplingPlan),
            (electrochemical, ElectrochemicalCouplingPlan),
        )
        if any(not isinstance(value, expected) for value, expected in required):
            raise TypeError("Coupled multiphysics plan received an incompatible term.")
        tolerances = tuple(
            float(value)
            for value in (
                energy_tolerance,
                exchange_tolerance,
                conservation_tolerance,
                entropy_tolerance,
            )
        )
        if any(value < 0.0 for value in tolerances):
            raise ValueError("Coupled multiphysics tolerances must be nonnegative.")
        self.graph = graph
        self.thermal = thermal
        self.anti_trapping = anti_trapping
        self.nucleation = nucleation
        self.mechanics = mechanics
        self.flow = flow
        self.electrostatic = electrostatic
        self.electrochemical = electrochemical
        (
            self.energy_tolerance,
            self.exchange_tolerance,
            self.conservation_tolerance,
            self.entropy_tolerance,
        ) = tolerances
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coupled-phase-field-multiphysics-plan",
                "graph": graph.graph_id,
                "thermal": thermal.plan_id,
                "anti_trapping": anti_trapping.plan_id,
                "nucleation": nucleation.plan_id,
                "mechanics": mechanics.model_id,
                "flow": flow.plan_id,
                "electrostatic": electrostatic.plan_id,
                "electrochemical": electrochemical.plan_id,
                "tolerances": tolerances,
            }
        )

    def initialize(
        self,
        thermal: NonisothermalSolidificationState,
        /,
        *,
        mechanical_energy: ArrayLike,
        kinetic_energy: ArrayLike,
        electrostatic_energy: ArrayLike,
        component_inventory: ArrayLike,
        thermal_reservoir: ArrayLike,
    ) -> CoupledMultiphysicsState:
        if not isinstance(thermal, NonisothermalSolidificationState):
            raise TypeError("thermal must be NonisothermalSolidificationState.")
        scalars = tuple(
            jnp.asarray(value, dtype=thermal.temperature.dtype)
            for value in (
                mechanical_energy,
                kinetic_energy,
                electrostatic_energy,
                component_inventory,
                thermal_reservoir,
            )
        )
        if any(value.shape != () for value in scalars):
            raise ValueError("Coupled initial inventories must be scalar.")
        return CoupledMultiphysicsState(
            thermal,
            self.nucleation.initialize(),
            scalars[0],
            scalars[1],
            scalars[2],
            scalars[3],
            scalars[4],
            jnp.asarray(0, dtype=jnp.int32),
        )

    def step(
        self,
        state: CoupledMultiphysicsState,
        inputs: CoupledMultiphysicsStepInputs,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> CoupledMultiphysicsStepResult:
        if not isinstance(state, CoupledMultiphysicsState) or not isinstance(
            inputs, CoupledMultiphysicsStepInputs
        ):
            raise TypeError("Coupled step needs compatible state and inputs.")
        time_ = jnp.asarray(time, dtype=state.thermal.temperature.dtype)
        step = jnp.asarray(step_size, dtype=time_.dtype)
        candidate_clock, proposal = self.nucleation.propose(
            state.nucleation,
            time_,
            step,
            inputs.nucleation_driving_force,
            inputs.nucleation_temperature,
        )
        transaction = self.nucleation.transact(
            state.nucleation,
            candidate_clock,
            proposal,
            available_component=state.component_inventory,
            available_energy=state.thermal_reservoir,
        )
        thermal_state, thermal_evidence = self.thermal.step(
            state.thermal,
            inputs.phase_logits,
            inputs.chemical_potential,
            heat_input=inputs.heat_input - transaction.energy_used,
            entropy_flux=inputs.entropy_flux,
            entropy_production=inputs.entropy_production,
        )
        anti_trapping = self.anti_trapping.evaluate(
            inputs.phase_rate,
            inputs.phase_gradient,
            inputs.scalar_chemical_potential,
        )
        mechanics = self.mechanics.evaluate(
            inputs.phase_logits, inputs.displacement_gradient
        )
        phase_weights = jax.nn.softmax(inputs.phase_logits, axis=-1)
        flow = self.flow.evaluate(
            phase_weights,
            inputs.phase_value,
            inputs.phase_gradient,
            inputs.scalar_chemical_potential,
            inputs.scalar_chemical_gradient,
            inputs.velocity,
            inputs.velocity_gradient,
        )
        electrostatic = self.electrostatic.evaluate(
            inputs.phase_logits,
            inputs.potential_gradient,
            free_charge=inputs.free_charge,
            displacement_divergence=inputs.displacement_divergence,
        )
        electrochemical = self.electrochemical.evaluate(
            inputs.concentrations,
            inputs.component_chemical_potentials,
            inputs.component_chemical_gradients,
            inputs.electric_potential,
            electrostatic.electric_field,
            fixed_charge=inputs.free_charge,
        )
        thermal_before = jnp.sum(state.thermal.internal_energy)
        thermal_after = jnp.sum(thermal_state.internal_energy)
        elastic_after = jnp.sum(mechanics.energy)
        kinetic_after = jnp.sum(flow.kinetic_energy)
        electric_after = jnp.sum(electrostatic.field_energy)
        nucleation_energy = transaction.energy_used
        energy_channels = (
            EnergyChannelBalance(
                "thermochemical",
                thermal_before,
                thermal_after,
                external_work=jnp.sum(inputs.heat_input),
            ),
            EnergyChannelBalance(
                "elastic",
                state.mechanical_energy,
                elastic_after,
                external_work=jnp.asarray(inputs.mechanical_work),
            ),
            EnergyChannelBalance(
                "kinetic",
                state.kinetic_energy,
                kinetic_after,
                dissipation=step * jnp.sum(flow.viscous_dissipation),
                external_work=jnp.asarray(inputs.flow_work),
            ),
            EnergyChannelBalance(
                "electrostatic",
                state.electrostatic_energy,
                electric_after,
                dissipation=step * jnp.sum(electrochemical.electrical_dissipation),
                external_work=jnp.asarray(inputs.electrical_work),
            ),
            EnergyChannelBalance(
                "nucleation-interface",
                jnp.asarray(0.0, dtype=time_.dtype),
                nucleation_energy,
            ),
        )
        exchanges = (
            InternalEnergyExchange(
                "capillary",
                "thermochemical",
                "kinetic",
                step * jnp.sum(flow.phase_advection_power),
                step * jnp.sum(flow.flow_capillary_power),
            ),
            InternalEnergyExchange(
                "joule-heating",
                "electrostatic",
                "thermochemical",
                -step * jnp.sum(electrochemical.joule_heating),
                step * jnp.sum(electrochemical.joule_heating),
            ),
            InternalEnergyExchange(
                "nucleation-energy",
                "thermochemical",
                "nucleation-interface",
                -nucleation_energy,
                nucleation_energy,
            ),
        )
        component_before = jnp.sum(state.thermal.composition)
        component_after = jnp.sum(thermal_state.composition)
        conservation = (
            ConservationChannelBalance(
                "material-component",
                component_before,
                component_after,
                event_source=jnp.asarray(0.0, dtype=time_.dtype),
            ),
            ConservationChannelBalance(
                "electric-charge",
                jnp.sum(electrochemical.charge_density),
                jnp.sum(electrochemical.charge_density),
            ),
        )
        ledger = CoupledPhaseFieldLedger(
            energy_channels,
            exchanges,
            conservation,
            thermal_evidence.entropy,
            energy_tolerance=self.energy_tolerance,
            exchange_tolerance=self.exchange_tolerance,
            conservation_tolerance=self.conservation_tolerance,
            entropy_tolerance=self.entropy_tolerance,
            ledger_id=self.plan_id,
        )
        candidate = CoupledMultiphysicsState(
            thermal_state,
            transaction.candidate,
            elastic_after,
            kinetic_after,
            electric_after,
            state.component_inventory - transaction.component_used,
            state.thermal_reservoir - transaction.energy_used,
            state.accepted_steps + 1,
        )
        finite = tree_allfinite(candidate) & ledger.finite
        successful = (
            finite
            & thermal_evidence.successful
            & anti_trapping.successful
            & mechanics.successful
            & flow.successful
            & electrostatic.successful
            & electrochemical.successful
            & transaction.successful
            & ledger.successful
        )
        accepted = tree_where(successful, candidate, state)
        evidence = CoupledMultiphysicsEvidence(
            thermal_evidence,
            anti_trapping,
            mechanics,
            flow,
            electrostatic,
            electrochemical,
            proposal,
            transaction,
            ledger,
            finite,
            successful,
        )
        return CoupledMultiphysicsStepResult(
            candidate,
            accepted,
            successful,
            evidence,
        )


__all__ = [
    "CoupledMultiphysicsEvidence",
    "CoupledMultiphysicsPlan",
    "CoupledMultiphysicsState",
    "CoupledMultiphysicsStepInputs",
    "CoupledMultiphysicsStepResult",
]
