#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.discrete_velocity import (
    CompressibleKineticPopulationState,
    CompressibleKineticRuntimePlan,
    CompressibleKineticRuntimeResult,
    CompressibleKineticRuntimeState,
    KineticAuxiliaryState,
    KineticRadiationAblationEvidence,
    KineticRadiationAblationPlan,
    KineticSpeciesTransportEvidence,
    KineticSpeciesTransportPlan,
)


class KineticAerothermodynamicState(StrictModule):
    gas: CompressibleKineticRuntimeState
    auxiliary: KineticAuxiliaryState
    geometry_epoch: Array


class KineticAerothermodynamicEvidence(StrictModule):
    kinetic: CompressibleKineticRuntimeResult
    species: KineticSpeciesTransportEvidence | None
    radiation_ablation: KineticRadiationAblationEvidence | None
    successful: Array
    plan_id: str = eqx.field(static=True)


class KineticAerothermodynamicResult(StrictModule):
    candidate: KineticAerothermodynamicState
    accepted: KineticAerothermodynamicState
    evidence: KineticAerothermodynamicEvidence
    successful: Array


class KineticAerothermodynamicPlan(StrictModule, NonTrainableState):
    runtime: CompressibleKineticRuntimePlan
    species_transport: KineticSpeciesTransportPlan | None
    radiation_ablation: KineticRadiationAblationPlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: CompressibleKineticRuntimePlan,
        /,
        *,
        species_transport: KineticSpeciesTransportPlan | None = None,
        radiation_ablation: KineticRadiationAblationPlan | None = None,
    ):
        if not isinstance(runtime, CompressibleKineticRuntimePlan):
            raise TypeError("runtime must be CompressibleKineticRuntimePlan.")
        if species_transport is not None and not isinstance(
            species_transport, KineticSpeciesTransportPlan
        ):
            raise TypeError("species_transport must be KineticSpeciesTransportPlan.")
        if radiation_ablation is not None and not isinstance(
            radiation_ablation, KineticRadiationAblationPlan
        ):
            raise TypeError("radiation_ablation must be KineticRadiationAblationPlan.")
        self.runtime = runtime
        self.species_transport = species_transport
        self.radiation_ablation = radiation_ablation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kinetic-aerothermodynamic-plan",
                "runtime": runtime.runtime_id,
                "species_transport": None
                if species_transport is None
                else species_transport.plan_id,
                "radiation_ablation": None
                if radiation_ablation is None
                else radiation_ablation.plan_id,
            }
        )

    def advance(
        self,
        state: KineticAerothermodynamicState,
        relaxation_rate: ArrayLike,
        /,
        *,
        radiation_energy_exchange: ArrayLike | None = None,
        ablated_species: ArrayLike | None = None,
        wall_momentum: ArrayLike | None = None,
        wall_energy: ArrayLike | None = None,
        species_charges: ArrayLike | None = None,
        geometry_epoch_increment: ArrayLike = 0,
    ) -> KineticAerothermodynamicResult:
        kinetic = self.runtime.advance(state.gas, relaxation_rate)
        auxiliary = state.auxiliary
        species_evidence = None
        if self.species_transport is not None:
            transported, species_evidence = self.species_transport.advect(
                kinetic.accepted.kinetic, auxiliary.species_densities
            )
            auxiliary = KineticAuxiliaryState(
                transported,
                auxiliary.mode_energies,
                auxiliary.electron_energy,
                auxiliary.turbulence_variables,
                auxiliary.radiation_energy,
            )
        radiation_evidence = None
        if self.radiation_ablation is not None:
            values = (
                radiation_energy_exchange,
                ablated_species,
                wall_momentum,
                wall_energy,
                species_charges,
            )
            if any(value is None for value in values):
                raise ValueError(
                    "Radiation/ablation coupling requires every exchange input."
                )
            auxiliary, radiation_evidence = self.radiation_ablation.exchange(
                auxiliary,
                radiation_energy_exchange,
                ablated_species,
                wall_momentum,
                wall_energy,
                species_charges=species_charges,
            )
        successful = kinetic.successful
        if species_evidence is not None:
            successful &= jnp.all(species_evidence.successful)
        if radiation_evidence is not None:
            successful &= jnp.all(radiation_evidence.successful)
        candidate = KineticAerothermodynamicState(
            kinetic.accepted,
            auxiliary,
            state.geometry_epoch + jnp.asarray(geometry_epoch_increment),
        )
        accepted_kinetic = CompressibleKineticPopulationState(
            tuple(
                jnp.where(successful, new, old)
                for new, old in zip(
                    candidate.gas.kinetic.populations,
                    state.gas.kinetic.populations,
                    strict=True,
                )
            ),
            jnp.where(
                successful,
                candidate.gas.kinetic.equilibrium_dual,
                state.gas.kinetic.equilibrium_dual,
            ),
            jnp.where(
                successful,
                candidate.gas.kinetic.stabilizer,
                state.gas.kinetic.stabilizer,
            ),
            jnp.where(
                successful,
                candidate.gas.kinetic.frame_velocity,
                state.gas.kinetic.frame_velocity,
            ),
            jnp.where(
                successful,
                candidate.gas.kinetic.frame_temperature_scale,
                state.gas.kinetic.frame_temperature_scale,
            ),
            state.gas.kinetic.layout,
        )
        accepted_auxiliary = KineticAuxiliaryState(
            jnp.where(
                successful[..., None],
                candidate.auxiliary.species_densities,
                state.auxiliary.species_densities,
            ),
            jnp.where(
                successful[..., None],
                candidate.auxiliary.mode_energies,
                state.auxiliary.mode_energies,
            ),
            jnp.where(
                successful,
                candidate.auxiliary.electron_energy,
                state.auxiliary.electron_energy,
            ),
            jnp.where(
                successful[..., None],
                candidate.auxiliary.turbulence_variables,
                state.auxiliary.turbulence_variables,
            ),
            jnp.where(
                successful[..., None],
                candidate.auxiliary.radiation_energy,
                state.auxiliary.radiation_energy,
            ),
        )
        accepted = KineticAerothermodynamicState(
            CompressibleKineticRuntimeState(
                accepted_kinetic,
                jnp.where(successful, candidate.gas.time, state.gas.time),
                jnp.where(successful, candidate.gas.step_index, state.gas.step_index),
                jnp.where(successful, candidate.gas.parity, state.gas.parity),
            ),
            accepted_auxiliary,
            jnp.where(successful, candidate.geometry_epoch, state.geometry_epoch),
        )
        evidence = KineticAerothermodynamicEvidence(
            kinetic=kinetic,
            species=species_evidence,
            radiation_ablation=radiation_evidence,
            successful=successful,
            plan_id=self.plan_id,
        )
        return KineticAerothermodynamicResult(candidate, accepted, evidence, successful)


__all__ = [
    "KineticAerothermodynamicEvidence",
    "KineticAerothermodynamicPlan",
    "KineticAerothermodynamicResult",
    "KineticAerothermodynamicState",
]
