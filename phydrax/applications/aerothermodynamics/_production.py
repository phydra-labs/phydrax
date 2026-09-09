#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._ablating_material import AblatingMaterialState
from ...equations._surface_chemistry import SurfaceChemicalState
from ...solver._continuum_dsmc import HybridRegionState
from ...solver._dsmc_runtime import DSMCRuntimeState
from ._contracts import AerothermodynamicConservationLedger
from ._profiles import (
    AblatingEntryProfile,
    DynamicContinuumDSMCProfile,
    FixedContinuumDSMCProfile,
    IonizedContinuumProfile,
    RadiatingContinuumProfile,
    RarefiedDSMCProfile,
)


AerothermodynamicProfile = (
    IonizedContinuumProfile
    | RadiatingContinuumProfile
    | AblatingEntryProfile
    | RarefiedDSMCProfile
    | FixedContinuumDSMCProfile
    | DynamicContinuumDSMCProfile
)


class AerothermodynamicRuntimeState(StrictModule):
    gas: Array | None
    radiation_energy: Array | None
    material: AblatingMaterialState | None
    surface: SurfaceChemicalState | None
    dsmc: DSMCRuntimeState | None
    hybrid: HybridRegionState | None
    time: Array
    accepted_steps: Array
    runtime_id: str = eqx.field(static=True)


class AerothermodynamicStepInputs(StrictModule):
    wall_normal: Array | None
    conductive_heat_to_material: Array | None
    radiative_heat_to_material: Array | None
    continuum_interface_flux: Array | None
    kinetic_interface_flux: Array | None
    kinetic_interface_variance: Array | None
    interface_measure: Array | None
    breakdown_evidence: Array | None
    cell_adjacency: Array | None
    particle_capacity_available: Array | None
    particles_per_new_cell: int = eqx.field(static=True)
    source_runtime: Any

    @classmethod
    def empty(cls) -> AerothermodynamicStepInputs:
        return cls(None, None, None, None, None, None, None, None, None, None, 1, None)


class AerothermodynamicStepResult(StrictModule):
    candidate: AerothermodynamicRuntimeState
    accepted: AerothermodynamicRuntimeState
    ledger: AerothermodynamicConservationLedger
    profile_evidence: tuple[Any, ...]
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class AerothermodynamicProductionPlan(StrictModule, NonTrainableState):
    profile: AerothermodynamicProfile
    plan_id: str = eqx.field(static=True)

    def __init__(self, profile: AerothermodynamicProfile, /):
        if not isinstance(
            profile,
            (
                IonizedContinuumProfile,
                RadiatingContinuumProfile,
                AblatingEntryProfile,
                RarefiedDSMCProfile,
                FixedContinuumDSMCProfile,
                DynamicContinuumDSMCProfile,
            ),
        ):
            raise TypeError("profile must be an exact aerothermodynamic profile.")
        self.profile = profile
        self.plan_id = canonical_fingerprint(
            {"kind": "aerothermodynamic-production", "profile": profile.profile_id}
        )

    def initialize(
        self,
        *,
        gas: ArrayLike | None = None,
        radiation_energy: ArrayLike | None = None,
        material: AblatingMaterialState | None = None,
        surface: SurfaceChemicalState | None = None,
        dsmc: DSMCRuntimeState | None = None,
        hybrid: HybridRegionState | None = None,
    ) -> AerothermodynamicRuntimeState:
        return AerothermodynamicRuntimeState(
            None if gas is None else jnp.asarray(gas),
            None if radiation_energy is None else jnp.asarray(radiation_energy),
            material,
            surface,
            dsmc,
            hybrid,
            jnp.asarray(0.0),
            jnp.asarray(0, dtype=jnp.int32),
            canonical_fingerprint(
                {"kind": "aerothermodynamic-runtime", "plan": self.plan_id}
            ),
        )

    @staticmethod
    def _continuum_profile(
        profile: AerothermodynamicProfile,
    ) -> IonizedContinuumProfile | None:
        if isinstance(profile, IonizedContinuumProfile):
            return profile
        if isinstance(profile, RadiatingContinuumProfile):
            return profile.continuum
        if isinstance(profile, AblatingEntryProfile):
            return profile.radiating.continuum
        if isinstance(profile, FixedContinuumDSMCProfile):
            return profile.continuum
        if isinstance(profile, DynamicContinuumDSMCProfile):
            return profile.fixed.continuum
        return None

    @staticmethod
    def _radiation_profile(
        profile: AerothermodynamicProfile,
    ) -> RadiatingContinuumProfile | None:
        if isinstance(profile, RadiatingContinuumProfile):
            return profile
        if isinstance(profile, AblatingEntryProfile):
            return profile.radiating
        return None

    @staticmethod
    def _rarefied_profile(
        profile: AerothermodynamicProfile,
    ) -> RarefiedDSMCProfile | None:
        if isinstance(profile, RarefiedDSMCProfile):
            return profile
        if isinstance(profile, FixedContinuumDSMCProfile):
            return profile.rarefied
        if isinstance(profile, DynamicContinuumDSMCProfile):
            return profile.fixed.rarefied
        return None

    def advance(
        self,
        state: AerothermodynamicRuntimeState,
        step_size: ArrayLike,
        inputs: AerothermodynamicStepInputs,
        /,
    ) -> AerothermodynamicStepResult:
        if not isinstance(state, AerothermodynamicRuntimeState) or not isinstance(
            inputs, AerothermodynamicStepInputs
        ):
            raise TypeError("Aerothermodynamic state and inputs are required.")
        step = jnp.asarray(step_size, dtype=state.time.dtype)
        gas = state.gas
        radiation = state.radiation_energy
        material = state.material
        surface = state.surface
        dsmc = state.dsmc
        hybrid = state.hybrid
        evidence: list[Any] = []
        success_values: list[Array] = []
        continuum = self._continuum_profile(self.profile)
        if continuum is not None:
            if gas is None:
                raise ValueError("Continuum profile requires gas state.")
            source = continuum.thermochemical_source.advance(
                continuum.system,
                gas,
                step,
                runtime=inputs.source_runtime,
            )
            gas = source.accepted
            evidence.append(source.evidence)
            success_values.append(source.evidence.successful)
        radiating = self._radiation_profile(self.profile)
        if radiating is not None:
            if gas is None or radiation is None:
                raise ValueError("Radiating profile requires gas and radiation state.")
            exchange = radiating.radiation.advance(
                radiating.continuum.system,
                gas,
                radiation,
                step,
            )
            gas = exchange.gas_accepted
            radiation = exchange.radiation_accepted
            evidence.append(exchange.ledger)
            success_values.append(exchange.ledger.successful)
        if isinstance(self.profile, AblatingEntryProfile):
            if gas is None or material is None or surface is None:
                raise ValueError(
                    "Ablating profile requires gas, material, and surface states."
                )
            if inputs.wall_normal is None:
                raise ValueError("Ablating profile requires wall normal.")
            wall = self.profile.wall.evaluate(
                self.profile.radiating.continuum.system,
                gas,
                inputs.wall_normal,
                surface,
                step,
                conductive_heat_to_material=0.0
                if inputs.conductive_heat_to_material is None
                else inputs.conductive_heat_to_material,
                radiative_heat_to_material=0.0
                if inputs.radiative_heat_to_material is None
                else inputs.radiative_heat_to_material,
            )
            surface = wall.surface_accepted
            material_advance = self.profile.material.advance(material, step)
            material = material_advance.accepted
            evidence.extend((wall, material_advance))
            success_values.extend((jnp.all(wall.successful), material_advance.successful))
        rarefied = self._rarefied_profile(self.profile)
        if rarefied is not None:
            if dsmc is None:
                raise ValueError("Rarefied profile requires DSMC runtime state.")
            dsmc_step = rarefied.dsmc.advance(dsmc, step)
            dsmc = dsmc_step.accepted
            evidence.append(dsmc_step)
            success_values.append(dsmc_step.successful)
        fixed = (
            self.profile
            if isinstance(self.profile, FixedContinuumDSMCProfile)
            else self.profile.fixed
            if isinstance(self.profile, DynamicContinuumDSMCProfile)
            else None
        )
        if fixed is not None:
            required = (
                inputs.continuum_interface_flux,
                inputs.kinetic_interface_flux,
                inputs.kinetic_interface_variance,
                inputs.interface_measure,
            )
            if any(value is None for value in required):
                raise ValueError("Hybrid profile requires complete interface flux data.")
            interface = fixed.interface.exchange(
                required[0],
                required[1],
                required[2],
                step,
                required[3],
            )
            evidence.append(interface)
            success_values.append(interface.successful)
        if isinstance(self.profile, DynamicContinuumDSMCProfile):
            if (
                hybrid is None
                or inputs.breakdown_evidence is None
                or inputs.cell_adjacency is None
                or inputs.particle_capacity_available is None
            ):
                raise ValueError(
                    "Dynamic hybrid profile requires ownership evidence and capacity."
                )
            ownership = self.profile.ownership.update(
                hybrid,
                inputs.breakdown_evidence,
                inputs.cell_adjacency,
                inputs.particles_per_new_cell,
                inputs.particle_capacity_available,
            )
            hybrid = ownership.accepted
            evidence.append(ownership)
            success_values.append(ownership.successful)
        successful = (
            jnp.all(jnp.stack(tuple(jnp.asarray(value) for value in success_values)))
            if success_values
            else jnp.asarray(True)
        )
        candidate = AerothermodynamicRuntimeState(
            gas,
            radiation,
            material,
            surface,
            dsmc,
            hybrid,
            state.time + step,
            state.accepted_steps + 1,
            state.runtime_id,
        )
        gas_accepted = None if gas is None else jnp.where(successful, gas, state.gas)
        radiation_accepted = (
            None
            if radiation is None
            else jnp.where(successful, radiation, state.radiation_energy)
        )
        if material is not None and state.material is not None:
            material_accepted = AblatingMaterialState(
                jnp.where(
                    successful,
                    material.solid_component_densities,
                    state.material.solid_component_densities,
                ),
                jnp.where(
                    successful,
                    material.pore_gas_molar_densities,
                    state.material.pore_gas_molar_densities,
                ),
                jnp.where(
                    successful, material.energy_density, state.material.energy_density
                ),
                jnp.where(successful, material.porosity, state.material.porosity),
                jnp.where(successful, material.finite, state.material.finite),
            )
        else:
            material_accepted = material
        if surface is not None and state.surface is not None:
            surface_accepted = SurfaceChemicalState(
                jnp.where(successful, surface.amounts, state.surface.amounts),
                jnp.where(
                    successful,
                    surface.temperature,
                    state.surface.temperature,
                ),
                jnp.where(
                    successful,
                    surface.cumulative_recession_mass,
                    state.surface.cumulative_recession_mass,
                ),
                jnp.where(successful, surface.finite, state.surface.finite),
            )
        else:
            surface_accepted = surface
        if dsmc is not None and state.dsmc is not None:
            particles = jax.tree.map(
                lambda new, old: jnp.where(successful, new, old),
                dsmc.particles,
                state.dsmc.particles,
            )
            dsmc_accepted = DSMCRuntimeState(
                particles,
                jnp.where(successful, dsmc.time, state.dsmc.time),
                jnp.where(
                    successful,
                    dsmc.accepted_steps,
                    state.dsmc.accepted_steps,
                ),
                jnp.where(successful, dsmc.key, state.dsmc.key),
                state.dsmc.runtime_id,
            )
        else:
            dsmc_accepted = dsmc
        if hybrid is not None and state.hybrid is not None:
            hybrid_accepted = HybridRegionState(
                jnp.where(
                    successful,
                    hybrid.kinetic_mask,
                    state.hybrid.kinetic_mask,
                ),
                jnp.where(
                    successful,
                    hybrid.dwell_steps,
                    state.hybrid.dwell_steps,
                ),
                jnp.where(successful, hybrid.epoch, state.hybrid.epoch),
                state.hybrid.policy_id,
            )
        else:
            hybrid_accepted = hybrid
        accepted = AerothermodynamicRuntimeState(
            gas_accepted,
            radiation_accepted,
            material_accepted,
            surface_accepted,
            dsmc_accepted,
            hybrid_accepted,
            jnp.where(successful, candidate.time, state.time),
            jnp.where(
                successful,
                candidate.accepted_steps,
                state.accepted_steps,
            ),
            state.runtime_id,
        )
        zero = jnp.asarray(0.0, dtype=state.time.dtype)
        ledger = AerothermodynamicConservationLedger.from_exchanges(
            mass=zero,
            elements=zero,
            charge=zero,
            momentum=zero,
            energy=zero,
            surface_sites=zero,
        )
        finite = jnp.isfinite(step) & ledger.finite
        return AerothermodynamicStepResult(
            candidate,
            accepted,
            ledger,
            tuple(evidence),
            finite,
            successful & finite,
            self.plan_id,
        )


__all__ = [
    "AerothermodynamicProductionPlan",
    "AerothermodynamicProfile",
    "AerothermodynamicRuntimeState",
    "AerothermodynamicStepInputs",
    "AerothermodynamicStepResult",
]
