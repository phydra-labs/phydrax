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
from ...particle_physics import HEPProviderBinding
from ..lattice_field import FiniteDensityEOSTable, QCDTransportTable


class NuclearCollisionBatch(StrictModule, NonTrainableState):
    event_ids: Array
    impact_parameters: Array
    participant_counts: Array
    binary_collision_counts: Array
    eccentricities: Array
    active: Array
    valid: Array
    geometry_provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        event_ids: ArrayLike,
        impact_parameters: ArrayLike,
        participant_counts: ArrayLike,
        binary_collision_counts: ArrayLike,
        eccentricities: ArrayLike,
        active: ArrayLike,
        /,
        *,
        geometry_provider_id: str,
    ):
        event_ids_ = jnp.asarray(event_ids)
        impact = jnp.asarray(impact_parameters)
        participants = jnp.asarray(participant_counts, dtype=jnp.int32)
        collisions = jnp.asarray(binary_collision_counts, dtype=jnp.int32)
        eccentricities_ = jnp.asarray(eccentricities, dtype=impact.dtype)
        active_ = jnp.asarray(active, dtype=bool)
        if (
            event_ids_.ndim != 1
            or impact.shape != event_ids_.shape
            or participants.shape != event_ids_.shape
            or collisions.shape != event_ids_.shape
            or active_.shape != event_ids_.shape
            or eccentricities_.ndim != 2
            or eccentricities_.shape[0] != event_ids_.shape[0]
        ):
            raise ValueError("Nuclear collision fields must align with event support.")
        provider = str(geometry_provider_id).strip()
        if not provider:
            raise ValueError("geometry_provider_id is required.")
        valid = (
            jnp.isfinite(impact)
            & (impact >= 0.0)
            & (participants >= 0)
            & (collisions >= 0)
            & jnp.all(jnp.isfinite(eccentricities_), axis=-1)
        )
        self.event_ids = event_ids_
        self.impact_parameters = impact
        self.participant_counts = participants
        self.binary_collision_counts = collisions
        self.eccentricities = eccentricities_
        self.active = active_
        self.valid = jnp.where(active_, valid, True)
        self.geometry_provider_id = provider


class HeavyIonChainPlan(StrictModule, NonTrainableState):
    initial_state_provider: HEPProviderBinding
    pre_equilibrium_provider: HEPProviderBinding
    hydrodynamics_provider: HEPProviderBinding
    particlization_provider: HEPProviderBinding
    afterburner_provider: HEPProviderBinding
    equation_of_state: FiniteDensityEOSTable
    transport: QCDTransportTable
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_state_provider: HEPProviderBinding,
        pre_equilibrium_provider: HEPProviderBinding,
        hydrodynamics_provider: HEPProviderBinding,
        particlization_provider: HEPProviderBinding,
        afterburner_provider: HEPProviderBinding,
        equation_of_state: FiniteDensityEOSTable,
        transport: QCDTransportTable,
    ):
        providers = (
            initial_state_provider,
            pre_equilibrium_provider,
            hydrodynamics_provider,
            particlization_provider,
            afterburner_provider,
        )
        capabilities = (
            "hep.heavy-ion.initial-state",
            "hep.heavy-ion.pre-equilibrium",
            "hep.heavy-ion.hydrodynamics",
            "hep.heavy-ion.particlization",
            "hep.heavy-ion.afterburner",
        )
        if any(not isinstance(value, HEPProviderBinding) for value in providers):
            raise TypeError("Heavy-ion providers must use HEPProviderBinding.")
        if any(
            not provider.supports(capability)
            for provider, capability in zip(providers, capabilities, strict=True)
        ):
            raise ValueError("A heavy-ion provider lacks its required capability.")
        if not isinstance(equation_of_state, FiniteDensityEOSTable) or not isinstance(
            transport, QCDTransportTable
        ):
            raise TypeError(
                "Heavy-ion EoS and transport tables must be typed QCD products."
            )
        (
            self.initial_state_provider,
            self.pre_equilibrium_provider,
            self.hydrodynamics_provider,
            self.particlization_provider,
            self.afterburner_provider,
        ) = providers
        self.equation_of_state = equation_of_state
        self.transport = transport
        self.plan_id = canonical_fingerprint(
            {
                "kind": "heavy-ion-chain-plan",
                "providers": [value.binding_id for value in providers],
                "equation_of_state": equation_of_state.table_id,
                "transport": transport.table_id,
            }
        )


class HydrodynamicConservationEvidence(StrictModule, NonTrainableState):
    energy_residual: Array
    momentum_residual: Array
    charge_residual: Array
    maximum_relative_residual: Array
    valid: Array
    evidence_id: str = eqx.field(static=True)


def audit_hydrodynamic_conservation(
    initial_energy_momentum: ArrayLike,
    final_energy_momentum: ArrayLike,
    initial_charges: ArrayLike,
    final_charges: ArrayLike,
    /,
    *,
    maximum_relative_residual: float,
    source_id: str,
) -> HydrodynamicConservationEvidence:
    initial = jnp.asarray(initial_energy_momentum)
    final = jnp.asarray(final_energy_momentum, dtype=initial.dtype)
    initial_charges_ = jnp.asarray(initial_charges, dtype=initial.dtype)
    final_charges_ = jnp.asarray(final_charges, dtype=initial.dtype)
    if (
        initial.shape != final.shape
        or initial.shape[-1:] != (4,)
        or initial_charges_.shape != final_charges_.shape
        or initial_charges_.shape[-1:] != (3,)
    ):
        raise ValueError(
            "Hydrodynamic energy-momentum and B/Q/S charge supports are invalid."
        )
    energy_residual = final[..., 0] - initial[..., 0]
    momentum_residual = final[..., 1:] - initial[..., 1:]
    charge_residual = final_charges_ - initial_charges_
    scale = jnp.maximum(jnp.abs(initial[..., 0]), jnp.finfo(initial.dtype).tiny)
    maximum = jnp.maximum(
        jnp.abs(energy_residual) / scale,
        jnp.max(jnp.abs(momentum_residual), axis=-1) / scale,
    )
    maximum = jnp.maximum(
        maximum,
        jnp.max(jnp.abs(charge_residual), axis=-1)
        / jnp.maximum(jnp.max(jnp.abs(initial_charges_), axis=-1), 1.0),
    )
    finite = jnp.isfinite(maximum)
    valid = finite & (maximum <= float(maximum_relative_residual))
    evidence_id = canonical_fingerprint(
        {
            "kind": "hydrodynamic-conservation-evidence",
            "source": str(source_id),
            "maximum_relative_residual": float(maximum_relative_residual),
            "shape": list(initial.shape),
        }
    )
    return HydrodynamicConservationEvidence(
        energy_residual, momentum_residual, charge_residual, maximum, valid, evidence_id
    )


__all__ = [
    "HeavyIonChainPlan",
    "HydrodynamicConservationEvidence",
    "NuclearCollisionBatch",
    "audit_hydrodynamic_conservation",
]
