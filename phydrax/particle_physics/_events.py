#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import ENERGY, LENGTH, TIME, UnitDefinition
from ._identity import ParticleCatalogueReference, ParticleRole
from ._weights import EventWeightSet


class ParticleEventStatus(IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    DUPLICATE_EVENT_ID = 2
    INVALID_RELATION = 3
    CAPACITY_OVERFLOW = 4
    INVALID_WEIGHT = 5


class ParticleEventPlan(StrictModule, NonTrainableState):
    """Static event, truth-particle, and vertex capacities plus exact conventions."""

    catalogue: ParticleCatalogueReference
    momentum_unit: UnitDefinition
    length_unit: UnitDefinition
    time_unit: UnitDefinition
    event_capacity: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    vertex_capacity: int = eqx.field(static=True)
    provider_status_namespace: str = eqx.field(static=True)
    momentum_order: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        catalogue: ParticleCatalogueReference,
        momentum_unit: UnitDefinition,
        length_unit: UnitDefinition,
        time_unit: UnitDefinition,
        event_capacity: int,
        particle_capacity: int,
        vertex_capacity: int,
        provider_status_namespace: str,
        momentum_order: str = "E,px,py,pz",
    ):
        if not isinstance(catalogue, ParticleCatalogueReference):
            raise TypeError("catalogue must be ParticleCatalogueReference.")
        if (
            not isinstance(momentum_unit, UnitDefinition)
            or momentum_unit.dimension != ENERGY
        ):
            raise ValueError(
                "momentum_unit must have energy dimension under natural units."
            )
        if not isinstance(length_unit, UnitDefinition) or length_unit.dimension != LENGTH:
            raise ValueError("length_unit must have length dimension.")
        if not isinstance(time_unit, UnitDefinition) or time_unit.dimension != TIME:
            raise ValueError("time_unit must have time dimension.")
        capacities = tuple(map(int, (event_capacity, particle_capacity, vertex_capacity)))
        if any(value < 1 for value in capacities):
            raise ValueError("All event capacities must be positive.")
        namespace = str(provider_status_namespace).strip()
        if not namespace:
            raise ValueError("provider_status_namespace must be non-empty.")
        if momentum_order != "E,px,py,pz":
            raise ValueError("Only the explicit E,px,py,pz momentum order is supported.")
        self.catalogue = catalogue
        self.momentum_unit = momentum_unit
        self.length_unit = length_unit
        self.time_unit = time_unit
        self.event_capacity, self.particle_capacity, self.vertex_capacity = capacities
        self.provider_status_namespace = namespace
        self.momentum_order = momentum_order
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-event-plan",
                "catalogue": catalogue.catalogue_id,
                "units": [momentum_unit.unit_id, length_unit.unit_id, time_unit.unit_id],
                "capacities": list(capacities),
                "provider_status_namespace": namespace,
                "momentum_order": momentum_order,
            }
        )

    def prepare(self, /) -> PreparedParticleEvents:
        return PreparedParticleEvents(self)


class ParticleEventBatch(StrictModule, NonTrainableState):
    event_ids: Array
    subevent_ids: Array
    event_active: Array
    pdg_ids: Array
    roles: Array
    provider_status: Array
    momenta: Array
    rest_energies: Array
    particle_active: Array
    mother_indices: Array
    color_flow: Array
    production_vertices: Array
    vertex_active: Array
    weights: EventWeightSet
    overflow: Array
    finite: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    provider_status_namespace: str = eqx.field(static=True)
    momentum_unit_symbol: str = eqx.field(static=True)
    length_unit_symbol: str = eqx.field(static=True)
    time_unit_symbol: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.event_active)


class PreparedParticleEvents(StrictModule, NonTrainableState):
    plan: ParticleEventPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ParticleEventPlan, /):
        if not isinstance(plan, ParticleEventPlan):
            raise TypeError("plan must be ParticleEventPlan.")
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-particle-events", "plan": plan.plan_id}
        )

    def admit(
        self,
        *,
        event_ids: ArrayLike,
        subevent_ids: ArrayLike,
        event_active: ArrayLike,
        pdg_ids: ArrayLike,
        roles: ArrayLike,
        provider_status: ArrayLike,
        momenta: ArrayLike,
        rest_energies: ArrayLike,
        particle_active: ArrayLike,
        mother_indices: ArrayLike,
        color_flow: ArrayLike,
        production_vertices: ArrayLike,
        vertex_active: ArrayLike,
        weights: EventWeightSet,
        overflow: ArrayLike | None = None,
        source_id: str,
    ) -> ParticleEventBatch:
        event_capacity = self.plan.event_capacity
        particle_capacity = self.plan.particle_capacity
        vertex_capacity = self.plan.vertex_capacity
        event_ids_ = jnp.asarray(event_ids)
        subevent_ids_ = jnp.asarray(subevent_ids)
        event_active_ = jnp.asarray(event_active, dtype=bool)
        pdg_ids_ = jnp.asarray(pdg_ids, dtype=jnp.int32)
        roles_ = jnp.asarray(roles, dtype=jnp.int32)
        provider_status_ = jnp.asarray(provider_status, dtype=jnp.int32)
        momenta_ = jnp.asarray(momenta)
        rest_energies_ = jnp.asarray(rest_energies, dtype=momenta_.dtype)
        particle_active_ = jnp.asarray(particle_active, dtype=bool)
        mother_indices_ = jnp.asarray(mother_indices, dtype=jnp.int32)
        color_flow_ = jnp.asarray(color_flow, dtype=jnp.int32)
        vertices_ = jnp.asarray(production_vertices, dtype=momenta_.dtype)
        vertex_active_ = jnp.asarray(vertex_active, dtype=bool)
        expected_event = (event_capacity,)
        expected_particle = (event_capacity, particle_capacity)
        if (
            event_ids_.shape != expected_event
            or subevent_ids_.shape != expected_event
            or event_active_.shape != expected_event
        ):
            raise ValueError("Event identity arrays must align with event capacity.")
        if not jnp.issubdtype(event_ids_.dtype, jnp.integer) or not jnp.issubdtype(
            subevent_ids_.dtype, jnp.integer
        ):
            raise TypeError("Event identities must contain integers.")
        if (
            pdg_ids_.shape != expected_particle
            or roles_.shape != expected_particle
            or provider_status_.shape != expected_particle
        ):
            raise ValueError(
                "Particle identity arrays must align with event and particle capacity."
            )
        if (
            momenta_.shape != expected_particle + (4,)
            or rest_energies_.shape != expected_particle
            or particle_active_.shape != expected_particle
        ):
            raise ValueError("Particle state arrays have incompatible shapes.")
        if mother_indices_.shape != expected_particle + (
            2,
        ) or color_flow_.shape != expected_particle + (2,):
            raise ValueError("Particle relation arrays have incompatible shapes.")
        if vertices_.shape != (
            event_capacity,
            vertex_capacity,
            4,
        ) or vertex_active_.shape != (event_capacity, vertex_capacity):
            raise ValueError(
                "Production vertices must align with event and vertex capacity."
            )
        if (
            not isinstance(weights, EventWeightSet)
            or weights.event_capacity != event_capacity
        ):
            raise ValueError(
                "weights must be an EventWeightSet with matching event capacity."
            )
        if weights.event_active.shape != event_active_.shape:
            raise ValueError("Weight activity must align with event activity.")
        overflow_ = (
            jnp.zeros(expected_event, dtype=bool)
            if overflow is None
            else jnp.asarray(overflow, dtype=bool)
        )
        if overflow_.shape != expected_event:
            raise ValueError("overflow must align with event capacity.")
        if not str(source_id).strip():
            raise ValueError("source_id must be non-empty.")
        role_valid = (roles_ >= int(ParticleRole.UNKNOWN)) & (
            roles_ <= int(ParticleRole.OUTGOING)
        )
        mother_valid = (mother_indices_ == -1) | (
            (mother_indices_ >= 0) & (mother_indices_ < particle_capacity)
        )
        relation_valid = jnp.all(
            jnp.where(particle_active_[..., None], mother_valid, True), axis=(1, 2)
        )
        role_event_valid = jnp.all(jnp.where(particle_active_, role_valid, True), axis=1)
        finite = (
            jnp.all(
                jnp.where(particle_active_[..., None], jnp.isfinite(momenta_), True),
                axis=(1, 2),
            )
            & jnp.all(
                jnp.where(
                    particle_active_,
                    jnp.isfinite(rest_energies_) & (rest_energies_ >= 0.0),
                    True,
                ),
                axis=1,
            )
            & jnp.all(
                jnp.where(vertex_active_[..., None], jnp.isfinite(vertices_), True),
                axis=(1, 2),
            )
        )
        same_event = event_ids_[:, None] == event_ids_[None, :]
        same_subevent = subevent_ids_[:, None] == subevent_ids_[None, :]
        off_diagonal = ~jnp.eye(event_capacity, dtype=bool)
        duplicates = (
            jnp.any(
                same_event & same_subevent & off_diagonal & event_active_[None, :], axis=1
            )
            & event_active_
        )
        weight_valid = weights.finite & (weights.event_active == event_active_)
        valid = (
            event_active_
            & finite
            & relation_valid
            & role_event_valid
            & ~duplicates
            & ~overflow_
            & weight_valid
        )
        status = jnp.where(
            overflow_,
            int(ParticleEventStatus.CAPACITY_OVERFLOW),
            jnp.where(
                ~finite,
                int(ParticleEventStatus.NONFINITE),
                jnp.where(
                    duplicates,
                    int(ParticleEventStatus.DUPLICATE_EVENT_ID),
                    jnp.where(
                        ~(relation_valid & role_event_valid),
                        int(ParticleEventStatus.INVALID_RELATION),
                        jnp.where(
                            ~weight_valid,
                            int(ParticleEventStatus.INVALID_WEIGHT),
                            int(ParticleEventStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
        )
        return ParticleEventBatch(
            event_ids_,
            subevent_ids_,
            event_active_,
            pdg_ids_,
            roles_,
            provider_status_,
            momenta_,
            rest_energies_,
            particle_active_,
            mother_indices_,
            color_flow_,
            vertices_,
            vertex_active_,
            weights,
            overflow_,
            finite,
            valid,
            status.astype(jnp.int32),
            self.plan.plan_id,
            str(source_id).strip(),
            self.plan.provider_status_namespace,
            self.plan.momentum_unit.symbol,
            self.plan.length_unit.symbol,
            self.plan.time_unit.symbol,
        )


__all__ = [
    "ParticleEventBatch",
    "ParticleEventPlan",
    "ParticleEventStatus",
    "PreparedParticleEvents",
]
