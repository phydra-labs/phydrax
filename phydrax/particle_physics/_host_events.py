#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._events import ParticleEventBatch, ParticleEventPlan, PreparedParticleEvents
from ._identity import ParticleRole
from ._weights import EventWeightSet, WeightVariationKind


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _integer(
    value: object,
    name: str,
    /,
    *,
    minimum: int | None = None,
    bits: int = 64,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    info = np.iinfo(np.int32 if bits == 32 else np.int64)
    if result < info.min or result > info.max:
        raise OverflowError(f"{name} must fit signed int{bits}.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


@dataclass(frozen=True, slots=True)
class HostAttribute:
    """Opaque namespaced bytes retained by the authoritative host record."""

    namespace: str
    name: str
    content_type: str
    payload: bytes

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "namespace", _identifier(self.namespace, "Attribute namespace")
        )
        object.__setattr__(self, "name", _identifier(self.name, "Attribute name"))
        object.__setattr__(
            self, "content_type", _identifier(self.content_type, "Content type")
        )
        if not isinstance(self.payload, bytes):
            raise TypeError("Attribute payload must be bytes.")


@dataclass(frozen=True, slots=True)
class HostEventWeight:
    name: str
    value: float
    variation_kind: WeightVariationKind
    correlation_group: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "Weight name"))
        object.__setattr__(
            self,
            "correlation_group",
            _identifier(self.correlation_group, "Correlation group"),
        )
        if not isinstance(self.variation_kind, WeightVariationKind):
            raise TypeError("variation_kind must be WeightVariationKind.")
        if not math.isfinite(float(self.value)):
            raise ValueError("Host event weights must be finite.")


@dataclass(frozen=True, slots=True)
class HostParticleRecord:
    particle_id: int
    pdg_id: int
    role: ParticleRole
    provider_status: int
    momentum: tuple[float, float, float, float]
    rest_energy: float
    production_vertex_id: int | None = None
    end_vertex_id: int | None = None
    color_flow: tuple[int, int] = (0, 0)
    attributes: tuple[HostAttribute, ...] = ()

    def __post_init__(self) -> None:
        particle_id = _integer(self.particle_id, "particle_id", minimum=0)
        pdg_id = _integer(self.pdg_id, "pdg_id", bits=32)
        provider_status = _integer(self.provider_status, "provider_status", bits=32)
        if not isinstance(self.role, ParticleRole):
            raise TypeError("role must be ParticleRole.")
        momentum = tuple(float(value) for value in self.momentum)
        if len(momentum) != 4 or any(not math.isfinite(value) for value in momentum):
            raise ValueError("momentum must be a finite E,px,py,pz tuple.")
        rest_energy = float(self.rest_energy)
        if not math.isfinite(rest_energy) or rest_energy < 0.0:
            raise ValueError("rest_energy must be finite and nonnegative.")
        production_vertex = (
            None
            if self.production_vertex_id is None
            else _integer(self.production_vertex_id, "production_vertex_id", minimum=0)
        )
        end_vertex = (
            None
            if self.end_vertex_id is None
            else _integer(self.end_vertex_id, "end_vertex_id", minimum=0)
        )
        color = tuple(
            _integer(value, "color_flow", minimum=0, bits=32) for value in self.color_flow
        )
        if len(color) != 2:
            raise ValueError("color_flow must contain two integers.")
        attributes = tuple(self.attributes)
        if any(not isinstance(value, HostAttribute) for value in attributes):
            raise TypeError("Particle attributes must contain HostAttribute values.")
        attribute_keys = tuple((value.namespace, value.name) for value in attributes)
        if len(set(attribute_keys)) != len(attribute_keys):
            raise ValueError("Particle attribute names must be unique within namespaces.")
        object.__setattr__(self, "particle_id", particle_id)
        object.__setattr__(self, "pdg_id", pdg_id)
        object.__setattr__(self, "provider_status", provider_status)
        object.__setattr__(self, "momentum", momentum)
        object.__setattr__(self, "rest_energy", rest_energy)
        object.__setattr__(self, "production_vertex_id", production_vertex)
        object.__setattr__(self, "end_vertex_id", end_vertex)
        object.__setattr__(self, "color_flow", color)
        object.__setattr__(self, "attributes", attributes)


@dataclass(frozen=True, slots=True)
class HostVertexRecord:
    vertex_id: int
    position: tuple[float, float, float, float]
    incoming_particle_ids: tuple[int, ...] = ()
    outgoing_particle_ids: tuple[int, ...] = ()
    attributes: tuple[HostAttribute, ...] = ()

    def __post_init__(self) -> None:
        vertex_id = _integer(self.vertex_id, "vertex_id", minimum=0)
        position = tuple(float(value) for value in self.position)
        incoming = tuple(
            _integer(value, "incoming_particle_id", minimum=0)
            for value in self.incoming_particle_ids
        )
        outgoing = tuple(
            _integer(value, "outgoing_particle_id", minimum=0)
            for value in self.outgoing_particle_ids
        )
        attributes = tuple(self.attributes)
        if len(position) != 4 or any(not math.isfinite(value) for value in position):
            raise ValueError("Vertex identity and position are invalid.")
        if len(set(incoming)) != len(incoming) or len(set(outgoing)) != len(outgoing):
            raise ValueError("Vertex incoming/outgoing identities must be unique.")
        if set(incoming) & set(outgoing):
            raise ValueError(
                "A particle cannot be both incoming and outgoing at one vertex."
            )
        if any(not isinstance(value, HostAttribute) for value in attributes):
            raise TypeError("Vertex attributes must contain HostAttribute values.")
        object.__setattr__(self, "vertex_id", vertex_id)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "incoming_particle_ids", incoming)
        object.__setattr__(self, "outgoing_particle_ids", outgoing)
        object.__setattr__(self, "attributes", attributes)


@dataclass(frozen=True, slots=True)
class HostEventRecord:
    event_id: int
    subevent_id: int
    particles: tuple[HostParticleRecord, ...]
    vertices: tuple[HostVertexRecord, ...]
    weights: tuple[HostEventWeight, ...]
    provider_status_namespace: str
    source_id: str
    attributes: tuple[HostAttribute, ...] = ()

    def __post_init__(self) -> None:
        particles = tuple(self.particles)
        vertices = tuple(self.vertices)
        weights = tuple(self.weights)
        attributes = tuple(self.attributes)
        if any(not isinstance(value, HostParticleRecord) for value in particles):
            raise TypeError("particles must contain HostParticleRecord values.")
        if any(not isinstance(value, HostVertexRecord) for value in vertices):
            raise TypeError("vertices must contain HostVertexRecord values.")
        if any(not isinstance(value, HostEventWeight) for value in weights):
            raise TypeError("weights must contain HostEventWeight values.")
        if any(not isinstance(value, HostAttribute) for value in attributes):
            raise TypeError("attributes must contain HostAttribute values.")
        particle_ids = tuple(value.particle_id for value in particles)
        vertex_ids = tuple(value.vertex_id for value in vertices)
        weight_names = tuple(value.name for value in weights)
        if len(set(particle_ids)) != len(particle_ids):
            raise ValueError("Host particle identities must be unique per event.")
        if len(set(vertex_ids)) != len(vertex_ids):
            raise ValueError("Host vertex identities must be unique per event.")
        if len(set(weight_names)) != len(weight_names):
            raise ValueError("Host event weight names must be unique.")
        if (
            sum(value.variation_kind is WeightVariationKind.NOMINAL for value in weights)
            != 1
        ):
            raise ValueError("A host event requires exactly one nominal weight.")
        particle_id_set = set(particle_ids)
        vertex_id_set = set(vertex_ids)
        particles_by_id = {value.particle_id: value for value in particles}
        vertices_by_id = {value.vertex_id: value for value in vertices}
        for particle in particles:
            if particle.production_vertex_id is not None:
                if particle.production_vertex_id not in vertex_id_set:
                    raise ValueError("Particle references a missing production vertex.")
                if (
                    particle.particle_id
                    not in vertices_by_id[
                        particle.production_vertex_id
                    ].outgoing_particle_ids
                ):
                    raise ValueError("Particle and production-vertex incidence disagree.")
            if particle.end_vertex_id is not None:
                if particle.end_vertex_id not in vertex_id_set:
                    raise ValueError("Particle references a missing end vertex.")
                if (
                    particle.particle_id
                    not in vertices_by_id[particle.end_vertex_id].incoming_particle_ids
                ):
                    raise ValueError("Particle and end-vertex incidence disagree.")
        for vertex in vertices:
            if (
                not set(vertex.incoming_particle_ids + vertex.outgoing_particle_ids)
                <= particle_id_set
            ):
                raise ValueError("Vertex references a missing particle.")
            for particle_id in vertex.incoming_particle_ids:
                if particles_by_id[particle_id].end_vertex_id != vertex.vertex_id:
                    raise ValueError(
                        "Incoming vertex incidence is not bidirectionally consistent."
                    )
            for particle_id in vertex.outgoing_particle_ids:
                if particles_by_id[particle_id].production_vertex_id != vertex.vertex_id:
                    raise ValueError(
                        "Outgoing vertex incidence is not bidirectionally consistent."
                    )
        object.__setattr__(
            self, "event_id", _integer(self.event_id, "event_id", minimum=0)
        )
        object.__setattr__(
            self,
            "subevent_id",
            _integer(self.subevent_id, "subevent_id", minimum=0, bits=32),
        )
        object.__setattr__(self, "particles", particles)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "provider_status_namespace",
            _identifier(self.provider_status_namespace, "Provider status namespace"),
        )
        object.__setattr__(self, "source_id", _identifier(self.source_id, "Source ID"))
        object.__setattr__(self, "attributes", attributes)


class EventPackingStatus(StrEnum):
    ADMITTED = "admitted"
    PARTICLE_OVERFLOW = "particle-overflow"
    VERTEX_OVERFLOW = "vertex-overflow"
    RELATION_OVERFLOW = "relation-overflow"
    SOURCE_EVENT_OVERFLOW = "source-event-overflow"


class EventPackingReport(StrictModule, NonTrainableState):
    statuses: tuple[EventPackingStatus, ...] = eqx.field(static=True)
    source_event_count: int = eqx.field(static=True)
    admitted_event_count: int = eqx.field(static=True)
    rejected_event_count: int = eqx.field(static=True)
    attribute_loss_count: int = eqx.field(static=True)
    semantic_loss_fields: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        statuses: tuple[EventPackingStatus, ...],
        /,
        *,
        source_event_count: int,
        admitted_event_count: int,
        rejected_event_count: int,
        attribute_loss_count: int,
        semantic_loss_fields: tuple[str, ...],
        plan_id: str,
    ):
        statuses_ = tuple(statuses)
        if any(not isinstance(value, EventPackingStatus) for value in statuses_):
            raise TypeError("statuses must contain EventPackingStatus values.")
        count_inputs = (
            source_event_count,
            admitted_event_count,
            rejected_event_count,
            attribute_loss_count,
        )
        counts = tuple(
            _integer(value, "packing count", minimum=0) for value in count_inputs
        )
        losses = tuple(sorted(str(value).strip() for value in semantic_loss_fields))
        admitted_from_status = sum(
            value is EventPackingStatus.ADMITTED for value in statuses_
        )
        if (
            len(statuses_) != counts[0]
            or counts[1] + counts[2] != counts[0]
            or counts[1] != admitted_from_status
            or counts[2] != counts[0] - admitted_from_status
        ):
            raise ValueError("Packing statuses and counts are inconsistent.")
        if any(not value for value in losses) or len(set(losses)) != len(losses):
            raise ValueError(
                "semantic_loss_fields must contain distinct non-empty values."
            )
        self.statuses = statuses_
        (
            self.source_event_count,
            self.admitted_event_count,
            self.rejected_event_count,
            self.attribute_loss_count,
        ) = counts
        self.semantic_loss_fields = losses
        self.plan_id = _identifier(plan_id, "Plan ID")
        self.report_id = canonical_fingerprint(
            {
                "kind": "event-packing-report",
                "statuses": [value.value for value in statuses_],
                "counts": list(counts),
                "semantic_loss_fields": list(losses),
                "plan": self.plan_id,
            }
        )

    @property
    def successful(self) -> bool:
        return self.rejected_event_count == 0

    @property
    def lossless(self) -> bool:
        return (
            self.successful
            and self.attribute_loss_count == 0
            and not self.semantic_loss_fields
        )


class HostEventPackingResult(StrictModule, NonTrainableState):
    events: ParticleEventBatch
    report: EventPackingReport

    @property
    def successful(self) -> bool:
        return self.report.successful and bool(self.events.successful)


def _weight_layout(records: tuple[HostEventRecord, ...]) -> tuple[HostEventWeight, ...]:
    if not records:
        raise ValueError("At least one host event is required.")
    reference = records[0].weights
    reference_metadata = tuple(
        (value.name, value.variation_kind, value.correlation_group) for value in reference
    )
    for record in records[1:]:
        metadata = tuple(
            (value.name, value.variation_kind, value.correlation_group)
            for value in record.weights
        )
        if metadata != reference_metadata:
            raise ValueError(
                "Every packed event must use the same ordered weight layout."
            )
    return reference


def pack_host_events(
    records: tuple[HostEventRecord, ...],
    plan: ParticleEventPlan | PreparedParticleEvents,
    /,
    *,
    source_id: str,
) -> HostEventPackingResult:
    """Pack authoritative ragged records into one bounded JAX event projection."""
    records_ = tuple(records)
    if any(not isinstance(value, HostEventRecord) for value in records_):
        raise TypeError("records must contain HostEventRecord values.")
    prepared = plan.prepare() if isinstance(plan, ParticleEventPlan) else plan
    if not isinstance(prepared, PreparedParticleEvents):
        raise TypeError("plan must be ParticleEventPlan or PreparedParticleEvents.")
    layout = _weight_layout(records_)
    event_capacity = prepared.plan.event_capacity
    particle_capacity = prepared.plan.particle_capacity
    vertex_capacity = prepared.plan.vertex_capacity
    event_ids = np.zeros((event_capacity,), dtype=np.int64)
    subevent_ids = np.zeros((event_capacity,), dtype=np.int32)
    event_active = np.zeros((event_capacity,), dtype=np.bool_)
    particle_ids = np.zeros((event_capacity, particle_capacity), dtype=np.int64)
    pdg_ids = np.zeros((event_capacity, particle_capacity), dtype=np.int32)
    roles = np.zeros((event_capacity, particle_capacity), dtype=np.int32)
    provider_status = np.zeros((event_capacity, particle_capacity), dtype=np.int32)
    momenta = np.zeros((event_capacity, particle_capacity, 4), dtype=np.float64)
    rest_energies = np.zeros((event_capacity, particle_capacity), dtype=np.float64)
    particle_active = np.zeros((event_capacity, particle_capacity), dtype=np.bool_)
    mother_indices = np.full((event_capacity, particle_capacity, 2), -1, dtype=np.int32)
    production_vertex_indices = np.full(
        (event_capacity, particle_capacity), -1, dtype=np.int32
    )
    end_vertex_indices = np.full((event_capacity, particle_capacity), -1, dtype=np.int32)
    color_flow = np.zeros((event_capacity, particle_capacity, 2), dtype=np.int32)
    vertices = np.zeros((event_capacity, vertex_capacity, 4), dtype=np.float64)
    vertex_ids = np.zeros((event_capacity, vertex_capacity), dtype=np.int64)
    vertex_active = np.zeros((event_capacity, vertex_capacity), dtype=np.bool_)
    overflow = np.zeros((event_capacity,), dtype=np.bool_)
    weight_values = np.zeros((event_capacity, len(layout)), dtype=np.float64)
    statuses: list[EventPackingStatus] = []
    attribute_loss_count = 0
    admitted_source_records = records_[:event_capacity]
    for event_slot, record in enumerate(admitted_source_records):
        event_ids[event_slot] = record.event_id
        subevent_ids[event_slot] = record.subevent_id
        event_active[event_slot] = True
        if record.provider_status_namespace != prepared.plan.provider_status_namespace:
            raise ValueError("Host and device provider status namespaces differ.")
        particles = tuple(sorted(record.particles, key=lambda value: value.particle_id))
        vertices_ = tuple(sorted(record.vertices, key=lambda value: value.vertex_id))
        particle_overflow = len(particles) > particle_capacity
        vertex_overflow = len(vertices_) > vertex_capacity
        particle_slots = {
            value.particle_id: index
            for index, value in enumerate(particles[:particle_capacity])
        }
        vertex_slots = {
            value.vertex_id: index
            for index, value in enumerate(vertices_[:vertex_capacity])
        }
        relation_overflow = False
        for vertex in vertices_[:vertex_capacity]:
            if len(vertex.incoming_particle_ids) > 2:
                relation_overflow = True
        for particle_slot, particle in enumerate(particles[:particle_capacity]):
            particle_ids[event_slot, particle_slot] = particle.particle_id
            pdg_ids[event_slot, particle_slot] = particle.pdg_id
            roles[event_slot, particle_slot] = int(particle.role)
            provider_status[event_slot, particle_slot] = particle.provider_status
            momenta[event_slot, particle_slot] = particle.momentum
            rest_energies[event_slot, particle_slot] = particle.rest_energy
            particle_active[event_slot, particle_slot] = True
            color_flow[event_slot, particle_slot] = particle.color_flow
            if particle.production_vertex_id is not None:
                vertex_slot = vertex_slots.get(particle.production_vertex_id)
                if vertex_slot is None:
                    relation_overflow = True
                else:
                    production_vertex_indices[event_slot, particle_slot] = vertex_slot
                    incoming_ids = next(
                        value.incoming_particle_ids
                        for value in vertices_
                        if value.vertex_id == particle.production_vertex_id
                    )
                    incoming_slots = tuple(
                        particle_slots[value]
                        for value in incoming_ids
                        if value in particle_slots
                    )
                    if (
                        len(incoming_slots) != len(incoming_ids)
                        or len(incoming_slots) > 2
                    ):
                        relation_overflow = True
                    else:
                        mother_indices[
                            event_slot, particle_slot, : len(incoming_slots)
                        ] = incoming_slots
            if particle.end_vertex_id is not None:
                vertex_slot = vertex_slots.get(particle.end_vertex_id)
                if vertex_slot is None:
                    relation_overflow = True
                else:
                    end_vertex_indices[event_slot, particle_slot] = vertex_slot
            attribute_loss_count += len(particle.attributes)
        for vertex_slot, vertex in enumerate(vertices_[:vertex_capacity]):
            vertex_ids[event_slot, vertex_slot] = vertex.vertex_id
            vertices[event_slot, vertex_slot] = vertex.position
            vertex_active[event_slot, vertex_slot] = True
            attribute_loss_count += len(vertex.attributes)
        attribute_loss_count += len(record.attributes)
        weight_values[event_slot] = np.asarray([value.value for value in record.weights])
        event_overflow = particle_overflow or vertex_overflow or relation_overflow
        overflow[event_slot] = event_overflow
        if particle_overflow:
            statuses.append(EventPackingStatus.PARTICLE_OVERFLOW)
        elif vertex_overflow:
            statuses.append(EventPackingStatus.VERTEX_OVERFLOW)
        elif relation_overflow:
            statuses.append(EventPackingStatus.RELATION_OVERFLOW)
        else:
            statuses.append(EventPackingStatus.ADMITTED)
    extra_events = max(len(records_) - event_capacity, 0)
    statuses.extend(EventPackingStatus.SOURCE_EVENT_OVERFLOW for _ in range(extra_events))
    weights = EventWeightSet(
        weight_values,
        names=tuple(value.name for value in layout),
        variation_kinds=tuple(value.variation_kind for value in layout),
        correlation_groups=tuple(value.correlation_group for value in layout),
        event_active=event_active,
        nominal_name=next(
            value.name
            for value in layout
            if value.variation_kind is WeightVariationKind.NOMINAL
        ),
    )
    events = prepared.admit(
        event_ids=event_ids,
        subevent_ids=subevent_ids,
        event_active=event_active,
        particle_ids=particle_ids,
        pdg_ids=pdg_ids,
        roles=roles,
        provider_status=provider_status,
        momenta=momenta,
        rest_energies=rest_energies,
        particle_active=particle_active,
        mother_indices=mother_indices,
        production_vertex_indices=production_vertex_indices,
        end_vertex_indices=end_vertex_indices,
        color_flow=color_flow,
        production_vertices=vertices,
        vertex_active=vertex_active,
        vertex_ids=vertex_ids,
        weights=weights,
        overflow=overflow,
        source_id=source_id,
    )
    semantic_losses = () if attribute_loss_count == 0 else ("opaque-attributes",)
    rejected = sum(value is not EventPackingStatus.ADMITTED for value in statuses)
    report = EventPackingReport(
        tuple(statuses),
        source_event_count=len(records_),
        admitted_event_count=len(records_) - rejected,
        rejected_event_count=rejected,
        attribute_loss_count=attribute_loss_count,
        semantic_loss_fields=semantic_losses,
        plan_id=prepared.plan.plan_id,
    )
    return HostEventPackingResult(events, report)


def unpack_particle_events(events: ParticleEventBatch, /) -> tuple[HostEventRecord, ...]:
    """Recover the exact semantic subset carried by one bounded event batch."""
    if not isinstance(events, ParticleEventBatch):
        raise TypeError("events must be ParticleEventBatch.")
    active_events = np.asarray(events.event_active)
    valid_events = np.asarray(events.valid)
    if np.any(active_events & ~valid_events):
        raise ValueError("Cannot unpack active invalid or overflowed events.")
    records: list[HostEventRecord] = []
    for event_slot in np.flatnonzero(active_events):
        particle_mask = np.asarray(events.particle_active[event_slot])
        vertex_mask = np.asarray(events.vertex_active[event_slot])
        production = np.asarray(events.production_vertex_indices[event_slot])
        ending = np.asarray(events.end_vertex_indices[event_slot])
        particle_slots = tuple(np.flatnonzero(particle_mask))
        particle_ids = tuple(
            int(events.particle_ids[event_slot, slot]) for slot in particle_slots
        )
        particle_id_by_slot = dict(zip(particle_slots, particle_ids, strict=True))
        vertices: list[HostVertexRecord] = []
        for vertex_slot in np.flatnonzero(vertex_mask):
            incoming = tuple(
                particle_id_by_slot[particle_slot]
                for particle_slot in particle_slots
                if ending[particle_slot] == vertex_slot
            )
            outgoing = tuple(
                particle_id_by_slot[particle_slot]
                for particle_slot in particle_slots
                if production[particle_slot] == vertex_slot
            )
            vertices.append(
                HostVertexRecord(
                    int(events.vertex_ids[event_slot, vertex_slot]),
                    tuple(
                        float(value)
                        for value in np.asarray(
                            events.production_vertices[event_slot, vertex_slot]
                        )
                    ),
                    incoming,
                    outgoing,
                )
            )
        particles = tuple(
            HostParticleRecord(
                particle_id_by_slot[particle_slot],
                int(events.pdg_ids[event_slot, particle_slot]),
                ParticleRole(int(events.roles[event_slot, particle_slot])),
                int(events.provider_status[event_slot, particle_slot]),
                tuple(
                    float(value)
                    for value in np.asarray(events.momenta[event_slot, particle_slot])
                ),
                float(events.rest_energies[event_slot, particle_slot]),
                (
                    None
                    if production[particle_slot] < 0
                    else int(
                        events.vertex_ids[event_slot, int(production[particle_slot])]
                    )
                ),
                (
                    None
                    if ending[particle_slot] < 0
                    else int(events.vertex_ids[event_slot, int(ending[particle_slot])])
                ),
                tuple(np.asarray(events.color_flow[event_slot, particle_slot])),
            )
            for particle_slot in particle_slots
        )
        weights = tuple(
            HostEventWeight(
                name,
                float(events.weights.values[event_slot, weight_index]),
                events.weights.variation_kinds[weight_index],
                events.weights.correlation_groups[weight_index],
            )
            for weight_index, name in enumerate(events.weights.names)
        )
        records.append(
            HostEventRecord(
                int(events.event_ids[event_slot]),
                int(events.subevent_ids[event_slot]),
                particles,
                tuple(vertices),
                weights,
                events.provider_status_namespace,
                events.source_id,
            )
        )
    return tuple(records)


__all__ = [
    "EventPackingReport",
    "EventPackingStatus",
    "HostAttribute",
    "HostEventPackingResult",
    "HostEventRecord",
    "HostEventWeight",
    "HostParticleRecord",
    "HostVertexRecord",
    "pack_host_events",
    "unpack_particle_events",
]
