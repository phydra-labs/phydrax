#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping, Sequence

from ...particle_physics import (
    HostAttribute,
    HostEventRecord,
    HostEventWeight,
    HostParticleRecord,
    HostVertexRecord,
    ParticleRole,
    WeightVariationKind,
)
from .._report import AdapterFormatProfile, AdapterReport, AdapterStatus
from ._common import HEPOptionalDependencyError


def _attribute_record(value: HostAttribute, /) -> dict[str, object]:
    return {
        "namespace": value.namespace,
        "name": value.name,
        "content_type": value.content_type,
        "payload_hex": value.payload.hex(),
    }


def host_events_to_records(
    events: Sequence[HostEventRecord], /
) -> list[dict[str, object]]:
    """Convert authoritative host events to a canonical JSON/columnar-ready form."""
    events_ = tuple(events)
    if any(not isinstance(value, HostEventRecord) for value in events_):
        raise TypeError("events must contain HostEventRecord values.")
    return [
        {
            "event_id": event.event_id,
            "subevent_id": event.subevent_id,
            "provider_status_namespace": event.provider_status_namespace,
            "source_id": event.source_id,
            "attributes": [_attribute_record(value) for value in event.attributes],
            "weights": [
                {
                    "name": value.name,
                    "value": value.value,
                    "variation_kind": value.variation_kind.value,
                    "correlation_group": value.correlation_group,
                }
                for value in event.weights
            ],
            "particles": [
                {
                    "particle_id": value.particle_id,
                    "pdg_id": value.pdg_id,
                    "role": int(value.role),
                    "provider_status": value.provider_status,
                    "momentum": list(value.momentum),
                    "rest_energy": value.rest_energy,
                    "production_vertex_id": value.production_vertex_id,
                    "end_vertex_id": value.end_vertex_id,
                    "color_flow": list(value.color_flow),
                    "attributes": [_attribute_record(item) for item in value.attributes],
                }
                for value in event.particles
            ],
            "vertices": [
                {
                    "vertex_id": value.vertex_id,
                    "position": list(value.position),
                    "incoming_particle_ids": list(value.incoming_particle_ids),
                    "outgoing_particle_ids": list(value.outgoing_particle_ids),
                    "attributes": [_attribute_record(item) for item in value.attributes],
                }
                for value in event.vertices
            ],
        }
        for event in events_
    ]


def _attribute_from_record(record: Mapping[str, object], /) -> HostAttribute:
    return HostAttribute(
        str(record["namespace"]),
        str(record["name"]),
        str(record["content_type"]),
        bytes.fromhex(str(record["payload_hex"])),
    )


def host_events_from_records(
    records: Sequence[Mapping[str, object]], /
) -> tuple[HostEventRecord, ...]:
    """Reconstruct and validate canonical host-event records."""
    result: list[HostEventRecord] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Every host-event record must be a mapping.")
        particle_records = record["particles"]
        vertex_records = record["vertices"]
        weight_records = record["weights"]
        attribute_records = record["attributes"]
        for values, name in (
            (particle_records, "particles"),
            (vertex_records, "vertices"),
            (weight_records, "weights"),
            (attribute_records, "attributes"),
        ):
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise TypeError(f"{name} must be a sequence.")
        particles = tuple(
            HostParticleRecord(
                int(value["particle_id"]),
                int(value["pdg_id"]),
                ParticleRole(int(value["role"])),
                int(value["provider_status"]),
                tuple(float(item) for item in value["momentum"]),
                float(value["rest_energy"]),
                None
                if value["production_vertex_id"] is None
                else int(value["production_vertex_id"]),
                None if value["end_vertex_id"] is None else int(value["end_vertex_id"]),
                tuple(value["color_flow"]),
                tuple(_attribute_from_record(item) for item in value["attributes"]),
            )
            for value in particle_records
        )
        vertices = tuple(
            HostVertexRecord(
                int(value["vertex_id"]),
                tuple(float(item) for item in value["position"]),
                tuple(value["incoming_particle_ids"]),
                tuple(value["outgoing_particle_ids"]),
                tuple(_attribute_from_record(item) for item in value["attributes"]),
            )
            for value in vertex_records
        )
        weights = tuple(
            HostEventWeight(
                str(value["name"]),
                float(value["value"]),
                WeightVariationKind(str(value["variation_kind"])),
                str(value["correlation_group"]),
            )
            for value in weight_records
        )
        result.append(
            HostEventRecord(
                int(record["event_id"]),
                int(record["subevent_id"]),
                particles,
                vertices,
                weights,
                str(record["provider_status_namespace"]),
                str(record["source_id"]),
                tuple(_attribute_from_record(value) for value in attribute_records),
            )
        )
    return tuple(result)


def host_events_to_awkward(events: Sequence[HostEventRecord], /):
    """Create an optional Awkward host view without changing Phydrax ownership."""
    if importlib.util.find_spec("awkward") is None:
        raise HEPOptionalDependencyError(
            "Awkward host-event export requires optional awkward."
        )
    awkward = importlib.import_module("awkward")
    return awkward.Array(host_events_to_records(events))


def host_events_from_awkward(array, /) -> tuple[HostEventRecord, ...]:
    if importlib.util.find_spec("awkward") is None:
        raise HEPOptionalDependencyError(
            "Awkward host-event import requires optional awkward."
        )
    awkward = importlib.import_module("awkward")
    return host_events_from_records(awkward.to_list(array))


def columnar_host_event_report(
    *, source_format: str, source_id: str, target_id: str
) -> AdapterReport:
    """Describe the canonical host-event form used by Awkward/Arrow-style adapters."""
    fields = (
        "event identity",
        "particle and vertex topology",
        "named weights",
        "provider status",
        "opaque attributes",
    )
    return AdapterReport(
        AdapterStatus.LOSSLESS,
        str(source_format),
        "phydrax-host-particle-events",
        source_id=str(source_id),
        target_id=str(target_id),
        preserved_fields=fields,
        source_profile=AdapterFormatProfile(
            str(source_format), qualifiers={"form": "profile-specific"}
        ),
        target_profile=AdapterFormatProfile(
            "phydrax-host-particle-events", qualifiers={"form": "canonical-host-record"}
        ),
        stage="hep-columnar-host-event",
    )


__all__ = [
    "columnar_host_event_report",
    "host_events_from_awkward",
    "host_events_from_records",
    "host_events_to_awkward",
    "host_events_to_records",
]
