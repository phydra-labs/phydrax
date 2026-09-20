#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import numpy as np

from ...particle_physics import (
    EventWeightSet,
    ParticleEventBatch,
    ParticleEventPlan,
    ParticleRole,
    PreparedParticleEvents,
    WeightVariationKind,
)
from .._report import AdapterFormatProfile, AdapterLoss, AdapterReport, AdapterStatus
from ._common import HEPExportResult, HEPImportResult


def _source_id(text: str, declared: str | None, /) -> str:
    if declared is not None:
        value = str(declared).strip()
        if not value:
            raise ValueError("source_id must be non-empty when supplied.")
        return value
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_lhef(
    text: str,
    prepared: ParticleEventPlan | PreparedParticleEvents,
    /,
    *,
    source_id: str | None = None,
    event_number_start: int = 0,
) -> HEPImportResult:
    """Read the particle and nominal-weight subset of a Les Houches event document."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty LHEF document.")
    prepared = prepared.prepare() if isinstance(prepared, ParticleEventPlan) else prepared
    if not isinstance(prepared, PreparedParticleEvents):
        raise TypeError("prepared must be ParticleEventPlan or PreparedParticleEvents.")
    if prepared.plan.momentum_unit.symbol != "GeV":
        raise ValueError("The supported LHEF profile requires GeV momentum values.")
    root = ET.fromstring(text)
    event_nodes = tuple(root.iter("event"))
    if not event_nodes:
        raise ValueError("LHEF input contains no event elements.")
    event_capacity = prepared.plan.event_capacity
    particle_capacity = prepared.plan.particle_capacity
    vertex_capacity = prepared.plan.vertex_capacity
    event_ids = np.arange(
        event_number_start, event_number_start + event_capacity, dtype=np.int64
    )
    subevent_ids = np.zeros((event_capacity,), dtype=np.int32)
    event_active = np.zeros((event_capacity,), dtype=np.bool_)
    pdg_ids = np.zeros((event_capacity, particle_capacity), dtype=np.int32)
    roles = np.zeros_like(pdg_ids)
    provider_status = np.zeros_like(pdg_ids)
    momenta = np.zeros((event_capacity, particle_capacity, 4), dtype=np.float64)
    rest_energies = np.zeros((event_capacity, particle_capacity), dtype=np.float64)
    particle_active = np.zeros((event_capacity, particle_capacity), dtype=np.bool_)
    mother_indices = np.full((event_capacity, particle_capacity, 2), -1, dtype=np.int32)
    color_flow = np.zeros((event_capacity, particle_capacity, 2), dtype=np.int32)
    vertices = np.zeros((event_capacity, vertex_capacity, 4), dtype=np.float64)
    vertex_active = np.zeros((event_capacity, vertex_capacity), dtype=np.bool_)
    production_vertex_indices = np.full(
        (event_capacity, particle_capacity), -1, dtype=np.int32
    )
    end_vertex_indices = np.full((event_capacity, particle_capacity), -1, dtype=np.int32)
    overflow = np.zeros((event_capacity,), dtype=np.bool_)
    nominal_weights = np.zeros((event_capacity, 1), dtype=np.float64)
    dropped_alternates = False
    for event_index, node in enumerate(event_nodes[:event_capacity]):
        lines = tuple(
            line.strip()
            for line in (node.text or "").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if not lines:
            raise ValueError("LHEF event has no header.")
        header = lines[0].split()
        if len(header) < 3:
            raise ValueError("LHEF event header is incomplete.")
        particle_count = int(header[0])
        if particle_count < 0 or len(lines) < particle_count + 1:
            raise ValueError("LHEF particle count does not match event content.")
        event_active[event_index] = True
        nominal_weights[event_index, 0] = float(header[2])
        if particle_count > particle_capacity:
            overflow[event_index] = True
        for particle_index, line in enumerate(
            lines[1 : 1 + min(particle_count, particle_capacity)]
        ):
            fields = line.split()
            if len(fields) < 13:
                raise ValueError("LHEF particle row must contain thirteen fields.")
            pdg = int(fields[0])
            status = int(fields[1])
            mothers = (int(fields[2]), int(fields[3]))
            colors = (int(fields[4]), int(fields[5]))
            px, py, pz, energy, mass = map(float, fields[6:11])
            pdg_ids[event_index, particle_index] = pdg
            provider_status[event_index, particle_index] = status
            if status == -1:
                roles[event_index, particle_index] = int(ParticleRole.INCOMING)
            elif status == 1:
                roles[event_index, particle_index] = int(ParticleRole.OUTGOING)
            elif status == 2:
                roles[event_index, particle_index] = int(ParticleRole.INTERMEDIATE)
            else:
                roles[event_index, particle_index] = int(ParticleRole.UNKNOWN)
            momenta[event_index, particle_index] = (energy, px, py, pz)
            rest_energies[event_index, particle_index] = mass
            particle_active[event_index, particle_index] = True
            mother_indices[event_index, particle_index] = tuple(
                value - 1 if value > 0 else -1 for value in mothers
            )
            color_flow[event_index, particle_index] = colors
        dropped_alternates = dropped_alternates or any(True for _ in node.iter("wgt"))
    source_count = len(event_nodes)
    overflow_count = max(source_count - event_capacity, 0) + int(np.sum(overflow))
    if source_count > event_capacity:
        overflow[event_active] = True
    weights = EventWeightSet(
        nominal_weights,
        names=("nominal",),
        variation_kinds=(WeightVariationKind.NOMINAL,),
        correlation_groups=("nominal",),
        event_active=event_active,
    )
    resolved_source_id = _source_id(text, source_id)
    events = prepared.admit(
        event_ids=event_ids,
        subevent_ids=subevent_ids,
        event_active=event_active,
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
        weights=weights,
        overflow=overflow,
        source_id=resolved_source_id,
    )
    losses: tuple[AdapterLoss, ...] = ()
    if dropped_alternates:
        losses = (
            AdapterLoss(
                "event.rwgt",
                "import",
                "dropped",
                "This bounded LHEF profile admits only the nominal event weight.",
                changes_interpretation=True,
            ),
        )
    status = (
        AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
        if overflow_count
        else AdapterStatus.DECLARED_LOSS
        if losses
        else AdapterStatus.LOSSLESS
    )
    report = AdapterReport(
        status,
        "LHEF",
        "phydrax-particle-event",
        source_id=resolved_source_id,
        target_id=prepared.prepared_id,
        coordinate_mapping=(
            "px,py,pz,E -> E,px,py,pz",
            "one-based mothers -> zero-based mothers",
        ),
        preserved_fields=(
            "pdg_id",
            "status",
            "mothers",
            "color",
            "momentum",
            "mass",
            "nominal_weight",
        ),
        losses=losses,
        source_profile=AdapterFormatProfile(
            "LHEF", qualifiers={"weight_profile": "nominal"}
        ),
        target_profile=AdapterFormatProfile(
            "phydrax-particle-event",
            qualifiers={"momentum_unit": prepared.plan.momentum_unit.symbol},
        ),
        stage="lhef-import",
    )
    return HEPImportResult(
        events, report, source_count, min(source_count, event_capacity), overflow_count
    )


def write_lhef(
    events: ParticleEventBatch,
    /,
    *,
    generator_name: str = "phydrax",
) -> HEPExportResult:
    """Write active, valid events to the supported nominal-weight LHEF profile."""
    if not isinstance(events, ParticleEventBatch):
        raise TypeError("events must be ParticleEventBatch.")
    active = np.asarray(events.event_active)
    valid = np.asarray(events.valid)
    if np.any(active & ~valid):
        raise ValueError("LHEF export refuses active invalid or overflowed events.")
    if events.momentum_unit_symbol != "GeV":
        raise ValueError("The supported LHEF profile requires GeV momentum values.")
    lines = [
        '<LesHouchesEvents version="3.0">',
        "<header>",
        f"<!-- generated by {str(generator_name).strip() or 'phydrax'} -->",
        "</header>",
        "<init>",
        "0 0 0 0 0 0 0 0 3 1",
        "0 0 0 0",
        "</init>",
    ]
    exported = 0
    momenta = np.asarray(events.momenta)
    masses = np.asarray(events.rest_energies)
    pdg_ids = np.asarray(events.pdg_ids)
    roles = np.asarray(events.roles)
    provider_status = np.asarray(events.provider_status)
    mothers = np.asarray(events.mother_indices)
    colors = np.asarray(events.color_flow)
    particle_active = np.asarray(events.particle_active)
    nominal = np.asarray(events.weights.nominal)
    for event_index in np.flatnonzero(active):
        particles = np.flatnonzero(particle_active[event_index])
        lines.extend(("<event>", f"{len(particles)} 0 {nominal[event_index]:.17g} 0 0 0"))
        for particle_index in particles:
            energy, px, py, pz = momenta[event_index, particle_index]
            role = int(roles[event_index, particle_index])
            status = int(provider_status[event_index, particle_index])
            if status == 0:
                status = (
                    -1
                    if role == int(ParticleRole.INCOMING)
                    else 2
                    if role == int(ParticleRole.INTERMEDIATE)
                    else 1
                )
            mother_one, mother_two = mothers[event_index, particle_index]
            color_one, color_two = colors[event_index, particle_index]
            lines.append(
                f"{int(pdg_ids[event_index, particle_index])} {status} "
                f"{int(mother_one) + 1 if mother_one >= 0 else 0} "
                f"{int(mother_two) + 1 if mother_two >= 0 else 0} "
                f"{int(color_one)} {int(color_two)} {px:.17g} {py:.17g} {pz:.17g} "
                f"{energy:.17g} {masses[event_index, particle_index]:.17g} 0 9"
            )
        lines.append("</event>")
        exported += 1
    lines.append("</LesHouchesEvents>")
    payload = "\n".join(lines) + "\n"
    losses = (
        AdapterLoss(
            "event.vertices",
            "export",
            "unsupported",
            "The supported LHEF profile does not serialize spacetime production vertices.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "event.weights[non_nominal]",
            "export",
            "dropped",
            "The supported LHEF profile writes only the nominal event weight.",
            changes_interpretation=events.weights.weight_count > 1,
        ),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "phydrax-particle-event",
        "LHEF",
        source_id=events.source_id,
        target_id=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        coordinate_mapping=(
            "E,px,py,pz -> px,py,pz,E",
            "zero-based mothers -> one-based mothers",
        ),
        preserved_fields=(
            "pdg_id",
            "status",
            "mothers",
            "color",
            "momentum",
            "mass",
            "nominal_weight",
        ),
        losses=losses,
        stage="lhef-export",
    )
    return HEPExportResult(payload, report, exported)


__all__ = ["read_lhef", "write_lhef"]
