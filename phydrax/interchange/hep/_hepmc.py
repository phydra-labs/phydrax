#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib

import numpy as np

from ...particle_physics import ParticleEventBatch, ParticleRole
from .._report import AdapterLoss, AdapterReport, AdapterStatus
from ._common import HEPExportResult


def write_hepmc3_ascii(events: ParticleEventBatch, /) -> HEPExportResult:
    """Write the bounded particle subset of the HepMC3 ASCII profile."""
    if not isinstance(events, ParticleEventBatch):
        raise TypeError("events must be ParticleEventBatch.")
    active = np.asarray(events.event_active)
    valid = np.asarray(events.valid)
    if np.any(active & ~valid):
        raise ValueError("HepMC export refuses active invalid or overflowed events.")
    momentum_units = {"GeV": "GEV", "MeV": "MEV"}
    length_units = {"mm": "MM", "cm": "CM"}
    if events.momentum_unit_symbol not in momentum_units:
        raise ValueError(
            "HepMC3 ASCII export supports explicit GeV or MeV momentum units."
        )
    if events.length_unit_symbol not in length_units:
        raise ValueError("HepMC3 ASCII export supports explicit mm or cm length units.")
    momentum_unit = momentum_units[events.momentum_unit_symbol]
    length_unit = length_units[events.length_unit_symbol]
    lines = ["HepMC::Version 3.02.05", "HepMC::Asciiv3-START_EVENT_LISTING"]
    particle_active = np.asarray(events.particle_active)
    momenta = np.asarray(events.momenta)
    masses = np.asarray(events.rest_energies)
    pdg_ids = np.asarray(events.pdg_ids)
    status = np.asarray(events.provider_status)
    roles = np.asarray(events.roles)
    nominal = np.asarray(events.weights.nominal)
    event_ids = np.asarray(events.event_ids)
    exported = 0
    for event_index in np.flatnonzero(active):
        particle_indices = np.flatnonzero(particle_active[event_index])
        lines.append(f"E {int(event_ids[event_index])} 0 {len(particle_indices)}")
        lines.append(f"U {momentum_unit} {length_unit}")
        lines.append(f"W {nominal[event_index]:.17g}")
        for output_index, particle_index in enumerate(particle_indices, start=1):
            energy, px, py, pz = momenta[event_index, particle_index]
            provider_status = int(status[event_index, particle_index])
            if provider_status == 0:
                provider_status = (
                    4
                    if int(roles[event_index, particle_index])
                    == int(ParticleRole.OUTGOING)
                    else 3
                )
            lines.append(
                f"P {output_index} 0 {int(pdg_ids[event_index, particle_index])} "
                f"{px:.17g} {py:.17g} {pz:.17g} {energy:.17g} "
                f"{masses[event_index, particle_index]:.17g} {provider_status}"
            )
        exported += 1
    lines.append("HepMC::Asciiv3-END_EVENT_LISTING")
    payload = "\n".join(lines) + "\n"
    losses = (
        AdapterLoss(
            "event.particle_relations",
            "export",
            "dropped",
            "The bounded ASCII profile does not synthesize HepMC vertex topology from mother pairs.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "event.weights[non_nominal]",
            "export",
            "dropped",
            "The bounded ASCII profile writes only the nominal event weight.",
            changes_interpretation=events.weights.weight_count > 1,
        ),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "phydrax-particle-event",
        "HepMC3-ASCII",
        source_id=events.source_id,
        target_id=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        coordinate_mapping=("E,px,py,pz -> px,py,pz,E",),
        preserved_fields=(
            "event_id",
            "pdg_id",
            "status",
            "momentum",
            "mass",
            "nominal_weight",
        ),
        losses=losses,
        stage="hepmc3-ascii-export",
    )
    return HEPExportResult(payload, report, exported)


__all__ = ["write_hepmc3_ascii"]
