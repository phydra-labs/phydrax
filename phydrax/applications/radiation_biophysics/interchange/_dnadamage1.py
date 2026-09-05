#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Geant4 11.3.0 dnadamage1 ROOT ntuple profile, independently implemented.

Format authority (source pinned):
https://github.com/Geant4/geant4/blob/v11.3.0/examples/extended/medical/dna/dnadamage1/src/RunAction.cc
https://github.com/Geant4/geant4/blob/v11.3.0/examples/extended/medical/dna/dnadamage1/src/SteppingAction.cc
https://github.com/Geant4/geant4/blob/v11.3.0/examples/extended/medical/dna/dnadamage1/src/TimeStepAction.cc

ntuple/ntuple_1: x,y,z [nm], edep,diffKin [eV], volumeName, CopyNumber, EventID.
ntuple/ntuple_2: x,y,z [nm], RadName, EventID. Its writer selects OH reactions
producing damaged deoxyribose, not all radiolysis reactions. diffKin is kinetic
energy LOSS, never carried energy or a substitute for deposited energy.

Track/parent/process/physical species/time are absent. No time-dependent G-value
qualification is possible from this profile. Chemistry endpoint, run/fraction,
material table, source revision, RNG/configuration and rights must be supplied
from retained external run artifacts. Required absent semantics cause refusal.
ROOT loading is optional/lazy; the column route consumes the identical profile
without importing ROOT libraries. Original entry IDs MUST accompany column views.
Neither hand-authored column fixtures nor this adapter qualify the provider.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...._fingerprint import canonical_fingerprint, canonical_json
from ....interchange import AdapterLoss, AdapterReport, AdapterStatus
from ....units import conversion_factor, ELECTRONVOLT, METER, UnitDefinition
from .._interactions import (
    InteractionLedger,
    PhysicalInteraction,
    PrimaryHistoryKey,
    RadiationEventKey,
    RadiationSource,
)
from .._reactions import ChemicalReaction, ReactionLedger


DNADAMAGE1_REVISION = "v11.3.0"
DNADAMAGE1_PROFILE = "Geant4-dnadamage1-ROOT-11.3.0"
NANOMETER = UnitDefinition("nm", METER.dimension, METER.reference_system_id, "1e-9")
_PHYSICAL = ("x", "y", "z", "edep", "diffKin", "volumeName", "CopyNumber", "EventID")
_CHEMICAL = ("x", "y", "z", "RadName", "EventID")
_UNREPORTED = (
    "track_id",
    "parent_track_id",
    "process",
    "physical_species",
    "event_time",
    "carried_energy",
)


@dataclass(frozen=True, slots=True)
class ImportedRadiationLedgers:
    physical: InteractionLedger
    chemical: ReactionLedger
    report: AdapterReport


def dnadamage1_column_payload(
    physical: Mapping[str, Sequence],
    chemical: Mapping[str, Sequence],
    physical_entry_ids: Sequence[int],
    chemical_entry_ids: Sequence[int],
) -> bytes:
    """Canonical retained column-view bytes for checksum/rights admission.

    These bytes describe a column derivative, NOT the original ROOT bytes. Keep
    the ROOT artifact as a governed parent when creating such a derivative.
    """
    return canonical_json(
        {
            "profile": DNADAMAGE1_PROFILE,
            "physical": {
                name: np.asarray(values).tolist() for name, values in physical.items()
            },
            "chemical": {
                name: np.asarray(values).tolist() for name, values in chemical.items()
            },
            "physical_entry_ids": np.asarray(physical_entry_ids).tolist(),
            "chemical_entry_ids": np.asarray(chemical_entry_ids).tolist(),
        }
    ).encode("utf-8")


def _integer(value, name: str) -> int:
    integer = int(value)
    if isinstance(value, bool) or integer != value or integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer, not a truncated float.")
    return integer


def _check_columns(columns, expected, entry_ids):
    if set(columns) != set(expected):
        raise ValueError("Source columns do not match the pinned dnadamage1 profile.")
    n = len(entry_ids)
    if any(len(columns[name]) != n for name in expected):
        raise ValueError("Source column and original-entry capacities disagree.")
    ids = tuple(_integer(value, "original entry ID") for value in entry_ids)
    if len(set(ids)) != n:
        raise ValueError("Original ROOT entry IDs must be unique within an ntuple.")
    return ids


def _import_columns(
    physical_columns,
    chemical_columns,
    *,
    source,
    run_id,
    fraction_id,
    physical_entry_ids,
    chemical_entry_ids,
    volume_materials,
    required_semantics,
    commercial_use,
):
    source.require_rights(commercial_use=commercial_use)
    if (
        source.engine != "Geant4-dnadamage1"
        or source.engine_revision != DNADAMAGE1_REVISION
    ):
        raise ValueError(
            "Only the pinned Geant4-dnadamage1 v11.3.0 writer profile is supported."
        )
    if (
        conversion_factor(source.length_unit, NANOMETER) != 1
        or conversion_factor(source.energy_unit, ELECTRONVOLT) != 1
    ):
        raise ValueError(
            "dnadamage1 writer coordinates/deposition must be declared nm/eV."
        )
    if not source.random_lineage or not source.source_table_ids:
        raise ValueError(
            "External random lineage and source table artifacts are required."
        )
    missing = set(required_semantics) & set(_UNREPORTED)
    if missing:
        raise ValueError(f"Source profile omits required semantics: {sorted(missing)}")
    supported = set(_UNREPORTED) | {
        "event_identity",
        "primary_history",
        "deposited_energy",
        "reaction_channel",
        "coordinates",
        "material",
    }
    if set(required_semantics) - supported:
        raise ValueError("Unknown required adapter semantic.")
    physical_ids = _check_columns(physical_columns, _PHYSICAL, physical_entry_ids)
    chemical_ids = _check_columns(chemical_columns, _CHEMICAL, chemical_entry_ids)
    if chemical_ids and (
        source.chemistry_endpoint is None
        or source.chemistry_model_id is None
        or source.scavenging_model_id is None
    ):
        raise ValueError(
            "Untimed chemistry requires retained endpoint/model/scavenging metadata."
        )
    physical = []
    chemical = []
    for index, entry_id in enumerate(physical_ids):
        volume = _integer(physical_columns["volumeName"][index], "volume code")
        copy = _integer(physical_columns["CopyNumber"][index], "copy number")
        event = _integer(physical_columns["EventID"][index], "primary EventID")
        if volume not in volume_materials:
            raise ValueError(
                "Source volume lacks an explicitly supplied material mapping."
            )
        history = PrimaryHistoryKey(
            source.artifact.artifact_id, run_id, str(event), fraction_id
        )
        physical.append(
            PhysicalInteraction(
                RadiationEventKey(history, "physical", f"ntuple_1:{entry_id}"),
                tuple(float(physical_columns[name][index]) for name in ("x", "y", "z")),
                float(physical_columns["edep"][index]),
                source_site_id=f"{volume}:{copy}",
                material=volume_materials[volume],
                kinetic_energy_loss=float(physical_columns["diffKin"][index]),
            )
        )
    for index, entry_id in enumerate(chemical_ids):
        event = _integer(chemical_columns["EventID"][index], "primary EventID")
        radical = chemical_columns["RadName"][index]
        if not isinstance(radical, str):
            raise TypeError("RadName must be decoded text, not guessed from bytes.")
        history = PrimaryHistoryKey(
            source.artifact.artifact_id, run_id, str(event), fraction_id
        )
        chemical.append(
            ChemicalReaction(
                RadiationEventKey(history, "chemical", f"ntuple_2:{entry_id}"),
                tuple(float(chemical_columns[name][index]) for name in ("x", "y", "z")),
                "OH-deoxyribose-damage",
                (radical, "Deoxyribose"),
                ("DamagedDeoxyribose",),
            )
        )
    physical_ = InteractionLedger(source, tuple(physical))
    chemical_ = ReactionLedger(source, tuple(chemical))
    losses = tuple(
        AdapterLoss(
            f"events/{name}",
            "import",
            "unsupported",
            "Pinned source ntuple does not record this semantic; retained as unknown.",
            changes_interpretation=True,
            affected_capability_ids=(name,),
        )
        for name in _UNREPORTED
    )
    losses += (
        AdapterLoss(
            "chemical/coverage",
            "import",
            "unsupported",
            "Source records only selected OH/deoxyribose damage reactions, not a complete radiolysis ledger.",
            changes_interpretation=True,
            affected_capability_ids=("time-dependent-G-values",),
        ),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        DNADAMAGE1_PROFILE,
        "external-radiation-ledgers",
        source_id=source.artifact.artifact_id,
        target_id=canonical_fingerprint(
            [physical_.fingerprint(), chemical_.fingerprint()]
        ),
        losses=losses,
        coordinate_mapping=(source.coordinate_frame,),
        preserved_fields=(
            "EventID",
            "entry_id",
            "coordinates",
            "edep",
            "diffKin",
            "volumeName",
            "CopyNumber",
            "RadName",
        ),
        assumptions=(
            "nm/eV writer units",
            "external material map",
            "external run/fraction identity",
            "selected reaction channel from pinned writer",
        ),
    )
    return ImportedRadiationLedgers(physical_, chemical_, report)


def import_dnadamage1_columns(
    physical_columns,
    chemical_columns,
    *,
    source: RadiationSource,
    run_id: str,
    fraction_id: str,
    physical_entry_ids,
    chemical_entry_ids,
    volume_materials,
    required_semantics: tuple[str, ...] = (),
    commercial_use=False,
) -> ImportedRadiationLedgers:
    payload = dnadamage1_column_payload(
        physical_columns, chemical_columns, physical_entry_ids, chemical_entry_ids
    )
    manifest = source.rights[0]
    if (
        hashlib.new(manifest.checksum_algorithm, payload).hexdigest() != manifest.checksum
        or len(payload) != manifest.size_bytes
    ):
        raise ValueError("Column-view bytes do not match their admitted source artifact.")
    return _import_columns(
        physical_columns,
        chemical_columns,
        source=source,
        run_id=run_id,
        fraction_id=fraction_id,
        physical_entry_ids=physical_entry_ids,
        chemical_entry_ids=chemical_entry_ids,
        volume_materials=volume_materials,
        required_semantics=required_semantics,
        commercial_use=commercial_use,
    )


def import_dnadamage1_root(
    path: str | Path,
    *,
    source: RadiationSource,
    run_id: str,
    fraction_id: str,
    volume_materials,
    required_semantics: tuple[str, ...] = (),
    commercial_use=False,
) -> ImportedRadiationLedgers:
    """Import an admitted real ROOT file; requires the optional ``uproot`` package."""
    source.require_rights(commercial_use=commercial_use)
    path_ = Path(path)
    manifest = source.rights[0]
    with path_.open("rb") as stream:
        checksum = hashlib.file_digest(stream, manifest.checksum_algorithm).hexdigest()
    if checksum != manifest.checksum or path_.stat().st_size != manifest.size_bytes:
        raise ValueError("ROOT file bytes do not match their admitted source artifact.")
    import uproot

    with uproot.open(path_) as root:
        physical = root["ntuple/ntuple_1"].arrays(list(_PHYSICAL), library="np")
        chemical = root["ntuple/ntuple_2"].arrays(list(_CHEMICAL), library="np")
    return _import_columns(
        physical,
        chemical,
        source=source,
        run_id=run_id,
        fraction_id=fraction_id,
        physical_entry_ids=range(len(physical["EventID"])),
        chemical_entry_ids=range(len(chemical["EventID"])),
        volume_materials=volume_materials,
        required_semantics=required_semantics,
        commercial_use=commercial_use,
    )
