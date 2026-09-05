#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Immutable host-side external transport evidence; no native transport model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

from ..._fingerprint import canonical_fingerprint
from ...artifacts import ScientificArtifactEnvelope
from ...qualification import ReferenceArtifactManifest
from ...units import conversion_factor, ELECTRONVOLT, METER, SECOND, UnitDefinition


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string.")


def _nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _point(value: tuple[float, float, float]) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or not all(map(math.isfinite, value))
    ):
        raise ValueError("Coordinates must be a finite immutable three-vector.")


@dataclass(frozen=True, slots=True, order=True)
class PrimaryHistoryKey:
    """A primary history is local to a source, run and dose fraction."""

    source_id: str
    run_id: str
    primary_id: str
    fraction_id: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            _text(value, name)


@dataclass(frozen=True, slots=True, order=True)
class RadiationEventKey:
    history: PrimaryHistoryKey
    stage: str
    record_id: str

    def __post_init__(self):
        if self.stage not in ("physical", "chemical"):
            raise ValueError("Event stage must be physical or chemical.")
        _text(self.record_id, "record_id")


@dataclass(frozen=True, slots=True)
class RadiationSource:
    """Raw artifact, retained parent rights, and explicit external run configuration.

    None means unreported, not zero. Table/configuration identifiers must address
    caller-retained artifacts. Chemical endpoint is physical time, never a repair
    endpoint. Importers do not grant rights or generate parameter tables.
    """

    artifact: ScientificArtifactEnvelope
    rights: tuple[ReferenceArtifactManifest, ...]
    engine: str
    engine_revision: str
    configuration_id: str
    random_lineage: tuple[str, ...]
    source_table_ids: tuple[str, ...]
    cutoffs: tuple[tuple[str, float, UnitDefinition], ...]
    coordinate_frame: str
    length_unit: UnitDefinition
    energy_unit: UnitDefinition
    time_unit: UnitDefinition
    chemistry_endpoint: float | None = None
    chemistry_model_id: str | None = None
    scavenging_model_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.rights, tuple) or not self.rights:
            raise ValueError("Source requires immutable governing reference manifests.")
        if (
            not isinstance(self.artifact, ScientificArtifactEnvelope)
            or self.artifact.status != "complete"
            or any(
                not isinstance(item, ReferenceArtifactManifest) for item in self.rights
            )
        ):
            raise ValueError(
                "Radiation sources require a complete raw artifact and native rights manifests."
            )
        if self.artifact.content_digest != self.rights[0].checksum:
            raise ValueError("Raw artifact digest must match its governing manifest.")
        for name, value in (
            ("engine", self.engine),
            ("engine_revision", self.engine_revision),
            ("configuration_id", self.configuration_id),
            ("coordinate_frame", self.coordinate_frame),
        ):
            _text(value, name)
        for values in (self.random_lineage, self.source_table_ids, self.cutoffs):
            if not isinstance(values, tuple):
                raise TypeError("Source lineage and cutoffs must be tuples.")
        for value in (*self.random_lineage, *self.source_table_ids):
            _text(value, "lineage")
        conversion_factor(self.length_unit, METER)
        conversion_factor(self.energy_unit, ELECTRONVOLT)
        conversion_factor(self.time_unit, SECOND)
        for name, value, unit in self.cutoffs:
            _text(name, "cutoff name")
            _nonnegative(value, "cutoff")
            if not isinstance(unit, UnitDefinition):
                raise TypeError("Cutoffs require exact units.")
        if self.chemistry_endpoint is not None:
            _nonnegative(self.chemistry_endpoint, "chemistry endpoint")
        for value in (self.chemistry_model_id, self.scavenging_model_id):
            if value is not None:
                _text(value, "chemistry model")

    def require_rights(
        self,
        *,
        commercial_use=False,
        redistribution=False,
        training_use=False,
        export=False,
    ) -> None:
        """Re-admit every retained parent for each requested downstream use."""
        for manifest in self.rights:
            manifest.require_rights(
                commercial_use=commercial_use,
                redistribution=redistribution,
                training_use=training_use,
                export=export,
            )

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "artifact": self.artifact.artifact_id,
                "rights": [item.manifest_id for item in self.rights],
                "engine": self.engine,
                "revision": self.engine_revision,
                "configuration": self.configuration_id,
                "random": self.random_lineage,
                "tables": self.source_table_ids,
                "cutoffs": [
                    (name, value, unit.unit_id) for name, value, unit in self.cutoffs
                ],
                "frame": self.coordinate_frame,
                "units": [
                    self.length_unit.unit_id,
                    self.energy_unit.unit_id,
                    self.time_unit.unit_id,
                ],
                "endpoint": self.chemistry_endpoint,
                "chemistry": self.chemistry_model_id,
                "scavenging": self.scavenging_model_id,
            }
        )


@dataclass(frozen=True, slots=True)
class PhysicalInteraction:
    key: RadiationEventKey
    position: tuple[float, float, float]
    deposited_energy: float
    source_site_id: str | None = None
    track_id: str | None = None
    parent_track_id: str | None = None
    process: str | None = None
    species: str | None = None
    material: str | None = None
    time: float | None = None
    carried_energy: float | None = None
    kinetic_energy_loss: float | None = None

    def __post_init__(self):
        if self.key.stage != "physical":
            raise ValueError("Physical interactions require physical event keys.")
        _point(self.position)
        _nonnegative(self.deposited_energy, "deposited energy")
        for value in (
            self.source_site_id,
            self.track_id,
            self.parent_track_id,
            self.process,
            self.species,
            self.material,
        ):
            if value is not None:
                _text(value, "reported physical identity")
        for value in (self.time, self.carried_energy):
            if value is not None:
                _nonnegative(value, "time or carried energy")
        if self.kinetic_energy_loss is not None and not math.isfinite(
            self.kinetic_energy_loss
        ):
            raise ValueError("Kinetic-energy loss must be finite when reported.")


def _canonical_events(records: Iterable, source: RadiationSource) -> tuple:
    unique = {}
    for record in records:
        if record.key.history.source_id != source.artifact.artifact_id:
            raise ValueError("Event source does not match raw source artifact.")
        previous = unique.get(record.key)
        if previous is not None and previous != record:
            raise ValueError("Conflicting duplicate event identity.")
        unique[record.key] = record
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True, slots=True)
class InteractionLedger:
    source: RadiationSource
    records: tuple[PhysicalInteraction, ...]

    def __post_init__(self):
        if not all(isinstance(item, PhysicalInteraction) for item in self.records):
            raise TypeError("Interaction ledger requires physical interactions.")
        object.__setattr__(self, "records", _canonical_events(self.records, self.source))

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "source": self.source.fingerprint(),
                "physical": [asdict(item) for item in self.records],
            }
        )
