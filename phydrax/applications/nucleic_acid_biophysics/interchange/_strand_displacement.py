#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Admission of exact Zenodo 10090783 fluorescence workbooks.

The adapter is deliberately offline.  It accepts caller-owned paths and exact
byte manifests, verifies rights and identities before parsing, and recognizes
the CLARIOstar ``Table All Cycles`` and accompanying plate-setup layouts used
by the deposited source.  Notebook outputs are retained only as lineage.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....qualification import (
    CampaignRole,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
)
from ....units import conversion_factor, MOLE_PER_CUBIC_METER, UnitDefinition


_ZENODO_RECORD_ID = "zenodo:10090783"
_ZENODO_VERSION = "1"
_ZENODO_DOI = "10.5281/zenodo.10090783"
_LICENSE_ID = "CC-BY-4.0"
_DIRECTIONS = frozenset(("RNA>DNA", "DNA>RNA", "DNA>DNA", "reporter-characterization"))
_ROLES = frozenset(
    (
        "calibration",
        "model_selection",
        "interval_calibration",
        "locked_evaluation",
        "prospective",
    )
)
_PREPARED_TRACE_COLUMNS = (
    "case_id",
    "sample_label",
    "experiment_id",
    "plate_id",
    "well_id",
    "preparation_id",
    "replicate_id",
    "reporter_id",
    "sequence_family_id",
    "condition_id",
    "chemistry_direction",
    "temperature_kelvin",
    "time_seconds",
    "intensity",
    "intensity_unit_id",
    "saturation_state",
    "injection_reference_seconds",
    "saturation_threshold_intensity",
    "construct_ids_json",
    "initial_concentrations_molar_json",
    "source_manifest_ids_json",
)
_XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XLSX_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    result = tuple(_identifier(value, name) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and unique.")
    return result


def _positive(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _requested_use(value: Mapping[str, bool], /) -> tuple[tuple[str, bool], ...]:
    required = {"commercial_use", "redistribution", "training_use", "export"}
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or any(type(enabled) is not bool for enabled in value.values())
    ):
        raise ValueError("requested_use must explicitly declare all four use flags.")
    return tuple(sorted(value.items()))


def _require_manifest_bytes(
    content: bytes,
    manifest: ReferenceArtifactManifest,
    requested_use: tuple[tuple[str, bool], ...],
    /,
) -> None:
    manifest.require_rights(**dict(requested_use))
    if len(content) != manifest.size_bytes:
        raise ValueError(
            f"Artifact {manifest.artifact_name!r} byte size disagrees with its manifest."
        )
    digest = hashlib.new(manifest.checksum_algorithm, content).hexdigest()
    if digest != manifest.checksum:
        raise ValueError(
            f"Artifact {manifest.artifact_name!r} digest disagrees with its manifest."
        )


def _member_path(value: str, name: str, /) -> str:
    result = _identifier(value, name)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{name} must be a canonical relative archive member path.")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class PlateWellIdentity:
    """Exact instrument well and independent preparation identity."""

    experiment_id: str
    plate_id: str
    well_id: str
    preparation_id: str
    replicate_id: str

    def __post_init__(self):
        for value, name in (
            (self.experiment_id, "experiment_id"),
            (self.plate_id, "plate_id"),
            (self.well_id, "well_id"),
            (self.preparation_id, "preparation_id"),
            (self.replicate_id, "replicate_id"),
        ):
            _identifier(value, name)


@dataclass(frozen=True, slots=True, init=False)
class FluorescenceTimeTrace:
    """One raw independent well trace on an injection-relative second clock.

    Instrument saturation is represented by ``NaN`` intensity together with a
    true saturation mask.  It is never clipped into the observed range.  The
    injection marker itself is not a measurement and is omitted after its
    source position establishes ``injection_reference_seconds``.
    """

    case_id: str
    identity: PlateWellIdentity
    time_seconds: Array
    intensity: Array
    saturation_mask: Array
    has_saturated_observations: bool
    construct_ids: tuple[str, ...]
    initial_concentrations_molar: Array
    temperature_kelvin: float
    condition_id: str
    chemistry_direction: str
    reporter_id: str
    sequence_family_id: str
    source_manifest_ids: tuple[str, ...]
    injection_reference_seconds: float | None
    saturation_threshold_intensity: float | None
    intensity_unit_id: str
    trace_id: str

    def __init__(
        self,
        case_id: str,
        identity: PlateWellIdentity,
        time_seconds,
        intensity,
        saturation_mask,
        construct_ids: Sequence[str],
        initial_concentrations_molar,
        *,
        temperature_kelvin: float,
        condition_id: str,
        chemistry_direction: str,
        reporter_id: str,
        sequence_family_id: str,
        source_manifest_ids: Sequence[str],
        injection_reference_seconds: float | None,
        saturation_threshold_intensity: float | None,
        intensity_unit_id: str = "instrument-fluorescence-unit",
    ):
        if not isinstance(identity, PlateWellIdentity):
            raise TypeError("identity must be a PlateWellIdentity.")
        case = _identifier(case_id, "case_id")
        times = np.asarray(time_seconds, dtype=float)
        values = np.asarray(intensity, dtype=float)
        saturated = np.asarray(saturation_mask, dtype=bool)
        if times.ndim != 1 or times.size == 0:
            raise ValueError(
                "A fluorescence trace requires a non-empty one-dimensional clock."
            )
        if values.shape != times.shape or saturated.shape != times.shape:
            raise ValueError(
                "Time, intensity and saturation arrays must have identical shape."
            )
        if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
            raise ValueError("Trace times must be finite and strictly increasing.")
        if np.any(~saturated & ~np.isfinite(values)):
            raise ValueError("Every unsaturated fluorescence observation must be finite.")
        if np.any(saturated & ~np.isnan(values)):
            raise ValueError(
                "Saturated intensity must remain NaN rather than be clipped."
            )
        constructs = _identifiers(tuple(construct_ids), "construct_ids")
        concentrations = np.asarray(initial_concentrations_molar, dtype=float)
        if concentrations.shape != (len(constructs),):
            raise ValueError("Each construct requires one initial molar concentration.")
        if np.any(~np.isfinite(concentrations)) or np.any(concentrations < 0.0):
            raise ValueError(
                "Initial molar concentrations must be finite and non-negative."
            )
        temperature = _positive(temperature_kelvin, "temperature_kelvin")
        injection = (
            None
            if injection_reference_seconds is None
            else float(injection_reference_seconds)
        )
        if injection is not None and not math.isfinite(injection):
            raise ValueError("Injection reference time must be finite when present.")
        threshold = (
            None
            if saturation_threshold_intensity is None
            else _positive(
                saturation_threshold_intensity, "saturation_threshold_intensity"
            )
        )
        manifests = tuple(
            sorted(_identifiers(source_manifest_ids, "source_manifest_ids"))
        )
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "time_seconds", jnp.asarray(times))
        object.__setattr__(self, "intensity", jnp.asarray(values))
        object.__setattr__(self, "has_saturated_observations", bool(np.any(saturated)))
        object.__setattr__(self, "saturation_mask", jnp.asarray(saturated))
        object.__setattr__(self, "construct_ids", constructs)
        object.__setattr__(
            self, "initial_concentrations_molar", jnp.asarray(concentrations)
        )
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(
            self,
            "chemistry_direction",
            _identifier(chemistry_direction, "chemistry_direction"),
        )
        object.__setattr__(
            self, "condition_id", _identifier(condition_id, "condition_id")
        )
        object.__setattr__(self, "reporter_id", _identifier(reporter_id, "reporter_id"))
        object.__setattr__(
            self,
            "sequence_family_id",
            _identifier(sequence_family_id, "sequence_family_id"),
        )
        object.__setattr__(self, "source_manifest_ids", manifests)
        object.__setattr__(self, "injection_reference_seconds", injection)
        object.__setattr__(self, "saturation_threshold_intensity", threshold)
        object.__setattr__(
            self, "intensity_unit_id", _identifier(intensity_unit_id, "intensity_unit_id")
        )
        object.__setattr__(
            self,
            "trace_id",
            canonical_fingerprint(
                {
                    "kind": "fluorescence-time-trace",
                    "case": case,
                    "well": (
                        identity.experiment_id,
                        identity.plate_id,
                        identity.well_id,
                        identity.preparation_id,
                        identity.replicate_id,
                    ),
                    "constructs": constructs,
                    "concentrations_molar": concentrations.tolist(),
                    "temperature_kelvin": temperature,
                    "condition": condition_id,
                    "chemistry_direction": chemistry_direction,
                    "reporter": reporter_id,
                    "family": sequence_family_id,
                    "source_manifests": manifests,
                    "clock_and_signal": array_tree_fingerprint(
                        (times, values, saturated)
                    )["sha256"],
                    "injection_reference_seconds": injection,
                    "saturation_threshold_intensity": threshold,
                    "intensity_unit": intensity_unit_id,
                }
            ),
        )


_TRACE_DATA_FIELDS = (
    "time_seconds",
    "intensity",
    "saturation_mask",
    "initial_concentrations_molar",
)
_TRACE_META_FIELDS = (
    "case_id",
    "identity",
    "has_saturated_observations",
    "construct_ids",
    "temperature_kelvin",
    "condition_id",
    "chemistry_direction",
    "reporter_id",
    "sequence_family_id",
    "source_manifest_ids",
    "injection_reference_seconds",
    "saturation_threshold_intensity",
    "intensity_unit_id",
    "trace_id",
)


def _flatten_fluorescence_trace(trace):
    return (
        tuple(getattr(trace, name) for name in _TRACE_DATA_FIELDS),
        tuple(getattr(trace, name) for name in _TRACE_META_FIELDS),
    )


def _unflatten_fluorescence_trace(metadata, arrays):
    trace = object.__new__(FluorescenceTimeTrace)
    for name, value in zip(_TRACE_DATA_FIELDS, arrays, strict=True):
        object.__setattr__(trace, name, value)
    for name, value in zip(_TRACE_META_FIELDS, metadata, strict=True):
        object.__setattr__(trace, name, value)
    return trace


jax.tree_util.register_pytree_node(
    FluorescenceTimeTrace,
    _flatten_fluorescence_trace,
    _unflatten_fluorescence_trace,
)


@dataclass(frozen=True, slots=True)
class StrandDisplacementSourceMember:
    """One exact archive member or caller-supplied file."""

    relationship: str
    path: str
    manifest: ReferenceArtifactManifest

    def __post_init__(self):
        if self.relationship not in (
            "raw-workbook",
            "plate-layout",
            "processed-csv",
            "readme",
        ):
            raise ValueError("Unsupported strand-displacement source relationship.")
        object.__setattr__(self, "path", _member_path(self.path, "source member path"))
        if not isinstance(self.manifest, ReferenceArtifactManifest):
            raise TypeError("Source members require ReferenceArtifactManifest values.")
        if self.manifest.artifact_name != self.path:
            raise ValueError(
                "Member manifest artifact_name must equal its exact member path."
            )


@dataclass(frozen=True, slots=True, init=False)
class StrandDisplacementWellManifest:
    """Caller declaration connecting one source sample to physical semantics."""

    sample_label: str
    source_description: str
    case_id: str
    preparation_id: str
    replicate_id: str
    reporter_id: str
    sequence_family_id: str
    construct_ids: tuple[str, ...]
    initial_concentrations_molar: tuple[float, ...]
    role: str
    parent_case_ids: tuple[str, ...]
    saturation_threshold_intensity: float | None

    def __init__(
        self,
        sample_label: str,
        source_description: str,
        case_id: str,
        preparation_id: str,
        replicate_id: str,
        reporter_id: str,
        sequence_family_id: str,
        construct_ids: Sequence[str],
        initial_concentrations: Sequence[float],
        *,
        concentration_unit: UnitDefinition,
        role: str,
        parent_case_ids: Sequence[str] = (),
        saturation_threshold_intensity: float | None = None,
    ):
        constructs = _identifiers(tuple(construct_ids), "construct_ids")
        raw = np.asarray(tuple(initial_concentrations), dtype=float)
        if (
            raw.shape != (len(constructs),)
            or np.any(~np.isfinite(raw))
            or np.any(raw < 0.0)
        ):
            raise ValueError(
                "Each construct needs one finite non-negative concentration."
            )
        factor = conversion_factor(concentration_unit, MOLE_PER_CUBIC_METER) / 1000.0
        concentrations = tuple(float(value) for value in raw * factor)
        if role not in _ROLES:
            raise ValueError(
                "A well role must use the fixed scientific campaign vocabulary."
            )
        parents = tuple(parent_case_ids)
        if parents:
            parents = _identifiers(parents, "parent_case_ids")
        threshold = (
            None
            if saturation_threshold_intensity is None
            else _positive(
                saturation_threshold_intensity, "saturation_threshold_intensity"
            )
        )
        object.__setattr__(
            self, "sample_label", _identifier(sample_label, "sample_label")
        )
        object.__setattr__(
            self,
            "source_description",
            _identifier(source_description, "source_description"),
        )
        object.__setattr__(self, "case_id", _identifier(case_id, "case_id"))
        object.__setattr__(
            self, "preparation_id", _identifier(preparation_id, "preparation_id")
        )
        object.__setattr__(
            self, "replicate_id", _identifier(replicate_id, "replicate_id")
        )
        object.__setattr__(self, "reporter_id", _identifier(reporter_id, "reporter_id"))
        object.__setattr__(
            self,
            "sequence_family_id",
            _identifier(sequence_family_id, "sequence_family_id"),
        )
        object.__setattr__(self, "construct_ids", constructs)
        object.__setattr__(self, "initial_concentrations_molar", concentrations)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "parent_case_ids", parents)
        object.__setattr__(self, "saturation_threshold_intensity", threshold)


@dataclass(frozen=True, slots=True)
class StrandDisplacementSourceManifest:
    """Source and interpretation manifest for one exact raw plate workbook."""

    archive: ReferenceArtifactManifest | None
    raw_workbook: StrandDisplacementSourceMember
    plate_layout: StrandDisplacementSourceMember
    processed_csv: StrandDisplacementSourceMember | None
    experiment_id: str
    plate_id: str
    condition_id: str
    chemistry_direction: str
    temperature_kelvin: float
    wells: tuple[StrandDisplacementWellManifest, ...]
    record_id: str = _ZENODO_RECORD_ID
    version: str = _ZENODO_VERSION
    doi: str = _ZENODO_DOI
    license_id: str = _LICENSE_ID
    readme: StrandDisplacementSourceMember | None = None
    manifest_id: str = field(init=False)

    def __post_init__(self):
        if (self.record_id, self.version, self.doi, self.license_id) != (
            _ZENODO_RECORD_ID,
            _ZENODO_VERSION,
            _ZENODO_DOI,
            _LICENSE_ID,
        ):
            raise ValueError(
                "Strand-displacement sources must pin Zenodo 10090783 version 1 CC-BY-4.0."
            )
        if self.archive is not None and not isinstance(
            self.archive, ReferenceArtifactManifest
        ):
            raise TypeError("archive must be a ReferenceArtifactManifest or None.")
        if self.raw_workbook.relationship != "raw-workbook":
            raise ValueError("raw_workbook must declare the raw-workbook relationship.")
        if self.plate_layout.relationship != "plate-layout":
            raise ValueError("plate_layout must declare the plate-layout relationship.")
        if (
            self.processed_csv is not None
            and self.processed_csv.relationship != "processed-csv"
        ):
            raise ValueError("processed_csv must declare the processed-csv relationship.")
        if self.readme is not None and self.readme.relationship != "readme":
            raise ValueError("readme must declare the readme relationship.")
        if self.chemistry_direction not in _DIRECTIONS:
            raise ValueError("Unsupported chemistry direction for Zenodo 10090783.")
        _identifier(self.experiment_id, "experiment_id")
        _identifier(self.plate_id, "plate_id")
        _identifier(self.condition_id, "condition_id")
        temperature = _positive(self.temperature_kelvin, "temperature_kelvin")
        object.__setattr__(self, "temperature_kelvin", temperature)
        if not isinstance(self.wells, tuple) or not self.wells:
            raise ValueError("A source manifest requires a non-empty tuple of wells.")
        if any(
            not isinstance(well, StrandDisplacementWellManifest) for well in self.wells
        ):
            raise TypeError("wells must contain StrandDisplacementWellManifest values.")
        for coordinate, label in (
            ((well.sample_label for well in self.wells), "sample labels"),
            ((well.case_id for well in self.wells), "case IDs"),
        ):
            values = tuple(coordinate)
            if len(set(values)) != len(values):
                raise ValueError(f"Source {label} must be unique.")
        members = (
            (self.raw_workbook, self.plate_layout)
            + (() if self.processed_csv is None else (self.processed_csv,))
            + (() if self.readme is None else (self.readme,))
        )
        object.__setattr__(
            self,
            "manifest_id",
            canonical_fingerprint(
                {
                    "kind": "strand-displacement-source-manifest",
                    "record": self.record_id,
                    "version": self.version,
                    "doi": self.doi,
                    "license": self.license_id,
                    "archive": None if self.archive is None else self.archive.manifest_id,
                    "members": [
                        (member.relationship, member.path, member.manifest.manifest_id)
                        for member in members
                    ],
                    "experiment": self.experiment_id,
                    "plate": self.plate_id,
                    "condition": self.condition_id,
                    "chemistry_direction": self.chemistry_direction,
                    "temperature_kelvin": temperature,
                    "wells": [
                        {
                            "sample": well.sample_label,
                            "source_description": well.source_description,
                            "case": well.case_id,
                            "preparation": well.preparation_id,
                            "replicate": well.replicate_id,
                            "reporter": well.reporter_id,
                            "family": well.sequence_family_id,
                            "constructs": well.construct_ids,
                            "concentrations_molar": well.initial_concentrations_molar,
                            "role": well.role,
                            "parents": well.parent_case_ids,
                            "saturation_threshold": well.saturation_threshold_intensity,
                        }
                        for well in self.wells
                    ],
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class StrandDisplacementAdmission:
    """Verified raw traces and source identities; no fitted notebook values."""

    source: StrandDisplacementSourceManifest
    traces: tuple[FluorescenceTimeTrace, ...]
    admission_id: str


@dataclass(frozen=True, slots=True)
class StrandDisplacementCohort:
    """Cross-archive campaign with family, plate and preparation leakage barriers."""

    traces: tuple[FluorescenceTimeTrace, ...]
    campaign: ScientificCampaign
    source_manifest_ids: tuple[str, ...]
    cohort_id: str


def _xlsx_sheet_rows(
    content: bytes, sheet_name: str, /
) -> tuple[dict[str, str | None], ...]:
    """Parse scalar cells from one exact OOXML worksheet without formula evaluation."""

    with zipfile.ZipFile(io.BytesIO(content)) as workbook:
        names = tuple(item.filename for item in workbook.infolist())
        if len(set(names)) != len(names):
            raise ValueError("Workbook contains duplicate ZIP member names.")
        required = {"xl/workbook.xml", "xl/_rels/workbook.xml.rels"}
        if not required <= set(names):
            raise ValueError(
                "Source workbook is missing required OOXML workbook records."
            )
        shared: tuple[str, ...] = ()
        if "xl/sharedStrings.xml" in names:
            root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared = tuple(
                "".join(node.text or "" for node in item.iter(f"{{{_XLSX_MAIN_NS}}}t"))
                for item in root.findall(f"{{{_XLSX_MAIN_NS}}}si")
            )
        book = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
        relations = ElementTree.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relations.findall(f"{{{_XLSX_PACKAGE_REL_NS}}}Relationship")
        }
        sheets = book.find(f"{{{_XLSX_MAIN_NS}}}sheets")
        if sheets is None:
            raise ValueError("Source workbook has no worksheet table.")
        matches = tuple(
            sheet for sheet in sheets if sheet.attrib.get("name") == sheet_name
        )
        if len(matches) != 1:
            raise ValueError(f"Workbook must contain exactly one {sheet_name!r} sheet.")
        relation_id = matches[0].attrib[f"{{{_XLSX_REL_NS}}}id"]
        target = targets[relation_id]
        target = target[1:] if target.startswith("/") else f"xl/{target}"
        normalized = PurePosixPath(target)
        if ".." in normalized.parts or normalized.as_posix() not in names:
            raise ValueError(
                "Worksheet relationship does not identify a canonical member."
            )
        worksheet = ElementTree.fromstring(workbook.read(normalized.as_posix()))
        data = worksheet.find(f".//{{{_XLSX_MAIN_NS}}}sheetData")
        if data is None:
            raise ValueError(f"Worksheet {sheet_name!r} has no cell data.")
        result: list[dict[str, str | None]] = []
        for row in data.findall(f"{{{_XLSX_MAIN_NS}}}row"):
            values: dict[str, str | None] = {}
            for cell in row.findall(f"{{{_XLSX_MAIN_NS}}}c"):
                coordinate = cell.attrib["r"]
                column = "".join(
                    character for character in coordinate if character.isalpha()
                )
                kind = cell.attrib.get("t")
                scalar = cell.find(f"{{{_XLSX_MAIN_NS}}}v")
                if kind == "inlineStr":
                    inline = cell.find(f"{{{_XLSX_MAIN_NS}}}is")
                    value = (
                        None
                        if inline is None
                        else "".join(
                            node.text or ""
                            for node in inline.iter(f"{{{_XLSX_MAIN_NS}}}t")
                        )
                    )
                else:
                    value = None if scalar is None else scalar.text
                    if kind == "s" and value is not None:
                        index = int(value)
                        if not 0 <= index < len(shared):
                            raise ValueError("Workbook shared-string index is invalid.")
                        value = shared[index]
                values[column] = value
            if values:
                result.append(values)
        return tuple(result)


def _source_manifest_ids(source: StrandDisplacementSourceManifest, /) -> tuple[str, ...]:
    manifests = [
        source.raw_workbook.manifest.manifest_id,
        source.plate_layout.manifest.manifest_id,
    ]
    if source.archive is not None:
        manifests.append(source.archive.manifest_id)
    if source.processed_csv is not None:
        manifests.append(source.processed_csv.manifest.manifest_id)
    if source.readme is not None:
        manifests.append(source.readme.manifest.manifest_id)
    return tuple(sorted(manifests))


def _parse_admitted_workbooks(
    raw_content: bytes,
    plate_content: bytes,
    source: StrandDisplacementSourceManifest,
    /,
    *,
    saturation_markers: Sequence[str],
    injection_marker: str,
) -> StrandDisplacementAdmission:
    markers = frozenset(
        _identifier(value, "saturation marker") for value in saturation_markers
    )
    if not markers:
        raise ValueError("At least one exact instrument saturation marker is required.")
    injection = _identifier(injection_marker, "injection marker")
    plate_rows = _xlsx_sheet_rows(plate_content, "Sheet1")
    plate_labels: dict[str, str] = {}
    for row in plate_rows:
        sample, description = row.get("A"), row.get("B")
        if sample is None or not sample.startswith("Sample X"):
            continue
        if description is None:
            raise ValueError(f"Plate setup sample {sample!r} has no description.")
        if sample in plate_labels:
            raise ValueError(f"Plate setup sample {sample!r} is duplicated.")
        plate_labels[sample] = description

    raw_rows = _xlsx_sheet_rows(raw_content, "Table All Cycles")
    header_rows = tuple(row for row in raw_rows if row.get("B") == "Time [s]")
    if len(header_rows) != 1:
        raise ValueError("Raw workbook requires exactly one Time [s] row.")
    header = header_rows[0]
    measurement_rows = tuple(
        row
        for row in raw_rows
        if row.get("A") is not None and row.get("B", "").startswith("Sample X")
    )
    if not measurement_rows:
        raise ValueError("Raw workbook contains no well/sample fluorescence rows.")
    wells = tuple(row["A"] for row in measurement_rows)
    samples = tuple(row["B"] for row in measurement_rows)
    if len(set(wells)) != len(wells):
        raise ValueError("Raw workbook contains duplicate plate/well identities.")
    if len(set(samples)) != len(samples):
        raise ValueError("Raw workbook contains duplicate sample identities.")
    specifications = {well.sample_label: well for well in source.wells}
    if set(samples) != set(specifications):
        missing = sorted(set(samples) - set(specifications))
        absent = sorted(set(specifications) - set(samples))
        raise ValueError(
            f"Well manifest must exactly cover the raw workbook; missing={missing}, absent={absent}."
        )
    if set(samples) != set(plate_labels):
        raise ValueError(
            "Plate setup and raw workbook sample identities must agree exactly."
        )

    def column_number(name: str) -> int:
        value = 0
        for character in name:
            value = 26 * value + ord(character) - ord("A") + 1
        return value

    data_columns = tuple(
        sorted(
            (column for column in header if column not in ("A", "B")),
            key=column_number,
        )
    )
    if not data_columns or any(header[column] is None for column in data_columns):
        raise ValueError("Raw workbook time columns must be complete.")
    marker_columns_by_row = tuple(
        frozenset(column for column in data_columns if row.get(column) == injection)
        for row in measurement_rows
    )
    marker_columns = marker_columns_by_row[0]
    if any(columns != marker_columns for columns in marker_columns_by_row):
        raise ValueError(
            "Injection markers must occupy the same source column in every well."
        )
    if len(marker_columns) != 1:
        raise ValueError(
            "Raw kinetic traces require exactly one injection marker per workbook part."
        )
    kept_columns = tuple(
        column for column in data_columns if column not in marker_columns
    )
    source_times = np.asarray(
        [float(header[column]) for column in kept_columns], dtype=float
    )
    marker_column = next(iter(marker_columns))
    marker_position = data_columns.index(marker_column)
    if marker_position == 0:
        raise ValueError("Injection marker requires a preceding source time sample.")
    injection_reference = float(header[data_columns[marker_position - 1]])
    source_times = source_times - injection_reference
    if np.any(~np.isfinite(source_times)) or np.any(np.diff(source_times) <= 0.0):
        raise ValueError(
            "Source time order is not strictly increasing after injection alignment."
        )

    manifest_ids = _source_manifest_ids(source)
    traces: list[FluorescenceTimeTrace] = []
    for row in measurement_rows:
        sample = row["B"]
        specification = specifications[sample]
        if plate_labels[sample] != specification.source_description:
            raise ValueError(
                f"Plate description for {sample!r} disagrees with its manifest."
            )
        values: list[float] = []
        saturated: list[bool] = []
        for column in kept_columns:
            scalar = row.get(column)
            if scalar in markers:
                values.append(float("nan"))
                saturated.append(True)
            else:
                if scalar is None:
                    raise ValueError(
                        f"Well {row['A']!r} has a missing fluorescence value."
                    )
                number = float(scalar)
                if not math.isfinite(number):
                    raise ValueError(
                        f"Well {row['A']!r} has a non-finite fluorescence value."
                    )
                values.append(number)
                saturated.append(False)
        traces.append(
            FluorescenceTimeTrace(
                specification.case_id,
                PlateWellIdentity(
                    source.experiment_id,
                    source.plate_id,
                    row["A"],
                    specification.preparation_id,
                    specification.replicate_id,
                ),
                source_times,
                values,
                saturated,
                specification.construct_ids,
                specification.initial_concentrations_molar,
                temperature_kelvin=source.temperature_kelvin,
                condition_id=source.condition_id,
                chemistry_direction=source.chemistry_direction,
                reporter_id=specification.reporter_id,
                sequence_family_id=specification.sequence_family_id,
                source_manifest_ids=manifest_ids,
                injection_reference_seconds=injection_reference,
                saturation_threshold_intensity=specification.saturation_threshold_intensity,
            )
        )
    traces_ = tuple(traces)
    return StrandDisplacementAdmission(
        source,
        traces_,
        canonical_fingerprint(
            {
                "kind": "strand-displacement-admission",
                "source": source.manifest_id,
                "traces": [trace.trace_id for trace in traces_],
            }
        ),
    )


def admit_strand_displacement_archive(
    archive_path: str | Path,
    source: StrandDisplacementSourceManifest,
    /,
    *,
    requested_use: Mapping[str, bool],
    saturation_markers: Sequence[str] = ("OVER",),
    injection_marker: str = "Inj.",
) -> StrandDisplacementAdmission:
    """Verify and admit exact raw/layout members from a caller-supplied archive."""

    if source.archive is None:
        raise ValueError("Archive admission requires an exact archive manifest.")
    use = _requested_use(requested_use)
    content = Path(archive_path).read_bytes()
    _require_manifest_bytes(content, source.archive, use)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        infos = archive.infolist()
        names = tuple(info.filename for info in infos)
        if len(set(names)) != len(names):
            raise ValueError("Source archive contains duplicate member paths.")
        by_name = {info.filename: info for info in infos}
        members = (
            (source.raw_workbook, source.plate_layout)
            + (() if source.processed_csv is None else (source.processed_csv,))
            + (() if source.readme is None else (source.readme,))
        )
        admitted: dict[str, bytes] = {}
        for member in members:
            info = by_name.get(member.path)
            if info is None or info.is_dir() or info.flag_bits & 0x1:
                raise ValueError(
                    f"Required source member {member.path!r} is absent or unreadable."
                )
            if info.file_size != member.manifest.size_bytes:
                raise ValueError(
                    f"Archive member {member.path!r} size disagrees with its manifest."
                )
            payload = archive.read(info)
            _require_manifest_bytes(payload, member.manifest, use)
            admitted[member.relationship] = payload
    return _parse_admitted_workbooks(
        admitted["raw-workbook"],
        admitted["plate-layout"],
        source,
        saturation_markers=saturation_markers,
        injection_marker=injection_marker,
    )


def admit_strand_displacement_paths(
    raw_workbook_path: str | Path,
    plate_layout_path: str | Path,
    source: StrandDisplacementSourceManifest,
    /,
    *,
    requested_use: Mapping[str, bool],
    processed_csv_path: str | Path | None = None,
    readme_path: str | Path | None = None,
    saturation_markers: Sequence[str] = ("OVER",),
    injection_marker: str = "Inj.",
) -> StrandDisplacementAdmission:
    """Verify and admit exact caller-supplied standalone source files."""

    if source.archive is not None:
        raise ValueError(
            "Path admission requires a source manifest without an archive pin."
        )
    use = _requested_use(requested_use)
    raw = Path(raw_workbook_path).read_bytes()
    plate = Path(plate_layout_path).read_bytes()
    _require_manifest_bytes(raw, source.raw_workbook.manifest, use)
    _require_manifest_bytes(plate, source.plate_layout.manifest, use)
    if (source.processed_csv is None) != (processed_csv_path is None):
        raise ValueError(
            "Processed CSV path and manifest must either both be present or both absent."
        )
    if source.processed_csv is not None:
        processed = Path(processed_csv_path).read_bytes()
        _require_manifest_bytes(processed, source.processed_csv.manifest, use)
    if (source.readme is None) != (readme_path is None):
        raise ValueError(
            "README path and manifest must either both be present or both absent."
        )
    if source.readme is not None:
        readme = Path(readme_path).read_bytes()
        _require_manifest_bytes(readme, source.readme.manifest, use)
    return _parse_admitted_workbooks(
        raw,
        plate,
        source,
        saturation_markers=saturation_markers,
        injection_marker=injection_marker,
    )


def _csv_json_identifiers(value: str, name: str, /) -> tuple[str, ...]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must be an exact JSON string array.") from error
    if not isinstance(decoded, list):
        raise ValueError(f"{name} must be an exact JSON string array.")
    return _identifiers(tuple(decoded), name)


def _optional_csv_float(value: str, name: str, /) -> float | None:
    if value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be blank or finite.")
    return result


def admit_prepared_strand_displacement_csv(
    csv_path: str | Path,
    source: StrandDisplacementSourceManifest,
    prepared_trace_manifest: ReferenceArtifactManifest,
    /,
    *,
    requested_use: Mapping[str, bool],
) -> StrandDisplacementAdmission:
    """Admit a caller-prepared, exact long-form trace CSV with raw-source lineage.

    The schema is deliberately fixed to canonical seconds, molar concentrations,
    and explicit observed/right-censored saturation states.  Every repeated
    physical field is checked against ``source``; notebook-derived values are
    therefore unable to silently replace the declared raw-plate semantics.
    """

    if not isinstance(source, StrandDisplacementSourceManifest):
        raise TypeError("source must be a StrandDisplacementSourceManifest.")
    if not isinstance(prepared_trace_manifest, ReferenceArtifactManifest):
        raise TypeError("prepared_trace_manifest must be a ReferenceArtifactManifest.")
    raw_source_ids = _source_manifest_ids(source)
    required_lineage = {source.manifest_id, *raw_source_ids}
    if not required_lineage <= set(prepared_trace_manifest.lineage_ids):
        raise ValueError(
            "Prepared trace manifest lineage must include the source manifest "
            "and every retained raw source artifact."
        )
    content = Path(csv_path).read_bytes()
    _require_manifest_bytes(
        content, prepared_trace_manifest, _requested_use(requested_use)
    )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Prepared trace CSV must be exact UTF-8.") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != _PREPARED_TRACE_COLUMNS:
        raise ValueError(
            "Prepared trace CSV columns and order must exactly match "
            + ",".join(_PREPARED_TRACE_COLUMNS)
            + "."
        )
    rows = tuple(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError("Prepared trace CSV requires complete fixed-width rows.")

    grouped: dict[str, list[dict[str, str]]] = {}
    case_order: list[str] = []
    closed: set[str] = set()
    previous_case: str | None = None
    for row in rows:
        case_id = _identifier(row["case_id"], "CSV case_id")
        if case_id != previous_case:
            if case_id in closed:
                raise ValueError("Prepared trace rows for each case must be contiguous.")
            if previous_case is not None:
                closed.add(previous_case)
            case_order.append(case_id)
            previous_case = case_id
        grouped.setdefault(case_id, []).append(row)

    wells = {well.case_id: well for well in source.wells}
    if tuple(case_order) != tuple(well.case_id for well in source.wells):
        raise ValueError(
            "Prepared trace case order must exactly match the source well order."
        )
    trace_source_ids = tuple(
        sorted((*raw_source_ids, prepared_trace_manifest.manifest_id))
    )
    traces: list[FluorescenceTimeTrace] = []
    seen_wells: set[str] = set()
    for case_id in case_order:
        case_rows = grouped[case_id]
        first = case_rows[0]
        well = wells[case_id]
        constant_columns = tuple(
            column
            for column in _PREPARED_TRACE_COLUMNS
            if column not in ("time_seconds", "intensity", "saturation_state")
        )
        if any(
            any(row[column] != first[column] for column in constant_columns)
            for row in case_rows[1:]
        ):
            raise ValueError(f"Prepared trace metadata changes within case {case_id!r}.")
        expected_strings = {
            "case_id": well.case_id,
            "sample_label": well.sample_label,
            "experiment_id": source.experiment_id,
            "plate_id": source.plate_id,
            "preparation_id": well.preparation_id,
            "replicate_id": well.replicate_id,
            "reporter_id": well.reporter_id,
            "sequence_family_id": well.sequence_family_id,
            "condition_id": source.condition_id,
            "chemistry_direction": source.chemistry_direction,
        }
        if any(first[name] != expected for name, expected in expected_strings.items()):
            raise ValueError(
                f"Prepared trace metadata disagrees with source case {case_id!r}."
            )
        well_id = _identifier(first["well_id"], "CSV well_id")
        if well_id in seen_wells:
            raise ValueError(
                "Prepared trace CSV contains duplicate physical well identities."
            )
        seen_wells.add(well_id)
        if float(first["temperature_kelvin"]) != source.temperature_kelvin:
            raise ValueError(
                "Prepared trace temperature disagrees with its source manifest."
            )
        if (
            _csv_json_identifiers(first["construct_ids_json"], "CSV construct_ids")
            != well.construct_ids
        ):
            raise ValueError("Prepared trace constructs disagree with their source well.")
        concentrations = np.asarray(
            json.loads(first["initial_concentrations_molar_json"]), dtype=float
        )
        if concentrations.shape != (len(well.construct_ids),) or not np.array_equal(
            concentrations, np.asarray(well.initial_concentrations_molar)
        ):
            raise ValueError(
                "Prepared trace molar concentrations disagree with their source well."
            )
        if (
            tuple(
                sorted(
                    _csv_json_identifiers(
                        first["source_manifest_ids_json"], "CSV source_manifest_ids"
                    )
                )
            )
            != raw_source_ids
        ):
            raise ValueError("Prepared trace source lineage is incomplete or unexpected.")
        injection = _optional_csv_float(
            first["injection_reference_seconds"], "CSV injection_reference_seconds"
        )
        threshold = _optional_csv_float(
            first["saturation_threshold_intensity"],
            "CSV saturation_threshold_intensity",
        )
        if threshold != well.saturation_threshold_intensity:
            raise ValueError(
                "Prepared trace saturation threshold disagrees with its source well."
            )
        times: list[float] = []
        intensities: list[float] = []
        saturated: list[bool] = []
        for row in case_rows:
            time = float(row["time_seconds"])
            if not math.isfinite(time):
                raise ValueError("Prepared trace times must be finite.")
            state = row["saturation_state"]
            if state == "observed":
                value = float(row["intensity"])
                if not math.isfinite(value):
                    raise ValueError("Observed prepared fluorescence must be finite.")
                is_saturated = False
            elif state == "right-censored":
                if row["intensity"] != "" or threshold is None:
                    raise ValueError(
                        "Right-censored fluorescence must be blank and have a threshold."
                    )
                value = float("nan")
                is_saturated = True
            else:
                raise ValueError(
                    "Prepared saturation_state must be observed or right-censored."
                )
            times.append(time)
            intensities.append(value)
            saturated.append(is_saturated)
        traces.append(
            FluorescenceTimeTrace(
                case_id,
                PlateWellIdentity(
                    source.experiment_id,
                    source.plate_id,
                    well_id,
                    well.preparation_id,
                    well.replicate_id,
                ),
                times,
                intensities,
                saturated,
                well.construct_ids,
                well.initial_concentrations_molar,
                temperature_kelvin=source.temperature_kelvin,
                condition_id=source.condition_id,
                chemistry_direction=source.chemistry_direction,
                reporter_id=well.reporter_id,
                sequence_family_id=well.sequence_family_id,
                source_manifest_ids=trace_source_ids,
                injection_reference_seconds=injection,
                saturation_threshold_intensity=threshold,
                intensity_unit_id=_identifier(
                    first["intensity_unit_id"], "CSV intensity_unit_id"
                ),
            )
        )
    traces_ = tuple(traces)
    return StrandDisplacementAdmission(
        source,
        traces_,
        canonical_fingerprint(
            {
                "kind": "prepared-strand-displacement-csv-admission",
                "source": source.manifest_id,
                "prepared_trace_manifest": prepared_trace_manifest.manifest_id,
                "traces": [trace.trace_id for trace in traces_],
            }
        ),
    )


def prepare_strand_displacement_cohort(
    admissions: Sequence[StrandDisplacementAdmission],
    /,
    *,
    preprocessing_source_ids: Sequence[str] = (),
    criteria_ids: Sequence[str] = (),
) -> StrandDisplacementCohort:
    """Freeze a cross-source campaign after family/preparation/plate leakage checks."""

    values = tuple(admissions)
    if not values or any(
        not isinstance(value, StrandDisplacementAdmission) for value in values
    ):
        raise ValueError("A cohort requires admitted strand-displacement sources.")
    traces = tuple(trace for admission in values for trace in admission.traces)
    if len({trace.case_id for trace in traces}) != len(traces):
        raise ValueError("Cohort case IDs must be unique across admitted sources.")
    if len({trace.trace_id for trace in traces}) != len(traces):
        raise ValueError("Cohort trace identities must be unique.")
    physical_wells = tuple(
        (
            trace.identity.experiment_id,
            trace.identity.plate_id,
            trace.identity.well_id,
        )
        for trace in traces
    )
    if len(set(physical_wells)) != len(physical_wells):
        raise ValueError("Cohort physical plate/well identities must be unique.")
    well_by_case = {
        well.case_id: well for admission in values for well in admission.source.wells
    }
    role_by_case = {case_id: well.role for case_id, well in well_by_case.items()}
    for coordinate_name, coordinates in (
        ("sequence family", (trace.sequence_family_id for trace in traces)),
        ("preparation", (trace.identity.preparation_id for trace in traces)),
        ("plate", (trace.identity.plate_id for trace in traces)),
    ):
        roles_by_coordinate: dict[str, set[str]] = {}
        for trace, coordinate in zip(traces, coordinates, strict=True):
            roles_by_coordinate.setdefault(coordinate, set()).add(
                role_by_case[trace.case_id]
            )
        leaking = sorted(
            key for key, roles in roles_by_coordinate.items() if len(roles) != 1
        )
        if leaking:
            raise ValueError(
                f"{coordinate_name.title()} groups cross campaign roles: {leaking}."
            )
    cases = tuple(
        ScientificCase(
            case_id=trace.case_id,
            independent_unit_id=trace.sequence_family_id,
            construct_id=canonical_fingerprint(trace.construct_ids),
            condition_id=trace.condition_id,
            preparation_id=trace.identity.preparation_id,
            batch_id=trace.identity.plate_id,
            source_manifest_ids=trace.source_manifest_ids,
            parent_case_ids=well_by_case[trace.case_id].parent_case_ids,
        )
        for trace in traces
    )
    roles = tuple(
        CampaignRole(
            name,
            tuple(
                sorted(case_id for case_id, role in role_by_case.items() if role == name)
            ),
        )
        for name in (
            "calibration",
            "model_selection",
            "interval_calibration",
            "locked_evaluation",
            "prospective",
        )
        if any(role == name for role in role_by_case.values())
    )
    campaign = ScientificCampaign(
        cases,
        roles,
        preprocessing_source_ids=tuple(preprocessing_source_ids),
        criteria_ids=tuple(criteria_ids),
    )
    source_ids = tuple(sorted(admission.source.manifest_id for admission in values))
    return StrandDisplacementCohort(
        traces,
        campaign,
        source_ids,
        canonical_fingerprint(
            {
                "kind": "strand-displacement-cohort",
                "campaign": campaign.campaign_id,
                "sources": source_ids,
                "traces": [trace.trace_id for trace in traces],
            }
        ),
    )


__all__ = [
    "FluorescenceTimeTrace",
    "PlateWellIdentity",
    "StrandDisplacementAdmission",
    "StrandDisplacementCohort",
    "StrandDisplacementSourceManifest",
    "StrandDisplacementSourceMember",
    "StrandDisplacementWellManifest",
    "admit_strand_displacement_archive",
    "admit_prepared_strand_displacement_csv",
    "admit_strand_displacement_paths",
    "prepare_strand_displacement_cohort",
]
