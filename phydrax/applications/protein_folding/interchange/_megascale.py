#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Offline admission of exact Tsuboyama/MegaScale source tables.

Nothing in this module downloads data.  The caller supplies the bytes on disk and
an exact :class:`ReferenceArtifactManifest`; admission verifies both before any
scientific row is interpreted.  Derived figure tables remain labelled as such and
never acquire invented replicate uncertainty or raw-proteolysis lineage.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ...._fingerprint import canonical_fingerprint
from ....qualification import (
    CampaignRole,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
)
from .._construct import ProteinConstruct


MEGASCALE_RECORD_ID = "zenodo-7992926"
MEGASCALE_RECORD_VERSION = "v2_230420"
MEGASCALE_DOI = "10.5281/zenodo.7992926"
MEGASCALE_LICENSE_ID = "CC-BY-4.0"
MEGASCALE_DATA_TABLES_ARCHIVE_SHA256 = (
    "69f6d5a68ba961759879ab60b3f168b8fad1bce0b0b810ac490fe85001515890"
)
MEGASCALE_DATA_TABLES_ARCHIVE_SIZE = 24_125_970
MEGASCALE_FIGURE5_MEMBER = "Data_tables_for_figs/dG_non_redundant_natural_Fig5.csv"

_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_SET = frozenset(_AA)
_MUTATION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
_CENSORING = frozenset(("none", "lower", "upper", "interval", "unknown"))
_SIGN_CONVENTIONS = frozenset(("positive-is-stabilizing", "positive-is-destabilizing"))
_CAMPAIGN_ROLES = (
    "calibration",
    "model_selection",
    "interval_calibration",
    "locked_evaluation",
    "prospective",
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string.")
    return value


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of strings.")
    result = tuple(_identifier(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values.")
    return result


def _finite(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _optional_positive(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive when provided.")
    return result


def _source_fields(row: Mapping[str, str], headers: Sequence[str], /):
    return tuple((name, row[name]) for name in headers)


def _processed_stability_value(
    raw: str,
    censoring: Literal["none", "lower", "upper", "interval", "unknown"],
    name: str,
    /,
) -> float:
    """Parse an exact reported value or one-sided assay bound without clipping."""
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} must retain a non-empty source value.")
    if censoring == "lower":
        if not raw.startswith(">") or raw.startswith(">="):
            raise ValueError(f"{name} lower censoring requires an exact '>bound' value.")
        return _finite(float(raw[1:]), name)
    if censoring == "upper":
        if not raw.startswith("<") or raw.startswith("<="):
            raise ValueError(f"{name} upper censoring requires an exact '<bound' value.")
        return _finite(float(raw[1:]), name)
    if raw.startswith(("<", ">")):
        raise ValueError(
            f"{name} declares {censoring!r} censoring but contains a one-sided bound."
        )
    return _finite(float(raw), name)


def parse_mutation_code(
    mutation_code: str,
    /,
) -> tuple[tuple[str, int, str], ...]:
    """Parse explicit one-based point substitutions without guessing aliases."""
    code = _identifier(mutation_code, "mutation_code")
    if code == "WT":
        return ()
    substitutions: list[tuple[str, int, str]] = []
    positions: set[int] = set()
    for token in code.split(":"):
        match = _MUTATION.fullmatch(token)
        if match is None:
            raise ValueError(
                "Mutation codes must be WT or colon-separated one-based point "
                "substitutions such as A12V or A12V:G18D."
            )
        wild_type, position_text, mutant = match.groups()
        position = int(position_text)
        if position < 1 or position in positions or wild_type == mutant:
            raise ValueError(
                "Mutation positions must be positive and unique, and substitutions "
                "must change residue identity."
            )
        positions.add(position)
        substitutions.append((wild_type, position, mutant))
    substitutions.sort(key=lambda item: item[1])
    return tuple(substitutions)


def apply_mutation_code(sequence: str, mutation_code: str, /) -> str:
    """Apply a mutation only after exact WT residue and position validation."""
    if not isinstance(sequence, str) or not sequence or set(sequence) - _AA_SET:
        raise ValueError("sequence must contain uppercase canonical amino acids.")
    values = list(sequence)
    for wild_type, position, mutant in parse_mutation_code(mutation_code):
        index = position - 1
        if index >= len(values) or values[index] != wild_type:
            raise ValueError(
                f"Mutation {wild_type}{position}{mutant} does not match the exact "
                "WT construct sequence."
            )
        values[index] = mutant
    return "".join(values)


def convert_stability_sign(
    value: float,
    source: Literal["positive-is-stabilizing", "positive-is-destabilizing"],
    target: Literal["positive-is-stabilizing", "positive-is-destabilizing"],
    /,
) -> float:
    """Convert a declared sign convention reversibly; no implicit convention exists."""
    result = _finite(value, "stability value")
    if source not in _SIGN_CONVENTIONS or target not in _SIGN_CONVENTIONS:
        raise ValueError("Both source and target sign conventions must be explicit.")
    return result if source == target else -result


def convert_stability_censoring(
    censoring: Literal["none", "lower", "upper", "interval", "unknown"],
    bounds: tuple[float, float] | None,
    source: Literal["positive-is-stabilizing", "positive-is-destabilizing"],
    target: Literal["positive-is-stabilizing", "positive-is-destabilizing"],
    /,
) -> tuple[
    Literal["none", "lower", "upper", "interval", "unknown"],
    tuple[float, float] | None,
]:
    """Convert censor direction and interval bounds with an energy sign change."""
    if censoring not in _CENSORING:
        raise ValueError("censoring must be none/lower/upper/interval/unknown.")
    if source not in _SIGN_CONVENTIONS or target not in _SIGN_CONVENTIONS:
        raise ValueError("Both source and target sign conventions must be explicit.")
    if censoring == "interval":
        if bounds is None:
            raise ValueError("Interval censoring requires exact bounds.")
        lower, upper = (_finite(value, "censoring bound") for value in bounds)
        if lower >= upper:
            raise ValueError("Censoring lower bound must be below upper bound.")
    elif bounds is not None:
        raise ValueError("Only interval censoring accepts two censoring bounds.")
    if source == target:
        return censoring, bounds
    converted = {
        "none": "none",
        "lower": "upper",
        "upper": "lower",
        "interval": "interval",
        "unknown": "unknown",
    }[censoring]
    converted_bounds = None if bounds is None else (-float(bounds[1]), -float(bounds[0]))
    return converted, converted_bounds


@dataclass(frozen=True, slots=True)
class ProteinStabilityMeasurement:
    """One exact construct/mutation/condition measurement with source lineage."""

    measurement_id: str
    domain_id: str
    domain_family_id: str
    background_id: str
    sequence: str
    mutation_code: str
    mutation_order: int
    assay_channel: str
    library_id: str
    condition_id: str
    value_kcal_per_mol: float
    standard_error_kcal_per_mol: float | None
    censoring: Literal["none", "lower", "upper", "interval", "unknown"]
    quality_flags: tuple[str, ...]
    source_manifest_id: str
    observable: Literal["delta_g", "delta_delta_g", "thermodynamic_coupling"]
    sign_convention: Literal["positive-is-stabilizing", "positive-is-destabilizing"]
    source_mutant_sequence: str
    shared_wt_id: str
    source_fields: tuple[tuple[str, str], ...]
    inference_lineage: tuple[str, ...]
    nominal_model_temperature_kelvin: float | None = None
    experimental_temperature_kelvin: float | None = None
    censoring_bounds_kcal_per_mol: tuple[float, float] | None = None
    uncertainty_source_manifest_id: str | None = None
    source_manifest: ReferenceArtifactManifest | None = None
    uncertainty_source_manifest: ReferenceArtifactManifest | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.measurement_id, "measurement_id"),
            (self.domain_id, "domain_id"),
            (self.domain_family_id, "domain_family_id"),
            (self.background_id, "background_id"),
            (self.assay_channel, "assay_channel"),
            (self.library_id, "library_id"),
            (self.condition_id, "condition_id"),
            (self.source_manifest_id, "source_manifest_id"),
            (self.shared_wt_id, "shared_wt_id"),
        ):
            _identifier(value, name)
        substitutions = parse_mutation_code(self.mutation_code)
        if self.mutation_order != len(substitutions):
            raise ValueError("mutation_order must equal the parsed substitution count.")
        mutant_sequence = apply_mutation_code(self.sequence, self.mutation_code)
        if mutant_sequence != self.source_mutant_sequence:
            raise ValueError(
                "The source mutant sequence does not equal the exact mutation applied "
                "to the admitted WT construct."
            )
        _finite(self.value_kcal_per_mol, "value_kcal_per_mol")
        _optional_positive(
            self.standard_error_kcal_per_mol,
            "standard_error_kcal_per_mol",
        )
        if self.uncertainty_source_manifest_id is not None:
            _identifier(
                self.uncertainty_source_manifest_id,
                "uncertainty_source_manifest_id",
            )
        if self.source_manifest is not None and (
            not isinstance(self.source_manifest, ReferenceArtifactManifest)
            or self.source_manifest.manifest_id != self.source_manifest_id
        ):
            raise ValueError("Measurement source manifest identity is inconsistent.")
        if self.uncertainty_source_manifest is not None and (
            not isinstance(self.uncertainty_source_manifest, ReferenceArtifactManifest)
            or self.uncertainty_source_manifest.manifest_id
            != self.uncertainty_source_manifest_id
        ):
            raise ValueError("Measurement uncertainty manifest identity is inconsistent.")
        if self.censoring not in _CENSORING:
            raise ValueError("censoring must be none/lower/upper/interval/unknown.")
        if self.sign_convention not in _SIGN_CONVENTIONS:
            raise ValueError("sign_convention is unsupported.")
        if self.observable not in (
            "delta_g",
            "delta_delta_g",
            "thermodynamic_coupling",
        ):
            raise ValueError("observable is unsupported.")
        _identifiers(self.quality_flags, "quality flag")
        _identifiers(self.inference_lineage, "inference lineage")
        names = tuple(name for name, _ in self.source_fields)
        if not names or len(set(names)) != len(names):
            raise ValueError("source_fields must preserve one unique source column each.")
        for name, value in self.source_fields:
            _identifier(name, "source field name")
            if not isinstance(value, str):
                raise TypeError("source field values must remain unmodified strings.")
        model_temperature = _optional_positive(
            self.nominal_model_temperature_kelvin,
            "nominal_model_temperature_kelvin",
        )
        experimental_temperature = _optional_positive(
            self.experimental_temperature_kelvin,
            "experimental_temperature_kelvin",
        )
        if self.censoring == "interval":
            if self.censoring_bounds_kcal_per_mol is None:
                raise ValueError("Interval censoring requires exact lower/upper bounds.")
            lower, upper = self.censoring_bounds_kcal_per_mol
            if not _finite(lower, "censoring lower bound") < _finite(
                upper, "censoring upper bound"
            ):
                raise ValueError("Censoring lower bound must be below upper bound.")
        elif self.censoring_bounds_kcal_per_mol is not None:
            raise ValueError("Only interval censoring accepts two censoring bounds.")
        object.__setattr__(self, "quality_flags", tuple(self.quality_flags))
        object.__setattr__(self, "inference_lineage", tuple(self.inference_lineage))
        object.__setattr__(self, "source_fields", tuple(self.source_fields))
        object.__setattr__(self, "nominal_model_temperature_kelvin", model_temperature)
        object.__setattr__(
            self, "experimental_temperature_kelvin", experimental_temperature
        )

    @property
    def independent_group_id(self) -> str:
        """Return the sequence-clustered family unit shared by every background."""
        return canonical_fingerprint(
            {
                "kind": "protein-stability-independent-group",
                "domain_family_id": self.domain_family_id,
            }
        )

    @property
    def pair_unit_id(self) -> str | None:
        substitutions = parse_mutation_code(self.mutation_code)
        if len(substitutions) != 2:
            return None
        return canonical_fingerprint(
            {
                "kind": "protein-double-mutant-pair-unit",
                "independent_group_id": self.independent_group_id,
                "positions": sorted(item[1] for item in substitutions),
            }
        )

    @property
    def record_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "protein-stability-measurement",
                "measurement_id": self.measurement_id,
                "domain_id": self.domain_id,
                "family": self.domain_family_id,
                "background": self.background_id,
                "sequence": self.sequence,
                "mutation": self.mutation_code,
                "mutation_order": self.mutation_order,
                "source_mutant_sequence": self.source_mutant_sequence,
                "channel": self.assay_channel,
                "library": self.library_id,
                "condition": self.condition_id,
                "value": float(self.value_kcal_per_mol).hex(),
                "standard_error": (
                    None
                    if self.standard_error_kcal_per_mol is None
                    else float(self.standard_error_kcal_per_mol).hex()
                ),
                "censoring": self.censoring,
                "bounds": self.censoring_bounds_kcal_per_mol,
                "quality_flags": list(self.quality_flags),
                "observable": self.observable,
                "sign": self.sign_convention,
                "shared_wt_id": self.shared_wt_id,
                "inference_lineage": list(self.inference_lineage),
                "nominal_model_temperature_kelvin": self.nominal_model_temperature_kelvin,
                "experimental_temperature_kelvin": self.experimental_temperature_kelvin,
                "source_manifest_id": self.source_manifest_id,
                "uncertainty_source_manifest_id": self.uncertainty_source_manifest_id,
                "source_fields": list(self.source_fields),
            }
        )


@dataclass(frozen=True, slots=True)
class AdmittedProteinStabilitySource:
    """Verified source content before scientific campaign roles are assigned."""

    construct_records: tuple[ProteinConstruct, ...]
    measurements: tuple[ProteinStabilityMeasurement, ...]
    source_manifest: ReferenceArtifactManifest
    source_kind: str
    source_id: str

    def __post_init__(self) -> None:
        constructs = tuple(self.construct_records)
        measurements = tuple(self.measurements)
        if not constructs or not measurements:
            raise ValueError("An admitted source needs constructs and measurements.")
        if any(not isinstance(value, ProteinConstruct) for value in constructs):
            raise TypeError("construct_records must contain ProteinConstruct values.")
        if any(
            not isinstance(value, ProteinStabilityMeasurement) for value in measurements
        ):
            raise TypeError(
                "measurements must contain ProteinStabilityMeasurement values."
            )
        if not isinstance(self.source_manifest, ReferenceArtifactManifest):
            raise TypeError("source_manifest must be a ReferenceArtifactManifest.")
        source_kind = _identifier(self.source_kind, "source_kind")
        if any(
            value.source_manifest_id != self.source_manifest.manifest_id
            for value in measurements
        ):
            raise ValueError("Every measurement must cite the exact admitted manifest.")
        ids = tuple(value.measurement_id for value in measurements)
        if len(set(ids)) != len(ids):
            raise ValueError("Measurement IDs must be unique within an admitted source.")
        by_background: dict[str, set[tuple[str, str, str]]] = {}
        for value in measurements:
            by_background.setdefault(value.background_id, set()).add(
                (value.domain_family_id, value.sequence, value.shared_wt_id)
            )
        if any(len(identities) != 1 for identities in by_background.values()):
            raise ValueError(
                "Each mutation background must retain one family, WT sequence, "
                "and shared-WT identity."
            )
        construct_sequences = {
            sequence for construct in constructs for sequence in construct.sequences
        }
        if any(value.sequence not in construct_sequences for value in measurements):
            raise ValueError("Every measurement WT sequence needs an admitted construct.")
        expected = canonical_fingerprint(
            {
                "kind": "admitted-protein-stability-source",
                "source_kind": source_kind,
                "manifest_id": self.source_manifest.manifest_id,
                "measurement_record_ids": [value.record_id for value in measurements],
                "construct_ids": [value.fingerprint() for value in constructs],
            }
        )
        if self.source_id != expected:
            raise ValueError(
                "source_id is not the content address of the admitted source."
            )
        object.__setattr__(self, "construct_records", constructs)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "source_kind", source_kind)


@dataclass(frozen=True, slots=True)
class ProteinStabilityCohort:
    construct_records: tuple[ProteinConstruct, ...]
    measurements: tuple[ProteinStabilityMeasurement, ...]
    campaign: ScientificCampaign
    source_id: str
    cohort_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "construct_records", tuple(self.construct_records))
        object.__setattr__(self, "measurements", tuple(self.measurements))
        source_id = _identifier(self.source_id, "source_id")
        object.__setattr__(self, "source_id", source_id)
        if not self.construct_records or not self.measurements:
            raise ValueError("Protein stability cohorts cannot be empty.")
        if len({item.measurement_id for item in self.measurements}) != len(
            self.measurements
        ):
            raise ValueError("Cohort measurement IDs must be unique.")
        if not isinstance(self.campaign, ScientificCampaign):
            raise TypeError("campaign must be a ScientificCampaign.")
        measurement_ids = tuple(item.measurement_id for item in self.measurements)
        if set(measurement_ids) != set(self.campaign.case_ids):
            raise ValueError("Campaign cases must cover cohort measurements exactly.")
        construct_by_sequence = {
            construct.sequences[0]: construct for construct in self.construct_records
        }
        case_by_id = {case.case_id: case for case in self.campaign.cases}
        for measurement in self.measurements:
            case = case_by_id[measurement.measurement_id]
            expected_sources = tuple(
                sorted(
                    artifact_id
                    for artifact_id in (
                        measurement.source_manifest_id,
                        measurement.uncertainty_source_manifest_id,
                    )
                    if artifact_id is not None
                )
            )
            if (
                measurement.sequence not in construct_by_sequence
                or case.independent_unit_id != measurement.independent_group_id
                or case.construct_id
                != construct_by_sequence[measurement.sequence].fingerprint()
                or case.condition_id != measurement.condition_id
                or case.source_manifest_ids != expected_sources
            ):
                raise ValueError(
                    "Campaign cases must retain each exact cohort measurement's "
                    "group, construct, condition, and admitted sources."
                )
        expected = canonical_fingerprint(
            {
                "kind": "protein-stability-cohort",
                "source_id": source_id,
                "campaign_id": self.campaign.campaign_id,
                "measurement_record_ids": sorted(
                    item.record_id for item in self.measurements
                ),
                "construct_ids": sorted(
                    construct.fingerprint() for construct in self.construct_records
                ),
            }
        )
        if self.cohort_id != expected:
            raise ValueError("cohort_id is not the content address of this cohort.")


def megascale_data_tables_archive_manifest() -> ReferenceArtifactManifest:
    """Return the pinned manifest for the verified Zenodo figure-data archive."""
    return ReferenceArtifactManifest(
        "Data_tables_for_figs.zip",
        checksum_algorithm="sha256",
        checksum=MEGASCALE_DATA_TABLES_ARCHIVE_SHA256,
        size_bytes=MEGASCALE_DATA_TABLES_ARCHIVE_SIZE,
        license_id=MEGASCALE_LICENSE_ID,
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="publicly-available-scientific-data",
        nondimensionalization={"energy_kcal_per_mol": 1.0},
        uncertainty=None,
        lineage_ids=(
            MEGASCALE_RECORD_ID,
            MEGASCALE_RECORD_VERSION,
            f"doi:{MEGASCALE_DOI}",
        ),
    )


def verify_source_payload(
    payload: bytes,
    manifest: ReferenceArtifactManifest,
    /,
    *,
    artifact_name: str,
    training_use: bool = False,
    commercial_use: bool = False,
) -> None:
    """Verify exact bytes, filename, license, and requested-use rights offline."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("Source payload must be non-empty immutable bytes.")
    if not isinstance(manifest, ReferenceArtifactManifest):
        raise TypeError("manifest must be a ReferenceArtifactManifest.")
    if manifest.artifact_name != artifact_name:
        raise ValueError("Manifest artifact_name does not match the supplied source.")
    if manifest.size_bytes != len(payload):
        raise ValueError("Source size does not match its manifest.")
    digest = hashlib.new(manifest.checksum_algorithm, payload).hexdigest()
    if digest != manifest.checksum:
        raise ValueError("Source checksum does not match its manifest.")
    if manifest.license_id.lower() != MEGASCALE_LICENSE_ID.lower():
        raise ValueError("MegaScale admission requires the pinned CC BY 4.0 license.")
    if not {
        MEGASCALE_RECORD_ID,
        MEGASCALE_RECORD_VERSION,
        f"doi:{MEGASCALE_DOI}",
    }.issubset(manifest.lineage_ids):
        raise ValueError("Manifest lacks exact MegaScale record/version/DOI lineage.")
    manifest.require_rights(
        training_use=training_use,
        commercial_use=commercial_use,
    )


def verified_megascale_archive_member(
    archive_path: str | Path,
    archive_manifest: ReferenceArtifactManifest,
    member_name: str,
    /,
    *,
    training_use: bool = False,
    commercial_use: bool = False,
) -> tuple[bytes, ReferenceArtifactManifest]:
    """Read one exact member after verifying the caller-supplied Zenodo archive."""
    path = Path(archive_path)
    payload = path.read_bytes()
    verify_source_payload(
        payload,
        archive_manifest,
        artifact_name=path.name,
        training_use=training_use,
        commercial_use=commercial_use,
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if member_name not in names or names.count(member_name) != 1:
            raise ValueError("Requested archive member must occur exactly once.")
        if member_name.endswith("/"):
            raise ValueError("Requested archive member must be a file.")
        member_payload = archive.read(member_name)
    if not member_payload:
        raise ValueError("Requested archive member is empty.")
    member_manifest = ReferenceArtifactManifest(
        member_name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(member_payload).hexdigest(),
        size_bytes=len(member_payload),
        license_id=archive_manifest.license_id,
        commercial_use_permitted=archive_manifest.commercial_use_permitted,
        redistribution_permitted=archive_manifest.redistribution_permitted,
        training_use_permitted=archive_manifest.training_use_permitted,
        export_permitted=archive_manifest.export_permitted,
        export_classification=archive_manifest.export_classification,
        nondimensionalization=dict(archive_manifest.nondimensionalization),
        uncertainty=None,
        lineage_ids=(
            archive_manifest.manifest_id,
            MEGASCALE_RECORD_ID,
            MEGASCALE_RECORD_VERSION,
            f"doi:{MEGASCALE_DOI}",
        ),
    )
    return member_payload, member_manifest


def _csv_rows(payload: bytes, /) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise ValueError("Source CSV has no header.")
    headers = tuple(reader.fieldnames)
    if not headers or any(not name or name != name.strip() for name in headers):
        raise ValueError("Source CSV headers must be non-empty and whitespace exact.")
    if len(set(headers)) != len(headers):
        raise ValueError("Source CSV headers must be unique.")
    rows = tuple(dict(row) for row in reader)
    if not rows:
        raise ValueError("Source CSV has no data rows.")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Source CSV rows do not match the exact header width.")
    return headers, rows


def _figure_table_source(
    payload: bytes,
    manifest: ReferenceArtifactManifest,
    /,
    *,
    artifact_name: str,
    family_by_domain: Mapping[str, str],
    library_id: str,
    condition_id: str,
    nominal_model_temperature_kelvin: float = 298.0,
    experimental_temperature_kelvin: float | None = 295.15,
    training_use: bool = False,
    commercial_use: bool = False,
) -> AdmittedProteinStabilitySource:
    verify_source_payload(
        payload,
        manifest,
        artifact_name=artifact_name,
        training_use=training_use,
        commercial_use=commercial_use,
    )
    if not isinstance(family_by_domain, Mapping):
        raise TypeError("family_by_domain must be an explicit source-backed mapping.")
    library = _identifier(library_id, "library_id")
    condition = _identifier(condition_id, "condition_id")
    headers, rows = _csv_rows(payload)
    metadata = ("pdb_name", "pos", "wt_aa")
    if tuple(headers[:3]) != metadata:
        raise ValueError("Figure-5 source must begin exactly with pdb_name,pos,wt_aa.")
    mutant_columns = headers[3:]
    if not mutant_columns or any(column not in _AA_SET for column in mutant_columns):
        raise ValueError("Figure-5 mutation columns must be canonical residue letters.")
    if len(set(mutant_columns)) != len(mutant_columns):
        raise ValueError("Figure-5 mutation columns must be unique.")

    by_domain: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        domain = _identifier(row["pdb_name"], "pdb_name")
        by_domain.setdefault(domain, []).append(row)
    if set(family_by_domain) != set(by_domain):
        raise ValueError(
            "family_by_domain must cover source domains exactly; family labels are "
            "never inferred from PDB names."
        )

    constructs: list[ProteinConstruct] = []
    measurements: list[ProteinStabilityMeasurement] = []
    for domain, domain_rows in sorted(by_domain.items()):
        positioned: dict[int, dict[str, str]] = {}
        for row in domain_rows:
            position_text = row["pos"]
            if not position_text.isdigit() or int(position_text) < 1:
                raise ValueError("Figure-5 positions must be positive decimal integers.")
            position = int(position_text)
            if position in positioned:
                raise ValueError("Figure-5 domain positions must be unique.")
            if row["wt_aa"] not in _AA_SET:
                raise ValueError("Figure-5 wt_aa must be a canonical residue letter.")
            positioned[position] = row
        expected = tuple(range(1, len(positioned) + 1))
        if tuple(sorted(positioned)) != expected:
            raise ValueError(
                "Figure-5 rows must provide a complete contiguous WT sequence."
            )
        sequence = "".join(positioned[position]["wt_aa"] for position in expected)
        construct = ProteinConstruct(("A",), (sequence,))
        constructs.append(construct)
        family = _identifier(family_by_domain[domain], "domain family ID")
        for position in expected:
            row = positioned[position]
            wild_type = row["wt_aa"]
            for mutant in mutant_columns:
                raw = row[mutant]
                value = _finite(float(raw), f"{domain}:{position}:{mutant}")
                code = "WT" if mutant == wild_type else f"{wild_type}{position}{mutant}"
                measurement_id = f"figure5:{domain}:{position}:{mutant}"
                measurements.append(
                    ProteinStabilityMeasurement(
                        measurement_id,
                        domain,
                        family,
                        domain,
                        sequence,
                        code,
                        0 if code == "WT" else 1,
                        "combined-derived-figure-table",
                        library,
                        condition,
                        value,
                        None,
                        "unknown",
                        (
                            "derived-figure-table",
                            "uncertainty-not-reported",
                            "censoring-not-reported",
                            "not-raw-proteolysis",
                        ),
                        manifest.manifest_id,
                        "delta_g",
                        "positive-is-stabilizing",
                        apply_mutation_code(sequence, code),
                        f"figure5:{domain}:shared-wt",
                        _source_fields(row, headers),
                        (
                            "Tsuboyama2023-figure5-derived-dG",
                            "trypsin-K50-derived",
                            "chymotrypsin-K50-derived",
                            "proteolysis-assay-inference",
                        ),
                        nominal_model_temperature_kelvin,
                        experimental_temperature_kelvin,
                        source_manifest=manifest,
                    )
                )
    source_id = canonical_fingerprint(
        {
            "kind": "admitted-protein-stability-source",
            "source_kind": "derived-figure-table",
            "manifest_id": manifest.manifest_id,
            "measurement_record_ids": [item.record_id for item in measurements],
            "construct_ids": [item.fingerprint() for item in constructs],
        }
    )
    return AdmittedProteinStabilitySource(
        tuple(constructs),
        tuple(measurements),
        manifest,
        "derived-figure-table",
        source_id,
    )


def admit_megascale_figure_table(
    path: str | Path,
    manifest: ReferenceArtifactManifest,
    /,
    *,
    family_by_domain: Mapping[str, str],
    library_id: str,
    condition_id: str,
    nominal_model_temperature_kelvin: float = 298.0,
    experimental_temperature_kelvin: float | None = 295.15,
    training_use: bool = False,
    commercial_use: bool = False,
) -> AdmittedProteinStabilitySource:
    """Admit an extracted exact Figure-5 dG matrix with caller-supplied families."""
    source_path = Path(path)
    return _figure_table_source(
        source_path.read_bytes(),
        manifest,
        artifact_name=source_path.name,
        family_by_domain=family_by_domain,
        library_id=library_id,
        condition_id=condition_id,
        nominal_model_temperature_kelvin=nominal_model_temperature_kelvin,
        experimental_temperature_kelvin=experimental_temperature_kelvin,
        training_use=training_use,
        commercial_use=commercial_use,
    )


def admit_megascale_figure_archive(
    archive_path: str | Path,
    archive_manifest: ReferenceArtifactManifest,
    /,
    *,
    family_by_domain: Mapping[str, str],
    library_id: str,
    condition_id: str,
    member_name: str = MEGASCALE_FIGURE5_MEMBER,
    nominal_model_temperature_kelvin: float = 298.0,
    experimental_temperature_kelvin: float | None = 295.15,
    training_use: bool = False,
    commercial_use: bool = False,
) -> AdmittedProteinStabilitySource:
    """Admit the exact Figure-5 member from a verified caller-owned archive."""
    payload, member_manifest = verified_megascale_archive_member(
        archive_path,
        archive_manifest,
        member_name,
        training_use=training_use,
        commercial_use=commercial_use,
    )
    return _figure_table_source(
        payload,
        member_manifest,
        artifact_name=member_name,
        family_by_domain=family_by_domain,
        library_id=library_id,
        condition_id=condition_id,
        nominal_model_temperature_kelvin=nominal_model_temperature_kelvin,
        experimental_temperature_kelvin=experimental_temperature_kelvin,
        training_use=training_use,
        commercial_use=commercial_use,
    )


def admit_megascale_processed_table(
    path: str | Path,
    manifest: ReferenceArtifactManifest,
    /,
    *,
    selected_names: Sequence[str],
    wt_sequence_by_background: Mapping[str, str],
    censoring_by_name: Mapping[
        str, Literal["none", "lower", "upper", "interval", "unknown"]
    ],
    quality_flags_by_name: Mapping[str, Sequence[str]],
    library_id: str,
    condition_id: str,
    target_sign_convention: Literal[
        "positive-is-stabilizing", "positive-is-destabilizing"
    ] = "positive-is-stabilizing",
    nominal_model_temperature_kelvin: float = 298.0,
    experimental_temperature_kelvin: float | None = 295.15,
    standard_error_by_name: Mapping[str, float | None] | None = None,
    censoring_bounds_by_name: Mapping[str, tuple[float, float] | None] | None = None,
    standard_error_manifest: ReferenceArtifactManifest | None = None,
    training_use: bool = False,
    commercial_use: bool = False,
) -> AdmittedProteinStabilitySource:
    """Admit explicitly selected rows from the official processed Dataset-3 schema.

    Selection, WT sequences, censoring, and quality flags are mandatory because the
    processed table does not make all of them safely inferable.  Missing rows are
    refused rather than silently filtered. ``ddG_ML`` is preserved as positive-is-
    stabilizing unless the caller requests the explicit reversible conversion.
    """
    source_path = Path(path)
    payload = source_path.read_bytes()
    verify_source_payload(
        payload,
        manifest,
        artifact_name=source_path.name,
        training_use=training_use,
        commercial_use=commercial_use,
    )
    headers, rows = _csv_rows(payload)
    required = (
        "name",
        "aa_seq",
        "mut_type",
        "WT_name",
        "WT_cluster",
        "ddG_ML",
    )
    if any(name not in headers for name in required):
        raise ValueError("Processed MegaScale table lacks its exact required columns.")
    selected = _identifiers(selected_names, "selected source name")
    if not selected:
        raise ValueError("selected_names must identify at least one exact source row.")
    if set(censoring_by_name) != set(selected) or set(quality_flags_by_name) != set(
        selected
    ):
        raise ValueError(
            "Censoring and quality mappings must cover selected names exactly."
        )
    standard_errors = (
        {name: None for name in selected}
        if standard_error_by_name is None
        else dict(standard_error_by_name)
    )
    censoring_bounds = (
        {name: None for name in selected}
        if censoring_bounds_by_name is None
        else dict(censoring_bounds_by_name)
    )
    if set(standard_errors) != set(selected) or set(censoring_bounds) != set(selected):
        raise ValueError(
            "Uncertainty and censoring-bound mappings must cover selected names exactly."
        )
    uncertainty_manifest_id: str | None = None
    if standard_error_manifest is not None:
        if not isinstance(standard_error_manifest, ReferenceArtifactManifest):
            raise TypeError(
                "standard_error_manifest must be a ReferenceArtifactManifest."
            )
        standard_error_manifest.require_rights(
            training_use=training_use,
            commercial_use=commercial_use,
        )
        required_lineage = {
            "protein-stability-standard-error:"
            + name
            + ":"
            + (
                "unquantified"
                if standard_errors[name] is None
                else float(standard_errors[name]).hex()
            )
            for name in selected
        }
        if not required_lineage.issubset(standard_error_manifest.lineage_ids):
            raise ValueError(
                "Standard-error manifest lineage must bind every selected source "
                "name and exact uncertainty value."
            )
        uncertainty_manifest_id = standard_error_manifest.manifest_id
    rows_by_name: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_name.setdefault(row["name"], []).append(row)
    if any(name not in rows_by_name or len(rows_by_name[name]) != 1 for name in selected):
        raise ValueError("Every selected name must identify exactly one source row.")
    selected_backgrounds = {rows_by_name[name][0]["WT_name"] for name in selected}
    if set(wt_sequence_by_background) != selected_backgrounds:
        raise ValueError(
            "wt_sequence_by_background must cover selected backgrounds exactly."
        )

    measurements: list[ProteinStabilityMeasurement] = []
    constructs: dict[str, ProteinConstruct] = {}
    for name in selected:
        row = rows_by_name[name][0]
        background = _identifier(row["WT_name"], "WT_name")
        if background not in wt_sequence_by_background:
            raise ValueError("Every selected background needs an exact WT sequence.")
        sequence = wt_sequence_by_background[background]
        code = "WT" if row["mut_type"].lower() == "wt" else row["mut_type"]
        substitutions = parse_mutation_code(code)
        source_mutant = row["aa_seq"]
        if apply_mutation_code(sequence, code) != source_mutant:
            raise ValueError(
                f"Selected row {name!r} mutation/sequence does not match its exact WT."
            )
        raw_value = _processed_stability_value(
            row["ddG_ML"], censoring_by_name[name], f"{name}.ddG_ML"
        )
        value = convert_stability_sign(
            raw_value,
            "positive-is-stabilizing",
            target_sign_convention,
        )
        source_censoring = censoring_by_name[name]
        converted_censoring, converted_bounds = convert_stability_censoring(
            source_censoring,
            censoring_bounds[name],
            "positive-is-stabilizing",
            target_sign_convention,
        )
        flags = tuple(quality_flags_by_name[name])
        if standard_errors[name] is None and "uncertainty-not-reported" not in flags:
            flags = (*flags, "uncertainty-not-reported")
        family = _identifier(row["WT_cluster"], "WT_cluster")
        constructs[background] = ProteinConstruct(("A",), (sequence,))
        measurements.append(
            ProteinStabilityMeasurement(
                name,
                background,
                family,
                background,
                sequence,
                code,
                len(substitutions),
                "combined-protease-inference",
                _identifier(library_id, "library_id"),
                _identifier(condition_id, "condition_id"),
                value,
                standard_errors[name],
                converted_censoring,
                flags,
                manifest.manifest_id,
                "delta_delta_g",
                target_sign_convention,
                source_mutant,
                f"{background}:shared-wt",
                _source_fields(row, headers),
                (
                    "Tsuboyama2023-processed-ddG_ML",
                    "trypsin-K50-derived",
                    "chymotrypsin-K50-derived",
                    "proteolysis-assay-inference",
                ),
                nominal_model_temperature_kelvin,
                experimental_temperature_kelvin,
                converted_bounds,
                uncertainty_manifest_id,
                source_manifest=manifest,
                uncertainty_source_manifest=standard_error_manifest,
            )
        )
    source_id = canonical_fingerprint(
        {
            "kind": "admitted-protein-stability-source",
            "source_kind": "processed-proteolysis-inference",
            "manifest_id": manifest.manifest_id,
            "measurement_record_ids": [item.record_id for item in measurements],
            "construct_ids": [
                constructs[key].fingerprint() for key in sorted(constructs)
            ],
        }
    )
    return AdmittedProteinStabilitySource(
        tuple(constructs[key] for key in sorted(constructs)),
        tuple(measurements),
        manifest,
        "processed-proteolysis-inference",
        source_id,
    )


def prepare_protein_stability_cohort(
    source: AdmittedProteinStabilitySource,
    role_by_independent_group: Mapping[str, str],
    /,
    *,
    preparation_id_by_measurement: Mapping[str, str],
    batch_id_by_measurement: Mapping[str, str],
    preprocessing_source_ids: Sequence[str] = (),
    criteria_ids: Sequence[str] = (),
) -> ProteinStabilityCohort:
    """Bind complete biological groups to locked campaign roles, never rows."""
    if not isinstance(source, AdmittedProteinStabilitySource):
        raise TypeError("source must be an AdmittedProteinStabilitySource.")
    groups = {measurement.independent_group_id for measurement in source.measurements}
    if set(role_by_independent_group) != groups:
        raise ValueError("Role mapping must cover independent biological groups exactly.")
    measurement_ids = {measurement.measurement_id for measurement in source.measurements}
    if (
        set(preparation_id_by_measurement) != measurement_ids
        or set(batch_id_by_measurement) != measurement_ids
    ):
        raise ValueError(
            "Preparation and batch mappings must cover measurements exactly."
        )
    unknown_roles = set(role_by_independent_group.values()) - set(_CAMPAIGN_ROLES)
    if unknown_roles:
        raise ValueError(f"Unknown scientific campaign roles: {sorted(unknown_roles)!r}.")
    construct_by_sequence = {
        construct.sequences[0]: construct for construct in source.construct_records
    }
    cases = tuple(
        ScientificCase(
            measurement.measurement_id,
            measurement.independent_group_id,
            construct_by_sequence[measurement.sequence].fingerprint(),
            measurement.condition_id,
            _identifier(
                preparation_id_by_measurement[measurement.measurement_id],
                "preparation_id",
            ),
            _identifier(batch_id_by_measurement[measurement.measurement_id], "batch_id"),
            tuple(
                artifact_id
                for artifact_id in (
                    measurement.source_manifest_id,
                    measurement.uncertainty_source_manifest_id,
                )
                if artifact_id is not None
            ),
        )
        for measurement in source.measurements
    )
    role_case_ids = {role: [] for role in _CAMPAIGN_ROLES}
    for measurement in source.measurements:
        role = role_by_independent_group[measurement.independent_group_id]
        role_case_ids[role].append(measurement.measurement_id)
    roles = tuple(
        CampaignRole(role, tuple(role_case_ids[role]))
        for role in _CAMPAIGN_ROLES
        if role_case_ids[role]
    )
    campaign = ScientificCampaign(
        cases,
        roles,
        preprocessing_source_ids=tuple(preprocessing_source_ids),
        criteria_ids=tuple(criteria_ids),
    )
    cohort_id = canonical_fingerprint(
        {
            "kind": "protein-stability-cohort",
            "source_id": source.source_id,
            "campaign_id": campaign.campaign_id,
            "measurement_record_ids": sorted(
                item.record_id for item in source.measurements
            ),
            "construct_ids": sorted(
                construct.fingerprint() for construct in source.construct_records
            ),
        }
    )
    return ProteinStabilityCohort(
        source.construct_records,
        source.measurements,
        campaign,
        source.source_id,
        cohort_id,
    )


__all__ = [
    "AdmittedProteinStabilitySource",
    "MEGASCALE_DATA_TABLES_ARCHIVE_SHA256",
    "MEGASCALE_DATA_TABLES_ARCHIVE_SIZE",
    "MEGASCALE_DOI",
    "MEGASCALE_FIGURE5_MEMBER",
    "MEGASCALE_LICENSE_ID",
    "MEGASCALE_RECORD_ID",
    "MEGASCALE_RECORD_VERSION",
    "convert_stability_censoring",
    "ProteinStabilityCohort",
    "ProteinStabilityMeasurement",
    "admit_megascale_figure_archive",
    "admit_megascale_figure_table",
    "admit_megascale_processed_table",
    "apply_mutation_code",
    "convert_stability_sign",
    "megascale_data_tables_archive_manifest",
    "parse_mutation_code",
    "prepare_protein_stability_cohort",
    "verified_megascale_archive_member",
    "verify_source_payload",
]
