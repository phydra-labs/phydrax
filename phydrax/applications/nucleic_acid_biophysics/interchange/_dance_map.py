# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Admission of caller-supplied ShapeMapper parsed-mutation files for DANCE-MaP.

This module does not align reads, download sources, infer references, or run
DanceMapper. It retains ShapeMapper mapped depth separately from effective depth
and records every parsed row, including mapping categories excluded from analysis.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import numpy as np

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....qualification import ReferenceArtifactManifest
from ..observations._mutation_profiles import MutationProfileBatch, MutationProfileCase


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{owner} must be a non-empty canonical identifier.")
    return value


@dataclass(frozen=True, slots=True, init=False)
class DanceMapFile:
    """One externally produced parsed-mutation file and its exact assay lineage."""

    path: str
    source: ReferenceArtifactManifest
    nucleotide_ids: tuple[str, ...]
    construct_id: str
    condition_id: str
    preparation_id: str
    batch_id: str
    replicate_id: str
    reagent_id: str
    protocol_id: str
    upstream_tool: str
    upstream_version: str
    reference_sequence_id: str
    parent_case_ids: tuple[str, ...]
    file_id: str

    def __init__(
        self,
        path: str | Path,
        source: ReferenceArtifactManifest,
        /,
        *,
        nucleotide_ids: tuple[str, ...],
        construct_id: str,
        condition_id: str,
        preparation_id: str,
        batch_id: str,
        replicate_id: str,
        reagent_id: str,
        protocol_id: str,
        upstream_tool: str,
        upstream_version: str,
        reference_sequence_id: str,
        parent_case_ids: tuple[str, ...] = (),
    ):
        if not isinstance(source, ReferenceArtifactManifest):
            raise TypeError("source must be a ReferenceArtifactManifest.")
        path_ = str(Path(path).expanduser().resolve())
        if not isinstance(nucleotide_ids, tuple) or not nucleotide_ids:
            raise TypeError("nucleotide_ids must be a non-empty tuple.")
        nucleotides = tuple(
            _identifier(value, "nucleotide ID") for value in nucleotide_ids
        )
        if len(set(nucleotides)) != len(nucleotides):
            raise ValueError("nucleotide_ids must be unique.")
        values = tuple(
            _identifier(value, name)
            for value, name in (
                (construct_id, "construct ID"),
                (condition_id, "condition ID"),
                (preparation_id, "preparation ID"),
                (batch_id, "batch ID"),
                (replicate_id, "replicate ID"),
                (reagent_id, "reagent ID"),
                (protocol_id, "protocol ID"),
                (upstream_tool, "upstream mapping tool"),
                (upstream_version, "upstream mapping tool version"),
                (reference_sequence_id, "reference sequence ID"),
            )
        )
        if not isinstance(parent_case_ids, tuple) or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in parent_case_ids
        ):
            raise TypeError("parent_case_ids must be a tuple of canonical identifiers.")
        if len(set(parent_case_ids)) != len(parent_case_ids):
            raise ValueError("parent_case_ids must be unique.")
        object.__setattr__(self, "path", path_)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "nucleotide_ids", nucleotides)
        for name, value in zip(
            (
                "construct_id",
                "condition_id",
                "preparation_id",
                "batch_id",
                "replicate_id",
                "reagent_id",
                "protocol_id",
                "upstream_tool",
                "upstream_version",
                "reference_sequence_id",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "parent_case_ids", parent_case_ids)
        object.__setattr__(
            self,
            "file_id",
            canonical_fingerprint(
                {
                    "kind": "dance-map-source-file",
                    "source": source.manifest_id,
                    "nucleotides": nucleotides,
                    "case": values[:7],
                    "upstream": values[7:],
                    "parents": parent_case_ids,
                }
            ),
        )


class DanceMapAdmission(StrictModule, NonTrainableState):
    """Parsed batch plus retained per-category admission counts and lineage."""

    batch: MutationProfileBatch
    files: tuple[DanceMapFile, ...] = eqx.field(static=True)
    mapping_category_ids: tuple[str, ...] = eqx.field(static=True)
    category_counts: tuple[int, ...] = eqx.field(static=True)
    included_mapping_categories: tuple[str, ...] = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)

    def __init__(
        self,
        batch: MutationProfileBatch,
        files: tuple[DanceMapFile, ...],
        category_counts: tuple[int, ...],
        included_mapping_categories: tuple[str, ...],
    ):
        if not isinstance(batch, MutationProfileBatch):
            raise TypeError("batch must be a MutationProfileBatch.")
        if (
            not isinstance(files, tuple)
            or not files
            or any(not isinstance(value, DanceMapFile) for value in files)
        ):
            raise TypeError("files must contain the exact admitted DanceMapFile records.")
        if tuple(value.source.manifest_id for value in files) != batch.source_ids:
            raise ValueError(
                "DANCE-MaP files must align with batch source-manifest order."
            )
        included = tuple(included_mapping_categories)
        if (
            not included
            or len(set(included)) != len(included)
            or not set(included).issubset(batch.mapping_category_ids)
        ):
            raise ValueError(
                "Included mapping categories must be unique admitted categories."
            )
        if len(category_counts) != len(batch.mapping_category_ids) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in category_counts
        ):
            raise ValueError("category_counts must align with mapping category IDs.")
        actual_counts = tuple(
            int(np.sum(np.asarray(batch.mapping_category_index) == index))
            for index in range(len(batch.mapping_category_ids))
        )
        expected_included = np.asarray(
            tuple(
                batch.mapping_category_ids[int(index)] in included
                for index in np.asarray(batch.mapping_category_index)
            )
        )
        if tuple(category_counts) != actual_counts or not np.array_equal(
            expected_included, np.asarray(batch.mapping_included)
        ):
            raise ValueError(
                "DANCE-MaP category counts or inclusion mask disagree with the batch."
            )
        self.batch = batch
        self.files = files
        self.mapping_category_ids = batch.mapping_category_ids
        self.category_counts = tuple(category_counts)
        self.included_mapping_categories = included
        self.admission_id = canonical_fingerprint(
            {
                "kind": "dance-map-admission",
                "batch": batch.batch_fingerprint,
                "files": [value.file_id for value in files],
                "categories": list(
                    zip(batch.mapping_category_ids, category_counts, strict=True)
                ),
                "included": included_mapping_categories,
            }
        )


@dataclass(frozen=True, slots=True)
class _ParsedRead:
    mutation: np.ndarray
    observed: np.ndarray
    coverage: np.ndarray
    category: str
    source_row: int


def import_dance_map_files(
    files,
    /,
    *,
    included_mapping_categories: tuple[str, ...] = ("INCLUDED",),
    requested_use=None,
) -> DanceMapAdmission:
    """Verify and import caller paths using ShapeMapper ``parsed.mut`` columns.

    Expected columns are read type, read name, inclusive left/right reference
    coordinates, mapping category, primer-pair index, mapped-depth bit vector,
    effective-depth bit vector, and mutation-count bit vector. Additional detailed
    mutation columns are retained in the governed source artifact but are not used
    by this Bernoulli-profile adapter.
    """
    records = tuple(files)
    if not records or any(not isinstance(value, DanceMapFile) for value in records):
        raise TypeError("files must be a non-empty sequence of DanceMapFile records.")
    included = tuple(
        _identifier(value, "included mapping category")
        for value in included_mapping_categories
    )
    if not included or len(set(included)) != len(included):
        raise ValueError("Included mapping categories must be non-empty and unique.")
    nucleotide_ids = records[0].nucleotide_ids
    if any(value.nucleotide_ids != nucleotide_ids for value in records):
        raise ValueError(
            "One rectangular batch requires the same explicit nucleotide mapping for every file."
        )
    use = {} if requested_use is None else dict(requested_use)
    payload_identities = tuple(
        (
            record.source.checksum_algorithm,
            record.source.checksum,
            record.source.size_bytes,
        )
        for record in records
    )
    if len(set(payload_identities)) != len(payload_identities):
        raise ValueError(
            "Identical parsed-mutation payloads cannot be relabeled as distinct "
            "source manifests or preparations."
        )

    parsed_by_file: list[tuple[_ParsedRead, ...]] = []
    cases: list[MutationProfileCase] = []
    for record in records:
        record.source.require_rights(**use)
        source_path = Path(record.path)
        if not source_path.is_file():
            raise ValueError(
                f"Caller-supplied DANCE-MaP path is not a file: {source_path}."
            )
        payload = source_path.read_bytes()
        digest = hashlib.new(record.source.checksum_algorithm, payload).hexdigest()
        if digest != record.source.checksum or len(payload) != record.source.size_bytes:
            raise ValueError(
                "Caller-supplied parsed-mutation bytes do not match their source manifest."
            )
        parsed_by_file.append(_parse_shapemapper_mut(payload, len(nucleotide_ids)))
        cases.append(
            MutationProfileCase(
                construct_id=record.construct_id,
                condition_id=record.condition_id,
                preparation_id=record.preparation_id,
                batch_id=record.batch_id,
                replicate_id=record.replicate_id,
                reagent_id=record.reagent_id,
                protocol_id=record.protocol_id,
                source_manifest_ids=(record.source.manifest_id,),
                parent_case_ids=record.parent_case_ids,
            )
        )
    if sum(len(values) for values in parsed_by_file) == 0:
        raise ValueError("Parsed-mutation admission contains no mapped-read rows.")

    def vocabulary(values):
        return tuple(dict.fromkeys(values))

    construct_ids = vocabulary(record.construct_id for record in records)
    condition_ids = vocabulary(record.condition_id for record in records)
    replicate_ids = vocabulary(record.replicate_id for record in records)
    preparation_ids = vocabulary(record.preparation_id for record in records)
    batch_ids = vocabulary(record.batch_id for record in records)
    reagent_ids = vocabulary(record.reagent_id for record in records)
    protocol_ids = vocabulary(record.protocol_id for record in records)
    source_ids = tuple(record.source.manifest_id for record in records)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError(
            "Each source manifest may occur once per admission; combine its rows into one file record."
        )
    categories = vocabulary(read.category for values in parsed_by_file for read in values)
    unknown_included = set(included) - set(categories)
    if unknown_included:
        raise ValueError(
            "Included mapping categories are absent from the admitted source rows: "
            f"{tuple(sorted(unknown_included))!r}."
        )

    mutation, observed, coverage = [], [], []
    metadata = [[] for _ in range(10)]
    source_rows, selected = [], []
    for file_index, (record, case, reads) in enumerate(
        zip(records, cases, parsed_by_file, strict=True)
    ):
        fixed = (
            construct_ids.index(record.construct_id),
            condition_ids.index(record.condition_id),
            replicate_ids.index(record.replicate_id),
            preparation_ids.index(record.preparation_id),
            batch_ids.index(record.batch_id),
            reagent_ids.index(record.reagent_id),
            protocol_ids.index(record.protocol_id),
            file_index,
            cases.index(case),
        )
        for read in reads:
            mutation.append(read.mutation)
            observed.append(read.observed)
            coverage.append(read.coverage)
            for values, index in zip(metadata[:9], fixed, strict=True):
                values.append(index)
            metadata[9].append(categories.index(read.category))
            source_rows.append(read.source_row)
            selected.append(read.category in included)

    batch = MutationProfileBatch(
        np.stack(mutation),
        np.stack(observed),
        np.stack(coverage),
        np.asarray(metadata[0]),
        np.asarray(metadata[1]),
        np.asarray(metadata[2]),
        source_ids,
        nucleotide_ids=nucleotide_ids,
        preparation_index=np.asarray(metadata[3]),
        batch_index=np.asarray(metadata[4]),
        reagent_index=np.asarray(metadata[5]),
        protocol_index=np.asarray(metadata[6]),
        source_index=np.asarray(metadata[7]),
        case_index=np.asarray(metadata[8]),
        source_row_index=np.asarray(source_rows),
        mapping_category_index=np.asarray(metadata[9]),
        mapping_included=np.asarray(selected),
        construct_ids=construct_ids,
        condition_ids=condition_ids,
        replicate_ids=replicate_ids,
        preparation_ids=preparation_ids,
        batch_ids=batch_ids,
        reagent_ids=reagent_ids,
        protocol_ids=protocol_ids,
        mapping_category_ids=categories,
        cases=tuple(cases),
        sources=tuple(record.source for record in records),
    )
    counts = tuple(
        sum(read.category == category for values in parsed_by_file for read in values)
        for category in categories
    )
    return DanceMapAdmission(batch, records, counts, included)


def _parse_shapemapper_mut(
    payload: bytes, nucleotide_count: int
) -> tuple[_ParsedRead, ...]:
    text = payload.decode("utf-8")
    result: list[_ParsedRead] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line or raw_line.startswith("#"):
            continue
        fields = raw_line.split("\t")
        if len(fields) < 9:
            raise ValueError(
                f"ShapeMapper parsed-mutation row {line_number} has fewer than nine columns."
            )
        left, right = int(fields[2]), int(fields[3])
        if left < 0 or right < left or right >= nucleotide_count:
            raise ValueError(
                f"ShapeMapper row {line_number} maps outside the declared reference support."
            )
        width = right - left + 1
        vectors = fields[6:9]
        if any(len(value) != width or set(value) - {"0", "1"} for value in vectors):
            raise ValueError(
                f"ShapeMapper row {line_number} depth/mutation columns must be aligned binary vectors."
            )
        mapped = np.zeros((nucleotide_count,), dtype=np.int8)
        effective = np.zeros((nucleotide_count,), dtype=bool)
        mutation = np.zeros((nucleotide_count,), dtype=np.int8)
        mapped[left : right + 1] = np.fromiter(vectors[0], dtype=np.int8)
        effective[left : right + 1] = np.fromiter(vectors[1], dtype=np.int8).astype(bool)
        mutation[left : right + 1] = np.fromiter(vectors[2], dtype=np.int8)
        if np.any(effective & (mapped == 0)) or np.any((mutation == 1) & ~effective):
            raise ValueError(
                f"ShapeMapper row {line_number} violates mapped/effective/mutation nesting."
            )
        result.append(
            _ParsedRead(
                mutation,
                effective,
                mapped,
                _identifier(fields[4], f"mapping category on row {line_number}"),
                line_number,
            )
        )
    return tuple(result)


__all__ = ["DanceMapAdmission", "DanceMapFile", "import_dance_map_files"]
