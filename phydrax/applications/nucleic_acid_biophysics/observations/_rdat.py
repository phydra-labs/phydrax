# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Narrow RDAT 0.34 processed-reactivity adapter; no reads/alignment tooling."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ...._fingerprint import canonical_fingerprint
from ....artifacts import ScientificArtifactEnvelope
from ....qualification import ReferenceArtifactManifest
from .._construct import NucleicAcidConstruct, NucleotideKey
from ._chemical_mapping import ChemicalMappingCondition, ChemicalMappingObservation


@dataclass(frozen=True, slots=True)
class ProcessedRDATEntry:
    observation: ChemicalMappingObservation
    annotations: tuple[str, ...]
    declared_structure: str | None
    source_sequence_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProcessedRDAT:
    raw_bytes: bytes
    source: ScientificArtifactEnvelope
    entries: tuple[ProcessedRDATEntry, ...]
    records: tuple[tuple[str, str], ...]


def import_processed_rdat(
    payload: bytes,
    source: ReferenceArtifactManifest,
    *,
    requested_use,
    error_semantics: str,
    replicate_ids=None,
) -> ProcessedRDAT:
    """Read one supported version with exact source checksum and explicit SD law.

    REACTIVITY_ERROR semantics are admitted by the caller as standard-deviation;
    the adapter cannot discover a covariance or replicate grouping absent in the
    source. Every ANNOTATION_DATA row is its own construct/output, never silently
    pooled as a replicate. Designed STRUCTURE is retained as a declared hypothesis,
    not an experimentally determined pairing graph. Unsupported fields refuse.
    """
    if (
        not isinstance(payload, bytes)
        or source.checksum_algorithm != "sha256"
        or hashlib.sha256(payload).hexdigest() != source.checksum
        or len(payload) != source.size_bytes
    ):
        raise ValueError(
            "RDAT admission requires exact bytes matching its SHA256 rights manifest."
        )
    source.require_rights(**requested_use)
    if error_semantics != "standard-deviation":
        raise ValueError(
            "This profile needs explicit standard-deviation semantics for REACTIVITY_ERROR."
        )
    records = []
    header, data, errors, annotations = {}, {}, {}, {}
    repeated = {"COMMENT", "ANNOTATION"}
    supported = {
        "RDAT_VERSION",
        "NAME",
        "SEQUENCE",
        "STRUCTURE",
        "OFFSET",
        "SEQPOS",
        *repeated,
    }
    for line in payload.decode("utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = line.split(None, 1)
        if len(tokens) != 2:
            raise ValueError("RDAT records require a tag and payload.")
        tag, body = tokens[0], tokens[1].strip()
        records.append((tag, body))
        if ":" in tag:
            kind, index = tag.split(":", 1)
            if not index.isdigit() or int(index) < 1:
                raise ValueError("RDAT data row IDs must be positive integers.")
            target = (
                data
                if kind == "REACTIVITY"
                else errors
                if kind == "REACTIVITY_ERROR"
                else annotations
                if kind == "ANNOTATION_DATA"
                else None
            )
            if target is None or int(index) in target:
                raise ValueError("Unsupported or duplicate RDAT data field.")
            target[int(index)] = body
        elif tag in supported:
            if tag in header and tag not in repeated:
                raise ValueError("Duplicate singleton RDAT field.")
            header[tag] = (
                body if tag not in repeated else header.get(tag, "") + "\t" + body
            )
        else:
            raise ValueError(
                "Unsupported RDAT field; a new explicit adapter profile is required."
            )
    if (
        header.get("RDAT_VERSION") != "0.34"
        or not all(key in header for key in ("NAME", "SEQUENCE", "OFFSET", "SEQPOS"))
        or not data
        or set(data) != set(errors)
    ):
        raise ValueError(
            "RDAT 0.34 requires sequence, explicit offset, selected positions and aligned reactivity errors."
        )
    if set(annotations) - set(data):
        raise ValueError("RDAT annotations refer to missing data rows.")
    global_annotations = tuple(
        item for item in header.get("ANNOTATION", "").split("\t") if item
    )
    offset = int(header["OFFSET"])
    labels = tuple(header["SEQPOS"].split())
    if any(re.fullmatch(r"[ACGU]-?\d+", label) is None for label in labels):
        raise ValueError(
            "Supported SEQPOS profile is canonical RNA base plus integer source position."
        )
    positions = tuple(int(label[1:]) - offset - 1 for label in labels)
    if len(set(positions)) != len(positions):
        raise ValueError("SEQPOS source positions must be unique.")
    entries = []
    for index in sorted(data):
        local = tuple(
            item.strip()
            for item in annotations.get(index, "").split("\t")
            if item.strip()
        )
        sequences = [
            item[len("sequence:") :] for item in local if item.startswith("sequence:")
        ]
        structures = [
            item[len("structure:") :] for item in local if item.startswith("structure:")
        ]
        if len(sequences) > 1 or len(structures) > 1:
            raise ValueError("Ambiguous per-construct sequence/structure annotations.")
        sequence = sequences[0] if sequences else header["SEQUENCE"]
        construct = NucleicAcidConstruct(
            (f"rdat-{index}",), (sequence,), ("RNA",), (False,)
        )
        if any(position < 0 or position >= len(sequence) for position in positions):
            raise ValueError("SEQPOS mapping lies outside a declared construct sequence.")
        all_annotations = global_annotations + local
        modifiers = [
            item[len("modifier:") :]
            for item in all_annotations
            if item.startswith("modifier:")
        ]
        if len(set(modifiers)) != 1:
            raise ValueError(
                "This profile requires exactly one declared chemical reagent."
            )
        condition_annotations = tuple(
            item
            for item in all_annotations
            if not item.startswith(("sequence:", "structure:", "EteRNA:", "processing:"))
        )
        condition_id = canonical_fingerprint({"annotations": condition_annotations})
        condition = ChemicalMappingCondition(condition_id, condition_annotations, None)
        replicate = (
            f"unpooled-row-{index}" if replicate_ids is None else replicate_ids[index]
        )
        observation = ChemicalMappingObservation(
            construct,
            tuple(
                NucleotideKey(construct.strand_ids[0], position) for position in positions
            ),
            tuple(float(x) for x in data[index].split()),
            tuple(float(x) for x in errors[index].split()),
            reagent=modifiers[0],
            condition=condition,
            replicate_id=replicate,
            preprocessing=tuple(
                item for item in all_annotations if item.startswith("processing:")
            ),
            source=source,
            requested_use=requested_use,
        )
        structure = structures[0] if structures else header.get("STRUCTURE")
        if structure is not None and len(structure) != len(sequence):
            raise ValueError(
                "Declared structural hypothesis must align with its variant sequence."
            )
        entries.append(ProcessedRDATEntry(observation, local, structure, labels))
    envelope = ScientificArtifactEnvelope(
        artifact_kind="processed-RDAT-source",
        content_digest=source.checksum,
        producer="external-RMDB",
        producer_version="RDAT-0.34",
        build_id="source-bytes",
        license_id=source.license_id,
        resource_id=source.manifest_id,
        status="complete",
        parent_artifact_ids=source.lineage_ids,
    )
    return ProcessedRDAT(payload, envelope, tuple(entries), tuple(records))


__all__ = ["ProcessedRDATEntry", "ProcessedRDAT", "import_processed_rdat"]
