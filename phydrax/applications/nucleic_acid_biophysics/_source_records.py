# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Biological binding of explicitly selected native neutral structure records."""

from __future__ import annotations

from dataclasses import dataclass

from ...atomistic.interchange._structure_records import PDBAtomRecord
from ...units import ANGSTROM
from ._binding import NucleotideAtomMapping
from ._hypotheses import NucleicStructureHypothesis


@dataclass(frozen=True, slots=True)
class NucleicRecordHypothesis:
    source_records: tuple[PDBAtomRecord, ...]
    selected_records: tuple[PDBAtomRecord, ...]
    hypothesis: NucleicStructureHypothesis


def nucleic_hypothesis_from_pdb_records(
    source_records,
    selected_records,
    *,
    construct,
    record_assignments,
    source,
    rights,
    requested_use,
    image_policy,
):
    """Bind native reader output with exact record-ID→(nucleotide key,atom ID).

    The caller uses native read_pdb_atom_records/select_pdb_model. Source author
    numbering, insertion codes, chain aliases, model and alternate coordinates
    remain in source_records. Only canonical unmodified residue names are
    accepted; atom naming, stable IDs and selected model are never inferred.
    """
    raw, selected = tuple(source_records), tuple(selected_records)
    if (
        not raw
        or not selected
        or any(not isinstance(row, PDBAtomRecord) for row in raw + selected)
    ):
        raise TypeError("Binding requires native PDBAtomRecord values.")
    raw_by_id = {row.record_id: row for row in raw}
    if len(raw_by_id) != len(raw) or any(
        raw_by_id.get(row.record_id) != row for row in selected
    ):
        raise ValueError("Selected records must retain exact raw source rows.")
    if len({row.model_id for row in selected}) != 1 or set(record_assignments) != {
        row.record_id for row in selected
    }:
        raise ValueError(
            "One explicitly selected model and total record assignments are required."
        )
    chemistry = {
        key: (base, polymer)
        for strand, sequence, polymer in zip(
            construct.strand_ids,
            construct.sequences,
            construct.polymer_types,
            strict=True,
        )
        for key, base in zip(
            (key for key in construct.nucleotide_keys if key.strand_id == strand),
            sequence,
            strict=True,
        )
    }
    ids, keys, names, positions = [], [], [], []
    for row in selected:
        key, atom_id = record_assignments[row.record_id]
        if key not in chemistry:
            raise ValueError("Source assignment lies outside the nucleotide construct.")
        base, polymer = chemistry[key]
        if row.residue_name != ("D" + base if polymer == "DNA" else base):
            raise ValueError(
                "PDB residue chemistry does not match the declared canonical construct."
            )
        ids.append(atom_id)
        keys.append(key)
        names.append(row.atom_name)
        positions.append(row.position)
    mapping = NucleotideAtomMapping(construct, tuple(ids), tuple(keys), tuple(names))
    hypothesis = NucleicStructureHypothesis(
        mapping,
        positions,
        ANGSTROM,
        source,
        rights,
        requested_use=requested_use,
        image_policy=image_policy,
    )
    return NucleicRecordHypothesis(raw, selected, hypothesis)


__all__ = ["NucleicRecordHypothesis", "nucleic_hypothesis_from_pdb_records"]
