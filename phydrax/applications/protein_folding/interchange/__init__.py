# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Explicit source-record adapters; external parameterization runtimes stay lazy."""

from __future__ import annotations

from dataclasses import dataclass

from ....atomistic.interchange import AtomisticInterchangeReport
from ....atomistic.interchange._structure_records import PDBAtomRecord
from ....units import ANGSTROM
from .._binding import bind_protein, PreparedProteinBinding
from .._construct import ProteinAtomKey
from .._hypotheses import ProteinSourceAtom, ProteinStructureHypothesis


_ELEMENT_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
_RESIDUE_LETTERS = dict(
    zip(
        (
            "ALA",
            "CYS",
            "ASP",
            "GLU",
            "PHE",
            "GLY",
            "HIS",
            "ILE",
            "LYS",
            "LEU",
            "MET",
            "ASN",
            "PRO",
            "GLN",
            "ARG",
            "SER",
            "THR",
            "VAL",
            "TRP",
            "TYR",
        ),
        "ACDEFGHIKLMNPQRSTVWY",
        strict=True,
    )
)


def protein_hypothesis_from_pdb_records(
    records,
    construct,
    residue_map,
    *,
    source,
    rights,
    canonical_atom_names=None,
    provider="user-supplied-pdb",
    commercial_use=False,
    training_use=False,
    export=False,
):
    """Interpret a selected neutral PDB model using an explicit author→construct map.

    ``residue_map`` keys are (author chain, author residue number, insertion
    code). ``canonical_atom_names`` is an optional record-ID→name map that
    declares any provider renaming while preserving original source bytes.
    Unselected models/alternate conformers remain in the original source artifact.
    """
    rows = tuple(records)
    if not rows or any(not isinstance(record, PDBAtomRecord) for record in rows):
        raise TypeError("Supply explicitly selected neutral PDBAtomRecord values.")
    for manifest in rights:
        manifest.require_rights(
            commercial_use=commercial_use, training_use=training_use, export=export
        )
    aliases = {} if canonical_atom_names is None else dict(canonical_atom_names)
    if set(aliases) - {record.record_id for record in rows}:
        raise ValueError("Atom-name normalization references absent source records.")
    required_residues = {
        (r.chain_id, r.author_residue_number, r.insertion_code) for r in rows
    }
    if set(residue_map) != required_residues or set(residue_map.values()) != set(
        construct.residue_keys
    ):
        raise ValueError(
            "Author-residue mapping must cover the entire declared construct without dropped material."
        )
    if len(set(residue_map.values())) != len(residue_map):
        raise ValueError("Distinct author residues cannot alias one sequence position.")
    letters = {
        key: letter
        for key, letter in zip(
            construct.residue_keys, "".join(construct.sequences), strict=True
        )
    }
    atoms = []
    for record in rows:
        if record.record_kind != "ATOM" or record.element.upper() not in _ELEMENT_NUMBERS:
            raise ValueError(
                "Only canonical elemental protein ATOM records are supported; "
                "modified/hetero material requires another chemistry profile."
            )
        residue = residue_map[
            (record.chain_id, record.author_residue_number, record.insertion_code)
        ]
        if _RESIDUE_LETTERS.get(record.residue_name) != letters[residue]:
            raise ValueError(
                "Source residue chemistry differs from the declared sequence; "
                "mutations require an explicit new realization."
            )
        atom_key = ProteinAtomKey(
            residue, aliases.get(record.record_id, record.atom_name)
        )
        atoms.append(
            ProteinSourceAtom(
                record.record_id,
                atom_key,
                record.model_id,
                record.chain_id,
                record.author_residue_number,
                record.insertion_code,
                record.alternate_location,
                record.occupancy,
                _ELEMENT_NUMBERS[record.element.upper()],
            )
        )
    return ProteinStructureHypothesis(
        construct,
        tuple(atoms),
        [record.position for record in rows],
        ANGSTROM,
        source,
        tuple(rights),
        provider=provider,
    )


@dataclass(frozen=True, slots=True)
class ProteinOpenMMBinding:
    binding: PreparedProteinBinding
    interchange_report: AtomisticInterchangeReport
    source_record_ids_by_particle: tuple[str, ...]


def bind_protein_openmm(
    hypothesis,
    chemistry,
    system,
    units,
    *,
    source_record_ids_by_particle,
    parameter_rights,
    source_id,
    cutoff,
    accept_bounded_no_cutoff=False,
    commercial_use=False,
):
    """Use the existing full force-field converter with an explicit stable-row map.

    ``system`` is already parameterized by the caller. No force-field tables,
    charges or protonation states are generated here. OpenMM NoCutoff LJ is
    bounded by the existing native adapter's cutoff; explicit acceptance retains
    that approximation warning and is required for this particular profile.
    """
    from ....atomistic.interchange import from_openmm_system

    hypothesis.require_rights(commercial_use=commercial_use)
    for manifest in parameter_rights:
        manifest.require_rights(commercial_use=commercial_use)

    record_ids = tuple(source_record_ids_by_particle)
    rows = {row.record_id: row for row in hypothesis.source_atoms}
    if (
        len(record_ids) != len(rows)
        or len(set(record_ids)) != len(record_ids)
        or set(record_ids) != set(rows)
    ):
        raise ValueError(
            "Every OpenMM particle requires one unique original source record; "
            "no missing or imputed atoms are silently admitted."
        )
    if system.getNumParticles() != len(record_ids):
        raise ValueError("OpenMM particles and explicit source map disagree.")
    bundle = from_openmm_system(
        system,
        units,
        atomic_numbers=[rows[key].element for key in record_ids],
        cutoff=cutoff,
        source_id=source_id,
    )
    bundle.report.require_complete()
    if bundle.report.warnings and not accept_bounded_no_cutoff:
        raise ValueError(
            "Native interchange emitted approximation warnings; explicitly admit "
            "the bounded NoCutoff profile or supply a matching source model."
        )
    field = bundle.force_field.prepare()
    binding = bind_protein(
        hypothesis,
        chemistry,
        field,
        {
            rows[key].atom_key: int(field.system.plan.particle_ids[index])
            for index, key in enumerate(record_ids)
        },
        parameter_energy_unit=units.scale.energy_unit,
        parameter_rights=tuple(parameter_rights),
        commercial_use=commercial_use,
    )
    return ProteinOpenMMBinding(binding, bundle.report, record_ids)


__all__ = [
    "ProteinOpenMMBinding",
    "protein_hypothesis_from_pdb_records",
    "bind_protein_openmm",
]
