# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from ...units import ANGSTROM


_REAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")


@dataclass(frozen=True, slots=True)
class PDBAtomRecord:
    """Neutral PDB fixed-column atom record; no biological or bond inference."""

    source_id: str
    line_number: int
    raw_line: str
    record_kind: str
    model_id: str
    atom_serial: str
    atom_name: str
    alternate_location: str
    residue_name: str
    chain_id: str
    author_residue_number: str
    insertion_code: str
    position: tuple[float, float, float]
    occupancy: float
    temperature_factor: float
    element: str
    formal_charge: str

    @property
    def record_id(self):
        return f"{self.source_id}:line:{self.line_number}"

    @property
    def atom_identity(self):
        return (
            self.chain_id,
            self.author_residue_number,
            self.insertion_code,
            self.atom_name,
        )

    @property
    def length_unit(self):
        return ANGSTROM


def _number(value, field, line_number):
    text = value.strip()
    if _REAL.fullmatch(text) is None:
        raise ValueError(f"PDB line {line_number}: missing/invalid {field}.")
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"PDB line {line_number}: nonfinite {field}.")
    return number


def read_pdb_atom_records(text: str, *, source_id: str) -> tuple[PDBAtomRecord, ...]:
    """Read the strict numeric-serial PDB atom profile without silently losing rows.

    MODEL, source line, author IDs, insertion codes, alternate conformers,
    occupancy, B factor, element and charge are retained. Hybrid-36 serials,
    missing element/occupancy, malformed model nesting and duplicate record
    identities refuse. This is not an mmCIF reader and does not parse connectivity,
    infer elements, select models, complete atoms or guess residue chemistry.
    """
    if (
        not isinstance(text, str)
        or not isinstance(source_id, str)
        or not source_id
        or source_id.strip() != source_id
    ):
        raise ValueError("PDB text and nonempty canonical source identity are required.")
    records, models, seen = [], set(), set()
    current_model, explicit_models, ended_model = None, False, False
    for line_number, line in enumerate(text.splitlines(), 1):
        kind = line[:6].strip()
        if kind == "MODEL":
            model = line[10:14].strip()
            if (
                current_model is not None
                or records
                and not explicit_models
                or _INTEGER.fullmatch(model) is None
                or model in models
            ):
                raise ValueError(
                    f"PDB line {line_number}: invalid/duplicate MODEL support."
                )
            explicit_models, ended_model, current_model = True, False, model
            models.add(model)
        elif kind == "ENDMDL":
            if not explicit_models or current_model is None:
                raise ValueError(f"PDB line {line_number}: unmatched ENDMDL.")
            current_model, ended_model = None, True
        elif kind in ("ATOM", "HETATM"):
            if explicit_models and (current_model is None or ended_model):
                raise ValueError(
                    f"PDB line {line_number}: atom outside its declared MODEL."
                )
            if len(line) < 78:
                raise ValueError(
                    f"PDB line {line_number}: mandatory element columns absent."
                )
            model = current_model if explicit_models else "1"
            serial, atom, residue, author = (
                line[6:11].strip(),
                line[12:16].strip(),
                line[17:20].strip(),
                line[22:26].strip(),
            )
            element = line[76:78].strip()
            if (
                _INTEGER.fullmatch(serial) is None
                or _INTEGER.fullmatch(author) is None
                or not atom
                or not residue
                or re.fullmatch(r"[A-Za-z]{1,2}", element) is None
            ):
                raise ValueError(
                    f"PDB line {line_number}: unsupported serial/residue/atom/element profile."
                )
            identity = model, serial
            if identity in seen:
                raise ValueError(
                    f"PDB line {line_number}: repeated atom serial within one model."
                )
            seen.add(identity)
            xyz = tuple(
                _number(line[start : start + 8], "coordinate", line_number)
                for start in (30, 38, 46)
            )
            occupancy = _number(line[54:60], "occupancy", line_number)
            if not 0 <= occupancy <= 1:
                raise ValueError(f"PDB line {line_number}: occupancy outside [0,1].")
            records.append(
                PDBAtomRecord(
                    source_id,
                    line_number,
                    line,
                    kind,
                    model,
                    serial,
                    atom,
                    line[16:17].strip(),
                    residue,
                    line[21:22].strip(),
                    author,
                    line[26:27].strip(),
                    xyz,
                    occupancy,
                    _number(line[60:66], "temperature factor", line_number),
                    element,
                    line[78:80].strip(),
                )
            )
    if explicit_models and current_model is not None:
        raise ValueError("PDB model has no ENDMDL terminator.")
    if not records:
        raise ValueError("No atom records in the supported PDB profile.")
    return tuple(records)


def select_pdb_model(
    records: tuple[PDBAtomRecord, ...],
    model_id: str,
    *,
    alternate_locations: Mapping[tuple[str, str, str, str], str],
) -> tuple[PDBAtomRecord, ...]:
    """Explicit model/conformer view; caller retains the unselected raw records."""
    selected_model = tuple(record for record in records if record.model_id == model_id)
    if not selected_model:
        raise ValueError("Requested PDB model is absent.")
    grouped = {}
    for record in selected_model:
        grouped.setdefault(record.atom_identity, []).append(record)
    if set(alternate_locations) - set(grouped):
        raise ValueError("Alternate-location policy references absent atoms.")
    admitted = set()
    for identity, alternatives in grouped.items():
        if identity in alternate_locations:
            matches = tuple(
                record
                for record in alternatives
                if record.alternate_location == alternate_locations[identity]
            )
        elif len(alternatives) == 1 and alternatives[0].alternate_location == "":
            matches = tuple(alternatives)
        else:
            raise ValueError(
                "Every alternate-location atom requires an explicit conformer selection."
            )
        if len(matches) != 1 or matches[0].occupancy <= 0:
            raise ValueError(
                "Selected conformer must resolve one positive-occupancy source atom."
            )
        admitted.add(matches[0].record_id)
    return tuple(record for record in selected_model if record.record_id in admitted)


__all__ = ["PDBAtomRecord", "read_pdb_atom_records", "select_pdb_model"]
