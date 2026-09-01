#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import re
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ..structure._record import (
    AssemblyGenerator,
    AssemblyOperation,
    AtomRecord,
    BondRecord,
    ChainRecord,
    ChemicalComponent,
    ChemicalComponentAtom,
    ChemicalComponentBond,
    EntityRecord,
    MacromolecularRecord,
    MissingAtomRecord,
    MissingResidueRecord,
    ResidueRecord,
)
from ..structure._types import BondOrder, ConnectionKind, EntityKind, PolymerKind


_ELEMENT_SYMBOLS = (
    "",
    "H",
    "HE",
    "LI",
    "BE",
    "B",
    "C",
    "N",
    "O",
    "F",
    "NE",
    "NA",
    "MG",
    "AL",
    "SI",
    "P",
    "S",
    "CL",
    "AR",
    "K",
    "CA",
    "SC",
    "TI",
    "V",
    "CR",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "ZN",
    "GA",
    "GE",
    "AS",
    "SE",
    "BR",
    "KR",
    "RB",
    "SR",
    "Y",
    "ZR",
    "NB",
    "MO",
    "TC",
    "RU",
    "RH",
    "PD",
    "AG",
    "CD",
    "IN",
    "SN",
    "SB",
    "TE",
    "I",
    "XE",
    "CS",
    "BA",
    "LA",
    "CE",
    "PR",
    "ND",
    "PM",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
    "HF",
    "TA",
    "W",
    "RE",
    "OS",
    "IR",
    "PT",
    "AU",
    "HG",
    "TL",
    "PB",
    "BI",
    "PO",
    "AT",
    "RN",
    "FR",
    "RA",
    "AC",
    "TH",
    "PA",
    "U",
    "NP",
    "PU",
    "AM",
    "CM",
    "BK",
    "CF",
    "ES",
    "FM",
    "MD",
    "NO",
    "LR",
    "RF",
    "DB",
    "SG",
    "BH",
    "HS",
    "MT",
    "DS",
    "RG",
    "CN",
    "NH",
    "FL",
    "MC",
    "LV",
    "TS",
    "OG",
)
_ATOMIC_NUMBER = {
    symbol: index for index, symbol in enumerate(_ELEMENT_SYMBOLS) if symbol
}
_MISSING = {".", "?"}


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(";"):
            block = [line[1:]]
            index += 1
            while index < len(lines) and not lines[index].startswith(";"):
                block.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError("Unterminated semicolon-delimited mmCIF text field.")
            tokens.append("\n".join(block))
            index += 1
            continue
        lexer = shlex.shlex(line, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens.extend(list(lexer))
        index += 1
    return tokens


def _category(tag: str) -> str:
    return tag.split(".", 1)[0]


def _parse_document(
    text: str,
) -> tuple[str, dict[str, list[dict[str, str]]], dict[str, str]]:
    tokens = _tokens(text)
    block = "structure"
    categories: dict[str, list[dict[str, str]]] = defaultdict(list)
    scalars: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        lower = token.lower()
        if lower.startswith("data_"):
            block = token[5:] or "structure"
            index += 1
            continue
        if lower == "loop_":
            index += 1
            tags: list[str] = []
            while index < len(tokens) and tokens[index].startswith("_"):
                tags.append(tokens[index])
                index += 1
            if not tags:
                raise ValueError("mmCIF loop_ must be followed by data names.")
            category = _category(tags[0])
            if any(_category(tag) != category for tag in tags):
                raise ValueError(
                    "Mixed-category mmCIF loops are not supported because they violate PDBx practice."
                )
            values: list[str] = []
            while index < len(tokens):
                value = tokens[index]
                value_lower = value.lower()
                if (
                    value_lower == "loop_"
                    or value_lower.startswith("data_")
                    or value_lower.startswith("save_")
                ):
                    break
                if value.startswith("_") and len(values) % len(tags) == 0:
                    break
                values.append(value)
                index += 1
            if len(values) % len(tags):
                raise ValueError(f"Loop {category} has an incomplete final row.")
            for offset in range(0, len(values), len(tags)):
                categories[category].append(
                    dict(zip(tags, values[offset : offset + len(tags)], strict=True))
                )
            continue
        if token.startswith("_"):
            if index + 1 >= len(tokens):
                raise ValueError(f"Missing value for mmCIF data name {token}.")
            scalars[token] = tokens[index + 1]
            index += 2
            continue
        index += 1
    return block, dict(categories), scalars


def _value(row: dict[str, str], name: str, default: str | None = None) -> str | None:
    value = row.get(name, default)
    return None if value in _MISSING else value


def _integer(value: str | None) -> int | None:
    return None if value is None else int(value)


def _float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _entity_kind(value: str | None) -> EntityKind:
    normalized = "unknown" if value is None else value.lower().replace("_", "-")
    return {
        "polymer": EntityKind.POLYMER,
        "non-polymer": EntityKind.NON_POLYMER,
        "branched": EntityKind.BRANCHED,
        "water": EntityKind.WATER,
    }.get(normalized, EntityKind.UNKNOWN)


def _polymer_kind(value: str | None) -> PolymerKind:
    if value is None:
        return PolymerKind.NONE
    normalized = value.strip()
    for kind in PolymerKind:
        if kind.value.lower() == normalized.lower():
            return kind
    return PolymerKind.OTHER


def _bond_order(value: str | None) -> BondOrder:
    normalized = "" if value is None else value.lower()
    return {
        "sing": BondOrder.SINGLE,
        "single": BondOrder.SINGLE,
        "doub": BondOrder.DOUBLE,
        "double": BondOrder.DOUBLE,
        "trip": BondOrder.TRIPLE,
        "triple": BondOrder.TRIPLE,
        "quad": BondOrder.QUADRUPLE,
        "arom": BondOrder.UNKNOWN,
    }.get(normalized, BondOrder.UNKNOWN)


def _connection_kind(value: str | None) -> ConnectionKind:
    normalized = "" if value is None else value.lower()
    if normalized.startswith("covale"):
        return ConnectionKind.COVALENT
    if normalized.startswith("disulf"):
        return ConnectionKind.DISULFIDE
    if normalized.startswith("metal"):
        return ConnectionKind.METAL_COORDINATION
    if normalized.startswith("hydrog"):
        return ConnectionKind.HYDROGEN_BOND
    return ConnectionKind.OTHER


def _connection_token(kind: ConnectionKind) -> str:
    return {
        ConnectionKind.COVALENT: "covale",
        ConnectionKind.DISULFIDE: "disulf",
        ConnectionKind.METAL_COORDINATION: "metalc",
        ConnectionKind.HYDROGEN_BOND: "hydrog",
        ConnectionKind.OTHER: "other",
    }[kind]


def _expand_operation_group(group: str) -> list[str]:
    values: list[str] = []
    for item in group.split(","):
        item = item.strip()
        match = re.fullmatch(r"(-?\d+)-(-?\d+)", item)
        if match:
            start, stop = (int(match.group(1)), int(match.group(2)))
            step = 1 if stop >= start else -1
            values.extend(str(value) for value in range(start, stop + step, step))
        elif item:
            values.append(item)
    return values


def _operation_combinations(expression: str) -> list[tuple[str, ...]]:
    groups = re.findall(r"\(([^()]*)\)", expression)
    if not groups:
        groups = [expression]
    expanded = [_expand_operation_group(group) for group in groups]
    if any(not group for group in expanded):
        raise ValueError(f"Invalid assembly operation expression {expression!r}.")
    return list(itertools.product(*expanded))


def parse_mmcif(text: str, /) -> MacromolecularRecord:
    """Parse PDBx/mmCIF host data into a validated macromolecular record."""

    if not isinstance(text, str):
        raise TypeError("text must be a host string.")
    block, categories, scalars = _parse_document(text)
    entity_poly = {
        row["_entity_poly.entity_id"]: _polymer_kind(_value(row, "_entity_poly.type"))
        for row in categories.get("_entity_poly", [])
    }
    sequence_by_entity: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in categories.get("_entity_poly_seq", []):
        entity_id = row["_entity_poly_seq.entity_id"]
        sequence_by_entity[entity_id].append(
            (int(row["_entity_poly_seq.num"]), row["_entity_poly_seq.mon_id"])
        )
    entities: list[EntityRecord] = []
    for row in categories.get("_entity", []):
        entity_id = row["_entity.id"]
        sequence = tuple(
            value for _, value in sorted(sequence_by_entity.get(entity_id, []))
        )
        kind = _entity_kind(_value(row, "_entity.type"))
        entities.append(
            EntityRecord(
                entity_id,
                kind,
                _value(row, "_entity.pdbx_description", "") or "",
                entity_poly.get(
                    entity_id,
                    PolymerKind.NONE
                    if kind is not EntityKind.POLYMER
                    else PolymerKind.OTHER,
                ),
                sequence,
            )
        )
    if not entities:
        raise ValueError(
            "mmCIF requires an _entity category for lossless structure identity."
        )

    atom_rows = categories.get("_atom_site", [])
    if not atom_rows:
        raise ValueError("mmCIF requires a non-empty _atom_site category.")
    auth_by_label: dict[str, str] = {}
    for row in atom_rows:
        label = _value(row, "_atom_site.label_asym_id")
        auth = _value(row, "_atom_site.auth_asym_id")
        if label is not None and auth is not None:
            auth_by_label.setdefault(label, auth)
    chains = [
        ChainRecord(
            row["_struct_asym.id"],
            auth_by_label.get(row["_struct_asym.id"], row["_struct_asym.id"]),
            row["_struct_asym.entity_id"],
        )
        for row in categories.get("_struct_asym", [])
    ]
    if not chains:
        raise ValueError("mmCIF requires _struct_asym rows for label-chain identity.")
    chain_index = {chain.label_asym_id: index for index, chain in enumerate(chains)}

    residue_keys: dict[tuple[object, ...], int] = {}
    residues: list[ResidueRecord] = []
    row_residue_index: list[int] = []
    for row in atom_rows:
        label_chain = _value(row, "_atom_site.label_asym_id")
        if label_chain not in chain_index:
            raise ValueError(
                f"_atom_site references undeclared label_asym_id {label_chain!r}."
            )
        label_comp = _value(row, "_atom_site.label_comp_id")
        auth_comp = _value(row, "_atom_site.auth_comp_id", label_comp)
        label_seq = _integer(_value(row, "_atom_site.label_seq_id"))
        auth_seq = _integer(_value(row, "_atom_site.auth_seq_id"))
        insertion = _value(row, "_atom_site.pdbx_PDB_ins_code")
        hetero = (_value(row, "_atom_site.group_PDB", "ATOM") or "ATOM").upper() != "ATOM"
        key = (
            chain_index[label_chain],
            label_comp,
            auth_comp,
            label_seq,
            auth_seq,
            insertion,
            hetero,
        )
        if key not in residue_keys:
            residue_keys[key] = len(residues)
            residues.append(
                ResidueRecord(
                    chain_index[label_chain],
                    label_comp or auth_comp or "UNK",
                    auth_comp or label_comp or "UNK",
                    label_seq,
                    auth_seq,
                    insertion,
                    label_seq,
                    hetero,
                )
            )
        row_residue_index.append(residue_keys[key])

    components_by_id: dict[str, dict[str, str]] = {
        row["_chem_comp.id"]: row for row in categories.get("_chem_comp", [])
    }
    component_atoms: dict[str, list[ChemicalComponentAtom]] = defaultdict(list)
    for row in categories.get("_chem_comp_atom", []):
        symbol = (_value(row, "_chem_comp_atom.type_symbol") or "").upper()
        component_atoms[row["_chem_comp_atom.comp_id"]].append(
            ChemicalComponentAtom(
                row["_chem_comp_atom.atom_id"],
                symbol,
                _ATOMIC_NUMBER.get(symbol, 0),
                int(_value(row, "_chem_comp_atom.charge", "0") or 0),
                (_value(row, "_chem_comp_atom.pdbx_aromatic_flag", "N") or "N").upper()
                == "Y",
            )
        )
    component_bonds: dict[str, list[ChemicalComponentBond]] = defaultdict(list)
    for row in categories.get("_chem_comp_bond", []):
        order_value = _value(row, "_chem_comp_bond.value_order")
        component_bonds[row["_chem_comp_bond.comp_id"]].append(
            ChemicalComponentBond(
                row["_chem_comp_bond.atom_id_1"],
                row["_chem_comp_bond.atom_id_2"],
                _bond_order(order_value),
                (order_value or "").lower() == "arom"
                or (_value(row, "_chem_comp_bond.pdbx_aromatic_flag", "N") or "N").upper()
                == "Y",
            )
        )
    component_ids = sorted(
        set(components_by_id) | set(component_atoms) | set(component_bonds)
    )
    components = tuple(
        ChemicalComponent(
            component_id,
            _value(components_by_id.get(component_id, {}), "_chem_comp.name", "") or "",
            _value(components_by_id.get(component_id, {}), "_chem_comp.type", "") or "",
            tuple(component_atoms.get(component_id, [])),
            tuple(component_bonds.get(component_id, [])),
            _value(
                components_by_id.get(component_id, {}),
                "_chem_comp.mon_nstd_parent_comp_id",
            ),
        )
        for component_id in component_ids
    )

    atoms: list[AtomRecord] = []
    for row_index, (row, residue_index) in enumerate(
        zip(atom_rows, row_residue_index, strict=True)
    ):
        symbol = (_value(row, "_atom_site.type_symbol") or "").upper()
        x = _float(_value(row, "_atom_site.Cartn_x"))
        y = _float(_value(row, "_atom_site.Cartn_y"))
        z = _float(_value(row, "_atom_site.Cartn_z"))
        present = x is not None and y is not None and z is not None
        atoms.append(
            AtomRecord(
                _value(row, "_atom_site.id", str(row_index + 1)) or str(row_index + 1),
                residue_index,
                int(_value(row, "_atom_site.pdbx_PDB_model_num", "1") or 1),
                _value(row, "_atom_site.label_atom_id")
                or _value(row, "_atom_site.auth_atom_id")
                or "?",
                _value(row, "_atom_site.auth_atom_id")
                or _value(row, "_atom_site.label_atom_id")
                or "?",
                symbol,
                _ATOMIC_NUMBER.get(symbol, 0),
                (
                    np.nan if x is None else x,
                    np.nan if y is None else y,
                    np.nan if z is None else z,
                ),
                float(_value(row, "_atom_site.occupancy", "1") or 1.0),
                float(_value(row, "_atom_site.B_iso_or_equiv", "0") or 0.0),
                _value(row, "_atom_site.label_alt_id"),
                int(
                    (_value(row, "_atom_site.pdbx_formal_charge", "0") or "0").rstrip(
                        "+-"
                    )
                    or 0
                ),
                present,
            )
        )

    def resolve_partner(row: dict[str, str], prefix: str) -> int:
        label_chain = _value(row, f"_struct_conn.{prefix}_label_asym_id")
        auth_chain = _value(row, f"_struct_conn.{prefix}_auth_asym_id")
        label_seq = _integer(_value(row, f"_struct_conn.{prefix}_label_seq_id"))
        auth_seq = _integer(_value(row, f"_struct_conn.{prefix}_auth_seq_id"))
        atom_name = _value(row, f"_struct_conn.{prefix}_label_atom_id") or _value(
            row, f"_struct_conn.{prefix}_auth_atom_id"
        )
        alt = _value(row, f"_struct_conn.{prefix}_label_alt_id")
        insertion = _value(row, f"_struct_conn.pdbx_{prefix}_PDB_ins_code")
        matches: list[int] = []
        for atom_index, atom in enumerate(atoms):
            residue = residues[atom.residue_index]
            chain = chains[residue.chain_index]
            if label_chain is not None and chain.label_asym_id != label_chain:
                continue
            if (
                label_chain is None
                and auth_chain is not None
                and chain.auth_asym_id != auth_chain
            ):
                continue
            if label_seq is not None and residue.label_seq_id != label_seq:
                continue
            if (
                label_seq is None
                and auth_seq is not None
                and residue.auth_seq_id != auth_seq
            ):
                continue
            if insertion is not None and residue.insertion_code != insertion:
                continue
            if (
                atom_name is not None
                and atom.label_atom_id != atom_name
                and atom.auth_atom_id != atom_name
            ):
                continue
            if alt is not None and atom.altloc_id != alt:
                continue
            matches.append(atom_index)
        if not matches:
            raise ValueError(f"Unresolved _struct_conn partner {prefix}.")
        identities: dict[tuple[int, str, str | None], int] = {}
        for match in matches:
            identities.setdefault(atoms[match].identity_key, match)
        if alt is None and len(identities) > 1:
            shared = [value for key, value in identities.items() if key[2] is None]
            if len(shared) == 1:
                return shared[0]
            raise ValueError(
                f"Ambiguous alternate-location _struct_conn partner {prefix}."
            )
        return next(iter(identities.values()))

    bonds: list[BondRecord] = []
    for row in categories.get("_struct_conn", []):
        order_value = _value(row, "_struct_conn.pdbx_value_order")
        bonds.append(
            BondRecord(
                resolve_partner(row, "ptnr1"),
                resolve_partner(row, "ptnr2"),
                _bond_order(order_value),
                (order_value or "").lower() == "arom",
                _connection_kind(_value(row, "_struct_conn.conn_type_id")),
                _value(row, "_struct_conn.id"),
            )
        )

    missing_residues: list[MissingResidueRecord] = []
    for row in categories.get("_pdbx_unobs_or_zero_occ_residues", []):
        label_chain = _value(row, "_pdbx_unobs_or_zero_occ_residues.label_asym_id")
        if label_chain not in chain_index:
            continue
        missing_residues.append(
            MissingResidueRecord(
                chain_index[label_chain],
                _value(row, "_pdbx_unobs_or_zero_occ_residues.label_comp_id") or "UNK",
                _integer(_value(row, "_pdbx_unobs_or_zero_occ_residues.label_seq_id")),
                _integer(_value(row, "_pdbx_unobs_or_zero_occ_residues.auth_seq_id")),
                _value(row, "_pdbx_unobs_or_zero_occ_residues.PDB_ins_code"),
                _integer(_value(row, "_pdbx_unobs_or_zero_occ_residues.PDB_model_num")),
            )
        )
    missing_atoms: list[MissingAtomRecord] = []
    for row in categories.get("_pdbx_unobs_or_zero_occ_atoms", []):
        label_chain = _value(row, "_pdbx_unobs_or_zero_occ_atoms.label_asym_id")
        label_seq = _integer(_value(row, "_pdbx_unobs_or_zero_occ_atoms.label_seq_id"))
        comp = _value(row, "_pdbx_unobs_or_zero_occ_atoms.label_comp_id")
        candidates = [
            index
            for index, residue in enumerate(residues)
            if chains[residue.chain_index].label_asym_id == label_chain
            and residue.label_seq_id == label_seq
            and (comp is None or residue.label_comp_id == comp)
        ]
        if len(candidates) == 1:
            missing_atoms.append(
                MissingAtomRecord(
                    candidates[0],
                    _value(row, "_pdbx_unobs_or_zero_occ_atoms.label_atom_id") or "?",
                    _integer(_value(row, "_pdbx_unobs_or_zero_occ_atoms.PDB_model_num")),
                )
            )

    operations: list[AssemblyOperation] = []
    for row in categories.get("_pdbx_struct_oper_list", []):
        rotation_rows = [
            [
                float(row[f"_pdbx_struct_oper_list.matrix[{i}][{j}]"])
                for j in range(1, 4)
            ]
            for i in range(1, 4)
        ]
        translation_values = [
            float(row[f"_pdbx_struct_oper_list.vector[{i}]"])
            for i in range(1, 4)
        ]
        if len(rotation_rows) != 3 or any(
            len(rotation_row) != 3 for rotation_row in rotation_rows
        ):
            raise ValueError("Assembly rotation must contain exactly three rows of three values.")
        if len(translation_values) != 3:
            raise ValueError("Assembly translation must contain exactly three values.")
        rotation = (
            (rotation_rows[0][0], rotation_rows[0][1], rotation_rows[0][2]),
            (rotation_rows[1][0], rotation_rows[1][1], rotation_rows[1][2]),
            (rotation_rows[2][0], rotation_rows[2][1], rotation_rows[2][2]),
        )
        translation = (
            translation_values[0],
            translation_values[1],
            translation_values[2],
        )
        operations.append(
            AssemblyOperation(row["_pdbx_struct_oper_list.id"], rotation, translation)
        )
    operation_by_id = {operation.operation_id: operation for operation in operations}
    generators: list[AssemblyGenerator] = []
    composite_ids: dict[tuple[str, ...], str] = {}
    for row in categories.get("_pdbx_struct_assembly_gen", []):
        expression = row["_pdbx_struct_assembly_gen.oper_expression"]
        resolved_ids: list[str] = []
        for combination in _operation_combinations(expression):
            if any(identifier not in operation_by_id for identifier in combination):
                raise ValueError(
                    f"Assembly expression references unknown operation {combination}."
                )
            if len(combination) == 1:
                resolved_ids.append(combination[0])
                continue
            if combination not in composite_ids:
                rotation = np.eye(3)
                translation = np.zeros((3,))
                for identifier in combination:
                    operation = operation_by_id[identifier]
                    next_rotation = np.asarray(operation.rotation)
                    next_translation = np.asarray(operation.translation)
                    rotation = next_rotation @ rotation
                    translation = next_rotation @ translation + next_translation
                composite_id = "*".join(combination)
                suffix = 1
                while composite_id in operation_by_id:
                    suffix += 1
                    composite_id = f"{'*'.join(combination)}#{suffix}"
                composite = AssemblyOperation(
                    composite_id, tuple(map(tuple, rotation)), tuple(translation)
                )
                operations.append(composite)
                operation_by_id[composite_id] = composite
                composite_ids[combination] = composite_id
            resolved_ids.append(composite_ids[combination])
        asym_ids = [
            value.strip()
            for value in row["_pdbx_struct_assembly_gen.asym_id_list"].split(",")
            if value.strip()
        ]
        if any(value not in chain_index for value in asym_ids):
            raise ValueError(
                "Assembly generator references an unknown label asymmetric unit."
            )
        generators.append(
            AssemblyGenerator(
                row["_pdbx_struct_assembly_gen.assembly_id"],
                tuple(resolved_ids),
                tuple(chain_index[value] for value in asym_ids),
            )
        )

    experimental_method = scalars.get("_exptl.method")
    resolution = _float(scalars.get("_refine.ls_d_res_high"))
    return MacromolecularRecord(
        block,
        tuple(entities),
        tuple(chains),
        tuple(residues),
        tuple(atoms),
        components,
        tuple(bonds),
        tuple(missing_residues),
        tuple(missing_atoms),
        tuple(operations),
        tuple(generators),
        experimental_method,
        resolution,
    )


def load_mmcif(path: str | Path, /) -> MacromolecularRecord:
    """Read and parse a UTF-8 PDBx/mmCIF host file."""

    return parse_mmcif(Path(path).read_text(encoding="utf-8"))


def _quote(value: object) -> str:
    if value is None:
        return "."
    text = str(value)
    if not text:
        return "''"
    if "\n" in text:
        return f"\n;{text}\n;"
    if any(character.isspace() for character in text) or text.startswith(("#", "_", ";")):
        return "'" + text.replace("'", "''") + "'"
    return text


def _loop(
    lines: list[str], tags: tuple[str, ...], rows: Iterable[tuple[object, ...]]
) -> None:
    materialized = list(rows)
    if not materialized:
        return
    lines.extend(("#", "loop_", *tags))
    for row in materialized:
        lines.append(" ".join(_quote(value) for value in row))


def dumps_mmcif(record: MacromolecularRecord, /) -> str:
    """Serialize the represented identity, chemistry, coordinates, links, and assemblies."""

    if not isinstance(record, MacromolecularRecord):
        raise TypeError("record must be a MacromolecularRecord.")
    lines = [f"data_{record.data_block}"]
    _loop(
        lines,
        ("_entity.id", "_entity.type", "_entity.pdbx_description"),
        (
            (
                entity.entity_id,
                entity.kind.value.replace("-", "_"),
                entity.description or ".",
            )
            for entity in record.entities
        ),
    )
    _loop(
        lines,
        ("_entity_poly.entity_id", "_entity_poly.type"),
        (
            (entity.entity_id, entity.polymer_kind.value)
            for entity in record.entities
            if entity.kind is EntityKind.POLYMER
        ),
    )
    _loop(
        lines,
        ("_entity_poly_seq.entity_id", "_entity_poly_seq.num", "_entity_poly_seq.mon_id"),
        (
            (entity.entity_id, index + 1, component)
            for entity in record.entities
            for index, component in enumerate(entity.sequence_components)
        ),
    )
    _loop(
        lines,
        ("_struct_asym.id", "_struct_asym.entity_id"),
        ((chain.label_asym_id, chain.entity_id) for chain in record.chains),
    )
    _loop(
        lines,
        (
            "_chem_comp.id",
            "_chem_comp.name",
            "_chem_comp.type",
            "_chem_comp.mon_nstd_parent_comp_id",
        ),
        (
            (
                component.component_id,
                component.name or ".",
                component.component_type or ".",
                component.parent_component_id,
            )
            for component in record.chemical_components
        ),
    )
    _loop(
        lines,
        (
            "_chem_comp_atom.comp_id",
            "_chem_comp_atom.atom_id",
            "_chem_comp_atom.type_symbol",
            "_chem_comp_atom.charge",
            "_chem_comp_atom.pdbx_aromatic_flag",
        ),
        (
            (
                component.component_id,
                atom.atom_id,
                atom.element,
                atom.formal_charge,
                "Y" if atom.aromatic else "N",
            )
            for component in record.chemical_components
            for atom in component.atoms
        ),
    )
    _loop(
        lines,
        (
            "_chem_comp_bond.comp_id",
            "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2",
            "_chem_comp_bond.value_order",
            "_chem_comp_bond.pdbx_aromatic_flag",
        ),
        (
            (
                component.component_id,
                bond.atom_id_1,
                bond.atom_id_2,
                {
                    0: "arom" if bond.aromatic else "unk",
                    1: "sing",
                    2: "doub",
                    3: "trip",
                    4: "quad",
                }[int(bond.order)],
                "Y" if bond.aromatic else "N",
            )
            for component in record.chemical_components
            for bond in component.bonds
        ),
    )
    atom_tags = (
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    )
    atom_rows = []
    for atom in record.atoms:
        residue = record.residues[atom.residue_index]
        chain = record.chains[residue.chain_index]
        atom_rows.append(
            (
                "HETATM" if residue.hetero else "ATOM",
                atom.atom_site_id,
                atom.element,
                atom.label_atom_id,
                atom.altloc_id,
                residue.label_comp_id,
                chain.label_asym_id,
                residue.label_seq_id,
                residue.insertion_code,
                atom.position[0] if atom.present else None,
                atom.position[1] if atom.present else None,
                atom.position[2] if atom.present else None,
                atom.occupancy,
                atom.b_factor,
                atom.formal_charge,
                residue.auth_seq_id,
                residue.auth_comp_id,
                chain.auth_asym_id,
                atom.auth_atom_id,
                atom.model_number,
            )
        )
    _loop(lines, atom_tags, atom_rows)
    _loop(
        lines,
        (
            "_struct_conn.id",
            "_struct_conn.conn_type_id",
            "_struct_conn.ptnr1_label_asym_id",
            "_struct_conn.ptnr1_label_seq_id",
            "_struct_conn.ptnr1_label_atom_id",
            "_struct_conn.ptnr1_label_alt_id",
            "_struct_conn.ptnr2_label_asym_id",
            "_struct_conn.ptnr2_label_seq_id",
            "_struct_conn.ptnr2_label_atom_id",
            "_struct_conn.ptnr2_label_alt_id",
            "_struct_conn.pdbx_value_order",
        ),
        (
            (
                bond.connection_id or f"conn{index + 1}",
                _connection_token(bond.connection_kind),
                record.chains[
                    record.residues[
                        record.atoms[bond.atom_index_1].residue_index
                    ].chain_index
                ].label_asym_id,
                record.residues[
                    record.atoms[bond.atom_index_1].residue_index
                ].label_seq_id,
                record.atoms[bond.atom_index_1].label_atom_id,
                record.atoms[bond.atom_index_1].altloc_id,
                record.chains[
                    record.residues[
                        record.atoms[bond.atom_index_2].residue_index
                    ].chain_index
                ].label_asym_id,
                record.residues[
                    record.atoms[bond.atom_index_2].residue_index
                ].label_seq_id,
                record.atoms[bond.atom_index_2].label_atom_id,
                record.atoms[bond.atom_index_2].altloc_id,
                {0: "unk", 1: "sing", 2: "doub", 3: "trip", 4: "quad"}[int(bond.order)],
            )
            for index, bond in enumerate(record.bonds)
        ),
    )
    _loop(
        lines,
        (
            "_pdbx_unobs_or_zero_occ_residues.PDB_model_num",
            "_pdbx_unobs_or_zero_occ_residues.label_asym_id",
            "_pdbx_unobs_or_zero_occ_residues.label_comp_id",
            "_pdbx_unobs_or_zero_occ_residues.label_seq_id",
            "_pdbx_unobs_or_zero_occ_residues.auth_seq_id",
            "_pdbx_unobs_or_zero_occ_residues.PDB_ins_code",
        ),
        (
            (
                value.model_number,
                record.chains[value.chain_index].label_asym_id,
                value.label_comp_id,
                value.label_seq_id,
                value.auth_seq_id,
                value.insertion_code,
            )
            for value in record.missing_residues
        ),
    )
    _loop(
        lines,
        (
            "_pdbx_unobs_or_zero_occ_atoms.PDB_model_num",
            "_pdbx_unobs_or_zero_occ_atoms.label_asym_id",
            "_pdbx_unobs_or_zero_occ_atoms.label_comp_id",
            "_pdbx_unobs_or_zero_occ_atoms.label_seq_id",
            "_pdbx_unobs_or_zero_occ_atoms.label_atom_id",
        ),
        (
            (
                value.model_number,
                record.chains[
                    record.residues[value.residue_index].chain_index
                ].label_asym_id,
                record.residues[value.residue_index].label_comp_id,
                record.residues[value.residue_index].label_seq_id,
                value.label_atom_id,
            )
            for value in record.missing_atoms
        ),
    )
    _loop(
        lines,
        tuple(
            ["_pdbx_struct_oper_list.id"]
            + [
                f"_pdbx_struct_oper_list.matrix[{i}][{j}]"
                for i in range(1, 4)
                for j in range(1, 4)
            ]
            + [f"_pdbx_struct_oper_list.vector[{i}]" for i in range(1, 4)]
        ),
        (
            (
                operation.operation_id,
                *(value for row in operation.rotation for value in row),
                *operation.translation,
            )
            for operation in record.assembly_operations
        ),
    )
    _loop(
        lines,
        (
            "_pdbx_struct_assembly_gen.assembly_id",
            "_pdbx_struct_assembly_gen.oper_expression",
            "_pdbx_struct_assembly_gen.asym_id_list",
        ),
        (
            (
                generator.assembly_id,
                ",".join(generator.operation_ids),
                ",".join(
                    record.chains[index].label_asym_id
                    for index in generator.chain_indices
                ),
            )
            for generator in record.assembly_generators
        ),
    )
    if record.experimental_method is not None:
        lines.extend(("#", f"_exptl.method {_quote(record.experimental_method)}"))
    if record.resolution_angstrom is not None:
        lines.extend(("#", f"_refine.ls_d_res_high {record.resolution_angstrom}"))
    lines.append("#")
    return "\n".join(lines) + "\n"


def dump_mmcif(record: MacromolecularRecord, path: str | Path, /) -> None:
    """Serialize a macromolecular record to a UTF-8 host file."""

    Path(path).write_text(dumps_mmcif(record), encoding="utf-8")


__all__ = ["dump_mmcif", "dumps_mmcif", "load_mmcif", "parse_mmcif"]
