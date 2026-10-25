#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generate or check the admitted exact space-group operation table.

The admitted sources are pinned by revision, path and SHA-256 in `MANIFEST`.
Obtain the listed files under ``--source`` as ``<source name>/<path>`` (for
example ``spglib/src/spg_database.c``); the tool never downloads anything.

Before any table is written the tool verifies:

- every source and license file digest against the manifest, and the declared
  SPDX license identifier of each data file;
- for all 530 Hall entries, that the natively parsed Hall symbol closes to
  exactly the decoded spglib operation set (operation-by-operation);
- that every translation is a rational with the derived common denominator and
  numerators in ``[0, denominator)``;
- that the pinned Toolkit standard-setting map names one Hall entry of each
  International number and that its tabulated operations equal the native
  operations of that entry.

``--check`` regenerates in memory and fails unless ``--output`` is identical.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
from fractions import Fraction
from math import lcm
from pathlib import Path
from typing import NamedTuple

from phydrax._fingerprint import canonical_fingerprint
from phydrax.atomistic.crystal._hall import (
    ExactAffineOperation,
    HallSymbol,
    IntegerMatrix,
    RationalVector,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "phydrax" / "atomistic" / "crystal" / "_space_group_data.py"
HALL_ENTRY_COUNT = 530
INTERNATIONAL_NUMBER_COUNT = 230


class ManifestFile(NamedTuple):
    path: str
    sha256: str


class ManifestSource(NamedTuple):
    name: str
    repository: str
    revision: str
    commit: str
    data: ManifestFile
    license_file: ManifestFile
    license: str
    role: str


MANIFEST = (
    ManifestSource(
        name="spglib",
        repository="https://github.com/spglib/spglib",
        revision="v2.7.0",
        commit="12355c77fb7c505a55f52cae36341d73b781a065",
        data=ManifestFile(
            "src/spg_database.c",
            "bf1ab8a74e1d275e20ee4145bd409b62b5295e680599d3a112a41d396bde56bd",
        ),
        license_file=ManifestFile(
            "COPYING", "9c4e602ce15bf48206dad105b666ddaeedba747eeda6465807106bc11f4f587f"
        ),
        license="BSD-3-Clause",
        role="Hall symbols, setting choices and operation oracle for 530 Hall entries",
    ),
    ManifestSource(
        name="nvalchemi-toolkit",
        repository="https://github.com/NVIDIA/nvalchemi-toolkit",
        revision="f59e28dc6701583a13bb294e536076c2a7379e98",
        commit="f59e28dc6701583a13bb294e536076c2a7379e98",
        data=ManifestFile(
            "nvalchemi/csp/_space_group_tables.py",
            "e9f76edcfa028d8a177a4ff50fe134d91f8644b8a7befc6a734978a41caa59e2",
        ),
        license_file=ManifestFile(
            "LICENSE", "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
        ),
        license="Apache-2.0",
        role="standard-setting Hall map and standard-setting operation oracle",
    ),
)

# spglib setting-choice tokens, interpreted by crystal family.
_MONOCLINIC_CHOICE = re.compile(r"(-?[abc])([123]?)")
_ORTHORHOMBIC_CHOICE = re.compile(r"([12]?)(|ba-c|cab|-cba|bca|a-cb)")
_SPACEGROUP_ROW = re.compile(
    r'\{\s*(\d+),\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)",'
    r'\s*"([^"]*)",\s*"([^"]*)",\s*(\w+),\s*(\d+)\s*\}'
)


class HallEntry(NamedTuple):
    hall_number: int
    international_number: int
    hall_symbol: str
    setting_code: str
    origin_choice: int | None
    cell_choice: int | None
    unique_axis: str | None
    orthorhombic_axes: str | None
    rhombohedral_axes: str | None
    schoenflies_symbol: str
    international_short_symbol: str
    international_full_symbol: str
    operations: tuple[ExactAffineOperation, ...]


def _digest(path: Path, /) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_file(
    source_root: Path, source: ManifestSource, file: ManifestFile, /
) -> Path:
    path = source_root / source.name / file.path
    if not path.is_file():
        raise SystemExit(f"Missing admitted source file {path}.")
    digest = _digest(path)
    if digest != file.sha256:
        raise SystemExit(
            f"{path} has SHA-256 {digest}; the admitted {source.name} "
            f"{source.revision} file is {file.sha256}."
        )
    return path


def _verify_rights(source_root: Path, /) -> dict[str, str]:
    """Verify every pinned data and license digest and the declared SPDX license."""
    texts: dict[str, str] = {}
    for source in MANIFEST:
        _verified_file(source_root, source, source.license_file)
        data = _verified_file(source_root, source, source.data).read_text()
        if f"SPDX-License-Identifier: {source.license}" not in data:
            raise SystemExit(
                f"{source.name} {source.data.path} does not declare {source.license}."
            )
        texts[source.name] = data
    return texts


def _block(text: str, start: str, end: str, /) -> str:
    return text[text.index(start) : text.index(end)]


def _decode_spglib_operation(encoded: int, /) -> ExactAffineOperation:
    """Decode spglib's ternary rotation / duodecimal translation packing."""
    rotation_code = encoded % 19683
    digit = 6561
    rotation: list[int] = []
    for _ in range(9):
        rotation.append((rotation_code % (digit * 3)) // digit - 1)
        digit //= 3
    translation_code = encoded // 19683
    digit = 144
    translation: list[Fraction] = []
    for _ in range(3):
        translation.append(Fraction((translation_code % (digit * 12)) // digit, 12))
        digit //= 12
    matrix: IntegerMatrix = (
        rotation[0], rotation[1], rotation[2], rotation[3], rotation[4],
        rotation[5], rotation[6], rotation[7], rotation[8],
    )  # fmt: skip
    vector: RationalVector = (translation[0], translation[1], translation[2])
    return matrix, vector


def _choice_fields(
    number: int, code: str, /
) -> tuple[int | None, int | None, str | None, str | None, str | None]:
    """Interpret one spglib choice token as origin/cell/axis setting fields."""
    if 3 <= number <= 15:
        match = _MONOCLINIC_CHOICE.fullmatch(code)
        if match is None:
            raise SystemExit(f"Unrecognized monoclinic choice {code!r} for {number}.")
        return None, int(match[2]) if match[2] else None, match[1], None, None
    if 16 <= number <= 74:
        match = _ORTHORHOMBIC_CHOICE.fullmatch(code)
        if match is None:
            raise SystemExit(f"Unrecognized orthorhombic choice {code!r} for {number}.")
        origin = int(match[1]) if match[1] else None
        return origin, None, None, match[2] or "abc", None
    if code in ("H", "R"):
        if not 143 <= number <= 167:
            raise SystemExit(f"Rhombohedral choice {code!r} outside trigonal {number}.")
        return None, None, None, None, "hexagonal" if code == "H" else "rhombohedral"
    if code in ("", "1", "2"):
        if number <= 2:
            if code:
                raise SystemExit(f"Triclinic group {number} has a choice token.")
            return None, None, None, None, None
        return int(code) if code else None, None, None, None, None
    raise SystemExit(f"Unrecognized choice {code!r} for {number}.")


def _spglib_entries(text: str, /) -> tuple[HallEntry, ...]:
    rows = _SPACEGROUP_ROW.findall(
        _block(text, "spacegroup_types[] = {", "symmetry_operations[] = {")
    )
    codes = [
        int(value)
        for value in re.findall(
            r"^\s*(\d+),",
            _block(text, "symmetry_operations[] = {", "symmetry_operation_index"),
            re.MULTILINE,
        )
    ]
    index = [
        (int(count), int(start))
        for count, start in re.findall(
            r"\{(\d+),\s*(\d+)\}",
            _block(text, "symmetry_operation_index[][2] = {", "layer_group_types"),
        )
    ]
    if len(rows) != HALL_ENTRY_COUNT + 1 or len(index) != HALL_ENTRY_COUNT + 1:
        raise SystemExit("spglib table does not contain 530 Hall entries.")
    entries: list[HallEntry] = []
    for hall_number in range(1, HALL_ENTRY_COUNT + 1):
        row = rows[hall_number]
        number = int(row[0])
        # spglib stores '"' as '=' inside its C string literals.
        symbol = " ".join(row[2].split()).replace("=", '"')
        code = row[6].strip()
        count, start = index[hall_number]
        decoded = tuple(_decode_spglib_operation(codes[start + k]) for k in range(count))
        reference = frozenset(decoded)
        if len(reference) != count:
            raise SystemExit(f"Hall entry {hall_number} repeats an operation.")
        group = HallSymbol(symbol).group()
        if frozenset(group.operations) != reference:
            raise SystemExit(
                f"Hall entry {hall_number} {symbol!r}: native closure differs from spglib."
            )
        origin, cell, axis, axes, rhombohedral = _choice_fields(number, code)
        entries.append(
            HallEntry(
                hall_number,
                number,
                symbol,
                code,
                origin,
                cell,
                axis,
                axes,
                rhombohedral,
                row[1].strip(),
                row[5].strip(),
                row[4].strip(),
                group.operations,
            )
        )
    numbers = [entry.international_number for entry in entries]
    if numbers != sorted(numbers) or set(numbers) != set(
        range(1, INTERNATIONAL_NUMBER_COUNT + 1)
    ):
        raise SystemExit("spglib Hall entries do not cover International numbers 1-230.")
    return tuple(entries)


def _exact_literal(node: ast.expr, /) -> Fraction | list[object]:
    match node:
        case ast.Constant(value=int() as value) if not isinstance(value, bool):
            return Fraction(value)
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            value = _exact_literal(operand)
            if isinstance(value, list):
                raise SystemExit("Toolkit table negates a list literal.")
            return -value
        case ast.BinOp(left=left, op=ast.Div(), right=right):
            numerator, denominator = _exact_literal(left), _exact_literal(right)
            if isinstance(numerator, list) or isinstance(denominator, list):
                raise SystemExit("Toolkit table divides a list literal.")
            return numerator / denominator
        case ast.List(elts=elements):
            return [_exact_literal(element) for element in elements]
        case _:
            raise SystemExit(f"Toolkit table has a non-exact literal {ast.dump(node)}.")


def _toolkit_arrays(text: str, /) -> dict[str, list[Fraction]]:
    """Read the Toolkit's flat literal arrays exactly from their source text."""
    arrays: dict[str, list[Fraction]] = {}
    for statement in ast.parse(text).body:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and statement.value.args
        ):
            continue
        literal = _exact_literal(statement.value.args[0])
        if not isinstance(literal, list):
            raise SystemExit("Toolkit array literal is not a list.")
        flat: list[Fraction] = []
        for item in literal:
            if isinstance(item, Fraction):
                flat.append(item)
            elif isinstance(item, list) and all(isinstance(v, Fraction) for v in item):
                flat.extend(v for v in item if isinstance(v, Fraction))
            else:
                raise SystemExit("Toolkit array literal nests beyond one level.")
        arrays[statement.targets[0].id] = flat
    return arrays


def _integer(value: Fraction, /) -> int:
    if value.denominator != 1:
        raise SystemExit(f"Toolkit integer field holds {value}.")
    return value.numerator


def _standard_hall_numbers(
    text: str, entries: tuple[HallEntry, ...], /
) -> tuple[int, ...]:
    arrays = _toolkit_arrays(text)
    standard = tuple(_integer(value) for value in arrays["HALL_FROM_INTERNATIONAL"])
    operations = arrays["SYMM_OPS"]
    indices = [_integer(value) for value in arrays["SG_OPS_IDX"]]
    pointers = [_integer(value) for value in arrays["SG_OPS_PTR"]]
    if len(standard) != INTERNATIONAL_NUMBER_COUNT or len(operations) % 12:
        raise SystemExit("Toolkit standard map or operation rows are malformed.")
    if len(pointers) != INTERNATIONAL_NUMBER_COUNT + 1:
        raise SystemExit("Toolkit operation pointers are malformed.")
    for number, hall_number in enumerate(standard, start=1):
        entry = entries[hall_number - 1]
        if entry.international_number != number:
            raise SystemExit(f"Toolkit maps {number} to Hall entry {hall_number}.")
        tabulated: set[ExactAffineOperation] = set()
        for row in indices[pointers[number - 1] : pointers[number]]:
            values = operations[12 * row : 12 * row + 12]
            rotation = tuple(_integer(value) for value in values[:9])
            if len(rotation) != 9:
                raise SystemExit("Toolkit rotation row is malformed.")
            matrix: IntegerMatrix = (
                rotation[0], rotation[1], rotation[2], rotation[3], rotation[4],
                rotation[5], rotation[6], rotation[7], rotation[8],
            )  # fmt: skip
            vector: RationalVector = (values[9] % 1, values[10] % 1, values[11] % 1)
            tabulated.add((matrix, vector))
        if frozenset(tabulated) != frozenset(entry.operations):
            raise SystemExit(
                f"Toolkit operations of {number} differ from Hall {hall_number}."
            )
    return standard


def _translation_denominator(entries: tuple[HallEntry, ...], /) -> int:
    denominator = 1
    for entry in entries:
        for _, translation in entry.operations:
            denominator = lcm(denominator, *(value.denominator for value in translation))
    return denominator


def _operation_row(
    operation: ExactAffineOperation, denominator: int, /
) -> tuple[int, ...]:
    rotation, translation = operation
    numerators = tuple(value * denominator for value in translation)
    if any(
        value.denominator != 1 or not 0 <= value < denominator for value in numerators
    ):
        raise SystemExit("Translation numerator outside the derived exact range.")
    return rotation + tuple(value.numerator for value in numerators)


def _render_indices(indices: tuple[int, ...], /) -> str:
    lines = []
    for start in range(0, len(indices), 16):
        chunk = ", ".join(str(value) for value in indices[start : start + 16])
        lines.append(f"            {chunk},")
    return "\n".join(lines)


def _quoted(value: str, /) -> str:
    """Return the string literal spelling preferred by the configured formatter."""
    if '"' in value and "'" not in value:
        return repr(value)
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _optional(value: int | str | None, /) -> str:
    if value is None:
        return "None"
    return _quoted(value) if isinstance(value, str) else str(value)


def _render(entries: tuple[HallEntry, ...], standard: tuple[int, ...], /) -> str:
    denominator = _translation_denominator(entries)
    rows = sorted(
        {_operation_row(op, denominator) for e in entries for op in e.operations}
    )
    position = {row: index for index, row in enumerate(rows)}
    entry_indices = [
        tuple(position[_operation_row(op, denominator)] for op in entry.operations)
        for entry in entries
    ]
    data_id = canonical_fingerprint(
        {
            "kind": "admitted-space-group-table",
            "sources": [
                [s.name, s.commit, s.data.path, s.data.sha256, s.license]
                for s in MANIFEST
            ],
            "translation_denominator": denominator,
            "operations": [list(row) for row in rows],
            "entries": [
                [entry.hall_number, entry.international_number, list(indices)]
                for entry, indices in zip(entries, entry_indices, strict=True)
            ],
            "standard_hall_numbers": list(standard),
        }
    )
    sources = "\n".join(
        "    SpaceGroupDataSource(\n"
        f"        {_quoted(source.name)},\n"
        f"        {_quoted(source.repository)},\n"
        f"        {_quoted(source.revision)},\n"
        f"        {_quoted(source.commit)},\n"
        f"        {_quoted(source.data.path)},\n"
        f"        {_quoted(source.data.sha256)},\n"
        f"        {_quoted(source.license)},\n"
        f"        {_quoted(source.license_file.path)},\n"
        f"        {_quoted(source.license_file.sha256)},\n"
        f"        {_quoted(source.role)},\n"
        "    ),"
        for source in MANIFEST
    )
    operation_lines = "\n".join(f"    {row!r}," for row in rows)
    entry_lines = "\n".join(
        "    AdmittedHallEntry(\n"
        f"        {entry.hall_number}, {entry.international_number}, "
        f"{_quoted(entry.hall_symbol)}, {_quoted(entry.setting_code)},\n"
        f"        {_optional(entry.origin_choice)}, {_optional(entry.cell_choice)}, "
        f"{_optional(entry.unique_axis)}, {_optional(entry.orthorhombic_axes)}, "
        f"{_optional(entry.rhombohedral_axes)},\n"
        f"        {_quoted(entry.schoenflies_symbol)}, "
        f"{_quoted(entry.international_short_symbol)}, "
        f"{_quoted(entry.international_full_symbol)},\n"
        "        (\n"
        f"{_render_indices(indices)}\n"
        "        ),\n"
        "    ),"
        for entry, indices in zip(entries, entry_indices, strict=True)
    )
    standard_lines = "\n".join(
        "    " + ", ".join(str(value) for value in standard[start : start + 16]) + ","
        for start in range(0, len(standard), 16)
    )
    return f'''#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# Generated by tools/generate_space_group_tables.py; do not edit by hand.
# Hall entries, setting choices and the operation oracle come from spglib
# {MANIFEST[0].revision} ({MANIFEST[0].commit}) {MANIFEST[0].data.path},
# Copyright (C) 2010 Atsushi Togo; COPYING: Copyright (c) 2024, Spglib team;
# See LICENSES/SPGLIB-BSD-3-CLAUSE.txt for conditions and disclaimer.
# BSD-3-Clause.  The standard-setting Hall map comes from NVIDIA
# nvalchemi-toolkit {MANIFEST[1].commit}
# {MANIFEST[1].data.path}, Apache-2.0.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSES/NVALCHEMI-TOOLKIT-APACHE-2.0.txt.
# Modifications: native exact Hall closure, canonical integer operations and
# a standard-setting index map replace the source floating-point arrays.
#

"""Admitted exact space-group operations for 530 Hall settings.

``OPERATIONS`` rows are ``(R00, R01, ..., R22, n0, n1, n2)``: an integer
rotation acting on fractional column coordinates and the translation
``n / TRANSLATION_DENOMINATOR`` reduced to ``[0, 1)``.  Each Hall entry lists
its operations in the canonical centering-major order of the native Hall
closure, which the generator proved identical to the source table.
``STANDARD_HALL_NUMBERS[k]`` is the standard Hall entry of International
number ``k + 1``.  Symbols are source labels, not scientific identity.
"""

from __future__ import annotations

from typing import NamedTuple


class SpaceGroupDataSource(NamedTuple):
    name: str
    repository: str
    revision: str
    commit: str
    path: str
    sha256: str
    license: str
    license_path: str
    license_sha256: str
    role: str


class AdmittedHallEntry(NamedTuple):
    hall_number: int
    international_number: int
    hall_symbol: str
    setting_code: str
    origin_choice: int | None
    cell_choice: int | None
    unique_axis: str | None
    orthorhombic_axes: str | None
    rhombohedral_axes: str | None
    schoenflies_symbol: str
    international_short_symbol: str
    international_full_symbol: str
    operation_indices: tuple[int, ...]


SPACE_GROUP_SOURCES: tuple[SpaceGroupDataSource, ...] = (
{sources}
)
TRANSLATION_DENOMINATOR = {denominator}
SPACE_GROUP_DATA_ID = {_quoted(data_id)}

# fmt: off
OPERATIONS: tuple[tuple[int, ...], ...] = (
{operation_lines}
)

HALL_ENTRIES: tuple[AdmittedHallEntry, ...] = (
{entry_lines}
)

STANDARD_HALL_NUMBERS: tuple[int, ...] = (
{standard_lines}
)
# fmt: on


__all__ = [
    "AdmittedHallEntry",
    "HALL_ENTRIES",
    "OPERATIONS",
    "SPACE_GROUP_DATA_ID",
    "SPACE_GROUP_SOURCES",
    "SpaceGroupDataSource",
    "STANDARD_HALL_NUMBERS",
    "TRANSLATION_DENOMINATOR",
]
'''


def generate(source_root: Path, /) -> str:
    """Verify rights and every admitted entry, then render the data module."""
    texts = _verify_rights(source_root)
    entries = _spglib_entries(texts["spglib"])
    standard = _standard_hall_numbers(texts["nvalchemi-toolkit"], entries)
    return _render(entries, standard)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or check the admitted exact space-group table."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory holding <source name>/<path> files pinned in MANIFEST.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail unless --output equals the regenerated module.",
    )
    arguments = parser.parse_args()
    content = generate(arguments.source)
    output: Path = arguments.output
    if arguments.check:
        if not output.is_file() or output.read_text() != content:
            sys.exit(
                f"{output} differs from the table generated from {arguments.source}."
            )
        return
    output.write_text(content)


if __name__ == "__main__":
    main()
