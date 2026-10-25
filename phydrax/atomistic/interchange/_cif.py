#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host CIF 1.1 crystal admission with explicit refusal of unresolved content.

The parser reads one data block of a Crystallographic Information File
(syntax: S. R. Hall, F. H. Allen, I. D. Brown, Acta Cryst. A47 (1991) 655, and
the CIF 1.1 specification) into exact symmetry operations, a cell and fully
ordered atom sites.  Interpretation follows the core CIF dictionary: lengths
in angstrom, angles in degrees, fractional site coordinates, and an absent
``_atom_site_occupancy`` meaning 1.

Nothing is inferred: element identity requires ``_atom_site_type_symbol``
(labels are never parsed for elements), partial occupancy, disorder groups or
assemblies, dummy atoms, incommensurate/magnetic descriptions and symmetry
operations that do not form a space group are refused with an `AdapterError`.
Retained but unmodeled content (standard uncertainties, displacement
parameters, unconsumed tags) is enumerated as `AdapterLoss` records.  Symmetry
expansion, setting identification and site multiplicities are computed
natively from the exact operations.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float
from ...interchange import AdapterError, AdapterLoss, AdapterReport, AdapterStatus
from ...typing import Dim, HostFloat64, HostInt64, parse
from ..crystal._hall import (
    canonical_affine_group,
    close_affine_group,
    ExactAffineOperation,
    HallSymbol,
    reduce_translation,
)
from ..crystal._setting import (
    admitted_hall_numbers,
    HALL_ENTRY_COUNT,
    SpaceGroupSetting,
)


ChargedTypeSymbolPolicy: TypeAlias = Literal["refuse", "drop"]
"""Type symbols such as ``O2-`` carry an oxidation-state label: refuse them, or
drop the label (recorded as an interpretation-changing loss)."""

_ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np",
    "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
    "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)  # fmt: skip
_ATOMIC_NUMBERS: Mapping[str, int] = {
    symbol: number for number, symbol in enumerate(_ELEMENT_SYMBOLS, start=1)
}
_TYPE_SYMBOL = re.compile(r"^([A-Z][a-z]?)(\d*[+-])?$")
_UNCERTAINTY = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(\(\d+\))?$")
_BOND_ORDERS: Mapping[str, int] = {"S": 1, "D": 2, "T": 3, "A": 4}
_SOURCE_FORMAT = "cif"
_TARGET_FORMAT = "phydrax.atomistic.CIFCrystalStructure"
_REFUSED_PREFIXES: Mapping[str, str] = {
    "_space_group_magn": "magnetic space-group description",
    "_atom_site_moment": "magnetic moments",
    "_cell_wave_vector": "incommensurate modulation",
    "_atom_site_fourier": "incommensurate modulation",
    "_cell_subsystem": "composite structure",
}
_OPERATION_TAGS = ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz")
_HALL_TAGS = ("_space_group_name_hall", "_symmetry_space_group_name_hall")
_NUMBER_TAGS = ("_space_group_it_number", "_symmetry_int_tables_number")
_CELL_TAGS = (
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
)
_SITE_TAGS = (
    "_atom_site_label",
    "_atom_site_type_symbol",
    "_atom_site_fract_x",
    "_atom_site_fract_y",
    "_atom_site_fract_z",
    "_atom_site_occupancy",
    "_atom_site_disorder_group",
    "_atom_site_disorder_assembly",
    "_atom_site_calc_flag",
    "_atom_site_symmetry_multiplicity",
    "_atom_site_site_symmetry_multiplicity",
)
_BOND_TAGS = (
    "_geom_bond_atom_site_label_1",
    "_geom_bond_atom_site_label_2",
    "_geom_bond_site_symmetry_1",
    "_geom_bond_site_symmetry_2",
    "_ccdc_geom_bond_type",
)


class CIFSiteDim(Dim):
    """Asymmetric-unit atom sites of one CIF block."""


class CIFAtomDim(Dim):
    """Symmetry-expanded atoms of one CIF unit cell."""


def _refuse(status: AdapterStatus, message: str, /) -> AdapterError:
    return AdapterError(status, message)


class _Block(NamedTuple):
    name: str
    items: dict[str, str | None]
    loops: list[tuple[tuple[str, ...], list[tuple[str | None, ...]]]]


def _tokens(text: str, /) -> list[tuple[str, bool]]:
    """Return ``(token, quoted)`` pairs; ``quoted`` values are never keywords."""
    tokens: list[tuple[str, bool]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(";"):
            field = [line[1:]]
            index += 1
            while index < len(lines) and not lines[index].startswith(";"):
                field.append(lines[index])
                index += 1
            if index == len(lines):
                raise _refuse(
                    AdapterStatus.MALFORMED_SOURCE, "Unterminated CIF text field."
                )
            tokens.append(("\n".join(field).strip("\n"), True))
            index += 1
            continue
        position = 0
        while position < len(line):
            character = line[position]
            if character.isspace():
                position += 1
            elif character == "#":
                break
            elif character in "'\"":
                end = position + 1
                while end < len(line) and not (
                    line[end] == character
                    and (end + 1 == len(line) or line[end + 1].isspace())
                ):
                    end += 1
                if end >= len(line):
                    raise _refuse(
                        AdapterStatus.MALFORMED_SOURCE,
                        f"Unterminated quoted CIF value: {line!r}",
                    )
                tokens.append((line[position + 1 : end], True))
                position = end + 1
            else:
                end = position
                while end < len(line) and not line[end].isspace():
                    end += 1
                tokens.append((line[position:end], False))
                position = end
        index += 1
    return tokens


def _value(token: str, quoted: bool, /) -> str | None:
    return None if not quoted and token in ("?", ".") else token


def _blocks(text: str, /) -> list[_Block]:
    tokens = _tokens(text)
    blocks: list[_Block] = []
    index = 0
    while index < len(tokens):
        token, quoted = tokens[index]
        lowered = token.lower()
        if not quoted and lowered.startswith("data_"):
            blocks.append(_Block(token[5:], {}, []))
            index += 1
            continue
        if not quoted and (lowered.startswith("save_") or lowered == "global_"):
            raise _refuse(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "CIF save frames and global blocks are dictionary constructs, not crystals.",
            )
        if not blocks:
            raise _refuse(
                AdapterStatus.MALFORMED_SOURCE, "CIF content precedes any data_ block."
            )
        block = blocks[-1]
        if not quoted and lowered == "loop_":
            index += 1
            names: list[str] = []
            while (
                index < len(tokens)
                and not tokens[index][1]
                and tokens[index][0].startswith("_")
            ):
                names.append(tokens[index][0].lower())
                index += 1
            values: list[str | None] = []
            while index < len(tokens):
                candidate, candidate_quoted = tokens[index]
                if not candidate_quoted and (
                    candidate.startswith("_")
                    or candidate.lower() in ("loop_", "global_")
                    or candidate.lower().startswith(("data_", "save_"))
                ):
                    break
                values.append(_value(candidate, candidate_quoted))
                index += 1
            if not names or len(values) % len(names):
                raise _refuse(
                    AdapterStatus.MALFORMED_SOURCE,
                    f"CIF loop {names} has {len(values)} values, not a multiple of its tags.",
                )
            rows = [
                tuple(values[start : start + len(names)])
                for start in range(0, len(values), len(names))
            ]
            block.loops.append((tuple(names), rows))
            continue
        if not quoted and token.startswith("_"):
            if index + 1 >= len(tokens):
                raise _refuse(
                    AdapterStatus.MALFORMED_SOURCE, f"CIF tag {token} has no value."
                )
            name = token.lower()
            if name in block.items or any(name in loop[0] for loop in block.loops):
                raise _refuse(
                    AdapterStatus.MALFORMED_SOURCE, f"CIF tag {token} is repeated."
                )
            block.items[name] = _value(*tokens[index + 1])
            index += 2
            continue
        raise _refuse(AdapterStatus.MALFORMED_SOURCE, f"Unexpected CIF token {token!r}.")
    return blocks


def _number(text: str | None, name: str, /) -> tuple[float, bool]:
    """Parse a CIF numeric value; returns the value and whether an s.u. was present."""
    if text is None:
        raise _refuse(AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC, f"{name} is unknown.")
    match = _UNCERTAINTY.match(text.strip())
    if match is None:
        raise _refuse(AdapterStatus.MALFORMED_SOURCE, f"{name} is not numeric: {text!r}.")
    return float(match.group(1)), match.group(2) is not None


def _fraction(text: str, /) -> Fraction:
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        if not numerator.isdigit() or not denominator.isdigit() or int(denominator) == 0:
            raise ValueError(text)
        return Fraction(int(numerator), int(denominator))
    return Fraction(text)


def parse_symmetry_operation(text: str, /) -> ExactAffineOperation:
    """Parse one CIF ``x,y,z``-style operation into an exact Seitz pair.

    Coefficients must be integers and translations exact rationals; a decimal
    approximation such as ``0.3333`` is kept literally and will fail group
    closure rather than being snapped to ``1/3``.
    """
    rows = text.replace(" ", "").lower().split(",")
    if len(rows) != 3:
        raise _refuse(
            AdapterStatus.MALFORMED_SOURCE, f"Symmetry operation {text!r} needs 3 rows."
        )
    rotation: list[int] = []
    translation: list[Fraction] = []
    for row in rows:
        coefficients = [0, 0, 0]
        shift = Fraction(0)
        terms = re.findall(r"[+-]?[^+-]+", row)
        if not terms or "".join(terms) != row:
            raise _refuse(
                AdapterStatus.MALFORMED_SOURCE, f"Malformed symmetry row {row!r}."
            )
        for term in terms:
            sign = -1 if term.startswith("-") else 1
            body = term.lstrip("+-")
            variable = body[-1] if body and body[-1] in "xyz" else None
            magnitude = body[:-1].rstrip("*") if variable is not None else body
            try:
                value = _fraction(magnitude) if magnitude else Fraction(1)
            except (ValueError, ZeroDivisionError) as error:
                raise _refuse(
                    AdapterStatus.MALFORMED_SOURCE, f"Malformed symmetry term {term!r}."
                ) from error
            if variable is None:
                shift += sign * value
            else:
                if value.denominator != 1:
                    raise _refuse(
                        AdapterStatus.MALFORMED_SOURCE,
                        f"Symmetry coefficient {term!r} is not an integer.",
                    )
                coefficients["xyz".index(variable)] += sign * value.numerator
        rotation.extend(coefficients)
        translation.append(shift)
    return (
        (
            rotation[0],
            rotation[1],
            rotation[2],
            rotation[3],
            rotation[4],
            rotation[5],
            rotation[6],
            rotation[7],
            rotation[8],
        ),  # fmt: skip
        reduce_translation((translation[0], translation[1], translation[2])),
    )


def _cell_vectors(lengths: np.ndarray, angles: np.ndarray, /) -> np.ndarray:
    """Row cell vectors in the standard orientation (a along x, b in xy)."""
    alpha, beta, gamma = np.deg2rad(angles)
    a, b, c = lengths
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    squared = c**2 - cx**2 - cy**2
    if not np.isfinite(squared) or squared <= 0.0:
        raise _refuse(
            AdapterStatus.INCONSISTENT_SOURCE, "CIF cell angles admit no positive volume."
        )
    return np.asarray(
        [
            [a, 0.0, 0.0],
            [b * np.cos(gamma), b * np.sin(gamma), 0.0],
            [cx, cy, np.sqrt(squared)],
        ],
        dtype=np.float64,
    )


@final
class CIFBond(StrictModule):
    """One declared ``_geom_bond`` row.

    ``first``/``second`` index ASU sites; ``*_operation`` are 0-based indices
    into the block's *declared* operation list and ``*_translation`` the
    lattice translations of the CIF ``n_klm`` codes (``.`` is the identity).
    ``order`` is the ``_ccdc_geom_bond_type`` order (4 aromatic) or ``None``.
    """

    first: int = eqx.field(static=True)
    second: int = eqx.field(static=True)
    first_operation: int = eqx.field(static=True)
    second_operation: int = eqx.field(static=True)
    first_translation: tuple[int, int, int] = eqx.field(static=True)
    second_translation: tuple[int, int, int] = eqx.field(static=True)
    order: int | None = eqx.field(static=True)


@final
class CIFCrystalStructure(StrictModule, NonTrainableState):
    """Admitted ordered crystal structure of one CIF data block.

    ``operations`` are the canonical exact operations; ``declared_operations``
    keep the source order used by bond symmetry codes.  ``setting`` is the
    admitted `SpaceGroupSetting` with exactly these operations, or ``None``
    when no admitted Hall setting matches (the exact operations remain the
    identity).  ``expanded_*`` list every distinct unit-cell image of each
    site in site-major order with fractional coordinates in ``[0, 1)``,
    ``expanded_operation_indices`` indexing ``operations``.
    """

    __strict_contract__ = True

    cell_vectors: HostFloat64[Literal[3], Literal[3]]
    site_fractional: HostFloat64[CIFSiteDim, Literal[3]]
    expanded_fractional: HostFloat64[CIFAtomDim, Literal[3]]
    expanded_atomic_numbers: HostInt64[CIFAtomDim]
    expanded_site_indices: HostInt64[CIFAtomDim]
    expanded_operation_indices: HostInt64[CIFAtomDim]
    setting: SpaceGroupSetting | None
    report: AdapterReport
    bonds: tuple[CIFBond, ...]
    block_name: str = eqx.field(static=True)
    cell_lengths: tuple[float, float, float] = eqx.field(static=True)
    cell_angles: tuple[float, float, float] = eqx.field(static=True)
    site_labels: tuple[str, ...] = eqx.field(static=True)
    site_atomic_numbers: tuple[int, ...] = eqx.field(static=True)
    site_multiplicities: tuple[int, ...] = eqx.field(static=True)
    operations: tuple[ExactAffineOperation, ...] = eqx.field(static=True)
    declared_operations: tuple[ExactAffineOperation, ...] = eqx.field(static=True)
    declared_international_number: int | None = eqx.field(static=True)
    declared_hall_symbol: str | None = eqx.field(static=True)
    site_tolerance: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)


def _loop(
    block: _Block, tag: str, /
) -> tuple[tuple[str, ...], list[tuple[str | None, ...]]] | None:
    for names, rows in block.loops:
        if tag in names:
            return names, rows
    return None


def _column(
    names: tuple[str, ...], rows: list[tuple[str | None, ...]], tag: str, /
) -> list[str | None] | None:
    if tag not in names:
        return None
    position = names.index(tag)
    return [row[position] for row in rows]


def _first_item(block: _Block, tags: Sequence[str], /) -> str | None:
    values = [block.items[tag] for tag in tags if block.items.get(tag) is not None]
    if len(set(values)) > 1:
        raise _refuse(
            AdapterStatus.INCONSISTENT_SOURCE,
            f"CIF aliases {list(tags)} disagree: {values}.",
        )
    return values[0] if values else None


def _select_block(blocks: Sequence[_Block], name: str | None, /) -> _Block:
    """The sole data block, or the unique block named ``name`` (case-insensitive)."""
    if name is None:
        if len(blocks) != 1:
            raise _refuse(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                f"CIF has {len(blocks)} data blocks; select one by name.",
            )
        return blocks[0]
    matches = [item for item in blocks if item.name.lower() == name.lower()]
    if len(matches) != 1:
        raise _refuse(
            AdapterStatus.MALFORMED_SOURCE, f"CIF block {name!r} is not unique."
        )
    return matches[0]


def _admitted_tags(block: _Block, /) -> set[str]:
    """Every item and loop tag, refusing descriptions this admission does not model."""
    all_tags = set(block.items) | {name for names, _ in block.loops for name in names}
    for prefix, meaning in _REFUSED_PREFIXES.items():
        if any(tag.startswith(prefix) for tag in all_tags):
            raise _refuse(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                f"CIF declares {meaning} ({prefix}*), which this admission does not model.",
            )
    return all_tags


def _cell(
    block: _Block, consumed: set[str], uncertain: set[str], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Admissible cell ``(lengths, angles, row vectors)``; records consumed/s.u. tags."""
    cell_values: list[float] = []
    for tag in _CELL_TAGS:
        value, has_uncertainty = _number(block.items.get(tag), tag)
        cell_values.append(value)
        consumed.add(tag)
        if has_uncertainty:
            uncertain.add(tag)
    lengths = np.asarray(cell_values[:3], dtype=np.float64)
    angles = np.asarray(cell_values[3:], dtype=np.float64)
    if np.any(lengths <= 0.0) or np.any(angles <= 0.0) or np.any(angles >= 180.0):
        raise _refuse(
            AdapterStatus.INCONSISTENT_SOURCE, "CIF cell parameters are not admissible."
        )
    return lengths, angles, _cell_vectors(lengths, angles)


class _Symmetry(NamedTuple):
    """Admitted symmetry of one block; ``operations`` are canonically ordered."""

    hall_symbol: str | None
    declared_number: int | None
    declared: tuple[ExactAffineOperation, ...]
    operations: tuple[ExactAffineOperation, ...]
    setting: SpaceGroupSetting | None


def _declared_operations(
    block: _Block,
    hall_operations: tuple[ExactAffineOperation, ...] | None,
    consumed: set[str],
    assumptions: list[str],
    /,
) -> tuple[ExactAffineOperation, ...]:
    """Source-ordered operations: the operation loop, else the Hall generation."""
    operation_loop = next(
        (found for tag in _OPERATION_TAGS if (found := _loop(block, tag)) is not None),
        None,
    )
    if operation_loop is not None:
        names, rows = operation_loop
        column = next(
            (
                values
                for tag in _OPERATION_TAGS
                if (values := _column(names, rows, tag)) is not None
            ),
            None,
        )
        if column is None or any(value is None for value in column):
            raise _refuse(
                AdapterStatus.MALFORMED_SOURCE, "CIF symmetry operations are incomplete."
            )
        declared = tuple(
            parse_symmetry_operation(value) for value in column if value is not None
        )
        consumed.update(names)
        return declared
    if hall_operations is not None:
        assumptions.append("Operations were generated from the declared Hall symbol.")
        return hall_operations
    raise _refuse(
        AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
        "CIF declares neither symmetry operations nor a Hall symbol.",
    )


def _space_group(
    declared: tuple[ExactAffineOperation, ...],
    hall_operations: tuple[ExactAffineOperation, ...] | None,
    /,
) -> tuple[frozenset[ExactAffineOperation], tuple[ExactAffineOperation, ...]]:
    """Distinct, closed declared operations agreeing with any Hall symbol.

    Returns the operation set and its canonical order.
    """
    operation_set = frozenset(declared)
    if len(operation_set) != len(declared):
        raise _refuse(
            AdapterStatus.INCONSISTENT_SOURCE, "CIF repeats a symmetry operation."
        )
    try:
        if (
            close_affine_group(operation_set, maximum_order=len(operation_set))
            != operation_set
        ):
            raise ValueError("Declared operations are not closed.")
        canonical = canonical_affine_group(operation_set)
    except ValueError as error:
        raise _refuse(
            AdapterStatus.INCONSISTENT_SOURCE,
            f"CIF symmetry operations do not form a space group: {error}",
        ) from error
    if hall_operations is not None and frozenset(hall_operations) != operation_set:
        raise _refuse(
            AdapterStatus.INCONSISTENT_SOURCE,
            "The declared Hall symbol disagrees with the declared operations.",
        )
    return operation_set, canonical.operations


def _check_metric_invariance(
    operations: Sequence[ExactAffineOperation],
    vectors: np.ndarray,
    metric_bound: float,
    /,
) -> None:
    """Refuse a cell whose metric ``G`` violates ``R^T G R = G`` beyond the bound."""
    metric = vectors @ vectors.T
    scale = float(np.max(np.abs(metric)))
    for rotation, _ in operations:
        matrix = np.asarray(rotation, dtype=np.float64).reshape((3, 3))
        if np.max(np.abs(matrix.T @ metric @ matrix - metric)) > metric_bound * scale:
            raise _refuse(
                AdapterStatus.INCONSISTENT_SOURCE,
                "The declared cell is not invariant under the declared operations.",
            )


def _admitted_setting(
    declared_number: int | None,
    operation_set: frozenset[ExactAffineOperation],
    assumptions: list[str],
    /,
) -> SpaceGroupSetting | None:
    """First admitted Hall setting, in Hall-number order, with these operations."""
    candidates = (
        admitted_hall_numbers(declared_number)
        if declared_number is not None
        else tuple(range(1, HALL_ENTRY_COUNT + 1))
    )
    setting = next(
        (
            candidate_setting
            for hall in candidates
            if frozenset(
                (candidate_setting := SpaceGroupSetting(hall)).exact_operations()
            )
            == operation_set
        ),
        None,
    )
    if setting is None:
        assumptions.append(
            "The operations match no admitted Hall setting; the exact operations are retained."
        )
    return setting


def _symmetry(
    block: _Block,
    vectors: np.ndarray,
    metric_bound: float,
    consumed: set[str],
    assumptions: list[str],
    /,
) -> _Symmetry:
    """Admit the declared Hall symbol, IT number and operations of ``block``."""
    hall_symbol = _first_item(block, _HALL_TAGS)
    number_text = _first_item(block, _NUMBER_TAGS)
    consumed.update(_HALL_TAGS + _NUMBER_TAGS)
    declared_number = (
        None if number_text is None else int(_number(number_text, "IT number")[0])
    )
    hall_operations: tuple[ExactAffineOperation, ...] | None = None
    if hall_symbol is not None:
        try:
            hall_operations = HallSymbol(hall_symbol).group().operations
        except ValueError as error:
            raise _refuse(
                AdapterStatus.MALFORMED_SOURCE,
                f"Invalid Hall symbol {hall_symbol!r}: {error}",
            ) from error
    declared = _declared_operations(block, hall_operations, consumed, assumptions)
    operation_set, operations = _space_group(declared, hall_operations)
    _check_metric_invariance(operations, vectors, metric_bound)
    setting = _admitted_setting(declared_number, operation_set, assumptions)
    return _Symmetry(hall_symbol, declared_number, declared, operations, setting)


class _Sites(NamedTuple):
    """Ordered asymmetric-unit sites and the ``_atom_site`` loop they came from."""

    names: tuple[str, ...]
    rows: list[tuple[str | None, ...]]
    labels: tuple[str, ...]
    numbers: list[int]
    coordinates: np.ndarray


def _site_elements(
    labels: Sequence[str],
    symbols: Sequence[str | None],
    charge_policy: ChargedTypeSymbolPolicy,
    losses: list[AdapterLoss],
    /,
) -> list[int]:
    """Atomic numbers of explicit type symbols under the oxidation-label policy."""
    numbers: list[int] = []
    for label, symbol in zip(labels, symbols, strict=True):
        match = _TYPE_SYMBOL.match(symbol or "")
        if match is None or match.group(1) not in _ATOMIC_NUMBERS:
            raise _refuse(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                f"Site {label} type symbol {symbol!r} is not a chemical element.",
            )
        if match.group(2):
            if charge_policy == "refuse":
                raise _refuse(
                    AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                    f"Site {label} type symbol {symbol!r} carries an oxidation-state label.",
                )
            losses.append(
                AdapterLoss(
                    f"_atom_site_type_symbol[{label}]",
                    "import",
                    "dropped",
                    f"Oxidation-state label of {symbol!r} dropped; it is not a formal charge.",
                    changes_interpretation=True,
                )
            )
        numbers.append(_ATOMIC_NUMBERS[match.group(1)])
    return numbers


def _site_coordinates(
    names: tuple[str, ...],
    rows: list[tuple[str | None, ...]],
    site_count: int,
    uncertain: set[str],
    /,
) -> np.ndarray:
    """Fractional site coordinates, axis-major in source row order."""
    coordinates = np.zeros((site_count, 3), dtype=np.float64)
    for axis, tag in enumerate(
        ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")
    ):
        column = _column(names, rows, tag)
        if column is None:
            raise _refuse(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "CIF atom sites need fractional coordinates.",
            )
        for site, value in enumerate(column):
            coordinates[site, axis], has_uncertainty = _number(value, tag)
            if has_uncertainty:
                uncertain.add(tag)
    return coordinates


def _refuse_unresolved_sites(
    names: tuple[str, ...],
    rows: list[tuple[str | None, ...]],
    labels: Sequence[str],
    uncertain: set[str],
    assumptions: list[str],
    /,
) -> None:
    """Refuse partial/unknown occupancy, declared disorder and dummy sites."""
    occupancy = _column(names, rows, "_atom_site_occupancy")
    if occupancy is None:
        assumptions.append("Absent _atom_site_occupancy takes its dictionary default 1.")
    else:
        for label, value in zip(labels, occupancy, strict=True):
            if value is None:
                raise _refuse(
                    AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                    f"Site {label} occupancy is unknown.",
                )
            fraction, has_uncertainty = _number(value, "_atom_site_occupancy")
            if has_uncertainty:
                uncertain.add("_atom_site_occupancy")
            if abs(fraction - 1.0) > 1.0e-6:
                raise _refuse(
                    AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                    f"Site {label} has partial occupancy {fraction}; unresolved "
                    "occupancy is refused, not resolved into a structure.",
                )
    for tag in ("_atom_site_disorder_group", "_atom_site_disorder_assembly"):
        column = _column(names, rows, tag)
        if column is not None and any(value is not None for value in column):
            raise _refuse(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                f"CIF declares disorder ({tag}); unresolved disorder is refused.",
            )
    flags = _column(names, rows, "_atom_site_calc_flag")
    if flags is not None and any((value or "").lower() == "dum" for value in flags):
        raise _refuse(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC, "CIF declares dummy atom sites."
        )


def _atom_sites(
    block: _Block,
    charge_policy: ChargedTypeSymbolPolicy,
    consumed: set[str],
    uncertain: set[str],
    assumptions: list[str],
    losses: list[AdapterLoss],
    /,
) -> _Sites:
    """Admit uniquely labelled, explicitly typed, fully ordered atom sites."""
    site_loop = _loop(block, "_atom_site_label")
    if site_loop is None:
        raise _refuse(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC, "CIF declares no atom sites."
        )
    names, rows = site_loop
    consumed.update(name for name in names if name in _SITE_TAGS)
    labels = _column(names, rows, "_atom_site_label")
    symbols = _column(names, rows, "_atom_site_type_symbol")
    if labels is None or any(label is None for label in labels):
        raise _refuse(
            AdapterStatus.MALFORMED_SOURCE, "Every CIF atom site needs a label."
        )
    site_labels = tuple(label for label in labels if label is not None)
    if len(set(site_labels)) != len(site_labels):
        raise _refuse(AdapterStatus.INCONSISTENT_SOURCE, "CIF atom-site labels repeat.")
    if symbols is None or any(symbol is None for symbol in symbols):
        raise _refuse(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "CIF atom sites lack _atom_site_type_symbol; elements are never inferred from labels.",
        )
    numbers = _site_elements(site_labels, symbols, charge_policy, losses)
    coordinates = _site_coordinates(names, rows, len(site_labels), uncertain)
    _refuse_unresolved_sites(names, rows, site_labels, uncertain, assumptions)
    return _Sites(names, rows, site_labels, numbers, coordinates)


def _check_declared_multiplicities(
    sites: _Sites, multiplicities: Sequence[int], /
) -> None:
    """Refuse a declared site multiplicity that disagrees with the computed one."""
    for tag in (
        "_atom_site_symmetry_multiplicity",
        "_atom_site_site_symmetry_multiplicity",
    ):
        column = _column(sites.names, sites.rows, tag)
        if column is None:
            continue
        for label, value, computed in zip(
            sites.labels, column, multiplicities, strict=True
        ):
            if value is not None and int(_number(value, tag)[0]) != computed:
                raise _refuse(
                    AdapterStatus.INCONSISTENT_SOURCE,
                    f"Site {label} declares multiplicity {value}, computed {computed}.",
                )


def _retained_losses(uncertain: set[str], unconsumed: set[str], /) -> list[AdapterLoss]:
    """Losses for dropped standard uncertainties, then unconsumed tags by category."""
    losses: list[AdapterLoss] = []
    if uncertain:
        losses.append(
            AdapterLoss(
                "standard-uncertainties",
                "import",
                "dropped",
                "Standard uncertainties of "
                + ", ".join(sorted(uncertain))
                + " are not retained.",
                changes_interpretation=False,
            )
        )
    categories: dict[str, list[str]] = {}
    for tag in sorted(unconsumed):
        categories.setdefault(
            tag.split("_")[1] if tag.count("_") > 1 else tag, []
        ).append(tag)
    for category, tags in sorted(categories.items()):
        losses.append(
            AdapterLoss(
                f"_{category}",
                "import",
                "dropped",
                "Unmodeled CIF items dropped: " + ", ".join(tags),
                changes_interpretation=False,
            )
        )
    return losses


def cif_crystal_structure(
    text: str,
    /,
    *,
    block: str | None = None,
    site_tolerance: float = 0.01,
    metric_tolerance: float = 1.0e-3,
    charged_type_symbols: ChargedTypeSymbolPolicy = "refuse",
    source_id: str | None = None,
) -> CIFCrystalStructure:
    """Admit one CIF data block as an ordered crystal structure.

    ``site_tolerance`` (angstrom) merges symmetry images of one site (special
    positions) and refuses distinct sites closer than it.  ``metric_tolerance``
    bounds the relative violation of ``R^T G R = G`` by the declared cell.
    """
    if not isinstance(text, str):
        raise TypeError("CIF text must be a string.")
    tolerance = positive_finite_float(site_tolerance, "site_tolerance")
    metric_bound = positive_finite_float(metric_tolerance, "metric_tolerance")
    charge_policy = parse(
        charged_type_symbols, ChargedTypeSymbolPolicy, "charged_type_symbols"
    )
    selected = _select_block(_blocks(text), block)
    all_tags = _admitted_tags(selected)
    losses: list[AdapterLoss] = []
    assumptions: list[str] = [
        "CIF core dictionary units: cell lengths in angstrom, angles in degrees.",
    ]
    consumed: set[str] = set()
    uncertain: set[str] = set()
    lengths, angles, vectors = _cell(selected, consumed, uncertain)
    symmetry = _symmetry(selected, vectors, metric_bound, consumed, assumptions)
    sites = _atom_sites(selected, charge_policy, consumed, uncertain, assumptions, losses)
    expanded, site_index, operation_index, multiplicities = _expand(
        sites.coordinates, symmetry.operations, vectors, tolerance, sites.labels
    )
    _check_declared_multiplicities(sites, multiplicities)
    bonds = _bonds(selected, sites.labels, len(symmetry.declared), consumed)
    losses.extend(_retained_losses(uncertain, all_tags - consumed))
    digest = canonical_fingerprint({"kind": "cif-source", "text": text})
    resolved_source = digest if source_id is None else str(source_id)
    structure_id = canonical_fingerprint(
        {
            "kind": "cif-crystal-structure",
            "cell": array_tree_fingerprint(vectors),
            "operations": [
                [list(rotation), [str(value) for value in translation]]
                for rotation, translation in symmetry.operations
            ],
            "sites": array_tree_fingerprint(sites.coordinates),
            "numbers": sites.numbers,
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        _SOURCE_FORMAT,
        _TARGET_FORMAT,
        source_id=resolved_source,
        target_id=structure_id,
        preserved_fields=(
            "cell",
            "symmetry-operations",
            "atom-site-labels",
            "atom-site-elements",
            "atom-site-fractional-coordinates",
            *(("geom-bonds",) if bonds else ()),
        ),
        assumptions=assumptions,
        losses=losses,
    )
    return CIFCrystalStructure(
        cell_vectors=vectors,
        site_fractional=sites.coordinates,
        expanded_fractional=expanded,
        expanded_atomic_numbers=np.asarray(sites.numbers, dtype=np.int64)[site_index],
        expanded_site_indices=site_index,
        expanded_operation_indices=operation_index,
        setting=symmetry.setting,
        report=report,
        bonds=bonds,
        block_name=selected.name,
        cell_lengths=(float(lengths[0]), float(lengths[1]), float(lengths[2])),
        cell_angles=(float(angles[0]), float(angles[1]), float(angles[2])),
        site_labels=sites.labels,
        site_atomic_numbers=tuple(sites.numbers),
        site_multiplicities=multiplicities,
        operations=symmetry.operations,
        declared_operations=symmetry.declared,
        declared_international_number=symmetry.declared_number,
        declared_hall_symbol=symmetry.hall_symbol,
        site_tolerance=tolerance,
        source_id=resolved_source,
        structure_id=structure_id,
    )


def _expand(
    coordinates: np.ndarray,
    operations: Sequence[ExactAffineOperation],
    vectors: np.ndarray,
    tolerance: float,
    labels: Sequence[str],
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Distinct unit-cell images per site under minimum-image Cartesian distance."""
    rotations = np.asarray(
        [rotation for rotation, _ in operations], dtype=np.float64
    ).reshape((-1, 3, 3))
    translations = np.asarray(
        [[float(value) for value in translation] for _, translation in operations],
        dtype=np.float64,
    )
    shifts = np.asarray(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=np.float64,
    )

    def separation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        delta = first[:, None, :] - second[None, :, :]
        delta = delta - np.round(delta)
        images = (delta[:, :, None, :] + shifts[None, None, :, :]) @ vectors
        return np.min(np.linalg.norm(images, axis=-1), axis=-1)

    positions: list[np.ndarray] = []
    sites: list[int] = []
    operation_indices: list[int] = []
    multiplicities: list[int] = []
    for site, coordinate in enumerate(coordinates):
        images = np.einsum("oij,j->oi", rotations, coordinate) + translations
        images = images - np.floor(images)
        images = np.where(images >= 1.0, images - 1.0, images)
        kept: list[int] = []
        for index in range(images.shape[0]):
            if (
                kept
                and np.min(separation(images[index : index + 1], images[kept]))
                <= tolerance
            ):
                continue
            kept.append(index)
        if positions:
            existing = np.concatenate(positions, axis=0)
            if np.min(separation(images[kept], existing)) <= tolerance:
                raise _refuse(
                    AdapterStatus.INCONSISTENT_SOURCE,
                    f"Site {labels[site]} overlaps another site within {tolerance} Å "
                    "(unresolved disorder or duplicated sites).",
                )
        positions.append(images[kept])
        sites.extend([site] * len(kept))
        operation_indices.extend(kept)
        multiplicities.append(len(kept))
    return (
        np.concatenate(positions, axis=0),
        np.asarray(sites, dtype=np.int64),
        np.asarray(operation_indices, dtype=np.int64),
        tuple(multiplicities),
    )


_SYMMETRY_CODE = re.compile(r"^(\d+)(?:_(\d)(\d)(\d))?$")


def _symmetry_code(
    text: str | None, operation_count: int, /
) -> tuple[int, tuple[int, int, int]]:
    if text is None:
        return 0, (0, 0, 0)
    match = _SYMMETRY_CODE.match(text.strip())
    if match is None:
        raise _refuse(
            AdapterStatus.MALFORMED_SOURCE, f"Malformed CIF symmetry code {text!r}."
        )
    operation = int(match.group(1)) - 1
    if not 0 <= operation < operation_count:
        raise _refuse(
            AdapterStatus.INCONSISTENT_SOURCE,
            f"Symmetry code {text!r} names no declared operation.",
        )
    if match.group(2) is None:
        return operation, (0, 0, 0)
    return operation, (
        int(match.group(2)) - 5,
        int(match.group(3)) - 5,
        int(match.group(4)) - 5,
    )


def _bonds(
    block: _Block, labels: Sequence[str], operation_count: int, consumed: set[str], /
) -> tuple[CIFBond, ...]:
    found = _loop(block, "_geom_bond_atom_site_label_1")
    if found is None:
        return ()
    names, rows = found
    consumed.update(name for name in names if name in _BOND_TAGS)
    first = _column(names, rows, "_geom_bond_atom_site_label_1") or []
    second = _column(names, rows, "_geom_bond_atom_site_label_2")
    if second is None:
        raise _refuse(
            AdapterStatus.MALFORMED_SOURCE, "CIF bonds lack their second site label."
        )
    first_codes = _column(names, rows, "_geom_bond_site_symmetry_1") or [None] * len(rows)
    second_codes = _column(names, rows, "_geom_bond_site_symmetry_2") or [None] * len(
        rows
    )
    types = _column(names, rows, "_ccdc_geom_bond_type") or [None] * len(rows)
    index = {label: position for position, label in enumerate(labels)}
    bonds: list[CIFBond] = []
    for left, right, left_code, right_code, kind in zip(
        first, second, first_codes, second_codes, types, strict=True
    ):
        if left not in index or right not in index:
            raise _refuse(
                AdapterStatus.INCONSISTENT_SOURCE,
                f"CIF bond {left}-{right} names unknown sites.",
            )
        if kind is not None and kind.upper() not in _BOND_ORDERS:
            raise _refuse(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                f"Unsupported CIF bond type {kind!r}.",
            )
        left_operation, left_translation = _symmetry_code(left_code, operation_count)
        right_operation, right_translation = _symmetry_code(right_code, operation_count)
        bonds.append(
            CIFBond(
                first=index[left],
                second=index[right],
                first_operation=left_operation,
                second_operation=right_operation,
                first_translation=left_translation,
                second_translation=right_translation,
                order=None if kind is None else _BOND_ORDERS[kind.upper()],
            )
        )
    return tuple(bonds)


__all__ = [
    "CIFAtomDim",
    "CIFBond",
    "CIFCrystalStructure",
    "CIFSiteDim",
    "ChargedTypeSymbolPolicy",
    "cif_crystal_structure",
    "parse_symmetry_operation",
]
