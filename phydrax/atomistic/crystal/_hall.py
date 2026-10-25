#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact Hall notation and affine space-group algebra.

The Hall grammar and default-axis rules follow S. R. Hall, "Space-group
notation with an explicit origin", Acta Cryst. A37 (1981) 517-525, as
reproduced in International Tables for Crystallography Vol. B, Section A1.4.2.
A symbol ``L N_A^T ... (V)`` names a lattice symbol ``L`` (an optional leading
``-`` adds the inversion at the origin), up to four matrix symbols and an
optional origin-shift vector ``V`` in twelfths.

Everything here is host-only exact preparation.  An operation is the Seitz
pair ``x -> R x + t`` acting on fractional column coordinates, with an integer
row-major rotation ``R`` and a `fractions.Fraction` translation reduced modulo
one.  A space group is represented modulo its integer lattice ``Z^3``; the
non-integral lattice translations are its centering vectors.  This module owns
composition, bounded closure, group-law verification and the canonical
operation order; it never imports generated data, so the table generator can
use it to derive and check that data.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import final, Literal, NamedTuple, TypeAlias

import equinox as eqx

from ..._strict import StrictModule
from ...algebraic._exact_integer import exact_integer_determinant
from ...typing import parse


HallLatticeSymbol: TypeAlias = Literal["P", "A", "B", "C", "I", "R", "S", "T", "F"]
"""Hall lattice symbols; ``R``, ``S`` and ``T`` are the triple rhombohedral cells."""

type IntegerMatrix = tuple[int, int, int, int, int, int, int, int, int]
type RationalVector = tuple[Fraction, Fraction, Fraction]
type ExactAffineOperation = tuple[IntegerMatrix, RationalVector]

MAXIMUM_POINT_GROUP_ORDER = 48
"""Largest crystallographic point-group order (m-3m)."""

_ZERO = Fraction(0)
_HALF = Fraction(1, 2)
_THIRD = Fraction(1, 3)
_QUARTER = Fraction(1, 4)
_ZERO_VECTOR: RationalVector = (_ZERO, _ZERO, _ZERO)
IDENTITY_ROTATION: IntegerMatrix = (1, 0, 0, 0, 1, 0, 0, 0, 1)
INVERSION_ROTATION: IntegerMatrix = (-1, 0, 0, 0, -1, 0, 0, 0, -1)
IDENTITY_OPERATION: ExactAffineOperation = (IDENTITY_ROTATION, _ZERO_VECTOR)

# Table A1.4.2.2: lattice translations besides the zero vector.
_CENTERING: dict[str, tuple[RationalVector, ...]] = {
    "P": (),
    "A": ((_ZERO, _HALF, _HALF),),
    "B": ((_HALF, _ZERO, _HALF),),
    "C": ((_HALF, _HALF, _ZERO),),
    "I": ((_HALF, _HALF, _HALF),),
    "R": ((2 * _THIRD, _THIRD, _THIRD), (_THIRD, 2 * _THIRD, 2 * _THIRD)),
    "S": ((_THIRD, _THIRD, 2 * _THIRD), (2 * _THIRD, 2 * _THIRD, _THIRD)),
    "T": ((_THIRD, 2 * _THIRD, _THIRD), (2 * _THIRD, _THIRD, 2 * _THIRD)),
    "F": ((_ZERO, _HALF, _HALF), (_HALF, _ZERO, _HALF), (_HALF, _HALF, _ZERO)),
}

# Table A1.4.2.4: proper rotations along the principal axes.
_PRINCIPAL_ROTATIONS: dict[tuple[int, str], IntegerMatrix] = {
    (2, "x"): (1, 0, 0, 0, -1, 0, 0, 0, -1),
    (2, "y"): (-1, 0, 0, 0, 1, 0, 0, 0, -1),
    (2, "z"): (-1, 0, 0, 0, -1, 0, 0, 0, 1),
    (3, "x"): (1, 0, 0, 0, 0, -1, 0, 1, -1),
    (3, "y"): (-1, 0, 1, 0, 1, 0, -1, 0, 0),
    (3, "z"): (0, -1, 0, 1, -1, 0, 0, 0, 1),
    (4, "x"): (1, 0, 0, 0, 0, -1, 0, 1, 0),
    (4, "y"): (0, 0, 1, 0, 1, 0, -1, 0, 0),
    (4, "z"): (0, -1, 0, 1, 0, 0, 0, 0, 1),
    (6, "x"): (1, 0, 0, 0, 1, -1, 0, 1, 0),
    (6, "y"): (0, 0, 1, 0, 1, 0, -1, 0, 1),
    (6, "z"): (1, -1, 0, 1, 0, 0, 0, 0, 1),
}
# Table A1.4.2.4: twofold rotations along face diagonals perpendicular to the
# preceding principal axis (' and ").
_PRIMED_ROTATIONS: dict[str, IntegerMatrix] = {
    "x": (-1, 0, 0, 0, 0, -1, 0, -1, 0),
    "y": (0, 0, -1, 0, -1, 0, -1, 0, 0),
    "z": (0, -1, 0, -1, 0, 0, 0, 0, -1),
}
_DOUBLE_PRIMED_ROTATIONS: dict[str, IntegerMatrix] = {
    "x": (-1, 0, 0, 0, 0, 1, 0, 1, 0),
    "y": (0, 0, 1, 0, -1, 0, 1, 0, 0),
    "z": (0, 1, 0, 1, 0, 0, 0, 0, -1),
}
_BODY_DIAGONAL_ROTATION: IntegerMatrix = (0, 0, 1, 1, 0, 0, 0, 1, 0)

# Table A1.4.2.3: translation symbols.
_TRANSLATIONS: dict[str, RationalVector] = {
    "a": (_HALF, _ZERO, _ZERO),
    "b": (_ZERO, _HALF, _ZERO),
    "c": (_ZERO, _ZERO, _HALF),
    "n": (_HALF, _HALF, _HALF),
    "u": (_QUARTER, _ZERO, _ZERO),
    "v": (_ZERO, _QUARTER, _ZERO),
    "w": (_ZERO, _ZERO, _QUARTER),
    "d": (_QUARTER, _QUARTER, _QUARTER),
}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_ROTATION_ORDERS = (1, 2, 3, 4, 6)


def reduce_translation(vector: RationalVector, /) -> RationalVector:
    """Return the representative of a rational translation in ``[0, 1)^3``."""
    return (vector[0] % 1, vector[1] % 1, vector[2] % 1)


def add_vectors(left: RationalVector, right: RationalVector, /) -> RationalVector:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def multiply_rotations(left: IntegerMatrix, right: IntegerMatrix, /) -> IntegerMatrix:
    """Return the integer product ``left @ right`` of row-major 3x3 matrices."""
    return (
        left[0] * right[0] + left[1] * right[3] + left[2] * right[6],
        left[0] * right[1] + left[1] * right[4] + left[2] * right[7],
        left[0] * right[2] + left[1] * right[5] + left[2] * right[8],
        left[3] * right[0] + left[4] * right[3] + left[5] * right[6],
        left[3] * right[1] + left[4] * right[4] + left[5] * right[7],
        left[3] * right[2] + left[4] * right[5] + left[5] * right[8],
        left[6] * right[0] + left[7] * right[3] + left[8] * right[6],
        left[6] * right[1] + left[7] * right[4] + left[8] * right[7],
        left[6] * right[2] + left[7] * right[5] + left[8] * right[8],
    )


def rotate_vector(matrix: IntegerMatrix, vector: RationalVector, /) -> RationalVector:
    """Return ``matrix @ vector`` for an integer matrix and a rational vector."""
    return (
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    )


def negate_rotation(matrix: IntegerMatrix, /) -> IntegerMatrix:
    return (
        -matrix[0],
        -matrix[1],
        -matrix[2],
        -matrix[3],
        -matrix[4],
        -matrix[5],
        -matrix[6],
        -matrix[7],
        -matrix[8],
    )


def compose_operations(
    left: ExactAffineOperation, right: ExactAffineOperation, /
) -> ExactAffineOperation:
    """Return ``left o right`` modulo the integer lattice."""
    rotation = multiply_rotations(left[0], right[0])
    translation = add_vectors(rotate_vector(left[0], right[1]), left[1])
    return rotation, reduce_translation(translation)


def rotation_determinant(matrix: IntegerMatrix, /) -> int:
    """Return the exact determinant of an integer rotation through the owner."""
    return exact_integer_determinant(
        ((matrix[0], matrix[1], matrix[2]), (matrix[3], matrix[4], matrix[5]),
         (matrix[6], matrix[7], matrix[8]))
    )  # fmt: skip


def rotation_order(matrix: IntegerMatrix, /) -> int:
    """Return the crystallographic order of an integer rotation, refusing others."""
    power = matrix
    for order in range(1, 7):
        if power == IDENTITY_ROTATION:
            if order not in _ROTATION_ORDERS:
                break
            return order
        power = multiply_rotations(power, matrix)
    raise ValueError(
        "Integer matrix is not a crystallographic rotation of order 1-4 or 6."
    )


def close_affine_group(
    generators: Iterable[ExactAffineOperation], /, *, maximum_order: int
) -> frozenset[ExactAffineOperation]:
    """Return the finite group generated modulo ``Z^3``, refusing ``maximum_order``.

    Breadth-first right multiplication by the generators enumerates the
    generated monoid, which equals the generated group because every element
    of a finite set of invertible affine maps has finite order.
    """
    reduced = tuple(
        (rotation, reduce_translation(translation))
        for rotation, translation in generators
    )
    elements = {IDENTITY_OPERATION}
    frontier = [IDENTITY_OPERATION]
    while frontier:
        discovered: list[ExactAffineOperation] = []
        for element in frontier:
            for generator in reduced:
                product = compose_operations(element, generator)
                if product in elements:
                    continue
                elements.add(product)
                discovered.append(product)
                if len(elements) > maximum_order:
                    raise ValueError(
                        f"Affine group closure exceeds maximum_order={maximum_order}."
                    )
        frontier = discovered
    return frozenset(elements)


class CanonicalAffineGroup(NamedTuple):
    """Canonically ordered operations of one space group modulo ``Z^3``.

    ``operations`` is centering-major: entry ``c * point_group_order + r`` is
    coset representative ``r`` translated by centering vector ``c``.  The zero
    centering vector comes first, so the leading ``point_group_order`` entries
    are the coset representatives; the identity is entry zero.  Representatives
    follow the rotation order (identity first, then lexicographic integer
    entries) and each carries the lexicographically least translation in its
    coset.
    """

    operations: tuple[ExactAffineOperation, ...]
    centering_vectors: tuple[RationalVector, ...]
    point_group_order: int


def canonical_affine_group(
    group: frozenset[ExactAffineOperation], /
) -> CanonicalAffineGroup:
    """Verify the space-group laws of ``group`` and return its canonical order."""
    if IDENTITY_OPERATION not in group:
        raise ValueError("Space-group operations must contain the identity.")
    for left in group:
        for right in group:
            if compose_operations(left, right) not in group:
                raise ValueError(
                    "Space-group operations are not closed under composition."
                )
    cosets: dict[IntegerMatrix, list[RationalVector]] = {}
    for rotation, translation in group:
        cosets.setdefault(rotation, []).append(translation)
    for rotation in cosets:
        if abs(rotation_determinant(rotation)) != 1:
            raise ValueError("Space-group rotations must be unimodular.")
        rotation_order(rotation)
    if len(cosets) > MAXIMUM_POINT_GROUP_ORDER:
        raise ValueError("Space-group point group exceeds the crystallographic bound.")
    centering = tuple(sorted(cosets[IDENTITY_ROTATION]))
    representatives: list[ExactAffineOperation] = []
    for rotation in sorted(cosets, key=lambda value: (value != IDENTITY_ROTATION, value)):
        translations = cosets[rotation]
        if len(translations) != len(centering):
            raise RuntimeError("Closed space-group cosets must share the centering size.")
        representatives.append((rotation, min(translations)))
    operations = tuple(
        (rotation, reduce_translation(add_vectors(translation, vector)))
        for vector in centering
        for rotation, translation in representatives
    )
    if frozenset(operations) != group:
        raise RuntimeError("Canonical space-group order lost an operation.")
    return CanonicalAffineGroup(operations, centering, len(representatives))


class _MatrixSymbol(NamedTuple):
    order: int
    improper: bool
    axis: str | None
    screw: int
    translation: RationalVector


def _matrix_symbol(token: str, /) -> _MatrixSymbol:
    improper = token.startswith("-")
    body = token[1:] if improper else token
    if not body or not body[0].isdigit() or int(body[0]) not in _ROTATION_ORDERS:
        raise ValueError(
            f"Hall matrix symbol {token!r} needs a rotation order 1, 2, 3, 4 or 6."
        )
    order = int(body[0])
    rest = body[1:]
    axis: str | None = None
    if rest[:1] in ("x", "y", "z", "'", '"', "*"):
        axis = rest[0]
        rest = rest[1:]
    screw = 0
    if rest[:1].isdigit():
        screw = int(rest[0])
        rest = rest[1:]
        if not 0 < screw < order:
            raise ValueError(f"Hall screw component in {token!r} must lie in 1..N-1.")
    translation = _ZERO_VECTOR
    for letter in rest:
        if letter not in _TRANSLATIONS:
            raise ValueError(
                f"Hall matrix symbol {token!r} has an unknown translation {letter!r}."
            )
        translation = add_vectors(translation, _TRANSLATIONS[letter])
    return _MatrixSymbol(order, improper, axis, screw, translation)


def _resolve_axis(
    symbol: _MatrixSymbol, position: int, previous: _MatrixSymbol | None, /
) -> str:
    """Apply the Hall default-axis rules to one matrix symbol."""
    if symbol.axis is not None:
        return symbol.axis
    if symbol.order == 1 or position == 0:
        return "z"
    if position == 1 and symbol.order == 2 and previous is not None:
        if previous.order in (2, 4):
            return "x"
        if previous.order in (3, 6):
            return "'"
    if position == 2 and symbol.order == 3:
        return "*"
    raise ValueError(
        f"Hall matrix symbol at position {position} with order {symbol.order} "
        "needs an explicit axis."
    )


def _rotation(
    symbol: _MatrixSymbol, axis: str, previous_axis: str | None, /
) -> IntegerMatrix:
    if symbol.order == 1:
        if axis not in _AXIS_INDEX:
            raise ValueError("The Hall identity symbol admits only a principal axis.")
        return IDENTITY_ROTATION
    match axis:
        case "x" | "y" | "z":
            return _PRINCIPAL_ROTATIONS[(symbol.order, axis)]
        case "'" | '"':
            if symbol.order != 2:
                raise ValueError("Primed Hall axes admit only twofold rotations.")
            # After a body-diagonal threefold (rhombohedral axes) the primed
            # twofold lies along a - b, the same matrix as after z.
            reference = "z" if previous_axis == "*" else previous_axis
            if reference is None or reference not in _AXIS_INDEX:
                raise ValueError("A primed Hall axis must follow a rotation-axis symbol.")
            table = _PRIMED_ROTATIONS if axis == "'" else _DOUBLE_PRIMED_ROTATIONS
            return table[reference]
        case "*":
            if symbol.order != 3:
                raise ValueError(
                    "The Hall body-diagonal axis admits only threefold rotations."
                )
            return _BODY_DIAGONAL_ROTATION
        case _:
            raise ValueError(f"Unknown Hall axis {axis!r}.")


def _screw_translation(symbol: _MatrixSymbol, axis: str, /) -> RationalVector:
    if symbol.screw == 0:
        return _ZERO_VECTOR
    if axis not in _AXIS_INDEX:
        raise ValueError("Hall screw components are defined only along x, y or z.")
    vector = [_ZERO, _ZERO, _ZERO]
    vector[_AXIS_INDEX[axis]] = Fraction(symbol.screw, symbol.order)
    return (vector[0], vector[1], vector[2])


def _origin_shift(text: str, /) -> tuple[str, RationalVector]:
    if "(" not in text:
        if ")" in text:
            raise ValueError("Hall symbol has an unmatched ')'.")
        return text, _ZERO_VECTOR
    body, _, shift = text.partition("(")
    if not shift.endswith(")") or ")" in shift[:-1] or "(" in shift:
        raise ValueError("Hall origin shift must be one trailing '(vx vy vz)' group.")
    components = shift[:-1].split()
    if len(components) != 3 or "," in shift:
        # The general (x,y,z)-style change-of-basis extension is outside the
        # admitted grammar; refuse it rather than reinterpret it as a shift.
        raise ValueError(
            "Hall origin shift must contain three integer twelfths; general "
            "change-of-basis operators are not admitted."
        )
    values: list[Fraction] = []
    for component in components:
        digits = component[1:] if component[:1] in "+-" else component
        if not digits.isdigit():
            raise ValueError(
                f"Hall origin shift component {component!r} is not an integer."
            )
        sign = -1 if component.startswith("-") else 1
        values.append(Fraction(sign * int(digits), 12))
    return body, (values[0], values[1], values[2])


@final
class HallSymbol(StrictModule):
    """Parsed Hall symbol: centering, exact generators and origin shift.

    ``generator_rotations[k]`` and ``generator_translations[k]`` are Seitz
    operators already expressed relative to the origin selected by
    ``origin_shift``; the inversion of a centrosymmetric lattice symbol is the
    first generator.  ``symbol`` is the whitespace-normalized source spelling
    and is not a scientific identity; the generated operations are.
    """

    symbol: str = eqx.field(static=True)
    lattice: HallLatticeSymbol = eqx.field(static=True)
    centrosymmetric: bool = eqx.field(static=True)
    centering_vectors: tuple[RationalVector, ...] = eqx.field(static=True)
    generator_rotations: tuple[IntegerMatrix, ...] = eqx.field(static=True)
    generator_translations: tuple[RationalVector, ...] = eqx.field(static=True)
    origin_shift: RationalVector = eqx.field(static=True)

    def __init__(self, symbol: str, /) -> None:
        if not isinstance(symbol, str):
            raise TypeError("Hall symbol must be a string.")
        text = " ".join(symbol.split())
        if not text:
            raise ValueError("Hall symbol must be non-empty.")
        body, shift = _origin_shift(text)
        tokens = body.split()
        lattice_token = tokens[0]
        centrosymmetric = lattice_token.startswith("-")
        lattice = parse(
            lattice_token[1:] if centrosymmetric else lattice_token,
            HallLatticeSymbol,
            "Hall lattice symbol",
        )
        matrix_tokens = tokens[1:]
        if not 1 <= len(matrix_tokens) <= 4:
            raise ValueError("Hall symbols contain between one and four matrix symbols.")
        rotations: list[IntegerMatrix] = []
        translations: list[RationalVector] = []
        if centrosymmetric:
            rotations.append(INVERSION_ROTATION)
            translations.append(_ZERO_VECTOR)
        previous: _MatrixSymbol | None = None
        previous_axis: str | None = None
        for position, token in enumerate(matrix_tokens):
            matrix_symbol = _matrix_symbol(token)
            axis = _resolve_axis(matrix_symbol, position, previous)
            rotation = _rotation(matrix_symbol, axis, previous_axis)
            rotations.append(
                negate_rotation(rotation) if matrix_symbol.improper else rotation
            )
            translations.append(
                add_vectors(
                    matrix_symbol.translation, _screw_translation(matrix_symbol, axis)
                )
            )
            if matrix_symbol.order != 1:
                previous = matrix_symbol
                previous_axis = axis
        # The symbol's operators act on coordinates x_symbol = x - V, so in the
        # target coordinates every operator becomes (R, t - (R - I) V).  This
        # sign is checked against every shifted entry of the admitted table.
        self.symbol = text
        self.lattice = lattice
        self.centrosymmetric = centrosymmetric
        self.centering_vectors = _CENTERING[lattice]
        self.generator_rotations = tuple(rotations)
        self.generator_translations = tuple(
            reduce_translation(
                add_vectors(
                    add_vectors(translation, shift),
                    rotate_vector(negate_rotation(rotation), shift),
                )
            )
            for rotation, translation in zip(rotations, translations, strict=True)
        )
        self.origin_shift = shift

    def generators(self) -> tuple[ExactAffineOperation, ...]:
        """Return centering translations followed by the matrix-symbol generators."""
        centering = tuple(
            (IDENTITY_ROTATION, vector) for vector in self.centering_vectors
        )
        return centering + tuple(
            zip(self.generator_rotations, self.generator_translations, strict=True)
        )

    def group(self) -> CanonicalAffineGroup:
        """Return the verified canonical space group generated by this symbol."""
        bound = MAXIMUM_POINT_GROUP_ORDER * (len(self.centering_vectors) + 1)
        return canonical_affine_group(
            close_affine_group(self.generators(), maximum_order=bound)
        )


__all__ = [
    "CanonicalAffineGroup",
    "ExactAffineOperation",
    "HallLatticeSymbol",
    "HallSymbol",
    "IDENTITY_OPERATION",
    "IDENTITY_ROTATION",
    "INVERSION_ROTATION",
    "IntegerMatrix",
    "MAXIMUM_POINT_GROUP_ORDER",
    "RationalVector",
    "add_vectors",
    "canonical_affine_group",
    "close_affine_group",
    "compose_operations",
    "multiply_rotations",
    "negate_rotation",
    "reduce_translation",
    "rotate_vector",
    "rotation_determinant",
    "rotation_order",
]
