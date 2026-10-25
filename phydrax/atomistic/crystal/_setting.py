#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact space-group settings, setting transformations and derived properties.

A `SpaceGroupSetting` is always derived from one admitted Hall entry (its
*reference*) and an exact change of basis.  Its scientific identity is the
International number together with the canonical exact operation set modulo
the integer lattice; Hall symbols and setting-choice labels are provenance.
All preparation is host-only exact arithmetic; numerical consumers receive
explicit host arrays or the JAX action of `SpaceGroupSetting.apply`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import lcm
from numbers import Rational
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...algebraic._exact_integer import (
    exact_integer_determinant,
    smith_normal_decomposition,
)
from ...ein import contract
from ...typing import Dim, HostInt64, parse
from . import _space_group_data as data
from ._hall import (
    add_vectors,
    canonical_affine_group,
    CanonicalAffineGroup,
    compose_operations,
    ExactAffineOperation,
    IntegerMatrix,
    INVERSION_ROTATION,
    negate_rotation,
    RationalVector,
    reduce_translation,
    rotate_vector,
    rotation_determinant,
    rotation_order,
)


CrystalSystem: TypeAlias = Literal[
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]
MonoclinicAxis: TypeAlias = Literal["a", "b", "c", "-a", "-b", "-c"]
"""Monoclinic unique axis of an admitted setting; ``-`` marks the reversed setting."""
OrthorhombicAxes: TypeAlias = Literal["abc", "ba-c", "cab", "-cba", "bca", "a-cb"]
RhombohedralAxes: TypeAlias = Literal["hexagonal", "rhombohedral"]
OriginChoice: TypeAlias = Literal[1, 2]
CellChoice: TypeAlias = Literal[1, 2, 3]

type RationalMatrix = tuple[
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    Fraction,
]

HALL_ENTRY_COUNT = len(data.HALL_ENTRIES)
DEFAULT_MAXIMUM_OPERATION_COUNT = 192
"""Default operation bound of a transformed setting (Fm-3m modulo its lattice)."""


class OperationDim(Dim, minimum=1):
    """Space-group operations modulo the integer lattice."""


_ONE = Fraction(1)
_NIL = Fraction(0)
_IDENTITY_MATRIX: RationalMatrix = (_ONE, _NIL, _NIL, _NIL, _ONE, _NIL, _NIL, _NIL, _ONE)
_ZERO_VECTOR: RationalVector = (Fraction(0), Fraction(0), Fraction(0))


def _rational(value: object, name: str, /) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Rational):
        raise TypeError(f"{name} must contain exact integers or fractions.")
    return Fraction(int(value.numerator), int(value.denominator))


def _rational_product(left: RationalMatrix, right: RationalMatrix, /) -> RationalMatrix:
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


def _rational_apply(matrix: RationalMatrix, vector: RationalVector, /) -> RationalVector:
    return (
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    )


def _as_rational_matrix(matrix: IntegerMatrix, /) -> RationalMatrix:
    return (
        Fraction(matrix[0]), Fraction(matrix[1]), Fraction(matrix[2]),
        Fraction(matrix[3]), Fraction(matrix[4]), Fraction(matrix[5]),
        Fraction(matrix[6]), Fraction(matrix[7]), Fraction(matrix[8]),
    )  # fmt: skip


def _as_integer_matrix(matrix: RationalMatrix, /) -> IntegerMatrix | None:
    if any(value.denominator != 1 for value in matrix):
        return None
    return (
        matrix[0].numerator, matrix[1].numerator, matrix[2].numerator,
        matrix[3].numerator, matrix[4].numerator, matrix[5].numerator,
        matrix[6].numerator, matrix[7].numerator, matrix[8].numerator,
    )  # fmt: skip


def _scaled_integer_rows(
    matrix: RationalMatrix, /
) -> tuple[int, tuple[tuple[int, int, int], ...]]:
    scale = lcm(*(value.denominator for value in matrix))
    values = [value * scale for value in matrix]
    rows = tuple(
        (
            values[3 * r].numerator,
            values[3 * r + 1].numerator,
            values[3 * r + 2].numerator,
        )
        for r in range(3)
    )
    return scale, rows


def _rational_inverse(matrix: RationalMatrix, /) -> RationalMatrix:
    """Invert ``matrix = Q / s`` exactly from the Smith form ``U Q V = D``."""
    scale, rows = _scaled_integer_rows(matrix)
    diagonal, left, right = smith_normal_decomposition(rows)
    pivots = [int(diagonal[index, index]) for index in range(3)]
    if any(pivot == 0 for pivot in pivots):
        raise ValueError("Setting transformation matrix must be nonsingular.")
    # Q^-1 = V D^-1 U, so (Q / s)^-1 = s V D^-1 U.
    entries = [
        scale
        * sum(
            (
                Fraction(int(right[row, k]) * int(left[k, column]), pivots[k])
                for k in range(3)
            ),
            Fraction(0),
        )
        for row in range(3)
        for column in range(3)
    ]
    return (
        entries[0], entries[1], entries[2], entries[3], entries[4],
        entries[5], entries[6], entries[7], entries[8],
    )  # fmt: skip


@final
class SettingTransformation(StrictModule):
    """Exact change of setting ``(a', b', c') = (a, b, c) P`` with new origin ``p``.

    ``matrix`` holds ``P`` row-major (its columns are the new basis vectors in
    old fractional coordinates) and ``origin_shift`` is the new origin in old
    fractional coordinates, so fractional coordinates map as ``x = P x' + p``
    (International Tables Vol. A, Section 1.5).  Entries are exact integers or
    `fractions.Fraction`; floating values are refused.
    """

    matrix: RationalMatrix = eqx.field(static=True)
    origin_shift: RationalVector = eqx.field(static=True)
    inverse_matrix: RationalMatrix = eqx.field(static=True)
    determinant: Fraction = eqx.field(static=True)
    transformation_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: Sequence[Sequence[int | Fraction]] | None = None,
        origin_shift: Sequence[int | Fraction] | None = None,
    ) -> None:
        if matrix is None:
            matrix_ = _IDENTITY_MATRIX
        else:
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                raise ValueError("Setting transformation matrix must be 3x3.")
            values = [
                _rational(value, "transformation matrix")
                for row in matrix
                for value in row
            ]
            matrix_ = (
                values[0], values[1], values[2], values[3], values[4],
                values[5], values[6], values[7], values[8],
            )  # fmt: skip
        if origin_shift is None:
            shift = _ZERO_VECTOR
        else:
            if len(origin_shift) != 3:
                raise ValueError("Setting origin shift must have three components.")
            shift = (
                _rational(origin_shift[0], "origin shift"),
                _rational(origin_shift[1], "origin shift"),
                _rational(origin_shift[2], "origin shift"),
            )
        scale, rows = _scaled_integer_rows(matrix_)
        determinant = Fraction(exact_integer_determinant(rows), scale**3)
        if determinant == 0:
            raise ValueError("Setting transformation matrix must be nonsingular.")
        self.matrix = matrix_
        self.origin_shift = shift
        self.inverse_matrix = _rational_inverse(matrix_)
        self.determinant = determinant
        self.transformation_id = canonical_fingerprint(
            {
                "kind": "space-group-setting-transformation",
                "matrix": [str(value) for value in matrix_],
                "origin_shift": [str(value) for value in shift],
            }
        )

    @property
    def orientation_preserving(self) -> bool:
        return self.determinant > 0

    @property
    def is_identity(self) -> bool:
        return self.matrix == _IDENTITY_MATRIX and self.origin_shift == _ZERO_VECTOR

    def followed_by(self, other: SettingTransformation, /) -> SettingTransformation:
        """Return the composite of this change of setting and then ``other``."""
        if not isinstance(other, SettingTransformation):
            raise TypeError("other must be a SettingTransformation.")
        matrix = _rational_product(self.matrix, other.matrix)
        moved = _rational_apply(self.matrix, other.origin_shift)
        shift = tuple(moved[i] + self.origin_shift[i] for i in range(3))
        return SettingTransformation(
            tuple(tuple(matrix[3 * r : 3 * r + 3]) for r in range(3)), shift
        )


def _admitted_entry(hall_number: int, /) -> data.AdmittedHallEntry:
    if isinstance(hall_number, bool) or not isinstance(hall_number, int):
        raise TypeError("hall_number must be an integer.")
    if not 1 <= hall_number <= HALL_ENTRY_COUNT:
        raise ValueError(f"hall_number must lie in 1..{HALL_ENTRY_COUNT}.")
    return data.HALL_ENTRIES[hall_number - 1]


def _decoded_operations(
    entry: data.AdmittedHallEntry, /
) -> tuple[ExactAffineOperation, ...]:
    denominator = data.TRANSLATION_DENOMINATOR
    operations: list[ExactAffineOperation] = []
    for index in entry.operation_indices:
        row = data.OPERATIONS[index]
        rotation: IntegerMatrix = (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
        )  # fmt: skip
        translation: RationalVector = (
            Fraction(row[9], denominator),
            Fraction(row[10], denominator),
            Fraction(row[11], denominator),
        )
        operations.append((rotation, translation))
    return tuple(operations)


@lru_cache(maxsize=None)
def _admitted_group(hall_number: int, /) -> CanonicalAffineGroup:
    """Return the group-law-verified canonical group of one admitted entry."""
    operations = _decoded_operations(_admitted_entry(hall_number))
    group = canonical_affine_group(frozenset(operations))
    if group.operations != operations:
        raise RuntimeError(
            f"Admitted Hall entry {hall_number} is not canonically ordered."
        )
    return group


@lru_cache(maxsize=1)
def _admitted_identity_index() -> dict[frozenset[ExactAffineOperation], tuple[int, ...]]:
    """Map every admitted exact operation set to the Hall numbers carrying it."""
    index: dict[frozenset[ExactAffineOperation], list[int]] = {}
    for entry in data.HALL_ENTRIES:
        key = frozenset(_decoded_operations(entry))
        index.setdefault(key, []).append(entry.hall_number)
    return {key: tuple(values) for key, values in index.items()}


def _shifted(
    operations: frozenset[ExactAffineOperation], shift: RationalVector, /
) -> frozenset[ExactAffineOperation]:
    """Return the operations after moving the origin to ``shift`` (``x = x' + p``)."""
    negative: RationalVector = (-shift[0], -shift[1], -shift[2])
    return frozenset(
        (
            rotation,
            reduce_translation(
                add_vectors(
                    add_vectors(translation, rotate_vector(rotation, shift)), negative
                )
            ),
        )
        for rotation, translation in operations
    )


@lru_cache(maxsize=None)
def _mirror_international_number(hall_number: int, /) -> int:
    """Return the International number of the mirror image of an admitted entry.

    A group with an improper operation is its own mirror image.  For a Sohncke
    group the mirrored operations ``(R, -t)`` are matched exactly against the
    admitted table, first directly and then after origin shifts on the bounded
    grid ``(1 / 2D) Z^3`` with ``D`` the group's translation denominator.  An
    unmatched mirror is refused rather than assigned a guessed type.
    """
    entry = _admitted_entry(hall_number)
    group = _admitted_group(hall_number)
    if any(rotation_determinant(rotation) != 1 for rotation, _ in group.operations):
        return entry.international_number
    mirrored = frozenset(
        (rotation, reduce_translation((-t[0], -t[1], -t[2])))
        for rotation, t in group.operations
    )
    index = _admitted_identity_index()
    denominator = 2 * lcm(
        *(
            value.denominator
            for _, translation in group.operations
            for value in translation
        )
    )
    for numerators in product(range(denominator), repeat=3):
        shift: RationalVector = (
            Fraction(numerators[0], denominator),
            Fraction(numerators[1], denominator),
            Fraction(numerators[2], denominator),
        )
        matches = index.get(_shifted(mirrored, shift))
        if matches is not None:
            return data.HALL_ENTRIES[matches[0] - 1].international_number
    raise ValueError(
        f"The mirror image of Hall entry {hall_number} matches no admitted setting "
        "within the bounded origin search; its enantiomorph type is undetermined."
    )


def _transformed_group(
    group: CanonicalAffineGroup,
    transformation: SettingTransformation,
    maximum_operation_count: int,
    /,
) -> CanonicalAffineGroup:
    """Express ``group`` in the setting ``x = P x' + p`` modulo the new lattice."""
    matrix, inverse, shift = (
        transformation.matrix,
        transformation.inverse_matrix,
        transformation.origin_shift,
    )
    lattice = frozenset((_ZERO_VECTOR, *group.centering_vectors))
    for column in range(3):
        basis = (matrix[column], matrix[3 + column], matrix[6 + column])
        if reduce_translation(basis) not in lattice:
            raise ValueError(
                "Every new basis vector must be a lattice translation of the setting."
            )
    centering_count = abs(transformation.determinant) * len(lattice)
    if centering_count.denominator != 1:
        raise RuntimeError(
            "Lattice index of an admitted change of basis is not integral."
        )
    if centering_count.numerator * group.point_group_order > maximum_operation_count:
        raise ValueError(
            f"Transformed setting needs {centering_count.numerator * group.point_group_order} "
            f"operations, exceeding maximum_operation_count={maximum_operation_count}."
        )
    axes: tuple[RationalVector, ...] = (
        (_ONE, _NIL, _NIL),
        (_NIL, _ONE, _NIL),
        (_NIL, _NIL, _ONE),
    )
    generators = [
        reduce_translation(_rational_apply(inverse, vector))
        for vector in (*axes, *group.centering_vectors)
    ]
    centering = {_ZERO_VECTOR}
    frontier = [_ZERO_VECTOR]
    while frontier:
        discovered: list[RationalVector] = []
        for vector in frontier:
            for generator in generators:
                candidate = reduce_translation(
                    (
                        vector[0] + generator[0],
                        vector[1] + generator[1],
                        vector[2] + generator[2],
                    )
                )
                if candidate not in centering:
                    centering.add(candidate)
                    discovered.append(candidate)
        if len(centering) > centering_count.numerator:
            raise RuntimeError("Transformed centering exceeds the exact lattice index.")
        frontier = discovered
    if len(centering) != centering_count.numerator:
        raise RuntimeError(
            "Transformed centering does not match the exact lattice index."
        )
    operations: set[ExactAffineOperation] = set()
    for rotation, translation in group.operations[: group.point_group_order]:
        conjugated = _as_integer_matrix(
            _rational_product(
                _rational_product(inverse, _as_rational_matrix(rotation)), matrix
            )
        )
        if conjugated is None:
            raise ValueError(
                "The new basis is not invariant under the point group; the "
                "transformed rotations are not integral."
            )
        moved = add_vectors(rotate_vector(rotation, shift), translation)
        image = _rational_apply(
            inverse, (moved[0] - shift[0], moved[1] - shift[1], moved[2] - shift[2])
        )
        for vector in centering:
            operations.add(
                (
                    conjugated,
                    reduce_translation(
                        (image[0] + vector[0], image[1] + vector[1], image[2] + vector[2])
                    ),
                )
            )
    return canonical_affine_group(frozenset(operations))


def _crystal_system(rotations: tuple[IntegerMatrix, ...], /) -> CrystalSystem:
    """Classify the Laue class through the orders of the proper rotations ``det(R) R``."""
    proper = {
        rotation if rotation_determinant(rotation) == 1 else negate_rotation(rotation)
        for rotation in rotations
    }
    orders = Counter(rotation_order(rotation) for rotation in proper)
    if orders[3] >= 8:
        return "cubic"
    if orders[6]:
        return "hexagonal"
    if orders[3]:
        return "trigonal"
    if orders[4]:
        return "tetragonal"
    if orders[2] >= 3:
        return "orthorhombic"
    if orders[2] == 1:
        return "monoclinic"
    return "triclinic"


@final
class SpaceGroupSetting(StrictModule):
    """One exact space-group setting derived from an admitted Hall entry.

    ``SpaceGroupSetting(hall_number)`` is the admitted setting itself;
    ``transformation`` re-expresses it through ``x = P x' + p``.  Operations are
    canonical and centering-major (see `CanonicalAffineGroup`): entry
    ``c * point_group_order + r`` is coset representative ``r`` translated by
    centering vector ``c``, and entry zero is the identity.

    ``setting_id`` is the scientific identity (International number and exact
    operations); two settings with equal ``setting_id`` are the same setting
    whatever their reference or transformation provenance.  ``hall_numbers``
    lists every admitted entry with exactly these operations.  The choice
    fields come from the reference entry for an untransformed setting and
    otherwise from the lowest matching admitted entry; they are ``None`` when
    no admitted entry matches.  An orientation-reversing transformation maps a
    chiral type to its enantiomorph, whose number is derived exactly.
    """

    international_number: int = eqx.field(static=True)
    reference_hall_number: int = eqx.field(static=True)
    transformation: SettingTransformation
    hall_numbers: tuple[int, ...] = eqx.field(static=True)
    hall_symbol: str | None = eqx.field(static=True)
    origin_choice: OriginChoice | None = eqx.field(static=True)
    cell_choice: CellChoice | None = eqx.field(static=True)
    unique_axis: MonoclinicAxis | None = eqx.field(static=True)
    orthorhombic_axes: OrthorhombicAxes | None = eqx.field(static=True)
    rhombohedral_axes: RhombohedralAxes | None = eqx.field(static=True)
    crystal_system: CrystalSystem = eqx.field(static=True)
    rotations: tuple[IntegerMatrix, ...] = eqx.field(static=True)
    translation_numerators: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    translation_denominator: int = eqx.field(static=True)
    centering_count: int = eqx.field(static=True)
    point_group_order: int = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    setting_id: str = eqx.field(static=True)

    def __init__(
        self,
        hall_number: int,
        /,
        *,
        transformation: SettingTransformation | None = None,
        maximum_operation_count: int = DEFAULT_MAXIMUM_OPERATION_COUNT,
    ) -> None:
        entry = _admitted_entry(hall_number)
        transformation_ = (
            SettingTransformation() if transformation is None else transformation
        )
        if not isinstance(transformation_, SettingTransformation):
            raise TypeError("transformation must be a SettingTransformation or None.")
        if isinstance(maximum_operation_count, bool) or not isinstance(
            maximum_operation_count, int
        ):
            raise TypeError("maximum_operation_count must be an integer.")
        if maximum_operation_count < 1:
            raise ValueError("maximum_operation_count must be positive.")
        reference = _admitted_group(hall_number)
        group = (
            reference
            if transformation_.is_identity
            else _transformed_group(reference, transformation_, maximum_operation_count)
        )
        if len(group.operations) > maximum_operation_count:
            raise ValueError(
                f"Setting has {len(group.operations)} operations, exceeding "
                f"maximum_operation_count={maximum_operation_count}."
            )
        number = (
            entry.international_number
            if transformation_.orientation_preserving
            else _mirror_international_number(hall_number)
        )
        matches = _admitted_identity_index().get(frozenset(group.operations), ())
        if transformation_.is_identity and hall_number not in matches:
            raise RuntimeError("An admitted setting is missing from its identity index.")
        if any(data.HALL_ENTRIES[h - 1].international_number != number for h in matches):
            raise RuntimeError("An exact admitted match disagrees with the derived type.")
        label = (
            entry
            if transformation_.is_identity
            else data.HALL_ENTRIES[matches[0] - 1]
            if matches
            else None
        )
        denominator = lcm(
            *(
                value.denominator
                for _, translation in group.operations
                for value in translation
            )
        )
        rotations = tuple(rotation for rotation, _ in group.operations)
        numerators = tuple(
            (
                (translation[0] * denominator).numerator,
                (translation[1] * denominator).numerator,
                (translation[2] * denominator).numerator,
            )
            for _, translation in group.operations
        )
        self.international_number = number
        self.reference_hall_number = hall_number
        self.transformation = transformation_
        self.hall_numbers = matches
        self.hall_symbol = None if label is None else label.hall_symbol
        self.origin_choice = (
            None
            if label is None
            else parse(label.origin_choice, OriginChoice | None, "origin_choice")
        )
        self.cell_choice = (
            None
            if label is None
            else parse(label.cell_choice, CellChoice | None, "cell_choice")
        )
        self.unique_axis = (
            None
            if label is None
            else parse(label.unique_axis, MonoclinicAxis | None, "unique_axis")
        )
        self.orthorhombic_axes = (
            None
            if label is None
            else parse(
                label.orthorhombic_axes, OrthorhombicAxes | None, "orthorhombic_axes"
            )
        )
        self.rhombohedral_axes = (
            None
            if label is None
            else parse(
                label.rhombohedral_axes, RhombohedralAxes | None, "rhombohedral_axes"
            )
        )
        self.crystal_system = _crystal_system(rotations)
        self.rotations = rotations
        self.translation_numerators = numerators
        self.translation_denominator = denominator
        self.centering_count = len(group.centering_vectors)
        self.point_group_order = group.point_group_order
        self.data_id = data.SPACE_GROUP_DATA_ID
        self.setting_id = canonical_fingerprint(
            {
                "kind": "space-group-setting",
                "international_number": number,
                "rotations": [list(rotation) for rotation in rotations],
                "translation_numerators": [list(row) for row in numerators],
                "translation_denominator": denominator,
            }
        )

    @property
    def operation_count(self) -> int:
        return len(self.rotations)

    @property
    def determinants(self) -> tuple[int, ...]:
        """Exact rotation determinants, ``+1`` proper and ``-1`` improper."""
        return tuple(rotation_determinant(rotation) for rotation in self.rotations)

    @property
    def sohncke(self) -> bool:
        """Whether every operation is proper, so handedness is preserved."""
        return all(value == 1 for value in self.determinants)

    @property
    def centrosymmetric(self) -> bool:
        """Whether the point group contains the inversion."""
        return INVERSION_ROTATION in self.rotations

    @property
    def centering_vectors(self) -> tuple[RationalVector, ...]:
        """Exact centering translations, the zero vector first."""
        step = self.point_group_order
        return tuple(
            self._translation(index) for index in range(0, self.operation_count, step)
        )

    def _translation(self, index: int, /) -> RationalVector:
        numerators = self.translation_numerators[index]
        denominator = self.translation_denominator
        return (
            Fraction(numerators[0], denominator),
            Fraction(numerators[1], denominator),
            Fraction(numerators[2], denominator),
        )

    def exact_operations(self) -> tuple[ExactAffineOperation, ...]:
        """Return the canonical exact operations ``(R, t)``."""
        return tuple(
            (rotation, self._translation(index))
            for index, rotation in enumerate(self.rotations)
        )

    def transformed(
        self,
        transformation: SettingTransformation,
        /,
        *,
        maximum_operation_count: int = DEFAULT_MAXIMUM_OPERATION_COUNT,
    ) -> SpaceGroupSetting:
        """Return this setting re-expressed through a further change of setting."""
        if not isinstance(transformation, SettingTransformation):
            raise TypeError("transformation must be a SettingTransformation.")
        return SpaceGroupSetting(
            self.reference_hall_number,
            transformation=self.transformation.followed_by(transformation),
            maximum_operation_count=maximum_operation_count,
        )

    def rotation_array(self) -> HostInt64[OperationDim, Literal[3], Literal[3]]:
        """Return the integer rotations as a host array of shape ``(operations, 3, 3)``."""
        return parse(
            np.asarray(self.rotations, dtype=np.int64).reshape((-1, 3, 3)),
            HostInt64[OperationDim, Literal[3], Literal[3]],
            "rotations",
        )

    def translation_numerator_array(self) -> HostInt64[OperationDim, Literal[3]]:
        """Return translation numerators over ``translation_denominator``."""
        return parse(
            np.asarray(self.translation_numerators, dtype=np.int64),
            HostInt64[OperationDim, Literal[3]],
            "translation_numerators",
        )

    def multiplication_table(
        self,
    ) -> tuple[
        HostInt64[OperationDim, OperationDim],
        HostInt64[OperationDim, OperationDim, Literal[3]],
    ]:
        """Return ``k[i, j]`` and integer ``n[i, j]`` with ``g_i g_j = g_k + n``."""
        operations = self.exact_operations()
        position = {operation: index for index, operation in enumerate(operations)}
        count = len(operations)
        products = np.empty((count, count), dtype=np.int64)
        shifts = np.empty((count, count, 3), dtype=np.int64)
        for i, left in enumerate(operations):
            for j, right in enumerate(operations):
                reduced = compose_operations(left, right)
                k = position[reduced]
                rotation = left[0]
                full = tuple(
                    sum(
                        (rotation[3 * r + c] * right[1][c] for c in range(3)), Fraction(0)
                    )
                    + left[1][r]
                    for r in range(3)
                )
                products[i, j] = k
                difference = tuple(full[r] - reduced[1][r] for r in range(3))
                if any(value.denominator != 1 for value in difference):
                    raise RuntimeError("Space-group product lost lattice integrality.")
                shifts[i, j] = [value.numerator for value in difference]
        return products, shifts

    def apply(self, fractional: ArrayLike, /, *, dtype: DTypeLike) -> Array:
        """Return every operation image ``R s + t`` with shape ``(operations, ..., 3)``.

        ``fractional`` has trailing extent three; images are not wrapped into
        the unit cell, so molecular whole-image branches remain explicit.
        """
        points = jnp.asarray(fractional, dtype=dtype)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("fractional coordinates must have trailing extent 3.")
        if not jnp.issubdtype(points.dtype, jnp.floating):
            raise TypeError("dtype must be a real floating dtype.")
        rotations = jnp.asarray(self.rotation_array(), dtype=points.dtype)
        translations = jnp.asarray(
            np.asarray(self.translation_numerators, dtype=np.float64)
            / self.translation_denominator,
            dtype=points.dtype,
        )
        images = contract("oij,...j->o...i", rotations, points, backend="jax")
        return images + translations.reshape(
            (translations.shape[0],) + (1,) * (points.ndim - 1) + (3,)
        )


def standard_space_group_setting(international_number: int, /) -> SpaceGroupSetting:
    """Return the admitted standard setting of an International number (1-230).

    The standard Hall entries are the pinned Toolkit choices: origin choice 2
    where two origins exist and hexagonal axes for rhombohedral groups.
    """
    if isinstance(international_number, bool) or not isinstance(
        international_number, int
    ):
        raise TypeError("international_number must be an integer.")
    if not 1 <= international_number <= len(data.STANDARD_HALL_NUMBERS):
        raise ValueError("international_number must lie in 1..230.")
    return SpaceGroupSetting(data.STANDARD_HALL_NUMBERS[international_number - 1])


def admitted_hall_numbers(international_number: int, /) -> tuple[int, ...]:
    """Return every admitted Hall entry of an International number, ascending."""
    if isinstance(international_number, bool) or not isinstance(
        international_number, int
    ):
        raise TypeError("international_number must be an integer.")
    if not 1 <= international_number <= len(data.STANDARD_HALL_NUMBERS):
        raise ValueError("international_number must lie in 1..230.")
    return tuple(
        entry.hall_number
        for entry in data.HALL_ENTRIES
        if entry.international_number == international_number
    )


__all__ = [
    "admitted_hall_numbers",
    "CellChoice",
    "CrystalSystem",
    "DEFAULT_MAXIMUM_OPERATION_COUNT",
    "MonoclinicAxis",
    "OperationDim",
    "OriginChoice",
    "OrthorhombicAxes",
    "RhombohedralAxes",
    "SettingTransformation",
    "SpaceGroupSetting",
    "standard_space_group_setting",
]
