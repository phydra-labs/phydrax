#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact site symmetry, special-position enumeration and site orbits.

A *site* is an affine subspace ``S = {x0 + W u}`` of fractional coordinates
that is the fixed-point set of its generic stabilizer: the space-group
elements ``x -> R x + t + n`` (operation ``(R, t)`` of the setting together
with an integer lattice shift ``n``) that fix every point of ``S``.  The
general position is the whole space with the identity stabilizer; special
positions are lower-dimensional sites with a larger stabilizer.

Everything here is host-only exact arithmetic over the integers and
`fractions.Fraction`; the native exact Smith decomposition supplies kernels,
saturated lattice bases and the solution of affine congruences.  Special
positions are enumerated by intersecting a known site with the fixed set of
one more operation: in the direction lattice ``W`` of a site the congruence
``(R - I) W u = -((R - I) x0 + t) - n`` has, by the Smith form
``U (R - I) W V = D``, exactly ``prod D_ii`` solution classes modulo lattice
translations along the site, so the integer shifts follow from the operation
instead of a guessed search box.  Every site is reached by such a chain, and
the search runs only from one representative per orbit.  Exhausting either
declared bound returns an explicitly incomplete table.

Site orbits are the classes of sites under the space group; for a complete
table they coincide with the Wyckoff positions of the setting.  Wyckoff
letters are deliberately not produced: they come only from an admitted
authoritative letter mapping, which is not part of the admitted data.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction
from functools import lru_cache
from itertools import product
from numbers import Rational
from typing import final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...algebraic._exact_integer import smith_normal_decomposition, smith_rank
from ...ein import contract
from ._hall import (
    compose_operations,
    ExactAffineOperation,
    IDENTITY_ROTATION,
    IntegerMatrix,
    multiply_rotations,
    RationalVector,
    reduce_translation,
    rotate_vector,
    rotation_determinant,
)
from ._setting import SpaceGroupSetting


type LatticeShift = tuple[int, int, int]
type StabilizerElement = tuple[int, LatticeShift]
"""Operation index ``k`` and integer shift ``n`` of the map ``x -> R_k x + t_k + n``."""

SiteTableStatus: TypeAlias = Literal["complete", "site-limit", "system-limit"]

_ZERO = Fraction(0)
_ZERO_SHIFT: LatticeShift = (0, 0, 0)
_UNIT_DIRECTIONS: tuple[LatticeShift, ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _rational(value: object, name: str, /) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Rational):
        raise TypeError(f"{name} must contain exact integers or fractions.")
    return Fraction(int(value.numerator), int(value.denominator))


def _rational_vector(values: Sequence[int | Fraction], name: str, /) -> RationalVector:
    if len(values) != 3:
        raise ValueError(f"{name} must have three components.")
    return (
        _rational(values[0], name),
        _rational(values[1], name),
        _rational(values[2], name),
    )


def _integer_vector(values: Sequence[int], name: str, /) -> LatticeShift:
    if len(values) != 3:
        raise ValueError(f"{name} must have three components.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError(f"{name} must contain integers.")
    return (values[0], values[1], values[2])


def _minus_identity(
    rotation: IntegerMatrix, /
) -> tuple[LatticeShift, LatticeShift, LatticeShift]:
    return (
        (rotation[0] - 1, rotation[1], rotation[2]),
        (rotation[3], rotation[4] - 1, rotation[5]),
        (rotation[6], rotation[7], rotation[8] - 1),
    )


def _apply_rows(
    rows: Sequence[Sequence[int]], vector: Sequence[Fraction | int], /
) -> tuple[Fraction, ...]:
    return tuple(
        sum((Fraction(row[k]) * vector[k] for k in range(len(vector))), _ZERO)
        for row in rows
    )


def _column(matrix: np.ndarray, column: int, /) -> tuple[int, ...]:
    return tuple(int(matrix[row, column]) for row in range(matrix.shape[0]))


def _integer_kernel(rows: Sequence[Sequence[int]], /) -> tuple[LatticeShift, ...]:
    """Return a saturated integer basis of ``{x in Z^3 : rows x = 0}`` via Smith."""
    if not rows:
        return _UNIT_DIRECTIONS
    diagonal, _, right = smith_normal_decomposition([list(row) for row in rows])
    rank = smith_rank(diagonal)
    return tuple(
        (int(right[0, column]), int(right[1, column]), int(right[2, column]))
        for column in range(rank, 3)
    )


class _SiteFrame(NamedTuple):
    """Canonical direction lattice, annihilator and completion of a rotation set."""

    directions: tuple[LatticeShift, ...]
    constraints: tuple[LatticeShift, ...]
    # Point completion x0 = completion @ (left @ offsets) for A x0 = offsets.
    completion: tuple[tuple[int, ...], ...]
    left: tuple[tuple[int, ...], ...]


@lru_cache(maxsize=4096)
def _site_frame(rotations: tuple[IntegerMatrix, ...], /) -> _SiteFrame:
    """Return the canonical frame of the fixed directions of a sorted rotation set.

    Directions are the saturated integer kernel of the stacked ``R - I``;
    constraints ``A`` are the saturated integer annihilator of the directions,
    so ``A Z^3 = Z^(3-d)`` and two parallel sites coincide modulo ``Z^3`` exactly
    when ``A x mod 1`` agrees.  The Smith form of ``A`` (unit invariant factors)
    supplies a deterministic point with prescribed offsets.
    """
    rows = [row for rotation in rotations for row in _minus_identity(rotation)]
    directions = _integer_kernel(rows) if rows else _UNIT_DIRECTIONS
    dimension = len(directions)
    if dimension == 3:
        return _SiteFrame(directions, (), (), ())
    if dimension == 0:
        constraints: tuple[LatticeShift, ...] = _UNIT_DIRECTIONS
    else:
        transposed = [list(direction) for direction in directions]
        diagonal, _, right = smith_normal_decomposition(transposed)
        if smith_rank(diagonal) != dimension:
            raise RuntimeError("Site directions lost full rank.")
        constraints = tuple(
            (int(right[0, column]), int(right[1, column]), int(right[2, column]))
            for column in range(dimension, 3)
        )
    count = 3 - dimension
    diagonal, left, right = smith_normal_decomposition([list(row) for row in constraints])
    if any(int(diagonal[index, index]) != 1 for index in range(count)):
        raise RuntimeError("Site annihilator is not saturated.")
    completion = tuple(
        tuple(int(right[row, column]) for column in range(count)) for row in range(3)
    )
    left_rows = tuple(
        tuple(int(left[row, column]) for column in range(count)) for row in range(count)
    )
    return _SiteFrame(directions, constraints, completion, left_rows)


class _GroupContext(NamedTuple):
    operations: tuple[ExactAffineOperation, ...]
    index: dict[ExactAffineOperation, int]
    inverse_rotation: dict[IntegerMatrix, IntegerMatrix]


@lru_cache(maxsize=64)
def _group_context(operations: tuple[ExactAffineOperation, ...], /) -> _GroupContext:
    rotations = sorted({rotation for rotation, _ in operations})
    inverse = {}
    for rotation in rotations:
        matches = [
            other
            for other in rotations
            if multiply_rotations(rotation, other) == IDENTITY_ROTATION
        ]
        if len(matches) != 1:
            raise RuntimeError("Setting rotations are not closed under inversion.")
        inverse[rotation] = matches[0]
    return _GroupContext(
        operations,
        {operation: position for position, operation in enumerate(operations)},
        inverse,
    )


def _context(setting: SpaceGroupSetting, /) -> _GroupContext:
    return _group_context(setting.exact_operations())


def _wrapped(values: Sequence[Fraction], /) -> tuple[Fraction, ...]:
    return tuple(value % 1 for value in values)


def _site_key(
    setting_id: str,
    rotations: tuple[IntegerMatrix, ...],
    offsets: tuple[Fraction, ...],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "crystal-site",
            "setting_id": setting_id,
            "rotations": [list(rotation) for rotation in rotations],
            "offsets": [str(value) for value in offsets],
        }
    )


def _image_key(
    setting_id: str,
    context: _GroupContext,
    operation: ExactAffineOperation,
    point: RationalVector,
    stabilizer_rotations: tuple[IntegerMatrix, ...],
    /,
) -> str:
    """Return the key of ``g(S)`` from the conjugated stabilizer rotations."""
    rotation, translation = operation
    inverse = context.inverse_rotation[rotation]
    rotations = tuple(
        sorted(
            {
                multiply_rotations(multiply_rotations(rotation, member), inverse)
                for member in stabilizer_rotations
            }
        )
    )
    image = rotate_vector(rotation, point)
    image = (
        image[0] + translation[0],
        image[1] + translation[1],
        image[2] + translation[2],
    )
    frame = _site_frame(rotations)
    return _site_key(
        setting_id, rotations, _wrapped(_apply_rows(frame.constraints, image))
    )


@final
class SitePosition(StrictModule, NonTrainableState):
    """One exact site subspace modulo ``Z^3`` with its generic stabilizer.

    ``point`` is the canonical exact point and ``directions`` the saturated
    integer basis of the direction lattice, so the site is ``point + W u`` with
    ``W`` the columns ``directions``.  ``constraints`` ``A`` and ``offsets``
    ``b`` describe the same site as ``A x = b (mod 1)``.  ``stabilizer`` lists
    every element ``(k, n)`` with ``R_k x + t_k + n = x`` on the whole site,
    identity first.  ``coset_representatives`` are the lowest operation
    indices of the left cosets of the stabilizer, one per image of a generic
    site point, so ``multiplicity = operation_count / stabilizer_order``.
    ``site_key`` is the canonical identity of the site modulo ``Z^3``.
    """

    setting_id: str = eqx.field(static=True)
    point: RationalVector = eqx.field(static=True)
    directions: tuple[LatticeShift, ...] = eqx.field(static=True)
    constraints: tuple[LatticeShift, ...] = eqx.field(static=True)
    offsets: tuple[Fraction, ...] = eqx.field(static=True)
    stabilizer: tuple[StabilizerElement, ...] = eqx.field(static=True)
    stabilizer_rotations: tuple[IntegerMatrix, ...] = eqx.field(static=True)
    coset_representatives: tuple[int, ...] = eqx.field(static=True)
    coset_operations: tuple[ExactAffineOperation, ...] = eqx.field(static=True)
    operation_count: int = eqx.field(static=True)
    site_key: str = eqx.field(static=True)

    def __init__(
        self,
        setting: SpaceGroupSetting,
        point: Sequence[int | Fraction],
        /,
        directions: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Admit the site through ``point`` spanned by integer ``directions``.

        Without ``directions`` the site is the fixed set of the stabilizer of
        ``point`` itself.  With ``directions`` the subspace must be exactly the
        fixed set of its generic stabilizer; otherwise it is refused, because a
        subspace that is not a fixed set is not a site.
        """
        if not isinstance(setting, SpaceGroupSetting):
            raise TypeError("setting must be a SpaceGroupSetting.")
        context = _context(setting)
        base = _rational_vector(point, "point")
        spanning = (
            None
            if directions is None
            else tuple(
                _integer_vector(direction, "direction") for direction in directions
            )
        )
        stabilizer = _site_stabilizer(context, base, () if spanning is None else spanning)
        rotations = tuple(sorted({context.operations[k][0] for k, _ in stabilizer}))
        frame = _site_frame(rotations)
        if spanning is not None:
            rank = (
                smith_rank(smith_normal_decomposition([list(v) for v in spanning])[0])
                if spanning
                else 0
            )
            if rank != len(frame.directions):
                raise ValueError(
                    "The subspace is not the fixed set of its generic stabilizer; "
                    f"its dimension {rank} differs from the fixed dimension "
                    f"{len(frame.directions)}."
                )
        offsets = _wrapped(_apply_rows(frame.constraints, base))
        canonical = _canonical_point(frame, offsets)
        canonical_stabilizer = _site_stabilizer(context, canonical, frame.directions)
        if len(canonical_stabilizer) != len(stabilizer):
            raise RuntimeError("Canonical site point changed the generic stabilizer.")
        representatives = _coset_representatives(context, canonical_stabilizer)
        self.setting_id = setting.setting_id
        self.point = canonical
        self.directions = frame.directions
        self.constraints = frame.constraints
        self.offsets = offsets
        self.stabilizer = canonical_stabilizer
        self.stabilizer_rotations = rotations
        self.coset_representatives = representatives
        self.coset_operations = tuple(context.operations[k] for k in representatives)
        self.operation_count = len(context.operations)
        self.site_key = _site_key(setting.setting_id, rotations, offsets)

    @property
    def dimension(self) -> int:
        """Number of free site parameters (3 for the general position)."""
        return len(self.directions)

    @property
    def stabilizer_order(self) -> int:
        return len(self.stabilizer)

    @property
    def multiplicity(self) -> int:
        """Orbit size of a generic site point modulo ``Z^3`` (centering included)."""
        return self.operation_count // len(self.stabilizer)

    @property
    def special(self) -> bool:
        return len(self.stabilizer) > 1

    @property
    def improper_stabilizer(self) -> bool:
        """Whether a stabilizer element reverses handedness."""
        return any(
            rotation_determinant(rotation) < 0 for rotation in self.stabilizer_rotations
        )

    def direction_matrix(self) -> np.ndarray:
        """Return ``W`` as a host integer array of shape ``(3, dimension)``."""
        matrix = np.zeros((3, self.dimension), dtype=np.int64)
        for column, direction in enumerate(self.directions):
            matrix[:, column] = direction
        return matrix

    def point_array(self) -> np.ndarray:
        """Return the canonical point as a host float64 array (exact values rounded once)."""
        return np.asarray([float(value) for value in self.point], dtype=np.float64)

    def point_at(self, parameters: ArrayLike, /, *, dtype: DTypeLike) -> Array:
        """Return ``point + W u`` for parameters of shape ``(..., dimension)``."""
        values = jnp.asarray(parameters, dtype=dtype)
        if values.ndim < 1 or values.shape[-1] != self.dimension:
            raise ValueError(f"site parameters must end in extent {self.dimension}.")
        origin = jnp.asarray(self.point_array(), dtype=values.dtype)
        basis = jnp.asarray(self.direction_matrix(), dtype=values.dtype)
        return origin + contract("ij,...j->...i", basis, values, backend="jax")

    def constraint_residual(self, fractional: ArrayLike, /) -> Array:
        """Return ``A x - b`` wrapped to ``[-1/2, 1/2)`` for points of shape ``(..., 3)``.

        A zero residual means the point lies on a lattice translate of the site.
        """
        values = jnp.asarray(fractional)
        if values.ndim < 1 or values.shape[-1] != 3:
            raise ValueError("fractional coordinates must have trailing extent 3.")
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("fractional coordinates must be real floating values.")
        count = 3 - self.dimension
        if count == 0:
            return jnp.zeros(values.shape[:-1] + (0,), dtype=values.dtype)
        constraints = jnp.asarray(
            np.asarray(self.constraints, dtype=np.float64), dtype=values.dtype
        )
        offsets = jnp.asarray(
            np.asarray([float(value) for value in self.offsets], dtype=np.float64),
            dtype=values.dtype,
        )
        raw = contract("ij,...j->...i", constraints, values, backend="jax") - offsets
        return raw - jnp.round(raw)

    def orbit(
        self, parameters: Sequence[int | Fraction] = ()
    ) -> tuple[tuple[int, RationalVector, LatticeShift], ...]:
        """Return the exact orbit of the site point at rational ``parameters``.

        Each entry is ``(k, image, n)`` with ``image = R_k x + t_k + n`` in
        ``[0, 1)^3`` for coset representative ``k``; ``x = point + W u``.  For
        parameters at which ``x`` is a more special point the images repeat.
        """
        if len(parameters) != self.dimension:
            raise ValueError(f"parameters must have {self.dimension} entries.")
        values = [_rational(value, "parameters") for value in parameters]
        point = tuple(
            self.point[i]
            + sum(
                (self.directions[k][i] * values[k] for k in range(self.dimension)), _ZERO
            )
            for i in range(3)
        )
        result = []
        for index, (rotation, translation) in zip(
            self.coset_representatives, self.coset_operations, strict=True
        ):
            moved = rotate_vector(rotation, (point[0], point[1], point[2]))
            raw = (
                moved[0] + translation[0],
                moved[1] + translation[1],
                moved[2] + translation[2],
            )
            wrapped = reduce_translation(raw)
            result.append(
                (
                    index,
                    wrapped,
                    (
                        int(wrapped[0] - raw[0]),
                        int(wrapped[1] - raw[1]),
                        int(wrapped[2] - raw[2]),
                    ),
                )
            )
        return tuple(result)


def _fixes_directions(
    rotation: IntegerMatrix, directions: Sequence[LatticeShift], /
) -> bool:
    return all(
        rotate_vector(rotation, (Fraction(v[0]), Fraction(v[1]), Fraction(v[2])))
        == (Fraction(v[0]), Fraction(v[1]), Fraction(v[2]))
        for v in directions
    )


def _site_stabilizer(
    context: _GroupContext, point: RationalVector, directions: Sequence[LatticeShift], /
) -> tuple[StabilizerElement, ...]:
    """Return every ``(k, n)`` fixing ``point`` and the ``directions``, by index."""
    result: list[StabilizerElement] = []
    for index, (rotation, translation) in enumerate(context.operations):
        if not _fixes_directions(rotation, directions):
            continue
        moved = rotate_vector(rotation, point)
        drift = tuple(moved[i] + translation[i] - point[i] for i in range(3))
        if all(value.denominator == 1 for value in drift):
            result.append(
                (index, (-drift[0].numerator, -drift[1].numerator, -drift[2].numerator))
            )
    if not result or result[0] != (0, _ZERO_SHIFT):
        raise RuntimeError("A site stabilizer must start with the identity.")
    return tuple(result)


def _canonical_point(
    frame: _SiteFrame, offsets: tuple[Fraction, ...], /
) -> RationalVector:
    count = len(frame.constraints)
    if count == 0:
        return (_ZERO, _ZERO, _ZERO)
    reduced = [
        sum((Fraction(frame.left[row][k]) * offsets[k] for k in range(count)), _ZERO)
        for row in range(count)
    ]
    point = tuple(
        sum(
            (Fraction(frame.completion[row][k]) * reduced[k] for k in range(count)), _ZERO
        )
        for row in range(3)
    )
    return (point[0], point[1], point[2])


def _coset_representatives(
    context: _GroupContext, stabilizer: tuple[StabilizerElement, ...], /
) -> tuple[int, ...]:
    members = [context.operations[k] for k, _ in stabilizer]
    seen: set[int] = set()
    representatives: list[int] = []
    for index, operation in enumerate(context.operations):
        if index in seen:
            continue
        representatives.append(index)
        for member in members:
            seen.add(context.index[compose_operations(operation, member)])
    if len(representatives) * len(stabilizer) != len(context.operations):
        raise RuntimeError("Stabilizer cosets do not partition the operations.")
    return tuple(representatives)


def site_of_point(
    setting: SpaceGroupSetting, point: Sequence[int | Fraction], /
) -> SitePosition:
    """Return the exact site whose generic stabilizer is the stabilizer of ``point``."""
    return SitePosition(setting, point)


@final
class SiteOrbit(StrictModule, NonTrainableState):
    """One class of sites under the space group.

    ``representative`` is the member with the least ``site_key``;
    ``member_keys`` lists every member site modulo ``Z^3``.  For a complete
    table the orbits are the Wyckoff positions of the setting.  No Wyckoff
    letter is attached: letters require an admitted authoritative mapping.
    """

    representative: SitePosition
    member_keys: tuple[str, ...] = eqx.field(static=True)
    orbit_id: str = eqx.field(static=True)

    @property
    def multiplicity(self) -> int:
        return self.representative.multiplicity

    @property
    def dimension(self) -> int:
        return self.representative.dimension

    @property
    def stabilizer_order(self) -> int:
        return self.representative.stabilizer_order


@final
class SiteSymmetryTable(StrictModule, NonTrainableState):
    """Site orbits of one setting with truthful completeness evidence.

    ``status`` is ``"complete"`` only when the bounded search finished; then
    every site of the setting belongs to exactly one listed orbit.  Otherwise
    the listed orbits are correct but the table may miss sites, and
    `orbit_of` cannot refute membership.
    """

    setting_id: str = eqx.field(static=True)
    orbits: tuple[SiteOrbit, ...]
    status: SiteTableStatus = eqx.field(static=True)
    sites_found: int = eqx.field(static=True)
    systems_solved: int = eqx.field(static=True)
    maximum_sites: int = eqx.field(static=True)
    maximum_systems: int = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    def orbit_of(self, site: SitePosition, /) -> SiteOrbit | None:
        """Return the orbit containing ``site``; ``None`` only for an incomplete table."""
        if not isinstance(site, SitePosition):
            raise TypeError("site must be a SitePosition.")
        if site.setting_id != self.setting_id:
            raise ValueError("site belongs to another setting.")
        for orbit in self.orbits:
            if site.site_key in orbit.member_keys:
                return orbit
        if self.complete:
            raise RuntimeError("A complete site table misses an admitted site.")
        return None


def _orbit_keys(
    setting_id: str, context: _GroupContext, site: SitePosition, /
) -> frozenset[str]:
    return frozenset(
        _image_key(setting_id, context, operation, site.point, site.stabilizer_rotations)
        for operation in context.operations
    )


def _image_site(
    setting: SpaceGroupSetting, context: _GroupContext, site: SitePosition, index: int, /
) -> SitePosition:
    rotation, translation = context.operations[index]
    moved = rotate_vector(rotation, site.point)
    point = (
        moved[0] + translation[0],
        moved[1] + translation[1],
        moved[2] + translation[2],
    )
    directions = [
        tuple(
            int(value)
            for value in rotate_vector(
                rotation, (Fraction(v[0]), Fraction(v[1]), Fraction(v[2]))
            )
        )
        for v in site.directions
    ]
    return SitePosition(setting, point, directions)


class _Congruence(NamedTuple):
    """Smith-reduced ``(R - I) W u = -((R - I) x0 + t) - n`` along one site.

    ``projected`` is the left-transformed target, ``factors`` the nonzero
    invariant factors and ``right`` the right Smith transform; each class
    ``c`` with ``0 <= c_i < factors[i]`` is one solution modulo the site lattice.
    """

    rank: int
    factors: list[int]
    projected: tuple[Fraction, ...]
    right: np.ndarray


def _congruence(
    site: SitePosition, operation: ExactAffineOperation, /
) -> _Congruence | None:
    """Smith-reduce the fixed-set congruence of ``operation`` along ``site``.

    ``None`` when the operation fixes every site direction or the congruence
    has no integer solution.
    """
    rotation, translation = operation
    directions = site.directions
    dimension = len(directions)
    rows = _minus_identity(rotation)
    moved = [
        _apply_rows(rows, (Fraction(v[0]), Fraction(v[1]), Fraction(v[2])))
        for v in directions
    ]
    matrix = [
        [int(moved[column][row]) for column in range(dimension)] for row in range(3)
    ]
    if all(value == 0 for row in matrix for value in row):
        return None
    diagonal, left, right = smith_normal_decomposition(matrix)
    rank = smith_rank(diagonal)
    drift = _apply_rows(rows, site.point)
    target = tuple(-(drift[i] + translation[i]) for i in range(3))
    projected = _apply_rows(
        [[int(left[r, c]) for c in range(3)] for r in range(3)], target
    )
    if any(projected[i].denominator != 1 for i in range(rank, 3)):
        return None
    factors = [int(diagonal[i, i]) for i in range(rank)]
    return _Congruence(rank, factors, projected, right)


def _congruence_classes(
    context: _GroupContext, site: SitePosition, /
) -> Iterator[tuple[_Congruence, tuple[int, ...]]]:
    """Lazily yield every solvable congruence class of ``site``, operation-major.

    Operations in the site stabilizer contribute no class.  Laziness keeps
    each Smith decomposition behind the system bound of the consumer.
    """
    stabilizing = {k for k, _ in site.stabilizer}
    if len(site.directions) == 0:
        return
    for index, operation in enumerate(context.operations):
        if index in stabilizing:
            continue
        congruence = _congruence(site, operation)
        if congruence is None:
            continue
        for classes in product(*(range(factor) for factor in congruence.factors)):
            yield congruence, classes


def _intersection(
    context: _GroupContext,
    setting_id: str,
    site: SitePosition,
    congruence: _Congruence,
    classes: tuple[int, ...],
    /,
) -> tuple[str, RationalVector, list[LatticeShift]]:
    """Return ``(site_key, point, remaining directions)`` of one congruence class."""
    directions = site.directions
    dimension = len(directions)
    rank, factors, projected, right = congruence
    reduced = [(projected[i] - classes[i]) / factors[i] for i in range(rank)] + [
        _ZERO
    ] * (dimension - rank)
    parameters = [
        sum(
            (Fraction(int(right[row, k])) * reduced[k] for k in range(dimension)),
            _ZERO,
        )
        for row in range(dimension)
    ]
    first, second, third = (
        site.point[i]
        + sum(
            (directions[k][i] * parameters[k] for k in range(dimension)),
            _ZERO,
        )
        for i in range(3)
    )
    point = (first, second, third)
    remaining: list[LatticeShift] = []
    for column in range(rank, dimension):
        first_shift, second_shift, third_shift = (
            sum(directions[k][i] * int(right[k, column]) for k in range(dimension))
            for i in range(3)
        )
        remaining.append((first_shift, second_shift, third_shift))
    stabilizer = _site_stabilizer(context, point, remaining)
    rotations = tuple(sorted({context.operations[k][0] for k, _ in stabilizer}))
    frame = _site_frame(rotations)
    if len(frame.directions) != len(remaining):
        raise RuntimeError("A stabilizer intersection is not a fixed set.")
    key = _site_key(
        setting_id, rotations, _wrapped(_apply_rows(frame.constraints, point))
    )
    return key, point, remaining


class _SiteSearch(NamedTuple):
    """Outcome of the bounded search: one discovered site per orbit, in order."""

    orbit_sites: list[tuple[SitePosition, frozenset[str]]]
    sites_found: int
    systems: int
    status: SiteTableStatus


def _search_sites(
    setting: SpaceGroupSetting,
    context: _GroupContext,
    maximum_sites: int,
    maximum_systems: int,
    /,
) -> _SiteSearch:
    """Breadth-first special-position search from the general position.

    Each new site registers its whole orbit, so only one representative per
    orbit is searched; reaching either bound stops with that status.
    """
    setting_id = setting.setting_id
    general = SitePosition(setting, (0, 0, 0), _UNIT_DIRECTIONS)
    known: dict[str, int] = {}
    orbit_sites: list[tuple[SitePosition, frozenset[str]]] = []
    systems = 0

    def register(site: SitePosition) -> bool:
        keys = _orbit_keys(setting_id, context, site)
        if len(known) + len(keys) > maximum_sites:
            return False
        for key in keys:
            known[key] = len(orbit_sites)
        orbit_sites.append((site, keys))
        return True

    if not register(general):
        return _SiteSearch(orbit_sites, len(known), systems, "site-limit")
    queue = [general]
    while queue:
        site = queue.pop(0)
        for congruence, classes in _congruence_classes(context, site):
            systems += 1
            if systems > maximum_systems:
                return _SiteSearch(orbit_sites, len(known), systems, "system-limit")
            key, point, remaining = _intersection(
                context, setting_id, site, congruence, classes
            )
            if key in known:
                continue
            candidate = SitePosition(setting, point, remaining)
            if not register(candidate):
                return _SiteSearch(orbit_sites, len(known), systems, "site-limit")
            queue.append(candidate)
    return _SiteSearch(orbit_sites, len(known), systems, "complete")


def _site_orbit(
    setting: SpaceGroupSetting,
    context: _GroupContext,
    site: SitePosition,
    keys: frozenset[str],
    /,
) -> SiteOrbit:
    """Orbit of ``site`` represented by its member with the least site key."""
    least = min(keys)
    representative = site
    if site.site_key != least:
        for index in range(len(context.operations)):
            image = _image_site(setting, context, site, index)
            if image.site_key == least:
                representative = image
                break
        else:
            raise RuntimeError("An orbit member key has no realizing operation.")
    ordered = tuple(sorted(keys))
    return SiteOrbit(
        representative,
        ordered,
        canonical_fingerprint(
            {
                "kind": "crystal-site-orbit",
                "setting_id": setting.setting_id,
                "members": ordered,
            }
        ),
    )


def enumerate_site_orbits(
    setting: SpaceGroupSetting,
    /,
    *,
    maximum_sites: int,
    maximum_systems: int,
) -> SiteSymmetryTable:
    """Enumerate every site orbit of ``setting`` within explicit bounds.

    ``maximum_sites`` bounds the distinct sites modulo ``Z^3`` (orbit members
    included) and ``maximum_systems`` the solved congruence classes.  A search
    that reaches either bound stops with the corresponding incomplete status.
    """
    if not isinstance(setting, SpaceGroupSetting):
        raise TypeError("setting must be a SpaceGroupSetting.")
    for name, value in (
        ("maximum_sites", maximum_sites),
        ("maximum_systems", maximum_systems),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < 1:
            raise ValueError(f"{name} must be positive.")
    context = _context(setting)
    setting_id = setting.setting_id
    search = _search_sites(setting, context, maximum_sites, maximum_systems)
    orbits = [
        _site_orbit(setting, context, site, keys) for site, keys in search.orbit_sites
    ]
    orbits.sort(key=lambda orbit: (orbit.multiplicity, orbit.dimension, orbit.orbit_id))
    return SiteSymmetryTable(
        setting_id,
        tuple(orbits),
        search.status,
        search.sites_found,
        search.systems,
        maximum_sites,
        maximum_systems,
        canonical_fingerprint(
            {
                "kind": "crystal-site-table",
                "setting_id": setting_id,
                "status": search.status,
                "orbits": [orbit.orbit_id for orbit in orbits],
            }
        ),
    )


__all__ = [
    "enumerate_site_orbits",
    "LatticeShift",
    "site_of_point",
    "SiteOrbit",
    "SitePosition",
    "SiteSymmetryTable",
    "SiteTableStatus",
    "StabilizerElement",
]
