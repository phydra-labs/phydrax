#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Trigonometric nodal level sets on periodic lattices (TPMS families).

A table is a finite trigonometric polynomial in reduced lattice coordinates
``u in R^3 / Z^3``: exact rational coefficients times products of
``sin``/``cos(2 pi n u_i)``. With ``x, y, z = 2 pi u`` the named tables are the
nodal approximations of von Schnering and Nesper, "Nodal surfaces of Fourier
series: fundamental invariants of structured matter", Z. Phys. B 83 (1991)
407-412, and of Lord and Mackay, "Periodic minimal surfaces of cubic symmetry",
Current Science 85 (2003) 346-362:

- gyroid: ``sin x cos y + sin y cos z + sin z cos x``;
- gyroid_mirror: ``cos x sin y + cos y sin z + cos z sin x``, the gyroid composed
  with the reflection ``(u_1, u_2, u_3) -> (u_1, u_3, u_2)`` (its enantiomorph);
- schwarz_p: ``cos x + cos y + cos z``;
- schwarz_d: ``sin x sin y sin z + sin x cos y cos z + cos x sin y cos z
  + cos x cos y sin z``;
- iwp: ``2 (cos x cos y + cos y cos z + cos z cos x) - (cos 2x + cos 2y + cos 2z)``;
- lidinoid: ``1/2 (sin 2x cos y sin z + sin 2y cos z sin x + sin 2z cos x sin y)
  - 1/2 (cos 2x cos 2y + cos 2y cos 2z + cos 2z cos 2x) + 3/20``.

These are nodal approximations of the minimal surfaces, not exact minimal
surfaces: their mean curvature is evaluated, never assumed zero.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from fractions import Fraction
from typing import assert_never, Literal, NoReturn, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.custom_derivatives import SymbolicZero

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._periodic_cell import PeriodicCell
from ...typing import checked, parse, PRNGKey
from .._capabilities import GeometryCapability
from .._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from .._contracts import (
    ContactCurvatureResult,
    GeometryKernel,
    GeometryKind,
    GeometrySource,
)
from .._interval_enclosure import (
    Interval,
    interval_add,
    interval_multiply,
    interval_periodic_harmonic,
    IntervalExtensionError,
    register_custom_jvp_interval_extension,
)
from .._periodic_harmonics import periodic_harmonic_jet
from .._sampling import RejectionSamplingPlan
from .._validity import GeometryValidityEvidence
from ..brep._intersection_kernel import interval_times_interval, krawczyk, round_interval
from ..design._schema import _ParameterCollector, DesignState
from ._primitives import _check_points, _feature_id


TrigonometricFactor: TypeAlias = Literal["sin", "cos"]
TrigonometricFamily: TypeAlias = Literal[
    "gyroid", "gyroid_mirror", "schwarz_p", "schwarz_d", "iwp", "lidinoid"
]
TrigonometricRegion: TypeAlias = Literal["network", "complement", "sheet"]

type _Factors = tuple[
    tuple[TrigonometricFactor, int],
    tuple[TrigonometricFactor, int],
    tuple[TrigonometricFactor, int],
]

_CAPABILITIES = frozenset(
    {
        GeometryCapability.REGION_QUERY,
        GeometryCapability.BOUNDARY_NORMAL,
        GeometryCapability.CONTACT_CURVATURE,
    }
)
_TWO_PI_UPPER = float(np.nextafter(2.0 * math.pi, np.inf))
_EPSILON = float(np.finfo(np.float64).eps)
# Initial critical-point search grid per reduced axis: 4^3 boxes.
_CRITICAL_GRID = 4
# Boxes per critical-search work item; bounds the interval workspace.
_CRITICAL_CHUNK = 16384

type _Bounds = tuple[np.ndarray, np.ndarray]


def _host(value: Interval, /) -> _Bounds:
    """Host float64 endpoint arrays of an interval-enclosure result."""
    return np.asarray(value[0], dtype=np.float64), np.asarray(value[1], dtype=np.float64)


def _up(value: float, /) -> float:
    return float(np.nextafter(value, np.inf))


def _down(value: float, /) -> float:
    return float(np.nextafter(value, -np.inf))


def _rational(value: object, name: str, /) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(
            f"{name} must be an exact int or Fraction, not {type(value).__name__}."
        )
    return Fraction(value)


def _fraction_interval(value: Fraction, /) -> tuple[float, float]:
    nearest = float(value)
    exact = Fraction(nearest)
    return (
        _down(nearest) if exact > value else nearest,
        _up(nearest) if exact < value else nearest,
    )


def _factor_kind(value: object, /) -> TrigonometricFactor:
    return parse(value, TrigonometricFactor, "factor")


class TrigonometricTerm(StrictModule):
    """One coefficient times a product of per-axis harmonics ``sin/cos(2 pi n u_i)``.

    Factors are canonical: frequencies are nonnegative (``sin(-n)`` folds its
    sign into the coefficient) and a zero frequency is the constant ``cos 0``.
    """

    coefficient: Fraction = eqx.field(static=True)
    factors: _Factors = eqx.field(static=True)

    def __init__(
        self, coefficient: int | Fraction, factors: Sequence[tuple[str, int]], /
    ) -> None:
        value = _rational(coefficient, "coefficient")
        if value == 0:
            raise ValueError("A trigonometric term coefficient must be nonzero.")
        entries = tuple(factors)
        if len(entries) != 3:
            raise ValueError("A trigonometric term needs one factor per lattice axis.")
        canonical: list[tuple[TrigonometricFactor, int]] = []
        for entry in entries:
            if len(entry) != 2:
                raise ValueError("Each factor is a (kind, frequency) pair.")
            kind = _factor_kind(entry[0])
            frequency = entry[1]
            if isinstance(frequency, bool) or not isinstance(frequency, int):
                raise TypeError("Factor frequencies must be integers.")
            if frequency < 0:
                frequency = -frequency
                match kind:
                    case "sin":
                        value = -value
                    case "cos":
                        pass
                    case _:
                        assert_never(kind)
            if frequency == 0:
                match kind:
                    case "sin":
                        raise ValueError(
                            "A zero-frequency sine factor vanishes identically."
                        )
                    case "cos":
                        pass
                    case _:
                        assert_never(kind)
            canonical.append((kind, frequency))
        self.coefficient = value
        self.factors = (canonical[0], canonical[1], canonical[2])


def _term(coefficient: int | Fraction, *factors: tuple[str, int]) -> TrigonometricTerm:
    return TrigonometricTerm(coefficient, factors)


def _family_terms(family: TrigonometricFamily, /) -> tuple[TrigonometricTerm, ...]:
    s1, c1, s2, c2, one = ("sin", 1), ("cos", 1), ("sin", 2), ("cos", 2), ("cos", 0)
    half = Fraction(1, 2)
    match family:
        case "gyroid":
            return (
                _term(1, s1, c1, one),
                _term(1, one, s1, c1),
                _term(1, c1, one, s1),
            )
        case "gyroid_mirror":
            return (
                _term(1, c1, s1, one),
                _term(1, one, c1, s1),
                _term(1, s1, one, c1),
            )
        case "schwarz_p":
            return (
                _term(1, c1, one, one),
                _term(1, one, c1, one),
                _term(1, one, one, c1),
            )
        case "schwarz_d":
            return (
                _term(1, s1, s1, s1),
                _term(1, s1, c1, c1),
                _term(1, c1, s1, c1),
                _term(1, c1, c1, s1),
            )
        case "iwp":
            return (
                _term(2, c1, c1, one),
                _term(2, one, c1, c1),
                _term(2, c1, one, c1),
                _term(-1, c2, one, one),
                _term(-1, one, c2, one),
                _term(-1, one, one, c2),
            )
        case "lidinoid":
            return (
                _term(half, s2, c1, s1),
                _term(half, s1, s2, c1),
                _term(half, c1, s1, s2),
                _term(-half, c2, c2, one),
                _term(-half, one, c2, c2),
                _term(-half, c2, one, c2),
                _term(Fraction(3, 20), one, one, one),
            )
        case _:
            assert_never(family)


def _quarter_shift(
    kind: TrigonometricFactor, quarter_turns: int, /
) -> tuple[int, TrigonometricFactor]:
    """``trig(theta + k pi / 2)`` as ``sign * trig'(theta)``."""
    match kind:
        case "sin":
            return ((1, "sin"), (1, "cos"), (-1, "sin"), (-1, "cos"))[quarter_turns]
        case "cos":
            return ((1, "cos"), (-1, "sin"), (-1, "cos"), (1, "sin"))[quarter_turns]
        case _:
            assert_never(kind)


class TrigonometricTable(StrictModule):
    """Canonical finite trigonometric polynomial in reduced lattice coordinates.

    Terms with equal factors are merged, zero sums are dropped, and terms are
    ordered by factors, so equal polynomials have equal ``table_id``.
    """

    terms: tuple[TrigonometricTerm, ...] = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(self, terms: Sequence[TrigonometricTerm], /) -> None:
        merged: dict[_Factors, Fraction] = {}
        for term in terms:
            if not isinstance(term, TrigonometricTerm):
                raise TypeError(
                    "TrigonometricTable terms must be TrigonometricTerm values."
                )
            merged[term.factors] = (
                merged.get(term.factors, Fraction(0)) + term.coefficient
            )
        canonical = tuple(
            TrigonometricTerm(coefficient, factors)
            for factors, coefficient in sorted(merged.items())
            if coefficient != 0
        )
        if not any(any(frequency for _, frequency in term.factors) for term in canonical):
            raise ValueError("A trigonometric table needs at least one nonconstant term.")
        self.terms = canonical
        self.table_id = canonical_fingerprint(
            {
                "kind": "trigonometric-table",
                "terms": [
                    [str(term.coefficient), [list(factor) for factor in term.factors]]
                    for term in canonical
                ],
            }
        )

    @classmethod
    def family(cls, family: TrigonometricFamily, /) -> TrigonometricTable:
        """Return a named nodal family table (see the module documentation)."""
        return cls(_family_terms(parse(family, TrigonometricFamily, "family")))

    def scaled(self, factor: int | Fraction, /) -> TrigonometricTable:
        """Return ``factor * F`` exactly."""
        value = _rational(factor, "factor")
        if value == 0:
            raise ValueError("A trigonometric table cannot be scaled to zero.")
        return TrigonometricTable(
            tuple(
                TrigonometricTerm(value * term.coefficient, term.factors)
                for term in self.terms
            )
        )

    def composed(
        self, matrix: Sequence[Sequence[int]], shift: Sequence[int | Fraction], /
    ) -> TrigonometricTable:
        """Return the exact table of ``u -> F(A u + b)``.

        ``A`` must be a signed permutation matrix, so ``u -> A u + b`` is a
        measure-preserving affine bijection of the quotient cell. Each shifted
        harmonic ``trig(2 pi n (s u_p + b))`` stays a table harmonic only when
        ``4 n b`` is an integer (quarter turns); any other phase is refused.
        """
        rows = tuple(tuple(row) for row in matrix)
        shifts = tuple(_rational(value, "shift") for value in shift)
        if len(rows) != 3 or any(len(row) != 3 for row in rows) or len(shifts) != 3:
            raise ValueError("A lattice map needs a 3x3 matrix and a three-vector shift.")
        targets: list[tuple[int, int]] = []
        for row in rows:
            if any(
                isinstance(entry, bool) or not isinstance(entry, int) for entry in row
            ):
                raise TypeError("Lattice map matrix entries must be integers.")
            support = [column for column, entry in enumerate(row) if entry]
            if len(support) != 1 or abs(row[support[0]]) != 1:
                raise ValueError("The lattice map matrix must be a signed permutation.")
            targets.append((support[0], row[support[0]]))
        if sorted(column for column, _ in targets) != [0, 1, 2]:
            raise ValueError("The lattice map matrix must be a signed permutation.")
        composed_terms: list[TrigonometricTerm] = []
        for term in self.terms:
            coefficient = term.coefficient
            placed: list[tuple[str, int]] = [("cos", 0)] * 3
            for axis, (kind, frequency) in enumerate(term.factors):
                column, sign = targets[axis]
                phase = 4 * frequency * shifts[axis]
                if phase.denominator != 1:
                    raise ValueError(
                        "The lattice shift moves a harmonic by a non-quarter turn; "
                        "the composition is not a table polynomial."
                    )
                factor_sign, shifted = _quarter_shift(kind, int(phase) % 4)
                coefficient *= factor_sign
                placed[column] = (shifted, sign * frequency)
            composed_terms.append(TrigonometricTerm(coefficient, placed))
        return TrigonometricTable(composed_terms)

    def symmetry_sign(
        self, matrix: Sequence[Sequence[int]], shift: Sequence[int | Fraction], /
    ) -> int:
        """Prove ``F(A u + b) = sigma F(u)`` from the table, returning ``sigma``.

        Returns ``1`` or ``-1`` when the composed table equals ``F`` or ``-F``
        exactly and ``0`` otherwise. With ``sigma = -1`` the measure-preserving
        map exchanges ``{F < c}`` and ``{F > -c}``; at ``c = 0`` the network
        volume fraction is exactly one half.
        """
        composed = self.composed(matrix, shift).table_id
        if composed == self.table_id:
            return 1
        if composed == self.scaled(-1).table_id:
            return -1
        return 0

    @property
    def amplitude_bound(self) -> float:
        """Upper bound of ``sup |F|``: the sum of absolute coefficients."""
        total = 0.0
        for term in self.terms:
            total = _up(total + _fraction_interval(abs(term.coefficient))[1])
        return total

    @property
    def gradient_bound(self) -> float:
        """Upper bound of ``sup ||grad_u F||``: ``sum |a| 2 pi ||n||``."""
        total = 0.0
        for term in self.terms:
            norm = _up(math.sqrt(sum(n * n for _, n in term.factors)))
            weight = _up(_fraction_interval(abs(term.coefficient))[1] * _TWO_PI_UPPER)
            total = _up(total + _up(weight * norm))
        return total

    @property
    def hessian_bound(self) -> float:
        """Upper bound of ``sup ||Hess_u F||_2``: ``sum |a| (2 pi)^2 ||n||^2``."""
        square = _up(_TWO_PI_UPPER * _TWO_PI_UPPER)
        total = 0.0
        for term in self.terms:
            norm = float(sum(n * n for _, n in term.factors))
            weight = _up(_fraction_interval(abs(term.coefficient))[1] * square)
            total = _up(total + _up(weight * norm))
        return total

    def axis_frequencies(self, axis: int, /) -> tuple[int, ...]:
        """Sorted distinct nonzero frequencies of one reduced axis."""
        return tuple(sorted({term.factors[axis][1] for term in self.terms} - {0}))

    def evaluate_reduced(self, fractional: Array, /) -> Array:
        """Evaluate ``F`` at reduced coordinates with binary64 coefficients.

        Coordinates are reduced exactly, so lattice-equivalent binary64 inputs
        ``u`` and ``u + m`` evaluate to bitwise-identical values. Dyadic
        coefficients are exact; any other rational is rounded to nearest.
        Interval programs enclose the field and its derivatives in mean-value
        form from the exact table (see ``_table_jet``).
        """
        value = jnp.asarray(fractional, dtype=jnp.float64)
        if value.ndim == 0 or value.shape[-1] != 3:
            raise ValueError("Reduced coordinates must end in lattice rank 3.")
        # One flat (points, 3) operand puts every batch axis first, the layout
        # the declared interval extension of the table jet encloses.
        flat = value.reshape((-1, 3))
        return _table_jet(self, 0)(flat).reshape(value.shape[:-1])


def _table_derivative(
    table: TrigonometricTable, fractional: Array, order: int, /
) -> Array:
    """The order-``k`` derivative tensor of ``F`` at reduced coordinates."""
    harmonics: list[tuple[tuple[Array, ...], tuple[int, ...]] | None] = []
    for axis in range(3):
        frequencies = table.axis_frequencies(axis)
        harmonics.append(
            None
            if not frequencies
            else (
                tuple(
                    periodic_harmonic_jet(
                        fractional[..., axis, None],
                        jnp.asarray(frequencies, dtype=jnp.float64),
                        order=derivative,
                    )
                    for derivative in range(order + 1)
                ),
                frequencies,
            )
        )
    shape = fractional.shape[:-1]
    entries = []
    for index in itertools.product(range(3), repeat=order):
        counts = tuple(index.count(axis) for axis in range(3))
        total = jnp.zeros(shape, dtype=jnp.float64)
        for term in table.terms:
            if any(
                frequency == 0 and count
                for (_, frequency), count in zip(term.factors, counts, strict=True)
            ):
                continue
            product = jnp.full(shape, float(term.coefficient), dtype=jnp.float64)
            for axis, (kind, frequency) in enumerate(term.factors):
                entry = harmonics[axis]
                if frequency == 0 or entry is None:
                    continue
                jets, frequencies = entry
                product = (
                    product
                    * jets[counts[axis]][
                        ..., frequencies.index(frequency), _component(kind)
                    ]
                )
            total = total + product
        entries.append(total)
    return jnp.stack(entries, axis=-1).reshape(shape + (3,) * order)


# Table jet functions keyed by (table identity, derivative order).
_TABLE_JETS: dict[tuple[str, int], jax.custom_jvp[Array]] = {}
# Orders whose interval programs use mean-value enclosures from the table.
_ENCLOSED_TABLE_ORDERS = 3


def _table_jet(table: TrigonometricTable, order: int, /) -> jax.custom_jvp[Array]:
    """Custom-JVP derivative tensor of ``F`` whose tangent is the next tensor.

    Orders 0-2 declare interval extensions in mean-value form, which shrink
    quadratically with the box even near critical points where the natural
    extension of the harmonic products overestimates a vanishing gradient.
    """
    key = (table.table_id, order)
    cached = _TABLE_JETS.get(key)
    if cached is not None:
        return cached

    @jax.custom_jvp
    def jet(fractional: Array) -> Array:
        return _table_derivative(table, fractional, order)

    def rule(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
        (fractional,), (tangent,) = primals, tangents
        values = jet(fractional)
        if isinstance(tangent, SymbolicZero):
            tangent = jnp.zeros_like(fractional)
        following = _table_jet(table, order + 1)(fractional)
        direction = tangent.reshape(tangent.shape[:-1] + (1,) * order + (3,))
        return values, jnp.sum(following * direction, axis=-1)

    jet.defjvp(rule, symbolic_zeros=True)
    if order < _ENCLOSED_TABLE_ORDERS:
        enclosure = _TableEnclosure(table)

        def extension(operands: tuple[Interval, ...]) -> Interval:
            return _table_interval(enclosure, order, operands[0])

        register_custom_jvp_interval_extension(rule, extension)
    _TABLE_JETS[key] = jet
    return jet


def _table_interval(
    enclosure: _TableEnclosure, order: int, fractional: Interval, /
) -> Interval:
    lower, upper = _host(fractional)
    if lower.ndim < 2 or lower.shape[-1] != 3 or upper.shape != lower.shape:
        raise IntervalExtensionError(
            "A trigonometric table jet needs leading point axes and a last axis of 3."
        )
    shape = lower.shape[:-1] + (3,) * order
    lower, upper = lower.reshape((-1, 3)), upper.reshape((-1, 3))
    if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
        enclosed = enclosure.centered(lower, upper)
    else:
        natural = enclosure.evaluate(lower, upper, 2)
        if natural[1] is None or natural[2] is None:
            raise RuntimeError("The table derivative enclosures were not evaluated.")
        enclosed = (natural[0], natural[1], natural[2])
    low, high = enclosed[order]
    return low.reshape(shape), high.reshape(shape)


def _component(kind: TrigonometricFactor, /) -> int:
    match kind:
        case "sin":
            return 0
        case "cos":
            return 1
        case _:
            assert_never(kind)


class _TableEnclosure:
    """Outward interval value, gradient, and Hessian of a table over reduced boxes.

    Rational coefficients enter as outward binary64 intervals, so every
    enclosure holds for the exact table and for its binary64-coefficient
    evaluation alike.
    """

    def __init__(self, table: TrigonometricTable, /) -> None:
        self.frequencies = tuple(
            np.asarray(table.axis_frequencies(axis), dtype=np.float64)
            for axis in range(3)
        )
        columns = tuple(
            {
                frequency: index
                for index, frequency in enumerate(table.axis_frequencies(axis))
            }
            for axis in range(3)
        )
        self.terms = tuple(
            (
                _fraction_interval(term.coefficient),
                tuple(
                    None
                    if frequency == 0
                    else (columns[axis][frequency], _component(kind))
                    for axis, (kind, frequency) in enumerate(term.factors)
                ),
            )
            for term in table.terms
        )

    def _jets(
        self, lower: np.ndarray, upper: np.ndarray, order: int, /
    ) -> tuple[tuple[_Bounds, ...], ...]:
        return tuple(
            tuple(
                _host(
                    interval_periodic_harmonic(
                        (lower[:, axis, None], upper[:, axis, None]),
                        self.frequencies[axis][None, :],
                        derivative,
                    )
                )
                if self.frequencies[axis].size
                else (np.empty((0,)), np.empty((0,)))
                for derivative in range(order + 1)
            )
            for axis in range(3)
        )

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, order: int, /
    ) -> tuple[_Bounds, _Bounds | None, _Bounds | None]:
        """Return value ``(N,)``, gradient ``(N, 3)``, and Hessian ``(N, 3, 3)``."""
        count = lower.shape[0]
        jets = self._jets(lower, upper, order)
        zero: _Bounds = (np.zeros((count,)), np.zeros((count,)))

        def factor(axis: int, derivative: int, entry: tuple[int, int]) -> _Bounds:
            low, high = jets[axis][derivative]
            return low[:, entry[0], entry[1]], high[:, entry[0], entry[1]]

        def product(coefficient: tuple[float, float], parts: list[_Bounds]) -> _Bounds:
            result: _Bounds = (
                np.full((count,), coefficient[0]),
                np.full((count,), coefficient[1]),
            )
            for part in parts:
                result = _host(interval_multiply(result, part))
            return result

        def add(first: _Bounds, second: _Bounds) -> _Bounds:
            return _host(interval_add(first, second))

        value = zero
        gradient = [zero, zero, zero]
        hessian = [[zero] * 3 for _ in range(3)]
        for coefficient, entries in self.terms:
            values = [
                None if entry is None else factor(axis, 0, entry)
                for axis, entry in enumerate(entries)
            ]
            constant = [part for part in values if part is not None]
            value = add(value, product(coefficient, constant))
            if order < 1:
                continue
            for first in range(3):
                first_entry = entries[first]
                if first_entry is None:
                    continue
                others = [
                    part
                    for axis, part in enumerate(values)
                    if axis != first and part is not None
                ]
                gradient[first] = add(
                    gradient[first],
                    product(coefficient, [factor(first, 1, first_entry), *others]),
                )
                if order < 2:
                    continue
                hessian[first][first] = add(
                    hessian[first][first],
                    product(coefficient, [factor(first, 2, first_entry), *others]),
                )
                for second in range(first + 1, 3):
                    second_entry = entries[second]
                    if second_entry is None:
                        continue
                    rest = [
                        part
                        for axis, part in enumerate(values)
                        if axis not in (first, second) and part is not None
                    ]
                    hessian[first][second] = add(
                        hessian[first][second],
                        product(
                            coefficient,
                            [
                                factor(first, 1, first_entry),
                                factor(second, 1, second_entry),
                                *rest,
                            ],
                        ),
                    )
        gradient_interval = (
            None
            if order < 1
            else (
                np.stack([part[0] for part in gradient], axis=1),
                np.stack([part[1] for part in gradient], axis=1),
            )
        )
        hessian_interval = None
        if order >= 2:
            for first in range(3):
                for second in range(first):
                    hessian[first][second] = hessian[second][first]
            hessian_interval = (
                np.stack(
                    [np.stack([entry[0] for entry in row], axis=1) for row in hessian],
                    axis=1,
                ),
                np.stack(
                    [np.stack([entry[1] for entry in row], axis=1) for row in hessian],
                    axis=1,
                ),
            )
        return value, gradient_interval, hessian_interval

    def centered(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[_Bounds, _Bounds, _Bounds]:
        """Mean-value enclosures intersected with the natural ones.

        ``grad F(X) in grad F(m) + Hess F(X) (X - m)`` and ``F(X) in F(m) +
        grad F(X) . (X - m)`` shrink quadratically with the box even where
        the natural extension overestimates a vanishing gradient.
        """
        middle = 0.5 * lower + 0.5 * upper
        point_value, point_gradient, _ = self.evaluate(middle, middle, 1)
        value, gradient, hessian = self.evaluate(lower, upper, 2)
        if point_gradient is None or gradient is None or hessian is None:
            raise RuntimeError("The table derivative enclosures were not evaluated.")
        offset = round_interval(lower - middle, upper - middle)
        change = interval_times_interval(
            hessian[0], hessian[1], offset[0][..., None], offset[1][..., None]
        )
        centered_gradient = round_interval(
            point_gradient[0] + change[0][..., 0], point_gradient[1] + change[1][..., 0]
        )
        gradient = (
            np.maximum(gradient[0], centered_gradient[0]),
            np.minimum(gradient[1], centered_gradient[1]),
        )
        slope = interval_times_interval(
            gradient[0][:, None, :],
            gradient[1][:, None, :],
            offset[0][..., None],
            offset[1][..., None],
        )
        centered_value = round_interval(
            point_value[0] + slope[0][:, 0, 0], point_value[1] + slope[1][:, 0, 0]
        )
        value = (
            np.maximum(value[0], centered_value[0]),
            np.minimum(value[1], centered_value[1]),
        )
        return value, gradient, hessian


class _GradientValues:
    def __init__(self, enclosure: _TableEnclosure, /) -> None:
        self._enclosure = enclosure

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        gradient = self._enclosure.evaluate(lower, upper, 1)[1]
        if gradient is None:
            raise RuntimeError("The table gradient enclosure was not evaluated.")
        return gradient


class _HessianValues:
    def __init__(self, enclosure: _TableEnclosure, /) -> None:
        self._enclosure = enclosure

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        hessian = self._enclosure.evaluate(lower, upper, 2)[2]
        if hessian is None:
            raise RuntimeError("The table Hessian enclosure was not evaluated.")
        return hessian


class _CriticalPointSystem:
    """Krawczyk system ``grad_u F = 0`` with Hessian interval Jacobians."""

    def __init__(self, enclosure: _TableEnclosure, /) -> None:
        self._value = _GradientValues(enclosure)
        self._jacobian = _HessianValues(enclosure)

    @property
    def value(self) -> _GradientValues:
        return self._value

    @property
    def jacobian(self) -> _HessianValues:
        return self._jacobian

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        lower, upper = self._jacobian.evaluate(points, points)
        return 0.5 * lower + 0.5 * upper


def _merge_intervals(
    lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(lower, kind="stable")
    merged_lower: list[float] = []
    merged_upper: list[float] = []
    for low, high in zip(lower[order].tolist(), upper[order].tolist(), strict=True):
        if merged_upper and low <= merged_upper[-1]:
            merged_upper[-1] = max(merged_upper[-1], high)
        else:
            merged_lower.append(low)
            merged_upper.append(high)
    return np.asarray(merged_lower, dtype=np.float64), np.asarray(
        merged_upper, dtype=np.float64
    )


def _readonly(values: np.ndarray, /) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


_QUARTER_SIN = (0, 1, 0, -1)
_QUARTER_COS = (1, 0, -1, 0)


def _quarter_jet(
    table: TrigonometricTable, point: tuple[int, int, int], /
) -> tuple[Fraction, tuple[Fraction, Fraction, Fraction]]:
    """Exact ``F`` and ``grad_u F / (2 pi)`` at the quarter-lattice point ``point / 4``.

    Every harmonic ``sin``/``cos(2 pi n k / 4)`` is exactly ``0`` or ``+-1``,
    so the value and the scaled gradient are exact rationals.
    """
    value = Fraction(0)
    gradient = [Fraction(0), Fraction(0), Fraction(0)]
    for term in table.terms:
        factors: list[int] = []
        slopes: list[int] = []
        for (kind, frequency), quarter in zip(term.factors, point, strict=True):
            phase = (frequency * quarter) % 4
            sine, cosine = _QUARTER_SIN[phase], _QUARTER_COS[phase]
            match kind:
                case "sin":
                    factors.append(sine)
                    slopes.append(frequency * cosine)
                case "cos":
                    factors.append(cosine)
                    slopes.append(-frequency * sine)
                case _:
                    assert_never(kind)
        value += term.coefficient * factors[0] * factors[1] * factors[2]
        for axis in range(3):
            product = slopes[axis]
            for other in range(3):
                if other != axis:
                    product *= factors[other]
            gradient[axis] += term.coefficient * product
    return value, (gradient[0], gradient[1], gradient[2])


def _quarter_critical_points(
    table: TrigonometricTable, /
) -> tuple[np.ndarray, tuple[Fraction, ...]]:
    """Quarter-lattice points with exactly vanishing gradient and their exact values."""
    points: list[tuple[int, int, int]] = []
    values: list[Fraction] = []
    for point in itertools.product(range(4), repeat=3):
        value, gradient = _quarter_jet(table, (point[0], point[1], point[2]))
        if not any(gradient):
            points.append((point[0], point[1], point[2]))
            values.append(value)
    return np.asarray(points, dtype=np.float64).reshape((-1, 3)) / 4.0, tuple(values)


class TrigonometricCriticalLevels(StrictModule, NonTrainableState):
    """Certified enclosure of every critical value of a table on the quotient cell.

    ``root_boxes`` are Krawczyk-certified boxes, each holding exactly one
    regular critical point of ``F`` (a point may recur across adjacent or
    periodically identified boxes). ``unresolved_boxes`` may hold critical
    points that were not isolated: their value enclosure narrowed to the
    declared ``value_resolution`` of the global cover, or the box reached
    binary64 resolution (for example around a degenerate or non-isolated
    critical set); their whole value enclosures are retained. Every critical
    value lies in one of the merged ``value_lower``/``value_upper`` intervals.

    The global cover is coarse but sound. ``separate`` resolves one requested
    level on demand by refining only the boxes whose value enclosure meets it.
    ``quarter_points`` are the quarter-lattice points where ``grad_u F``
    vanishes exactly, with their exact rational ``quarter_values``.
    """

    table: TrigonometricTable
    root_boxes: np.ndarray
    root_values: np.ndarray
    unresolved_boxes: np.ndarray
    unresolved_values: np.ndarray
    value_lower: np.ndarray
    value_upper: np.ndarray
    quarter_points: np.ndarray
    quarter_values: tuple[Fraction, ...] = eqx.field(static=True)
    value_resolution: float = eqx.field(static=True)

    def __init__(
        self,
        table: TrigonometricTable,
        /,
        *,
        root_boxes: np.ndarray,
        root_values: np.ndarray,
        unresolved_boxes: np.ndarray,
        unresolved_values: np.ndarray,
        value_resolution: float,
    ) -> None:
        boxes = _readonly(root_boxes).reshape((-1, 2, 3))
        values = _readonly(root_values).reshape((-1, 2))
        open_boxes = _readonly(unresolved_boxes).reshape((-1, 2, 3))
        open_values = _readonly(unresolved_values).reshape((-1, 2))
        if (
            boxes.shape[0] != values.shape[0]
            or open_boxes.shape[0] != open_values.shape[0]
        ):
            raise ValueError("Critical boxes and value enclosures must align.")
        everything = np.concatenate((values, open_values), axis=0)
        if np.any(everything[:, 0] > everything[:, 1]) or not np.all(
            np.isfinite(everything)
        ):
            raise ValueError("Critical value enclosures must be finite and ordered.")
        lower, upper = _merge_intervals(everything[:, 0], everything[:, 1])
        points, exact = _quarter_critical_points(table)
        self.table = table
        self.root_boxes = boxes
        self.root_values = values
        self.unresolved_boxes = open_boxes
        self.unresolved_values = open_values
        self.value_lower = _readonly(lower)
        self.value_upper = _readonly(upper)
        self.quarter_points = _readonly(points)
        self.quarter_values = exact
        self.value_resolution = float(value_resolution)

    @property
    def table_id(self) -> str:
        return self.table.table_id

    @property
    def resolved(self) -> bool:
        return self.unresolved_boxes.shape[0] == 0

    def enclosure_meeting(self, level: float, /) -> tuple[float, float] | None:
        """The merged critical-value enclosure containing ``level``, if any."""
        value = float(level)
        hits = np.flatnonzero((self.value_lower <= value) & (value <= self.value_upper))
        if not hits.size:
            return None
        return float(self.value_lower[hits[0]]), float(self.value_upper[hits[0]])

    def separation(self, level: float, /) -> float:
        """Lower bound of the distance from ``level`` to every critical value."""
        value = float(level)
        if not self.value_lower.size:
            return math.inf
        gaps = np.maximum(self.value_lower - value, value - self.value_upper)
        return _down(float(np.min(gaps)))

    def separate(self, level: float, /) -> TrigonometricLevelSeparation:
        """Resolve ``level`` against the critical values on demand.

        Only boxes whose value enclosure contains ``level`` are refined, until
        the level is separated from every enclosure, a critical point whose
        value enclosure still contains it is proved (Krawczyk, contracted to
        its binary64 fixed point, or an exact quarter-lattice critical point),
        or a box meeting it reaches binary64 resolution. The returned
        certificate is this one with the refined boxes substituted.
        """
        return _separate_level(self, level)


TrigonometricLevelOutcome: TypeAlias = Literal[
    "separated", "critical_point", "quarter_point", "binary64_resolution"
]


class TrigonometricLevelSeparation(StrictModule, NonTrainableState):
    """Decided outcome of resolving one level against the critical values.

    ``separated``: no critical-value enclosure of ``critical_levels`` contains
    ``level``. ``critical_point``: a Krawczyk-certified critical point at its
    binary64 fixed point has a value enclosure containing ``level``.
    ``quarter_point``: ``level`` equals the exact critical value at a
    quarter-lattice point where the gradient vanishes exactly.
    ``binary64_resolution``: a box that may hold a critical point and whose
    value enclosure contains ``level`` cannot be bisected in binary64.
    ``evidence_box`` (``(2, 3)``, reduced coordinates) and ``evidence_value``
    carry the deciding box and value enclosure of a refusal.
    """

    critical_levels: TrigonometricCriticalLevels
    evidence_box: np.ndarray | None
    level: float = eqx.field(static=True)
    outcome: TrigonometricLevelOutcome = eqx.field(static=True)
    evidence_value: tuple[float, float] | None = eqx.field(static=True)
    refined_boxes: int = eqx.field(static=True)

    def __init__(
        self,
        critical_levels: TrigonometricCriticalLevels,
        /,
        *,
        level: float,
        outcome: TrigonometricLevelOutcome,
        evidence_box: np.ndarray | None,
        evidence_value: tuple[float, float] | None,
        refined_boxes: int,
    ) -> None:
        self.critical_levels = critical_levels
        self.evidence_box = (
            None if evidence_box is None else _readonly(evidence_box).reshape((2, 3))
        )
        self.level = float(level)
        self.outcome = outcome
        self.evidence_value = evidence_value
        self.refined_boxes = int(refined_boxes)

    @property
    def separated(self) -> bool:
        return self.outcome == "separated"


def _contract_root(
    system: _CriticalPointSystem, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect certified boxes with their Krawczyk images to the binary64 fixed point.

    Each image holds the unique root; the boxes shrink monotonically over a
    finite set of binary64 endpoints, so the loop ends.
    """
    while True:
        result = krawczyk(system, lower, upper)
        next_lower = np.maximum(lower, result.lower)
        next_upper = np.minimum(upper, result.upper)
        if np.array_equal(next_lower, lower) and np.array_equal(next_upper, upper):
            return lower, upper
        lower, upper = next_lower, next_upper


type _Deferral = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class _CriticalStep:
    """One branch-and-bound pass over search boxes for ``grad_u F = 0``."""

    def __init__(self, table: TrigonometricTable, /) -> None:
        self.enclosure = _TableEnclosure(table)
        self.system = _CriticalPointSystem(self.enclosure)

    def roots(self, lower: np.ndarray, upper: np.ndarray, /) -> tuple[_Bounds, _Bounds]:
        """Contract certified boxes to their fixed point; return boxes and values."""
        lower, upper = _contract_root(self.system, lower, upper)
        value = self.enclosure.centered(lower, upper)[0]
        return (lower, upper), value

    def __call__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        value_resolution: float | None = None,
        level: float | None = None,
    ) -> tuple[list[_Bounds], _Deferral, _Deferral, _Bounds]:
        """Decide one batch of search boxes.

        Exactly one deferral rule applies: an undecided box is deferred when its
        value enclosure is at most ``value_resolution`` wide (the global cover),
        or when it excludes ``level`` (on-demand separation). Returns certified
        root boxes, deferred boxes, boxes at binary64 resolution (each with
        value enclosures), and the bisected children.
        """
        empty = np.empty((0, 3)), np.empty((0, 3)), np.empty((0,)), np.empty((0,))
        value, gradient, _ = self.enclosure.centered(lower, upper)
        possible = np.all((gradient[0] <= 0.0) & (gradient[1] >= 0.0), axis=1)
        lower, upper = lower[possible], upper[possible]
        value = (value[0][possible], value[1][possible])
        roots: list[_Bounds] = []
        if not lower.shape[0]:
            return roots, empty, empty, (np.empty((0, 3)), np.empty((0, 3)))
        result = krawczyk(self.system, lower, upper)
        certified = result.certified
        if np.any(certified):
            roots.append((lower[certified], upper[certified]))
        open_rows = np.flatnonzero(~certified & ~result.excluded)
        regular = result.contraction[open_rows] < 1.0
        if np.any(regular):
            rows = open_rows[regular]
            # A regular root on a box face never reaches a bisected interior;
            # one inflated Krawczyk test certifies or excludes it.
            margin = 0.25 * (upper[rows] - lower[rows])
            inflated_lower, inflated_upper = lower[rows] - margin, upper[rows] + margin
            inflated = krawczyk(self.system, inflated_lower, inflated_upper)
            if np.any(inflated.certified):
                roots.append(
                    (
                        inflated_lower[inflated.certified],
                        inflated_upper[inflated.certified],
                    )
                )
            settled = np.zeros(open_rows.shape[0], dtype=np.bool_)
            settled[regular] = inflated.certified | inflated.excluded
            open_rows = open_rows[~settled]
        lower, upper = lower[open_rows], upper[open_rows]
        value = (value[0][open_rows], value[1][open_rows])
        if value_resolution is not None and level is None:
            deferred = value[1] - value[0] <= value_resolution
        elif level is not None and value_resolution is None:
            deferred = (value[1] < level) | (level < value[0])
        else:
            raise ValueError("Exactly one critical-search deferral rule applies.")
        held = (lower[deferred], upper[deferred], value[0][deferred], value[1][deferred])
        lower, upper = lower[~deferred], upper[~deferred]
        value = (value[0][~deferred], value[1][~deferred])
        middle = 0.5 * lower + 0.5 * upper
        splittable = (middle > lower) & (middle < upper)
        divisible = np.any(splittable, axis=1)
        resolution = (
            lower[~divisible],
            upper[~divisible],
            value[0][~divisible],
            value[1][~divisible],
        )
        lower, upper = lower[divisible], upper[divisible]
        middle, splittable = middle[divisible], splittable[divisible]
        axis = np.argmax(np.where(splittable, upper - lower, -1.0), axis=1)
        rows = np.arange(lower.shape[0])
        left_upper = upper.copy()
        left_upper[rows, axis] = middle[rows, axis]
        right_lower = lower.copy()
        right_lower[rows, axis] = middle[rows, axis]
        return (
            roots,
            held,
            resolution,
            (
                np.concatenate((lower, right_lower), axis=0),
                np.concatenate((left_upper, upper), axis=0),
            ),
        )


def _stack_boxes(parts: Sequence[_Bounds], /) -> tuple[np.ndarray, np.ndarray]:
    if not parts:
        return np.empty((0, 3)), np.empty((0, 3))
    return (
        np.concatenate([part[0] for part in parts], axis=0),
        np.concatenate([part[1] for part in parts], axis=0),
    )


def _stack_deferrals(parts: Sequence[_Deferral], /) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.concatenate(
        [np.stack(part[:2], axis=1) for part in parts] or [np.empty((0, 2, 3))]
    )
    values = np.concatenate(
        [np.stack(part[2:], axis=1) for part in parts] or [np.empty((0, 2))]
    )
    return boxes, values


@checked
def certify_trigonometric_critical_levels(
    table: TrigonometricTable,
    /,
    *,
    value_resolution: float = 2.0**-8,
) -> TrigonometricCriticalLevels:
    """Enclose all solutions of ``grad_u F = 0`` on the quotient cell and their values.

    Branch and bound covers the closed reduced cube, which meets every orbit of
    the torus. A box is discarded when its gradient enclosure excludes zero or
    its Krawczyk image is disjoint from it, and certified when the image lies in
    its interior (inflating a regular box once when its root sits on a face).
    ``value_resolution`` is the declared resolution of this global cover: an
    undecided box whose value enclosure is at most that wide is retained as
    unresolved evidence with its whole value enclosure, as is a box at binary64
    resolution. It never refuses a level by itself: a requested level meeting a
    retained enclosure is resolved on demand by
    ``TrigonometricCriticalLevels.separate``. A degenerate critical set never
    admits a Krawczyk proof: I-WP, for example, is critical with value 3 along
    every body diagonal. Critical points of ``F(x)`` in Cartesian coordinates
    coincide with these for every lattice.
    """
    resolution = float(value_resolution)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("value_resolution must be finite and positive.")
    step = _CriticalStep(table)
    grid = np.stack(
        np.meshgrid(*(np.arange(_CRITICAL_GRID, dtype=np.float64),) * 3, indexing="ij"),
        axis=-1,
    ).reshape((-1, 3))
    lower, upper = grid / _CRITICAL_GRID, (grid + 1.0) / _CRITICAL_GRID
    roots: list[_Bounds] = []
    unresolved: list[_Deferral] = []
    while lower.shape[0]:
        children: list[_Bounds] = []
        for start in range(0, lower.shape[0], _CRITICAL_CHUNK):
            stop = start + _CRITICAL_CHUNK
            found, held, resolution_boxes, split = step(
                lower[start:stop], upper[start:stop], value_resolution=resolution
            )
            roots.extend(found)
            unresolved.extend((held, resolution_boxes))
            children.append(split)
        lower, upper = _stack_boxes(children)

    root_lower, root_upper = _stack_boxes(roots)
    if root_lower.shape[0]:
        (root_lower, root_upper), root_value = step.roots(root_lower, root_upper)
        root_values = np.stack(root_value, axis=1)
    else:
        root_values = np.empty((0, 2))
    unresolved_boxes, unresolved_values = _stack_deferrals(unresolved)
    return TrigonometricCriticalLevels(
        table,
        root_boxes=np.stack((root_lower, root_upper), axis=1),
        root_values=root_values,
        unresolved_boxes=unresolved_boxes,
        unresolved_values=unresolved_values,
        value_resolution=resolution,
    )


def _separate_level(
    certificate: TrigonometricCriticalLevels, level: float, /
) -> TrigonometricLevelSeparation:
    value = float(level)
    if not math.isfinite(value):
        raise ValueError("level must be finite.")

    def decided(
        outcome: TrigonometricLevelOutcome,
        refined: TrigonometricCriticalLevels,
        box: np.ndarray | None,
        enclosure: tuple[float, float] | None,
        count: int,
    ) -> TrigonometricLevelSeparation:
        return TrigonometricLevelSeparation(
            refined,
            level=value,
            outcome=outcome,
            evidence_box=box,
            evidence_value=enclosure,
            refined_boxes=count,
        )

    exact = Fraction(value)
    for point, critical in zip(
        certificate.quarter_points, certificate.quarter_values, strict=True
    ):
        if critical == exact:
            return decided(
                "quarter_point", certificate, np.stack((point, point)), (value, value), 0
            )
    if certificate.enclosure_meeting(value) is None:
        return decided("separated", certificate, None, None, 0)

    step = _CriticalStep(certificate.table)
    root_lower = certificate.root_boxes[:, 0]
    root_upper = certificate.root_boxes[:, 1]
    root_values = certificate.root_values
    meets = (root_values[:, 0] <= value) & (value <= root_values[:, 1])
    if np.any(meets):
        (lower, upper), contracted = step.roots(root_lower[meets], root_upper[meets])
        root_lower, root_upper = root_lower.copy(), root_upper.copy()
        root_values = root_values.copy()
        root_lower[meets], root_upper[meets] = lower, upper
        root_values[meets] = np.stack(contracted, axis=1)
        hits = np.flatnonzero(
            meets & (root_values[:, 0] <= value) & (value <= root_values[:, 1])
        )
        if hits.size:
            first = int(hits[0])
            return decided(
                "critical_point",
                certificate,
                np.stack((root_lower[first], root_upper[first])),
                (float(root_values[first, 0]), float(root_values[first, 1])),
                0,
            )
    roots: list[_Bounds] = [(root_lower, root_upper)]
    root_value_parts = [root_values]
    open_values = certificate.unresolved_values
    meets = (open_values[:, 0] <= value) & (value <= open_values[:, 1])
    kept: list[_Deferral] = [
        (
            certificate.unresolved_boxes[~meets, 0],
            certificate.unresolved_boxes[~meets, 1],
            open_values[~meets, 0],
            open_values[~meets, 1],
        )
    ]
    lower = certificate.unresolved_boxes[meets, 0]
    upper = certificate.unresolved_boxes[meets, 1]
    refined = 0
    while lower.shape[0]:
        refined += lower.shape[0]
        children: list[_Bounds] = []
        for start in range(0, lower.shape[0], _CRITICAL_CHUNK):
            stop = start + _CRITICAL_CHUNK
            found, held, resolution_boxes, split = step(
                lower[start:stop], upper[start:stop], level=value
            )
            if found:
                boxes, contracted = step.roots(*_stack_boxes(found))
                values = np.stack(contracted, axis=1)
                hits = np.flatnonzero((values[:, 0] <= value) & (value <= values[:, 1]))
                if hits.size:
                    first = int(hits[0])
                    return decided(
                        "critical_point",
                        certificate,
                        np.stack((boxes[0][first], boxes[1][first])),
                        (float(values[first, 0]), float(values[first, 1])),
                        refined,
                    )
                roots.append(boxes)
                root_value_parts.append(values)
            if resolution_boxes[0].shape[0]:
                return decided(
                    "binary64_resolution",
                    certificate,
                    np.stack((resolution_boxes[0][0], resolution_boxes[1][0])),
                    (float(resolution_boxes[2][0]), float(resolution_boxes[3][0])),
                    refined,
                )
            kept.append(held)
            children.append(split)
        lower, upper = _stack_boxes(children)
    root_lower, root_upper = _stack_boxes(roots)
    unresolved_boxes, unresolved_values = _stack_deferrals(kept)
    resolved = TrigonometricCriticalLevels(
        certificate.table,
        root_boxes=np.stack((root_lower, root_upper), axis=1),
        root_values=np.concatenate(root_value_parts, axis=0),
        unresolved_boxes=unresolved_boxes,
        unresolved_values=unresolved_values,
        value_resolution=certificate.value_resolution,
    )
    return decided("separated", resolved, None, None, refined)


class TrigonometricSingularLevelError(ValueError):
    """A requested boundary level is not separated from the critical values.

    ``separation`` carries the decided outcome and its evidence box/value.
    """

    def __init__(self, separation: TrigonometricLevelSeparation, /) -> None:
        enclosure = separation.evidence_value
        if separation.separated or enclosure is None:
            raise ValueError("A singular-level refusal needs a decided refusal.")
        super().__init__(
            f"Level {separation.level!r} is not separated from the critical values "
            f"({separation.outcome}; value enclosure [{enclosure[0]!r}, "
            f"{enclosure[1]!r}]): the level set is singular (a certified "
            "topology-change event) and is refused."
        )
        self.level = separation.level
        self.enclosure = enclosure
        self.critical_levels = separation.critical_levels
        self.separation = separation


def _matrix_norm_upper(matrix: np.ndarray, /) -> float:
    """Upper bound of the spectral norm through the outward Frobenius norm."""
    total = 0.0
    for value in np.abs(np.asarray(matrix, dtype=np.float64)).reshape((-1,)).tolist():
        total = _up(total + _up(value * value))
    return _up(math.sqrt(total))


def _cartesian_gradient_norm_upper(
    inverse: np.ndarray, gradient: _Bounds, /
) -> np.ndarray:
    """Outward ``||M grad_u F||`` over boxes, with ``grad_x F = M grad_u F``."""
    count = gradient[0].shape[0]
    squares = np.zeros((count,), dtype=np.float64)
    for row in range(3):
        component: _Bounds = (np.zeros((count,)), np.zeros((count,)))
        for column in range(3):
            entry = np.float64(inverse[row, column])
            component = _host(
                interval_add(
                    component,
                    interval_multiply(
                        (entry, entry), (gradient[0][:, column], gradient[1][:, column])
                    ),
                )
            )
        magnitude = np.maximum(np.abs(component[0]), np.abs(component[1]))
        squares = np.nextafter(
            squares + np.nextafter(magnitude * magnitude, np.inf), np.inf
        )
    return np.nextafter(np.sqrt(squares), np.inf)


def _require_cell(cell: PeriodicCell, /) -> np.ndarray:
    if cell.rank != 3 or cell.ambient_dimension != 3 or not cell.fully_periodic:
        raise ValueError(
            "A trigonometric level set needs a fully periodic rank-3 cell in 3D."
        )
    if cell.vectors.dtype != jnp.float64:
        raise ValueError("A trigonometric level set needs a float64 PeriodicCell.")
    return np.asarray(cell.inverse_vectors, dtype=np.float64)


class TrigonometricSheetThickness(StrictModule, NonTrainableState):
    """Certified minimum wall thickness of level bands ``{|F - c| <= tau}``.

    A reduced-cell box cover of the band ``tau <= maximum_half_width`` gives,
    for each box, a lower bound ``d`` of ``|F - c|`` and an upper bound of
    ``||grad_x F||``. ``g(s)``, the largest gradient bound over boxes with
    ``d <= s``, bounds ``sup ||grad_x F||`` over the band of half width ``s``.
    Every path between ``F = c - tau`` and ``F = c + tau`` crosses each slab
    ``s <= |F - c| <= s + ds`` inside it, so the wall is at least
    ``2 * integral_0^tau ds / g(s)`` thick. That certified bound dominates
    ``2 tau / g(tau)``, the uniform ``2 tau / sup ||grad F||`` bound, and is
    strictly increasing in ``tau``. No average gradient enters either bound.
    """

    offsets: np.ndarray
    gradient_upper: np.ndarray
    level: float = eqx.field(static=True)
    maximum_half_width: float = eqx.field(static=True)
    resolution_level: int = eqx.field(static=True)
    sampled_gradient_norm: float = eqx.field(static=True)
    box_count: int = eqx.field(static=True)
    table_id: str = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        offsets: np.ndarray,
        gradient_upper: np.ndarray,
        level: float,
        maximum_half_width: float,
        resolution_level: int,
        sampled_gradient_norm: float,
        box_count: int,
        table_id: str,
        cell_id: str,
    ) -> None:
        steps = _readonly(offsets)
        bounds = _readonly(gradient_upper)
        if (
            steps.ndim != 1
            or bounds.shape != steps.shape
            or not steps.size
            or steps[0] != 0.0
            or np.any(np.diff(steps) < 0.0)
            or np.any(np.diff(bounds) < 0.0)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds <= 0.0)
        ):
            raise ValueError(
                "Sheet thickness steps need sorted offsets from 0 and positive nondecreasing bounds."
            )
        self.offsets = steps
        self.gradient_upper = bounds
        self.level = float(level)
        self.maximum_half_width = float(maximum_half_width)
        self.resolution_level = int(resolution_level)
        self.sampled_gradient_norm = float(sampled_gradient_norm)
        self.box_count = int(box_count)
        self.table_id = table_id
        self.cell_id = cell_id

    def _require_half_width(self, half_width: float, /) -> float:
        value = float(half_width)
        if not math.isfinite(value) or not 0.0 < value <= self.maximum_half_width:
            raise ValueError("half_width must lie in (0, maximum_half_width].")
        return value

    def gradient_norm_upper(self, half_width: float, /) -> float:
        """Certified upper bound of ``sup ||grad_x F||`` over the band."""
        value = self._require_half_width(half_width)
        return float(
            self.gradient_upper[np.searchsorted(self.offsets, value, side="right") - 1]
        )

    def minimum_thickness(self, half_width: float, /) -> float:
        """Certified lower bound of the band's wall thickness (binary64 rounded down)."""
        value = self._require_half_width(half_width)
        active = np.searchsorted(self.offsets, value, side="left")
        ends = np.append(self.offsets[1:active], value)
        widths = np.nextafter(ends - self.offsets[:active], -np.inf)
        terms = np.nextafter(
            np.maximum(widths, 0.0) / self.gradient_upper[:active], -np.inf
        )
        return _down(2.0 * _down(math.fsum(terms.tolist())))

    def half_width_for_thickness(self, thickness: float, /) -> float:
        """Smallest certified half width whose minimum wall thickness meets ``thickness``.

        The bound is piecewise linear in ``tau``; the crossing is solved on its
        segment and rounded up until the certified bound meets the target.
        """
        target = float(thickness)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("thickness must be finite and positive.")
        if self.minimum_thickness(self.maximum_half_width) < target:
            raise ValueError(
                "The thickness target is not certified within maximum_half_width."
            )
        steps = np.append(self.offsets, self.maximum_half_width)
        widths = np.maximum(np.diff(steps), 0.0)
        reached = 2.0 * np.concatenate(((0.0,), np.cumsum(widths / self.gradient_upper)))
        segment = min(int(np.searchsorted(reached, target, side="left")), steps.size - 1)
        segment = max(segment - 1, 0)
        estimate = (
            steps[segment]
            + 0.5 * (target - reached[segment]) * self.gradient_upper[segment]
        )
        candidate = min(
            max(float(estimate), float(np.nextafter(0.0, 1.0))), self.maximum_half_width
        )
        while (
            candidate > float(np.nextafter(0.0, 1.0))
            and self.minimum_thickness(_down(candidate)) >= target
        ):
            candidate = _down(candidate)
        while self.minimum_thickness(candidate) < target:
            candidate = min(_up(candidate), self.maximum_half_width)
        return candidate


@checked
def certify_trigonometric_sheet_thickness(
    table: TrigonometricTable,
    cell: PeriodicCell,
    /,
    *,
    level: float,
    maximum_half_width: float,
    resolution_level: int = 6,
) -> TrigonometricSheetThickness:
    """Certify minimum wall thicknesses of every band up to ``maximum_half_width``.

    The reduced cell is bisected uniformly through ``resolution_level`` dyadic
    levels, keeping only boxes whose value enclosure meets the widest band.
    ``sampled_gradient_norm`` is the largest gradient norm measured at kept box
    centers inside the band: measured evidence of the bound's sharpness, not a
    certified quantity.
    """
    inverse = _require_cell(cell)
    center = float(level)
    limit = float(maximum_half_width)
    if not math.isfinite(center):
        raise ValueError("level must be finite.")
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("maximum_half_width must be finite and positive.")
    if isinstance(resolution_level, bool) or not isinstance(resolution_level, int):
        raise TypeError("resolution_level must be an integer.")
    if not 1 <= resolution_level <= 12:
        raise ValueError("resolution_level must lie in [1, 12].")
    enclosure = _TableEnclosure(table)
    corners = np.stack(np.meshgrid(*((0.0, 0.5),) * 3, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    lower = np.zeros((1, 3), dtype=np.float64)
    upper = np.ones((1, 3), dtype=np.float64)

    def distance(value: _Bounds) -> np.ndarray:
        below = np.nextafter(value[0] - center, -np.inf)
        above = np.nextafter(center - value[1], -np.inf)
        return np.maximum(np.maximum(below, above), 0.0)

    for depth in range(resolution_level):
        size = 0.5**depth
        lower = (lower[:, None, :] + size * corners[None, :, :]).reshape((-1, 3))
        upper = lower + 0.5 * size
        kept = distance(enclosure.centered(lower, upper)[0]) <= limit
        lower, upper = lower[kept], upper[kept]
        if not lower.shape[0]:
            raise ValueError("The level band is empty: the level is not attained.")
    value, gradient, _ = enclosure.centered(lower, upper)
    offsets = distance(value)
    lipschitz = _up(_matrix_norm_upper(inverse) * table.gradient_bound)
    norms = np.minimum(_cartesian_gradient_norm_upper(inverse, gradient), lipschitz)
    order = np.argsort(offsets, kind="stable")
    offsets, norms = offsets[order], np.maximum.accumulate(norms[order])
    if offsets[0] != 0.0:
        raise ValueError("The level band is empty: the level is not attained.")
    middle = 0.5 * lower + 0.5 * upper
    point_value, point_gradient, _ = enclosure.evaluate(middle, middle, 1)
    if point_gradient is None:
        raise RuntimeError("The table gradient enclosure was not evaluated.")
    inside = np.abs(0.5 * point_value[0] + 0.5 * point_value[1] - center) <= limit
    sampled = (0.5 * point_gradient[0] + 0.5 * point_gradient[1]) @ inverse.T
    sampled_norm = float(np.max(np.linalg.norm(sampled[inside], axis=1), initial=0.0))
    return TrigonometricSheetThickness(
        offsets=offsets,
        gradient_upper=norms,
        level=center,
        maximum_half_width=limit,
        resolution_level=resolution_level,
        sampled_gradient_norm=sampled_norm,
        box_count=lower.shape[0],
        table_id=table.table_id,
        cell_id=cell.cell_id,
    )


class TrigonometricSheetBand(StrictModule):
    """Sheet semantics: the level band ``{|F - c| <= half_width}`` as requested."""

    half_width: float = eqx.field(static=True)

    def __init__(self, half_width: float, /) -> None:
        value = float(half_width)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("half_width must be finite and positive.")
        self.half_width = value


class TrigonometricWallThickness(StrictModule):
    """Sheet semantics: the thinnest band whose certified wall meets ``minimum_thickness``."""

    minimum_thickness: float = eqx.field(static=True)
    resolution_level: int = eqx.field(static=True)

    def __init__(self, minimum_thickness: float, /, *, resolution_level: int = 6) -> None:
        value = float(minimum_thickness)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("minimum_thickness must be finite and positive.")
        if isinstance(resolution_level, bool) or not isinstance(resolution_level, int):
            raise TypeError("resolution_level must be an integer.")
        self.minimum_thickness = value
        self.resolution_level = resolution_level


def _boundary_levels(
    region: TrigonometricRegion, level: float, half_width: float | None, /
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    match region:
        case "network" | "complement":
            return (level,), ("level",)
        case "sheet":
            if half_width is None:
                raise ValueError("A sheet region needs a half width.")
            return (level - half_width, level + half_width), (
                "lower_level",
                "upper_level",
            )
        case _:
            assert_never(region)


class TrigonometricLevelSet(GeometrySource):
    """Region of a trigonometric nodal level set on a periodic affine lattice.

    ``F(x) = table(u)`` with reduced coordinates ``u = (x - origin) @
    cell.inverse_vectors``. Regions: ``network`` is ``{F <= c}``,
    ``complement`` is ``{F >= c}``, and ``sheet`` is ``{|F - c| <= tau}`` with
    ``tau`` from a ``TrigonometricSheetBand`` or solved from a
    ``TrigonometricWallThickness`` target. Each boundary level (``c``, or
    ``c - tau`` and ``c + tau`` for a sheet) is resolved on demand against the
    critical values (``TrigonometricCriticalLevels.separate``); one that is not
    separated raises ``TrigonometricSingularLevelError`` with the decided
    evidence. ``critical_levels`` is the certificate refined for these levels.
    """

    table: TrigonometricTable
    cell: PeriodicCell
    critical_levels: TrigonometricCriticalLevels
    sheet_thickness: TrigonometricSheetThickness | None
    region: TrigonometricRegion = eqx.field(static=True)
    level: float = eqx.field(static=True)
    half_width: float | None = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        table: TrigonometricTable,
        cell: PeriodicCell,
        /,
        *,
        level: float = 0.0,
        region: TrigonometricRegion = "network",
        sheet: TrigonometricSheetBand | TrigonometricWallThickness | None = None,
        critical_levels: TrigonometricCriticalLevels | None = None,
        feature_id: str | None = None,
    ) -> None:
        _require_cell(cell)
        selected = parse(region, TrigonometricRegion, "region")
        center = float(level)
        if not math.isfinite(center):
            raise ValueError("level must be finite.")
        certificate = (
            certify_trigonometric_critical_levels(table)
            if critical_levels is None
            else critical_levels
        )
        if certificate.table_id != table.table_id:
            raise ValueError("critical_levels certify a different table.")
        half_width, thickness = _sheet_half_width(selected, sheet, table, cell, center)
        for boundary in _boundary_levels(selected, center, half_width)[0]:
            separation = certificate.separate(boundary)
            if not separation.separated:
                raise TrigonometricSingularLevelError(separation)
            certificate = separation.critical_levels
        self.table = table
        self.cell = cell
        self.critical_levels = certificate
        self.sheet_thickness = thickness
        self.region = selected
        self.level = center
        self.half_width = half_width
        self.feature_id = _feature_id(feature_id, "trigonometric-level-set")

    @property
    def lipschitz_bound(self) -> float:
        """Upper bound of ``sup ||grad_x F||`` on the lattice."""
        inverse = np.asarray(self.cell.inverse_vectors, dtype=np.float64)
        return _up(_matrix_norm_upper(inverse) * self.table.gradient_bound)

    @property
    def hessian_bound(self) -> float:
        """Upper bound of ``sup ||Hess_x F||_2 = ||M Hess_u F M^T||_2``."""
        norm = _matrix_norm_upper(np.asarray(self.cell.inverse_vectors, dtype=np.float64))
        return _up(_up(norm * norm) * self.table.hessian_bound)

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        del context
        levels, names = _boundary_levels(self.region, self.level, self.half_width)
        lipschitz = self.lipschitz_bound
        match self.region:
            case "network" | "complement":
                field_lipschitz = lipschitz
            case "sheet":
                if self.half_width is None:
                    raise ValueError("A sheet region needs a half width.")
                spread = _up(self.table.amplitude_bound + abs(self.level))
                field_lipschitz = _up(_up(spread * lipschitz) / _down(self.half_width))
            case _:
                assert_never(self.region)
        return _TrigonometricLevelSetKernel(
            self.table,
            self.cell,
            region=self.region,
            level=self.level,
            half_width=self.half_width,
            lipschitz=field_lipschitz,
            margins=tuple(self.critical_levels.separation(value) for value in levels),
            margin_names=names,
            source_id=self.feature_id,
        )


def _sheet_half_width(
    region: TrigonometricRegion,
    sheet: TrigonometricSheetBand | TrigonometricWallThickness | None,
    table: TrigonometricTable,
    cell: PeriodicCell,
    level: float,
    /,
) -> tuple[float | None, TrigonometricSheetThickness | None]:
    match region:
        case "network" | "complement":
            if sheet is not None:
                raise ValueError("Only a sheet region takes sheet semantics.")
            return None, None
        case "sheet":
            if isinstance(sheet, TrigonometricSheetBand):
                return sheet.half_width, None
            if isinstance(sheet, TrigonometricWallThickness):
                inverse = _require_cell(cell)
                lipschitz = _up(_matrix_norm_upper(inverse) * table.gradient_bound)
                # The certified wall is at least 2 tau / lipschitz, so this band
                # brackets the target; beyond sup |F - c| the band is the cell.
                bracket = min(
                    _up(
                        _up(0.5 * sheet.minimum_thickness * lipschitz)
                        * (1.0 + 16.0 * _EPSILON)
                    ),
                    _up(table.amplitude_bound + abs(level)),
                )
                thickness = certify_trigonometric_sheet_thickness(
                    table,
                    cell,
                    level=level,
                    maximum_half_width=bracket,
                    resolution_level=sheet.resolution_level,
                )
                return thickness.half_width_for_thickness(
                    sheet.minimum_thickness
                ), thickness
            raise ValueError(
                "A sheet region needs TrigonometricSheetBand or TrigonometricWallThickness."
            )
        case _:
            assert_never(region)


class _TrigonometricLevelSetKernel(GeometryKernel):
    table: TrigonometricTable
    cell: PeriodicCell
    region: TrigonometricRegion = eqx.field(static=True)
    level: float = eqx.field(static=True)
    half_width: float | None = eqx.field(static=True)
    lipschitz: float = eqx.field(static=True)
    margins: tuple[float, ...] = eqx.field(static=True)
    margin_names: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        table: TrigonometricTable,
        cell: PeriodicCell,
        /,
        *,
        region: TrigonometricRegion,
        level: float,
        half_width: float | None,
        lipschitz: float,
        margins: tuple[float, ...],
        margin_names: tuple[str, ...],
        source_id: str,
    ) -> None:
        self.table = table
        self.cell = cell
        self.region = region
        self.level = level
        self.half_width = half_width
        self.lipschitz = lipschitz
        self.margins = margins
        self.margin_names = margin_names
        self.source_id = source_id

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 3

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return _CAPABILITIES

    @property
    def field_certificate(self) -> FieldCertificate:
        # A pointwise evaluation-error bound would grow with |x| through the
        # Cartesian-to-reduced map, so none is declared for the unbounded field.
        return FieldCertificate(
            zero_set_accuracy=ZeroSetAccuracy.EXACT,
            sign_reliability=SignReliability.RELIABLE,
            distance_semantics=DistanceSemantics.LEVEL_SET,
            regularity=FieldRegularity.SMOOTH,
            safe_step_factor=None,
            validity_region=(
                "nodal trigonometric level set of the declared table on the declared "
                "lattice; a nodal approximation, not an exact minimal surface"
            ),
            parameter_differentiable=False,
            provenance=("trigonometric_nodal_level_set", self.table.table_id),
            lipschitz_upper_bound=self.lipschitz,
        )

    def geometry_validity(self, state: DesignState, /) -> GeometryValidityEvidence:
        del state
        margins = jnp.asarray(self.margins, dtype=jnp.float64)
        return GeometryValidityEvidence(
            finite=jnp.all(jnp.isfinite(margins)),
            conditions_satisfied=jnp.all(margins > 0.0),
            resolved=True,
            margins=margins,
            margin_names=tuple(
                f"critical_separation_{name}" for name in self.margin_names
            ),
            contract_id="trigonometric_level_set",
        )

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        del state
        points_ = _check_points(points, 3)
        value = self.table.evaluate_reduced(self.cell.fractional(points_))
        match self.region:
            case "network":
                return value - self.level
            case "complement":
                return self.level - value
            case "sheet":
                if self.half_width is None:
                    raise ValueError("A sheet region needs a half width.")
                offset = value - self.level
                return (jnp.square(offset) - self.half_width**2) / (2.0 * self.half_width)
            case _:
                assert_never(self.region)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        points_ = _check_points(points, 3)

        def field(point: Array) -> Array:
            return self.boundary_field(state, point[None, :])[0]

        gradient = jax.vmap(jax.grad(field))(points_)
        norm = jnp.linalg.norm(gradient, axis=-1, keepdims=True)
        return gradient / jnp.maximum(norm, jnp.finfo(jnp.float64).tiny)

    def contact_curvature(
        self, state: DesignState, points: Array, /
    ) -> ContactCurvatureResult:
        """Principal curvatures of the level set from the exact field Hessian."""
        points_ = _check_points(points, 3)

        def field(point: Array) -> Array:
            return self.boundary_field(state, point[None, :])[0]

        gradient = jax.vmap(jax.grad(field))(points_)
        hessian = jax.vmap(jax.hessian(field))(points_)
        gradient_norm = jnp.linalg.norm(gradient, axis=-1)
        normal = gradient / jnp.maximum(
            gradient_norm[:, None], jnp.finfo(jnp.float64).tiny
        )
        identity = jnp.eye(3, dtype=jnp.float64)
        seed = identity[jnp.argmin(jnp.abs(normal), axis=-1)]
        first = jnp.cross(normal, seed)
        first = first / jnp.linalg.norm(first, axis=-1, keepdims=True)
        second = jnp.cross(normal, first)
        frame = jnp.stack((first, second), axis=-1)
        shape = jnp.swapaxes(frame, -1, -2) @ hessian @ frame
        shape = shape / jnp.maximum(
            gradient_norm[:, None, None], jnp.finfo(jnp.float64).tiny
        )
        principal = jnp.linalg.eigvalsh(shape)
        valid = jnp.all(jnp.isfinite(principal), axis=-1) & (gradient_norm > 0.0)
        return ContactCurvatureResult(
            principal, valid, gradient_norm, ambient_dimension=3
        )

    def bounds(self, state: DesignState, /) -> Array:
        del state
        return jnp.asarray(
            ((-jnp.inf, -jnp.inf, -jnp.inf), (jnp.inf, jnp.inf, jnp.inf)),
            dtype=jnp.float64,
        )

    def measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError(
            "A periodic lattice region is unbounded; enclose a finite domain's volume "
            "from an adaptive discovery cover."
        )

    def boundary_measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("A periodic lattice surface has unbounded area.")

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: PRNGKey,
        plan: RejectionSamplingPlan | None = None,
    ) -> NoReturn:
        del state, num_points, key, plan
        raise NotImplementedError("An unbounded periodic region has no interior sampler.")

    def sample_boundary(
        self, state: DesignState, num_points: int, /, *, key: PRNGKey
    ) -> NoReturn:
        del state, num_points, key
        raise NotImplementedError(
            "An unbounded periodic surface has no boundary sampler."
        )

    def boundary_atlas(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("A periodic lattice surface has no boundary atlas.")


__all__ = [
    "TrigonometricCriticalLevels",
    "TrigonometricFactor",
    "TrigonometricFamily",
    "TrigonometricLevelSet",
    "TrigonometricRegion",
    "TrigonometricSheetBand",
    "TrigonometricSheetThickness",
    "TrigonometricSingularLevelError",
    "TrigonometricTable",
    "TrigonometricTerm",
    "TrigonometricWallThickness",
    "certify_trigonometric_critical_levels",
    "certify_trigonometric_sheet_thickness",
]
