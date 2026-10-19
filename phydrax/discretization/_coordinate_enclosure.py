#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only source-level polynomial bounds with rational/algebraic scalar coefficients.

No basis tabulation, floating interpolation inverse, or partition-of-unity test
is used as an error bound. Pyramids use the collapsed cube; its top face is one
physical apex, and the physical determinant divides out the rational collapse.
"""

from __future__ import annotations

import itertools as imported_itertools
import math
import operator
from collections.abc import ItemsView, Iterable, Iterator, Mapping, Sequence, ValuesView
from contextlib import AbstractContextManager
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, partial
from itertools import product
from types import MappingProxyType
from typing import (
    assert_never,
    Callable,
    Literal,
    overload,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
    TypeAlias,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from typing_extensions import TypeIs

import phydrax.discretization.fem._reference_types as reference_types_owner
from phydrax.discretization.fem import (
    _hybrid_reference as hybrid_reference_owner,
    _jacobi as jacobi_owner,
)
from phydrax.discretization.fem._reference_types import FiniteElementSpec

from .. import _fingerprint as imported_fingerprint
from .._host_storage import (
    declared_integer_storage_bytes,
    declared_object_storage_bytes,
)
from .._host_table_cache import host_table_cache
from .._validation import nonnegative_integer
from ..geometry._rational import rational_vector
from ..linalg import _small_batched as imported_linalg__small_batched
from ..linalg._exact import solve_exact
from ..typing import checked, parse
from . import (
    _reference_cell as imported_reference_cell,
)
from ._coordinate_budget import (
    CoordinateEnclosureBudget,
    RationalEnclosureError,
    reserve_polynomial,
)
from ._coordinate_identity import coordinate_source_bank_key
from ._coordinate_source_snapshot import BarycentricSourceSnapshot, SplineSourceSnapshot
from ._exact_algebraic_geometry import (
    coefficient_endpoint,
    RadicalElement,
    rational_coefficient,
    reserve_algebraic_work,
)
from ._reference_cell_types import CellGeometryElement
from .fem import (
    _form_elements as imported_fem__form_elements,
    _form_expressions as imported_fem__form_expressions,
    _high_order as imported_fem__high_order,
    _reference as imported_fem__reference,
)
from .fem._reference import lagrange_element


if TYPE_CHECKING:
    from . import _cell_geometry as cell_geometry_types
    from ._cell_geometry import (
        CellGeometrySpec,
        LayerColumnCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        ScalarCoordinateElement,
        SplineCellGeometryElement,
    )
    from .fem._high_order import SimplexNodalFamily


class _CellGeometryDispatch(Protocol):
    @property
    def BarycentricCellGeometryElement(
        self,
    ) -> type[cell_geometry_types.BarycentricCellGeometryElement]: ...
    @property
    def CoordinateTabulator(self) -> type[cell_geometry_types.CoordinateTabulator]: ...
    @property
    def LayerColumnCellGeometryElement(
        self,
    ) -> type[cell_geometry_types.LayerColumnCellGeometryElement]: ...
    @property
    def PolynomialComposedCellGeometryElement(
        self,
    ) -> type[cell_geometry_types.PolynomialComposedCellGeometryElement]: ...
    @property
    def RationalComposedCellGeometryElement(
        self,
    ) -> type[cell_geometry_types.RationalComposedCellGeometryElement]: ...
    @property
    def RestrictedCellGeometryElement(
        self,
    ) -> type[cell_geometry_types.RestrictedCellGeometryElement]: ...
    @property
    def SplineCellGeometryElement(
        self,
    ) -> type[cell_geometry_types.SplineCellGeometryElement]: ...
    @property
    def SweptCoordinateTabulator(
        self,
    ) -> type[cell_geometry_types.SweptCoordinateTabulator]: ...

    @staticmethod
    def require_p1_cardinal_source(
        element: CellGeometryElement, /
    ) -> FiniteElementSpec: ...

    @staticmethod
    def require_scalar_coordinate_element(
        element: CellGeometryElement, role: str, /
    ) -> ScalarCoordinateElement: ...

    @staticmethod
    def coordinate_source_snapshot_preparation_bounds(
        element: CellGeometryElement, /
    ) -> tuple[int, int]: ...

    @staticmethod
    def prepare_barycentric_source_snapshot(
        element: cell_geometry_types.BarycentricCellGeometryElement,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> BarycentricSourceSnapshot: ...

    @staticmethod
    def prepare_spline_source_snapshot(
        element: cell_geometry_types.SplineCellGeometryElement,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> SplineSourceSnapshot: ...

    @staticmethod
    def owned_coordinate_preparation(
        maximum_work_units: int,
        maximum_memory_bytes: int,
        /,
        *,
        maximum_geometry_queries: int = 0,
    ) -> AbstractContextManager[CoordinateEnclosureBudget]: ...


@cache
def _cell_geometry_dispatch() -> _CellGeometryDispatch:
    # Deferred once: enclosure → cell geometry → periodic geometry → enclosure.
    from . import _cell_geometry

    return _cell_geometry


type ExactScalar = Fraction | RadicalElement
type Polynomial = dict[tuple[int, ...], ExactScalar]
type PreparedPolynomialCoefficients = tuple[tuple[tuple[int, ...], ExactScalar], ...]
type PreparedReferenceDenominators = tuple[tuple[Polynomial, ...], tuple[int, ...]]
type CoordinateSourceBank = tuple[tuple[ExactScalar, ...], ...]
type CoordinatePreparationKey = (
    tuple[str, str, CoordinateSourceBank] | tuple[str, str, CoordinateSourceBank, bytes]
)
type CoordinatePointKey = tuple[ExactScalar, ...] | tuple[tuple[ExactScalar, ...], bytes]
type CoordinateLinearKey = (
    tuple[tuple[str, str], CoordinateSourceBank, tuple[tuple[int, ExactScalar], ...]]
    | tuple[
        tuple[str, str],
        CoordinateSourceBank,
        tuple[tuple[int, ExactScalar], ...],
        bytes,
        bytes,
    ]
)
type CoordinateCoefficients = np.ndarray | CoordinateSourceBank
type CoordinateCornerWeights = tuple[tuple[tuple[int, ExactScalar], ...], ...]
type OrderedReferenceArguments = tuple[PreparedPolynomialCoefficients, ...]
BernsteinDomain: TypeAlias = Literal["simplex", "prism", "box"]
"""Reference domain of a Bernstein control net: simplex, triangle x interval, or unit box."""

type PolynomialRepresentationKey = (
    PreparedPolynomialCoefficients | tuple[PreparedPolynomialCoefficients, bytes]
)
type BernsteinCoefficientKey = (
    tuple[BernsteinDomain, int, PreparedPolynomialCoefficients]
    | tuple[BernsteinDomain, int, PreparedPolynomialCoefficients, bytes]
)


class RationalPolynomial:
    """Rational quotient of source expressions; no interpolation or sampled fit."""

    def __init__(self, numerator: Polynomial, denominator: Polynomial, /) -> None:
        if not denominator:
            raise ValueError(
                "RationalPolynomial.denominator must be nonzero; got zero support."
            )
        self.numerator = numerator
        self.denominator = denominator

    def __bool__(self) -> bool:
        return bool(self.numerator)

    def equals(self, other: object, /, *, budget: CoordinateEnclosureBudget) -> bool:
        """Compare rational quotients, counting both cross products on the explicit budget."""
        if not isinstance(other, RationalPolynomial):
            return False
        return multiply(self.numerator, other.denominator, budget=budget) == multiply(
            other.numerator, self.denominator, budget=budget
        )

    def __eq__(self, other: object) -> bool:
        raise TypeError(
            "RationalPolynomial.comparison must use equals(other, budget=budget); got implicit equality."
        )

    __hash__ = None


type Expression = Polynomial | RationalPolynomial


def expression_equal(
    first: Expression, second: Expression, /, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Compare source expressions with explicit quotient work accounting."""
    if isinstance(first, RationalPolynomial):
        return first.equals(second, budget=budget)
    if isinstance(second, RationalPolynomial):
        return False
    return first == second


def expression_sequence_equal(
    first: Sequence[Expression],
    second: Sequence[Expression],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Compare source banks with sequence identity shortcuts and ordered rational work."""
    if first is second:
        return True
    for a, b in zip(first, second):
        if a is not b and not expression_equal(a, b, budget=budget):
            return False
    return len(first) == len(second)


@dataclass(frozen=True, slots=True)
class PreparedCoordinateCells:
    """Retained source cell records earned by one full mapped preparation."""

    cell_ids: tuple[int, ...]
    descriptors: tuple[
        tuple[str, CellGeometryElement, CoordinateSourceBank, tuple[int, ...]], ...
    ]
    coordinates: tuple[tuple[Expression, ...], ...]


_BINARY64_MAXIMUM = Fraction(float(np.finfo(np.float64).max))


def _scalar_kind_key(values: Iterable[ExactScalar], /) -> bytes:
    """Tag radical-containing payloads; rational payloads keep their existing keys."""
    kinds = bytes(isinstance(value, RadicalElement) for value in values)
    return kinds if any(kinds) else b""


def _polynomial_representation_key(
    polynomial: Mapping[tuple[int, ...], ExactScalar], /
) -> PolynomialRepresentationKey:
    entries = tuple(polynomial.items())
    kinds = _scalar_kind_key(polynomial.values())
    return (entries, kinds) if kinds else entries


def _coordinate_linear_key(
    source_identity: tuple[str, str],
    coefficients: CoordinateSourceBank,
    row: tuple[tuple[int, ExactScalar], ...],
    bank_kinds: bytes,
    /,
) -> CoordinateLinearKey:
    row_kinds = _scalar_kind_key(value for _, value in row)
    if bank_kinds or row_kinds:
        return source_identity, coefficients, row, bank_kinds, row_kinds
    return source_identity, coefficients, row


def _scalar_profile(value: ExactScalar, /) -> tuple[int, int]:
    """Profile constituent rational coefficients and radicands, never a surrogate."""
    if isinstance(value, Fraction):
        return abs(value.numerator).bit_length(), value.denominator
    return (
        max(
            1,
            sum(
                abs(coefficient.numerator).bit_length() + radicand.bit_length() + 1
                for radicand, coefficient in value.terms
            ),
        ),
        math.lcm(*(coefficient.denominator for _, coefficient in value.terms)),
    )


def _scalar_bits(value: ExactScalar | int, /) -> int:
    if isinstance(value, int):
        return abs(value).bit_length()
    numerator, denominator = _scalar_profile(value)
    return numerator + denominator.bit_length()


def _scalar_hash_storage(value: ExactScalar, /) -> int:
    if isinstance(value, Fraction):
        return declared_object_storage_bytes(
            value.numerator
        ) + declared_object_storage_bytes(value.denominator)
    return 512 + sum(
        512 + declared_object_storage_bytes(radicand) + _scalar_hash_storage(coefficient)
        for radicand, coefficient in value.terms
    )


def _all_rational(
    polynomial: Mapping[tuple[int, ...], ExactScalar], /
) -> TypeIs[Mapping[tuple[int, ...], Fraction]]:
    return all(isinstance(value, Fraction) for value in polynomial.values())


def _all_rational_values(
    values: tuple[ExactScalar, ...], /
) -> TypeIs[tuple[Fraction, ...]]:
    return all(isinstance(value, Fraction) for value in values)


def _scalar_endpoint(
    value: ExactScalar, /, *, upper: bool, budget: CoordinateEnclosureBudget
) -> float:
    return coefficient_endpoint(value, upper=upper, saturate=True, budget=budget)


def require_rational_reference_scalar(value: ExactScalar, /) -> Fraction:
    """Narrow a declared rational reference chart without rounding its coefficients."""
    return rational_coefficient(value, role="coordinate_enclosure.reference_scalar")


def _reserve_coefficient_work(
    polynomials: tuple[Mapping[tuple[int, ...], ExactScalar], ...] = (),
    scalars: tuple[ExactScalar | int, ...] = (),
    banks: tuple[CoordinateSourceBank, ...] = (),
    *,
    work: int,
    count: int,
    degree: int = 1,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Admit radical constituent arithmetic without changing rational ledger entries."""
    if (
        all(_all_rational(polynomial) for polynomial in polynomials)
        and not any(isinstance(value, RadicalElement) for value in scalars)
        and not any(
            isinstance(value, RadicalElement)
            for bank in banks
            for row in bank
            for value in row
        )
    ):
        return
    reserve_algebraic_work(
        imported_itertools.chain(
            (value for polynomial in polynomials for value in polynomial.values()),
            (
                value
                if isinstance(value, (Fraction, RadicalElement))
                else Fraction(value)
                for value in scalars
            ),
            (value for bank in banks for row in bank for value in row),
        ),
        work,
        count,
        degree,
        budget=budget,
    )


def _scalar_quotient(
    numerator: ExactScalar,
    denominator: ExactScalar,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactScalar:
    if isinstance(numerator, Fraction) and isinstance(denominator, Fraction):
        return numerator / denominator
    _reserve_coefficient_work(
        scalars=(numerator, denominator),
        work=1,
        count=1,
        degree=2,
        budget=budget,
    )
    return numerator / denominator


@dataclass(frozen=True, slots=True, eq=False, init=False)
class PreparedPolynomialSupport(Mapping[tuple[int, ...], ExactScalar]):
    """Immutable coefficient support with a profile earned from every retained coefficient."""

    _coefficients: Mapping[tuple[int, ...], ExactScalar]
    numerator_bits: int
    denominator: int

    def __init__(
        self,
        coefficients: Mapping[tuple[int, ...], ExactScalar],
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        with budget.temporary_scope():
            budget.reserve(len(coefficients), 96 + 64 * len(coefficients))
            snapshot: Polynomial = {}
            numerator, denominator = 1, 1
            for index, coefficient in coefficients.items():
                snapshot[index] = coefficient
                coefficient_numerator, coefficient_denominator = _scalar_profile(
                    coefficient
                )
                numerator = max(numerator, coefficient_numerator)
                denominator = math.lcm(denominator, coefficient_denominator)
            view = MappingProxyType(snapshot)
            object.__setattr__(self, "_coefficients", view)
            object.__setattr__(self, "numerator_bits", numerator)
            object.__setattr__(self, "denominator", denominator)
            budget.retain_basis((self, snapshot, view, numerator, denominator))

    def __getitem__(self, index: tuple[int, ...]) -> ExactScalar:
        return self._coefficients[index]

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return iter(self._coefficients)

    def __len__(self) -> int:
        return len(self._coefficients)

    def items(self) -> ItemsView[tuple[int, ...], ExactScalar]:
        return self._coefficients.items()

    def values(self) -> ValuesView[ExactScalar]:
        return self._coefficients.values()


def coefficient_profile(
    polynomials: tuple[Mapping[tuple[int, ...], ExactScalar], ...],
    /,
    *,
    exact_denominator: bool = False,
    budget: CoordinateEnclosureBudget,
) -> tuple[int, int]:
    budget.reserve(
        sum(
            1 if isinstance(polynomial, PreparedPolynomialSupport) else len(polynomial)
            for polynomial in polynomials
        )
    )
    numerator_bits, denominator = 1, 1
    for polynomial in polynomials:
        if isinstance(polynomial, PreparedPolynomialSupport):
            numerator_bits = max(numerator_bits, polynomial.numerator_bits)
            denominator = math.lcm(denominator, polynomial.denominator)
            continue
        for value in polynomial.values():
            value_numerator, value_denominator = _scalar_profile(value)
            numerator_bits = max(numerator_bits, value_numerator)
            denominator = math.lcm(denominator, value_denominator)
    return numerator_bits, denominator if exact_denominator else denominator.bit_length()


@runtime_checkable
class _BoundSourceTabulator(Protocol):
    @property
    def __self__(self) -> object: ...

    @property
    def __func__(self) -> Callable[..., tuple[ArrayLike, ArrayLike]]: ...


def constant(
    value: ExactScalar | int, dimension: int, *, budget: CoordinateEnclosureBudget
) -> Polynomial:
    if (
        type(dimension) is int
        and (type(value) is int or type(value) is Fraction)
        and not value
    ):
        return {}
    bits = (
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        if isinstance(value, Fraction)
        else _scalar_bits(value)
    )
    reserve_polynomial(1, int(bool(value)), dimension, bits, budget=budget)
    coefficient = value if isinstance(value, RadicalElement) else Fraction(value)
    return {(0,) * dimension: coefficient} if coefficient else {}


def _copy_disjoint_terms(
    target: Polynomial, second: Polynomial, /, *, budget: CoordinateEnclosureBudget
) -> None:
    """Append immutable coefficients after their disjoint support was established."""
    budget.reserve(len(second), 256 + 64 * (len(target) + len(second)))
    for index, value in second.items():
        if value:
            target[index] = value


def _accumulate(
    target: Polynomial,
    second: Polynomial,
    /,
    *,
    target_profile: tuple[int, int] | None = None,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Fuse the support probe with its earned coefficient profile."""
    if not second:
        return
    if not target:
        _copy_disjoint_terms(target, second, budget=budget)
        return
    numerator = denominator = 1
    with budget.temporary_scope():
        budget.reserve(len(second), 384)
        disjoint = True
        second_numerator = second_common = 1
        for index, value in second.items():
            disjoint = disjoint and index not in target
            value_numerator, value_denominator = _scalar_profile(value)
            second_numerator = max(second_numerator, value_numerator)
            second_common = math.lcm(second_common, value_denominator)
        budget.reserve(
            0,
            max(
                0,
                declared_object_storage_bytes(second_common)
                - declared_object_storage_bytes(1),
            ),
        )
        if not disjoint:
            if target_profile is None:
                target_numerator, target_common = coefficient_profile(
                    (target,), exact_denominator=True, budget=budget
                )
            else:
                budget.reserve(1)
                target_numerator, target_common = target_profile
            budget.reserve(
                0,
                96
                + 2
                * (
                    declared_object_storage_bytes(target_common)
                    + declared_object_storage_bytes(second_common)
                ),
            )
            numerator = max(target_numerator, second_numerator)
            denominator = math.lcm(target_common, second_common).bit_length()
            del target_common
    del second_common
    if disjoint:
        _copy_disjoint_terms(target, second, budget=budget)
        return
    dimension = len(next(iter(target)))
    reserve_polynomial(
        len(target) + len(second),
        len(target) + len(second),
        dimension,
        numerator + denominator + 1,
        budget=budget,
    )
    _reserve_coefficient_work(
        (target, second), work=len(second), count=len(target) + len(second), budget=budget
    )
    for index, value in second.items():
        previous = target.get(index)
        total = value if previous is None else previous + value
        if total:
            target[index] = total
        else:
            target.pop(index, None)


def add(
    first: Polynomial, second: Polynomial, *, budget: CoordinateEnclosureBudget
) -> Polynomial:
    if not second:
        return scale(first, 1, budget=budget)
    result = dict(first)
    _accumulate(result, second, budget=budget)
    return result


def scale(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    value: ExactScalar | int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    if not value or not polynomial:
        return {}
    if value == 1:
        # Unit scaling allocates a dictionary, not new rational coefficients.
        budget.reserve(len(polynomial), 256 + 64 * len(polynomial))
        return dict(polynomial)
    numerator, denominator = coefficient_profile((polynomial,), budget=budget)
    value_bits = _scalar_bits(value)
    bits = numerator + denominator + value_bits
    reserve_polynomial(
        2 * len(polynomial),
        len(polynomial),
        len(next(iter(polynomial), ())),
        bits,
        budget=budget,
    )
    _reserve_coefficient_work(
        (polynomial,),
        (value,),
        work=len(polynomial),
        count=len(polynomial),
        degree=2,
        budget=budget,
    )
    return {
        index: scaled_value
        for index, coefficient in polynomial.items()
        if (scaled_value := coefficient * value)
    }


def _constant_factor_product(
    first: Mapping[tuple[int, ...], ExactScalar],
    second: Mapping[tuple[int, ...], ExactScalar],
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial | None:
    """Reduce a scalar support to scaling only when both parameter ranks agree."""
    if len(first) != 1 and len(second) != 1:
        return None
    dimension = len(next(iter(first)))
    zero = (0,) * dimension
    if not (
        (len(first) == 1 and zero in first) or (len(second) == 1 and zero in second)
    ) or any(
        len(index) != dimension for index in imported_itertools.chain(first, second)
    ):
        return None
    return (
        scale(second, first[zero], budget=budget)
        if len(first) == 1 and zero in first
        else scale(first, second[zero], budget=budget)
    )


def _multiply_with_first_profile(
    first: Mapping[tuple[int, ...], ExactScalar],
    second: Mapping[tuple[int, ...], ExactScalar],
    first_profile: tuple[int, int],
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    """Multiply after the unchanged first operand earned its coefficient profile."""
    if not first or not second:
        return {}
    scalar_product = _constant_factor_product(first, second, budget=budget)
    if scalar_product is not None:
        return scalar_product
    second_profile = (
        first_profile
        if first is second
        else coefficient_profile((second,), exact_denominator=True, budget=budget)
    )
    return _multiply_profiled(first, second, first_profile, second_profile, budget=budget)


def multiply(
    first: Mapping[tuple[int, ...], ExactScalar],
    second: Mapping[tuple[int, ...], ExactScalar],
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    # An empty rational support is the zero polynomial. No coefficient profile,
    # dimension traversal or product storage is needed for its annihilator.
    if not first or not second:
        return {}
    scalar_product = _constant_factor_product(first, second, budget=budget)
    if scalar_product is not None:
        return scalar_product
    square = first is second
    first_profile = coefficient_profile((first,), exact_denominator=True, budget=budget)
    second_profile = (
        first_profile
        if square
        else coefficient_profile((second,), exact_denominator=True, budget=budget)
    )
    return _multiply_profiled(first, second, first_profile, second_profile, budget=budget)


def _multiply_profiled(
    first: Mapping[tuple[int, ...], ExactScalar],
    second: Mapping[tuple[int, ...], ExactScalar],
    first_profile: tuple[int, int],
    second_profile: tuple[int, int],
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    """Expand nonconstant factors using the profiles earned by their caller."""
    square = first is second
    first_num, first_common = first_profile
    second_num, second_common = second_profile
    first_den, second_den = first_common.bit_length(), second_common.bit_length()
    full_pairs = len(first) * len(second)
    pairs = len(first) * (len(first) + 1) // 2 if square else full_pairs
    bits = (
        first_num + second_num + first_den + second_den + max(full_pairs, 1).bit_length()
    )
    dimension = len(next(iter(first or second), ()))
    budget.reserve(dimension * (len(first) + len(second)))
    reachable = math.prod(
        max((index[axis] for index in first), default=0)
        + max((index[axis] for index in second), default=0)
        + 1
        for axis in range(dimension)
    )
    reserve_polynomial(pairs, min(full_pairs, reachable), dimension, bits, budget=budget)
    dimension = len(next(iter(first)))
    if any(len(index) != dimension for index in first) or any(
        len(index) != dimension for index in second
    ):
        raise ValueError(
            "coordinate_enclosure.factors must share one parameter dimension; got unequal exponent ranks."
        )
    if not _all_rational(first) or not _all_rational(second):
        _reserve_coefficient_work(
            (first, second),
            work=pairs,
            count=min(full_pairs, reachable),
            degree=2,
            budget=budget,
        )
        totals_exact: Polynomial = {}
        left_exact = tuple(first.items())
        right_exact = left_exact if square else tuple(second.items())
        for row, (a, x) in enumerate(left_exact):
            start = row if square else 0
            for column in range(start, len(right_exact)):
                b, y = right_exact[column]
                index = tuple(map(operator.add, a, b))
                coefficient = x * y
                if square and row != column:
                    coefficient *= 2
                totals_exact[index] = totals_exact.get(index, Fraction(0)) + coefficient
        return {index: total for index, total in totals_exact.items() if total}
    # Rational products are accumulated as integers over the product of the two
    # common denominators; insertion order and normalized values are unchanged.
    left = tuple(
        (a, x.numerator * (first_common // x.denominator)) for a, x in first.items()
    )
    right = (
        left
        if square
        else tuple(
            (b, y.numerator * (second_common // y.denominator)) for b, y in second.items()
        )
    )
    totals: dict[tuple[int, ...], int] = {}
    if square:
        for row, (a, x) in enumerate(left):
            for column in range(row, len(left)):
                b, y = left[column]
                index = tuple(map(operator.add, a, b))
                coefficient = x * y
                totals[index] = totals.get(index, 0) + (
                    coefficient if row == column else 2 * coefficient
                )
    else:
        for a, x in left:
            for b, y in right:
                index = tuple(map(operator.add, a, b))
                totals[index] = totals.get(index, 0) + x * y
    common = first_common * second_common
    return {index: Fraction(total, common) for index, total in totals.items() if total}


def power(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    exponent: int,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    if exponent <= 0:
        return constant(1, dimension, budget=budget)
    if not polynomial:
        return {}
    if exponent == 1:
        budget.reserve(len(polynomial), 256 + 64 * len(polynomial))
        seed: Polynomial = {}
        for index, coefficient in polynomial.items():
            if len(index) != dimension:
                raise ValueError(
                    "coordinate_enclosure.factors must share one parameter dimension; got unequal exponent ranks."
                )
            seed[index] = coefficient
        return seed
    budget.reserve(len(polynomial))
    numerator, denominator = 1, 1
    for index, coefficient in polynomial.items():
        if len(index) != dimension:
            raise ValueError(
                "coordinate_enclosure.factors must share one parameter dimension; got unequal exponent ranks."
            )
        coefficient_numerator, coefficient_denominator = _scalar_profile(coefficient)
        numerator = max(numerator, coefficient_numerator)
        denominator = math.lcm(denominator, coefficient_denominator)
    source_profile = (numerator, denominator)
    result = _constant_factor_product(polynomial, polynomial, budget=budget)
    if result is None:
        result = _multiply_profiled(
            polynomial,
            polynomial,
            source_profile,
            source_profile,
            budget=budget,
        )
    for _ in range(2, exponent):
        if not result:
            return {}
        scaled = _constant_factor_product(result, polynomial, budget=budget)
        if scaled is not None:
            result = scaled
            continue
        result_profile = coefficient_profile(
            (result,), exact_denominator=True, budget=budget
        )
        result = _multiply_profiled(
            result, polynomial, result_profile, source_profile, budget=budget
        )
    return result


def axes(dimension: int, *, budget: CoordinateEnclosureBudget) -> tuple[Polynomial, ...]:
    reserve_polynomial(dimension, dimension, dimension, 1, budget=budget)
    return tuple(
        {tuple(int(i == axis) for i in range(dimension)): Fraction(1)}
        for axis in range(dimension)
    )


def derivative(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    axis: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    profile = coefficient_profile((polynomial,), exact_denominator=True, budget=budget)
    return _derivative_profiled(polynomial, axis, profile, budget=budget)


def _derivative_profiled(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    axis: int,
    profile: tuple[int, int],
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    """Differentiate after the unchanged support earned its coefficient profile."""
    numerator, common_denominator = profile
    denominator = common_denominator.bit_length()
    degree = max((index[axis] for index in polynomial), default=0)
    reserve_polynomial(
        len(polynomial),
        len(polynomial),
        len(next(iter(polynomial), ())),
        numerator + denominator + max(degree, 1).bit_length(),
        budget=budget,
    )
    _reserve_coefficient_work(
        (polynomial,), work=len(polynomial), count=len(polynomial), budget=budget
    )
    return {
        tuple(value - int(i == axis) for i, value in enumerate(index)): coefficient
        * index[axis]
        for index, coefficient in polynomial.items()
        if index[axis]
    }


def _axis_affine_arguments(
    arguments: tuple[Mapping[tuple[int, ...], ExactScalar], ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Fraction, Fraction, bool], ...] | None:
    if len(arguments) != dimension:
        return None
    budget.reserve(sum(len(argument) for argument in arguments))
    zero = (0,) * dimension
    result = []
    for axis, argument in enumerate(arguments):
        if not _all_rational(argument):
            return None
        unit = tuple(int(index == axis) for index in range(dimension))
        if any(index not in (zero, unit) and value for index, value in argument.items()):
            return None
        keys = tuple(index for index, value in argument.items() if value)
        result.append(
            (
                argument.get(zero, Fraction(0)),
                argument.get(unit, Fraction(0)),
                bool(keys and keys[0] == unit),
            )
        )
    return tuple(result)


def _affine_integer_factors(
    arguments: tuple[tuple[Fraction, Fraction, bool], ...],
    degrees: tuple[int, ...],
) -> list[list[tuple[tuple[int, int], ...]]]:
    """Prepare ordered binomial factors over the admitted common denominator."""
    factors = []
    for (offset, factor, descending), maximum in zip(arguments, degrees, strict=True):
        table = []
        for exponent in range(maximum + 1):
            orders = (
                (range(exponent, -1, -1) if descending else range(exponent + 1))
                if offset
                else (exponent,)
            )
            table.append(
                tuple(
                    (
                        order,
                        math.comb(exponent, order)
                        * offset.numerator ** (exponent - order)
                        * offset.denominator ** (maximum - exponent + order)
                        * factor.numerator**order
                        * factor.denominator ** (maximum - order),
                    )
                    for order in orders
                )
            )
        factors.append(table)
    return factors


def _axis_affine_polynomial(
    polynomial: Mapping[tuple[int, ...], Fraction],
    arguments: tuple[tuple[Fraction, Fraction, bool], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    """Rational separable affine composition, retaining the source accumulator order."""
    dimension = len(arguments)
    budget.reserve(len(polynomial))
    if any(len(index) != dimension for index in polynomial):
        raise ValueError(
            "coordinate_enclosure.arguments must match the polynomial parameter dimension; got unequal exponent ranks."
        )
    if not polynomial:
        return {}
    if all(not offset and factor == 1 for offset, factor, _ in arguments):
        budget.reserve(len(polynomial), 256 + 64 * len(polynomial))
        return {index: value for index, value in polynomial.items() if value}
    cache_key = _axis_affine_polynomial_key(polynomial, arguments, budget=budget)
    cached = budget.axis_affine_polynomial_cache.get(cache_key)
    if cached is not None:
        budget.reserve(len(cached), 256 + 64 * len(cached))
        return dict(cached)
    degrees = tuple(max(index[axis] for index in polynomial) for axis in range(dimension))
    coefficient_common = math.lcm(*(value.denominator for value in polynomial.values()))
    denominator = coefficient_common * math.prod(
        (offset.denominator * factor.denominator) ** degree
        for (offset, factor, _), degree in zip(arguments, degrees, strict=True)
    )
    contributions = sum(
        math.prod(
            exponent + 1 if offset else 1
            for exponent, (offset, _, _) in zip(index, arguments, strict=True)
        )
        for index, value in polynomial.items()
        if value
    )
    factor_terms = sum(
        (degree + 1) * (degree + 2) // 2 if offset else degree + 1
        for (offset, _, _), degree in zip(arguments, degrees, strict=True)
    )
    bits = (
        max(abs(value.numerator).bit_length() for value in polynomial.values())
        + coefficient_common.bit_length()
        + sum(
            degree
            * (
                abs(offset.numerator).bit_length()
                + offset.denominator.bit_length()
                + abs(factor.numerator).bit_length()
                + factor.denominator.bit_length()
                + 2
            )
            for (offset, factor, _), degree in zip(arguments, degrees, strict=True)
        )
        + max(len(polynomial), 1).bit_length()
        + 2
    )
    reserve_polynomial(
        contributions * (dimension + 1) + factor_terms,
        math.prod(degree + 1 for degree in degrees) + factor_terms,
        dimension,
        bits,
        budget=budget,
    )
    factors = _affine_integer_factors(arguments, degrees)
    totals: dict[tuple[int, ...], int] = {}
    for index, value in polynomial.items():
        if not value:
            continue
        scalar = value.numerator * (coefficient_common // value.denominator)
        for terms in product(
            *(table[exponent] for table, exponent in zip(factors, index, strict=True))
        ):
            key = tuple(term[0] for term in terms)
            coefficient = scalar * math.prod(term[1] for term in terms)
            updated = totals.get(key, 0) + coefficient
            if updated:
                totals[key] = updated
            else:
                totals.pop(key, None)
    result: Polynomial = {
        index: Fraction(value, denominator) for index, value in totals.items()
    }
    budget.reserve(len(result), 512 + 320 * len(result))
    snapshot = tuple(result.items())
    budget.retain_polynomial_preparation((cache_key, snapshot))
    budget.axis_affine_polynomial_cache[cache_key] = snapshot
    return result


def _axis_affine_polynomial_key(
    polynomial: Mapping[tuple[int, ...], Fraction],
    arguments: tuple[tuple[Fraction, Fraction, bool], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> str:
    """Identify ordered rational source and affine chart coefficients."""
    budget.reserve(
        len(polynomial) + 2 * len(arguments),
        8192
        + 256 * (len(polynomial) + len(arguments)) * (len(arguments) + 1)
        + 4
        * sum(
            declared_object_storage_bytes(value.numerator)
            + declared_object_storage_bytes(value.denominator)
            for value in imported_itertools.chain(
                polynomial.values(),
                (value for offset, factor, _ in arguments for value in (offset, factor)),
            )
        ),
    )
    return imported_fingerprint.canonical_fingerprint(
        {
            "kind": "axis-affine-polynomial",
            "coefficients": tuple(
                (index, value.numerator, value.denominator)
                for index, value in polynomial.items()
            ),
            "arguments": tuple(
                (
                    offset.numerator,
                    offset.denominator,
                    factor.numerator,
                    factor.denominator,
                    descending,
                )
                for offset, factor, descending in arguments
            ),
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class PreparedPolynomialArguments:
    """Full immutable reference support and its earned classification."""

    arguments: tuple[Mapping[tuple[int, ...], ExactScalar], ...]
    ordered_key: OrderedReferenceArguments
    dimension: int
    axis_affine: tuple[tuple[Fraction, Fraction, bool], ...] | None


class ArgumentComposition:
    """Composition with fixed polynomial arguments.

    Argument powers and their prefix products are formed once and reused by
    every polynomial composed with the same arguments. ``c * (P0^a P1^b ...)``
    has the same rational coefficients and key order as ``((c * P0^a) * P1^b) ...``
    because a nonzero scalar creates no zeros.
    """

    def __init__(
        self,
        arguments: tuple[Mapping[tuple[int, ...], ExactScalar], ...]
        | PreparedPolynomialArguments,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        if isinstance(arguments, PreparedPolynomialArguments):
            self.arguments = arguments.arguments
            self.dimension = arguments.dimension
            self._axis_affine = arguments.axis_affine
            self._reference_key: OrderedReferenceArguments | None = arguments.ordered_key
        else:
            self.arguments = arguments
            self.dimension = next(
                (len(index) for argument in arguments for index in argument),
                len(arguments),
            )
            self._axis_affine = _axis_affine_arguments(
                arguments, self.dimension, budget=budget
            )
            self._reference_key = None
            if self._axis_affine is None:
                terms = sum(len(argument) for argument in arguments)
                # Dynamic caller banks still earn their current full snapshot.
                budget.reserve(terms, 64 + 64 * len(arguments) + 64 * terms)
                self._reference_key = tuple(
                    tuple(argument.items()) for argument in arguments
                )
        self._scalar_kinds = _scalar_kind_key(
            value for argument in self.arguments for value in argument.values()
        )
        self._powers: tuple[dict[int, Polynomial], ...] = tuple(
            {} for _ in self.arguments
        )
        self._products: dict[tuple[int, ...], Mapping[tuple[int, ...], ExactScalar]] = {
            (): constant(1, self.dimension, budget=budget)
        }

    def _monomial(
        self, index: tuple[int, ...], /, *, budget: CoordinateEnclosureBudget
    ) -> Mapping[tuple[int, ...], ExactScalar]:
        local = self._products.get(index)
        if local is not None:
            return local
        supports = self.arguments if self._reference_key is None else self._reference_key
        if any(
            exponent > 0 and not argument
            for exponent, argument in zip(index, supports, strict=True)
        ):
            # Inspect the full factor support before executing a prefix.
            # A later rational zero annihilates all earlier reference powers.
            return {}
        key = None
        if self._reference_key is None:
            terms = sum(len(argument) for argument in self.arguments)
            budget.reserve(terms, 64 + 64 * len(self.arguments) + 64 * terms)
            self._reference_key = tuple(
                tuple(argument.items()) for argument in self.arguments
            )
        key = (
            (self._reference_key, index, self._scalar_kinds)
            if self._scalar_kinds
            else (self._reference_key, index)
        )
        immutable = budget.reference_monomial_cache.get(key)
        if immutable is not None:
            self._products[index] = immutable
            return immutable
        factor = self._products[()]
        for axis in range(len(index)):
            prefix = index[: axis + 1]
            cached = self._products.get(prefix)
            if cached is None:
                exponent = index[axis]
                argument_power = self._powers[axis].get(exponent)
                if argument_power is None:
                    argument = self.arguments[axis]
                    if self._reference_key is not None:
                        entries = self._reference_key[axis]
                        budget.reserve(len(entries), 256 + 128 * len(entries))
                        argument = dict(entries)
                    argument_power = self._powers[axis][exponent] = power(
                        argument, exponent, self.dimension, budget=budget
                    )
                cached = self._products[prefix] = multiply(
                    factor, argument_power, budget=budget
                )
            factor = cached
        if key is not None:
            # Snapshot and profile the full retained support once. Its
            # read-only coefficients and rational profile then share this owner.
            immutable = PreparedPolynomialSupport(factor, budget=budget)
            budget.retain_basis((key, immutable))
            budget.reference_monomial_cache[key] = immutable
            self._products[index] = immutable
            return immutable
        return factor

    def __call__(
        self,
        polynomial: Mapping[tuple[int, ...], ExactScalar],
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> Polynomial:
        if self._axis_affine is not None and _all_rational(polynomial):
            return _axis_affine_polynomial(polynomial, self._axis_affine, budget=budget)
        monomials: list[Mapping[tuple[int, ...], ExactScalar]] = []
        weights: list[ExactScalar] = []
        budget.reserve(0, 256 + 64 * len(polynomial))
        for index, value in polynomial.items():
            if len(index) != len(self.arguments):
                raise ValueError(
                    "coordinate_enclosure.arguments must match the polynomial parameter dimension; got unequal exponent ranks."
                )
            monomial = self._monomial(index, budget=budget)
            if not monomial:
                # The full reference support already proved this product
                # zero. No coefficient addition or target mutation executes,
                # so do not re-profile/re-reserve the unchanged accumulator.
                continue
            monomials.append(monomial)
            weights.append(value)
        if not monomials:
            return {}
        # Prepare the full bank once rather than repeatedly scaling terms,
        # scanning the growing accumulator and replacing its storage bound.
        # The weighted reduction retains the same insertion/cancellation order.
        return linear_combinations(tuple(monomials), (tuple(weights),), budget=budget)[0]


def compose(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    arguments: tuple[Mapping[tuple[int, ...], ExactScalar], ...]
    | PreparedPolynomialArguments,
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    return ArgumentComposition(arguments, budget=budget)(polynomial, budget=budget)


def affine_arguments(
    origin: np.ndarray, matrix: np.ndarray, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    result: list[Polynomial] = []
    result_storage = 0
    with budget.temporary_scope():
        variables = axes(matrix.shape[1], budget=budget)
        for start, row in zip(origin, matrix, strict=True):
            with budget.temporary_scope():
                offset = constant(
                    start
                    if isinstance(start, (Fraction, RadicalElement))
                    else Fraction(float(start)),
                    matrix.shape[1],
                    budget=budget,
                )
                linear = sum_polynomials(
                    tuple(
                        scale(
                            variable,
                            value
                            if isinstance(value, (Fraction, RadicalElement))
                            else Fraction(float(value)),
                            budget=budget,
                        )
                        for variable, value in zip(variables, row, strict=True)
                    ),
                    budget=budget,
                )
                before_add = budget.temporary_bytes_upper
                result.append(add(offset, linear, budget=budget))
                row_storage = budget.temporary_bytes_upper - before_add
            budget.reserve(0, row_storage)
            result_storage += row_storage
    # Coefficients escape; axis banks and scaled/replaced accumulators do not.
    budget.reserve(0, result_storage)
    return tuple(result)


def sum_polynomials(
    polynomials: tuple[Polynomial, ...], *, budget: CoordinateEnclosureBudget
) -> Polynomial:
    result: Polynomial = {}
    result_storage = 0
    with budget.temporary_scope(), budget.live_storage() as live:
        for polynomial in polynomials:
            with budget.temporary_scope():
                before_add = budget.temporary_bytes_upper
                result = add(result, polynomial, budget=budget)
                result_storage = budget.temporary_bytes_upper - before_add
            live.set_bound(result_storage)
    budget.reserve(0, result_storage)
    return result


def _linear_combination_profiled(
    polynomials: tuple[Mapping[tuple[int, ...], ExactScalar], ...],
    weights: tuple[ExactScalar, ...],
    *,
    dimension: int,
    coefficient_bits: int,
    active_terms: int,
    scaled_terms: int,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    """Accumulate after this operation earned its support and weight bounds."""
    reserve_polynomial(
        active_terms + scaled_terms,
        active_terms,
        dimension,
        coefficient_bits,
        budget=budget,
    )
    _reserve_coefficient_work(
        polynomials,
        weights,
        work=active_terms + scaled_terms,
        count=active_terms,
        degree=2,
        budget=budget,
    )
    result: Polynomial = {}
    for polynomial, weight in zip(polynomials, weights, strict=True):
        if not weight:
            continue
        for index, coefficient in polynomial.items():
            value = coefficient if weight == 1 else coefficient * weight
            if not value:
                continue
            previous = result.get(index)
            total = value if previous is None else previous + value
            if total:
                result[index] = total
            else:
                result.pop(index, None)
    return result


def linear_combinations(
    polynomials: tuple[Mapping[tuple[int, ...], ExactScalar], ...],
    weight_bank: tuple[tuple[ExactScalar, ...], ...],
    *,
    prepared_profile: tuple[int, int, int] | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...]:
    """Share bank preparation across rational accumulators without output copies."""
    if any(len(polynomials) != len(weights) for weights in weight_bank):
        raise ValueError(
            "coordinate_enclosure.weights must match the polynomial bank; got unequal row counts."
        )
    if prepared_profile is None:
        numerator, denominator = coefficient_profile(polynomials, budget=budget)
        dimension = next(
            (len(index) for polynomial in polynomials for index in polynomial), 0
        )
    else:
        budget.reserve(1)
        numerator, denominator, dimension = prepared_profile
    results = []
    for weights in weight_bank:
        budget.reserve(len(weights))
        weight_bits, active_terms, scaled_terms = 1, 0, 0
        for polynomial, weight in zip(polynomials, weights, strict=True):
            weight_bits = max(
                weight_bits,
                _scalar_bits(weight),
            )
            if weight:
                active_terms += len(polynomial)
                if weight != 1:
                    scaled_terms += len(polynomial)
        results.append(
            _linear_combination_profiled(
                polynomials,
                weights,
                dimension=dimension,
                coefficient_bits=numerator
                + denominator
                + weight_bits
                + max(active_terms, 1).bit_length(),
                active_terms=active_terms,
                scaled_terms=scaled_terms,
                budget=budget,
            )
        )
    return tuple(results)


def _nodal_axis(
    nodes: np.ndarray, axis: int, dimension: int, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    variable = axes(dimension, budget=budget)[axis]
    result = []
    for i, node in enumerate(nodes):
        polynomial = constant(1, dimension, budget=budget)
        x = Fraction(float(node))
        for j, other in enumerate(nodes):
            if i != j:
                y = Fraction(float(other))
                polynomial = multiply(
                    polynomial,
                    scale(
                        add(
                            variable,
                            constant(-y, dimension, budget=budget),
                            budget=budget,
                        ),
                        1 / (x - y),
                        budget=budget,
                    ),
                    budget=budget,
                )
        result.append(polynomial)
    return tuple(result)


def _simplex_basis(
    owner: SimplexNodalFamily, dimension: int, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    variables = axes(dimension, budget=budget)
    barycentric = (
        add(
            constant(1, dimension, budget=budget),
            scale(sum_polynomials(variables, budget=budget), -1, budget=budget),
            budget=budget,
        ),
        *variables,
    )
    modal = []
    for index in owner.multiindices:
        weight = math.factorial(owner.order) // math.prod(
            math.factorial(value) for value in index
        )
        term = constant(weight, dimension, budget=budget)
        for argument, exponent in zip(barycentric, index, strict=True):
            term = multiply(
                term, power(argument, exponent, dimension, budget=budget), budget=budget
            )
        modal.append(term)
    budget.reserve(
        owner.coefficients.size,
        owner.coefficients.size * np.dtype(np.float64).itemsize,
    )
    coefficients = np.asarray(owner.coefficients, dtype=np.float64)
    return tuple(
        sum_polynomials(
            tuple(
                scale(term, Fraction(float(value)), budget=budget)
                for term, value in zip(modal, coefficients[:, i], strict=True)
            ),
            budget=budget,
        )
        for i in range(coefficients.shape[1])
    )


def jacobi(
    degree: int,
    alpha: int,
    axis: int,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    variable = axes(dimension, budget=budget)[axis]
    shifted = add(variable, constant(-1, dimension, budget=budget), budget=budget)
    return sum_polynomials(
        tuple(
            scale(
                multiply(
                    power(shifted, degree - i, dimension, budget=budget),
                    power(variable, i, dimension, budget=budget),
                    budget=budget,
                ),
                math.comb(degree + alpha, i) * math.comb(degree, degree - i),
                budget=budget,
            )
            for i in range(degree + 1)
        ),
        budget=budget,
    )


@checked
def source_basis(
    element: CellGeometryElement, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...] | None:
    """Prepare declared expressions once within an explicit resource ledger."""
    key = element_identity(element)
    if key not in budget.basis_cache:
        with budget.temporary_scope():
            basis = _source_basis(element, budget=budget)
            if basis is not None:
                budget.retain_basis(basis)
            budget.basis_cache[key] = basis
    return budget.basis_cache[key]


def element_identity(element: CellGeometryElement, /) -> tuple[str, str]:
    """Declared identity of an element: its declared ID and full array contents."""

    return element.element_id, imported_fingerprint.canonical_fingerprint(
        imported_fingerprint.array_tree_fingerprint(element)
    )


def reference_composition_arguments(
    element: CellGeometryElement, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    """Prepare the immutable source chart once on its preparation resource ledger."""
    key = element_identity(element)
    cached = budget.reference_composition_cache.get(key)
    if cached is not None:
        return cached
    with budget.temporary_scope():
        arguments = _prepare_reference_composition_arguments(element, budget=budget)
        budget.retain_basis((key, arguments))
        budget.reference_composition_cache[key] = arguments
        return arguments


def prepared_reference_arguments(
    element: CellGeometryElement, /, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...] | PreparedPolynomialArguments:
    """Reuse immutable rational reference preparation, never physical coefficient bindings."""
    source_key = element_identity(element)
    cached = budget.prepared_reference_cache.get(source_key)
    if cached is not None:
        return cached
    original = reference_composition_arguments(element, budget=budget)
    if any(isinstance(argument, RationalPolynomial) for argument in original):
        # Rational actions keep their existing preparation homogeneous preparation.
        return original
    with budget.temporary_scope():
        terms = sum(len(argument) for argument in original)
        budget.reserve(terms, 64 + 64 * len(original) + 64 * terms)
        ordered_key = tuple(tuple(argument.items()) for argument in original)
        budget.reserve(terms, 128 + 128 * len(original) + 64 * terms)
        snapshots = tuple(dict(entries) for entries in ordered_key)
        arguments = tuple(MappingProxyType(snapshot) for snapshot in snapshots)
        dimension = next(
            (len(index) for argument in arguments for index in argument), len(arguments)
        )
        axis_affine = _axis_affine_arguments(arguments, dimension, budget=budget)
        prepared = PreparedPolynomialArguments(
            arguments, ordered_key, dimension, axis_affine
        )
        # The record, full ordered key, underlying snapshot dictionaries,
        # immutable views and classification all belong to this source ledger.
        budget.retain_basis(
            (source_key, prepared, ordered_key, snapshots, arguments, axis_affine)
        )
        budget.prepared_reference_cache[source_key] = prepared
        return prepared


def _prepare_reference_composition_arguments(
    element: CellGeometryElement, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    """Rational reference chart, independent of rounded tabulation values."""

    if not isinstance(
        element,
        (
            _cell_geometry_dispatch().PolynomialComposedCellGeometryElement,
            _cell_geometry_dispatch().RationalComposedCellGeometryElement,
        ),
    ):
        raise TypeError(
            "coordinate_enclosure.chart_element must be a composed coordinate element; got another source kind."
        )
    if isinstance(element, _cell_geometry_dispatch().RationalComposedCellGeometryElement):
        expected = np.asarray(
            [
                [float(Fraction(a, b)) for a, b in row]
                for row in element.chart_coefficients
            ],
            dtype=np.float64,
        )
        if not np.array_equal(
            expected.view(np.uint64),
            np.asarray(element.chart_coordinates).view(np.uint64),
        ):
            raise ValueError(
                "coordinate_enclosure.chart_coefficients must be the RNE of their rational source; got inconsistent execution coefficients."
            )
    return prepare_reference_chart_arguments(
        element.chart_element,
        element.chart_coefficients,
        element.source_element.topological_dimension,
        budget=budget,
    )


@checked
def prepare_reference_chart_arguments(
    chart_element: CellGeometryElement,
    coefficients: tuple[tuple[tuple[int, int], ...], ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...]:
    """Prepare admitted source chart inputs without a partial preparation instance."""
    basis = source_basis(chart_element, budget=budget)
    if basis is None:
        raise ValueError(
            "coordinate_enclosure.chart_basis must have supported source expressions; got unsupported tabulator."
        )
    return tuple(
        sum_polynomials(
            tuple(
                scale(term, Fraction(*row[axis]), budget=budget)
                for term, row in zip(basis, coefficients, strict=True)
            ),
            budget=budget,
        )
        for axis in range(dimension)
    )


def _spline_axis_basis(
    knots: np.ndarray,
    degree: int,
    span: int,
    axis: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...]:
    """Rational Cox-de Boor source polynomials on one normalized knot span."""
    vector = rational_vector(knots)
    count = len(vector) - degree - 1
    parameter = add(
        constant(vector[span], 2, budget=budget),
        scale(
            axes(2, budget=budget)[axis], vector[span + 1] - vector[span], budget=budget
        ),
        budget=budget,
    )
    basis = tuple(
        constant(1 if row == span else 0, 2, budget=budget)
        for row in range(len(vector) - 1)
    )
    for order in range(1, degree + 1):
        following = []
        for row in range(len(vector) - order - 1):
            left, right = (
                vector[row + order] - vector[row],
                vector[row + order + 1] - vector[row + 1],
            )
            first = (
                {}
                if not left
                else scale(
                    multiply(
                        add(
                            parameter,
                            constant(-vector[row], 2, budget=budget),
                            budget=budget,
                        ),
                        basis[row],
                        budget=budget,
                    ),
                    1 / left,
                    budget=budget,
                )
            )
            second = (
                {}
                if not right
                else scale(
                    multiply(
                        add(
                            constant(vector[row + order + 1], 2, budget=budget),
                            scale(parameter, -1, budget=budget),
                            budget=budget,
                        ),
                        basis[row + 1],
                        budget=budget,
                    ),
                    1 / right,
                    budget=budget,
                )
            )
            following.append(add(first, second, budget=budget))
        basis = tuple(following)
    return basis[:count]


@checked
def source_expressions(
    element: CellGeometryElement, *, budget: CoordinateEnclosureBudget
) -> tuple[Expression, ...] | None:
    """Prepare the source scalar source expression once on the preparation ledger."""
    key = element_identity(element)
    if key in budget.source_expression_cache:
        return budget.source_expression_cache[key]
    with budget.temporary_scope():
        expressions = _prepare_source_expressions(element, budget=budget)
        if expressions is not None:
            budget.retain_basis((key, expressions))
        budget.source_expression_cache[key] = expressions
        return expressions


def _prepare_source_expressions(
    element: CellGeometryElement, *, budget: CoordinateEnclosureBudget
) -> tuple[Expression, ...] | None:
    """Retained scalar coordinate source basis, including source rational weights."""

    if isinstance(element, _cell_geometry_dispatch().SplineCellGeometryElement):
        snapshot = _cell_geometry_dispatch().prepare_spline_source_snapshot(
            element, budget=budget
        )
        u = _spline_axis_basis(
            snapshot.u_knots,
            element.u_degree,
            element.span_indices[0],
            0,
            budget=budget,
        )
        v = _spline_axis_basis(
            snapshot.v_knots,
            element.v_degree,
            element.span_indices[1],
            1,
            budget=budget,
        )
        numerators = tuple(
            scale(
                multiply(first, second, budget=budget),
                Fraction(float(weight)),
                budget=budget,
            )
            for first, row in zip(u, snapshot.weights, strict=True)
            for second, weight in zip(v, np.asarray(row), strict=True)
        )
        denominator = sum_polynomials(numerators, budget=budget)
        return tuple(
            rational_expression(term, denominator, budget=budget) for term in numerators
        )
    if isinstance(
        element,
        (
            _cell_geometry_dispatch().PolynomialComposedCellGeometryElement,
            _cell_geometry_dispatch().RationalComposedCellGeometryElement,
        ),
    ):
        source = source_expressions(element.source_element, budget=budget)
        if source is None:
            return None
        arguments = prepared_reference_arguments(element, budget=budget)
        return tuple(
            expression_compose(term, arguments, budget=budget) for term in source
        )
    if isinstance(element, _cell_geometry_dispatch().RestrictedCellGeometryElement):
        source = source_expressions(element.source_element, budget=budget)
        if source is None:
            return None
        return restrict_chart_expressions(
            source,
            element.source_element.cell_kind,
            element.cell_kind,
            np.asarray(element.offset),
            np.asarray(element.matrix),
            budget=budget,
        )
    return source_basis(element, budget=budget)


def _source_basis(
    element: CellGeometryElement, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...] | None:
    """Lower a supported coordinate source into polynomial dictionary support."""
    geometry = _cell_geometry_dispatch()
    match element:
        case geometry.BarycentricCellGeometryElement():
            return _barycentric_source_basis(element, budget=budget)
        case geometry.LayerColumnCellGeometryElement():
            return _layer_column_source_basis(element, budget=budget)
        case geometry.SplineCellGeometryElement():
            expressions = source_expressions(element, budget=budget)
            if expressions is None or any(
                isinstance(term, RationalPolynomial) for term in expressions
            ):
                return None
            return tuple(
                term for term in expressions if not isinstance(term, RationalPolynomial)
            )
        case (
            geometry.PolynomialComposedCellGeometryElement()
            | geometry.RationalComposedCellGeometryElement()
        ):
            basis = source_basis(element.source_element, budget=budget)
            if basis is None:
                return None
            arguments = prepared_reference_arguments(element, budget=budget)
            return tuple(compose(term, arguments, budget=budget) for term in basis)
        case geometry.RestrictedCellGeometryElement():
            if element.source_element.cell_kind == "pyramid":
                # Physical affine restriction becomes rational under collapse.
                return None
            basis = source_basis(element.source_element, budget=budget)
            if basis is None:
                return None
            arguments = affine_arguments(
                np.asarray(element.offset), np.asarray(element.matrix), budget=budget
            )
            return tuple(compose(term, arguments, budget=budget) for term in basis)
        case FiniteElementSpec():
            return _finite_element_source_basis(element, budget=budget)
        case _:
            return None


def _barycentric_source_basis(
    element: cell_geometry_types.BarycentricCellGeometryElement,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...] | None:
    """Apply the retained cardinal-action chain in its declared order."""
    root = element.source_element
    while isinstance(root, _cell_geometry_dispatch().BarycentricCellGeometryElement):
        root = root.source_element
    basis = source_basis(root, budget=budget)
    if basis is None:
        raise ValueError(
            "coordinate_enclosure.barycentric_source must retain its P1 cardinal source; got unsupported root basis."
        )
    current = element
    while isinstance(current, _cell_geometry_dispatch().BarycentricCellGeometryElement):
        budget.reserve(
            current.barycentric_weights.size,
            128 + 512 * current.barycentric_weights.size,
        )
        weights = (
            _cell_geometry_dispatch()
            .prepare_barycentric_source_snapshot(current, budget=budget)
            .barycentric_weights
        )
        columns = tuple(
            tuple(
                Fraction(float(weights[target, source]))
                for target in range(weights.shape[0])
            )
            for source in range(weights.shape[1])
        )
        basis = linear_combinations(basis, columns, budget=budget)
        current = current.source_element
    return basis


def _layer_column_source_basis(
    element: LayerColumnCellGeometryElement, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...] | None:
    """Form wall profiles and signed authored corner corrections."""
    wall, corners = (
        source_basis(element.wall_element, budget=budget),
        source_basis(element.corner_element, budget=budget),
    )
    if wall is None or corners is None or element.station_axis != 2:
        return None
    if element.local_dof_count != 2 * len(wall) + 4 * len(corners):
        return None
    station = axes(3, budget=budget)[2]
    factors = (
        add(
            constant(1, 3, budget=budget),
            scale(station, -1, budget=budget),
            budget=budget,
        ),
        station,
    )
    lifted_wall: tuple[Polynomial, ...] = tuple(
        {(*index, 0): value for index, value in term.items()} for term in wall
    )
    lifted_corners: tuple[Polynomial, ...] = tuple(
        {(*index, 0): value for index, value in term.items()} for term in corners
    )
    profiles = tuple(
        multiply(term, factor, budget=budget)
        for factor in factors
        for term in lifted_wall
    )
    actual = tuple(
        multiply(term, factor, budget=budget)
        for factor in factors
        for term in lifted_corners
    )
    return (*profiles, *actual, *(scale(term, -1, budget=budget) for term in actual))


def _finite_element_source_basis(
    element: FiniteElementSpec, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...] | None:
    """Validate scalar execution provenance before preparing its reference family."""
    if element.mapping != "identity" or element.value_shape:
        return None
    if type(element.tabulator) is _cell_geometry_dispatch().SweptCoordinateTabulator:
        source = element.tabulator.source
        expected = "prism" if source.cell_kind == "triangle" else "hexahedron"
        if (
            source.cell_kind not in ("triangle", "quadrilateral")
            or element.cell_kind != expected
            or element.degree != source.degree
            or element.local_dof_count != 2 * source.local_dof_count
        ):
            return None
        basis = source_basis(source, budget=budget)
        if basis is None:
            return None
        lifted: tuple[Polynomial, ...] = tuple(
            {(*index, 0): value for index, value in term.items()} for term in basis
        )
        w = axes(3, budget=budget)[2]
        return tuple(
            multiply(term, factor, budget=budget)
            for factor in (
                add(
                    constant(1, 3, budget=budget),
                    scale(w, -1, budget=budget),
                    budget=budget,
                ),
                w,
            )
            for term in lifted
        )
    if type(element.tabulator) is imported_fem__form_elements.ProxyTabulator:
        owner = element.tabulator.basis
        if (
            type(owner) is not imported_fem__form_elements.FormBasis
            or owner.form_degree != 0
            or owner.dimension != element.topological_dimension
            or owner.order != element.degree
            or owner.local_dof_count != element.local_dof_count
            or element.tabulator.value_spec != element.value_spec
            or element.representation not in ("polynomial_moment", "rational_moment")
        ):
            return None
        return _scalar_form_basis(owner, budget=budget)
    if element.representation != "point_value":
        return None
    dimension = element.topological_dimension
    if (
        element.degree == 0
        and element.family == "DiscontinuousLagrange"
        and element.tabulator is None
    ):
        canonical = imported_fem__reference.discontinuous_element(element.cell_kind, 0)
        if element.element_id == canonical.element_id and element.local_dof_count == 1:
            return (constant(1, dimension, budget=budget),)
        return None
    if isinstance(element.tabulator, _cell_geometry_dispatch().CoordinateTabulator):
        if (
            element.cell_kind != element.tabulator.cell_kind
            or element.degree != element.tabulator.degree
        ):
            return None
        return lattice_basis(
            element.tabulator.cell_kind, element.tabulator.degree, budget=budget
        )
    owner = owning_tabulator_source(element.tabulator)
    if owner is not None:
        return _owned_nodal_source_basis(element, owner, budget=budget)
    return _lagrange_source_basis(element, budget=budget)


def _owned_nodal_source_basis(
    element: FiniteElementSpec,
    owner: FiniteElementSpec
    | SimplexNodalFamily
    | imported_fem__high_order.ReferenceNodalFamily
    | hybrid_reference_owner.HybridReferenceFamily
    | imported_fem__form_elements.FormBasis,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...] | None:
    """Lower a reference family only when its declared element semantics agree."""
    dimension = element.topological_dimension
    match owner:
        case FiniteElementSpec():
            if (element.cell_kind, element.degree, element.local_dof_count) != (
                owner.cell_kind,
                owner.degree,
                owner.local_dof_count,
            ):
                return None
            return source_basis(owner, budget=budget)
        case imported_fem__high_order.SimplexNodalFamily():
            if element.cell_kind != owner.cell_kind or element.degree != owner.order:
                return None
            return _simplex_basis(owner, dimension, budget=budget)
        case imported_fem__high_order.ReferenceNodalFamily():
            if element.cell_kind != owner.cell_kind or element.degree != max(
                owner.orders
            ):
                return None
            factors = tuple(
                _nodal_axis(np.asarray(nodes), axis, dimension, budget=budget)
                for axis, nodes in enumerate(owner.nodes_by_axis)
            )
            return tuple(
                math_product_polynomials(choice, dimension, budget=budget)
                for choice in product(*factors)
            )
        case hybrid_reference_owner.HybridReferenceFamily():
            return _hybrid_source_basis(element, owner, budget=budget)
        case imported_fem__form_elements.FormBasis():
            return None
        case _:
            assert_never(owner)


def _hybrid_source_basis(
    element: FiniteElementSpec,
    owner: hybrid_reference_owner.HybridReferenceFamily,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...] | None:
    """Assemble prism tensor factors or collapsed pyramid modes."""
    if element.cell_kind != owner.cell_kind or element.degree != owner.degree:
        return None
    if owner.cell_kind == "prism":
        triangle = _simplex_basis(
            imported_fem__high_order.SimplexNodalFamily("triangle", owner.orders[0]),
            2,
            budget=budget,
        )
        lifted: tuple[Polynomial, ...] = tuple(
            {(*index, 0): value for index, value in term.items()} for term in triangle
        )
        axial = _nodal_axis(np.unique(np.asarray(owner.nodes)[:, 2]), 2, 3, budget=budget)
        terms = tuple(multiply(a, b, budget=budget) for a, b in product(lifted, axial))
        return tuple(terms[index] for index in owner.basis_permutation)
    collapse = add(
        constant(1, 3, budget=budget),
        scale(axes(3, budget=budget)[2], -1, budget=budget),
        budget=budget,
    )
    modes = tuple(
        math_product_polynomials(
            (
                jacobi(i, 0, 0, 3, budget=budget),
                jacobi(j, 0, 1, 3, budget=budget),
                power(collapse, max(i, j), 3, budget=budget),
                jacobi(k, 2 * max(i, j) + 2, 2, 3, budget=budget),
            ),
            3,
            budget=budget,
        )
        for i, j, k in owner.modal_indices
    )
    budget.reserve(
        owner.coefficients.size,
        owner.coefficients.size * np.dtype(np.float64).itemsize,
    )
    coefficients = np.asarray(owner.coefficients, dtype=np.float64)
    return tuple(
        sum_polynomials(
            tuple(
                scale(term, Fraction(float(value)), budget=budget)
                for term, value in zip(modes, coefficients[:, i], strict=True)
            ),
            budget=budget,
        )
        for i in range(coefficients.shape[1])
    )


def _lagrange_source_basis(
    element: FiniteElementSpec, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...] | None:
    """Prepare the admitted built-in scalar Lagrange basis."""
    dimension = element.topological_dimension
    if element.tabulator is not None:
        return None
    if (
        element.element_id
        != lagrange_element(element.cell_kind, element.degree).element_id
    ):
        return None
    variables = axes(dimension, budget=budget)
    if element.cell_kind in ("triangle", "tetrahedron"):
        barycentric = (
            add(
                constant(1, dimension, budget=budget),
                scale(sum_polynomials(variables, budget=budget), -1, budget=budget),
                budget=budget,
            ),
            *variables,
        )
        if element.degree == 1:
            return barycentric
        vertices = tuple(
            multiply(
                value,
                add(
                    scale(value, 2, budget=budget),
                    constant(-1, dimension, budget=budget),
                    budget=budget,
                ),
                budget=budget,
            )
            for value in barycentric
        )
        edges = tuple(
            scale(
                multiply(barycentric[a], barycentric[b], budget=budget), 4, budget=budget
            )
            for a, b in ((0, 1), (1, 2), (2, 0))
        )
        return (*vertices, *edges)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    factors = tuple(
        _nodal_axis(np.unique(nodes[:, axis]), axis, dimension, budget=budget)
        for axis in range(dimension)
    )
    return tuple(
        math_product_polynomials(
            tuple(factors[axis][int(value)] for axis, value in enumerate(row)),
            dimension,
            budget=budget,
        )
        for row in nodes
    )


def _scalar_form_basis(
    owner: imported_fem__form_elements.FormBasis, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    """Extract the retained declared scalar form execution, including tensor factors."""

    if owner.form_degree != 0:
        raise TypeError(
            "coordinate_enclosure.form_source must have scalar FormBasis degree zero; got another form degree."
        )
    if owner.hybrid_factors is not None:
        source = tuple(
            field[0]
            for field in imported_fem__form_expressions.component_expressions(
                owner, budget=budget
            )
        )
        if owner.family == "pyramid-trimmed":
            arguments = chart_arguments("pyramid", 3, budget=budget)
            source = tuple(
                expression_compose(term, arguments, budget=budget) for term in source
            )
        if any(isinstance(term, RationalPolynomial) for term in source):
            raise RationalEnclosureError(
                "coordinate_enclosure.hybrid_source must have a resolved collapsed-chart denominator; got a quotient."
            )
        return tuple(term for term in source if not isinstance(term, RationalPolynomial))
    if owner.tensor_factors is None:
        budget.reserve(
            owner.coefficients.size,
            owner.coefficients.size * np.dtype(np.float64).itemsize,
        )
        coefficients = np.asarray(owner.coefficients, dtype=np.float64)
        return tuple(
            {
                index: Fraction(float(value))
                for index, value in zip(
                    owner.exponents, coefficients[:, 0, dof], strict=True
                )
                if value
            }
            for dof in range(owner.local_dof_count)
        )
    factors = owner.tensor_factors
    budget.reserve(factors.zero.size + factors.indices.size + factors.components.size)
    zero = np.asarray(factors.zero, dtype=np.float64)
    indices = np.asarray(factors.indices, dtype=np.int32)
    components = np.asarray(factors.components, dtype=np.float64)
    if np.any(np.asarray(factors.differential_axes)):
        raise ValueError(
            "coordinate_enclosure.tensor_factors must contain no differential axes; got active differential axes."
        )
    dimension = owner.dimension
    variables = axes(dimension, budget=budget)
    axial = []
    for axis, variable in enumerate(variables):
        collapse = add(
            constant(1, dimension, budget=budget),
            scale(variable, -1, budget=budget),
            budget=budget,
        )
        bubble = multiply(variable, collapse, budget=budget)
        modes = tuple(
            jacobi(order, 0, axis, dimension, budget=budget)
            for order in range(owner.order - 1)
        )
        axial.append(
            tuple(
                add(
                    add(
                        scale(collapse, Fraction(float(zero[0, dof])), budget=budget),
                        scale(variable, Fraction(float(zero[1, dof])), budget=budget),
                        budget=budget,
                    ),
                    multiply(
                        bubble,
                        sum_polynomials(
                            tuple(
                                scale(mode, Fraction(float(value)), budget=budget)
                                for mode, value in zip(modes, zero[2:, dof], strict=True)
                            ),
                            budget=budget,
                        ),
                        budget=budget,
                    ),
                    budget=budget,
                )
                for dof in range(zero.shape[1])
            )
        )
    return tuple(
        scale(
            math_product_polynomials(
                tuple(axial[axis][indices[axis, dof]] for axis in range(dimension)),
                dimension,
                budget=budget,
            ),
            Fraction(float(components[dof, 0])),
            budget=budget,
        )
        for dof in range(owner.local_dof_count)
    )


def math_product_polynomials(
    polynomials: tuple[Polynomial, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    if any(not polynomial for polynomial in polynomials):
        return {}
    result = constant(1, dimension, budget=budget)
    for polynomial in polynomials:
        result = multiply(result, polynomial, budget=budget)
    return result


def coordinate_source_rows(local: CoordinateCoefficients) -> CoordinateSourceBank:
    if isinstance(local, np.ndarray):
        if local.ndim != 2:
            raise ValueError(
                "coordinate_enclosure.coefficients must be a rank-two host bank; got another rank."
            )
        rows = tuple(
            tuple(
                value
                if isinstance(value, (Fraction, RadicalElement))
                else Fraction(float(value))
                for value in row
            )
            for row in local
        )
    else:
        rows = local
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(
            "coordinate_enclosure.coefficients must be a nonempty rectangular host bank; got empty or unequal rows."
        )
    return rows


@checked
def coordinate_polynomials(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...] | None:
    """Reuse full rational polynomial maps in the preparation preparation ledger."""
    with budget.temporary_scope():
        coefficients, key = coordinate_preparation_key(element, local, budget=budget)
        return _coordinate_polynomials_from_prepared(
            element, coefficients, key, budget=budget
        )


def _coordinate_polynomials_from_prepared(
    element: CellGeometryElement,
    coefficients: CoordinateSourceBank,
    key: CoordinatePreparationKey,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...] | None:
    """Consume a full element/bank binding earned in the enclosing operation."""
    cached = budget.polynomial_cache.get(key)
    if cached is not None:
        return cached
    prepared = _prepare_multi_affine_coordinates(
        element, coefficients, prepared_key=key, budget=budget
    )
    if prepared is not None:
        return prepared[0]
    coordinates = _affine_simplex_coordinates(element, coefficients, budget=budget)
    if coordinates is None:
        coordinates = _prepare_coordinate_polynomials(
            element, coefficients, budget=budget
        )
    if coordinates is not None:
        budget.retain_basis((key, coordinates))
        budget.coordinate_cache[key] = coordinates
        budget.polynomial_cache[key] = coordinates
    return coordinates


def coordinate_bank_cache_key(
    element: CellGeometryElement, coefficients: CoordinateSourceBank, /
) -> CoordinatePreparationKey:
    """Bind an immutable coefficient bank and its scalar representations to an element."""
    identity = element_identity(element)
    kinds = _scalar_kind_key(value for row in coefficients for value in row)
    return (*identity, coefficients, kinds) if kinds else (*identity, coefficients)


@checked
def coordinate_preparation_key(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[CoordinateSourceBank, CoordinatePreparationKey]:
    immutable_key = None
    if isinstance(local, tuple) and all((isinstance(row, tuple) for row in local)):
        immutable_key = coordinate_bank_cache_key(element, local)
        prepared = budget.coefficient_preparation_cache.get(immutable_key)
        if prepared is not None:
            return prepared
    budget.reserve(
        local.size if isinstance(local, np.ndarray) else sum(len(row) for row in local)
    )
    if isinstance(local, np.ndarray):
        budget.reserve(0, 128 + 512 * local.size)
    coefficients = coordinate_source_rows(local)
    if len(coefficients) != element.local_dof_count:
        raise ValueError(
            "coordinate_enclosure.coefficients must include every source degree of freedom; got unequal row count."
        )
    key = (
        immutable_key
        if immutable_key is not None
        else coordinate_bank_cache_key(element, coefficients)
    )
    prepared = coefficients, key
    if immutable_key is not None:
        # Retain the validated full immutable bank and element binding.
        # A mutable host array still earns every current normalization; only
        # identical immutable source data can skip that executed preparation.
        budget.reserve(1, 128)
        budget.retain_basis((immutable_key, prepared))
        budget.coefficient_preparation_cache[immutable_key] = prepared
    return prepared


def prepared_coordinate_source_bank(
    geometry: CellGeometrySpec, /, *, budget: CoordinateEnclosureBudget
) -> CoordinateSourceBank:
    """Retain one immutable current source bank in the source preparation ledger.

    The geometry identity binds source definitions, routes and scientific source
    IDs; the full live numerical tree additionally binds every preparation source
    bank. Equal carrier arrays alone never authorize reuse of a foreign source.
    """

    key = coordinate_source_bank_key(geometry)
    cached = budget.source_bank_cache.get(key)
    if cached is not None:
        return cached
    with budget.temporary_scope():
        bank = geometry.source_coordinates(budget=budget)
        budget.retain_basis((key, bank))
    budget.source_bank_cache[key] = bank
    return bank


def _prepare_coordinate_polynomials(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...] | None:

    if isinstance(element, _cell_geometry_dispatch().BarycentricCellGeometryElement):
        # Tabulation multiplies the full cardinal basis by the retained
        # outer-to-inner action stack. Contract physical coefficients in reverse
        # order, exactly, rather than materializing/recombining cardinal maps.
        actions = []
        root = element
        while isinstance(root, _cell_geometry_dispatch().BarycentricCellGeometryElement):
            actions.append(
                _cell_geometry_dispatch()
                .prepare_barycentric_source_snapshot(root, budget=budget)
                .barycentric_weights
            )
            root = root.source_element
        rows = coordinate_source_rows(local)
        for action in reversed(actions):
            weights = action
            budget.reserve(weights.size)
            exact = tuple(
                tuple(
                    (index, Fraction(float(value)))
                    for index, value in enumerate(row)
                    if value
                )
                for row in weights
            )
            transformed = []
            for coefficients in exact:
                image = []
                for axis in range(len(rows[0])):
                    budget.reserve(3 * len(coefficients))
                    image.append(
                        sum(
                            (
                                coefficient * rows[index][axis]
                                for index, coefficient in coefficients
                            ),
                            Fraction(0),
                        )
                    )
                transformed.append(tuple(image))
            rows = tuple(transformed)
        return coordinate_polynomials(root, rows, budget=budget)

    if isinstance(element, _cell_geometry_dispatch().RestrictedCellGeometryElement):
        source = coordinate_polynomials(element.source_element, local, budget=budget)
        if source is None:
            return None
        return restrict_chart_coordinates(
            source,
            element.source_element.cell_kind,
            element.cell_kind,
            np.asarray(element.offset),
            np.asarray(element.matrix),
            budget=budget,
        )
    basis = source_basis(element, budget=budget)
    if basis is None:
        return None
    rows = coordinate_source_rows(local)
    profile = None
    key = element_identity(element)
    profile = budget.basis_profile_cache.get(key)
    if profile is None:
        numerator, denominator = coefficient_profile(basis, budget=budget)
        profile = (
            numerator,
            denominator,
            next((len(index) for term in basis for index in term), 0),
        )
        budget.reserve(1, 128)
        budget.retain_basis((key, profile))
        budget.basis_profile_cache[key] = profile
    return linear_combinations(
        basis,
        tuple(tuple(row[axis] for row in rows) for axis in range(len(rows[0]))),
        prepared_profile=profile,
        budget=budget,
    )


def determinant(
    matrix: tuple[tuple[Polynomial, ...], ...], *, budget: CoordinateEnclosureBudget
) -> Polynomial:

    dimension = len(matrix)
    reduction: tuple[list[Polynomial], list[Fraction]] | None = None
    active_terms = scaled_terms = 0
    for permutation in imported_itertools.permutations(range(dimension)):
        factors = tuple(matrix[i][permutation[i]] for i in range(dimension))
        if any(not factor for factor in factors):
            continue
        sign = (-1) ** sum(
            permutation[i] > permutation[j]
            for i in range(dimension)
            for j in range(i + 1, dimension)
        )
        if reduction is None:
            budget.reserve(0, 800)
            reduction = ([], [])
        terms, weights = reduction
        budget.reserve(0, 320)
        term = math_product_polynomials(
            factors, len(next(iter(factors[0]))), budget=budget
        )
        terms.append(term)
        active_terms += len(term)
        if sign == -1:
            scaled_terms += len(term)
        weights.append(Fraction(sign))
    if reduction is None:
        return {}
    terms, weights = reduction
    polynomials = tuple(terms)
    numerator, denominator = coefficient_profile(polynomials, budget=budget)
    return _linear_combination_profiled(
        polynomials,
        tuple(weights),
        dimension=next((len(index) for term in terms for index in term), 0),
        coefficient_bits=numerator + denominator + 2 + max(active_terms, 1).bit_length(),
        active_terms=active_terms,
        scaled_terms=scaled_terms,
        budget=budget,
    )


def divide_collapse(polynomial: Polynomial, repetitions: int) -> Polynomial | None:
    result = polynomial
    for _ in range(repetitions):
        quotient: Polynomial = {}
        grouped: dict[tuple[int, ...], dict[int, ExactScalar]] = {}
        for index, value in result.items():
            grouped.setdefault(index[:-1], {})[index[-1]] = value
        for prefix, values in grouped.items():
            coefficient: ExactScalar = Fraction(0)
            for degree in range(max(values) + 1):
                coefficient += values.get(degree, Fraction(0))
                if degree < max(values) and coefficient:
                    quotient[(*prefix, degree)] = coefficient
            if coefficient:
                return None
        result = quotient
    return result


def determinant_polynomial(
    coordinates: tuple[Polynomial, ...],
    cell_kind: str,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial | None:
    jacobian = tuple(
        tuple(derivative(value, axis, budget=budget) for axis in range(dimension))
        for value in coordinates
    )
    if len(coordinates) == dimension:
        value = determinant(jacobian, budget=budget)
        return divide_collapse(value, 2) if cell_kind == "pyramid" else value
    gram = tuple(
        tuple(
            sum_polynomials(
                tuple(multiply(row[i], row[j], budget=budget) for row in jacobian),
                budget=budget,
            )
            for j in range(dimension)
        )
        for i in range(dimension)
    )
    return determinant(gram, budget=budget)


def _prepare_bernstein_lattice(
    reference: BernsteinDomain,
    degrees: tuple[int, ...],
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[int, ...], ...]:
    visits = (
        (degrees[0] + 1) ** dimension
        if reference == "simplex"
        else (degrees[0] + 1) ** 2 * (degrees[1] + 1)
        if reference == "prism"
        else math.prod(degree + 1 for degree in degrees)
    )
    budget.reserve(visits)
    match reference:
        case "simplex":
            return tuple(
                index
                for index in product(range(degrees[0] + 1), repeat=dimension)
                if sum(index) <= degrees[0]
            )
        case "prism":
            return tuple(
                (*index, height)
                for index in product(range(degrees[0] + 1), repeat=2)
                if sum(index) <= degrees[0]
                for height in range(degrees[1] + 1)
            )
        case "box":
            return tuple(product(*(range(degree + 1) for degree in degrees)))
        case _:
            assert_never(reference)


def _bernstein_lattice(
    reference: BernsteinDomain,
    degrees: tuple[int, ...],
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[int, ...], ...]:
    owner = budget.bernstein_lattice_cache
    key = (reference, degrees, dimension)
    lattice = owner.get(key)
    if lattice is None:
        with budget.temporary_scope():
            lattice = _prepare_bernstein_lattice(
                reference, degrees, dimension, budget=budget
            )
            budget.retain_basis(lattice)
            owner[key] = lattice
    return lattice


def _dominating_simplex_indices(
    alpha: tuple[int, ...], degree: int
) -> Iterator[tuple[int, ...]]:
    """Generate only dominating simplex indices in declared lexicographic order."""
    dimension = len(alpha)
    suffix = tuple(sum(alpha[axis + 1 :]) for axis in range(dimension))

    def descend(
        axis: int, remaining: int, prefix: tuple[int, ...]
    ) -> Iterator[tuple[int, ...]]:
        if axis == dimension:
            yield prefix
            return
        for value in range(alpha[axis], remaining - suffix[axis] + 1):
            yield from descend(axis + 1, remaining - value, (*prefix, value))

    yield from descend(0, degree, ())


def _simplex_control_position(beta: tuple[int, ...], degree: int) -> int:
    position, remaining = 0, degree
    for axis, value in enumerate(beta):
        dimensions = len(beta) - axis
        position += math.comb(remaining + dimensions, dimensions) - math.comb(
            remaining - value + dimensions, dimensions
        )
        remaining -= value
    return position


def _bernstein_support_profile(
    value: Mapping[tuple[int, ...], ExactScalar],
    reference: BernsteinDomain,
    degrees: tuple[int, ...],
    dimension: int,
) -> tuple[int, int, int]:
    """Pre-admit retained control initialization, term and accumulation visits."""
    match reference:
        case "simplex":
            nodes = math.comb(degrees[0] + dimension, dimension)
        case "prism":
            nodes = math.comb(degrees[0] + 2, 2) * (degrees[1] + 1)
        case "box":
            nodes = math.prod(degree + 1 for degree in degrees)
        case invalid:
            assert_never(invalid)
    entries = 0
    zero = (0,) * dimension
    for alpha in value:
        if alpha == zero:
            continue
        if reference == "box":
            count = math.prod(
                degree - exponent + 1
                for exponent, degree in zip(alpha, degrees, strict=True)
            )
        else:
            simplex_dimension = dimension if reference == "simplex" else 2
            remaining = degrees[0] - sum(alpha[:simplex_dimension])
            count = math.comb(remaining + simplex_dimension, simplex_dimension)
            if reference == "prism":
                count *= degrees[1] - alpha[2] + 1
        entries += count
    return nodes + len(value) + entries, nodes, entries


def _bernstein_weight_entry(
    position: int, numerator: int, *, budget: CoordinateEnclosureBudget
) -> tuple[int, int]:
    """Share an integer entry without sharing its table's denominator or chart."""
    budget.reserve(1)
    entry = (position, numerator)
    cached = budget.bernstein_weight_entry_cache.get(entry)
    if cached is not None:
        return cached
    with budget.temporary_scope():
        budget.retain_basis((entry,))
        budget.bernstein_weight_entry_cache[entry] = entry
    return entry


def _prepare_bernstein_weights(
    reference: BernsteinDomain,
    degrees: tuple[int, ...],
    dimension: int,
    alpha: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Rational monomial-to-Bernstein weights of ``alpha`` as integers over one denominator.

    Box weights are ``prod comb(b, a) / comb(n, a)``; simplex weights use
    ``prod comb(b, a)`` over the matching sequential binomial denominator.
    This cancels the common ``prod a!`` factor before allocating the table.
    Lattice points not dominating ``alpha`` receive no contribution.
    """
    entries: list[tuple[int, int]] = []
    if reference == "box":
        lengths = tuple(
            degree - exponent + 1 for exponent, degree in zip(alpha, degrees, strict=True)
        )
        count = math.prod(lengths)
        prefixes = sum(math.prod(lengths[: axis + 1]) for axis in range(dimension))
    else:
        simplex_dimension = dimension if reference == "simplex" else 2
        remaining = degrees[0] - sum(alpha[:simplex_dimension])
        count = math.comb(remaining + simplex_dimension, simplex_dimension)
        prefixes = sum(
            math.comb(remaining + axis, axis) for axis in range(1, simplex_dimension + 1)
        )
        if reference == "prism":
            count *= degrees[1] - alpha[2] + 1
            prefixes += count
    budget.reserve(prefixes + count)
    if reference == "box":
        denominator = math.prod(
            math.comb(n, a) for a, n in zip(alpha, degrees, strict=True)
        )
        strides = tuple(
            math.prod(degree + 1 for degree in degrees[axis + 1 :])
            for axis in range(dimension)
        )
        for beta in product(
            *(range(a, degree + 1) for a, degree in zip(alpha, degrees, strict=True))
        ):
            position = sum(
                value * stride for value, stride in zip(beta, strides, strict=True)
            )
            entries.append(
                _bernstein_weight_entry(
                    position,
                    math.prod(math.comb(b, a) for a, b in zip(alpha, beta, strict=True)),
                    budget=budget,
                )
            )
        return denominator, tuple(entries)
    simplex_dimension = dimension if reference == "simplex" else 2
    denominator = 1
    remaining = degrees[0]
    for axis in range(simplex_dimension):
        denominator *= math.comb(remaining, alpha[axis])
        remaining -= alpha[axis]
    if reference == "prism":
        denominator *= math.comb(degrees[1], alpha[2])
    budget.reserve(0, 256 + 64 * (count + 1))
    shared_numerators = {denominator: denominator}
    for beta in _dominating_simplex_indices(alpha[:simplex_dimension], degrees[0]):
        position = _simplex_control_position(beta, degrees[0])
        numerator = math.prod(
            math.comb(beta[i], alpha[i]) for i in range(simplex_dimension)
        )
        if reference == "prism":
            for height in range(alpha[2], degrees[1] + 1):
                weight = numerator * math.comb(height, alpha[2])
                entries.append(
                    _bernstein_weight_entry(
                        position * (degrees[1] + 1) + height,
                        shared_numerators.setdefault(weight, weight),
                        budget=budget,
                    )
                )
        else:
            entries.append(
                _bernstein_weight_entry(
                    position,
                    shared_numerators.setdefault(numerator, numerator),
                    budget=budget,
                )
            )
    return denominator, tuple(entries)


def _bernstein_weights(
    reference: BernsteinDomain,
    degrees: tuple[int, ...],
    dimension: int,
    alpha: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    key = (reference, degrees, dimension, alpha)
    weights = budget.bernstein_weight_cache.get(key)
    if weights is None:
        with budget.temporary_scope():
            weights = _prepare_bernstein_weights(
                reference, degrees, dimension, alpha, budget=budget
            )
            budget.retain_basis(weights)
            budget.bernstein_weight_cache[key] = weights
    return weights


def _require_canonical_bernstein_term(
    index: tuple[int, ...], coefficient: ExactScalar
) -> None:
    if type(index) is not tuple or any(type(exponent) is not int for exponent in index):
        raise TypeError(
            "coordinate_enclosure.bernstein_exponents must be built-in tuples of built-in int values; got unsupported exponent types."
        )
    if type(coefficient) not in (Fraction, RadicalElement):
        raise TypeError(
            "coordinate_enclosure.bernstein_coefficients must be Fraction values without subclasses or RadicalElement values without subclasses; got an unsupported coefficient type."
        )


def _bernstein_preparation_key(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    reference: BernsteinDomain,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> BernsteinCoefficientKey:
    """Bind ordered rational/algebraic coefficients and integer exponents to their chart."""
    budget.reserve(max(len(polynomial), 1), 1024 + 128 * len(polynomial))
    entries: list[tuple[tuple[int, ...], ExactScalar]] = []
    maximum_hash_scratch = 0
    for index, value in polynomial.items():
        _require_canonical_bernstein_term(index, value)
        maximum_hash_scratch = max(
            maximum_hash_scratch,
            _scalar_hash_storage(value),
        )
        entries.append((index, value))
    budget.reserve(0, 4 * maximum_hash_scratch)
    kinds = _scalar_kind_key(polynomial.values())
    key = reference, dimension, tuple(entries)
    return (*key, kinds) if kinds else key


def bernstein_coefficients(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    domain: str,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[ExactScalar, ...]:
    """Reuse an immutable control net only for the same full coefficient support."""
    reference = parse(domain, BernsteinDomain, "domain")
    if len(polynomial) == 1 and (reference != "prism" or dimension == 3):
        index = next(iter(polynomial))
        if len(index) == dimension and not any(index):
            coefficient = polynomial[index]
            _require_canonical_bernstein_term(index, coefficient)
            budget.reserve(1, 256)
            return (coefficient,)
    with budget.temporary_scope():
        key = _bernstein_preparation_key(polynomial, reference, dimension, budget=budget)
        cached = budget.bernstein_coefficient_cache.get(key)
        if cached is not None:
            return cached
        coefficients = _prepare_bernstein_coefficients(
            polynomial, reference, dimension, budget=budget
        )
        budget.retain_polynomial_preparation((key, coefficients))
        budget.bernstein_coefficient_cache[key] = coefficients
        return coefficients


def _prepare_bernstein_coefficients(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    reference: BernsteinDomain,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[ExactScalar, ...]:
    match reference:
        case "simplex":
            degrees = (max((sum(index) for index in polynomial), default=0),)
        case "prism":
            degrees = (
                max((sum(index[:2]) for index in polynomial), default=0),
                max((index[2] for index in polynomial), default=0),
            )
        case "box":
            degrees = tuple(
                max((index[axis] for index in polynomial), default=0)
                for axis in range(dimension)
            )
        case _:
            assert_never(reference)
    if reference == "simplex" and degrees[0] <= 1:
        # Degree-one Bernstein coefficients are the rational vertex images.
        # Avoid constructing rational weight tables for this identity action.
        numerator, denominator = coefficient_profile((polynomial,), budget=budget)
        reserve_polynomial(
            len(polynomial) + dimension,
            dimension + 1,
            dimension,
            numerator + denominator + 1,
            budget=budget,
        )
        _reserve_coefficient_work(
            (polynomial,),
            work=len(polynomial) + dimension,
            count=dimension + 1,
            budget=budget,
        )
        origin = polynomial.get((0,) * dimension, Fraction(0))
        if degrees[0] == 0:
            return (origin,)
        return (
            origin,
            *(
                origin
                + polynomial.get(
                    tuple(int(axis == selected) for axis in range(dimension)),
                    Fraction(0),
                )
                for selected in reversed(range(dimension))
            ),
        )
    numerator, denominator = coefficient_profile((polynomial,), budget=budget)
    visits, nodes, entries = _bernstein_support_profile(
        polynomial, reference, degrees, dimension
    )
    degree = max((sum(index) for index in polynomial), default=0)
    weight_bits = 2 * dimension * degree * max(degree + 1, 1).bit_length()
    reserve_polynomial(
        visits,
        nodes + entries,
        dimension,
        numerator + denominator + weight_bits + max(len(polynomial), 1).bit_length(),
        budget=budget,
    )
    _reserve_coefficient_work(
        (polynomial,), work=visits, count=nodes + entries, budget=budget
    )
    if not _all_rational(polynomial):
        zero = (0,) * dimension
        controls = [polynomial.get(zero, Fraction(0))] * len(
            _bernstein_lattice(reference, degrees, dimension, budget=budget)
        )
        for alpha, coefficient in polynomial.items():
            if alpha == zero:
                continue
            common, weights = _bernstein_weights(
                reference, degrees, dimension, alpha, budget=budget
            )
            for position, weight in weights:
                controls[position] += coefficient * Fraction(weight, common)
        return tuple(controls)
    # Rational rational sums are accumulated as integers over one common
    # denominator; ``Fraction`` normalization yields the identical coefficients.
    zero = (0,) * dimension
    origin = polynomial.get(zero)
    if origin is None:
        origin = Fraction(0)
    terms = []
    common = origin.denominator
    for alpha, value in polynomial.items():
        if alpha == zero:
            continue
        weight_denominator, entries = _bernstein_weights(
            reference, degrees, dimension, alpha, budget=budget
        )
        denominator = value.denominator * weight_denominator
        terms.append((value.numerator, denominator, entries))
        common = math.lcm(common, denominator)
    initial = origin.numerator * (common // origin.denominator)
    totals = [initial] * len(
        _bernstein_lattice(reference, degrees, dimension, budget=budget)
    )
    for numerator, denominator, entries in terms:
        factor = numerator * (common // denominator)
        for position, weight in entries:
            totals[position] += factor * weight
    return tuple(Fraction(total, common) for total in totals)


def polynomial_bounds(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    domain: str,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:
    coefficients = bernstein_coefficients(polynomial, domain, dimension, budget=budget)
    return _scalar_endpoint(
        min(coefficients), upper=False, budget=budget
    ), _scalar_endpoint(max(coefficients), upper=True, budget=budget)


def _affine_polynomial_jacobian(
    coordinates: tuple[Expression, ...],
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Polynomial, ...], ...] | None:
    polynomials = []
    for value in coordinates:
        if isinstance(value, RationalPolynomial):
            return None
        budget.reserve(len(value))
        if any(sum(index) > 1 for index in value):
            return None
        polynomials.append(value)
    numerator, denominator = coefficient_profile(tuple(polynomials), budget=budget)
    reserve_polynomial(
        len(coordinates) * dimension,
        len(coordinates) * dimension,
        dimension,
        numerator + denominator,
        budget=budget,
    )
    indices = tuple(
        tuple(int(axis == selected) for axis in range(dimension))
        for selected in range(dimension)
    )
    zero = (0,) * dimension
    return tuple(
        tuple(
            {zero: coefficient} if (coefficient := value.get(index, Fraction(0))) else {}
            for index in indices
        )
        for value in polynomials
    )


def physical_jacobian(
    coordinates: tuple[Polynomial, ...],
    cell_kind: str,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Polynomial, ...], ...] | None:
    if cell_kind != "pyramid":
        affine = _affine_polynomial_jacobian(coordinates, dimension, budget=budget)
        if affine is not None:
            return affine
    rows = tuple(
        tuple(derivative(value, axis, budget=budget) for axis in range(dimension))
        for value in coordinates
    )
    if cell_kind != "pyramid":
        return rows
    variables = axes(3, budget=budget)
    result = []
    for first, second, height in rows:
        x = divide_collapse(first, 1)
        y = divide_collapse(second, 1)
        if x is None or y is None:
            return None
        z = add(
            height,
            add(
                multiply(
                    add(
                        variables[0],
                        constant(Fraction(-1, 2), 3, budget=budget),
                        budget=budget,
                    ),
                    x,
                    budget=budget,
                ),
                multiply(
                    add(
                        variables[1],
                        constant(Fraction(-1, 2), 3, budget=budget),
                        budget=budget,
                    ),
                    y,
                    budget=budget,
                ),
                budget=budget,
            ),
            budget=budget,
        )
        result.append((x, y, z))
    return tuple(result)


@overload
def evaluate(
    polynomial: Mapping[tuple[int, ...], Fraction],
    point: tuple[Fraction, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction: ...


@overload
def evaluate(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    point: tuple[ExactScalar, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactScalar: ...


def evaluate(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    point: tuple[ExactScalar, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactScalar:
    if not polynomial:
        return Fraction(0)
    if not any(point):
        budget.reserve(1)
        origin = polynomial.get((0,) * len(point))
        return Fraction(0) if origin is None else origin
    budget.reserve(len(polynomial))
    if not _all_rational(polynomial) or not _all_rational_values(point):
        _reserve_coefficient_work(
            (polynomial,),
            point,
            work=len(polynomial),
            count=len(polynomial),
            degree=1 + max((sum(index) for index in polynomial), default=0),
            budget=budget,
        )
        result: ExactScalar = Fraction(0)
        for index, coefficient in polynomial.items():
            term: ExactScalar = coefficient
            for coordinate, exponent in zip(point, index, strict=True):
                term *= coordinate**exponent
            result += term
        return result
    # Rational evaluation over the common denominator of the coefficients and of
    # the point's maximal powers; ``Fraction`` normalizes the identical value.
    degrees = [0] * len(point)
    common = 1
    for index, coefficient in polynomial.items():
        for axis, exponent in enumerate(index):
            if exponent > degrees[axis]:
                degrees[axis] = exponent
        common = math.lcm(common, coefficient.denominator)
    numerators = tuple(
        tuple(value.numerator**exponent for exponent in range(degree + 1))
        for value, degree in zip(point, degrees, strict=True)
    )
    denominators = tuple(
        tuple(value.denominator**exponent for exponent in range(degree + 1))
        for value, degree in zip(point, degrees, strict=True)
    )
    total = 0
    for index, coefficient in polynomial.items():
        term = coefficient.numerator * (common // coefficient.denominator)
        for axis, exponent in enumerate(index):
            term *= (
                numerators[axis][exponent] * denominators[axis][degrees[axis] - exponent]
            )
        total += term
    return Fraction(total, common * math.prod(powers[-1] for powers in denominators))


def reference_image(
    polynomials: tuple[Polynomial, ...],
    cell_kind: str,
    point: tuple[ExactScalar, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[ExactScalar, ...]:
    """Evaluate one physical reference point under chart expressions.

    Pyramid expressions live on the collapsed cube ``x = u (1 - w) + w / 2``.
    The apex is the rational ``w = 1`` slice, which a single-valued map must leave
    independent of ``u`` and ``v``; no representative collapsed point is chosen.
    """
    if cell_kind != "pyramid":
        return tuple(evaluate(value, point, budget=budget) for value in polynomials)
    x, y, z = point
    if z != 1:
        return tuple(
            evaluate(
                value, ((x - z / 2) / (1 - z), (y - z / 2) / (1 - z), z), budget=budget
            )
            for value in polynomials
        )
    if x != Fraction(1, 2) or y != Fraction(1, 2):
        raise ValueError(
            "coordinate_enclosure.pyramid_point must be the apex at apex height; got other horizontal coordinates."
        )
    image = []
    for value in polynomials:
        budget.reserve(len(value))
        apex: dict[tuple[int, int], ExactScalar] = {}
        for (first, second, _), coefficient in value.items():
            apex[first, second] = apex.get((first, second), Fraction(0)) + coefficient
        if any(coefficient for index, coefficient in apex.items() if index != (0, 0)):
            raise ValueError(
                "coordinate_enclosure.pyramid_expression must be single-valued at its apex; got nonconstant horizontal terms."
            )
        image.append(apex.get((0, 0), Fraction(0)))
    return tuple(image)


def corner_images(
    polynomials: tuple[Polynomial, ...],
    cell_kind: str,
    *,
    budget: CoordinateEnclosureBudget,
) -> CoordinateSourceBank:
    """Evaluate the reference corners of chart expressions."""

    return tuple(
        reference_image(
            polynomials,
            cell_kind,
            rational_vector(corner),
            budget=budget,
        )
        for corner in imported_reference_cell.reference_cell_topology(cell_kind).vertices
    )


def affine_simplex_corner_images(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    budget: CoordinateEnclosureBudget,
) -> CoordinateSourceBank | None:
    """Return rational corners only after proving the full simplex source affine."""
    source = element
    geometry = _cell_geometry_dispatch()
    while isinstance(
        source,
        (geometry.RestrictedCellGeometryElement, geometry.BarycentricCellGeometryElement),
    ):
        if source.cell_kind not in ("interval", "triangle", "tetrahedron"):
            return None
        source = source.source_element
    match source:
        case FiniteElementSpec() as source:
            if (
                source.cell_kind not in ("interval", "triangle", "tetrahedron")
                or source.degree != 1
            ):
                return None
        case _:
            return None
    key = element_identity(source)
    affine = budget.affine_simplex_source_cache.get(key)
    if affine is None:
        basis = source_basis(source, budget=budget)
        if basis is None:
            affine = False
        else:
            budget.reserve(sum(len(value) for value in basis))
            affine = all(
                sum(index) <= 1
                for value in basis
                for index, coefficient in value.items()
                if coefficient
            )
        budget.retain_basis((key, affine))
        budget.affine_simplex_source_cache[key] = affine
    return coordinate_corner_images(element, local, budget=budget) if affine else None


def _affine_simplex_coordinates(
    element: CellGeometryElement,
    local: CoordinateSourceBank,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...] | None:
    """Recover a proven affine map from its corners without parent composition."""
    geometry = _cell_geometry_dispatch()
    if not isinstance(
        element,
        (geometry.RestrictedCellGeometryElement, geometry.BarycentricCellGeometryElement),
    ):
        return None
    if isinstance(element, geometry.BarycentricCellGeometryElement):
        # Without earned corners, the coefficient action costs less than
        # constructing and retaining its full cardinal corner-weight graph.
        _, key = coordinate_preparation_key(element, local, budget=budget)
        if key not in budget.coordinate_corner_cache:
            return None
    corners = affine_simplex_corner_images(element, local, budget=budget)
    if corners is None:
        return None
    dimension, ambient = len(corners) - 1, len(corners[0])
    budget.reserve(len(corners) * ambient)
    numerator_bits, denominator = 1, 1
    for point in corners:
        for value in point:
            value_numerator, value_denominator = _scalar_profile(value)
            numerator_bits = max(numerator_bits, value_numerator)
            denominator = math.lcm(denominator, value_denominator)
    reserve_polynomial(
        dimension * ambient,
        len(corners) * ambient,
        dimension,
        numerator_bits + 2 * denominator.bit_length() + 1,
        budget=budget,
    )
    _reserve_coefficient_work(
        banks=(corners,),
        work=dimension * ambient,
        count=len(corners) * ambient,
        budget=budget,
    )
    zero = (0,) * dimension
    directions = tuple(
        tuple(int(axis == parameter) for axis in range(dimension))
        for parameter in range(dimension)
    )
    coordinates = []
    for component in range(ambient):
        value = {zero: corners[0][component]} if corners[0][component] else {}
        for index, point in zip(directions, corners[1:], strict=True):
            coefficient = point[component] - corners[0][component]
            if coefficient:
                value[index] = coefficient
        coordinates.append(value)
    return tuple(coordinates)


@checked
def coordinate_corner_images(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    budget: CoordinateEnclosureBudget,
) -> CoordinateSourceBank | None:
    """Evaluate reference-corner images of a full or restricted coordinate map.

    Each target corner passes through the stored affine restriction chain into
    the root reference, whose expression uses rational/algebraic field arithmetic.
    Restrictions that are rational on the target chart, such as tetrahedral
    children of a curved pyramid, need no polynomial form.
    """
    with budget.temporary_scope():
        return _prepared_coordinate_corner_images(element, local, budget=budget)


def _prepared_coordinate_corner_images(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    budget: CoordinateEnclosureBudget,
) -> CoordinateSourceBank | None:
    """Reuse rational source-basis corner weights within one bounded preparation."""
    coefficients, map_key = coordinate_preparation_key(element, local, budget=budget)
    return _coordinate_corner_images_from_prepared(
        element, coefficients, map_key, budget=budget
    )


def _prepare_reference_denominators(
    basis: tuple[Expression, ...], *, budget: CoordinateEnclosureBudget
) -> PreparedReferenceDenominators:
    """Group ordered denominator terms for a source bank."""
    budget.reserve(0, 1024 + 64 * len(basis))
    groups: dict[PolynomialRepresentationKey, int] = {}
    denominators: list[Polynomial] = []
    routes: list[int] = []
    for term in basis:
        match term:
            case RationalPolynomial():
                budget.reserve(len(term.denominator), 256 + 320 * len(term.denominator))
                key = _polynomial_representation_key(term.denominator)
                route = groups.get(key)
                if route is None:
                    route = len(denominators)
                    groups[key] = route
                    denominators.append(term.denominator)
                routes.append(route)
            case dict():
                routes.append(-1)
            case invalid:
                assert_never(invalid)
    return tuple(denominators), tuple(routes)


def _reference_basis_values(
    basis: tuple[Expression, ...],
    points: CoordinateSourceBank,
    cell_kind: str,
    prepared: PreparedReferenceDenominators,
    *,
    budget: CoordinateEnclosureBudget,
) -> Iterator[tuple[ExactScalar, ...]]:
    """Evaluate a source bank with one shared-denominator evaluation per point."""
    denominators, routes = prepared
    domain = "simplex" if cell_kind in ("triangle", "tetrahedron") else "box"
    for point in points:
        budget.reserve(0, 512 + 16 * (len(basis) + len(denominators)))
        evaluated = tuple(
            evaluate(denominator, point, budget=budget) for denominator in denominators
        )
        row = []
        for term, route in zip(basis, routes, strict=True):
            match term:
                case RationalPolynomial():
                    denominator = evaluated[route]
                    if denominator:
                        row.append(
                            _scalar_quotient(
                                evaluate(term.numerator, point, budget=budget),
                                denominator,
                                budget=budget,
                            )
                        )
                    else:
                        row.append(
                            _rational_reference_corner_limit(
                                term, point, domain, budget=budget
                            )
                        )
                case dict():
                    row.append(
                        reference_image((term,), cell_kind, point, budget=budget)[0]
                    )
                case invalid:
                    assert_never(invalid)
        yield tuple(row)


def _prepare_coordinate_corner_weights(
    element: CellGeometryElement, *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[str, str], CoordinateCornerWeights] | None:
    """Prepare sparse corner rows from the declared root and reference chain."""
    with budget.temporary_scope():
        if has_cartesian_reference_chain(element):
            root, arguments = coordinate_reference_chain(element, budget=budget)
        else:
            # A full coefficient action owns its cardinal source law;
            # it is not an authored Cartesian restriction/reference chain.
            root = element
            scalar = _cell_geometry_dispatch().require_scalar_coordinate_element(
                element, "Exact coordinate corner preparation"
            )
            arguments = chart_arguments(
                scalar.cell_kind, scalar.topological_dimension, budget=budget
            )
        basis = source_expressions(root, budget=budget)
        if basis is None:
            return None
        source_identity = element_identity(root)
        if any(isinstance(term, RationalPolynomial) for term in basis):
            budget.reserve(1)
            prepared = budget.reference_denominator_group_cache.get(source_identity)
            if prepared is None:
                prepared = _prepare_reference_denominators(basis, budget=budget)
                budget.retain_basis((source_identity, prepared))
                budget.reference_denominator_group_cache[source_identity] = prepared
        else:
            budget.reserve(0, 256 + 16 * len(basis))
            prepared = ((), (-1,) * len(basis))
        corners = tuple(
            rational_vector(point)
            for point in imported_reference_cell.reference_cell_topology(
                element.cell_kind
            ).vertices
        )
        points = tuple(
            tuple(
                expression_reference_evaluate(
                    value,
                    corner,
                    "box"
                    if element.cell_kind == "pyramid"
                    else "simplex"
                    if element.cell_kind in ("triangle", "tetrahedron")
                    else "box",
                    budget=budget,
                )
                for value in arguments
            )
            for corner in corners
        )
        rows = []
        for values in _reference_basis_values(
            basis, points, root.cell_kind, prepared, budget=budget
        ):
            row = []
            for node, value in enumerate(values):
                if value:
                    row.append((node, value))
            rows.append(tuple(row))
        weights = tuple(rows)
        budget.retain_basis((source_identity, weights))
    return source_identity, weights


def _retained_coordinate_point(
    point: tuple[ExactScalar, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[ExactScalar, ...]:
    """Share immutable source point values without sharing source bindings."""
    with budget.temporary_scope():
        budget.reserve(
            len(point),
            # Fraction hashing needs integer scratch, not a serialized point copy.
            256
            + 4
            * max(
                (_scalar_hash_storage(value) for value in point),
                default=0,
            ),
        )
        kinds = _scalar_kind_key(point)
        key: CoordinatePointKey = (point, kinds) if kinds else point
        cached = budget.coordinate_point_value_cache.get(key)
        if cached is not None:
            return cached
        budget.retain_basis((key,))
        budget.coordinate_point_value_cache[key] = point
        return point


def _coordinate_corner_images_from_prepared(
    element: CellGeometryElement,
    coefficients: CoordinateSourceBank,
    map_key: CoordinatePreparationKey,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CoordinateSourceBank | None:
    """Consume this call's validated declared bank/key without rescanning it."""

    cached = budget.coordinate_corner_cache.get(map_key)
    if cached is not None:
        return cached
    key = map_key[:2]
    prepared_weights = budget.corner_weight_cache.get(key)
    if prepared_weights is None:
        prepared_weights = _prepare_coordinate_corner_weights(element, budget=budget)
        if prepared_weights is None:
            return None
        budget.corner_weight_cache[key] = prepared_weights
    source_identity, weights = prepared_weights
    dimension = len(coefficients[0])
    budget.reserve(len(weights))
    weighted = tuple(row for row in weights if len(row) != 1 or row[0][1] != 1)
    budget.reserve(dimension * (len(weights) - len(weighted)))
    # A completed cardinal row and full source bank define the rational
    # linear operation. Different chart owners retain their own map bindings;
    # only an identical operation under the same authentic source law shares
    # its immutable numerical image.
    bank_kinds = map_key[3] if len(map_key) == 4 else b""
    operation_keys = tuple(
        _coordinate_linear_key(source_identity, coefficients, row, bank_kinds)
        for row in weights
    )
    pending = tuple(
        dict.fromkeys(
            operation_key
            for row, operation_key in zip(weights, operation_keys, strict=True)
            if (len(row) != 1 or row[0][1] != 1)
            and operation_key not in budget.corner_linear_cache
        )
    )
    budget.reserve(0, 128 + 8 * len(weights) + (72 + 8 * dimension) * len(pending))
    if pending:
        budget.reserve(
            sum(len(row) for row in coefficients)
            + sum(len(operation_key[2]) for operation_key in pending)
        )
        coefficient_numerator = max(
            _scalar_profile(value)[0] for row in coefficients for value in row
        )
        coefficient_denominator = math.lcm(
            *(_scalar_profile(value)[1] for row in coefficients for value in row)
        ).bit_length()
        # Every dot product has the source denominator times its weight
        # denominator. Scaling numerators to those denominators and adding its
        # retained terms gives this expression-derived bound before expansion.
        bits = max(
            coefficient_numerator
            + coefficient_denominator
            + max((_scalar_profile(weight)[0] for _, weight in row), default=0)
            + math.lcm(*(_scalar_profile(weight)[1] for _, weight in row)).bit_length()
            + max(len(row), 1).bit_length()
            for operation_key in pending
            for row in (operation_key[2],)
        )
        reserve_polynomial(
            dimension * sum(len(operation_key[2]) for operation_key in pending),
            dimension * len(pending),
            0,
            bits,
            budget=budget,
        )
        _reserve_coefficient_work(
            banks=(coefficients,),
            work=dimension * sum(len(operation_key[2]) for operation_key in pending),
            count=dimension * len(pending),
            degree=2,
            budget=budget,
        )
        for operation_key in pending:
            row = operation_key[2]
            image = tuple(
                sum(
                    (weight * coefficients[node][axis] for node, weight in row),
                    Fraction(0),
                )
                for axis in range(dimension)
            )
            image = _retained_coordinate_point(image, budget=budget)
            budget.retain_basis((operation_key, image))
            budget.corner_linear_cache[operation_key] = image
    # A nodal corner image is its unit-weight coefficient row itself; sharing that
    # rational record keeps one retained copy per source node across its cells.
    images = tuple(
        coefficients[row[0][0]]
        if len(row) == 1 and row[0][1] == 1
        else budget.corner_linear_cache[operation_key]
        for row, operation_key in zip(weights, operation_keys, strict=True)
    )
    budget.retain_basis((map_key, images))
    budget.coordinate_corner_cache[map_key] = images
    return images


def _multi_affine_source(
    element: CellGeometryElement, /, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Whether every source scalar has degree at most one in each axis."""
    key = element_identity(element)
    if key is not None:
        cached = budget.multi_affine_cache.get(key)
        if cached is not None:
            return cached
    basis = source_basis(element, budget=budget)
    if basis is not None:
        budget.reserve(sum(len(term) for term in basis))
    decided = basis is not None and all(
        exponent <= 1 for term in basis for index in term for exponent in index
    )
    if key is not None:
        budget.multi_affine_cache[key] = decided
    return decided


def multi_affine_coordinates(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Polynomial, ...], CoordinateSourceBank] | None:
    """Return multi-affine box coordinates and corner images, or ``None``.

    Direct quadrilateral/hexahedron sources of degree at most one in each axis
    have a unique multi-affine interpolant of their rational/algebraic corner
    images. Inclusion-exclusion on the corner lattice yields its source-basis
    polynomial without rounding, preserving reference vertex order.
    Restricted, composed, spline, and other sources return ``None``.
    """
    with budget.temporary_scope():
        return _prepare_multi_affine_coordinates(element, local, budget=budget)


def _prepare_multi_affine_coordinates(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    prepared_key: CoordinatePreparationKey | None = None,
) -> tuple[tuple[Polynomial, ...], CoordinateSourceBank] | None:
    """Prove and retain the full source interpolant, not a sampled fit."""

    if (
        element.cell_kind not in ("quadrilateral", "hexahedron")
        or isinstance(
            element,
            (
                _cell_geometry_dispatch().PolynomialComposedCellGeometryElement,
                _cell_geometry_dispatch().RationalComposedCellGeometryElement,
                _cell_geometry_dispatch().RestrictedCellGeometryElement,
                _cell_geometry_dispatch().SplineCellGeometryElement,
            ),
        )
        or not _multi_affine_source(element, budget=budget)
    ):
        return None
    if prepared_key is None:
        coefficients, key = coordinate_preparation_key(element, local, budget=budget)
    else:
        key, coefficients = prepared_key, prepared_key[2]
    cached = budget.polynomial_cache.get(key)
    corners = budget.coordinate_corner_cache.get(key)
    if cached is not None and corners is not None:
        return cached, corners
    with budget.temporary_scope():
        corners = _coordinate_corner_images_from_prepared(
            element, coefficients, key, budget=budget
        )
    if corners is None:
        return None
    masks, exponents = box_corner_lattice(element.cell_kind)
    dimension, count, ambient = len(exponents[0]), len(exponents), len(corners[0])
    values: list[tuple[ExactScalar, ...]] = [()] * count
    for mask, image in zip(masks, corners, strict=True):
        values[mask] = image
    # Every power coefficient is a signed sum of at most ``count`` images over
    # their common denominator; each axis pass adds at most one numerator bit.
    budget.reserve(count * ambient)
    numerator_bits, common = 1, 1
    for image in corners:
        for value in image:
            value_numerator, value_denominator = _scalar_profile(value)
            numerator_bits = max(numerator_bits, value_numerator)
            common = math.lcm(common, value_denominator)
    reserve_polynomial(
        dimension * (count // 2) * ambient,
        count * ambient,
        dimension,
        numerator_bits + common.bit_length() + dimension,
        budget=budget,
    )
    _reserve_coefficient_work(
        banks=(corners,),
        work=dimension * (count // 2) * ambient,
        count=count * ambient,
        budget=budget,
    )
    for axis in range(dimension):
        bit = 1 << axis
        for mask in range(count):
            if mask & bit:
                values[mask] = tuple(
                    a - b for a, b in zip(values[mask], values[mask ^ bit], strict=True)
                )
    polynomials = tuple(
        {
            exponent: value[coordinate]
            for exponent, value in zip(exponents, values, strict=True)
            if value[coordinate]
        }
        for coordinate in range(ambient)
    )
    if key is not None:
        budget.retain_basis((key, polynomials, corners))
        budget.coordinate_cache[key] = polynomials
        budget.polynomial_cache[key] = polynomials
        budget.coordinate_corner_cache[key] = corners
    return polynomials, corners


@host_table_cache(max_entries=256, max_bytes=1048576)
def box_corner_lattice(
    cell_kind: str, /
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Axis bitmask of each reference corner and the monomial exponent of each mask.

    The exponent tuples are shared by every multi-affine coordinate expression.
    """

    reference = np.asarray(
        imported_reference_cell.reference_cell_topology(cell_kind).vertices,
        dtype=np.int64,
    )
    dimension = reference.shape[1]
    masks = tuple(
        int(sum(int(value) << axis for axis, value in enumerate(point)))
        for point in reference
    )
    if sorted(masks) != list(range(1 << dimension)):
        raise ValueError(
            "coordinate_enclosure.corner_lattice must contain the vertices of a unit reference box; got another vertex set."
        )
    exponents = tuple(
        tuple((mask >> axis) & 1 for axis in range(dimension))
        for mask in range(1 << dimension)
    )
    return masks, exponents


def rounded_point(point: tuple[ExactScalar, ...], /) -> np.ndarray:
    """Return the nearest, ties-to-even binary64 image of a source point."""
    return np.asarray([float(value) for value in point], dtype=np.float64)


def lattice_basis(
    cell_kind: str, degree: int, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    key = (cell_kind, degree)
    owner = budget.lattice_cache
    basis = owner.get(key)
    with budget.temporary_scope():
        if basis is None:
            dimension = imported_reference_cell.reference_cell_topology(
                cell_kind
            ).dimension
            candidates = (degree + 1) ** dimension
            reserve_polynomial(
                candidates,
                candidates,
                dimension,
                max(degree, 1).bit_length(),
                budget=budget,
            )
            basis = _lattice_basis(cell_kind, degree, budget=budget)
        budget.retain_basis(basis)
        owner[key] = basis
    return basis


def _lattice_basis(
    cell_kind: str, degree: int, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    """Ideal equispaced coordinate basis, with rational rational lattice identity."""

    dimension = imported_reference_cell.reference_cell_topology(cell_kind).dimension
    if degree == 1:
        return _linear_reference_basis(cell_kind, budget=budget)
    variables = axes(dimension, budget=budget)
    if cell_kind in ("triangle", "tetrahedron", "prism") or cell_kind.startswith(
        "simplex:"
    ):
        simplex_dimension = 2 if cell_kind == "prism" else dimension
        barycentric = (
            add(
                constant(1, dimension, budget=budget),
                scale(
                    sum_polynomials(variables[:simplex_dimension], budget=budget),
                    -1,
                    budget=budget,
                ),
                budget=budget,
            ),
            *variables[:simplex_dimension],
        )
        simplex = []
        indices = tuple(
            index
            for index in product(range(degree + 1), repeat=simplex_dimension)
            if sum(index) <= degree
        )
        for index in indices:
            with budget.live_storage() as term_owner:
                with budget.temporary_scope():
                    before_term = budget.temporary_bytes_upper
                    term = constant(1, dimension, budget=budget)
                    term_storage = budget.temporary_bytes_upper - before_term
                term_owner.set_bound(term_storage)
                for variable, exponent in zip(
                    barycentric, (degree - sum(index), *index), strict=True
                ):
                    for offset in range(exponent):
                        with budget.temporary_scope():
                            factor = scale(
                                add(
                                    scale(variable, degree, budget=budget),
                                    constant(-offset, dimension, budget=budget),
                                    budget=budget,
                                ),
                                Fraction(1, offset + 1),
                                budget=budget,
                            )
                            before_product = budget.temporary_bytes_upper
                            term = multiply(term, factor, budget=budget)
                            term_storage = budget.temporary_bytes_upper - before_product
                        term_owner.set_bound(term_storage)
            budget.reserve(0, term_storage)
            simplex.append(term)
        if cell_kind != "prism":
            return tuple(simplex)
        axial = _nodal_axis_exact(
            tuple(Fraction(i, degree) for i in range(degree + 1)), 2, 3, budget=budget
        )
        return tuple(multiply(a, b, budget=budget) for a, b in product(simplex, axial))
    if cell_kind != "pyramid":
        factors = tuple(
            _nodal_axis_exact(
                tuple(Fraction(i, degree) for i in range(degree + 1)),
                axis,
                dimension,
                budget=budget,
            )
            for axis in range(dimension)
        )
        return tuple(
            math_product_polynomials(choice, dimension, budget=budget)
            for choice in product(*factors)
        )
    collapse = add(
        constant(1, 3, budget=budget),
        scale(variables[2], -1, budget=budget),
        budget=budget,
    )
    indices = tuple(
        (i, j, k)
        for i in range(degree + 1)
        for j in range(degree + 1)
        for k in range(degree - max(i, j) + 1)
    )
    modes = tuple(
        math_product_polynomials(
            (
                jacobi(i, 0, 0, 3, budget=budget),
                jacobi(j, 0, 1, 3, budget=budget),
                power(collapse, max(i, j), 3, budget=budget),
                jacobi(k, 2 * max(i, j) + 2, 2, 3, budget=budget),
            ),
            3,
            budget=budget,
        )
        for i, j, k in indices
    )
    nodes = tuple(
        (Fraction(i, degree - k), Fraction(j, degree - k), Fraction(k, degree))
        if k < degree
        else (Fraction(1, 2), Fraction(1, 2), Fraction(1))
        for k in range(degree + 1)
        for i in range(degree - k + 1)
        for j in range(degree - k + 1)
    )
    table = [
        [
            require_rational_reference_scalar(evaluate(mode, point, budget=budget))
            for mode in modes
        ]
        for point in nodes
    ]
    coefficients = solve_exact(
        table,
        [[Fraction(int(i == j)) for j in range(len(nodes))] for i in range(len(nodes))],
    )
    return tuple(
        sum_polynomials(
            tuple(
                scale(mode, row[i], budget=budget)
                for mode, row in zip(modes, coefficients, strict=True)
            ),
            budget=budget,
        )
        for i in range(len(nodes))
    )


def _nodal_axis_exact(
    nodes: tuple[Fraction, ...],
    axis: int,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...]:
    variable = axes(dimension, budget=budget)[axis]
    result = []
    for i, x in enumerate(nodes):
        term = constant(1, dimension, budget=budget)
        for j, y in enumerate(nodes):
            if i != j:
                term = multiply(
                    term,
                    scale(
                        add(
                            variable,
                            constant(-y, dimension, budget=budget),
                            budget=budget,
                        ),
                        1 / (x - y),
                        budget=budget,
                    ),
                    budget=budget,
                )
        result.append(term)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PreparedCoordinateLatticeExecution:
    """Rational static coefficients and physical gradients for numerical tabulation."""

    cell_kind: str
    degree: int
    basis: tuple[PreparedPolynomialCoefficients, ...]
    gradients: tuple[tuple[PreparedPolynomialCoefficients, ...], ...]

    def validate_restored(self) -> None:
        """Validate all retained fields against the rational lattice preparation."""

        if type(self.cell_kind) is not str:
            raise TypeError(
                "PreparedCoordinateLatticeExecution.cell_kind must be a string; "
                f"got {type(self.cell_kind).__name__}"
            )
        if type(self.degree) is not int or self.degree < 0:
            raise ValueError(
                "PreparedCoordinateLatticeExecution.degree must be a nonnegative "
                f"integer; got {self.degree!r}"
            )
        dimension = imported_reference_cell.reference_cell_topology(
            self.cell_kind
        ).dimension
        if type(self.basis) is not tuple or type(self.gradients) is not tuple:
            raise TypeError(
                "PreparedCoordinateLatticeExecution.coefficients must be immutable "
                "tuple banks; got non-tuple bank"
            )
        if len(self.basis) != len(self.gradients):
            raise ValueError(
                "PreparedCoordinateLatticeExecution.gradients must match the basis "
                f"count; got {len(self.gradients)} for {len(self.basis)}"
            )
        if dimension > 0 and self.degree + 1 > len(self.basis):
            raise ValueError(
                "PreparedCoordinateLatticeExecution.basis must contain the "
                "declared degree's nodal basis before rational reconstruction; "
                f"got {len(self.basis)} entries for degree {self.degree}"
            )

        def require_polynomial(
            polynomial: PreparedPolynomialCoefficients, path: str
        ) -> None:
            if type(polynomial) is not tuple:
                raise TypeError(
                    f"{path} must be an immutable coefficient tuple; got non-tuple"
                )
            for term in polynomial:
                if type(term) is not tuple or len(term) != 2:
                    raise ValueError(
                        f"{path} must contain exponent/coefficient pairs; "
                        "got malformed term"
                    )
                exponent, coefficient = term
                if (
                    type(exponent) is not tuple
                    or len(exponent) != dimension
                    or any(type(power) is not int or power < 0 for power in exponent)
                ):
                    raise ValueError(
                        f"{path}.exponent must contain {dimension} nonnegative integers; "
                        f"got {exponent!r}"
                    )
                if type(coefficient) is not Fraction or coefficient == 0:
                    raise ValueError(
                        f"{path}.coefficient must be a nonzero Fraction; "
                        f"got {coefficient!r}"
                    )

        for index, polynomial in enumerate(self.basis):
            require_polynomial(
                polynomial, f"PreparedCoordinateLatticeExecution.basis[{index}]"
            )
        for index, row in enumerate(self.gradients):
            if type(row) is not tuple or len(row) != dimension:
                raise ValueError(
                    "PreparedCoordinateLatticeExecution.gradients must contain "
                    f"{dimension} physical components; got malformed row {index}"
                )
            for axis, polynomial in enumerate(row):
                require_polynomial(
                    polynomial,
                    f"PreparedCoordinateLatticeExecution.gradients[{index}][{axis}]",
                )
        estimate = coordinate_lattice_estimate(self.cell_kind, self.degree)
        with _cell_geometry_dispatch().owned_coordinate_preparation(
            estimate.work_upper, estimate.bytes_upper
        ) as budget:
            expected = prepare_coordinate_lattice_basis(
                self.cell_kind, self.degree, budget=budget
            )
        if self.basis != expected.basis or self.gradients != expected.gradients:
            raise ValueError(
                "PreparedCoordinateLatticeExecution.coefficients must match the "
                "declared rational lattice and physical gradients; got inconsistent bank"
            )


@checked
def prepare_coordinate_lattice_basis(
    cell_kind: str, degree: int, *, budget: CoordinateEnclosureBudget
) -> PreparedCoordinateLatticeExecution:
    """Prepare the rational fallback lattice before entering numerical execution."""

    dimension = imported_reference_cell.reference_cell_topology(cell_kind).dimension
    basis = lattice_basis(cell_kind, degree, budget=budget)
    gradients = []
    for polynomial in basis:
        rows = physical_jacobian((polynomial,), cell_kind, dimension, budget=budget)
        if rows is None:
            raise ValueError(
                "coordinate_enclosure.source must have a resolved rational denominator; got a quotient."
            )
        gradients.append(rows[0])
    terms = sum(len(value) for value in basis) + sum(
        len(value) for row in gradients for value in row
    )
    budget.reserve(terms, 256 + 64 * terms)
    prepared = PreparedCoordinateLatticeExecution(
        cell_kind,
        degree,
        tuple(tuple(value.items()) for value in basis),
        tuple(tuple(tuple(value.items()) for value in row) for row in gradients),
    )
    budget.retain_basis((prepared, prepared.basis, prepared.gradients))
    return prepared


@partial(jax.jit, static_argnames=("cell_kind", "degree", "preparation"))
def evaluate_coordinate_lattice_basis(
    cell_kind: str,
    degree: int,
    points: ArrayLike,
    *,
    preparation: PreparedCoordinateLatticeExecution | None = None,
) -> tuple[Array, Array]:
    """Execute numerical lattice tabulation using precomputed fallback coefficients."""

    reference = jnp.asarray(points, dtype=jnp.float64)
    dimension = imported_reference_cell.reference_cell_topology(cell_kind).dimension
    if reference.ndim != 2 or reference.shape[1] != dimension:
        raise ValueError(
            "coordinate_enclosure.reference_points must have the declared rank-two parameter shape; got incompatible shape."
        )
    if cell_kind in ("interval", "quadrilateral", "hexahedron") or cell_kind.startswith(
        "tensor:"
    ):
        factors, derivatives = [], []
        for axis in range(dimension):
            value, gradient = _evaluate_nodal_axis(degree, reference[:, axis])
            factors.append(value)
            derivatives.append(gradient)
        values = _tensor_jax(tuple(factors))
        gradients = jnp.stack(
            tuple(
                _tensor_jax(
                    tuple(
                        derivatives[j] if i == j else factors[j] for j in range(dimension)
                    )
                )
                for i in range(dimension)
            ),
            axis=-1,
        )
        if degree == 1:
            permutation = np.ravel_multi_index(
                np.asarray(
                    imported_reference_cell.reference_cell_topology(cell_kind).vertices,
                    dtype=np.int64,
                ).T,
                (2,) * dimension,
            )
            values = values[:, permutation]
            gradients = gradients[:, permutation]
        return values, gradients
    if degree > 1 and (
        cell_kind in ("triangle", "tetrahedron", "prism")
        or cell_kind.startswith("simplex:")
    ):
        simplex_dimension = 2 if cell_kind == "prism" else dimension
        values, gradients = _evaluate_simplex_lattice(
            degree, reference[:, :simplex_dimension]
        )
        if cell_kind == "prism":
            axial, derivative_axial = _evaluate_nodal_axis(degree, reference[:, 2])
            gradients = jnp.stack(
                (
                    _tensor_jax((gradients[:, :, 0], axial)),
                    _tensor_jax((gradients[:, :, 1], axial)),
                    _tensor_jax((values, derivative_axial)),
                ),
                axis=-1,
            )
            values = _tensor_jax((values, axial))
        return values, gradients
    parameters = reference
    if cell_kind == "pyramid":
        collapse = 1.0 - reference[:, 2:3]
        safe = jnp.where(collapse != 0.0, collapse, 1.0)
        horizontal = jnp.where(
            collapse != 0.0, (reference[:, :2] - 0.5 * reference[:, 2:3]) / safe, 0.5
        )
        parameters = jnp.concatenate((horizontal, reference[:, 2:3]), axis=1)
        if degree > 1:
            return _evaluate_pyramid_lattice(degree, parameters)
    if preparation is None:
        raise ValueError(
            "coordinate_enclosure.lattice_fallback must receive prepared coefficients; got None."
        )
    if (preparation.cell_kind, preparation.degree) != (cell_kind, degree):
        raise ValueError(
            "coordinate_enclosure.lattice_preparation must match cell kind and degree; got a different lattice."
        )
    values = []
    gradients = []
    for polynomial, row in zip(preparation.basis, preparation.gradients, strict=True):
        values.append(_evaluate_jax_polynomial(polynomial, parameters))
        gradients.append(
            jnp.stack(
                tuple(_evaluate_jax_polynomial(entry, parameters) for entry in row),
                axis=-1,
            )
        )
    return jnp.stack(values, axis=-1), jnp.stack(gradients, axis=1)


def _evaluate_nodal_axis(degree: int, points: Array) -> tuple[Array, Array]:
    """Factored rational lattice evaluation without monomial cancellation."""

    values, derivatives = [], []
    for node in range(degree + 1):
        value, derivative_value = jnp.ones_like(points), jnp.zeros_like(points)
        for other in range(degree + 1):
            if other != node:
                factor = (degree * points - other) / (node - other)
                derivative_value = derivative_value * factor + value * (
                    degree / (node - other)
                )
                value = value * factor
        values.append(value)
        derivatives.append(derivative_value)
    return jnp.stack(values, axis=-1), jnp.stack(derivatives, axis=-1)


def _evaluate_simplex_lattice(degree: int, points: Array) -> tuple[Array, Array]:
    """Evaluate the rational falling-factorial barycentric source in factored form."""

    dimension = points.shape[1]
    barycentric = jnp.concatenate(
        (1.0 - jnp.sum(points, axis=-1, keepdims=True), points), axis=-1
    )
    factors, derivatives = [], []
    value, gradient = jnp.ones_like(barycentric), jnp.zeros_like(barycentric)
    for exponent in range(degree + 1):
        factors.append(value)
        derivatives.append(gradient)
        factor = (degree * barycentric - exponent) / (exponent + 1)
        gradient = gradient * factor + value * (degree / (exponent + 1))
        value = value * factor
    values, gradients = [], []
    for index in product(range(degree + 1), repeat=dimension):
        if sum(index) > degree:
            continue
        exponents = (degree - sum(index), *index)
        selected = tuple(
            factors[exponent][:, axis] for axis, exponent in enumerate(exponents)
        )
        values.append(jnp.prod(jnp.stack(selected, axis=-1), axis=-1))
        partials = tuple(
            derivatives[exponent][:, axis]
            * jnp.prod(
                jnp.stack(
                    tuple(value for other, value in enumerate(selected) if other != axis),
                    axis=-1,
                ),
                axis=-1,
            )
            for axis, exponent in enumerate(exponents)
        )
        gradients.append(
            jnp.stack(
                tuple(partials[axis + 1] - partials[0] for axis in range(dimension)),
                axis=-1,
            )
        )
    return jnp.stack(values, axis=-1), jnp.stack(gradients, axis=1)


def _tensor_jax(factors: tuple[Array, ...]) -> Array:
    result = factors[0]
    for factor in factors[1:]:
        result = (result[:, :, None] * factor[:, None, :]).reshape((factor.shape[0], -1))
    return result


def _evaluate_jax_polynomial(
    polynomial: PreparedPolynomialCoefficients, points: Array
) -> Array:

    if not polynomial:
        return jnp.zeros((points.shape[0],), dtype=jnp.float64)
    coefficients = jnp.asarray(
        tuple(float(value) for _, value in polynomial), dtype=jnp.float64
    )
    # Exponents are static source data: Python integer powers lower to rational
    # repeated multiplication and stay valid under strict dtype promotion.
    monomials = tuple(
        jnp.prod(
            jnp.stack(
                tuple(points[:, axis] ** exponent for axis, exponent in enumerate(index)),
                axis=-1,
            ),
            axis=-1,
        )
        for index, _ in polynomial
    )
    return jnp.stack(monomials, axis=-1) @ coefficients


def _divide_monomial(
    numerator: Polynomial,
    exponent: tuple[int, ...],
    coefficient: ExactScalar,
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial | None:
    """Divide independent terms without constructing a long-division remainder."""
    dimension = len(exponent)
    budget.reserve(dimension * len(numerator))
    if any(
        any(a < b for a, b in zip(index, exponent, strict=True)) for index in numerator
    ):
        return None
    numerator_bits, denominator_bits = coefficient_profile((numerator,), budget=budget)
    reserve_polynomial(
        len(numerator),
        len(numerator),
        dimension,
        numerator_bits + denominator_bits + _scalar_bits(coefficient),
        budget=budget,
    )
    # Long division visits maximal monomials first; retain that ordering.
    return {
        tuple(a - b for a, b in zip(index, exponent, strict=True)): _scalar_quotient(
            numerator[index], coefficient, budget=budget
        )
        for index in sorted(numerator, reverse=True)
    }


def divide_polynomial(
    numerator: Polynomial, denominator: Polynomial, *, budget: CoordinateEnclosureBudget
) -> Polynomial | None:
    if not denominator:
        raise ValueError(
            "coordinate_enclosure.denominator must be nonzero; got zero support."
        )
    if not numerator:
        return {}
    leading = max(denominator)
    coefficient = denominator[leading]
    if len(denominator) == 1:
        return _divide_monomial(numerator, leading, coefficient, budget=budget)
    if any(a < b for a, b in zip(max(numerator), leading, strict=True)):
        return None
    numerator_bits, numerator_denominator = coefficient_profile(
        (numerator,), exact_denominator=True, budget=budget
    )
    denominator_bits, denominator_common = coefficient_profile(
        (denominator,), exact_denominator=True, budget=budget
    )
    # Profiles bound reduced numerators; the persistent remainder bound uses
    # integer numerators over its common denominator instead.
    numerator_bits += numerator_denominator.bit_length()
    denominator_bits += denominator_common.bit_length()
    quotient: Polynomial = {}
    quotient_storage = 256
    with budget.temporary_scope(), budget.live_storage() as remainder_storage:
        # The copied dictionary shares its input coefficients. Each following
        # remainder owns the newly admitted addition buffer, not its history.
        remainder_storage.set_bound(256 + 64 * len(numerator))
        remainder = dict(numerator)
        budget.reserve(0, quotient_storage)
        while remainder:
            index = max(remainder)
            if any(a < b for a, b in zip(index, leading, strict=True)):
                return None
            exponent = tuple(a - b for a, b in zip(index, leading, strict=True))
            value = _scalar_quotient(remainder[index], coefficient, budget=budget)
            _reserve_coefficient_work(
                (remainder, denominator),
                (value,),
                work=2 * len(denominator),
                count=len(remainder) + len(denominator),
                degree=2,
                budget=budget,
            )
            total = quotient.get(exponent, Fraction(0)) + value
            total_numerator, total_denominator = _scalar_profile(total)
            bits = max(total_numerator, total_denominator.bit_length())
            entry_storage = (
                512 + 32 * len(exponent) + 2 * declared_integer_storage_bytes(bits)
            )
            budget.reserve(0, entry_storage)
            quotient_storage += entry_storage
            quotient[exponent] = total
            # Subtract the shifted divisor directly. Long division owns this
            # private remainder, so materializing and re-profiling a monomial
            # product, its negation and a replacement remainder adds
            # no evidence. Bound the next coefficients from the earned input
            # profiles and the current rational quotient coefficient.
            value_bits, value_denominator = _scalar_profile(value)
            common = math.lcm(
                numerator_denominator, denominator_common * value_denominator
            )
            next_bits = (
                max(
                    numerator_bits + (common // numerator_denominator).bit_length(),
                    denominator_bits
                    + value_bits
                    + (common // (denominator_common * value_denominator)).bit_length(),
                )
                + 1
            )
            with budget.temporary_scope():
                before_add = budget.temporary_bytes_upper
                reserve_polynomial(
                    2 * len(denominator),
                    len(remainder) + len(denominator),
                    len(exponent),
                    next_bits + common.bit_length(),
                    budget=budget,
                )
                for power_index, divisor_value in denominator.items():
                    shifted = tuple(map(operator.add, exponent, power_index))
                    difference = (
                        remainder.get(shifted, Fraction(0)) - value * divisor_value
                    )
                    if difference:
                        remainder[shifted] = difference
                    else:
                        remainder.pop(shifted, None)
                next_storage = budget.temporary_bytes_upper - before_add
            numerator_bits, numerator_denominator = next_bits, common
            remainder_storage.set_bound(next_storage)
        # Only the quotient escapes; products and superseded remainders end here.
    budget.reserve(0, quotient_storage)
    return quotient


def _restrict_arguments(
    coordinates: tuple[Polynomial, ...],
    source_kind: str,
    arguments: tuple[Polynomial, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...] | None:
    """Compose source chart expressions with source-reference arguments.

    Pyramid sources are homogenized in ``1 - z`` and divided exactly; ``None``
    reports a collapsed denominator that does not cancel.
    """
    if source_kind != "pyramid":
        return tuple(compose(value, arguments, budget=budget) for value in coordinates)
    height = arguments[2]
    denominator = add(
        constant(1, dimension, budget=budget),
        scale(height, -1, budget=budget),
        budget=budget,
    )
    horizontal = (
        add(arguments[0], scale(height, Fraction(-1, 2), budget=budget), budget=budget),
        add(arguments[1], scale(height, Fraction(-1, 2), budget=budget), budget=budget),
    )
    result = []
    for polynomial in coordinates:
        degree = max((index[0] + index[1] for index in polynomial), default=0)
        numerator = sum_polynomials(
            tuple(
                scale(
                    math_product_polynomials(
                        (
                            power(horizontal[0], index[0], dimension, budget=budget),
                            power(horizontal[1], index[1], dimension, budget=budget),
                            power(height, index[2], dimension, budget=budget),
                            power(
                                denominator,
                                degree - index[0] - index[1],
                                dimension,
                                budget=budget,
                            ),
                        ),
                        dimension,
                        budget=budget,
                    ),
                    value,
                    budget=budget,
                )
                for index, value in polynomial.items()
            ),
            budget=budget,
        )
        quotient = divide_polynomial(
            numerator, power(denominator, degree, dimension, budget=budget), budget=budget
        )
        if quotient is None:
            return None
        result.append(quotient)
    return tuple(result)


def restrict_coordinates(
    coordinates: tuple[Polynomial, ...],
    source_kind: str,
    origin: np.ndarray,
    matrix: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...] | None:
    """Rational physical reference restriction; cancel removable apex denominators."""
    return _restrict_arguments(
        coordinates,
        source_kind,
        affine_arguments(origin, matrix, budget=budget),
        matrix.shape[1],
        budget=budget,
    )


def chart_arguments(
    cell_kind: str, dimension: int, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    """Physical reference coordinates as expressions on the cell's chart.

    Pyramid expressions use the collapsed cube; other cells use their reference.
    """
    variables = axes(dimension, budget=budget)
    if cell_kind != "pyramid":
        return variables
    u, v, w = variables
    collapse = add(
        constant(1, 3, budget=budget), scale(w, -1, budget=budget), budget=budget
    )
    return (
        add(
            multiply(u, collapse, budget=budget),
            scale(w, Fraction(1, 2), budget=budget),
            budget=budget,
        ),
        add(
            multiply(v, collapse, budget=budget),
            scale(w, Fraction(1, 2), budget=budget),
            budget=budget,
        ),
        w,
    )


def restrict_chart_coordinates(
    coordinates: tuple[Polynomial, ...],
    source_kind: str,
    target_kind: str,
    origin: np.ndarray,
    matrix: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...] | None:
    """Rational restriction from the source chart to the target chart.

    The target chart is inserted before cancelling pyramid denominators, so a
    restriction that is polynomial on the target chart (for example an
    apex-sharing sub-pyramid) is found even when it is rational on the target
    physical reference. ``None`` reports a rational restriction.
    """
    dimension = matrix.shape[1]
    chart = chart_arguments(target_kind, dimension, budget=budget)
    return _restrict_arguments(
        coordinates,
        source_kind,
        tuple(
            compose(term, chart, budget=budget)
            for term in affine_arguments(origin, matrix, budget=budget)
        ),
        dimension,
        budget=budget,
    )


def _linear_reference_basis(
    cell_kind: str, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:

    topology = imported_reference_cell.reference_cell_topology(cell_kind)
    dimension = topology.dimension
    variables = axes(dimension, budget=budget)
    if cell_kind in ("triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        return (
            add(
                constant(1, dimension, budget=budget),
                scale(sum_polynomials(variables, budget=budget), -1, budget=budget),
                budget=budget,
            ),
            *variables,
        )
    if cell_kind == "prism":
        x, y, z = variables
        triangle = (
            add(
                constant(1, 3, budget=budget),
                scale(add(x, y, budget=budget), -1, budget=budget),
                budget=budget,
            ),
            x,
            y,
        )
        collapse = add(
            constant(1, 3, budget=budget), scale(z, -1, budget=budget), budget=budget
        )
        return tuple(
            multiply(value, collapse, budget=budget) for value in triangle
        ) + tuple(multiply(value, z, budget=budget) for value in triangle)
    if cell_kind == "pyramid":
        u, v, w = variables
        collapse = add(
            constant(1, 3, budget=budget), scale(w, -1, budget=budget), budget=budget
        )
        factors = tuple(
            math_product_polynomials(
                tuple(
                    variable
                    if int(value)
                    else add(
                        constant(1, 3, budget=budget),
                        scale(variable, -1, budget=budget),
                        budget=budget,
                    )
                    for variable, value in zip((u, v), vertex[:2], strict=True)
                ),
                3,
                budget=budget,
            )
            for vertex in topology.vertices[:4]
        )
        return tuple(multiply(value, collapse, budget=budget) for value in factors) + (w,)
    return tuple(
        math_product_polynomials(
            tuple(
                variable
                if int(value)
                else add(
                    constant(1, dimension, budget=budget),
                    scale(variable, -1, budget=budget),
                    budget=budget,
                )
                for variable, value in zip(variables, vertex, strict=True)
            ),
            dimension,
            budget=budget,
        )
        for vertex in topology.vertices
    )


@host_table_cache(max_entries=256, max_bytes=1048576)
def _pyramid_execution_table(
    degree: int,
) -> tuple[tuple[tuple[int, int, int], ...], np.ndarray]:
    """Prepare the factored modal execution of the ideal rational lattice."""

    indices = tuple(
        (i, j, k)
        for i in range(degree + 1)
        for j in range(degree + 1)
        for k in range(degree - max(i, j) + 1)
    )
    nodes = np.asarray(
        [
            ((i + 0.5 * k) / degree, (j + 0.5 * k) / degree, k / degree)
            for k in range(degree + 1)
            for i in range(degree - k + 1)
            for j in range(degree - k + 1)
        ],
        dtype=np.float64,
    )
    table = hybrid_reference_owner.pyramid_modal_tabulation(nodes, indices)[0]
    coefficients = np.linalg.solve(table, np.eye(table.shape[0], dtype=np.float64))
    coefficients.flags.writeable = False
    return indices, coefficients


def _evaluate_pyramid_lattice(degree: int, parameters: Array) -> tuple[Array, Array]:

    indices, coefficients = _pyramid_execution_table(degree)
    u, v, w = (parameters[:, axis] for axis in range(3))
    collapse = 1.0 - w
    horizontal = tuple(
        (
            jacobi_owner.jacobi_jax(i, 0, 0, u),
            jacobi_owner.jacobi_jax(i, 0, 0, v),
            jacobi_owner.jacobi_derivative_jax(i, 0, 0, u),
            jacobi_owner.jacobi_derivative_jax(i, 0, 0, v),
        )
        for i in range(degree + 1)
    )
    vertical = {
        (m, k): (
            jacobi_owner.jacobi_jax(k, 2 * m + 2, 0, w),
            jacobi_owner.jacobi_derivative_jax(k, 2 * m + 2, 0, w),
        )
        for m in range(degree + 1)
        for k in range(degree - m + 1)
    }
    values = []
    gradients = []
    for i, j, k in indices:
        maximum = max(i, j)
        first, _, first_derivative, _ = horizontal[i]
        _, second, _, second_derivative = horizontal[j]
        height, height_derivative = vertical[maximum, k]
        factor = collapse**maximum
        reduced = collapse ** max(maximum - 1, 0)
        values.append(first * second * factor * height)
        dx = first_derivative * second * reduced * height
        dy = first * second_derivative * reduced * height
        dz = (
            first_derivative * (u - 0.5) * second
            + first * second_derivative * (v - 0.5)
            - maximum * first * second
        ) * reduced * height + first * second * factor * height_derivative
        gradients.append(jnp.stack((dx, dy, dz), axis=-1))
    conversion = jnp.asarray(coefficients, dtype=jnp.float64)
    value_table = jnp.stack(values, axis=-1) @ conversion
    gradient_table = jnp.stack(
        tuple(
            jnp.stack(tuple(value[:, axis] for value in gradients), axis=-1) @ conversion
            for axis in range(3)
        ),
        axis=-1,
    )
    return value_table, gradient_table


def bernstein_node_count(polynomial: Polynomial, domain: str, dimension: int) -> int:
    """Count the full control net before allocating its coefficient table."""
    reference = parse(domain, BernsteinDomain, "domain")
    match reference:
        case "simplex":
            return math.comb(
                max((sum(index) for index in polynomial), default=0) + dimension,
                dimension,
            )
        case "prism":
            degree = max((sum(index[:2]) for index in polynomial), default=0)
            axial = max((index[2] for index in polynomial), default=0)
            return math.comb(degree + 2, 2) * (axial + 1)
        case "box":
            return math.prod(
                max((index[axis] for index in polynomial), default=0) + 1
                for axis in range(dimension)
            )
        case _:
            assert_never(reference)


def coordinate_source_signature(element: CellGeometryElement) -> dict[str, object]:
    """Bind source semantics, independently of cached element IDs."""

    result: dict[str, object] = {
        "declared_element": element.element_id,
        "cell_kind": element.cell_kind,
        "local_dof_count": element.local_dof_count,
    }
    geometry = _cell_geometry_dispatch()
    match element:
        case geometry.BarycentricCellGeometryElement():
            result["operator"] = "affine-p1-full-barycentric-action"
            result["source"] = coordinate_source_signature(element.source_element)
            result["target_dimension"] = element.topological_dimension
            result["barycentric_weights"] = tuple(
                tuple(row) for row in np.asarray(element.barycentric_weights).tolist()
            )
        case geometry.LayerColumnCellGeometryElement():
            result["operator"] = "endpoint-profile-with-authored-corner-corrections"
            result["profile"] = coordinate_source_signature(element.wall_element)
            result["corners"] = coordinate_source_signature(element.corner_element)
            result["profile_axes"], result["station_axis"] = (0, 1), element.station_axis
            result["control_banks"] = (
                "bottom_profile",
                "top_profile",
                "actual_bottom",
                "actual_top",
                "profile_bottom_corners",
                "profile_top_corners",
            )
        case geometry.SplineCellGeometryElement():
            result["operator"] = "original-rational-bspline-span"
            result["source_id"], result["source_revision"] = (
                element.source_id,
                element.source_revision,
            )
            result["degrees"], result["spans"] = (
                (element.u_degree, element.v_degree),
                element.span_indices,
            )
            result["knots"] = (
                tuple(np.asarray(element.u_knots).tolist()),
                tuple(np.asarray(element.v_knots).tolist()),
            )
            result["weights"] = tuple(
                tuple(row) for row in np.asarray(element.weights).tolist()
            )
        case (
            geometry.PolynomialComposedCellGeometryElement()
            | geometry.RationalComposedCellGeometryElement()
        ):
            result["operator"] = (
                "exact-rational-collapsed-pyramid-reference-composition"
                if isinstance(
                    element, _cell_geometry_dispatch().RationalComposedCellGeometryElement
                )
                else "exact-polynomial-reference-composition"
            )
            result["source"] = coordinate_source_signature(element.source_element)
            result["chart"] = coordinate_source_signature(element.chart_element)
            result["chart_coefficients"] = element.chart_coefficients
            result["target_dimension"] = element.topological_dimension
        case geometry.RestrictedCellGeometryElement():
            result["operator"] = "exact-reference-restriction"
            result["source"] = coordinate_source_signature(element.source_element)
            result["target_dimension"] = element.topological_dimension
            result["matrix"] = tuple(
                tuple(row) for row in np.asarray(element.matrix).tolist()
            )
            result["offset"] = tuple(np.asarray(element.offset).tolist())
        case FiniteElementSpec():
            _finite_element_source_signature(element, result)
        case _:
            result["operator"] = "vertex-coordinate-source"
    return result


def _finite_element_source_signature(
    element: FiniteElementSpec, result: dict[str, object], /
) -> None:
    """Append reference execution provenance to the source identity payload."""
    result.update(
        {
            "family": element.family,
            "degree": element.degree,
            "mapping": element.mapping,
            "representation": element.representation,
            "value_shape": element.value_shape,
            "entity_dofs": element.entity_dofs,
        }
    )
    tabulator = element.tabulator
    geometry = _cell_geometry_dispatch()
    match tabulator:
        case geometry.SweptCoordinateTabulator() if (
            type(tabulator) is geometry.SweptCoordinateTabulator
        ):
            result["operator"] = "exact-profile-times-linear-station"
            result["profile"] = coordinate_source_signature(tabulator.source)
            result["station_axis"] = 2
            result["station_degree"] = 1
            result["endpoint_order"] = (0, 1)
        case geometry.CoordinateTabulator():
            result["operator"] = "exact-rational-coordinate-lattice"
            result["lattice"] = (
                tabulator.cell_kind,
                tabulator.degree,
                tabulator.source_basis_semantics,
            )
        case _:
            owner = owning_tabulator_source(tabulator)
            match owner:
                case FiniteElementSpec():
                    result["operator"] = "reference-source-forwarding"
                    result["source"] = coordinate_source_signature(owner)
                case imported_fem__form_elements.FormBasis():
                    result["operator"] = "scalar-form-source"
                    result["source_order"] = (
                        owner.dimension,
                        owner.form_degree,
                        owner.order,
                        owner.family,
                        owner.twist,
                        owner.exponents,
                        owner.dof_labels,
                    )
                case imported_fem__high_order.SimplexNodalFamily():
                    result["operator"] = "simplex-bernstein-source"
                    result["source_order"] = (
                        owner.cell_kind,
                        owner.order,
                        owner.multiindices,
                    )
                case imported_fem__high_order.ReferenceNodalFamily():
                    result["operator"] = "tensor-nodal-source"
                    result["source_order"] = (owner.cell_kind, owner.orders)
                case hybrid_reference_owner.HybridReferenceFamily():
                    result["operator"] = "hybrid-source"
                    result["source_order"] = (
                        owner.cell_kind,
                        owner.orders,
                        owner.modal_indices,
                        owner.basis_permutation,
                    )
                case None:
                    result["operator"] = (
                        "builtin-reference-source"
                        if tabulator is None
                        else "unsupported-tabulator"
                    )
                case _:
                    assert_never(owner)


def owning_tabulator_source(
    tabulator: object,
) -> (
    FiniteElementSpec
    | imported_fem__form_elements.FormBasis
    | imported_fem__high_order.SimplexNodalFamily
    | imported_fem__high_order.ReferenceNodalFamily
    | hybrid_reference_owner.HybridReferenceFamily
    | None
):
    """Unwrap a declared immutable reference-owner/method pair."""

    if type(tabulator) is imported_fem__form_elements.ProxyTabulator:
        match tabulator.basis:
            case imported_fem__form_elements.FormBasis() as basis:
                return basis
            case _:
                return None

    if (
        isinstance(tabulator, reference_types_owner.FiniteElementTabulator)
        and type(tabulator) is reference_types_owner.FiniteElementTabulator
    ):
        source = tabulator.source
        match source:
            case (
                FiniteElementSpec()
                | imported_fem__high_order.ReferenceNodalFamily()
                | imported_fem__high_order.SimplexNodalFamily()
            ):
                if (
                    type(source)
                    in (
                        FiniteElementSpec,
                        imported_fem__high_order.ReferenceNodalFamily,
                        imported_fem__high_order.SimplexNodalFamily,
                    )
                    and tabulator.method == "tabulate"
                ):
                    return source
            case hybrid_reference_owner.HybridReferenceFamily():
                if (
                    type(source) is hybrid_reference_owner.HybridReferenceFamily
                    and tabulator.method == "tabulate"
                ):
                    return source
            case _:
                return None
        return None
    if isinstance(tabulator, _BoundSourceTabulator):
        source = tabulator.__self__
        if (
            type(source) is imported_fem__high_order.SimplexNodalFamily
            and tabulator.__func__ is imported_fem__high_order.SimplexNodalFamily.tabulate
        ):
            return source
        if (
            type(source) is imported_fem__high_order.ReferenceNodalFamily
            and tabulator.__func__
            is imported_fem__high_order.ReferenceNodalFamily.tabulate
        ):
            return source
        if (
            type(source) is hybrid_reference_owner.HybridReferenceFamily
            and tabulator.__func__
            is hybrid_reference_owner.HybridReferenceFamily.tabulate
        ):
            return source
        if (
            type(source) is FiniteElementSpec
            and tabulator.__func__ is FiniteElementSpec.tabulate
        ):
            return source
    return None


def rational_expression(
    numerator: Polynomial, denominator: Polynomial, *, budget: CoordinateEnclosureBudget
) -> Expression:
    """Cancel a removable denominator exactly, retaining a quotient."""
    if numerator and denominator:
        dimension = len(next(iter(denominator)))
        budget.reserve(len(numerator) + len(denominator))
        shift = tuple(
            min(
                min(index[axis] for index in numerator),
                min(index[axis] for index in denominator),
            )
            for axis in range(dimension)
        )
        if any(shift):
            numerator_bits, denominator_bits = coefficient_profile(
                (numerator, denominator), budget=budget
            )
            reserve_polynomial(
                len(numerator) + len(denominator),
                len(numerator) + len(denominator),
                dimension,
                numerator_bits + denominator_bits,
                budget=budget,
            )
            numerator = {
                tuple(a - b for a, b in zip(index, shift, strict=True)): value
                for index, value in numerator.items()
            }
            denominator = {
                tuple(a - b for a, b in zip(index, shift, strict=True)): value
                for index, value in denominator.items()
            }
    quotient = divide_polynomial(numerator, denominator, budget=budget)
    return RationalPolynomial(numerator, denominator) if quotient is None else quotient


def expression_parts(
    value: Expression, dimension: int, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, Polynomial]:
    return (
        (value.numerator, value.denominator)
        if isinstance(value, RationalPolynomial)
        else (value, constant(1, dimension, budget=budget))
    )


def expression_add(
    first: Expression, second: Expression, *, budget: CoordinateEnclosureBudget
) -> Expression:
    match first, second:
        case (RationalPolynomial() as rational, _) | (
            _,
            RationalPolynomial() as rational,
        ):
            pass
        case dict() as first, dict() as second:
            return add(first, second, budget=budget)
        case _:
            raise TypeError(
                "coordinate_enclosure.expression_add.operands must be polynomial expressions; got unsupported operands."
            )
    dimension = len(next(iter(rational.denominator)))
    a, b = expression_parts(first, dimension, budget=budget)
    c, d = expression_parts(second, dimension, budget=budget)
    if b == d:
        return rational_expression(add(a, c, budget=budget), b, budget=budget)
    quotient = divide_polynomial(b, d, budget=budget)
    if quotient is not None:
        return rational_expression(
            add(a, multiply(c, quotient, budget=budget), budget=budget), b, budget=budget
        )
    quotient = divide_polynomial(d, b, budget=budget)
    if quotient is not None:
        return rational_expression(
            add(multiply(a, quotient, budget=budget), c, budget=budget), d, budget=budget
        )
    return rational_expression(
        add(multiply(a, d, budget=budget), multiply(c, b, budget=budget), budget=budget),
        multiply(b, d, budget=budget),
        budget=budget,
    )


def expression_scale(
    value: Expression,
    coefficient: ExactScalar | int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Expression:
    if isinstance(value, RationalPolynomial):
        return rational_expression(
            scale(value.numerator, coefficient, budget=budget),
            value.denominator,
            budget=budget,
        )
    return scale(value, coefficient, budget=budget)


def expression_multiply(
    first: Expression, second: Expression, *, budget: CoordinateEnclosureBudget
) -> Expression:
    match first, second:
        case (RationalPolynomial() as rational, _) | (
            _,
            RationalPolynomial() as rational,
        ):
            pass
        case dict() as first, dict() as second:
            return multiply(first, second, budget=budget)
        case _:
            raise TypeError(
                "coordinate_enclosure.expression_multiply.operands must be polynomial expressions; got unsupported operands."
            )
    dimension = len(next(iter(rational.denominator)))
    a, b = expression_parts(first, dimension, budget=budget)
    c, d = expression_parts(second, dimension, budget=budget)
    return rational_expression(
        multiply(a, c, budget=budget), multiply(b, d, budget=budget), budget=budget
    )


def expression_sum(
    values: tuple[Expression, ...], *, budget: CoordinateEnclosureBudget
) -> Expression:
    result: Expression = {}
    for value in values:
        result = expression_add(result, value, budget=budget)
    return result


def expression_linear_combinations(
    values: tuple[Expression, ...],
    weight_bank: tuple[tuple[ExactScalar, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...]:
    """Share polynomial bank preparation; retain rational arithmetic."""
    if any(len(values) != len(weights) for weights in weight_bank):
        raise ValueError(
            "coordinate_enclosure.weights must match the expression bank; got unequal row counts."
        )
    polynomials: list[Polynomial] = []
    for value in values:
        if isinstance(value, RationalPolynomial):
            return tuple(
                expression_sum(
                    tuple(
                        expression_scale(entry, weight, budget=budget)
                        for entry, weight in zip(values, weights, strict=True)
                    ),
                    budget=budget,
                )
                for weights in weight_bank
            )
        polynomials.append(value)
    return linear_combinations(tuple(polynomials), weight_bank, budget=budget)


def _rational_derivative(
    value: RationalPolynomial,
    axis: int,
    numerator_profile: tuple[int, int],
    denominator: Mapping[tuple[int, ...], ExactScalar],
    denominator_derivative: Polynomial,
    squared_denominator: Polynomial,
    *,
    budget: CoordinateEnclosureBudget,
) -> Expression:
    """Apply the quotient rule using the prepared denominator operands."""
    first = multiply(
        _derivative_profiled(value.numerator, axis, numerator_profile, budget=budget),
        denominator,
        budget=budget,
    )
    second = _multiply_with_first_profile(
        value.numerator,
        denominator_derivative,
        numerator_profile,
        budget=budget,
    )
    if not first:
        numerator = scale(second, -1, budget=budget)
    elif not second:
        numerator = first
    else:
        terms = (first, second)
        numerator_bits, denominator_bits = coefficient_profile(terms, budget=budget)
        active_terms = len(first) + len(second)
        numerator = _linear_combination_profiled(
            terms,
            (Fraction(1), Fraction(-1)),
            dimension=len(next(iter(first))),
            coefficient_bits=numerator_bits
            + denominator_bits
            + 2
            + max(active_terms, 1).bit_length(),
            active_terms=active_terms,
            scaled_terms=len(second),
            budget=budget,
        )
    return rational_expression(
        numerator,
        squared_denominator,
        budget=budget,
    )


def expression_derivative(
    value: Expression, axis: int, *, budget: CoordinateEnclosureBudget
) -> Expression:
    match value:
        case RationalPolynomial():
            return _rational_derivative(
                value,
                axis,
                coefficient_profile(
                    (value.numerator,), exact_denominator=True, budget=budget
                ),
                value.denominator,
                derivative(value.denominator, axis, budget=budget),
                multiply(value.denominator, value.denominator, budget=budget),
                budget=budget,
            )
        case dict():
            return derivative(value, axis, budget=budget)
        case invalid:
            assert_never(invalid)


def expression_compose(
    value: Expression,
    arguments: tuple[Expression, ...] | PreparedPolynomialArguments,
    *,
    budget: CoordinateEnclosureBudget,
) -> Expression:
    return ExpressionComposition(arguments, budget=budget)(value, budget=budget)


def _expression_evaluate_or_none(
    value: Expression,
    point: tuple[ExactScalar, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactScalar | None:
    if isinstance(value, RationalPolynomial):
        denominator = evaluate(value.denominator, point, budget=budget)
        if not denominator:
            return None
        return _scalar_quotient(
            evaluate(value.numerator, point, budget=budget), denominator, budget=budget
        )
    return evaluate(value, point, budget=budget)


@checked
def expression_evaluate(
    value: Expression,
    point: tuple[ExactScalar, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactScalar:
    """Evaluate an expression, refusing a zero rational denominator."""
    result = _expression_evaluate_or_none(value, point, budget=budget)
    if result is None:
        raise RationalEnclosureError(
            "coordinate_enclosure.evaluation must have a resolved chart singularity; got a zero denominator."
        )
    return result


@checked
def expression_evaluate_or_none(
    value: Expression,
    point: tuple[ExactScalar, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactScalar | None:
    """Evaluate an expression or return None for a zero rational denominator.

    Resource refusal and other invalid-input errors propagate unchanged.
    """
    return _expression_evaluate_or_none(value, point, budget=budget)


@overload
def expression_control_net(
    value: Mapping[tuple[int, ...], Fraction],
    domain: str,
    degrees: tuple[int, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, ...]: ...


@overload
def expression_control_net(
    value: Mapping[tuple[int, ...], ExactScalar],
    domain: str,
    degrees: tuple[int, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[ExactScalar, ...]: ...


def expression_control_net(
    value: Mapping[tuple[int, ...], ExactScalar],
    domain: str,
    degrees: tuple[int, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[ExactScalar, ...]:
    reference = parse(domain, BernsteinDomain, "domain")
    numerator, denominator = coefficient_profile((value,), budget=budget)
    visits, nodes, entries = _bernstein_support_profile(
        value, reference, degrees, dimension
    )
    reserve_polynomial(
        visits,
        nodes + entries,
        dimension,
        numerator
        + denominator
        + 2
        * dimension
        * max(degrees, default=0)
        * max(max(degrees, default=0) + 1, 1).bit_length(),
        budget=budget,
    )
    _reserve_coefficient_work((value,), work=visits, count=nodes + entries, budget=budget)
    lattice = _bernstein_lattice(reference, degrees, dimension, budget=budget)
    zero = (0,) * dimension
    origin = value.get(zero)
    if origin is None:
        origin = Fraction(0)
    controls = [origin] * len(lattice)
    for index, coefficient in value.items():
        if index == zero:
            continue
        denominator, entries = _bernstein_weights(
            reference, degrees, dimension, index, budget=budget
        )
        for position, numerator in entries:
            controls[position] += coefficient * Fraction(numerator, denominator)
    return tuple(controls)


def _common_control_degrees(
    indices: tuple[tuple[int, ...], ...], domain: str, dimension: int
) -> tuple[int, ...]:
    """Compute shared numerator/denominator degrees before domain admission."""
    match domain:
        case "simplex":
            return (max((sum(index) for index in indices), default=0),)
        case "prism":
            return (
                max((sum(index[:2]) for index in indices), default=0),
                max((index[2] for index in indices), default=0),
            )
        case _:
            return tuple(
                max((index[axis] for index in indices), default=0)
                for axis in range(dimension)
            )


def _expression_bernstein_coefficients_or_none(
    value: Expression, domain: str, dimension: int, *, budget: CoordinateEnclosureBudget
) -> tuple[ExactScalar, ...] | None:
    """Prepare polynomial controls or sign-definite rational control ratios."""
    if not isinstance(value, RationalPolynomial):
        return bernstein_coefficients(value, domain, dimension, budget=budget)
    indices = tuple(value.numerator) + tuple(value.denominator)
    degrees = _common_control_degrees(indices, domain, dimension)
    numerator = expression_control_net(
        value.numerator, domain, degrees, dimension, budget=budget
    )
    denominator = expression_control_net(
        value.denominator, domain, degrees, dimension, budget=budget
    )
    if max(denominator) < 0:
        numerator = tuple(-coefficient for coefficient in numerator)
        denominator = tuple(-coefficient for coefficient in denominator)
    if (
        min(denominator) < 0
        or not any(denominator)
        or any(a and not b for a, b in zip(numerator, denominator, strict=True))
    ):
        return None
    return tuple(
        _scalar_quotient(a, b, budget=budget)
        for a, b in zip(numerator, denominator, strict=True)
        if b
    )


@checked
def expression_bernstein_coefficients(
    value: Expression, domain: str, dimension: int, *, budget: CoordinateEnclosureBudget
) -> tuple[ExactScalar, ...]:
    """Return polynomial controls or bounding rational control ratios.

    The common-degree rational net uses denominator-weighted Bernstein basis
    functions. A zero denominator control is omitted only when its numerator
    control also vanishes; unresolved denominator controls are refused.
    """
    result = _expression_bernstein_coefficients_or_none(
        value, domain, dimension, budget=budget
    )
    if result is None:
        raise RationalEnclosureError(
            "coordinate_enclosure.denominator must have a sign-definite Bernstein enclosure; got controls of unresolved sign."
        )
    return result


@checked
def expression_bernstein_coefficients_or_none(
    value: Expression, domain: str, dimension: int, *, budget: CoordinateEnclosureBudget
) -> tuple[ExactScalar, ...] | None:
    """Return bounding controls or None for unresolved denominator controls.

    Resource refusal and other invalid-input errors propagate unchanged.
    """
    return _expression_bernstein_coefficients_or_none(
        value, domain, dimension, budget=budget
    )


def expression_bounds(
    value: Expression, domain: str, dimension: int, *, budget: CoordinateEnclosureBudget
) -> tuple[float, float]:
    coefficients = expression_bernstein_coefficients(
        value, domain, dimension, budget=budget
    )
    return _scalar_endpoint(
        min(coefficients), upper=False, budget=budget
    ), _scalar_endpoint(max(coefficients), upper=True, budget=budget)


def expression_determinant(
    matrix: tuple[tuple[Expression, ...], ...],
    *,
    variable_dimension: int | None = None,
    budget: CoordinateEnclosureBudget,
) -> Expression:

    dimension = len(matrix)
    variables = dimension if variable_dimension is None else variable_dimension
    if isinstance(variables, bool) or not isinstance(variables, (int, np.integer)):
        raise TypeError(
            "coordinate_enclosure.variable_dimension must be an integer; got another scalar kind."
        )
    if variables < 0 or any(len(row) != dimension for row in matrix):
        raise ValueError(
            "coordinate_enclosure.determinant must have a square matrix and nonnegative variable dimension; got incompatible layout."
        )
    for row in matrix:
        for value in row:
            numerator, denominator = expression_parts(value, variables, budget=budget)
            if any(
                len(index) != variables
                for polynomial in (numerator, denominator)
                for index in polynomial
            ):
                raise ValueError(
                    "coordinate_enclosure.determinant_entries must share declared parameter axes; got unequal exponent ranks."
                )
    result: Expression = {}
    for permutation in imported_itertools.permutations(range(dimension)):
        if any(not matrix[row][column] for row, column in enumerate(permutation)):
            continue
        term: Expression = constant(1, variables, budget=budget)
        for row, column in enumerate(permutation):
            term = expression_multiply(term, matrix[row][column], budget=budget)
        sign = (-1) ** sum(
            permutation[i] > permutation[j]
            for i in range(dimension)
            for j in range(i + 1, dimension)
        )
        result = expression_add(
            result, expression_scale(term, sign, budget=budget), budget=budget
        )
    return result


def _homogeneous_jacobian(
    jacobian: tuple[tuple[Expression, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[Polynomial, ...], ...], Polynomial | None] | None:
    """Extract one shared quotient denominator without cross-multiplying unrelated rows."""
    denominator: Polynomial | None = None
    for row in jacobian:
        for value in row:
            if isinstance(value, RationalPolynomial):
                if denominator is None:
                    denominator = value.denominator
                elif value.denominator != denominator:
                    return None
    numerators = tuple(
        tuple(
            value.numerator
            if isinstance(value, RationalPolynomial)
            else value
            if denominator is None
            else multiply(value, denominator, budget=budget)
            for value in row
        )
        for row in jacobian
    )
    return numerators, denominator


def _gram_metric_determinant(
    jacobian: tuple[tuple[Expression, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> Expression:
    """Retain quotient-by-quotient Gram expansion for unrelated denominators."""
    dimension = len(jacobian[0])
    metric = tuple(
        tuple(
            expression_sum(
                tuple(
                    expression_multiply(row[i], row[j], budget=budget) for row in jacobian
                ),
                budget=budget,
            )
            for j in range(dimension)
        )
        for i in range(dimension)
    )
    return expression_determinant(metric, budget=budget)


def expression_gram_determinant(
    jacobian: tuple[tuple[Expression, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> Expression:
    """Form the rational Gram determinant through Cauchy--Binet minors.

    A shared rational denominator is factored before any products. This keeps
    cancellation in the smaller minors instead of expanding products of Gram
    entries; the resulting numerator and denominator are the same polynomials.
    """
    dimension = len(jacobian[0])
    homogeneous = _homogeneous_jacobian(jacobian, budget=budget)
    if homogeneous is None:
        return _gram_metric_determinant(jacobian, budget=budget)
    numerators, denominator = homogeneous
    minors = tuple(
        determinant(tuple(numerators[row] for row in rows), budget=budget)
        for rows in imported_itertools.combinations(range(len(jacobian)), dimension)
    )
    squared = sum_polynomials(
        tuple(multiply(minor, minor, budget=budget) for minor in minors), budget=budget
    )
    return (
        squared
        if denominator is None
        else rational_expression(
            squared,
            power(denominator, 2 * dimension, dimension, budget=budget),
            budget=budget,
        )
    )


def restrict_chart_expressions(
    coordinates: tuple[Expression, ...],
    source_kind: str,
    target_kind: str,
    origin: np.ndarray,
    matrix: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...]:
    dimension = matrix.shape[1]
    chart = chart_arguments(target_kind, dimension, budget=budget)
    arguments = tuple(
        compose(term, chart, budget=budget)
        for term in affine_arguments(origin, matrix, budget=budget)
    )
    if source_kind != "pyramid":
        composition = ExpressionComposition(arguments, budget=budget)
        return tuple(composition(value, budget=budget) for value in coordinates)
    height = arguments[2]
    denominator = add(
        constant(1, dimension, budget=budget),
        scale(height, -1, budget=budget),
        budget=budget,
    )
    horizontal = (
        add(arguments[0], scale(height, Fraction(-1, 2), budget=budget), budget=budget),
        add(arguments[1], scale(height, Fraction(-1, 2), budget=budget), budget=budget),
    )

    def homogenize(polynomial: Polynomial) -> tuple[Polynomial, Polynomial]:
        degree = max((index[0] + index[1] for index in polynomial), default=0)
        numerator = sum_polynomials(
            tuple(
                scale(
                    math_product_polynomials(
                        (
                            power(horizontal[0], index[0], dimension, budget=budget),
                            power(horizontal[1], index[1], dimension, budget=budget),
                            power(height, index[2], dimension, budget=budget),
                            power(
                                denominator,
                                degree - index[0] - index[1],
                                dimension,
                                budget=budget,
                            ),
                        ),
                        dimension,
                        budget=budget,
                    ),
                    value,
                    budget=budget,
                )
                for index, value in polynomial.items()
            ),
            budget=budget,
        )
        return numerator, power(denominator, degree, dimension, budget=budget)

    result = []
    for value in coordinates:
        first, second = expression_parts(value, len(arguments), budget=budget)
        a, b = homogenize(first)
        c, d = homogenize(second)
        result.append(
            rational_expression(
                multiply(a, d, budget=budget),
                multiply(b, c, budget=budget),
                budget=budget,
            )
        )
    return tuple(result)


@checked
def coordinate_expressions(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...] | None:
    """Retained rational source, including rational nested restrictions."""
    with budget.temporary_scope():
        coefficients, cache_key = coordinate_preparation_key(
            element, local, budget=budget
        )
        return _coordinate_expressions_from_prepared(
            element, coefficients, cache_key, budget=budget
        )


def _coordinate_expressions_from_prepared(
    element: CellGeometryElement,
    coefficients: CoordinateSourceBank,
    key: CoordinatePreparationKey,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...] | None:
    """Share this operation's full binding between expression and polynomial work."""
    cached = budget.coordinate_cache.get(key)
    if cached is not None:
        return cached
    coordinates = _prepare_coordinate_expressions(
        element, coefficients, key, budget=budget
    )
    if coordinates is not None and key not in budget.coordinate_cache:
        budget.retain_basis((key, coordinates))
        budget.coordinate_cache[key] = coordinates
    return coordinates


@checked
def coordinate_map_images(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    include_expressions: bool = True,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Expression, ...] | None, CoordinateSourceBank | None]:
    """Prepare expressions and corners under one freshly hashed full input binding.

    The binding cannot escape this synchronous operation. Independent calls,
    including calls with mutable protocol owners, hash every current element leaf.
    """
    with budget.temporary_scope():
        coefficients, key = coordinate_preparation_key(element, local, budget=budget)
        coordinates = (
            _coordinate_expressions_from_prepared(
                element, coefficients, key, budget=budget
            )
            if include_expressions
            else None
        )
        corners = _coordinate_corner_images_from_prepared(
            element, coefficients, key, budget=budget
        )
        return coordinates, corners


@checked
def prepare_coordinate_expressions(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...] | None:
    """Prepare source expressions under a freshly earned full input binding."""
    with budget.temporary_scope():
        coefficients, key = coordinate_preparation_key(element, local, budget=budget)
        return _prepare_coordinate_expressions(element, coefficients, key, budget=budget)


def _prepare_coordinate_expressions(
    element: CellGeometryElement,
    local: CoordinateSourceBank,
    key: CoordinatePreparationKey,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...] | None:

    if isinstance(
        element,
        (
            _cell_geometry_dispatch().PolynomialComposedCellGeometryElement,
            _cell_geometry_dispatch().RationalComposedCellGeometryElement,
        ),
    ):
        # Form the source physical map once per authentic coefficient bank,
        # then pull it back. Recombining every composed cardinal basis repeats
        # the same parent affine action for every child/reference chart.
        source = coordinate_expressions(element.source_element, local, budget=budget)
        if source is None:
            return None
        composition = ExpressionComposition(
            prepared_reference_arguments(element, budget=budget), budget=budget
        )
        return tuple(composition(value, budget=budget) for value in source)

    if isinstance(element, _cell_geometry_dispatch().SplineCellGeometryElement):
        basis = source_expressions(element, budget=budget)
        if basis is None:
            return None
        rows = coordinate_source_rows(local)
        return expression_linear_combinations(
            basis,
            tuple(tuple(row[axis] for row in rows) for axis in range(len(rows[0]))),
            budget=budget,
        )

    if isinstance(element, _cell_geometry_dispatch().RestrictedCellGeometryElement):
        source = coordinate_expressions(element.source_element, local, budget=budget)
        if source is None:
            return None
        return restrict_chart_expressions(
            source,
            element.source_element.cell_kind,
            element.cell_kind,
            np.asarray(element.offset),
            np.asarray(element.matrix),
            budget=budget,
        )
    return _coordinate_polynomials_from_prepared(element, local, key, budget=budget)


class ExpressionComposition:
    """Share rational powers on polynomial or preparation rational reference actions."""

    def __init__(
        self,
        arguments: tuple[Expression, ...] | PreparedPolynomialArguments,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        self.composition: ArgumentComposition | None = None
        self.denominator: Polynomial | None = None
        self.rational_composition: ArgumentComposition | None = None
        self._denominator_snapshots: dict[
            PolynomialRepresentationKey, PreparedPolynomialCoefficients
        ] = {}
        if isinstance(arguments, PreparedPolynomialArguments):
            self.composition = ArgumentComposition(arguments, budget=budget)
            return
        if not any(isinstance(value, RationalPolynomial) for value in arguments):
            polynomials = tuple(
                value for value in arguments if not isinstance(value, RationalPolynomial)
            )
            self.composition = ArgumentComposition(polynomials, budget=budget)
            return
        rational = next(
            value for value in arguments if isinstance(value, RationalPolynomial)
        )
        dimension = len(next(iter(rational.denominator)))
        parts = tuple(
            expression_parts(value, dimension, budget=budget) for value in arguments
        )
        denominator = constant(1, dimension, budget=budget)
        for _, divisor in parts:
            if (
                divisor == denominator
                or divide_polynomial(denominator, divisor, budget=budget) is not None
            ):
                continue
            if divide_polynomial(divisor, denominator, budget=budget) is not None:
                denominator = divisor
            else:
                denominator = multiply(denominator, divisor, budget=budget)
        numerators: list[Polynomial] = []
        for numerator, divisor in parts:
            factor = divide_polynomial(denominator, divisor, budget=budget)
            if factor is None:
                raise ValueError(
                    "coordinate_enclosure.reference_composition must retain its common denominator; got a failed polynomial division."
                )
            numerators.append(multiply(numerator, factor, budget=budget))
        self.denominator = denominator
        self.rational_composition = ArgumentComposition(
            (*numerators, denominator), budget=budget
        )

    def _polynomial(
        self, value: Polynomial, *, budget: CoordinateEnclosureBudget
    ) -> tuple[Polynomial, Polynomial]:
        if self.rational_composition is None or self.denominator is None:
            raise ValueError(
                "coordinate_enclosure.reference_action must retain its homogeneous preparation; got missing composition or denominator."
            )
        dimension = len(next(iter(self.denominator)))
        numerator_bits, denominator_bits = coefficient_profile((value,), budget=budget)
        source_dimension = len(next(iter(value), ()))
        reserve_polynomial(
            len(value),
            len(value),
            source_dimension + 1,
            numerator_bits + denominator_bits,
            budget=budget,
        )
        degree = max((sum(index) for index in value), default=0)
        lifted: Polynomial = {
            (*index, degree - sum(index)): coefficient
            for index, coefficient in value.items()
        }
        return self.rational_composition(lifted, budget=budget), power(
            self.denominator, degree, dimension, budget=budget
        )

    def _denominator_pullback(
        self, value: Polynomial, *, budget: CoordinateEnclosureBudget
    ) -> Polynomial:
        """Reuse a full-content denominator pullback within this fixed argument bank."""
        composition = self.composition
        if composition is None:
            raise ValueError(
                "coordinate_enclosure.denominator_pullback must use polynomial arguments; got a rational argument bank."
            )
        if composition._axis_affine is not None:
            return composition(value, budget=budget)
        budget.reserve(
            len(value),
            512
            + 320 * len(value)
            + 4
            * sum(_scalar_hash_storage(coefficient) for coefficient in value.values()),
        )
        key = _polynomial_representation_key(value)
        cached = self._denominator_snapshots.get(key)
        if cached is not None:
            budget.reserve(len(cached), 256 + 64 * len(cached))
            return dict(cached)
        result = composition(value, budget=budget)
        budget.reserve(len(result), 512 + 320 * len(result))
        self._denominator_snapshots[key] = tuple(result.items())
        return result

    def __call__(
        self, value: Expression, /, *, budget: CoordinateEnclosureBudget
    ) -> Expression:
        if self.composition is not None:
            if isinstance(value, RationalPolynomial):
                return rational_expression(
                    self.composition(value.numerator, budget=budget),
                    self._denominator_pullback(value.denominator, budget=budget),
                    budget=budget,
                )
            return self.composition(value, budget=budget)
        if isinstance(value, RationalPolynomial):
            numerator, numerator_divisor = self._polynomial(
                value.numerator, budget=budget
            )
            denominator, denominator_divisor = self._polynomial(
                value.denominator, budget=budget
            )
            return rational_expression(
                multiply(numerator, denominator_divisor, budget=budget),
                multiply(numerator_divisor, denominator, budget=budget),
                budget=budget,
            )
        numerator, denominator = self._polynomial(value, budget=budget)
        return rational_expression(numerator, denominator, budget=budget)


def expression_node_count(value: Expression, domain: str, dimension: int) -> int:
    if isinstance(value, RationalPolynomial):
        indices: Polynomial = {
            index: Fraction(1) for index in (*value.numerator, *value.denominator)
        }
        return 2 * bernstein_node_count(indices, domain, dimension)
    return bernstein_node_count(value, domain, dimension)


def _expression_jacobian_rows(
    coordinates: tuple[Expression, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Expression, ...], ...]:
    """Prepare shared quotient denominator jets once for the coordinate bank."""
    if len(coordinates) <= 1 or dimension == 0:
        return tuple(
            tuple(
                expression_derivative(value, axis, budget=budget)
                for axis in range(dimension)
            )
            for value in coordinates
        )
    budget.reserve(0, 512 + 64 * len(coordinates))
    groups: dict[
        PolynomialRepresentationKey,
        tuple[PreparedPolynomialSupport, tuple[Polynomial, ...], Polynomial],
    ] = {}
    rows = []
    for value in coordinates:
        match value:
            case RationalPolynomial():
                budget.reserve(len(value.denominator), 256 + 320 * len(value.denominator))
                key = _polynomial_representation_key(value.denominator)
                prepared = groups.get(key)
                if prepared is None:
                    support = PreparedPolynomialSupport(value.denominator, budget=budget)
                    derivatives = tuple(
                        derivative(support, axis, budget=budget)
                        for axis in range(dimension)
                    )
                    square = multiply(support, support, budget=budget)
                    prepared = (support, derivatives, square)
                    groups[key] = prepared
                support, derivatives, square = prepared
                numerator_profile = coefficient_profile(
                    (value.numerator,), exact_denominator=True, budget=budget
                )
                rows.append(
                    tuple(
                        _rational_derivative(
                            value,
                            axis,
                            numerator_profile,
                            support,
                            derivatives[axis],
                            square,
                            budget=budget,
                        )
                        for axis in range(dimension)
                    )
                )
            case dict():
                rows.append(
                    tuple(
                        derivative(value, axis, budget=budget)
                        for axis in range(dimension)
                    )
                )
            case invalid:
                assert_never(invalid)
    return tuple(rows)


def expression_physical_jacobian(
    coordinates: tuple[Expression, ...],
    kind: str,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Expression, ...], ...]:
    if kind != "pyramid":
        affine = _affine_polynomial_jacobian(coordinates, dimension, budget=budget)
        if affine is not None:
            return affine
    rows = _expression_jacobian_rows(coordinates, dimension, budget=budget)
    if kind != "pyramid":
        return rows
    u, v, w = axes(3, budget=budget)
    collapse = add(
        constant(1, 3, budget=budget), scale(w, -1, budget=budget), budget=budget
    )
    reciprocal = RationalPolynomial(constant(1, 3, budget=budget), collapse)
    result = []
    for first, second, height in rows:
        x, y = (
            expression_multiply(first, reciprocal, budget=budget),
            expression_multiply(second, reciprocal, budget=budget),
        )
        z = expression_sum(
            (
                height,
                expression_multiply(
                    add(u, constant(Fraction(-1, 2), 3, budget=budget), budget=budget),
                    x,
                    budget=budget,
                ),
                expression_multiply(
                    add(v, constant(Fraction(-1, 2), 3, budget=budget), budget=budget),
                    y,
                    budget=budget,
                ),
            ),
            budget=budget,
        )
        result.append((x, y, z))
    return tuple(result)


def expression_reference_evaluate(
    value: Expression,
    point: tuple[ExactScalar, ...],
    domain: str,
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactScalar:
    """Evaluate including a proved continuous rational chart corner.

    The denominator must be homogeneous about the singular corner. Bounded
    rational Bernstein controls exclude unresolved transverse poles; the
    lowest homogeneous numerator must equal one constant times the denominator.
    Higher homogeneous terms then tend to zero on the occupied reference cone.
    """
    match value:
        case RationalPolynomial():
            evaluated_denominator = evaluate(value.denominator, point, budget=budget)
            if evaluated_denominator:
                return _scalar_quotient(
                    evaluate(value.numerator, point, budget=budget),
                    evaluated_denominator,
                    budget=budget,
                )
        case dict():
            return evaluate(value, point, budget=budget)
        case invalid:
            assert_never(invalid)
    return _rational_reference_corner_limit(value, point, domain, budget=budget)


def _rational_reference_corner_limit(
    value: RationalPolynomial,
    point: tuple[ExactScalar, ...],
    domain: str,
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactScalar:
    """Prove the homogeneous corner limit after its denominator evaluated to zero."""
    expression_bernstein_coefficients(value, domain, len(point), budget=budget)
    arguments = tuple(
        add(constant(x, len(point), budget=budget), variable, budget=budget)
        for x, variable in zip(point, axes(len(point), budget=budget), strict=True)
    )
    numerator, denominator = (
        compose(value.numerator, arguments, budget=budget),
        compose(value.denominator, arguments, budget=budget),
    )
    degrees = {sum(index) for index in denominator}
    if len(degrees) != 1:
        raise RationalEnclosureError(
            "coordinate_enclosure.corner_limit must have a homogeneous denominator; got mixed total degrees."
        )
    degree = next(iter(degrees))
    if any(sum(index) < degree for index in numerator):
        raise RationalEnclosureError(
            "coordinate_enclosure.corner_limit must have a removable pole; got a lower-degree numerator term."
        )
    leading = {
        index: coefficient
        for index, coefficient in numerator.items()
        if sum(index) == degree
    }
    pivot = next(iter(denominator))
    limit = _scalar_quotient(
        leading.get(pivot, Fraction(0)), denominator[pivot], budget=budget
    )
    if leading != scale(denominator, limit, budget=budget):
        raise RationalEnclosureError(
            "coordinate_enclosure.corner_image must be independent of the reference approach; got unequal homogeneous leading terms."
        )
    return limit


def expression_control_points(
    coordinates: tuple[Expression, ...],
    domain: str,
    dimension: int,
    maximum_nodes: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> CoordinateSourceBank | None:
    """Bound a rational image by its common-denominator projective control net."""
    denominators: list[Polynomial] = []
    for value in coordinates:
        if (
            isinstance(value, RationalPolynomial)
            and value.denominator not in denominators
        ):
            denominators.append(value.denominator)
    denominator = math_product_polynomials(tuple(denominators), dimension, budget=budget)
    numerators = []
    for value in coordinates:
        if isinstance(value, RationalPolynomial):
            factor = math_product_polynomials(
                tuple(d for d in denominators if d != value.denominator),
                dimension,
                budget=budget,
            )
            numerators.append(multiply(value.numerator, factor, budget=budget))
        else:
            numerators.append(multiply(value, denominator, budget=budget))
    indices = tuple(
        index for polynomial in (*numerators, denominator) for index in polynomial
    )
    degrees = _common_control_degrees(indices, domain, dimension)
    reference = parse(domain, BernsteinDomain, "domain")
    node_count = (
        math.comb(degrees[0] + dimension, dimension)
        if reference == "simplex"
        else math.comb(degrees[0] + 2, 2) * (degrees[1] + 1)
        if reference == "prism"
        else math.prod(degree + 1 for degree in degrees)
    )
    if node_count > maximum_nodes:
        return None
    weights = expression_control_net(
        denominator, domain, degrees, dimension, budget=budget
    )
    controls = tuple(
        expression_control_net(value, domain, degrees, dimension, budget=budget)
        for value in numerators
    )
    if max(weights) < 0:
        weights = tuple(-value for value in weights)
        controls = tuple(tuple(-value for value in row) for row in controls)
    if min(weights) < 0 or not any(weights):
        return None
    points = []
    for weight, row in zip(weights, zip(*controls, strict=True), strict=True):
        if weight:
            points.append(
                tuple(_scalar_quotient(value, weight, budget=budget) for value in row)
            )
        elif any(row):
            return None
    return tuple(points)


def has_cartesian_reference_chain(element: CellGeometryElement, /) -> bool:
    """Distinguish Cartesian pullbacks from full coefficient-action ancestry."""

    current = element
    while isinstance(
        current,
        (
            _cell_geometry_dispatch().RestrictedCellGeometryElement,
            _cell_geometry_dispatch().PolynomialComposedCellGeometryElement,
            _cell_geometry_dispatch().RationalComposedCellGeometryElement,
        ),
    ):
        current = current.source_element
    return not isinstance(
        current, _cell_geometry_dispatch().BarycentricCellGeometryElement
    )


def coordinate_reference_chain(
    element: CellGeometryElement,
    *,
    ancestor: CellGeometryElement | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    FiniteElementSpec
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement,
    tuple[Expression, ...],
]:
    return _prepare_coordinate_reference_chain(
        element, ancestor=ancestor, prove_partition_unity=False, budget=budget
    )


def coordinate_partition_unity_reference_chain(
    element: CellGeometryElement,
    *,
    ancestor: CellGeometryElement | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    FiniteElementSpec
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement,
    tuple[Expression, ...],
]:
    """Recover source references only after proving the full P1 unity law."""
    return _prepare_coordinate_reference_chain(
        element, ancestor=ancestor, prove_partition_unity=True, budget=budget
    )


@checked
def try_partition_unity_reference_chain(
    element: CellGeometryElement,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellGeometryElement, tuple[Expression, ...]] | None:
    """Return a source-reference chain when its P1 unity identity holds.

    Unsupported cardinal support or a failed unity identity returns None.
    Malformed source coefficients and resource refusals propagate.
    """
    return _prepare_coordinate_reference_chain(
        element,
        ancestor=None,
        prove_partition_unity=True,
        partition_unity_optional=True,
        budget=budget,
    )


type CoordinateReferenceChain = tuple[
    FiniteElementSpec
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement,
    tuple[Expression, ...],
]


@overload
def _prepare_coordinate_reference_chain(
    element: CellGeometryElement,
    *,
    ancestor: CellGeometryElement | None,
    prove_partition_unity: bool,
    budget: CoordinateEnclosureBudget,
    partition_unity_optional: Literal[False] = False,
) -> CoordinateReferenceChain: ...


@overload
def _prepare_coordinate_reference_chain(
    element: CellGeometryElement,
    *,
    ancestor: CellGeometryElement | None,
    prove_partition_unity: bool,
    budget: CoordinateEnclosureBudget,
    partition_unity_optional: Literal[True],
) -> CoordinateReferenceChain | None: ...


def _prepare_coordinate_reference_chain(
    element: CellGeometryElement,
    *,
    ancestor: CellGeometryElement | None,
    prove_partition_unity: bool,
    budget: CoordinateEnclosureBudget,
    partition_unity_optional: bool = False,
) -> CoordinateReferenceChain | None:
    """Rational physical-root reference map of affine and polynomial pullbacks.

    Every stored binary64 affine coefficient and every authored rational chart
    coefficient participates in exact arithmetic; no chain is collapsed through
    floating matrix products or interpolated target corners.
    """

    current = _cell_geometry_dispatch().require_scalar_coordinate_element(
        element, "Exact coordinate reference chain"
    )
    arguments = chart_arguments(
        current.cell_kind, current.topological_dimension, budget=budget
    )
    while isinstance(
        current,
        (
            _cell_geometry_dispatch().RestrictedCellGeometryElement,
            _cell_geometry_dispatch().PolynomialComposedCellGeometryElement,
            _cell_geometry_dispatch().RationalComposedCellGeometryElement,
        ),
    ):
        if ancestor is not None and current.element_id == ancestor.element_id:
            if coordinate_source_signature(current) != coordinate_source_signature(
                ancestor
            ):
                raise ValueError(
                    "coordinate_enclosure.ancestor must retain live source semantics; got changed source semantics."
                )
            return current, arguments
        if isinstance(current, _cell_geometry_dispatch().RestrictedCellGeometryElement):
            chart = affine_arguments(
                np.asarray(current.offset), np.asarray(current.matrix), budget=budget
            )
        else:
            chart = reference_composition_arguments(current, budget=budget)
        if isinstance(
            current, _cell_geometry_dispatch().RationalComposedCellGeometryElement
        ):
            height = arguments[2]
            divisor = expression_add(
                constant(1, current.topological_dimension, budget=budget),
                expression_scale(height, -1, budget=budget),
                budget=budget,
            )
            a, b = expression_parts(divisor, current.topological_dimension, budget=budget)
            reciprocal = rational_expression(b, a, budget=budget)
            collapsed = (
                expression_multiply(
                    expression_add(
                        arguments[0],
                        expression_scale(height, Fraction(-1, 2), budget=budget),
                        budget=budget,
                    ),
                    reciprocal,
                    budget=budget,
                ),
                expression_multiply(
                    expression_add(
                        arguments[1],
                        expression_scale(height, Fraction(-1, 2), budget=budget),
                        budget=budget,
                    ),
                    reciprocal,
                    budget=budget,
                ),
                height,
            )
            composition = ExpressionComposition(collapsed, budget=budget)
        else:
            composition = ExpressionComposition(arguments, budget=budget)
        arguments = tuple(composition(value, budget=budget) for value in chart)
        current = current.source_element
    if isinstance(current, _cell_geometry_dispatch().BarycentricCellGeometryElement):
        if not prove_partition_unity:
            raise ValueError(
                "coordinate_enclosure.reference_chain must have a Cartesian source-reference chain; got full barycentric coefficient actions."
            )
        root = current
        while isinstance(root, _cell_geometry_dispatch().BarycentricCellGeometryElement):
            root = root.source_element
        basis = source_basis(current, budget=budget)
        if basis is None or len(basis) != root.topological_dimension + 1:
            if partition_unity_optional:
                return None
            raise ValueError(
                "coordinate_enclosure.reference_action must retain its full P1 cardinal basis; got unsupported or unequal basis count."
            )
        unity = constant(1, root.topological_dimension, budget=budget)
        if sum_polynomials(basis, budget=budget) != unity or basis[0] != add(
            unity,
            scale(sum_polynomials(basis[1:], budget=budget), -1, budget=budget),
            budget=budget,
        ):
            if partition_unity_optional:
                return None
            raise ValueError(
                "coordinate_enclosure.reference_action must preserve partition of unity; got a failed polynomial identity."
            )
        composition = ExpressionComposition(arguments, budget=budget)
        arguments = tuple(composition(value, budget=budget) for value in basis[1:])
        current = root
    if ancestor is not None:
        if current.element_id != ancestor.element_id or coordinate_source_signature(
            current
        ) != coordinate_source_signature(ancestor):
            raise ValueError(
                "coordinate_enclosure.ancestor must be a current authored ancestor; got another source identity."
            )
        return current, arguments
    if not isinstance(
        current,
        (
            FiniteElementSpec,
            _cell_geometry_dispatch().SplineCellGeometryElement,
            _cell_geometry_dispatch().LayerColumnCellGeometryElement,
        ),
    ):
        raise TypeError(
            "coordinate_enclosure.reference_chain must retain a scalar source root; got another source kind."
        )
    return current, arguments


def prepare_p1_source_packet_pullback(
    element: CellGeometryElement,
    root_element: CellGeometryElement,
    root_uv: np.ndarray,
    root_xyz: np.ndarray,
    actual_xyz: np.ndarray,
    original_root_expressions: tuple[Expression, ...],
    *,
    root_corner_ids: np.ndarray,
    retained_corner_ids: np.ndarray,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellGeometryElement, tuple[Polynomial, ...], tuple[Expression, ...]]:
    """Prove a packet-bound full P1 UV map and retain its whole XYZ remainder.

    All source cardinal columns contribute even when their sum is not one.
    Only the source UV triangle frame is inverted; no target XYZ, RNE
    carrier or field samples are fitted. The caller owns root namespace/SCI
    and source-bound admission and must decide the returned whole remainder.
    """

    root = element
    while isinstance(root, _cell_geometry_dispatch().BarycentricCellGeometryElement):
        root = root.source_element
    _cell_geometry_dispatch().require_p1_cardinal_source(root)
    if root.cell_kind != "triangle" or coordinate_source_signature(
        root
    ) != coordinate_source_signature(root_element):
        raise ValueError(
            "coordinate_enclosure.p1_packet must retain its triangular source basis; got another source signature."
        )
    ids, retained = np.asarray(root_corner_ids), np.asarray(retained_corner_ids)
    if (
        ids.dtype.kind not in "iu"
        or retained.dtype.kind not in "iu"
        or ids.shape != (3,)
        or not np.array_equal(ids, retained)
        or len(set(ids.tolist())) != 3
    ):
        raise ValueError(
            "coordinate_enclosure.p1_packet must retain ordered source corner SCI; got changed or repeated corner identities."
        )
    uv, xyz, controls = (
        np.asarray(value, dtype=np.float64) for value in (root_uv, root_xyz, actual_xyz)
    )
    if (
        uv.shape != (3, 2)
        or xyz.shape != (3, 3)
        or controls.shape != (3, 3)
        or not all(np.all(np.isfinite(value)) for value in (uv, xyz, controls))
    ):
        raise ValueError(
            "coordinate_enclosure.p1_packet must have finite source UV and XYZ coefficient banks; got incompatible shape or nonfinite values."
        )
    if not np.array_equal(xyz.view(np.uint64), controls.view(np.uint64)):
        raise ValueError(
            "coordinate_enclosure.p1_packet must retain the physical coefficient bank; got changed coefficient bits."
        )
    uv_source = coordinate_polynomials(element, uv, budget=budget)
    physical = coordinate_expressions(element, controls, budget=budget)
    if uv_source is None or physical is None or len(original_root_expressions) != 3:
        raise ValueError(
            "coordinate_enclosure.p1_packet must retain full UV/physical source expressions; got unsupported basis or unequal component count."
        )
    original_uv = tuple(rational_vector(row) for row in uv)
    matrix = tuple(
        tuple(original_uv[column + 1][axis] - original_uv[0][axis] for column in range(2))
        for axis in range(2)
    )
    prepared = imported_linalg__small_batched.prepare_exact_small_linear_actions(
        matrix,
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
        budget=budget,
    )
    if prepared.actions is None:
        raise ValueError(
            "coordinate_enclosure.source_uv_triangle must have full rational rank; got deficient rank."
        )
    differences = tuple(
        add(value, constant(-original_uv[0][axis], 2, budget=budget), budget=budget)
        for axis, value in enumerate(uv_source)
    )
    arguments = linear_combinations(differences, prepared.actions, budget=budget)
    composition = ExpressionComposition(arguments, budget=budget)
    restricted = tuple(
        composition(value, budget=budget) for value in original_root_expressions
    )
    remainder = tuple(
        expression_add(
            actual, expression_scale(expected, -1, budget=budget), budget=budget
        )
        for actual, expected in zip(physical, restricted, strict=True)
    )
    return root, arguments, remainder


@dataclass(frozen=True, slots=True)
class CoordinateEnclosureEstimate:
    """Structural support and preparation bounds for one declared rational operation tree.

    Denominator bits bound a common denominator, not only one coefficient.
    A non-None denominator describes a quotient; no coefficients are evaluated.
    """

    terms: int
    dimension: int
    degree: int
    numerator_bits: int
    denominator_bits: int
    work_upper: int
    bytes_upper: int
    denominator: CoordinateEnclosureEstimate | None = None


def coordinate_algebraic_preparation_bounds(
    work_upper: int,
    bytes_upper: int,
    /,
    *,
    field_degree_upper: int,
    polynomial_degree_upper: int,
) -> tuple[int, int]:
    """Lift one full rational circuit budget to its declared coefficient field.

    Apply once after circuit aggregation, not once per expression node. The
    field owner bounds scalar operations by cubic field degree and storage by
    the retained field dimension times coefficient-height growth.
    """
    degree = nonnegative_integer(
        field_degree_upper, "coordinate_estimate.field_degree_upper"
    )
    polynomial_degree = nonnegative_integer(
        polynomial_degree_upper, "coordinate_estimate.polynomial_degree_upper"
    )
    if not degree:
        raise ValueError(
            "coordinate_estimate.field_degree_upper must be positive; got zero."
        )
    if degree == 1:
        return work_upper, bytes_upper
    return (
        work_upper * degree**3,
        bytes_upper * degree * max(degree, polynomial_degree + 1),
    )


def _estimate_storage(terms: int, dimension: int, degree: int, bits: int, /) -> int:
    # Includes support dictionaries, index tuples, rational integers, traversal
    # indices, cache bindings and coexistence of scratch and retained snapshots.
    integer = declared_integer_storage_bytes(max(bits, 1))
    exponent = declared_integer_storage_bytes(max(degree, 1).bit_length())
    return 4096 + terms * (4096 + dimension * (128 + exponent) + 4 * integer)


def coordinate_polynomial_estimate(
    *,
    terms: int,
    dimension: int,
    degree: int,
    numerator_bits: int,
    denominator_bits: int,
) -> CoordinateEnclosureEstimate:
    """Bound a source leaf from its declared support and coefficient bit lengths."""
    values = tuple(
        nonnegative_integer(value, "coordinate_estimate." + name)
        for name, value in (
            ("terms", terms),
            ("dimension", dimension),
            ("degree", degree),
            ("numerator_bits", numerator_bits),
            ("denominator_bits", denominator_bits),
        )
    )
    terms, dimension, degree, numerator_bits, denominator_bits = values
    return CoordinateEnclosureEstimate(
        terms,
        dimension,
        degree,
        max(numerator_bits, 1),
        max(denominator_bits, 1),
        2 + terms * (dimension + 5),
        2
        * _estimate_storage(terms, dimension, degree, numerator_bits + denominator_bits),
    )


def coordinate_expression_estimate(value: Expression, /) -> CoordinateEnclosureEstimate:
    """Read rational support metadata without expanding, composing or normalizing it."""
    if isinstance(value, RationalPolynomial):
        numerator = coordinate_expression_estimate(value.numerator)
        denominator = coordinate_expression_estimate(value.denominator)
        return CoordinateEnclosureEstimate(
            numerator.terms,
            numerator.dimension,
            numerator.degree,
            numerator.numerator_bits,
            numerator.denominator_bits,
            numerator.work_upper + denominator.work_upper,
            numerator.bytes_upper + denominator.bytes_upper,
            denominator,
        )
    return coordinate_polynomial_estimate(
        terms=len(value),
        dimension=len(next(iter(value), ())),
        degree=max((sum(index) for index in value), default=0),
        numerator_bits=max((_scalar_profile(v)[0] for v in value.values()), default=1),
        denominator_bits=sum(_scalar_profile(v)[1].bit_length() for v in value.values()),
    )


def _estimate_support(terms: int, dimension: int, degree: int, /) -> int:
    """Bound merged dictionary support by the total-degree monomial count."""
    if terms <= 1:
        return terms
    result = 1
    order = min(degree, dimension)
    for index in range(1, order + 1):
        result = result * (degree + dimension - order + index) // index
        if result >= terms:
            return terms
    return min(result, terms)


def _bounded_integer_power(base: int, exponent: int, bound: int, /) -> int:
    """Return min(base**exponent, bound) without constructing a larger integer."""
    result = 1
    factor = min(base, bound)
    while exponent:
        if exponent & 1:
            result = min(result * factor, bound)
        exponent >>= 1
        if exponent:
            factor = min(factor * factor, bound)
    return min(result, bound)


def _estimate_result(
    first: CoordinateEnclosureEstimate,
    second: CoordinateEnclosureEstimate,
    *,
    terms: int,
    degree: int,
    numerator_bits: int,
    denominator_bits: int,
    work: int,
    storage_terms: int | None = None,
) -> CoordinateEnclosureEstimate:
    dimension = max(first.dimension, second.dimension)
    storage = _estimate_storage(
        terms if storage_terms is None else storage_terms,
        dimension,
        degree,
        numerator_bits + denominator_bits,
    )
    return CoordinateEnclosureEstimate(
        terms,
        dimension,
        degree,
        numerator_bits,
        denominator_bits,
        first.work_upper + second.work_upper + work + 2 + terms * (dimension + 5),
        first.bytes_upper + second.bytes_upper + 2 * storage,
    )


def _estimate_parts(
    value: CoordinateEnclosureEstimate, /
) -> tuple[CoordinateEnclosureEstimate, CoordinateEnclosureEstimate]:
    return value, (
        coordinate_polynomial_estimate(
            terms=1,
            dimension=value.dimension,
            degree=0,
            numerator_bits=1,
            denominator_bits=1,
        )
        if value.denominator is None
        else value.denominator
    )


def coordinate_estimate_add(
    first: CoordinateEnclosureEstimate, second: CoordinateEnclosureEstimate, /
) -> CoordinateEnclosureEstimate:
    """Bound rational addition, including rational cross products and cancellation."""
    if first.denominator is not None or second.denominator is not None:
        a, b = _estimate_parts(first)
        c, d = _estimate_parts(second)
        # Strip quotient markers only: each numerator keeps its already earned cost.
        a = _polynomial_estimate_part(a)
        c = _polynomial_estimate_part(c)
        numerator = coordinate_estimate_add(
            coordinate_estimate_multiply(a, d), coordinate_estimate_multiply(c, b)
        )
        return coordinate_estimate_rational(numerator, coordinate_estimate_multiply(b, d))
    return _estimate_result(
        first,
        second,
        terms=_estimate_support(
            first.terms + second.terms,
            max(first.dimension, second.dimension),
            max(first.degree, second.degree),
        ),
        degree=max(first.degree, second.degree),
        numerator_bits=max(
            first.numerator_bits + second.denominator_bits,
            second.numerator_bits + first.denominator_bits,
        )
        + 1,
        denominator_bits=first.denominator_bits + second.denominator_bits,
        work=2 * (first.terms + second.terms),
        storage_terms=first.terms + second.terms,
    )


def _polynomial_estimate_part(
    value: CoordinateEnclosureEstimate, /
) -> CoordinateEnclosureEstimate:
    return CoordinateEnclosureEstimate(
        value.terms,
        value.dimension,
        value.degree,
        value.numerator_bits,
        value.denominator_bits,
        value.work_upper,
        value.bytes_upper,
    )


def coordinate_estimate_multiply(
    first: CoordinateEnclosureEstimate, second: CoordinateEnclosureEstimate, /
) -> CoordinateEnclosureEstimate:
    """Bound the full rational product and its source-expression preparation."""
    if first.denominator is not None or second.denominator is not None:
        a, b = _estimate_parts(first)
        c, d = _estimate_parts(second)
        return coordinate_estimate_rational(
            coordinate_estimate_multiply(
                _polynomial_estimate_part(a), _polynomial_estimate_part(c)
            ),
            coordinate_estimate_multiply(b, d),
        )
    pairs = first.terms * second.terms
    dimension = max(first.dimension, second.dimension)
    return _estimate_result(
        first,
        second,
        terms=_estimate_support(pairs, dimension, first.degree + second.degree),
        degree=first.degree + second.degree,
        numerator_bits=first.numerator_bits
        + second.numerator_bits
        + first.denominator_bits
        + second.denominator_bits
        + max(pairs, 1).bit_length(),
        denominator_bits=first.denominator_bits + second.denominator_bits,
        work=(dimension + 1) * (first.terms + second.terms) + pairs,
        storage_terms=_bounded_integer_power(
            first.degree + second.degree + 1, dimension, pairs
        ),
    )


def coordinate_estimate_scale(
    value: CoordinateEnclosureEstimate, coefficient: ExactScalar | int, /
) -> CoordinateEnclosureEstimate:
    """Bound rational scalar multiplication from the declared scalar itself."""
    scalar = (
        coefficient if isinstance(coefficient, RadicalElement) else Fraction(coefficient)
    )
    numerator_bits, denominator = _scalar_profile(scalar)
    leaf = coordinate_polynomial_estimate(
        terms=int(bool(scalar)),
        dimension=value.dimension,
        degree=0,
        numerator_bits=numerator_bits,
        denominator_bits=denominator.bit_length(),
    )
    if value.denominator is not None:
        return coordinate_estimate_rational(
            coordinate_estimate_scale(_polynomial_estimate_part(value), coefficient),
            value.denominator,
        )
    return _estimate_result(
        value,
        leaf,
        terms=value.terms if scalar else 0,
        degree=value.degree,
        numerator_bits=value.numerator_bits + leaf.numerator_bits,
        denominator_bits=value.denominator_bits + leaf.denominator_bits,
        work=3 * value.terms,
    )


def coordinate_estimate_power(
    value: CoordinateEnclosureEstimate, exponent: int, /
) -> CoordinateEnclosureEstimate:
    """Bound the declared repeated-product circuit without materializing its powers."""
    exponent = nonnegative_integer(exponent, "coordinate_estimate.exponent")
    if value.denominator is not None:
        return coordinate_estimate_rational(
            coordinate_estimate_power(_polynomial_estimate_part(value), exponent),
            coordinate_estimate_power(value.denominator, exponent),
        )
    dimension, degree = value.dimension, value.degree * exponent
    if value.terms <= 1:
        terms = value.terms if exponent else 1
        previous_terms = value.terms if exponent > 1 else 1
    else:
        support_limit = math.comb(degree + dimension, dimension)
        terms = _bounded_integer_power(value.terms, exponent, support_limit)
        previous_terms = _bounded_integer_power(
            value.terms,
            max(exponent - 1, 0),
            math.comb(value.degree * max(exponent - 1, 0) + dimension, dimension),
        )
    # power() performs exponent sequential multiplies. Dictionary merging
    # bounds each intermediate support by C(k*degree+dimension, dimension);
    # every multiply still visits all pairs before merging. Include both
    # profiles and their dimension traversals, including the initial constant.
    visits = (
        exponent * (previous_terms * value.terms + previous_terms + value.terms + 1) + 1
    )
    storage_terms = (
        exponent
        * _bounded_integer_power(degree + 1, dimension, previous_terms * value.terms)
        + 1
    )
    numerator = max(
        1,
        exponent
        * (
            value.numerator_bits
            + value.denominator_bits
            + max(value.terms, 1).bit_length()
        ),
    )
    denominator = max(1, exponent * value.denominator_bits)
    return CoordinateEnclosureEstimate(
        terms,
        dimension,
        degree,
        numerator,
        denominator,
        exponent * value.work_upper
        + (dimension + 8) * visits
        + exponent * (dimension + 2) * value.terms
        + 4,
        exponent * value.bytes_upper
        + 2
        * _estimate_storage(storage_terms, dimension, degree, numerator + denominator),
    )


def coordinate_estimate_rational(
    numerator: CoordinateEnclosureEstimate, denominator: CoordinateEnclosureEstimate, /
) -> CoordinateEnclosureEstimate:
    """Bound rational denominator cancellation using a decreasing monomial weight.

    For divisor total degree D, weights (D+1)**(dimension-axis-1) make
    every nonleading divisor monomial lighter than its lexicographic leader.
    Every elimination replaces a monomial with lighter, lexicographically
    smaller monomials, so successive leaders are distinct monomials of weight
    at most the numerator's. Visits are bounded by the smaller of the
    uncancelled expansion tree and the count of those monomials.
    """
    if numerator.denominator is not None or denominator.denominator is not None:
        raise ValueError(
            "coordinate_estimate.rational must receive polynomial parts; got a quotient."
        )
    if not denominator.terms:
        raise ValueError(
            "coordinate_estimate.denominator must be nonzero; got empty support."
        )
    dimension = max(numerator.dimension, denominator.dimension)
    weight = numerator.degree * (denominator.degree + 1) ** max(dimension - 1, 0)
    base = denominator.degree + 1
    # Exponent vectors i with sum(i[a] * base**(dimension-a-1)) <= weight.
    leaders = math.prod(weight // base**axis + 1 for axis in range(dimension))
    branches = denominator.terms
    if branches == 1:
        iterations = min(numerator.terms * (weight + 1), leaders)
    elif weight * (branches.bit_length() - 1) >= leaders.bit_length():
        # branches**weight already exceeds the leader count; skip the tree.
        iterations = leaders
    else:
        expansions = (branches ** (weight + 1) - 1) // (branches - 1)
        iterations = min(numerator.terms * expansions, leaders)
    scratch_degree = max(numerator.degree, dimension * weight)
    scratch_terms = max(numerator.terms, (scratch_degree + 1) ** dimension)
    # Successful rational division has deg(quotient) <= deg(numerator); failed
    # cancellation retains its source numerator. Elimination scratch is not
    # output support and must not inflate subsequent source-expression degrees.
    degree = numerator.degree
    terms = max(numerator.terms, (degree + 1) ** dimension)
    bits = (
        numerator.numerator_bits
        + numerator.denominator_bits
        + denominator.numerator_bits
        + denominator.denominator_bits
        + 2
    ) * (iterations + 1)
    storage = _estimate_storage(
        scratch_terms + denominator.terms, dimension, scratch_degree, 2 * bits
    )
    return CoordinateEnclosureEstimate(
        terms,
        dimension,
        degree,
        bits,
        bits,
        numerator.work_upper
        + denominator.work_upper
        + 4 * (numerator.terms + denominator.terms)
        + iterations * ((dimension + 8) * (scratch_terms + denominator.terms + 1))
        + 2
        + terms * (dimension + 5),
        numerator.bytes_upper + denominator.bytes_upper + 8 * (iterations + 1) * storage,
        denominator,
    )


def coordinate_estimate_compose(
    value: CoordinateEnclosureEstimate,
    arguments: tuple[CoordinateEnclosureEstimate, ...],
    /,
) -> CoordinateEnclosureEstimate:
    """Bound the uncached source monomial circuit, including rational arguments."""
    dimension = max(
        (argument.dimension for argument in arguments), default=value.dimension
    )
    monomial = coordinate_polynomial_estimate(
        terms=1,
        dimension=dimension,
        degree=0,
        numerator_bits=value.numerator_bits,
        denominator_bits=value.denominator_bits,
    )
    for argument in arguments:
        monomial = coordinate_estimate_multiply(
            monomial, coordinate_estimate_power(argument, value.degree)
        )
    if value.denominator is not None:
        numerator = coordinate_estimate_compose(
            _polynomial_estimate_part(value), arguments
        )
        denominator = coordinate_estimate_compose(value.denominator, arguments)
        a, b = _estimate_parts(numerator)
        c, d = _estimate_parts(denominator)
        return coordinate_estimate_rational(
            coordinate_estimate_multiply(_polynomial_estimate_part(a), d),
            coordinate_estimate_multiply(b, _polynomial_estimate_part(c)),
        )
    if monomial.denominator is not None:
        # Rational composition also prepares a common homogeneous denominator.
        result = coordinate_estimate_rational(
            _polynomial_estimate_part(monomial), monomial.denominator
        )
    else:
        result = monomial
    count = value.terms
    unmerged_terms = count * result.terms
    terms = _estimate_support(unmerged_terms, dimension, result.degree)
    bits = (
        result.numerator_bits
        + count * result.denominator_bits
        + max(count, 1).bit_length()
    )
    return CoordinateEnclosureEstimate(
        terms,
        dimension,
        result.degree,
        bits,
        max(1, count * result.denominator_bits),
        value.work_upper
        + count * result.work_upper
        + (count + 1) * (count + 1) * (terms + 1) * (dimension + 8),
        value.bytes_upper
        + count * result.bytes_upper
        + 4 * _estimate_storage(unmerged_terms, dimension, result.degree, 2 * bits),
        result.denominator,
    )


def coordinate_estimate_bank(
    values: tuple[CoordinateEnclosureEstimate, ...], /
) -> CoordinateEnclosureEstimate:
    """Aggregate independently prepared source-bank circuits, not an arithmetic sum."""
    return CoordinateEnclosureEstimate(
        sum(value.terms for value in values),
        max((value.dimension for value in values), default=0),
        max((value.degree for value in values), default=0),
        max((value.numerator_bits for value in values), default=1),
        sum(value.denominator_bits for value in values),
        sum(value.work_upper for value in values) + len(values) + 1,
        sum(value.bytes_upper for value in values) + 4096 + 128 * len(values),
    )


def coordinate_estimate_constant(
    value: ExactScalar | int, dimension: int, /
) -> CoordinateEnclosureEstimate:
    coefficient = value if isinstance(value, RadicalElement) else Fraction(value)
    numerator_bits, denominator = _scalar_profile(coefficient)
    return coordinate_polynomial_estimate(
        terms=int(bool(coefficient)),
        dimension=dimension,
        degree=0,
        numerator_bits=numerator_bits,
        denominator_bits=denominator.bit_length(),
    )


def coordinate_estimate_variable(dimension: int, /) -> CoordinateEnclosureEstimate:
    leaf = coordinate_polynomial_estimate(
        terms=1, dimension=dimension, degree=1, numerator_bits=1, denominator_bits=1
    )
    return CoordinateEnclosureEstimate(
        leaf.terms,
        leaf.dimension,
        leaf.degree,
        leaf.numerator_bits,
        leaf.denominator_bits,
        leaf.work_upper + dimension,
        leaf.bytes_upper + _estimate_storage(dimension, dimension, 1, 2),
    )


def coordinate_estimate_sum(
    values: tuple[CoordinateEnclosureEstimate, ...], dimension: int, /
) -> CoordinateEnclosureEstimate:
    result = coordinate_estimate_constant(0, dimension)
    for value in values:
        result = coordinate_estimate_add(result, value)
    return result


def coordinate_estimate_product(
    values: tuple[CoordinateEnclosureEstimate, ...], dimension: int, /
) -> CoordinateEnclosureEstimate:
    result = coordinate_estimate_constant(1, dimension)
    for value in values:
        result = coordinate_estimate_multiply(result, value)
    return result


def coordinate_jacobi_estimate(
    degree: int, alpha: int, axis: int, dimension: int, /
) -> CoordinateEnclosureEstimate:
    """Bound the same binomial source circuit used by the rational Jacobi polynomial."""
    if not 0 <= axis < dimension:
        raise ValueError(
            "coordinate_estimate.axis must belong to the parameter dimension; got invalid axis."
        )
    variable = coordinate_estimate_variable(dimension)
    shifted = coordinate_estimate_add(
        variable, coordinate_estimate_constant(-1, dimension)
    )
    return coordinate_estimate_sum(
        tuple(
            coordinate_estimate_scale(
                coordinate_estimate_multiply(
                    coordinate_estimate_power(shifted, degree - i),
                    coordinate_estimate_power(variable, i),
                ),
                math.comb(degree + alpha, i) * math.comb(degree, degree - i),
            )
            for i in range(degree + 1)
        ),
        dimension,
    )


def _estimate_nodal_axis(
    nodes: tuple[Fraction, ...], dimension: int, /
) -> tuple[CoordinateEnclosureEstimate, ...]:
    variable = coordinate_estimate_variable(dimension)
    result = []
    for i, x in enumerate(nodes):
        term = coordinate_estimate_constant(1, dimension)
        for j, y in enumerate(nodes):
            if i != j:
                term = coordinate_estimate_multiply(
                    term,
                    coordinate_estimate_scale(
                        coordinate_estimate_add(
                            variable, coordinate_estimate_constant(-y, dimension)
                        ),
                        1 / (x - y),
                    ),
                )
        result.append(term)
    return tuple(result)


def _estimate_linear_reference_basis(
    cell_kind: str, /
) -> tuple[CoordinateEnclosureEstimate, ...]:

    topology = imported_reference_cell.reference_cell_topology(cell_kind)
    dimension = topology.dimension
    variables = tuple(coordinate_estimate_variable(dimension) for _ in range(dimension))
    if cell_kind in ("triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        return (
            coordinate_estimate_add(
                coordinate_estimate_constant(1, dimension),
                coordinate_estimate_scale(
                    coordinate_estimate_sum(variables, dimension), -1
                ),
            ),
            *variables,
        )
    if cell_kind == "prism":
        x, y, z = variables
        triangle = (
            coordinate_estimate_add(
                coordinate_estimate_constant(1, 3),
                coordinate_estimate_scale(coordinate_estimate_add(x, y), -1),
            ),
            x,
            y,
        )
        collapse = coordinate_estimate_add(
            coordinate_estimate_constant(1, 3), coordinate_estimate_scale(z, -1)
        )
        return tuple(
            coordinate_estimate_multiply(value, factor)
            for factor in (collapse, z)
            for value in triangle
        )
    if cell_kind == "pyramid":
        u, v, w = variables
        collapse = coordinate_estimate_add(
            coordinate_estimate_constant(1, 3), coordinate_estimate_scale(w, -1)
        )
        factors = tuple(
            coordinate_estimate_product(
                tuple(
                    variable
                    if int(value)
                    else coordinate_estimate_add(
                        coordinate_estimate_constant(1, 3),
                        coordinate_estimate_scale(variable, -1),
                    )
                    for variable, value in zip((u, v), vertex[:2], strict=True)
                ),
                3,
            )
            for vertex in topology.vertices[:4]
        )
        return (
            *tuple(coordinate_estimate_multiply(value, collapse) for value in factors),
            w,
        )
    return tuple(
        coordinate_estimate_product(
            tuple(
                variable
                if int(value)
                else coordinate_estimate_add(
                    coordinate_estimate_constant(1, dimension),
                    coordinate_estimate_scale(variable, -1),
                )
                for variable, value in zip(variables, vertex, strict=True)
            ),
            dimension,
        )
        for vertex in topology.vertices
    )


def _estimate_lattice_basis(
    cell_kind: str, degree: int, /
) -> tuple[CoordinateEnclosureEstimate, ...]:

    dimension = imported_reference_cell.reference_cell_topology(cell_kind).dimension
    if degree == 1:
        return _estimate_linear_reference_basis(cell_kind)
    variables = tuple(coordinate_estimate_variable(dimension) for _ in range(dimension))
    if cell_kind in ("triangle", "tetrahedron", "prism") or cell_kind.startswith(
        "simplex:"
    ):
        simplex_dimension = 2 if cell_kind == "prism" else dimension
        barycentric = (
            coordinate_estimate_add(
                coordinate_estimate_constant(1, dimension),
                coordinate_estimate_scale(
                    coordinate_estimate_sum(variables[:simplex_dimension], dimension), -1
                ),
            ),
            *variables[:simplex_dimension],
        )
        simplex = []
        for index in product(range(degree + 1), repeat=simplex_dimension):
            if sum(index) > degree:
                continue
            term = coordinate_estimate_constant(1, dimension)
            for variable, exponent in zip(
                barycentric, (degree - sum(index), *index), strict=True
            ):
                for offset in range(exponent):
                    term = coordinate_estimate_multiply(
                        term,
                        coordinate_estimate_scale(
                            coordinate_estimate_add(
                                coordinate_estimate_scale(variable, degree),
                                coordinate_estimate_constant(-offset, dimension),
                            ),
                            Fraction(1, offset + 1),
                        ),
                    )
            simplex.append(term)
        if cell_kind != "prism":
            return tuple(simplex)
        axial = _estimate_nodal_axis(
            tuple(Fraction(i, degree) for i in range(degree + 1)), 3
        )
        return tuple(
            coordinate_estimate_multiply(a, b) for a, b in product(simplex, axial)
        )
    if cell_kind != "pyramid":
        factors = tuple(
            _estimate_nodal_axis(
                tuple(Fraction(i, degree) for i in range(degree + 1)), dimension
            )
            for _ in range(dimension)
        )
        return tuple(
            coordinate_estimate_product(choice, dimension) for choice in product(*factors)
        )
    return _estimate_pyramid_lattice(degree, variables)


def _estimate_pyramid_lattice(
    degree: int, variables: tuple[CoordinateEnclosureEstimate, ...], /
) -> tuple[CoordinateEnclosureEstimate, ...]:
    collapse = coordinate_estimate_add(
        coordinate_estimate_constant(1, 3), coordinate_estimate_scale(variables[2], -1)
    )
    indices = tuple(
        (i, j, k)
        for i in range(degree + 1)
        for j in range(degree + 1)
        for k in range(degree - max(i, j) + 1)
    )
    modes = tuple(
        coordinate_estimate_product(
            (
                coordinate_jacobi_estimate(i, 0, 0, 3),
                coordinate_jacobi_estimate(j, 0, 1, 3),
                coordinate_estimate_power(collapse, max(i, j)),
                coordinate_jacobi_estimate(k, 2 * max(i, j) + 2, 2, 3),
            ),
            3,
        )
        for i, j, k in indices
    )
    # Modal coefficients are integers. Every matrix row has one denominator
    # dividing (degree-height)**(2*degree) * degree**degree away from the apex
    # (and 2**(2*degree) at the apex). Cramer's column substitution preserves
    # that row denominator, which cancels in the ratio.
    # The n! signed products then bound the two integer determinants.
    count = len(indices)
    node_bits = max(degree, 1).bit_length()
    entry_bits = max(
        (
            m.numerator_bits
            + m.denominator_bits
            + 3 * degree * node_bits
            + max(m.terms, 1).bit_length()
            + 2
            for m in modes
        ),
        default=1,
    )
    coefficient_bits = 2 * count * entry_bits + math.factorial(count).bit_length() + 2
    coefficient = coordinate_polynomial_estimate(
        terms=1,
        dimension=3,
        degree=0,
        numerator_bits=coefficient_bits,
        denominator_bits=coefficient_bits,
    )
    cardinal = coordinate_estimate_sum(
        tuple(coordinate_estimate_multiply(mode, coefficient) for mode in modes), 3
    )
    evaluation_work = count * sum(mode.terms for mode in modes)
    return tuple(
        CoordinateEnclosureEstimate(
            cardinal.terms,
            3,
            cardinal.degree,
            cardinal.numerator_bits,
            cardinal.denominator_bits,
            cardinal.work_upper + evaluation_work,
            cardinal.bytes_upper,
        )
        for _ in indices
    )


def coordinate_estimate_derivative(
    value: CoordinateEnclosureEstimate, /
) -> CoordinateEnclosureEstimate:
    """Bound any declared source-axis derivative without inspecting coefficients."""
    if value.denominator is not None:
        a = _polynomial_estimate_part(value)
        b = value.denominator
        return coordinate_estimate_rational(
            coordinate_estimate_add(
                coordinate_estimate_multiply(coordinate_estimate_derivative(a), b),
                coordinate_estimate_multiply(a, coordinate_estimate_derivative(b)),
            ),
            coordinate_estimate_multiply(b, b),
        )
    zero = coordinate_estimate_constant(0, value.dimension)
    return _estimate_result(
        value,
        zero,
        terms=value.terms,
        degree=max(0, value.degree - 1),
        numerator_bits=value.numerator_bits + max(value.degree, 1).bit_length(),
        denominator_bits=value.denominator_bits,
        work=2 * value.terms,
    )


def coordinate_lattice_estimate(
    cell_kind: str, degree: int, /
) -> CoordinateEnclosureEstimate:
    """Bound declared lattice basis and physical fallback-gradient preparation."""

    degree = nonnegative_integer(degree, "coordinate_estimate.degree")
    dimension = imported_reference_cell.reference_cell_topology(cell_kind).dimension
    basis = _estimate_lattice_basis(cell_kind, degree)
    prepared = list(basis)
    for value in basis:
        rows = tuple(coordinate_estimate_derivative(value) for _ in range(dimension))
        prepared.extend(rows)
        if cell_kind == "pyramid":
            shifted = coordinate_estimate_add(
                coordinate_estimate_variable(3),
                coordinate_estimate_constant(Fraction(-1, 2), 3),
            )
            prepared.append(
                coordinate_estimate_add(
                    rows[2],
                    coordinate_estimate_add(
                        coordinate_estimate_multiply(shifted, rows[0]),
                        coordinate_estimate_multiply(shifted, rows[1]),
                    ),
                )
            )
    return coordinate_estimate_bank(tuple(prepared))


def _estimate_lift(
    value: CoordinateEnclosureEstimate, dimension: int, /
) -> CoordinateEnclosureEstimate:
    return CoordinateEnclosureEstimate(
        value.terms,
        dimension,
        value.degree,
        value.numerator_bits,
        value.denominator_bits,
        value.work_upper + value.terms,
        value.bytes_upper
        + _estimate_storage(
            value.terms,
            dimension,
            value.degree,
            value.numerator_bits + value.denominator_bits,
        ),
        None
        if value.denominator is None
        else _estimate_lift(value.denominator, dimension),
    )


def _estimate_weighted_bank(
    basis: tuple[CoordinateEnclosureEstimate, ...],
    rows: CoordinateSourceBank,
    /,
) -> tuple[CoordinateEnclosureEstimate, ...]:
    dimension = max((term.dimension for term in basis), default=0)
    return tuple(
        coordinate_estimate_sum(
            tuple(
                coordinate_estimate_scale(term, weight)
                for term, weight in zip(basis, row, strict=True)
            ),
            dimension,
        )
        for row in rows
    )


def coordinate_chart_argument_estimates(
    cell_kind: str, dimension: int, /
) -> tuple[CoordinateEnclosureEstimate, ...]:
    """Bound the declared physical-reference chart circuit."""
    variables = tuple(coordinate_estimate_variable(dimension) for _ in range(dimension))
    if cell_kind != "pyramid":
        return variables
    u, v, w = variables
    collapse = coordinate_estimate_add(
        coordinate_estimate_constant(1, 3), coordinate_estimate_scale(w, -1)
    )
    return (
        coordinate_estimate_add(
            coordinate_estimate_multiply(u, collapse),
            coordinate_estimate_scale(w, Fraction(1, 2)),
        ),
        coordinate_estimate_add(
            coordinate_estimate_multiply(v, collapse),
            coordinate_estimate_scale(w, Fraction(1, 2)),
        ),
        w,
    )


def coordinate_affine_argument_estimates(
    origin: np.ndarray, matrix: np.ndarray, /
) -> tuple[CoordinateEnclosureEstimate, ...]:
    """Bound an affine reference action from its declared binary64 coefficients."""
    dimension = matrix.shape[1]
    variables = tuple(coordinate_estimate_variable(dimension) for _ in range(dimension))
    return tuple(
        coordinate_estimate_add(
            coordinate_estimate_constant(Fraction(float(start)), dimension),
            coordinate_estimate_sum(
                tuple(
                    coordinate_estimate_scale(variable, Fraction(float(value)))
                    for variable, value in zip(variables, row, strict=True)
                ),
                dimension,
            ),
        )
        for start, row in zip(origin, matrix, strict=True)
    )


def _estimate_reference_arguments(
    element: CellGeometryElement, /
) -> tuple[CoordinateEnclosureEstimate, ...]:

    if not isinstance(
        element,
        (
            _cell_geometry_dispatch().PolynomialComposedCellGeometryElement,
            _cell_geometry_dispatch().RationalComposedCellGeometryElement,
        ),
    ):
        raise TypeError(
            "coordinate_estimate.reference must receive a composed source chart; got another element."
        )
    basis = source_expression_estimates(element.chart_element)
    if basis is None:
        raise ValueError(
            "coordinate_estimate.chart must have supported source expressions; got unsupported tabulator."
        )
    coefficients = tuple(
        tuple(Fraction(*pair) for pair in row) for row in element.chart_coefficients
    )
    return _estimate_weighted_bank(
        basis,
        tuple(
            tuple(row[axis] for row in coefficients)
            for axis in range(element.source_element.topological_dimension)
        ),
    )


def coordinate_chart_restriction_estimates(
    source: tuple[CoordinateEnclosureEstimate, ...],
    source_kind: str,
    target_kind: str,
    origin: np.ndarray,
    matrix: np.ndarray,
    /,
) -> tuple[CoordinateEnclosureEstimate, ...]:
    """Bound the declared source-to-target chart action and physical pullback."""
    dimension = matrix.shape[1]
    target_chart = coordinate_chart_argument_estimates(target_kind, dimension)
    arguments = tuple(
        coordinate_estimate_compose(value, target_chart)
        for value in coordinate_affine_argument_estimates(origin, matrix)
    )
    if source_kind == "pyramid":
        height = arguments[2]
        divisor = coordinate_estimate_add(
            coordinate_estimate_constant(1, dimension),
            coordinate_estimate_scale(height, -1),
        )
        reciprocal = coordinate_estimate_rational(
            coordinate_estimate_constant(1, dimension), divisor
        )
        arguments = (
            coordinate_estimate_multiply(
                coordinate_estimate_add(
                    arguments[0], coordinate_estimate_scale(height, Fraction(-1, 2))
                ),
                reciprocal,
            ),
            coordinate_estimate_multiply(
                coordinate_estimate_add(
                    arguments[1], coordinate_estimate_scale(height, Fraction(-1, 2))
                ),
                reciprocal,
            ),
            height,
        )
    return tuple(coordinate_estimate_compose(value, arguments) for value in source)


def _estimate_spline_axis(
    knots: np.ndarray, degree: int, span: int, /
) -> tuple[CoordinateEnclosureEstimate, ...]:
    vector = rational_vector(knots)
    count = len(vector) - degree - 1
    parameter = coordinate_estimate_add(
        coordinate_estimate_constant(vector[span], 2),
        coordinate_estimate_scale(
            coordinate_estimate_variable(2), vector[span + 1] - vector[span]
        ),
    )
    basis = tuple(
        coordinate_estimate_constant(int(row == span), 2)
        for row in range(len(vector) - 1)
    )
    for order in range(1, degree + 1):
        following = []
        for row in range(len(vector) - order - 1):
            left, right = (
                vector[row + order] - vector[row],
                vector[row + order + 1] - vector[row + 1],
            )
            first = (
                coordinate_estimate_constant(0, 2)
                if not left
                else coordinate_estimate_scale(
                    coordinate_estimate_multiply(
                        coordinate_estimate_add(
                            parameter, coordinate_estimate_constant(-vector[row], 2)
                        ),
                        basis[row],
                    ),
                    1 / left,
                )
            )
            second = (
                coordinate_estimate_constant(0, 2)
                if not right
                else coordinate_estimate_scale(
                    coordinate_estimate_multiply(
                        coordinate_estimate_add(
                            coordinate_estimate_constant(vector[row + order + 1], 2),
                            coordinate_estimate_scale(parameter, -1),
                        ),
                        basis[row + 1],
                    ),
                    1 / right,
                )
            )
            following.append(coordinate_estimate_add(first, second))
        basis = tuple(following)
    return basis[:count]


def _estimate_simplex_modes(
    indices: tuple[tuple[int, ...], ...], order: int, dimension: int, /
) -> tuple[CoordinateEnclosureEstimate, ...]:
    variables = tuple(coordinate_estimate_variable(dimension) for _ in range(dimension))
    barycentric = (
        coordinate_estimate_add(
            coordinate_estimate_constant(1, dimension),
            coordinate_estimate_scale(coordinate_estimate_sum(variables, dimension), -1),
        ),
        *variables,
    )
    modes = []
    for index in indices:
        weight = math.factorial(order) // math.prod(
            math.factorial(value) for value in index
        )
        term = coordinate_estimate_constant(weight, dimension)
        for argument, exponent in zip(barycentric, index, strict=True):
            term = coordinate_estimate_multiply(
                term, coordinate_estimate_power(argument, exponent)
            )
        modes.append(term)
    return tuple(modes)


def _estimate_simplex_owner(
    owner: SimplexNodalFamily, dimension: int, /
) -> tuple[CoordinateEnclosureEstimate, ...]:
    modes = _estimate_simplex_modes(owner.multiindices, owner.order, dimension)
    coefficients = np.asarray(owner.coefficients, dtype=np.float64)
    return _estimate_weighted_bank(
        modes,
        tuple(
            rational_vector(coefficients[:, column])
            for column in range(coefficients.shape[1])
        ),
    )


def _estimate_hybrid_owner(
    owner: hybrid_reference_owner.HybridReferenceFamily, /
) -> tuple[CoordinateEnclosureEstimate, ...]:
    if owner.cell_kind == "prism":
        degree = owner.orders[0]
        indices = tuple(
            (degree - i - j, i, j)
            for i in range(degree + 1)
            for j in range(degree - i + 1)
        )
        modes = _estimate_simplex_modes(indices, degree, 2)
        # The constructor solves binary64 coefficients. Their finite source
        # values have at most 1075 numerator/denominator bits, including subnormals.
        coefficient = coordinate_polynomial_estimate(
            terms=1, dimension=2, degree=0, numerator_bits=1075, denominator_bits=1075
        )
        triangle = tuple(
            coordinate_estimate_sum(
                tuple(coordinate_estimate_multiply(mode, coefficient) for mode in modes),
                2,
            )
            for _ in indices
        )
        lifted = tuple(_estimate_lift(value, 3) for value in triangle)
        axial = _estimate_nodal_axis(
            rational_vector(np.unique(np.asarray(owner.nodes)[:, 2])),
            3,
        )
        terms = tuple(
            coordinate_estimate_multiply(a, b) for a, b in product(lifted, axial)
        )
        return tuple(terms[index] for index in owner.basis_permutation)
    collapse = coordinate_estimate_add(
        coordinate_estimate_constant(1, 3),
        coordinate_estimate_scale(coordinate_estimate_variable(3), -1),
    )
    modes = tuple(
        coordinate_estimate_product(
            (
                coordinate_jacobi_estimate(i, 0, 0, 3),
                coordinate_jacobi_estimate(j, 0, 1, 3),
                coordinate_estimate_power(collapse, max(i, j)),
                coordinate_jacobi_estimate(k, 2 * max(i, j) + 2, 2, 3),
            ),
            3,
        )
        for i, j, k in owner.modal_indices
    )
    coefficients = np.asarray(owner.coefficients, dtype=np.float64)
    return _estimate_weighted_bank(
        modes,
        tuple(
            rational_vector(coefficients[:, column])
            for column in range(coefficients.shape[1])
        ),
    )


def _estimate_scalar_form_source(
    owner: imported_fem__form_elements.FormBasis,
) -> tuple[CoordinateEnclosureEstimate, ...]:
    """Bound form expressions and the same coefficient-storage admissions as lowering."""
    values = tuple(
        field[0]
        for field in imported_fem__form_expressions.component_expression_estimates(owner)
    )
    if owner.hybrid_factors is None and values:
        factors = owner.tensor_factors
        admission = (
            owner.coefficients.size
            if factors is None
            else factors.zero.size + factors.indices.size + factors.components.size
        )
        allocation = (
            owner.coefficients.size * np.dtype(np.float64).itemsize
            if factors is None
            else 0
        )
        first = values[0]
        values = (
            CoordinateEnclosureEstimate(
                first.terms,
                first.dimension,
                first.degree,
                first.numerator_bits,
                first.denominator_bits,
                first.work_upper + admission,
                first.bytes_upper + allocation,
                first.denominator,
            ),
            *values[1:],
        )
    if owner.family == "pyramid-trimmed":
        chart = coordinate_chart_argument_estimates("pyramid", 3)
        return tuple(coordinate_estimate_compose(value, chart) for value in values)
    return values


def _estimate_builtin_lagrange_source(
    element: FiniteElementSpec,
) -> tuple[CoordinateEnclosureEstimate, ...] | None:
    """Bound the built-in scalar reference basis after execution admission."""
    dimension = element.topological_dimension
    if (
        element.element_id
        != lagrange_element(element.cell_kind, element.degree).element_id
    ):
        return None
    if element.cell_kind in ("triangle", "tetrahedron"):
        variables = tuple(
            coordinate_estimate_variable(dimension) for _ in range(dimension)
        )
        barycentric = (
            coordinate_estimate_add(
                coordinate_estimate_constant(1, dimension),
                coordinate_estimate_scale(
                    coordinate_estimate_sum(variables, dimension), -1
                ),
            ),
            *variables,
        )
        if element.degree == 1:
            return barycentric
        vertices = tuple(
            coordinate_estimate_multiply(
                value,
                coordinate_estimate_add(
                    coordinate_estimate_scale(value, 2),
                    coordinate_estimate_constant(-1, dimension),
                ),
            )
            for value in barycentric
        )
        edges = tuple(
            coordinate_estimate_scale(
                coordinate_estimate_multiply(barycentric[a], barycentric[b]), 4
            )
            for a, b in ((0, 1), (1, 2), (2, 0))
        )
        return (*vertices, *edges)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    factors = tuple(
        _estimate_nodal_axis(
            rational_vector(np.unique(nodes[:, axis])),
            dimension,
        )
        for axis in range(dimension)
    )
    return tuple(
        coordinate_estimate_product(
            tuple(factors[axis][int(value)] for axis, value in enumerate(row)), dimension
        )
        for row in nodes
    )


# Owner identity checks and estimate routes stay together so unsupported scientific representations fail closed in order.
def _estimate_finite_element_source(
    element: FiniteElementSpec, /
) -> tuple[CoordinateEnclosureEstimate, ...] | None:

    if element.mapping != "identity" or element.value_shape:
        return None
    dimension = element.topological_dimension
    if type(element.tabulator) is _cell_geometry_dispatch().SweptCoordinateTabulator:
        owner = element.tabulator.source
        expected = "prism" if owner.cell_kind == "triangle" else "hexahedron"
        if (
            owner.cell_kind not in ("triangle", "quadrilateral")
            or element.cell_kind != expected
            or element.degree != owner.degree
            or element.local_dof_count != 2 * owner.local_dof_count
        ):
            return None
        basis = source_expression_estimates(owner)
        if basis is None:
            return None
        station = coordinate_estimate_variable(3)
        collapse = coordinate_estimate_add(
            coordinate_estimate_constant(1, 3), coordinate_estimate_scale(station, -1)
        )
        return tuple(
            coordinate_estimate_multiply(_estimate_lift(term, 3), factor)
            for factor in (collapse, station)
            for term in basis
        )
    if type(element.tabulator) is imported_fem__form_elements.ProxyTabulator:
        owner = element.tabulator.basis
        if (
            type(owner) is not imported_fem__form_elements.FormBasis
            or owner.form_degree != 0
            or owner.dimension != dimension
            or owner.order != element.degree
            or owner.local_dof_count != element.local_dof_count
            or element.tabulator.value_spec != element.value_spec
            or element.representation not in ("polynomial_moment", "rational_moment")
        ):
            return None
        return _estimate_scalar_form_source(owner)
    if element.representation != "point_value":
        return None
    if (
        element.degree == 0
        and element.family == "DiscontinuousLagrange"
        and element.tabulator is None
    ):
        canonical = imported_fem__reference.discontinuous_element(element.cell_kind, 0)
        return (
            (coordinate_estimate_constant(1, dimension),)
            if element.element_id == canonical.element_id and element.local_dof_count == 1
            else None
        )
    if isinstance(element.tabulator, _cell_geometry_dispatch().CoordinateTabulator):
        if (
            element.cell_kind != element.tabulator.cell_kind
            or element.degree != element.tabulator.degree
        ):
            return None
        return _estimate_lattice_basis(element.cell_kind, element.degree)
    owner = owning_tabulator_source(element.tabulator)
    if isinstance(owner, FiniteElementSpec):
        if (element.cell_kind, element.degree, element.local_dof_count) != (
            owner.cell_kind,
            owner.degree,
            owner.local_dof_count,
        ):
            return None
        return source_expression_estimates(owner)
    if isinstance(owner, imported_fem__high_order.SimplexNodalFamily):
        if element.cell_kind != owner.cell_kind or element.degree != owner.order:
            return None
        return _estimate_simplex_owner(owner, dimension)
    if isinstance(owner, imported_fem__high_order.ReferenceNodalFamily):
        if element.cell_kind != owner.cell_kind or element.degree != max(owner.orders):
            return None
        factors = tuple(
            _estimate_nodal_axis(rational_vector(np.asarray(nodes)), dimension)
            for nodes in owner.nodes_by_axis
        )
        return tuple(
            coordinate_estimate_product(choice, dimension) for choice in product(*factors)
        )
    if isinstance(owner, hybrid_reference_owner.HybridReferenceFamily):
        if element.cell_kind != owner.cell_kind or element.degree != owner.degree:
            return None
        return _estimate_hybrid_owner(owner)
    if owner is not None or element.tabulator is not None:
        return None
    return _estimate_builtin_lagrange_source(element)


def source_expression_estimates(
    element: CellGeometryElement, /
) -> tuple[CoordinateEnclosureEstimate, ...] | None:
    """Bound source expressions using one admitted host source snapshot traversal."""
    geometry = _cell_geometry_dispatch()
    work, memory = geometry.coordinate_source_snapshot_preparation_bounds(element)
    with geometry.owned_coordinate_preparation(work, memory) as budget:
        return _source_expression_estimates(element, budget=budget)


def _source_expression_estimates(
    element: CellGeometryElement, /, *, budget: CoordinateEnclosureBudget
) -> tuple[CoordinateEnclosureEstimate, ...] | None:
    """Traverse declared source routes using the preparation ledger's host banks."""
    if isinstance(element, _cell_geometry_dispatch().BarycentricCellGeometryElement):
        root = element.source_element
        while isinstance(root, _cell_geometry_dispatch().BarycentricCellGeometryElement):
            root = root.source_element
        basis = _source_expression_estimates(root, budget=budget)
        if basis is None:
            return None
        current = element
        while isinstance(
            current, _cell_geometry_dispatch().BarycentricCellGeometryElement
        ):
            weights = (
                _cell_geometry_dispatch()
                .prepare_barycentric_source_snapshot(current, budget=budget)
                .barycentric_weights
            )
            basis = _estimate_weighted_bank(
                basis,
                tuple(
                    rational_vector(weights[:, column])
                    for column in range(weights.shape[1])
                ),
            )
            current = current.source_element
        return basis
    if isinstance(element, _cell_geometry_dispatch().LayerColumnCellGeometryElement):
        wall, corners = (
            _source_expression_estimates(element.wall_element, budget=budget),
            _source_expression_estimates(element.corner_element, budget=budget),
        )
        if (
            wall is None
            or corners is None
            or element.station_axis != 2
            or element.local_dof_count != 2 * len(wall) + 4 * len(corners)
        ):
            return None
        station = coordinate_estimate_variable(3)
        factors = (
            coordinate_estimate_add(
                coordinate_estimate_constant(1, 3), coordinate_estimate_scale(station, -1)
            ),
            station,
        )
        profiles = tuple(
            coordinate_estimate_multiply(_estimate_lift(term, 3), factor)
            for factor in factors
            for term in wall
        )
        actual = tuple(
            coordinate_estimate_multiply(_estimate_lift(term, 3), factor)
            for factor in factors
            for term in corners
        )
        return (
            *profiles,
            *actual,
            *(coordinate_estimate_scale(term, -1) for term in actual),
        )
    if isinstance(element, _cell_geometry_dispatch().SplineCellGeometryElement):
        snapshot = _cell_geometry_dispatch().prepare_spline_source_snapshot(
            element, budget=budget
        )
        u = _estimate_spline_axis(
            snapshot.u_knots, element.u_degree, element.span_indices[0]
        )
        v = _estimate_spline_axis(
            snapshot.v_knots, element.v_degree, element.span_indices[1]
        )
        numerators = tuple(
            coordinate_estimate_scale(
                coordinate_estimate_multiply(first, second), Fraction(float(weight))
            )
            for first, row in zip(u, snapshot.weights, strict=True)
            for second, weight in zip(v, np.asarray(row), strict=True)
        )
        denominator = coordinate_estimate_sum(numerators, 2)
        return tuple(
            coordinate_estimate_rational(term, denominator) for term in numerators
        )
    if isinstance(
        element,
        (
            _cell_geometry_dispatch().PolynomialComposedCellGeometryElement,
            _cell_geometry_dispatch().RationalComposedCellGeometryElement,
        ),
    ):
        source = _source_expression_estimates(element.source_element, budget=budget)
        if source is None:
            return None
        arguments = _estimate_reference_arguments(element)
        return tuple(coordinate_estimate_compose(term, arguments) for term in source)
    if isinstance(element, _cell_geometry_dispatch().RestrictedCellGeometryElement):
        source = _source_expression_estimates(element.source_element, budget=budget)
        if source is None:
            return None
        return coordinate_chart_restriction_estimates(
            source,
            element.source_element.cell_kind,
            element.cell_kind,
            np.asarray(element.offset),
            np.asarray(element.matrix),
        )
    if isinstance(element, FiniteElementSpec):
        return _estimate_finite_element_source(element)
    return None


def source_basis_estimate(
    element: CellGeometryElement, /
) -> CoordinateEnclosureEstimate | None:
    """Aggregate declared source-basis preparation; unsupported tabulators remain unsupported."""
    values = source_expression_estimates(element)
    if values is None:
        return None
    bank = coordinate_estimate_bank(values)
    work, memory = (
        _cell_geometry_dispatch().coordinate_source_snapshot_preparation_bounds(element)
    )
    return CoordinateEnclosureEstimate(
        bank.terms,
        bank.dimension,
        bank.degree,
        bank.numerator_bits,
        bank.denominator_bits,
        bank.work_upper + work,
        bank.bytes_upper + memory,
        bank.denominator,
    )


def coordinate_polynomials_estimate(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    /,
    *,
    coefficient_bits_upper: int | None = None,
) -> CoordinateEnclosureEstimate | None:
    """Bound map, source-corner and cache preparation using one declared source tree."""
    values = coordinate_map_expression_estimates(
        element, local, coefficient_bits_upper=coefficient_bits_upper
    )
    if values is None:
        return None
    bank = coordinate_estimate_bank(values)
    metadata = coordinate_preparation_metadata_estimate(
        element, local, coefficient_bits_upper=coefficient_bits_upper
    )
    return CoordinateEnclosureEstimate(
        bank.terms,
        bank.dimension,
        bank.degree,
        bank.numerator_bits,
        bank.denominator_bits,
        bank.work_upper + metadata.work_upper,
        bank.bytes_upper + metadata.bytes_upper,
    )


def _coordinate_coefficient_bits(local: CoordinateCoefficients, /) -> int:
    if isinstance(local, np.ndarray):
        if not local.dtype.hasobject:
            return 1075
        values: Iterator[object] = iter(local.flat)
    else:
        values = (value for row in local for value in row)
    return max(
        (
            max(abs(value.numerator).bit_length(), value.denominator.bit_length())
            if isinstance(value, Fraction)
            else _scalar_bits(value)
            if isinstance(value, RadicalElement)
            else 1075
            for value in values
        ),
        default=1,
    )


def coordinate_map_expression_estimates(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    /,
    *,
    coefficient_bits_upper: int | None = None,
) -> tuple[CoordinateEnclosureEstimate, ...] | None:
    """Bound each physical source expression without binding carrier values as rational data."""
    basis = source_expression_estimates(element)
    if basis is None:
        return None
    if isinstance(local, np.ndarray):
        if local.ndim != 2:
            raise ValueError(
                "coordinate_estimate.coefficients must be rank two; got another rank."
            )
        count, ambient = local.shape
        bits = (
            _coordinate_coefficient_bits(local)
            if coefficient_bits_upper is None
            else coefficient_bits_upper
        )
    else:
        count = len(local)
        ambient = len(local[0]) if local else 0
        if any(len(row) != ambient for row in local):
            raise ValueError(
                "coordinate_estimate.coefficients must be rectangular; got unequal rows."
            )
        bits = (
            _coordinate_coefficient_bits(local)
            if coefficient_bits_upper is None
            else coefficient_bits_upper
        )
    if count != element.local_dof_count or not ambient:
        raise ValueError(
            "coordinate_estimate.coefficients must include every source row and physical axis; got incompatible shape."
        )
    bits = nonnegative_integer(bits, "coordinate_estimate.coefficient_bits_upper")

    topology = imported_reference_cell.reference_cell_topology(element.cell_kind)
    dimension = topology.dimension
    coefficient = coordinate_polynomial_estimate(
        terms=1, dimension=dimension, degree=0, numerator_bits=bits, denominator_bits=bits
    )
    result = coordinate_estimate_sum(
        tuple(coordinate_estimate_multiply(term, coefficient) for term in basis),
        dimension,
    )
    corners = len(topology.vertices)
    corner_work = corners * sum(value.work_upper for value in basis)
    corner_bytes = corners * sum(value.bytes_upper for value in basis)
    return tuple(
        CoordinateEnclosureEstimate(
            result.terms,
            result.dimension,
            result.degree,
            result.numerator_bits,
            result.denominator_bits,
            result.work_upper + corner_work,
            result.bytes_upper + corner_bytes,
            result.denominator,
        )
        for _ in range(ambient)
    )


def _estimate_bernstein_part(
    value: CoordinateEnclosureEstimate,
    domain: BernsteinDomain,
    dimension: int,
    degree: int,
    /,
) -> CoordinateEnclosureEstimate:
    match domain:
        case "simplex":
            nodes = math.comb(degree + dimension, dimension)
            lattice_visits = (degree + 1) ** dimension
        case "prism":
            nodes = math.comb(degree + 2, 2) * (degree + 1)
            lattice_visits = (degree + 1) ** 3
        case "box":
            nodes = (degree + 1) ** dimension
            lattice_visits = nodes
        case invalid:
            assert_never(invalid)
    entries = value.terms * nodes
    weight_bits = 2 * dimension * degree * max(degree + 1, 1).bit_length()
    denominator = value.denominator_bits + value.terms * weight_bits + 1
    numerator = (
        value.numerator_bits
        + denominator
        + weight_bits
        + max(value.terms, 1).bit_length()
        + 1
    )
    visits = (
        lattice_visits
        + nodes * (dimension + 2)
        + 2 * value.terms
        + entries * (dimension + 8)
        + 4
    )
    return CoordinateEnclosureEstimate(
        nodes,
        0,
        0,
        numerator,
        denominator,
        value.work_upper + visits,
        value.bytes_upper
        + 4
        * _estimate_storage(nodes + entries, dimension, degree, numerator + denominator),
    )


def coordinate_bernstein_estimate(
    value: CoordinateEnclosureEstimate, domain: str, dimension: int, /
) -> CoordinateEnclosureEstimate:
    """Bound rational polynomial controls or rational common-degree control ratios."""
    reference = parse(domain, BernsteinDomain, "coordinate_estimate.domain")
    if value.denominator is None:
        return _estimate_bernstein_part(value, reference, dimension, value.degree)
    degree = max(value.degree, value.denominator.degree)
    numerator = _estimate_bernstein_part(
        _polynomial_estimate_part(value), reference, dimension, degree
    )
    denominator = _estimate_bernstein_part(
        value.denominator, reference, dimension, degree
    )
    return _estimate_result(
        numerator,
        denominator,
        terms=numerator.terms,
        degree=0,
        numerator_bits=numerator.numerator_bits + denominator.denominator_bits,
        denominator_bits=denominator.numerator_bits + numerator.denominator_bits,
        work=numerator.terms + denominator.terms,
    )


def coordinate_physical_jacobian_estimates(
    coordinates: tuple[CoordinateEnclosureEstimate, ...], kind: str, dimension: int, /
) -> tuple[tuple[CoordinateEnclosureEstimate, ...], ...]:
    """Bound source derivatives and the declared pyramid physical-chart action."""
    rows = tuple(
        tuple(coordinate_estimate_derivative(value) for _ in range(dimension))
        for value in coordinates
    )
    if kind != "pyramid":
        return rows
    u, v, w = (coordinate_estimate_variable(3) for _ in range(3))
    collapse = coordinate_estimate_add(
        coordinate_estimate_constant(1, 3), coordinate_estimate_scale(w, -1)
    )
    reciprocal = coordinate_estimate_rational(
        coordinate_estimate_constant(1, 3), collapse
    )
    result = []
    for first, second, height in rows:
        x, y = (
            coordinate_estimate_multiply(first, reciprocal),
            coordinate_estimate_multiply(second, reciprocal),
        )
        z = coordinate_estimate_sum(
            (
                height,
                coordinate_estimate_multiply(
                    coordinate_estimate_add(
                        u, coordinate_estimate_constant(Fraction(-1, 2), 3)
                    ),
                    x,
                ),
                coordinate_estimate_multiply(
                    coordinate_estimate_add(
                        v, coordinate_estimate_constant(Fraction(-1, 2), 3)
                    ),
                    y,
                ),
            ),
            3,
        )
        result.append((x, y, z))
    return tuple(result)


def coordinate_determinant_estimate(
    matrix: tuple[tuple[CoordinateEnclosureEstimate, ...], ...],
    /,
    *,
    variable_dimension: int | None = None,
) -> CoordinateEnclosureEstimate:
    """Bound the explicit signed-permutation determinant source circuit."""

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("coordinate_estimate.matrix must be square; got unequal rows.")
    dimension = (
        max((value.dimension for row in matrix for value in row), default=0)
        if variable_dimension is None
        else variable_dimension
    )
    terms = []
    for permutation in imported_itertools.permutations(range(size)):
        term = coordinate_estimate_constant(1, dimension)
        for row, column in enumerate(permutation):
            term = coordinate_estimate_multiply(term, matrix[row][column])
        terms.append(coordinate_estimate_scale(term, 1))
    return coordinate_estimate_sum(tuple(terms), dimension)


def coordinate_preparation_metadata_estimate(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    /,
    *,
    coefficient_bits_upper: int | None = None,
) -> CoordinateEnclosureEstimate:
    """Bound source-bank normalization and cache binding even for unsupported sources."""

    if isinstance(local, np.ndarray):
        count = local.size
        bits = (
            _coordinate_coefficient_bits(local)
            if coefficient_bits_upper is None
            else coefficient_bits_upper
        )
    else:
        count = sum(len(row) for row in local)
        bits = (
            _coordinate_coefficient_bits(local)
            if coefficient_bits_upper is None
            else coefficient_bits_upper
        )
    leaf = coordinate_polynomial_estimate(
        terms=count,
        dimension=imported_reference_cell.reference_cell_topology(
            element.cell_kind
        ).dimension,
        degree=0,
        numerator_bits=bits,
        denominator_bits=max(count, 1) * bits,
    )
    # Two declared element identities, prepared-bank tuples and their cache
    # bindings contribute fewer than 64 visits in addition to each source row.
    snapshot_work, snapshot_memory = (
        _cell_geometry_dispatch().coordinate_source_snapshot_preparation_bounds(element)
    )
    return CoordinateEnclosureEstimate(
        leaf.terms,
        leaf.dimension,
        leaf.degree,
        leaf.numerator_bits,
        leaf.denominator_bits,
        leaf.work_upper + count + 64 + snapshot_work,
        leaf.bytes_upper + 4096 + snapshot_memory,
    )


def coordinate_estimate_equal(
    first: CoordinateEnclosureEstimate,
    second: CoordinateEnclosureEstimate,
    /,
) -> CoordinateEnclosureEstimate:
    """Bound rational quotient comparison without inventing denominator cancellation."""
    if first.denominator is None or second.denominator is None:
        # Polynomial equality and mixed polynomial/quotient kind comparisons
        # charge no arithmetic visits; source preparation is still included.
        return coordinate_estimate_bank((first, second))
    left = coordinate_estimate_multiply(
        _polynomial_estimate_part(first), second.denominator
    )
    right = coordinate_estimate_multiply(
        _polynomial_estimate_part(second), first.denominator
    )
    return coordinate_estimate_bank((left, right))


def coordinate_evaluate_estimate(
    value: CoordinateEnclosureEstimate,
    point: tuple[Fraction, ...],
    /,
) -> CoordinateEnclosureEstimate:
    """Bound rational point evaluation, including its common denominator and result retention."""
    if value.denominator is not None:
        numerator = coordinate_evaluate_estimate(_polynomial_estimate_part(value), point)
        denominator = coordinate_evaluate_estimate(value.denominator, point)
        result = coordinate_polynomial_estimate(
            terms=1,
            dimension=0,
            degree=0,
            numerator_bits=numerator.numerator_bits + denominator.denominator_bits,
            denominator_bits=numerator.denominator_bits + denominator.numerator_bits,
        )
        return CoordinateEnclosureEstimate(
            1,
            0,
            0,
            result.numerator_bits,
            result.denominator_bits,
            numerator.work_upper + denominator.work_upper + result.work_upper,
            numerator.bytes_upper + denominator.bytes_upper + result.bytes_upper,
        )
    point_bits = sum(
        max(abs(coordinate.numerator).bit_length(), coordinate.denominator.bit_length())
        for coordinate in point
    )
    numerator_bits = (
        value.numerator_bits
        + value.denominator_bits
        + value.degree * point_bits
        + max(value.terms, 1).bit_length()
        + 2
    )
    denominator_bits = value.denominator_bits + value.degree * sum(
        coordinate.denominator.bit_length() for coordinate in point
    )
    result = coordinate_polynomial_estimate(
        terms=1,
        dimension=0,
        degree=0,
        numerator_bits=numerator_bits,
        denominator_bits=denominator_bits,
    )
    # evaluate reserves one visit per support coefficient; result retention
    # adds the scalar Fraction graph without assuming affine source degree.
    return CoordinateEnclosureEstimate(
        1,
        0,
        0,
        result.numerator_bits,
        result.denominator_bits,
        value.work_upper + value.terms + result.work_upper,
        value.bytes_upper + result.bytes_upper,
    )
