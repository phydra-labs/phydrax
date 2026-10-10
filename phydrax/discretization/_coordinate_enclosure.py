#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only source-level polynomial bounds; every coefficient is exact dyadic data.

No basis tabulation, floating interpolation inverse, or partition-of-unity test
is used as an error bound. Pyramids use the collapsed cube; its top face is one
physical apex, and the physical determinant divides out the exact collapse.
"""

from __future__ import annotations

import math
import operator
import sys
from collections.abc import ItemsView, Iterator, Mapping, ValuesView
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, lru_cache
from itertools import product
from types import MappingProxyType
from typing import (
    assert_never,
    Callable,
    Literal,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
    TypeAlias,
)

import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..typing import parse
from .fem._reference import FiniteElementSpec, lagrange_element


if TYPE_CHECKING:
    from .._meshcore import NativeHostStorageWorkspace
    from ..geometry._mesh_certificates import _Facets
    from ..geometry.brep._patches import _MeridianSpans, RationalBezierPiece
    from ._cell_geometry import (
        CellGeometryElement,
        CellGeometrySpec,
        LayerColumnCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )
    from ._cell_mesh import CellMesh
    from .fem._high_order import SimplexNodalFamily

type Polynomial = dict[tuple[int, ...], Fraction]
type CoordinateSourceBank = tuple[tuple[Fraction, ...], ...]
type CoordinateCoefficients = np.ndarray | CoordinateSourceBank
type CoordinateCornerWeights = tuple[tuple[tuple[int, Fraction], ...], ...]
type OrderedReferenceArguments = tuple[tuple[tuple[tuple[int, ...], Fraction], ...], ...]
BernsteinDomain: TypeAlias = Literal["simplex", "prism", "box"]
"""Reference domain of a Bernstein control net: simplex, triangle x interval, or unit box."""

CoordinateEnclosureResource: TypeAlias = Literal[
    "coefficient_work", "polynomial_storage", "retained_basis"
]


class RationalEnclosureError(ValueError):
    """An exact rational expression has an unresolved denominator or chart limit."""


class RationalPolynomial:
    """Exact quotient of source expressions; no interpolation or sampled fit."""

    def __init__(self, numerator: Polynomial, denominator: Polynomial, /) -> None:
        if not denominator:
            raise ValueError(
                "A rational source expression cannot have a zero denominator."
            )
        self.numerator = numerator
        self.denominator = denominator

    def __bool__(self) -> bool:
        return bool(self.numerator)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RationalPolynomial):
            return False
        return multiply(self.numerator, other.denominator) == multiply(
            other.numerator, self.denominator
        )


type Expression = Polynomial | RationalPolynomial


@dataclass(frozen=True, slots=True)
class PreparedCoordinateCells:
    """Actual source cell records earned by one complete mapped preparation."""

    cell_ids: tuple[int, ...]
    descriptors: tuple[
        tuple[str, CellGeometryElement, CoordinateSourceBank, tuple[int, ...]], ...
    ]
    coordinates: tuple[tuple[Expression, ...], ...]


_BINARY64_MAXIMUM = Fraction(float(np.finfo(np.float64).max))


class CoordinateEnclosureResourceError(ValueError):
    """Pre-operation resource refusal with actual work/storage ledger evidence."""

    def __init__(
        self,
        resource: CoordinateEnclosureResource,
        limit: int,
        requested: int,
        completed: int,
        /,
        *,
        admission: bool = False,
    ) -> None:
        resource = parse(resource, CoordinateEnclosureResource, "resource")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in (limit, requested, completed)
        ):
            raise TypeError(
                "Coordinate refusal quantities must be actual integer ledger values."
            )
        if not isinstance(admission, bool):
            raise TypeError("Coordinate refusal admission context must be explicit.")
        if (
            min(limit, requested, completed) < 0
            or requested <= limit
            or requested < completed
            or ((requested == completed or completed > limit) and not admission)
        ):
            raise ValueError(
                "Coordinate refusal quantities must describe an actual pre-operation resource refusal."
            )
        self.admission = admission
        self.resource, self.limit, self.requested, self.completed = (
            resource,
            limit,
            requested,
            completed,
        )
        operation = "admission" if admission else "expansion"
        super().__init__(
            f"Coordinate enclosure {resource} requires {requested} above limit {limit} before {operation}; completed {completed}."
        )


class CoordinateLiveStorage:
    """One explicitly bounded live owner, released only after its last use."""

    def __init__(self, budget: CoordinateEnclosureBudget, /) -> None:
        self._budget = budget
        self.bound = 0
        self._closed = False

    def set_bound(self, total_bytes_upper: int, /, *, work: int = 0) -> None:
        if self._closed:
            raise RuntimeError("A closed coordinate live owner cannot grow.")
        if isinstance(total_bytes_upper, bool) or not isinstance(total_bytes_upper, int):
            raise TypeError("Live storage bounds must be explicit integers.")
        if total_bytes_upper < 0:
            raise ValueError("Live storage bounds must be nonnegative.")
        if isinstance(work, bool) or not isinstance(work, int):
            raise TypeError("Live storage work must be an explicit integer.")
        if work < 0:
            raise ValueError("Live storage work must be nonnegative.")
        budget = self._budget
        delta = total_bytes_upper - self.bound
        if delta >= 0:
            budget.reserve(work, delta)
        else:
            budget.reserve(work)
            budget._set_host_storage(
                budget.retained_basis_bytes + budget.temporary_bytes_upper + delta
            )
            budget.temporary_bytes_upper += delta
        budget._live_bytes_upper += delta
        self.bound = total_bytes_upper

    def close(self) -> None:
        if not self._closed:
            self.set_bound(0)
            self._closed = True


class CoordinateEnclosureBudget:
    """Host coefficient-term work and retained canonical-basis storage ledger.

    Primitive term visits are charged before their loops. Temporary polynomial
    storage has an explicit conservative CPython object-storage upper bound,
    not an RSS measurement; scopes release that bound between physical cells.
    Cached basis objects are counted by actual retained recursive object size.
    """

    def __init__(self, maximum_work_units: int, maximum_memory_bytes: int, /) -> None:
        if maximum_work_units < 0 or maximum_memory_bytes < 0:
            raise ValueError("Coordinate enclosure budgets must be nonnegative.")
        self.maximum_work_units = maximum_work_units
        self.maximum_memory_bytes = maximum_memory_bytes
        self.work_units = 0
        self.required_work_units = 0
        self.native_charged_work_units = 0
        self.retained_basis_bytes = 0
        self.temporary_bytes_upper = 0
        self._live_bytes_upper = 0
        self.peak_bytes_upper = 0
        # Element caches are keyed by canonical element identity (see
        # ``_element_identity``), never object identity: equal elements share
        # one prepared basis, so hits depend only on the actual call sequence.
        self.basis_cache: dict[tuple[str, str], tuple[Polynomial, ...] | None] = {}
        self.basis_profile_cache: dict[tuple[str, str], tuple[int, int, int]] = {}
        self.affine_injectivity_cache: set[tuple[tuple[Fraction, ...], ...]] = set()
        self.exact_orientation_preparation_cache: dict[
            tuple[object, ...], tuple[tuple[int | Fraction, ...], int | Fraction, int]
        ] = {}
        self.source_derivative_bounds_cache: dict[
            tuple[str, int, bytes],
            tuple[np.ndarray, np.ndarray],
        ] = {}
        self.lattice_cache: dict[tuple[str, int], tuple[Polynomial, ...]] = {}
        self.bernstein_lattice_cache: dict[
            tuple[BernsteinDomain, tuple[int, ...], int], tuple[tuple[int, ...], ...]
        ] = {}
        self.bernstein_weight_cache: dict[
            tuple[BernsteinDomain, tuple[int, ...], int, tuple[int, ...]],
            tuple[int, tuple[tuple[int, int], ...]],
        ] = {}
        self.reference_monomial_cache: dict[
            tuple[OrderedReferenceArguments, tuple[int, ...]],
            Mapping[tuple[int, ...], Fraction],
        ] = {}
        self.source_expression_cache: dict[
            tuple[str, str], tuple[Expression, ...] | None
        ] = {}
        self.reference_composition_cache: dict[
            tuple[str, str], tuple[Polynomial, ...]
        ] = {}
        self.prepared_reference_cache: dict[
            tuple[str, str], PreparedPolynomialArguments
        ] = {}
        self.corner_weight_cache: dict[
            tuple[str, str], tuple[tuple[str, str], CoordinateCornerWeights]
        ] = {}
        self.corner_linear_cache: dict[
            tuple[
                tuple[str, str], CoordinateSourceBank, tuple[tuple[int, Fraction], ...]
            ],
            tuple[Fraction, ...],
        ] = {}
        self.multi_affine_cache: dict[tuple[str, str], bool] = {}
        self.density_cache: dict[
            tuple[str, str, CoordinateSourceBank], tuple[Expression, ...]
        ] = {}
        self.coordinate_cache: dict[
            tuple[str, str, CoordinateSourceBank], tuple[Expression, ...]
        ] = {}
        self.coordinate_corner_cache: dict[
            tuple[str, str, CoordinateSourceBank], tuple[tuple[Fraction, ...], ...]
        ] = {}
        self.polynomial_cache: dict[
            tuple[str, str, CoordinateSourceBank], tuple[Polynomial, ...]
        ] = {}
        self.coefficient_preparation_cache: dict[
            tuple[str, str, CoordinateSourceBank],
            tuple[CoordinateSourceBank, tuple[str, str, CoordinateSourceBank]],
        ] = {}
        self.source_bank_cache: dict[tuple[str, str], CoordinateSourceBank] = {}
        self.scope_cache: dict[
            tuple[str, str, str], tuple[bool, np.ndarray, CoordinateSourceBank]
        ] = {}
        self.face_chart_cache: dict[
            tuple[str, str, CoordinateSourceBank, str, tuple[int, ...]],
            tuple[tuple[tuple[Expression, ...], str], ...],
        ] = {}
        self.face_flux_cache: dict[
            tuple[str, str, CoordinateSourceBank, str, tuple[int, ...]],
            Fraction | None,
        ] = {}
        self.bspline_span_cache: dict[str, tuple[RationalBezierPiece, ...]] = {}
        self.meridian_span_cache: dict[str, _MeridianSpans] = {}
        self.prepared_cell_cache: dict[tuple[str, str, str], PreparedCoordinateCells] = {}
        self.facet_cache: dict[tuple[str, str, str], _Facets] = {}
        self.retained_objects: dict[int, object] = {}
        self._stage_work_bounds: list[int] = []
        self._stage_memory_bounds: list[int] = []
        self._host_workspace: NativeHostStorageWorkspace | None = None

    def _set_host_storage(self, storage_upper: int, /) -> None:
        """Bind live exact objects to the original ambient native allowance."""
        workspace = self._host_workspace
        if workspace is None:
            return
        # The native pool may refuse work, queries, cavity, storage, metadata
        # allocation or time before resizing. Preserve that actual status and
        # its ended evidence; a generic capacity status is not a byte proof.
        workspace.set_bound(storage_upper)

    def reserve(self, work: int, storage_upper: int = 0, /) -> None:
        proposed_work = self.work_units + work
        proposed_storage = (
            self.retained_basis_bytes + self.temporary_bytes_upper + storage_upper
        )
        self.required_work_units = max(self.required_work_units, proposed_work)
        work_limit = min((self.maximum_work_units, *self._stage_work_bounds))
        storage_limit = min((self.maximum_memory_bytes, *self._stage_memory_bounds))
        if proposed_work > work_limit:
            raise CoordinateEnclosureResourceError(
                "coefficient_work",
                work_limit,
                proposed_work,
                self.work_units,
                admission=proposed_work == self.work_units
                or self.work_units > work_limit,
            )
        if proposed_storage > storage_limit:
            completed_storage = self.retained_basis_bytes + self.temporary_bytes_upper
            raise CoordinateEnclosureResourceError(
                "polynomial_storage",
                storage_limit,
                proposed_storage,
                completed_storage,
                admission=proposed_storage == completed_storage
                or completed_storage > storage_limit,
            )
        from .._meshcore import current_native_execution_budget

        native = current_native_execution_budget()
        if native is not None:
            # All measured but not yet debited host visits remain owed to this
            # actual root. Dry admission does not charge or renew the counter.
            native.admit_work_bound(
                work + self.work_units - self.native_charged_work_units
            )
        if storage_upper:
            self._set_host_storage(proposed_storage)
        self.work_units = proposed_work
        self.temporary_bytes_upper += storage_upper
        self.peak_bytes_upper = max(self.peak_bytes_upper, proposed_storage)

    def admit_work_bound(self, maximum_work: int, /) -> None:
        """Dry-admit an owning batch upper bound without inventing actual visits."""
        from .._meshcore import current_native_execution_budget

        if (
            isinstance(maximum_work, bool)
            or not isinstance(maximum_work, int)
            or maximum_work < 0
        ):
            raise TypeError(
                "A coordinate work bound must be a nonnegative explicit integer."
            )
        requested = self.work_units + maximum_work
        self.required_work_units = max(self.required_work_units, requested)
        limit = min((self.maximum_work_units, *self._stage_work_bounds))
        if requested > limit:
            raise CoordinateEnclosureResourceError(
                "coefficient_work",
                limit,
                requested,
                self.work_units,
                admission=requested == self.work_units or self.work_units > limit,
            )
        native = current_native_execution_budget()
        if native is not None:
            native.admit_work_bound(
                maximum_work + self.work_units - self.native_charged_work_units
            )

    def charge_native_work(self, actual_units: int, /) -> None:
        """Charge this ledger's actual host visits once, never native primitives.

        The counter advances only after a successful charge to the actual
        ambient native owner. Standalone algebra has no native owner and does
        not invent an execution record or a measured native memory debit.
        """
        from .._meshcore import current_native_execution_budget

        if isinstance(actual_units, bool) or not isinstance(actual_units, int):
            raise TypeError("Actual coordinate work must be an explicit integer.")
        if not 0 <= actual_units <= self.work_units - self.native_charged_work_units:
            raise ValueError(
                "Native coordinate work must be actual uncharged visits of this ledger."
            )
        native = current_native_execution_budget()
        if native is None:
            return
        native.charge(work=actual_units)
        self.native_charged_work_units += actual_units

    def _object_storage_profile(
        self,
        root: object,
        existing: Mapping[int, object],
        /,
        *,
        retained: bool,
    ) -> tuple[dict[int, object], int]:
        self.reserve(0, 256)
        pending: list[object] = [root]
        seen: dict[int, object] = {}
        size = 256 if retained and not existing else 0
        storage_limit = min((self.maximum_memory_bytes, *self._stage_memory_bounds))
        while pending:
            self.reserve(1)
            value = pending.pop()
            identifier = id(value)
            if identifier in seen or identifier in existing:
                continue
            # Conservative CPython hash-index slots. The transient traversal
            # index coexists with a retained index; live owners use the same
            # upper bound even though their lifetime is tracked separately.
            self.reserve(0, 128)
            size += sys.getsizeof(value) + 128
            if (
                retained
                and self.retained_basis_bytes + self.temporary_bytes_upper + size
                > storage_limit
            ):
                raise CoordinateEnclosureResourceError(
                    "retained_basis",
                    storage_limit,
                    self.retained_basis_bytes + self.temporary_bytes_upper + size,
                    self.retained_basis_bytes,
                )
            seen[identifier] = value
            if isinstance(value, dict):
                self.reserve(0, 64 + 32 * len(value))
                pending.extend(value.keys())
                pending.extend(value.values())
            elif isinstance(value, tuple):
                self.reserve(0, 64 + 16 * len(value))
                pending.extend(value)
            elif isinstance(value, RationalPolynomial):
                self.reserve(0, 96)
                pending.extend((value.numerator, value.denominator))
            elif isinstance(value, Fraction):
                self.reserve(0, 96)
                pending.extend((value.numerator, value.denominator))
        return seen, size

    def live_object_storage_upper(self, value: object, /) -> int:
        """Measure one still-live exact object graph without retaining it forever."""
        with self.temporary_scope():
            _, size = self._object_storage_profile(value, {}, retained=False)
        return size

    def retain_basis(self, basis: tuple[object, ...], /) -> None:
        # The cache must own its objects: ephemeral wrapper tuple IDs otherwise
        # recycle and incorrectly suppress distinct, still-live coefficients.
        with self.temporary_scope():
            seen, size = self._object_storage_profile(
                basis, self.retained_objects, retained=True
            )
            self._set_host_storage(
                self.retained_basis_bytes + size + self.temporary_bytes_upper
            )
            self.retained_basis_bytes += size
            self.retained_objects.update(seen)
            self.peak_bytes_upper = max(
                self.peak_bytes_upper,
                self.retained_basis_bytes + self.temporary_bytes_upper,
            )

    @contextmanager
    def bound_stage(
        self,
        maximum_work_units: int,
        maximum_memory_bytes: int,
        /,
        *,
        starting_work_units: int | None = None,
    ) -> Iterator[CoordinateEnclosureBudget]:
        """Bound a stage or resumed owner's original allowance on this ledger."""
        if (
            isinstance(maximum_work_units, bool)
            or not isinstance(maximum_work_units, int)
            or isinstance(maximum_memory_bytes, bool)
            or not isinstance(maximum_memory_bytes, int)
            or maximum_work_units < 0
            or maximum_memory_bytes < 0
        ):
            raise ValueError(
                "Stage work and memory bounds must be nonnegative explicit integers."
            )
        start = self.work_units if starting_work_units is None else starting_work_units
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or start > self.work_units
        ):
            raise ValueError(
                "Stage starting work must be an observed nonnegative ledger count."
            )
        self._stage_work_bounds.append(
            min(
                start + maximum_work_units,
                self.maximum_work_units,
                *self._stage_work_bounds,
            )
        )
        self._stage_memory_bounds.append(
            min(
                maximum_memory_bytes,
                self.maximum_memory_bytes,
                *self._stage_memory_bounds,
            )
        )
        try:
            self.reserve(0)
            yield self
        finally:
            self._stage_memory_bounds.pop()
            self._stage_work_bounds.pop()

    @contextmanager
    def live_storage(self) -> Iterator[CoordinateLiveStorage]:
        owner = CoordinateLiveStorage(self)
        primary: BaseException | None = None
        try:
            yield owner
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                owner.close()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f"Coordinate live-storage release also refused: {error}")

    @contextmanager
    def temporary_scope(self) -> Iterator[None]:
        prior = self.temporary_bytes_upper
        prior_live = self._live_bytes_upper
        primary: BaseException | None = None
        try:
            yield
        except BaseException as error:
            primary = error
            raise
        finally:
            remaining = prior + self._live_bytes_upper - prior_live
            try:
                self._set_host_storage(self.retained_basis_bytes + remaining)
                self.temporary_bytes_upper = remaining
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(
                    f"Coordinate temporary-storage release also refused: {error}"
                )

    @contextmanager
    def activate(self) -> Iterator[CoordinateEnclosureBudget]:
        from .._meshcore import current_native_execution_budget

        native = current_native_execution_budget()
        if _COORDINATE_BUDGET.get() is self and (
            self._host_workspace is not None or native is None
        ):
            yield self
            return
        token = _COORDINATE_BUDGET.set(self)
        try:
            if native is None or native.deferred_worker_active:
                yield self
            else:
                with native.host_workspace() as workspace:
                    self._host_workspace = workspace
                    try:
                        if self.retained_basis_bytes + self.temporary_bytes_upper:
                            self._set_host_storage(
                                self.retained_basis_bytes + self.temporary_bytes_upper
                            )
                        yield self
                    finally:
                        self._host_workspace = None
        finally:
            _COORDINATE_BUDGET.reset(token)


_COORDINATE_BUDGET: ContextVar[CoordinateEnclosureBudget | None] = ContextVar(
    "coordinate_enclosure_budget", default=None
)


def coordinate_enclosure_budget(
    maximum_work_units: int, maximum_memory_bytes: int, /
) -> CoordinateEnclosureBudget:
    """Borrow the original active ledger, or own one at a genuine direct entry."""
    budget = _COORDINATE_BUDGET.get()
    return (
        CoordinateEnclosureBudget(maximum_work_units, maximum_memory_bytes)
        if budget is None
        else budget
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class PreparedPolynomialSupport(Mapping[tuple[int, ...], Fraction]):
    """Immutable exact support with a profile earned from every actual coefficient."""

    _coefficients: Mapping[tuple[int, ...], Fraction]
    numerator_bits: int
    denominator: int

    def __init__(self, coefficients: Mapping[tuple[int, ...], Fraction], /) -> None:
        budget = _COORDINATE_BUDGET.get()
        if budget is not None:
            budget.reserve(len(coefficients), 96 + 64 * len(coefficients))
        snapshot = dict(coefficients)
        view = MappingProxyType(snapshot)
        numerator, denominator = _coefficient_profile((snapshot,), exact_denominator=True)
        object.__setattr__(self, "_coefficients", view)
        object.__setattr__(self, "numerator_bits", numerator)
        object.__setattr__(self, "denominator", denominator)
        if budget is not None:
            budget.retain_basis((self, snapshot, view, numerator, denominator))

    def __getitem__(self, index: tuple[int, ...]) -> Fraction:
        return self._coefficients[index]

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return iter(self._coefficients)

    def __len__(self) -> int:
        return len(self._coefficients)

    def items(self) -> ItemsView[tuple[int, ...], Fraction]:
        return self._coefficients.items()

    def values(self) -> ValuesView[Fraction]:
        return self._coefficients.values()


def _coefficient_profile(
    polynomials: tuple[Mapping[tuple[int, ...], Fraction], ...],
    /,
    *,
    exact_denominator: bool = False,
) -> tuple[int, int]:
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(
            sum(
                1
                if isinstance(polynomial, PreparedPolynomialSupport)
                else len(polynomial)
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
            numerator_bits = max(numerator_bits, abs(value.numerator).bit_length())
            denominator = math.lcm(denominator, value.denominator)
    return numerator_bits, denominator if exact_denominator else denominator.bit_length()


def _reserve_polynomial(work: int, terms: int, dimension: int, bits: int, /) -> None:
    budget = _COORDINATE_BUDGET.get()
    if budget is None:
        return
    # Dictionary capacity/key tuples/Fraction objects plus two bounded PyLongs.
    digits = (
        max(bits, 1) + sys.int_info.bits_per_digit - 1
    ) // sys.int_info.bits_per_digit
    per_term = (
        512 + 32 * dimension + 2 * (sys.getsizeof(0) + digits * sys.int_info.sizeof_digit)
    )
    budget.reserve(work, 256 + terms * per_term)


@runtime_checkable
class _BoundSourceTabulator(Protocol):
    @property
    def __self__(self) -> object: ...

    @property
    def __func__(self) -> Callable[..., tuple[ArrayLike, ArrayLike]]: ...


def constant(value: Fraction | int, dimension: int) -> Polynomial:
    if _COORDINATE_BUDGET.get() is not None:
        bits = (
            max(abs(value.numerator).bit_length(), value.denominator.bit_length())
            if isinstance(value, Fraction)
            else abs(value).bit_length()
        )
        _reserve_polynomial(1, int(bool(value)), dimension, bits)
    return {(0,) * dimension: Fraction(value)} if value else {}


def _accumulate(
    target: Polynomial,
    second: Polynomial,
    /,
    *,
    target_profile: tuple[int, int] | None = None,
) -> None:
    """Add in place; an owning reduction may reuse its unchanged target profile."""
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        if target_profile is None:
            numerator, denominator = _coefficient_profile((target, second))
        else:
            numerator, second_denominator = _coefficient_profile(
                (second,), exact_denominator=True
            )
            budget.reserve(0, 96 + sys.getsizeof(second_denominator))
            if target:
                # The private composition result has not changed since this
                # exact full-content profile was prepared after the last add.
                budget.reserve(1)
                numerator = max(numerator, target_profile[0])
                second_denominator = math.lcm(second_denominator, target_profile[1])
            denominator = second_denominator.bit_length()
        dimension = len(next(iter(target or second), ()))
        _reserve_polynomial(
            len(target) + len(second),
            len(target) + len(second),
            dimension,
            numerator + denominator + 1,
        )
    for index, value in second.items():
        total = target.get(index, Fraction(0)) + value
        if total:
            target[index] = total
        else:
            target.pop(index, None)


def add(first: Polynomial, second: Polynomial) -> Polynomial:
    result = dict(first)
    _accumulate(result, second)
    return result


def scale(
    polynomial: Mapping[tuple[int, ...], Fraction], value: Fraction | int
) -> Polynomial:
    if not value or not polynomial:
        return {}
    if _COORDINATE_BUDGET.get() is not None:
        numerator, denominator = _coefficient_profile((polynomial,))
        value_bits = (
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            if isinstance(value, Fraction)
            else abs(value).bit_length()
        )
        bits = numerator + denominator + value_bits
        _reserve_polynomial(
            2 * len(polynomial), len(polynomial), len(next(iter(polynomial), ())), bits
        )
    return {
        index: product
        for index, coefficient in polynomial.items()
        if (product := coefficient * value)
    }


def multiply(
    first: Mapping[tuple[int, ...], Fraction], second: Mapping[tuple[int, ...], Fraction]
) -> Polynomial:
    # An empty exact support is the zero polynomial. No coefficient profile,
    # dimension traversal or product storage is needed for its annihilator.
    if not first or not second:
        return {}
    square = first is second
    if _COORDINATE_BUDGET.get() is not None:
        first_num, first_den = _coefficient_profile((first,))
        second_num, second_den = _coefficient_profile((second,))
        full_pairs = len(first) * len(second)
        pairs = len(first) * (len(first) + 1) // 2 if square else full_pairs
        bits = (
            first_num
            + second_num
            + first_den
            + second_den
            + max(full_pairs, 1).bit_length()
        )
        dimension = len(next(iter(first or second), ()))
        budget = _COORDINATE_BUDGET.get()
        if budget is not None:
            budget.reserve(dimension * (len(first) + len(second)))
        reachable = math.prod(
            max((index[axis] for index in first), default=0)
            + max((index[axis] for index in second), default=0)
            + 1
            for axis in range(dimension)
        )
        _reserve_polynomial(pairs, min(full_pairs, reachable), dimension, bits)
    dimension = len(next(iter(first)))
    if any(len(index) != dimension for index in first) or any(
        len(index) != dimension for index in second
    ):
        raise ValueError("Polynomial factors must share one parameter dimension.")
    # Exact products are accumulated as integers over the product of the two
    # common denominators; insertion order and normalized values are unchanged.
    first_common = math.lcm(*(value.denominator for value in first.values()))
    second_common = (
        first_common
        if square
        else math.lcm(*(value.denominator for value in second.values()))
    )
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
    polynomial: Mapping[tuple[int, ...], Fraction], exponent: int, dimension: int
) -> Polynomial:
    result = constant(1, dimension)
    for _ in range(exponent):
        result = multiply(result, polynomial)
    return result


def axes(dimension: int) -> tuple[Polynomial, ...]:
    if _COORDINATE_BUDGET.get() is not None:
        _reserve_polynomial(dimension, dimension, dimension, 1)
    return tuple(
        {tuple(int(i == axis) for i in range(dimension)): Fraction(1)}
        for axis in range(dimension)
    )


def derivative(polynomial: Polynomial, axis: int) -> Polynomial:
    if _COORDINATE_BUDGET.get() is not None:
        numerator, denominator = _coefficient_profile((polynomial,))
        degree = max((index[axis] for index in polynomial), default=0)
        _reserve_polynomial(
            len(polynomial),
            len(polynomial),
            len(next(iter(polynomial), ())),
            numerator + denominator + max(degree, 1).bit_length(),
        )
    return {
        tuple(value - int(i == axis) for i, value in enumerate(index)): coefficient
        * index[axis]
        for index, coefficient in polynomial.items()
        if index[axis]
    }


def _axis_affine_arguments(
    arguments: tuple[Mapping[tuple[int, ...], Fraction], ...],
    dimension: int,
) -> tuple[tuple[Fraction, Fraction, bool], ...] | None:
    if len(arguments) != dimension:
        return None
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(sum(len(argument) for argument in arguments))
    zero = (0,) * dimension
    result = []
    for axis, argument in enumerate(arguments):
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


def _axis_affine_polynomial(
    polynomial: Polynomial,
    arguments: tuple[tuple[Fraction, Fraction, bool], ...],
) -> Polynomial:
    """Exact separable affine composition, retaining the original accumulator order."""
    dimension = len(arguments)
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(len(polynomial))
    if any(len(index) != dimension for index in polynomial):
        raise ValueError(
            "Composition arguments must match the polynomial parameter dimension."
        )
    if not polynomial:
        return {}
    if all(not offset and factor == 1 for offset, factor, _ in arguments):
        if budget is not None:
            budget.reserve(len(polynomial))
        bits = max(
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            for value in polynomial.values()
        )
        _reserve_polynomial(len(polynomial), len(polynomial), dimension, bits)
        return {index: value for index, value in polynomial.items() if value}
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
    _reserve_polynomial(
        contributions * (dimension + 1) + factor_terms,
        math.prod(degree + 1 for degree in degrees) + factor_terms,
        dimension,
        bits,
    )
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
    return {index: Fraction(value, denominator) for index, value in totals.items()}


@dataclass(frozen=True, slots=True, eq=False)
class PreparedPolynomialArguments:
    """Complete immutable reference support and its earned classification."""

    arguments: tuple[Mapping[tuple[int, ...], Fraction], ...]
    ordered_key: OrderedReferenceArguments
    dimension: int
    axis_affine: tuple[tuple[Fraction, Fraction, bool], ...] | None


class ArgumentComposition:
    """Composition with fixed polynomial arguments.

    Argument powers and their prefix products are formed once and reused by
    every polynomial composed with the same arguments. ``c * (P0^a P1^b ...)``
    has the same exact coefficients and key order as ``((c * P0^a) * P1^b) ...``
    because a nonzero scalar creates no zeros.
    """

    def __init__(
        self, arguments: tuple[Polynomial, ...] | PreparedPolynomialArguments, /
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
            self._axis_affine = _axis_affine_arguments(arguments, self.dimension)
            self._reference_key = None
            budget = _COORDINATE_BUDGET.get()
            if budget is not None and self._axis_affine is None:
                terms = sum(len(argument) for argument in arguments)
                # Dynamic caller banks still earn their current full snapshot.
                budget.reserve(terms, 64 + 64 * len(arguments) + 64 * terms)
                self._reference_key = tuple(
                    tuple(argument.items()) for argument in arguments
                )
        self._powers: tuple[dict[int, Polynomial], ...] = tuple(
            {} for _ in self.arguments
        )
        self._products: dict[tuple[int, ...], Mapping[tuple[int, ...], Fraction]] = {
            (): constant(1, self.dimension)
        }

    def _monomial(self, index: tuple[int, ...], /) -> Mapping[tuple[int, ...], Fraction]:
        local = self._products.get(index)
        if local is not None:
            return local
        budget = _COORDINATE_BUDGET.get()
        supports = self.arguments if self._reference_key is None else self._reference_key
        if any(
            exponent > 0 and not argument
            for exponent, argument in zip(index, supports, strict=True)
        ):
            # Inspect the complete factor support before executing a prefix.
            # A later exact zero annihilates all earlier reference powers.
            return {}
        key = None
        if budget is not None:
            if self._reference_key is None:
                terms = sum(len(argument) for argument in self.arguments)
                budget.reserve(terms, 64 + 64 * len(self.arguments) + 64 * terms)
                self._reference_key = tuple(
                    tuple(argument.items()) for argument in self.arguments
                )
            key = self._reference_key, index
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
                        if budget is not None:
                            budget.reserve(len(entries), 256 + 128 * len(entries))
                        argument = dict(entries)
                    argument_power = self._powers[axis][exponent] = power(
                        argument, exponent, self.dimension
                    )
                cached = self._products[prefix] = multiply(factor, argument_power)
            factor = cached
        if budget is not None and key is not None:
            # Snapshot and profile the complete actual support once. Its
            # read-only coefficients and exact profile then share this owner.
            immutable = PreparedPolynomialSupport(factor)
            budget.retain_basis((key, immutable))
            budget.reference_monomial_cache[key] = immutable
            self._products[index] = immutable
            return immutable
        return factor

    def __call__(self, polynomial: Polynomial, /) -> Polynomial:
        if self._axis_affine is not None:
            return _axis_affine_polynomial(polynomial, self._axis_affine)
        result: Polynomial = {}
        budget = _COORDINATE_BUDGET.get()
        target_storage = 0
        target_profile = (1, 1)
        for index, value in polynomial.items():
            if len(index) != len(self.arguments):
                raise ValueError(
                    "Composition arguments must match the polynomial parameter dimension."
                )
            monomial = self._monomial(index)
            if not monomial:
                # The complete reference support already proved this product
                # zero. No coefficient addition or target mutation executes,
                # so do not re-profile/re-reserve the unchanged accumulator.
                continue
            if budget is None:
                _accumulate(result, scale(monomial, value))
                continue
            # Powers/prefix products outlive each term and remain charged.
            # Scaled terms and replaced accumulator buffers do not. Preserve
            # the live result bound while releasing only completed scratch.
            with budget.temporary_scope():
                _accumulate(result, scale(monomial, value), target_profile=target_profile)
                target_profile = _coefficient_profile((result,), exact_denominator=True)
                numerator, denominator = target_profile
                before_target = budget.temporary_bytes_upper
                _reserve_polynomial(
                    0, len(result), self.dimension, numerator + denominator.bit_length()
                )
                budget.reserve(0, 128 + sys.getsizeof(denominator))
                next_target = budget.temporary_bytes_upper - before_target
            budget.temporary_bytes_upper -= target_storage
            budget.reserve(0, next_target)
            target_storage = next_target
        return result


def compose(
    polynomial: Polynomial,
    arguments: tuple[Polynomial, ...] | PreparedPolynomialArguments,
) -> Polynomial:
    return ArgumentComposition(arguments)(polynomial)


def affine_arguments(origin: np.ndarray, matrix: np.ndarray) -> tuple[Polynomial, ...]:
    variables = axes(matrix.shape[1])
    return tuple(
        add(
            constant(Fraction(float(start)), matrix.shape[1]),
            sum_polynomials(
                tuple(
                    scale(variable, Fraction(float(value)))
                    for variable, value in zip(variables, row, strict=True)
                )
            ),
        )
        for start, row in zip(origin, matrix, strict=True)
    )


def sum_polynomials(polynomials: tuple[Polynomial, ...]) -> Polynomial:
    result: Polynomial = {}
    for polynomial in polynomials:
        result = add(result, polynomial)
    return result


def linear_combinations(
    polynomials: tuple[Polynomial, ...],
    weight_bank: tuple[tuple[Fraction, ...], ...],
    *,
    prepared_profile: tuple[int, int, int] | None = None,
) -> tuple[Polynomial, ...]:
    """Share bank preparation across exact accumulators without output copies."""
    if any(len(polynomials) != len(weights) for weights in weight_bank):
        raise ValueError("Linear combination weights must match the polynomial bank.")
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        if prepared_profile is None:
            numerator, denominator = _coefficient_profile(polynomials)
            dimension = next(
                (len(index) for polynomial in polynomials for index in polynomial), 0
            )
        else:
            budget.reserve(1)
            numerator, denominator, dimension = prepared_profile
    results = []
    for weights in weight_bank:
        if budget is not None:
            budget.reserve(len(weights))
            weight_bits, active_terms = 1, 0
            for polynomial, weight in zip(polynomials, weights, strict=True):
                weight_bits = max(
                    weight_bits,
                    abs(weight.numerator).bit_length() + weight.denominator.bit_length(),
                )
                if weight:
                    active_terms += len(polynomial)
            _reserve_polynomial(
                2 * active_terms,
                active_terms,
                dimension,
                numerator + denominator + weight_bits + max(active_terms, 1).bit_length(),
            )
        result: Polynomial = {}
        for polynomial, weight in zip(polynomials, weights, strict=True):
            if not weight:
                continue
            for index, coefficient in polynomial.items():
                value = coefficient * weight
                if not value:
                    continue
                total = result.get(index, Fraction(0)) + value
                if total:
                    result[index] = total
                else:
                    result.pop(index, None)
        results.append(result)
    return tuple(results)


def _nodal_axis(nodes: np.ndarray, axis: int, dimension: int) -> tuple[Polynomial, ...]:
    variable = axes(dimension)[axis]
    result = []
    for i, node in enumerate(nodes):
        polynomial = constant(1, dimension)
        x = Fraction(float(node))
        for j, other in enumerate(nodes):
            if i != j:
                y = Fraction(float(other))
                polynomial = multiply(
                    polynomial, scale(add(variable, constant(-y, dimension)), 1 / (x - y))
                )
        result.append(polynomial)
    return tuple(result)


def _simplex_basis(owner: SimplexNodalFamily, dimension: int) -> tuple[Polynomial, ...]:
    variables = axes(dimension)
    barycentric = (
        add(constant(1, dimension), scale(sum_polynomials(variables), -1)),
        *variables,
    )
    modal = []
    for index in owner.multiindices:
        weight = math.factorial(owner.order) // math.prod(
            math.factorial(value) for value in index
        )
        term = constant(weight, dimension)
        for argument, exponent in zip(barycentric, index, strict=True):
            term = multiply(term, power(argument, exponent, dimension))
        modal.append(term)
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(
            owner.coefficients.size,
            owner.coefficients.size * np.dtype(np.float64).itemsize,
        )
    coefficients = np.asarray(owner.coefficients, dtype=np.float64)
    return tuple(
        sum_polynomials(
            tuple(
                scale(term, Fraction(float(value)))
                for term, value in zip(modal, coefficients[:, i], strict=True)
            )
        )
        for i in range(coefficients.shape[1])
    )


def _jacobi(degree: int, alpha: int, axis: int, dimension: int) -> Polynomial:
    variable = axes(dimension)[axis]
    shifted = add(variable, constant(-1, dimension))
    return sum_polynomials(
        tuple(
            scale(
                multiply(
                    power(shifted, degree - i, dimension), power(variable, i, dimension)
                ),
                math.comb(degree + alpha, i) * math.comb(degree, degree - i),
            )
            for i in range(degree + 1)
        )
    )


def source_basis(element: CellGeometryElement) -> tuple[Polynomial, ...] | None:
    """Prepare canonical expressions once within an explicit resource ledger."""
    budget = _COORDINATE_BUDGET.get()
    if budget is None:
        return _source_basis(element)
    key = _element_identity(element)
    if key not in budget.basis_cache:
        basis = _source_basis(element)
        if basis is not None:
            budget.retain_basis(basis)
        budget.basis_cache[key] = basis
    return budget.basis_cache[key]


def _element_identity(element: CellGeometryElement, /) -> tuple[str, str]:
    """Canonical identity of an element: its declared ID and complete array contents."""
    from .._fingerprint import array_tree_fingerprint, canonical_fingerprint

    return element.element_id, canonical_fingerprint(array_tree_fingerprint(element))


def reference_composition_arguments(
    element: CellGeometryElement,
) -> tuple[Polynomial, ...]:
    """Prepare the immutable original chart once on its owning resource ledger."""
    budget = _COORDINATE_BUDGET.get()
    if budget is None:
        return _prepare_reference_composition_arguments(element)
    key = _element_identity(element)
    cached = budget.reference_composition_cache.get(key)
    if cached is not None:
        return cached
    with budget.temporary_scope():
        arguments = _prepare_reference_composition_arguments(element)
        budget.retain_basis((key, arguments))
        budget.reference_composition_cache[key] = arguments
        return arguments


def prepared_reference_arguments(
    element: CellGeometryElement, /
) -> tuple[Polynomial, ...] | PreparedPolynomialArguments:
    """Reuse immutable exact reference preparation, never physical coefficient bindings."""
    budget = _COORDINATE_BUDGET.get()
    if budget is None:
        return reference_composition_arguments(element)
    source_key = _element_identity(element)
    cached = budget.prepared_reference_cache.get(source_key)
    if cached is not None:
        return cached
    original = reference_composition_arguments(element)
    if any(isinstance(argument, RationalPolynomial) for argument in original):
        # Rational actions keep their existing owning homogeneous preparation.
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
        axis_affine = _axis_affine_arguments(arguments, dimension)
        prepared = PreparedPolynomialArguments(
            arguments, ordered_key, dimension, axis_affine
        )
        # The record, complete ordered key, underlying snapshot dictionaries,
        # immutable views and classification all belong to this original ledger.
        budget.retain_basis(
            (source_key, prepared, ordered_key, snapshots, arguments, axis_affine)
        )
        budget.prepared_reference_cache[source_key] = prepared
        return prepared


def _prepare_reference_composition_arguments(
    element: CellGeometryElement,
) -> tuple[Polynomial, ...]:
    """Exact rational reference chart, independent of rounded tabulation values."""
    from ._cell_geometry import (
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
    )

    if not isinstance(
        element,
        (PolynomialComposedCellGeometryElement, RationalComposedCellGeometryElement),
    ):
        raise TypeError(
            "A reference composition chart requires its actual owning element."
        )
    if isinstance(element, RationalComposedCellGeometryElement):
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
                "Rational chart execution coefficients are not the RNE of their exact source."
            )
    return _prepare_reference_chart_arguments(
        element.chart_element,
        element.chart_coefficients,
        element.source_element.topological_dimension,
    )


def _prepare_reference_chart_arguments(
    chart_element: CellGeometryElement,
    coefficients: tuple[tuple[tuple[int, int], ...], ...],
    dimension: int,
) -> tuple[Polynomial, ...]:
    """Prepare admitted original chart inputs without a partial owning instance."""
    basis = source_basis(chart_element)
    if basis is None:
        raise ValueError(
            "A reference composition chart has no authoritative exact basis."
        )
    return tuple(
        sum_polynomials(
            tuple(
                scale(term, Fraction(*row[axis]))
                for term, row in zip(basis, coefficients, strict=True)
            )
        )
        for axis in range(dimension)
    )


def _spline_axis_basis(
    knots: np.ndarray,
    degree: int,
    span: int,
    axis: int,
    /,
) -> tuple[Polynomial, ...]:
    """Exact Cox-de Boor source polynomials on one normalized knot span."""
    vector = tuple(Fraction(float(value)) for value in knots)
    count = len(vector) - degree - 1
    parameter = add(
        constant(vector[span], 2), scale(axes(2)[axis], vector[span + 1] - vector[span])
    )
    basis = tuple(constant(1 if row == span else 0, 2) for row in range(len(vector) - 1))
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
                    multiply(add(parameter, constant(-vector[row], 2)), basis[row]),
                    1 / left,
                )
            )
            second = (
                {}
                if not right
                else scale(
                    multiply(
                        add(constant(vector[row + order + 1], 2), scale(parameter, -1)),
                        basis[row + 1],
                    ),
                    1 / right,
                )
            )
            following.append(add(first, second))
        basis = tuple(following)
    return basis[:count]


def source_expressions(element: CellGeometryElement) -> tuple[Expression, ...] | None:
    """Prepare the original scalar source expression once on the owning ledger."""
    budget = _COORDINATE_BUDGET.get()
    if budget is None:
        return _prepare_source_expressions(element)
    key = _element_identity(element)
    if key in budget.source_expression_cache:
        return budget.source_expression_cache[key]
    with budget.temporary_scope():
        expressions = _prepare_source_expressions(element)
        if expressions is not None:
            budget.retain_basis((key, expressions))
        budget.source_expression_cache[key] = expressions
        return expressions


def _prepare_source_expressions(
    element: CellGeometryElement,
) -> tuple[Expression, ...] | None:
    """Actual scalar coordinate source basis, including original rational weights."""
    from ._cell_geometry import (
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )

    if isinstance(element, SplineCellGeometryElement):
        u = _spline_axis_basis(
            np.asarray(element.u_knots), element.u_degree, element.span_indices[0], 0
        )
        v = _spline_axis_basis(
            np.asarray(element.v_knots), element.v_degree, element.span_indices[1], 1
        )
        numerators = tuple(
            scale(multiply(first, second), Fraction(float(weight)))
            for first, row in zip(u, np.asarray(element.weights), strict=True)
            for second, weight in zip(v, row, strict=True)
        )
        denominator = sum_polynomials(numerators)
        return tuple(rational_expression(term, denominator) for term in numerators)
    if isinstance(
        element,
        (PolynomialComposedCellGeometryElement, RationalComposedCellGeometryElement),
    ):
        source = source_expressions(element.source_element)
        if source is None:
            return None
        arguments = prepared_reference_arguments(element)
        return tuple(expression_compose(term, arguments) for term in source)
    if isinstance(element, RestrictedCellGeometryElement):
        source = source_expressions(element.source_element)
        if source is None:
            return None
        return restrict_chart_expressions(
            source,
            element.source_element.cell_kind,
            element.cell_kind,
            np.asarray(element.offset),
            np.asarray(element.matrix),
        )
    return source_basis(element)


def _source_basis(element: CellGeometryElement) -> tuple[Polynomial, ...] | None:
    """Extract only canonical source expressions, never a claimed family string."""
    from ._cell_geometry import (
        _CoordinateTabulator,
        _SweptCoordinateTabulator,
        BarycentricCellGeometryElement,
        LayerColumnCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )
    from .fem._form_elements import _ProxyTabulator, FormBasis
    from .fem._high_order import ReferenceNodalFamily, SimplexNodalFamily
    from .fem._spectral_hp_completion import HybridReferenceFamily

    if isinstance(element, BarycentricCellGeometryElement):
        root = element.source_element
        while isinstance(root, BarycentricCellGeometryElement):
            root = root.source_element
        basis = source_basis(root)
        if basis is None:
            raise ValueError(
                "A full barycentric action lost its canonical P1 cardinal source."
            )
        budget = _COORDINATE_BUDGET.get()
        current = element
        while isinstance(current, BarycentricCellGeometryElement):
            if budget is not None:
                budget.reserve(
                    current.barycentric_weights.size,
                    128 + 512 * current.barycentric_weights.size,
                )
            weights = np.asarray(current.barycentric_weights, dtype=np.float64)
            columns = tuple(
                tuple(
                    Fraction(float(weights[target, source]))
                    for target in range(weights.shape[0])
                )
                for source in range(weights.shape[1])
            )
            basis = linear_combinations(basis, columns)
            current = current.source_element
        return basis

    if isinstance(element, LayerColumnCellGeometryElement):
        wall, corners = (
            source_basis(element.wall_element),
            source_basis(element.corner_element),
        )
        if wall is None or corners is None or element.station_axis != 2:
            return None
        if element.local_dof_count != 2 * len(wall) + 4 * len(corners):
            return None
        station = axes(3)[2]
        factors = (add(constant(1, 3), scale(station, -1)), station)
        lifted_wall: tuple[Polynomial, ...] = tuple(
            {(*index, 0): value for index, value in term.items()} for term in wall
        )
        lifted_corners: tuple[Polynomial, ...] = tuple(
            {(*index, 0): value for index, value in term.items()} for term in corners
        )
        profiles = tuple(
            multiply(term, factor) for factor in factors for term in lifted_wall
        )
        actual = tuple(
            multiply(term, factor) for factor in factors for term in lifted_corners
        )
        return (*profiles, *actual, *(scale(term, -1) for term in actual))

    if isinstance(element, SplineCellGeometryElement):
        expressions = source_expressions(element)
        if expressions is None or any(
            isinstance(term, RationalPolynomial) for term in expressions
        ):
            return None
        return tuple(
            term for term in expressions if not isinstance(term, RationalPolynomial)
        )

    if isinstance(
        element,
        (PolynomialComposedCellGeometryElement, RationalComposedCellGeometryElement),
    ):
        basis = source_basis(element.source_element)
        if basis is None:
            return None
        arguments = prepared_reference_arguments(element)
        return tuple(compose(term, arguments) for term in basis)

    if isinstance(element, RestrictedCellGeometryElement):
        if element.source_element.cell_kind == "pyramid":
            # A general affine physical restriction is rational after collapse;
            # do not silently treat its inverse collapsed chart as affine.
            return None
        basis = source_basis(element.source_element)
        if basis is None:
            return None
        arguments = affine_arguments(
            np.asarray(element.offset), np.asarray(element.matrix)
        )
        return tuple(compose(term, arguments) for term in basis)

    if not isinstance(element, FiniteElementSpec):
        return None
    if element.mapping != "identity" or element.value_shape:
        return None
    if type(element.tabulator) is _SweptCoordinateTabulator:
        source = element.tabulator.source
        expected = "prism" if source.cell_kind == "triangle" else "hexahedron"
        if (
            source.cell_kind not in ("triangle", "quadrilateral")
            or element.cell_kind != expected
            or element.degree != source.degree
            or element.local_dof_count != 2 * source.local_dof_count
        ):
            return None
        basis = source_basis(source)
        if basis is None:
            return None
        lifted: tuple[Polynomial, ...] = tuple(
            {(*index, 0): value for index, value in term.items()} for term in basis
        )
        w = axes(3)[2]
        return tuple(
            multiply(term, factor)
            for factor in (add(constant(1, 3), scale(w, -1)), w)
            for term in lifted
        )
    if type(element.tabulator) is _ProxyTabulator:
        owner = element.tabulator.basis
        if (
            type(owner) is not FormBasis
            or owner.form_degree != 0
            or owner.dimension != element.topological_dimension
            or owner.order != element.degree
            or owner.local_dof_count != element.local_dof_count
            or element.tabulator.value_spec != element.value_spec
            or element.representation not in ("polynomial_moment", "rational_moment")
        ):
            return None
        return _scalar_form_basis(owner)
    if element.representation != "point_value":
        return None
    dimension = element.topological_dimension
    if (
        element.degree == 0
        and element.family == "DiscontinuousLagrange"
        and element.tabulator is None
    ):
        from .fem._reference import discontinuous_element

        canonical = discontinuous_element(element.cell_kind, 0)
        if element.element_id == canonical.element_id and element.local_dof_count == 1:
            return (constant(1, dimension),)
        return None
    if isinstance(element.tabulator, _CoordinateTabulator):
        if (
            element.cell_kind != element.tabulator.cell_kind
            or element.degree != element.tabulator.degree
        ):
            return None
        return lattice_basis(element.tabulator.cell_kind, element.tabulator.degree)
    owner = owning_tabulator_source(element.tabulator)
    if isinstance(owner, FiniteElementSpec):
        if (element.cell_kind, element.degree, element.local_dof_count) != (
            owner.cell_kind,
            owner.degree,
            owner.local_dof_count,
        ):
            return None
        return source_basis(owner)
    if owner is not None:
        if isinstance(owner, SimplexNodalFamily):
            if element.cell_kind != owner.cell_kind or element.degree != owner.order:
                return None
            return _simplex_basis(owner, dimension)
        if isinstance(owner, ReferenceNodalFamily):
            if element.cell_kind != owner.cell_kind or element.degree != max(
                owner.orders
            ):
                return None
            factors = tuple(
                _nodal_axis(np.asarray(nodes), axis, dimension)
                for axis, nodes in enumerate(owner.nodes_by_axis)
            )
            return tuple(
                math_product_polynomials(choice, dimension)
                for choice in product(*factors)
            )
        if isinstance(owner, HybridReferenceFamily):
            if element.cell_kind != owner.cell_kind or element.degree != owner.degree:
                return None
            if owner.cell_kind == "prism":
                triangle = _simplex_basis(
                    SimplexNodalFamily("triangle", owner.orders[0]), 2
                )
                lifted: tuple[Polynomial, ...] = tuple(
                    {(*index, 0): value for index, value in term.items()}
                    for term in triangle
                )
                axial = _nodal_axis(np.unique(np.asarray(owner.nodes)[:, 2]), 2, 3)
                terms = tuple(multiply(a, b) for a, b in product(lifted, axial))
                return tuple(terms[index] for index in owner.basis_permutation)
            collapse = add(constant(1, 3), scale(axes(3)[2], -1))
            modes = tuple(
                math_product_polynomials(
                    (
                        _jacobi(i, 0, 0, 3),
                        _jacobi(j, 0, 1, 3),
                        power(collapse, max(i, j), 3),
                        _jacobi(k, 2 * max(i, j) + 2, 2, 3),
                    ),
                    3,
                )
                for i, j, k in owner.modal_indices
            )
            budget = _COORDINATE_BUDGET.get()
            if budget is not None:
                budget.reserve(
                    owner.coefficients.size,
                    owner.coefficients.size * np.dtype(np.float64).itemsize,
                )
            coefficients = np.asarray(owner.coefficients, dtype=np.float64)
            return tuple(
                sum_polynomials(
                    tuple(
                        scale(term, Fraction(float(value)))
                        for term, value in zip(modes, coefficients[:, i], strict=True)
                    )
                )
                for i in range(coefficients.shape[1])
            )
        return None
    if element.tabulator is not None:
        return None
    if (
        element.element_id
        != lagrange_element(element.cell_kind, element.degree).element_id
    ):
        return None
    variables = axes(dimension)
    if element.cell_kind in ("triangle", "tetrahedron"):
        barycentric = (
            add(constant(1, dimension), scale(sum_polynomials(variables), -1)),
            *variables,
        )
        if element.degree == 1:
            return barycentric
        vertices = tuple(
            multiply(value, add(scale(value, 2), constant(-1, dimension)))
            for value in barycentric
        )
        edges = tuple(
            scale(multiply(barycentric[a], barycentric[b]), 4)
            for a, b in ((0, 1), (1, 2), (2, 0))
        )
        return (*vertices, *edges)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    factors = tuple(
        _nodal_axis(np.unique(nodes[:, axis]), axis, dimension)
        for axis in range(dimension)
    )
    return tuple(
        math_product_polynomials(
            tuple(factors[axis][int(value)] for axis, value in enumerate(row)), dimension
        )
        for row in nodes
    )


def _scalar_form_basis(owner: object) -> tuple[Polynomial, ...]:
    """Extract the actual canonical scalar form execution, including tensor factors."""
    from .fem._form_elements import FormBasis

    if not isinstance(owner, FormBasis) or owner.form_degree != 0:
        raise TypeError("Coordinate form sources require a scalar FormBasis.")
    budget = _COORDINATE_BUDGET.get()
    if owner.hybrid_factors is not None:
        source = tuple(field[0] for field in owner.component_expressions())
        if owner.family == "pyramid-trimmed":
            arguments = chart_arguments("pyramid", 3)
            source = tuple(expression_compose(term, arguments) for term in source)
        if any(isinstance(term, RationalPolynomial) for term in source):
            raise RationalEnclosureError(
                "Scalar hybrid source has an unresolved collapsed-chart denominator."
            )
        return tuple(term for term in source if not isinstance(term, RationalPolynomial))
    if owner.tensor_factors is None:
        if budget is not None:
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
    if budget is not None:
        budget.reserve(factors.zero.size + factors.indices.size + factors.components.size)
    zero = np.asarray(factors.zero, dtype=np.float64)
    indices = np.asarray(factors.indices, dtype=np.int32)
    components = np.asarray(factors.components, dtype=np.float64)
    if np.any(np.asarray(factors.differential_axes)):
        raise ValueError(
            "Scalar tensor coordinate factors cannot contain differential axes."
        )
    dimension = owner.dimension
    variables = axes(dimension)
    axial = []
    for axis, variable in enumerate(variables):
        collapse = add(constant(1, dimension), scale(variable, -1))
        bubble = multiply(variable, collapse)
        modes = tuple(
            _jacobi(order, 0, axis, dimension) for order in range(owner.order - 1)
        )
        axial.append(
            tuple(
                add(
                    add(
                        scale(collapse, Fraction(float(zero[0, dof]))),
                        scale(variable, Fraction(float(zero[1, dof]))),
                    ),
                    multiply(
                        bubble,
                        sum_polynomials(
                            tuple(
                                scale(mode, Fraction(float(value)))
                                for mode, value in zip(modes, zero[2:, dof], strict=True)
                            )
                        ),
                    ),
                )
                for dof in range(zero.shape[1])
            )
        )
    return tuple(
        scale(
            math_product_polynomials(
                tuple(axial[axis][indices[axis, dof]] for axis in range(dimension)),
                dimension,
            ),
            Fraction(float(components[dof, 0])),
        )
        for dof in range(owner.local_dof_count)
    )


def math_product_polynomials(
    polynomials: tuple[Polynomial, ...], dimension: int
) -> Polynomial:
    if any(not polynomial for polynomial in polynomials):
        return {}
    result = constant(1, dimension)
    for polynomial in polynomials:
        result = multiply(result, polynomial)
    return result


def _coordinate_source_rows(local: CoordinateCoefficients) -> CoordinateSourceBank:
    if isinstance(local, np.ndarray):
        if local.ndim != 2:
            raise ValueError(
                "Exact coordinate coefficients require a rank-two host bank."
            )
        rows = tuple(
            tuple(
                value if isinstance(value, Fraction) else Fraction(float(value))
                for value in row
            )
            for row in local
        )
    else:
        rows = local
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(
            "Exact coordinate coefficients require a nonempty rectangular host bank."
        )
    return rows


def coordinate_polynomials(
    element: CellGeometryElement, local: CoordinateCoefficients
) -> tuple[Polynomial, ...] | None:
    """Reuse complete exact polynomial maps in the owning preparation ledger."""
    budget = _COORDINATE_BUDGET.get()
    if budget is None:
        return _prepare_coordinate_polynomials(element, local)
    with budget.temporary_scope():
        coefficients, key = _coordinate_preparation_key(element, local)
        cached = budget.polynomial_cache.get(key)
        if cached is not None:
            return cached
        prepared = multi_affine_coordinates(element, coefficients)
        if prepared is not None:
            return prepared[0]
        coordinates = _prepare_coordinate_polynomials(element, coefficients)
        if coordinates is not None:
            budget.retain_basis((key, coordinates))
            budget.coordinate_cache[key] = coordinates
            budget.polynomial_cache[key] = coordinates
        return coordinates


def _coordinate_preparation_key(
    element: CellGeometryElement, local: CoordinateCoefficients, /
) -> tuple[CoordinateSourceBank, tuple[str, str, CoordinateSourceBank]]:
    budget = _COORDINATE_BUDGET.get()
    immutable_key = None
    if (
        budget is not None
        and isinstance(local, tuple)
        and all(isinstance(row, tuple) for row in local)
    ):
        immutable_key = (*_element_identity(element), local)
        prepared = budget.coefficient_preparation_cache.get(immutable_key)
        if prepared is not None:
            return prepared
    if budget is not None:
        budget.reserve(
            local.size
            if isinstance(local, np.ndarray)
            else sum(len(row) for row in local)
        )
        if isinstance(local, np.ndarray):
            budget.reserve(0, 128 + 512 * local.size)
    coefficients = _coordinate_source_rows(local)
    if len(coefficients) != element.local_dof_count:
        raise ValueError(
            "Exact coordinate coefficients must include every source degree of freedom."
        )
    prepared = coefficients, (*_element_identity(element), coefficients)
    if budget is not None and immutable_key is not None:
        # Retain the validated complete immutable bank and element binding.
        # A mutable host array still earns every current normalization; only
        # identical immutable source data can skip that executed preparation.
        budget.reserve(1, 128)
        budget.retain_basis((immutable_key, prepared))
        budget.coefficient_preparation_cache[immutable_key] = prepared
    return prepared


def coordinate_scope_key(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[str, str, str]:
    """Scientific source ownership and every live coordinate/reference bank."""
    from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from ._cell_geometry_validity import cell_geometry_id

    geometry_id = cell_geometry_id(geometry)
    live_tree_id = (
        geometry.storage_id
        if geometry.storage_id is not None
        else canonical_fingerprint(array_tree_fingerprint((mesh, geometry)))
    )
    return mesh.mesh_id, geometry_id, live_tree_id


def prepared_coordinate_source_bank(
    geometry: CellGeometrySpec, /
) -> CoordinateSourceBank:
    """Retain one immutable current source bank in the original owning ledger.

    The geometry identity binds source definitions, routes and scientific source
    IDs; the complete live numerical tree additionally binds every owning source
    bank. Equal carrier arrays alone never authorize reuse of a foreign source.
    """
    from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from ._cell_geometry_validity import cell_geometry_id

    budget = _COORDINATE_BUDGET.get()
    if budget is None:
        return geometry.source_coordinates()
    geometry_id = cell_geometry_id(geometry)
    live_tree_id = (
        geometry.storage_id
        if geometry.storage_id is not None
        else canonical_fingerprint(array_tree_fingerprint(geometry))
    )
    key = (geometry_id, live_tree_id)
    cached = budget.source_bank_cache.get(key)
    if cached is not None:
        return cached
    with budget.temporary_scope():
        bank = geometry.source_coordinates()
        budget.retain_basis((key, bank))
    budget.source_bank_cache[key] = bank
    return bank


def _prepare_coordinate_polynomials(
    element: CellGeometryElement, local: CoordinateCoefficients, /
) -> tuple[Polynomial, ...] | None:
    from ._cell_geometry import (
        BarycentricCellGeometryElement,
        RestrictedCellGeometryElement,
    )

    if isinstance(element, BarycentricCellGeometryElement):
        # Tabulation multiplies the complete cardinal basis by the retained
        # outer-to-inner action stack. Contract physical coefficients in reverse
        # order, exactly, rather than materializing/recombining cardinal maps.
        actions = []
        root = element
        while isinstance(root, BarycentricCellGeometryElement):
            actions.append(root.barycentric_weights)
            root = root.source_element
        rows = _coordinate_source_rows(local)
        budget = _COORDINATE_BUDGET.get()
        for action in reversed(actions):
            weights = np.asarray(action, dtype=np.float64)
            if budget is not None:
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
                    if budget is not None:
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
        return coordinate_polynomials(root, rows)

    if isinstance(element, RestrictedCellGeometryElement):
        source = coordinate_polynomials(element.source_element, local)
        if source is None:
            return None
        return restrict_chart_coordinates(
            source,
            element.source_element.cell_kind,
            element.cell_kind,
            np.asarray(element.offset),
            np.asarray(element.matrix),
        )
    basis = source_basis(element)
    if basis is None:
        return None
    rows = _coordinate_source_rows(local)
    budget = _COORDINATE_BUDGET.get()
    profile = None
    if budget is not None:
        key = _element_identity(element)
        profile = budget.basis_profile_cache.get(key)
        if profile is None:
            numerator, denominator = _coefficient_profile(basis)
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
    )


def determinant(matrix: tuple[tuple[Polynomial, ...], ...]) -> Polynomial:
    from itertools import permutations

    dimension = len(matrix)
    result: Polynomial = {}
    for permutation in permutations(range(dimension)):
        factors = tuple(matrix[i][permutation[i]] for i in range(dimension))
        if any(not factor for factor in factors):
            continue
        sign = (-1) ** sum(
            permutation[i] > permutation[j]
            for i in range(dimension)
            for j in range(i + 1, dimension)
        )
        result = add(result, scale(math_product_polynomials(factors, dimension), sign))
    return result


def divide_collapse(polynomial: Polynomial, repetitions: int) -> Polynomial | None:
    result = polynomial
    for _ in range(repetitions):
        quotient: Polynomial = {}
        grouped: dict[tuple[int, ...], dict[int, Fraction]] = {}
        for index, value in result.items():
            grouped.setdefault(index[:-1], {})[index[-1]] = value
        for prefix, values in grouped.items():
            coefficient = Fraction(0)
            for degree in range(max(values) + 1):
                coefficient += values.get(degree, Fraction(0))
                if degree < max(values) and coefficient:
                    quotient[(*prefix, degree)] = coefficient
            if coefficient:
                return None
        result = quotient
    return result


def determinant_polynomial(
    coordinates: tuple[Polynomial, ...], cell_kind: str, dimension: int
) -> Polynomial | None:
    jacobian = tuple(
        tuple(derivative(value, axis) for axis in range(dimension))
        for value in coordinates
    )
    if len(coordinates) == dimension:
        value = determinant(jacobian)
        return divide_collapse(value, 2) if cell_kind == "pyramid" else value
    gram = tuple(
        tuple(
            sum_polynomials(tuple(multiply(row[i], row[j]) for row in jacobian))
            for j in range(dimension)
        )
        for i in range(dimension)
    )
    return determinant(gram)


_BERNSTEIN_LATTICE_CACHE: dict[
    tuple[BernsteinDomain, tuple[int, ...], int], tuple[tuple[int, ...], ...]
] = {}


def _prepare_bernstein_lattice(
    reference: BernsteinDomain, degrees: tuple[int, ...], dimension: int, /
) -> tuple[tuple[int, ...], ...]:
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
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
    reference: BernsteinDomain, degrees: tuple[int, ...], dimension: int, /
) -> tuple[tuple[int, ...], ...]:
    budget = _COORDINATE_BUDGET.get()
    owner = _BERNSTEIN_LATTICE_CACHE if budget is None else budget.bernstein_lattice_cache
    key = (reference, degrees, dimension)
    lattice = owner.get(key)
    if lattice is None:
        lattice = _prepare_bernstein_lattice(reference, degrees, dimension)
        if budget is not None:
            budget.retain_basis(lattice)
        owner[key] = lattice
    return lattice


def _dominating_simplex_indices(
    alpha: tuple[int, ...], degree: int
) -> Iterator[tuple[int, ...]]:
    """Generate only dominating simplex indices in canonical lexicographic order."""
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
    value: Polynomial,
    reference: BernsteinDomain,
    degrees: tuple[int, ...],
    dimension: int,
) -> tuple[int, int, int]:
    """Pre-admit actual control initialization, term and accumulation visits."""
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
    for alpha in value:
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


def _prepare_bernstein_weights(
    reference: BernsteinDomain,
    degrees: tuple[int, ...],
    dimension: int,
    alpha: tuple[int, ...],
    /,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Exact monomial-to-Bernstein weights of ``alpha`` as integers over one denominator.

    Box weights are ``prod comb(b, a) / comb(n, a)``; simplex weights are falling
    factorials ``prod b!/(b-a)! / (n!/(n-|a|)!)``; prisms multiply the simplex
    weight by the axial box weight. Lattice points not dominating ``alpha``
    receive no contribution.
    """
    entries: list[tuple[int, int]] = []
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        if reference == "box":
            lengths = tuple(
                degree - exponent + 1
                for exponent, degree in zip(alpha, degrees, strict=True)
            )
            count = math.prod(lengths)
            prefixes = sum(math.prod(lengths[: axis + 1]) for axis in range(dimension))
        else:
            simplex_dimension = dimension if reference == "simplex" else 2
            remaining = degrees[0] - sum(alpha[:simplex_dimension])
            count = math.comb(remaining + simplex_dimension, simplex_dimension)
            prefixes = sum(
                math.comb(remaining + axis, axis)
                for axis in range(1, simplex_dimension + 1)
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
                (
                    position,
                    math.prod(math.comb(b, a) for a, b in zip(alpha, beta, strict=True)),
                )
            )
        return denominator, tuple(entries)
    simplex_dimension = dimension if reference == "simplex" else 2
    order = sum(alpha[:simplex_dimension])
    denominator = math.factorial(degrees[0]) // math.factorial(degrees[0] - order)
    if reference == "prism":
        denominator *= math.comb(degrees[1], alpha[2])
    for beta in _dominating_simplex_indices(alpha[:simplex_dimension], degrees[0]):
        position = _simplex_control_position(beta, degrees[0])
        numerator = math.prod(
            math.factorial(beta[i]) // math.factorial(beta[i] - alpha[i])
            for i in range(simplex_dimension)
        )
        if reference == "prism":
            for height in range(alpha[2], degrees[1] + 1):
                entries.append(
                    (
                        position * (degrees[1] + 1) + height,
                        numerator * math.comb(height, alpha[2]),
                    )
                )
        else:
            entries.append((position, numerator))
    return denominator, tuple(entries)


@lru_cache(maxsize=65536)
def _unbudgeted_bernstein_weights(
    reference: BernsteinDomain,
    degrees: tuple[int, ...],
    dimension: int,
    alpha: tuple[int, ...],
    /,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    return _prepare_bernstein_weights(reference, degrees, dimension, alpha)


def _bernstein_weights(
    reference: BernsteinDomain,
    degrees: tuple[int, ...],
    dimension: int,
    alpha: tuple[int, ...],
    /,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    budget = _COORDINATE_BUDGET.get()
    if budget is None:
        return _unbudgeted_bernstein_weights(reference, degrees, dimension, alpha)
    key = (reference, degrees, dimension, alpha)
    weights = budget.bernstein_weight_cache.get(key)
    if weights is None:
        weights = _prepare_bernstein_weights(reference, degrees, dimension, alpha)
        budget.retain_basis(weights)
        budget.bernstein_weight_cache[key] = weights
    return weights


def bernstein_coefficients(
    polynomial: Polynomial, domain: str, dimension: int
) -> tuple[Fraction, ...]:
    reference = parse(domain, BernsteinDomain, "domain")
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
        # Degree-one Bernstein coefficients are the exact vertex images.
        # Avoid constructing rational weight tables for this identity action.
        budget = _COORDINATE_BUDGET.get()
        if budget is not None:
            numerator, denominator = _coefficient_profile((polynomial,))
            _reserve_polynomial(
                len(polynomial) + dimension,
                dimension + 1,
                dimension,
                numerator + denominator + 1,
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
    if _COORDINATE_BUDGET.get() is not None:
        numerator, denominator = _coefficient_profile((polynomial,))
        visits, nodes, entries = _bernstein_support_profile(
            polynomial, reference, degrees, dimension
        )
        degree = max((sum(index) for index in polynomial), default=0)
        weight_bits = 2 * dimension * degree * max(degree + 1, 1).bit_length()
        _reserve_polynomial(
            visits,
            nodes + entries,
            dimension,
            numerator + denominator + weight_bits + max(len(polynomial), 1).bit_length(),
        )
    # Exact rational sums are accumulated as integers over one common
    # denominator; ``Fraction`` normalization yields the identical coefficients.
    terms = []
    common = 1
    for alpha, value in polynomial.items():
        weight_denominator, entries = _bernstein_weights(
            reference, degrees, dimension, alpha
        )
        denominator = value.denominator * weight_denominator
        terms.append((value.numerator, denominator, entries))
        common = math.lcm(common, denominator)
    totals = [0] * len(_bernstein_lattice(reference, degrees, dimension))
    for numerator, denominator, entries in terms:
        factor = numerator * (common // denominator)
        for position, weight in entries:
            totals[position] += factor * weight
    return tuple(Fraction(total, common) for total in totals)


def outward(value: Fraction, direction: float) -> float:
    if value > _BINARY64_MAXIMUM:
        return float(np.finfo(np.float64).max) if direction < 0.0 else math.inf
    if value < -_BINARY64_MAXIMUM:
        return -math.inf if direction < 0.0 else -float(np.finfo(np.float64).max)
    result = float(value)
    represented = Fraction(result)
    if represented == value:
        return result
    if (direction < 0.0 and represented > value) or (
        direction > 0.0 and represented < value
    ):
        return float(np.nextafter(result, direction))
    return result


def polynomial_bounds(
    polynomial: Polynomial, domain: str, dimension: int
) -> tuple[float, float]:
    coefficients = bernstein_coefficients(polynomial, domain, dimension)
    return outward(min(coefficients), -math.inf), outward(max(coefficients), math.inf)


def _affine_polynomial_jacobian(
    coordinates: tuple[Expression, ...], dimension: int, /
) -> tuple[tuple[Polynomial, ...], ...] | None:
    budget = _COORDINATE_BUDGET.get()
    polynomials = []
    for value in coordinates:
        if isinstance(value, RationalPolynomial):
            return None
        if budget is not None:
            budget.reserve(len(value))
        if any(sum(index) > 1 for index in value):
            return None
        polynomials.append(value)
    if budget is not None:
        numerator, denominator = _coefficient_profile(tuple(polynomials))
        _reserve_polynomial(
            len(coordinates) * dimension,
            len(coordinates) * dimension,
            dimension,
            numerator + denominator,
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
    coordinates: tuple[Polynomial, ...], cell_kind: str, dimension: int
) -> tuple[tuple[Polynomial, ...], ...] | None:
    if cell_kind != "pyramid":
        affine = _affine_polynomial_jacobian(coordinates, dimension)
        if affine is not None:
            return affine
    rows = tuple(
        tuple(derivative(value, axis) for axis in range(dimension))
        for value in coordinates
    )
    if cell_kind != "pyramid":
        return rows
    variables = axes(3)
    result = []
    for first, second, height in rows:
        x = divide_collapse(first, 1)
        y = divide_collapse(second, 1)
        if x is None or y is None:
            return None
        z = add(
            height,
            add(
                multiply(add(variables[0], constant(Fraction(-1, 2), 3)), x),
                multiply(add(variables[1], constant(Fraction(-1, 2), 3)), y),
            ),
        )
        result.append((x, y, z))
    return tuple(result)


def evaluate(polynomial: Polynomial, point: tuple[Fraction, ...]) -> Fraction:
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(len(polynomial))
    # Exact evaluation over the common denominator of the coefficients and of
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


def _reference_image(
    polynomials: tuple[Polynomial, ...], cell_kind: str, point: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    """Exact image of one physical reference point under chart expressions.

    Pyramid expressions live on the collapsed cube ``x = u (1 - w) + w / 2``.
    The apex is the exact ``w = 1`` slice, which a single-valued map must leave
    independent of ``u`` and ``v``; no representative collapsed point is chosen.
    """
    if cell_kind != "pyramid":
        return tuple(evaluate(value, point) for value in polynomials)
    x, y, z = point
    if z != 1:
        return tuple(
            evaluate(value, ((x - z / 2) / (1 - z), (y - z / 2) / (1 - z), z))
            for value in polynomials
        )
    if x != Fraction(1, 2) or y != Fraction(1, 2):
        raise ValueError("A pyramid reference point at apex height must be the apex.")
    budget = _COORDINATE_BUDGET.get()
    image = []
    for value in polynomials:
        if budget is not None:
            budget.reserve(len(value))
        apex: dict[tuple[int, int], Fraction] = {}
        for (first, second, _), coefficient in value.items():
            apex[first, second] = apex.get((first, second), Fraction(0)) + coefficient
        if any(coefficient for index, coefficient in apex.items() if index != (0, 0)):
            raise ValueError(
                "Pyramid coordinate expression is not single-valued at its apex."
            )
        image.append(apex.get((0, 0), Fraction(0)))
    return tuple(image)


def corner_images(
    polynomials: tuple[Polynomial, ...], cell_kind: str
) -> tuple[tuple[Fraction, ...], ...]:
    """Exact images of the reference corners of chart expressions."""
    from ._reference_cell import reference_cell_topology

    return tuple(
        _reference_image(
            polynomials, cell_kind, tuple(Fraction(float(value)) for value in corner)
        )
        for corner in reference_cell_topology(cell_kind).vertices
    )


def coordinate_corner_images(
    element: CellGeometryElement, local: CoordinateCoefficients
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Exact reference-corner images of a full or restricted coordinate map.

    Each target corner is carried exactly through the stored affine restriction
    chain into the root reference, where the root expression is evaluated in
    rational arithmetic. Restrictions that are rational on the target chart,
    such as tetrahedral children of a curved pyramid, need no polynomial form.
    """
    from ._cell_geometry import RestrictedCellGeometryElement
    from ._reference_cell import reference_cell_topology

    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        with budget.temporary_scope():
            return _prepared_coordinate_corner_images(element, local)
    from ._cell_geometry import (
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
    )

    budget = _COORDINATE_BUDGET.get()
    points = [
        tuple(Fraction(float(value)) for value in corner)
        for corner in reference_cell_topology(element.cell_kind).vertices
    ]
    root = element
    while isinstance(
        root,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        if isinstance(
            root,
            (PolynomialComposedCellGeometryElement, RationalComposedCellGeometryElement),
        ):
            arguments = reference_composition_arguments(root)
            points = [
                _reference_image(arguments, root.cell_kind, point) for point in points
            ]
            root = root.source_element
            continue
        rows = [
            [Fraction(value) for value in row]
            for row in np.asarray(root.matrix, dtype=np.float64).tolist()
        ]
        shifts = [
            Fraction(value)
            for value in np.asarray(root.offset, dtype=np.float64).tolist()
        ]
        if budget is not None:
            budget.reserve(len(points) * len(rows) * len(points[0]))
        points = [
            tuple(
                sum(
                    (
                        entry * coordinate
                        for entry, coordinate in zip(row, point, strict=True)
                    ),
                    shift,
                )
                for row, shift in zip(rows, shifts, strict=True)
            )
            for point in points
        ]
        root = root.source_element
    polynomials = coordinate_polynomials(root, local)
    if polynomials is None:
        expressions = coordinate_expressions(root, local)
        if expressions is not None and root.cell_kind != "pyramid":
            return tuple(
                tuple(expression_evaluate(term, point) for term in expressions)
                for point in points
            )
        return None
    return tuple(_reference_image(polynomials, root.cell_kind, point) for point in points)


def _prepared_coordinate_corner_images(
    element: CellGeometryElement, local: CoordinateCoefficients
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Reuse exact source-basis corner weights within one bounded preparation."""
    coefficients, map_key = _coordinate_preparation_key(element, local)
    return _coordinate_corner_images_from_prepared(element, coefficients, map_key)


def _coordinate_corner_images_from_prepared(
    element: CellGeometryElement,
    coefficients: CoordinateSourceBank,
    map_key: tuple[str, str, CoordinateSourceBank],
    /,
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Consume this call's validated canonical bank/key without rescanning it."""
    from ._cell_geometry import _require_scalar_coordinate_element
    from ._reference_cell import reference_cell_topology

    budget = _COORDINATE_BUDGET.get()
    if budget is None:
        raise RuntimeError(
            "Prepared corner weights require their active coordinate ledger."
        )
    cached = budget.coordinate_corner_cache.get(map_key)
    if cached is not None:
        return cached
    key = _element_identity(element)
    prepared_weights = budget.corner_weight_cache.get(key)
    if prepared_weights is None:
        with budget.temporary_scope():
            if _has_cartesian_reference_chain(element):
                root, arguments = coordinate_reference_chain(element)
            else:
                # A complete coefficient action owns its cardinal source law;
                # it is not an authored Cartesian restriction/reference chain.
                root = element
                scalar = _require_scalar_coordinate_element(
                    element, "Exact coordinate corner preparation"
                )
                arguments = chart_arguments(
                    scalar.cell_kind, scalar.topological_dimension
                )
            basis = source_expressions(root)
            if basis is None:
                return None
            corners = tuple(
                tuple(Fraction(float(value)) for value in point)
                for point in reference_cell_topology(element.cell_kind).vertices
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
                    )
                    for value in arguments
                )
                for corner in corners
            )
            rows = []
            for point in points:
                row = []
                for node, term in enumerate(basis):
                    value = (
                        _reference_image((term,), root.cell_kind, point)[0]
                        if not isinstance(term, RationalPolynomial)
                        else expression_reference_evaluate(
                            term,
                            point,
                            "box"
                            if root.cell_kind == "pyramid"
                            else "simplex"
                            if root.cell_kind in ("triangle", "tetrahedron")
                            else "box",
                        )
                    )
                    if value:
                        row.append((node, value))
                rows.append(tuple(row))
            weights = tuple(rows)
            source_identity = _element_identity(root)
            budget.retain_basis((source_identity, weights))
        budget.corner_weight_cache[key] = (source_identity, weights)
    else:
        source_identity, weights = prepared_weights
    dimension = len(coefficients[0])
    budget.reserve(len(weights))
    weighted = tuple(row for row in weights if len(row) != 1 or row[0][1] != 1)
    budget.reserve(dimension * (len(weights) - len(weighted)))
    # A completed cardinal row and complete original bank define the exact
    # linear operation. Different chart owners retain their own map bindings;
    # only an identical operation under the same authentic source law shares
    # its immutable numerical image.
    pending = tuple(
        dict.fromkeys(
            row
            for row in weighted
            if (source_identity, coefficients, row) not in budget.corner_linear_cache
        )
    )
    budget.reserve(0, 128 + 8 * len(weights) + (72 + 8 * dimension) * len(pending))
    if pending:
        budget.reserve(
            sum(len(row) for row in coefficients) + sum(len(row) for row in pending)
        )
        coefficient_numerator = max(
            abs(value.numerator).bit_length() for row in coefficients for value in row
        )
        coefficient_denominator = math.lcm(
            *(value.denominator for row in coefficients for value in row)
        ).bit_length()
        # Every dot product has the source denominator times its weight
        # denominator. Scaling numerators to those denominators and adding its
        # actual terms gives this expression-derived bound before expansion.
        bits = max(
            coefficient_numerator
            + coefficient_denominator
            + max((abs(weight.numerator).bit_length() for _, weight in row), default=0)
            + math.lcm(*(weight.denominator for _, weight in row)).bit_length()
            + max(len(row), 1).bit_length()
            for row in pending
        )
        _reserve_polynomial(
            dimension * sum(len(row) for row in pending),
            dimension * len(pending),
            0,
            bits,
        )
        for row in pending:
            operation_key = source_identity, coefficients, row
            image = tuple(
                sum(
                    (weight * coefficients[node][axis] for node, weight in row),
                    Fraction(0),
                )
                for axis in range(dimension)
            )
            budget.retain_basis((operation_key, image))
            budget.corner_linear_cache[operation_key] = image
    # A nodal corner image is its unit-weight coefficient row itself; sharing that
    # exact record keeps one retained copy per source node across its cells.
    images = tuple(
        coefficients[row[0][0]]
        if len(row) == 1 and row[0][1] == 1
        else budget.corner_linear_cache[source_identity, coefficients, row]
        for row in weights
    )
    budget.retain_basis((map_key, images))
    budget.coordinate_corner_cache[map_key] = images
    return images


def _multi_affine_source(element: CellGeometryElement, /) -> bool:
    """Whether every actual source scalar has degree at most one in each axis."""
    budget = _COORDINATE_BUDGET.get()
    key = None if budget is None else _element_identity(element)
    if budget is not None and key is not None:
        cached = budget.multi_affine_cache.get(key)
        if cached is not None:
            return cached
    basis = source_basis(element)
    if basis is not None and budget is not None:
        budget.reserve(sum(len(term) for term in basis))
    decided = basis is not None and all(
        exponent <= 1 for term in basis for index in term for exponent in index
    )
    if budget is not None and key is not None:
        budget.multi_affine_cache[key] = decided
    return decided


def multi_affine_coordinates(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
) -> tuple[tuple[Polynomial, ...], tuple[tuple[Fraction, ...], ...]] | None:
    """Exact multi-affine box coordinates and their corner images, or ``None``.

    When every scalar of the actual source basis of a direct quadrilateral or
    hexahedron source element has degree at most one in each reference axis, the
    coordinate map is the unique multi-affine interpolant of its exact
    reference-corner images. Its power coefficients follow from those images by
    the exact inclusion-exclusion transform over the corner lattice; the result
    equals the source-basis combination as an exact polynomial. Corner images
    keep the reference vertex order. Restricted, composed and spline charts and
    every other source return ``None`` for the general owner.
    """
    budget = _COORDINATE_BUDGET.get()
    if budget is None:
        return _prepare_multi_affine_coordinates(element, local)
    with budget.temporary_scope():
        return _prepare_multi_affine_coordinates(element, local)


def _prepare_multi_affine_coordinates(
    element: CellGeometryElement, local: CoordinateCoefficients, /
) -> tuple[tuple[Polynomial, ...], tuple[tuple[Fraction, ...], ...]] | None:
    """Prove and retain the complete source interpolant, not a sampled fit."""
    from ._cell_geometry import (
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )

    if (
        element.cell_kind not in ("quadrilateral", "hexahedron")
        or isinstance(
            element,
            (
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
                RestrictedCellGeometryElement,
                SplineCellGeometryElement,
            ),
        )
        or not _multi_affine_source(element)
    ):
        return None
    budget = _COORDINATE_BUDGET.get()
    key = None
    coefficients = None
    if budget is not None:
        coefficients, key = _coordinate_preparation_key(element, local)
        cached = budget.polynomial_cache.get(key)
        corners = budget.coordinate_corner_cache.get(key)
        if cached is not None and corners is not None:
            return cached, corners
    if budget is not None and key is not None and coefficients is not None:
        # The complete bank and scientific element identity were validated above.
        # Share that actual preparation with corner construction, not a second
        # coefficient-row normalization/validation traversal.
        with budget.temporary_scope():
            corners = _coordinate_corner_images_from_prepared(element, coefficients, key)
    else:
        corners = coordinate_corner_images(element, local)
    if corners is None:
        return None
    masks, exponents = _box_corner_lattice(element.cell_kind)
    dimension, count, ambient = len(exponents[0]), len(exponents), len(corners[0])
    values: list[tuple[Fraction, ...]] = [()] * count
    for mask, image in zip(masks, corners, strict=True):
        values[mask] = image
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        # Every power coefficient is a signed sum of at most ``count`` images over
        # their common denominator; each axis pass adds at most one numerator bit.
        budget.reserve(count * ambient)
        numerator_bits, common = 1, 1
        for image in corners:
            for value in image:
                numerator_bits = max(numerator_bits, abs(value.numerator).bit_length())
                common = math.lcm(common, value.denominator)
        _reserve_polynomial(
            dimension * (count // 2) * ambient,
            count * ambient,
            dimension,
            numerator_bits + common.bit_length() + dimension,
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
    if budget is not None and key is not None:
        budget.retain_basis((key, polynomials, corners))
        budget.coordinate_cache[key] = polynomials
        budget.polynomial_cache[key] = polynomials
        budget.coordinate_corner_cache[key] = corners
    return polynomials, corners


@cache
def _box_corner_lattice(
    cell_kind: str, /
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Axis bitmask of each reference corner and the monomial exponent of each mask.

    The exponent tuples are shared by every multi-affine coordinate expression.
    """
    from ._reference_cell import reference_cell_topology

    reference = np.asarray(reference_cell_topology(cell_kind).vertices, dtype=np.int64)
    dimension = reference.shape[1]
    masks = tuple(
        int(sum(int(value) << axis for axis, value in enumerate(point)))
        for point in reference
    )
    if sorted(masks) != list(range(1 << dimension)):
        raise ValueError(
            "A corner lattice requires the vertices of a unit reference box."
        )
    exponents = tuple(
        tuple((mask >> axis) & 1 for axis in range(dimension))
        for mask in range(1 << dimension)
    )
    return masks, exponents


def rounded_point(point: tuple[Fraction, ...], /) -> np.ndarray:
    """Correctly rounded (nearest, ties to even) binary64 image of an exact point."""
    return np.asarray([float(value) for value in point], dtype=np.float64)


def _solve_exact(
    matrix: list[list[Fraction]], right: list[list[Fraction]]
) -> list[list[Fraction]]:
    size = len(matrix)
    rows = [list(a) + list(b) for a, b in zip(matrix, right, strict=True)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if rows[row][column]), None)
        if pivot is None:
            raise ValueError("Canonical coordinate lattice is not unisolvent.")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for row in range(size):
            if row == column or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                a - factor * b for a, b in zip(rows[row], rows[column], strict=True)
            ]
    return [row[size:] for row in rows]


_LATTICE_BASIS_CACHE: dict[tuple[str, int], tuple[Polynomial, ...]] = {}


def lattice_basis(cell_kind: str, degree: int) -> tuple[Polynomial, ...]:
    key = (cell_kind, degree)
    budget = _COORDINATE_BUDGET.get()
    owner = _LATTICE_BASIS_CACHE if budget is None else budget.lattice_cache
    basis = owner.get(key)
    if basis is None:
        if budget is not None:
            from ._reference_cell import reference_cell_topology

            dimension = reference_cell_topology(cell_kind).dimension
            candidates = (degree + 1) ** dimension
            _reserve_polynomial(
                candidates, candidates, dimension, max(degree, 1).bit_length()
            )
        basis = _lattice_basis(cell_kind, degree)
    if budget is not None:
        budget.retain_basis(basis)
    owner[key] = basis
    return basis


def _lattice_basis(cell_kind: str, degree: int) -> tuple[Polynomial, ...]:
    """Ideal equispaced coordinate basis, with exact rational lattice identity."""
    from ._reference_cell import reference_cell_topology

    dimension = reference_cell_topology(cell_kind).dimension
    if degree == 1:
        return _linear_reference_basis(cell_kind)
    variables = axes(dimension)
    if cell_kind in ("triangle", "tetrahedron", "prism") or cell_kind.startswith(
        "simplex:"
    ):
        simplex_dimension = 2 if cell_kind == "prism" else dimension
        barycentric = (
            add(
                constant(1, dimension),
                scale(sum_polynomials(variables[:simplex_dimension]), -1),
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
            term = constant(1, dimension)
            for variable, exponent in zip(
                barycentric, (degree - sum(index), *index), strict=True
            ):
                for offset in range(exponent):
                    term = multiply(
                        term,
                        scale(
                            add(scale(variable, degree), constant(-offset, dimension)),
                            Fraction(1, offset + 1),
                        ),
                    )
            simplex.append(term)
        if cell_kind != "prism":
            return tuple(simplex)
        axial = _nodal_axis_exact(
            tuple(Fraction(i, degree) for i in range(degree + 1)), 2, 3
        )
        return tuple(multiply(a, b) for a, b in product(simplex, axial))
    if cell_kind != "pyramid":
        factors = tuple(
            _nodal_axis_exact(
                tuple(Fraction(i, degree) for i in range(degree + 1)), axis, dimension
            )
            for axis in range(dimension)
        )
        return tuple(
            math_product_polynomials(choice, dimension) for choice in product(*factors)
        )
    collapse = add(constant(1, 3), scale(variables[2], -1))
    indices = tuple(
        (i, j, k)
        for i in range(degree + 1)
        for j in range(degree + 1)
        for k in range(degree - max(i, j) + 1)
    )
    modes = tuple(
        math_product_polynomials(
            (
                _jacobi(i, 0, 0, 3),
                _jacobi(j, 0, 1, 3),
                power(collapse, max(i, j), 3),
                _jacobi(k, 2 * max(i, j) + 2, 2, 3),
            ),
            3,
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
    table = [[evaluate(mode, point) for mode in modes] for point in nodes]
    coefficients = _solve_exact(
        table,
        [[Fraction(int(i == j)) for j in range(len(nodes))] for i in range(len(nodes))],
    )
    return tuple(
        sum_polynomials(
            tuple(
                scale(mode, row[i]) for mode, row in zip(modes, coefficients, strict=True)
            )
        )
        for i in range(len(nodes))
    )


def _nodal_axis_exact(
    nodes: tuple[Fraction, ...], axis: int, dimension: int
) -> tuple[Polynomial, ...]:
    variable = axes(dimension)[axis]
    result = []
    for i, x in enumerate(nodes):
        term = constant(1, dimension)
        for j, y in enumerate(nodes):
            if i != j:
                term = multiply(
                    term, scale(add(variable, constant(-y, dimension)), 1 / (x - y))
                )
        result.append(term)
    return tuple(result)


def evaluate_coordinate_lattice_basis(
    cell_kind: str, degree: int, points: ArrayLike
) -> tuple[Array, Array]:
    """JAX execution of ideal source expressions; coefficients are static data."""
    import jax.numpy as jnp

    from ._reference_cell import reference_cell_topology

    reference = jnp.asarray(points, dtype=jnp.float64)
    dimension = reference_cell_topology(cell_kind).dimension
    if reference.ndim != 2 or reference.shape[1] != dimension:
        raise ValueError("Coordinate reference points have incompatible shape.")
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
                np.asarray(reference_cell_topology(cell_kind).vertices, dtype=np.int64).T,
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
    basis = lattice_basis(cell_kind, degree)
    values = []
    gradients = []
    for polynomial in basis:
        values.append(_evaluate_jax_polynomial(polynomial, parameters))
        rows = physical_jacobian((polynomial,), cell_kind, dimension)
        if rows is None:
            raise ValueError("Coordinate source has an unresolved rational denominator.")
        gradients.append(
            jnp.stack(
                tuple(_evaluate_jax_polynomial(entry, parameters) for entry in rows[0]),
                axis=-1,
            )
        )
    return jnp.stack(values, axis=-1), jnp.stack(gradients, axis=1)


def _evaluate_nodal_axis(degree: int, points: Array) -> tuple[Array, Array]:
    """Factored rational lattice evaluation without monomial cancellation."""
    import jax.numpy as jnp

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
    """Evaluate the exact falling-factorial barycentric source in factored form."""
    import jax.numpy as jnp

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


def _evaluate_jax_polynomial(polynomial: Polynomial, points: Array) -> Array:
    import jax.numpy as jnp

    if not polynomial:
        return jnp.zeros((points.shape[0],), dtype=jnp.float64)
    coefficients = jnp.asarray(
        tuple(float(value) for value in polynomial.values()), dtype=jnp.float64
    )
    # Exponents are static source data: Python integer powers lower to exact
    # repeated multiplication and stay valid under strict dtype promotion.
    monomials = tuple(
        jnp.prod(
            jnp.stack(
                tuple(points[:, axis] ** exponent for axis, exponent in enumerate(index)),
                axis=-1,
            ),
            axis=-1,
        )
        for index in polynomial
    )
    return jnp.stack(monomials, axis=-1) @ coefficients


def divide_polynomial(
    numerator: Polynomial, denominator: Polynomial
) -> Polynomial | None:
    if not denominator:
        raise ValueError("Source rational denominator is zero.")
    leading = max(denominator)
    coefficient = denominator[leading]
    remainder = dict(numerator)
    quotient: Polynomial = {}
    while remainder:
        index = max(remainder)
        if any(a < b for a, b in zip(index, leading, strict=True)):
            return None
        exponent = tuple(a - b for a, b in zip(index, leading, strict=True))
        value = remainder[index] / coefficient
        quotient[exponent] = quotient.get(exponent, Fraction(0)) + value
        remainder = add(
            remainder,
            scale(multiply({exponent: value}, denominator), -1),
        )
    return quotient


def _restrict_arguments(
    coordinates: tuple[Polynomial, ...],
    source_kind: str,
    arguments: tuple[Polynomial, ...],
    dimension: int,
) -> tuple[Polynomial, ...] | None:
    """Compose source chart expressions with source-reference arguments.

    Pyramid sources are homogenized in ``1 - z`` and divided exactly; ``None``
    reports a collapsed denominator that does not cancel.
    """
    if source_kind != "pyramid":
        return tuple(compose(value, arguments) for value in coordinates)
    height = arguments[2]
    denominator = add(constant(1, dimension), scale(height, -1))
    horizontal = (
        add(arguments[0], scale(height, Fraction(-1, 2))),
        add(arguments[1], scale(height, Fraction(-1, 2))),
    )
    result = []
    for polynomial in coordinates:
        degree = max((index[0] + index[1] for index in polynomial), default=0)
        numerator = sum_polynomials(
            tuple(
                scale(
                    math_product_polynomials(
                        (
                            power(horizontal[0], index[0], dimension),
                            power(horizontal[1], index[1], dimension),
                            power(height, index[2], dimension),
                            power(denominator, degree - index[0] - index[1], dimension),
                        ),
                        dimension,
                    ),
                    value,
                )
                for index, value in polynomial.items()
            )
        )
        quotient = divide_polynomial(numerator, power(denominator, degree, dimension))
        if quotient is None:
            return None
        result.append(quotient)
    return tuple(result)


def restrict_coordinates(
    coordinates: tuple[Polynomial, ...],
    source_kind: str,
    origin: np.ndarray,
    matrix: np.ndarray,
) -> tuple[Polynomial, ...] | None:
    """Exact physical reference restriction; cancel removable apex denominators."""
    return _restrict_arguments(
        coordinates, source_kind, affine_arguments(origin, matrix), matrix.shape[1]
    )


def chart_arguments(cell_kind: str, dimension: int) -> tuple[Polynomial, ...]:
    """Physical reference coordinates as expressions on the cell's chart.

    Pyramid expressions use the collapsed cube; other cells use their reference.
    """
    variables = axes(dimension)
    if cell_kind != "pyramid":
        return variables
    u, v, w = variables
    collapse = add(constant(1, 3), scale(w, -1))
    return (
        add(multiply(u, collapse), scale(w, Fraction(1, 2))),
        add(multiply(v, collapse), scale(w, Fraction(1, 2))),
        w,
    )


def restrict_chart_coordinates(
    coordinates: tuple[Polynomial, ...],
    source_kind: str,
    target_kind: str,
    origin: np.ndarray,
    matrix: np.ndarray,
) -> tuple[Polynomial, ...] | None:
    """Exact restriction from the source chart to the target chart.

    The target chart is inserted before cancelling pyramid denominators, so a
    restriction that is polynomial on the target chart (for example an
    apex-sharing sub-pyramid) is found even when it is rational on the target
    physical reference. ``None`` reports a genuinely rational restriction.
    """
    dimension = matrix.shape[1]
    chart = chart_arguments(target_kind, dimension)
    return _restrict_arguments(
        coordinates,
        source_kind,
        tuple(compose(term, chart) for term in affine_arguments(origin, matrix)),
        dimension,
    )


def _linear_reference_basis(cell_kind: str) -> tuple[Polynomial, ...]:
    from ._reference_cell import reference_cell_topology

    topology = reference_cell_topology(cell_kind)
    dimension = topology.dimension
    variables = axes(dimension)
    if cell_kind in ("triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        return (
            add(constant(1, dimension), scale(sum_polynomials(variables), -1)),
            *variables,
        )
    if cell_kind == "prism":
        x, y, z = variables
        triangle = (add(constant(1, 3), scale(add(x, y), -1)), x, y)
        collapse = add(constant(1, 3), scale(z, -1))
        return tuple(multiply(value, collapse) for value in triangle) + tuple(
            multiply(value, z) for value in triangle
        )
    if cell_kind == "pyramid":
        u, v, w = variables
        collapse = add(constant(1, 3), scale(w, -1))
        factors = tuple(
            math_product_polynomials(
                tuple(
                    variable if int(value) else add(constant(1, 3), scale(variable, -1))
                    for variable, value in zip((u, v), vertex[:2], strict=True)
                ),
                3,
            )
            for vertex in topology.vertices[:4]
        )
        return tuple(multiply(value, collapse) for value in factors) + (w,)
    return tuple(
        math_product_polynomials(
            tuple(
                variable
                if int(value)
                else add(constant(1, dimension), scale(variable, -1))
                for variable, value in zip(variables, vertex, strict=True)
            ),
            dimension,
        )
        for vertex in topology.vertices
    )


@cache
def _pyramid_execution_table(
    degree: int,
) -> tuple[tuple[tuple[int, int, int], ...], np.ndarray]:
    """Prepare the factored modal execution of the ideal rational lattice."""
    from .fem._spectral_hp_completion import _pyramid_modal_tabulation

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
    table = _pyramid_modal_tabulation(nodes, indices)[0]
    coefficients = np.linalg.solve(table, np.eye(table.shape[0], dtype=np.float64))
    return indices, coefficients


def _jacobi_jax(degree: int, alpha: int, beta: int, points: Array) -> Array:
    import jax.numpy as jnp

    previous = jnp.ones_like(points)
    if degree == 0:
        return previous
    x = 2.0 * points - 1.0
    current = 0.5 * (alpha - beta + (alpha + beta + 2) * x)
    for n in range(2, degree + 1):
        ab = alpha + beta
        numerator = (
            (2 * n + ab - 1)
            * ((2 * n + ab) * (2 * n + ab - 2) * x + alpha * alpha - beta * beta)
            * current
        )
        numerator -= 2 * (n + alpha - 1) * (n + beta - 1) * (2 * n + ab) * previous
        following = numerator / (2 * n * (n + ab) * (2 * n + ab - 2))
        previous, current = current, following
    return current


def _jacobi_derivative_jax(degree: int, alpha: int, beta: int, points: Array) -> Array:
    import jax.numpy as jnp

    return (
        jnp.zeros_like(points)
        if degree == 0
        else (degree + alpha + beta + 1)
        * _jacobi_jax(degree - 1, alpha + 1, beta + 1, points)
    )


def _evaluate_pyramid_lattice(degree: int, parameters: Array) -> tuple[Array, Array]:
    import jax.numpy as jnp

    indices, coefficients = _pyramid_execution_table(degree)
    u, v, w = (parameters[:, axis] for axis in range(3))
    collapse = 1.0 - w
    horizontal = tuple(
        (
            _jacobi_jax(i, 0, 0, u),
            _jacobi_jax(i, 0, 0, v),
            _jacobi_derivative_jax(i, 0, 0, u),
            _jacobi_derivative_jax(i, 0, 0, v),
        )
        for i in range(degree + 1)
    )
    vertical = {
        (m, k): (
            _jacobi_jax(k, 2 * m + 2, 0, w),
            _jacobi_derivative_jax(k, 2 * m + 2, 0, w),
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
    """Count the complete control net before allocating its coefficient table."""
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
    """Bind actual source semantics, independently of cached element IDs."""
    from ._cell_geometry import (
        _CoordinateTabulator,
        _SweptCoordinateTabulator,
        BarycentricCellGeometryElement,
        LayerColumnCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )
    from .fem._form_elements import FormBasis
    from .fem._high_order import ReferenceNodalFamily, SimplexNodalFamily
    from .fem._spectral_hp_completion import HybridReferenceFamily

    result: dict[str, object] = {
        "declared_element": element.element_id,
        "cell_kind": element.cell_kind,
        "local_dof_count": element.local_dof_count,
    }
    if isinstance(element, BarycentricCellGeometryElement):
        result["operator"] = "affine-p1-full-barycentric-action"
        result["source"] = coordinate_source_signature(element.source_element)
        result["target_dimension"] = element.topological_dimension
        result["barycentric_weights"] = tuple(
            tuple(row) for row in np.asarray(element.barycentric_weights).tolist()
        )
    elif isinstance(element, LayerColumnCellGeometryElement):
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
    elif isinstance(element, SplineCellGeometryElement):
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
    elif isinstance(
        element,
        (PolynomialComposedCellGeometryElement, RationalComposedCellGeometryElement),
    ):
        result["operator"] = (
            "exact-rational-collapsed-pyramid-reference-composition"
            if isinstance(element, RationalComposedCellGeometryElement)
            else "exact-polynomial-reference-composition"
        )
        result["source"] = coordinate_source_signature(element.source_element)
        result["chart"] = coordinate_source_signature(element.chart_element)
        result["chart_coefficients"] = element.chart_coefficients
        result["target_dimension"] = element.topological_dimension
    elif isinstance(element, RestrictedCellGeometryElement):
        result["operator"] = "exact-reference-restriction"
        result["source"] = coordinate_source_signature(element.source_element)
        result["target_dimension"] = element.topological_dimension
        result["matrix"] = tuple(
            tuple(row) for row in np.asarray(element.matrix).tolist()
        )
        result["offset"] = tuple(np.asarray(element.offset).tolist())
    elif isinstance(element, FiniteElementSpec):
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
        if type(tabulator) is _SweptCoordinateTabulator:
            result["operator"] = "exact-profile-times-linear-station"
            result["profile"] = coordinate_source_signature(tabulator.source)
            result["station_axis"] = 2
            result["station_degree"] = 1
            result["endpoint_order"] = (0, 1)
        elif isinstance(tabulator, _CoordinateTabulator):
            result["operator"] = "exact-rational-coordinate-lattice"
            result["lattice"] = (
                tabulator.cell_kind,
                tabulator.degree,
                tabulator.source_basis_semantics,
            )
        else:
            owner = owning_tabulator_source(tabulator)
            if isinstance(owner, FiniteElementSpec):
                result["operator"] = "reference-source-forwarding"
                result["source"] = coordinate_source_signature(owner)
            elif isinstance(owner, FormBasis):
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
            elif isinstance(owner, SimplexNodalFamily):
                result["operator"] = "simplex-bernstein-source"
                result["source_order"] = (
                    owner.cell_kind,
                    owner.order,
                    owner.multiindices,
                )
            elif isinstance(owner, ReferenceNodalFamily):
                result["operator"] = "tensor-nodal-source"
                result["source_order"] = (owner.cell_kind, owner.orders)
            elif isinstance(owner, HybridReferenceFamily):
                result["operator"] = "hybrid-source"
                result["source_order"] = (
                    owner.cell_kind,
                    owner.orders,
                    owner.modal_indices,
                    owner.basis_permutation,
                )
            else:
                result["operator"] = (
                    "builtin-reference-source"
                    if tabulator is None
                    else "unsupported-tabulator"
                )
    else:
        result["operator"] = "vertex-coordinate-source"
    return result


def owning_tabulator_source(tabulator: object) -> object | None:
    """Unwrap only an exact immutable reference-owner/method pair."""
    from .fem._form_elements import _ProxyTabulator
    from .fem._high_order import ReferenceNodalFamily, SimplexNodalFamily
    from .fem._reference import _FiniteElementTabulator
    from .fem._spectral_hp_completion import HybridReferenceFamily

    if type(tabulator) is _ProxyTabulator:
        return tabulator.basis

    if (
        isinstance(tabulator, _FiniteElementTabulator)
        and type(tabulator) is _FiniteElementTabulator
    ):
        source = tabulator.source
        if (
            type(source) in (FiniteElementSpec, ReferenceNodalFamily, SimplexNodalFamily)
            and tabulator.method == "tabulate"
        ):
            return source
        if (
            type(source) is HybridReferenceFamily
            and tabulator.method == "tabulate_with_gradients"
        ):
            return source
        return None
    if isinstance(tabulator, _BoundSourceTabulator):
        source = tabulator.__self__
        if (
            type(source) is SimplexNodalFamily
            and tabulator.__func__ is SimplexNodalFamily.tabulate
        ):
            return source
        if (
            type(source) is ReferenceNodalFamily
            and tabulator.__func__ is ReferenceNodalFamily.tabulate
        ):
            return source
        if (
            type(source) is HybridReferenceFamily
            and tabulator.__func__ is HybridReferenceFamily.tabulate_with_gradients
        ):
            return source
        if (
            type(source) is FiniteElementSpec
            and tabulator.__func__ is FiniteElementSpec.tabulate
        ):
            return source
    return None


def rational_expression(numerator: Polynomial, denominator: Polynomial) -> Expression:
    """Cancel a removable denominator exactly, retaining a genuine quotient."""
    if numerator and denominator:
        dimension = len(next(iter(denominator)))
        budget = _COORDINATE_BUDGET.get()
        if budget is not None:
            budget.reserve(len(numerator) + len(denominator))
        shift = tuple(
            min(
                min(index[axis] for index in numerator),
                min(index[axis] for index in denominator),
            )
            for axis in range(dimension)
        )
        if any(shift):
            if budget is not None:
                numerator_bits, denominator_bits = _coefficient_profile(
                    (numerator, denominator)
                )
                _reserve_polynomial(
                    len(numerator) + len(denominator),
                    len(numerator) + len(denominator),
                    dimension,
                    numerator_bits + denominator_bits,
                )
            numerator = {
                tuple(a - b for a, b in zip(index, shift, strict=True)): value
                for index, value in numerator.items()
            }
            denominator = {
                tuple(a - b for a, b in zip(index, shift, strict=True)): value
                for index, value in denominator.items()
            }
    quotient = divide_polynomial(numerator, denominator)
    return RationalPolynomial(numerator, denominator) if quotient is None else quotient


def expression_parts(value: Expression, dimension: int) -> tuple[Polynomial, Polynomial]:
    return (
        (value.numerator, value.denominator)
        if isinstance(value, RationalPolynomial)
        else (value, constant(1, dimension))
    )


def expression_add(first: Expression, second: Expression) -> Expression:
    if not isinstance(first, RationalPolynomial) and not isinstance(
        second, RationalPolynomial
    ):
        return add(first, second)
    rational = first if isinstance(first, RationalPolynomial) else second
    if not isinstance(rational, RationalPolynomial):
        raise TypeError("Rational expression addition requires a quotient.")
    dimension = len(next(iter(rational.denominator)))
    a, b = expression_parts(first, dimension)
    c, d = expression_parts(second, dimension)
    if b == d:
        return rational_expression(add(a, c), b)
    quotient = divide_polynomial(b, d)
    if quotient is not None:
        return rational_expression(add(a, multiply(c, quotient)), b)
    quotient = divide_polynomial(d, b)
    if quotient is not None:
        return rational_expression(add(multiply(a, quotient), c), d)
    return rational_expression(add(multiply(a, d), multiply(c, b)), multiply(b, d))


def expression_scale(value: Expression, coefficient: Fraction | int) -> Expression:
    if isinstance(value, RationalPolynomial):
        return rational_expression(scale(value.numerator, coefficient), value.denominator)
    return scale(value, coefficient)


def expression_multiply(first: Expression, second: Expression) -> Expression:
    if not isinstance(first, RationalPolynomial) and not isinstance(
        second, RationalPolynomial
    ):
        return multiply(first, second)
    rational = first if isinstance(first, RationalPolynomial) else second
    if not isinstance(rational, RationalPolynomial):
        raise TypeError("Rational expression multiplication requires a quotient.")
    dimension = len(next(iter(rational.denominator)))
    a, b = expression_parts(first, dimension)
    c, d = expression_parts(second, dimension)
    return rational_expression(multiply(a, c), multiply(b, d))


def expression_sum(values: tuple[Expression, ...]) -> Expression:
    result: Expression = {}
    for value in values:
        result = expression_add(result, value)
    return result


def expression_linear_combinations(
    values: tuple[Expression, ...],
    weight_bank: tuple[tuple[Fraction, ...], ...],
) -> tuple[Expression, ...]:
    """Share polynomial bank preparation; retain genuine rational arithmetic."""
    if any(len(values) != len(weights) for weights in weight_bank):
        raise ValueError("Linear combination weights must match the expression bank.")
    polynomials: list[Polynomial] = []
    for value in values:
        if isinstance(value, RationalPolynomial):
            return tuple(
                expression_sum(
                    tuple(
                        expression_scale(entry, weight)
                        for entry, weight in zip(values, weights, strict=True)
                    )
                )
                for weights in weight_bank
            )
        polynomials.append(value)
    return linear_combinations(tuple(polynomials), weight_bank)


def expression_derivative(value: Expression, axis: int) -> Expression:
    if isinstance(value, RationalPolynomial):
        return rational_expression(
            add(
                multiply(derivative(value.numerator, axis), value.denominator),
                scale(multiply(value.numerator, derivative(value.denominator, axis)), -1),
            ),
            multiply(value.denominator, value.denominator),
        )
    return derivative(value, axis)


def expression_compose(
    value: Expression, arguments: tuple[Expression, ...] | PreparedPolynomialArguments
) -> Expression:
    return ExpressionComposition(arguments)(value)


def expression_evaluate(value: Expression, point: tuple[Fraction, ...]) -> Fraction:
    if isinstance(value, RationalPolynomial):
        denominator = evaluate(value.denominator, point)
        if not denominator:
            # A singular chart point is admissible only when exact algebra
            # removes the singularity; never pick a sampled directional limit.
            raise RationalEnclosureError(
                "Rational source evaluation has an unresolved chart singularity."
            )
        return evaluate(value.numerator, point) / denominator
    return evaluate(value, point)


def _expression_control_net(
    value: Polynomial, domain: str, degrees: tuple[int, ...], dimension: int
) -> tuple[Fraction, ...]:
    reference = parse(domain, BernsteinDomain, "domain")
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        numerator, denominator = _coefficient_profile((value,))
        visits, nodes, entries = _bernstein_support_profile(
            value, reference, degrees, dimension
        )
        _reserve_polynomial(
            visits,
            nodes + entries,
            dimension,
            numerator
            + denominator
            + 2
            * dimension
            * max(degrees, default=0)
            * max(max(degrees, default=0) + 1, 1).bit_length(),
        )
    lattice = _bernstein_lattice(reference, degrees, dimension)
    controls = [Fraction(0) for _ in lattice]
    for index, coefficient in value.items():
        denominator, entries = _bernstein_weights(reference, degrees, dimension, index)
        for position, numerator in entries:
            controls[position] += coefficient * Fraction(numerator, denominator)
    return tuple(controls)


def expression_bernstein_coefficients(
    value: Expression, domain: str, dimension: int
) -> tuple[Fraction, ...]:
    """Polynomial controls or rigorously bounding rational control ratios.

    A common-degree rational Bernstein net is a convex combination of ratios
    with weights denominator_control * Bernstein_basis. Zero weights may be
    omitted only when the corresponding numerator control is also exactly zero.
    """
    if not isinstance(value, RationalPolynomial):
        return bernstein_coefficients(value, domain, dimension)
    indices = tuple(value.numerator) + tuple(value.denominator)
    if domain == "simplex":
        degrees = (max((sum(index) for index in indices), default=0),)
    elif domain == "prism":
        degrees = (
            max((sum(index[:2]) for index in indices), default=0),
            max((index[2] for index in indices), default=0),
        )
    else:
        degrees = tuple(
            max((index[axis] for index in indices), default=0)
            for axis in range(dimension)
        )
    numerator = _expression_control_net(value.numerator, domain, degrees, dimension)
    denominator = _expression_control_net(value.denominator, domain, degrees, dimension)
    if max(denominator) < 0:
        numerator = tuple(-coefficient for coefficient in numerator)
        denominator = tuple(-coefficient for coefficient in denominator)
    if (
        min(denominator) < 0
        or not any(denominator)
        or any(a and not b for a, b in zip(numerator, denominator, strict=True))
    ):
        raise RationalEnclosureError(
            "Rational source denominator has no sign-definite Bernstein enclosure."
        )
    return tuple(a / b for a, b in zip(numerator, denominator, strict=True) if b)


def expression_bounds(
    value: Expression, domain: str, dimension: int
) -> tuple[float, float]:
    coefficients = expression_bernstein_coefficients(value, domain, dimension)
    return outward(min(coefficients), -math.inf), outward(max(coefficients), math.inf)


def expression_determinant(
    matrix: tuple[tuple[Expression, ...], ...],
    *,
    variable_dimension: int | None = None,
) -> Expression:
    from itertools import permutations

    dimension = len(matrix)
    variables = dimension if variable_dimension is None else variable_dimension
    if isinstance(variables, bool) or not isinstance(variables, (int, np.integer)):
        raise TypeError("Expression determinant variable dimension must be an integer.")
    if variables < 0 or any(len(row) != dimension for row in matrix):
        raise ValueError(
            "Expression determinant requires a square matrix and nonnegative variable dimension."
        )
    for row in matrix:
        for value in row:
            numerator, denominator = expression_parts(value, variables)
            if any(
                len(index) != variables
                for polynomial in (numerator, denominator)
                for index in polynomial
            ):
                raise ValueError(
                    "Expression determinant entries use different declared parameter axes."
                )
    result: Expression = {}
    for permutation in permutations(range(dimension)):
        if any(not matrix[row][column] for row, column in enumerate(permutation)):
            continue
        term: Expression = constant(1, variables)
        for row, column in enumerate(permutation):
            term = expression_multiply(term, matrix[row][column])
        sign = (-1) ** sum(
            permutation[i] > permutation[j]
            for i in range(dimension)
            for j in range(i + 1, dimension)
        )
        result = expression_add(result, expression_scale(term, sign))
    return result


def restrict_chart_expressions(
    coordinates: tuple[Expression, ...],
    source_kind: str,
    target_kind: str,
    origin: np.ndarray,
    matrix: np.ndarray,
) -> tuple[Expression, ...]:
    dimension = matrix.shape[1]
    chart = chart_arguments(target_kind, dimension)
    arguments = tuple(compose(term, chart) for term in affine_arguments(origin, matrix))
    if source_kind != "pyramid":
        composition = ExpressionComposition(arguments)
        return tuple(composition(value) for value in coordinates)
    height = arguments[2]
    denominator = add(constant(1, dimension), scale(height, -1))
    horizontal = (
        add(arguments[0], scale(height, Fraction(-1, 2))),
        add(arguments[1], scale(height, Fraction(-1, 2))),
    )

    def homogenize(polynomial: Polynomial) -> tuple[Polynomial, Polynomial]:
        degree = max((index[0] + index[1] for index in polynomial), default=0)
        numerator = sum_polynomials(
            tuple(
                scale(
                    math_product_polynomials(
                        (
                            power(horizontal[0], index[0], dimension),
                            power(horizontal[1], index[1], dimension),
                            power(height, index[2], dimension),
                            power(denominator, degree - index[0] - index[1], dimension),
                        ),
                        dimension,
                    ),
                    value,
                )
                for index, value in polynomial.items()
            )
        )
        return numerator, power(denominator, degree, dimension)

    result = []
    for value in coordinates:
        first, second = expression_parts(value, len(arguments))
        a, b = homogenize(first)
        c, d = homogenize(second)
        result.append(rational_expression(multiply(a, d), multiply(b, c)))
    return tuple(result)


def coordinate_expressions(
    element: CellGeometryElement, local: CoordinateCoefficients
) -> tuple[Expression, ...] | None:
    """Actual exact source, including genuinely rational nested restrictions."""
    budget = _COORDINATE_BUDGET.get()
    if budget is None:
        return _prepare_coordinate_expressions(element, local)
    with budget.temporary_scope():
        coefficients, cache_key = _coordinate_preparation_key(element, local)
        cached = budget.coordinate_cache.get(cache_key)
        if cached is not None:
            return cached
        coordinates = _prepare_coordinate_expressions(element, coefficients)
        if coordinates is not None and cache_key not in budget.coordinate_cache:
            budget.retain_basis((cache_key, coordinates))
            budget.coordinate_cache[cache_key] = coordinates
        return coordinates


def _prepare_coordinate_expressions(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
) -> tuple[Expression, ...] | None:
    from ._cell_geometry import (
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )

    if isinstance(
        element,
        (PolynomialComposedCellGeometryElement, RationalComposedCellGeometryElement),
    ):
        # Form the original physical map once per authentic coefficient bank,
        # then pull it back. Recombining every composed cardinal basis repeats
        # the same parent affine action for every child/reference chart.
        source = coordinate_expressions(element.source_element, local)
        if source is None:
            return None
        composition = ExpressionComposition(prepared_reference_arguments(element))
        return tuple(composition(value) for value in source)

    if isinstance(element, SplineCellGeometryElement):
        basis = source_expressions(element)
        if basis is None:
            return None
        rows = _coordinate_source_rows(local)
        return expression_linear_combinations(
            basis, tuple(tuple(row[axis] for row in rows) for axis in range(len(rows[0])))
        )

    if isinstance(element, RestrictedCellGeometryElement):
        source = coordinate_expressions(element.source_element, local)
        if source is None:
            return None
        return restrict_chart_expressions(
            source,
            element.source_element.cell_kind,
            element.cell_kind,
            np.asarray(element.offset),
            np.asarray(element.matrix),
        )
    return coordinate_polynomials(element, local)


class ExpressionComposition:
    """Share exact powers on polynomial or owning rational reference actions."""

    def __init__(
        self, arguments: tuple[Expression, ...] | PreparedPolynomialArguments, /
    ) -> None:
        self.composition: ArgumentComposition | None = None
        self.denominator: Polynomial | None = None
        self.rational_composition: ArgumentComposition | None = None
        if isinstance(arguments, PreparedPolynomialArguments):
            self.composition = ArgumentComposition(arguments)
            return
        if not any(isinstance(value, RationalPolynomial) for value in arguments):
            polynomials = tuple(
                value for value in arguments if not isinstance(value, RationalPolynomial)
            )
            self.composition = ArgumentComposition(polynomials)
            return
        rational = next(
            value for value in arguments if isinstance(value, RationalPolynomial)
        )
        dimension = len(next(iter(rational.denominator)))
        parts = tuple(expression_parts(value, dimension) for value in arguments)
        denominator = constant(1, dimension)
        for _, divisor in parts:
            if (
                divisor == denominator
                or divide_polynomial(denominator, divisor) is not None
            ):
                continue
            if divide_polynomial(divisor, denominator) is not None:
                denominator = divisor
            else:
                denominator = multiply(denominator, divisor)
        numerators: list[Polynomial] = []
        for numerator, divisor in parts:
            factor = divide_polynomial(denominator, divisor)
            if factor is None:
                raise ValueError(
                    "Rational reference composition lost its exact common denominator."
                )
            numerators.append(multiply(numerator, factor))
        self.denominator = denominator
        self.rational_composition = ArgumentComposition((*numerators, denominator))

    def _polynomial(self, value: Polynomial) -> tuple[Polynomial, Polynomial]:
        if self.rational_composition is None or self.denominator is None:
            raise ValueError(
                "A rational reference action requires its owning homogeneous preparation."
            )
        dimension = len(next(iter(self.denominator)))
        numerator_bits, denominator_bits = _coefficient_profile((value,))
        source_dimension = len(next(iter(value), ()))
        _reserve_polynomial(
            len(value),
            len(value),
            source_dimension + 1,
            numerator_bits + denominator_bits,
        )
        degree = max((sum(index) for index in value), default=0)
        lifted: Polynomial = {
            (*index, degree - sum(index)): coefficient
            for index, coefficient in value.items()
        }
        return self.rational_composition(lifted), power(
            self.denominator, degree, dimension
        )

    def __call__(self, value: Expression, /) -> Expression:
        if self.composition is not None:
            if isinstance(value, RationalPolynomial):
                return rational_expression(
                    self.composition(value.numerator), self.composition(value.denominator)
                )
            return self.composition(value)
        if isinstance(value, RationalPolynomial):
            numerator, numerator_divisor = self._polynomial(value.numerator)
            denominator, denominator_divisor = self._polynomial(value.denominator)
            return rational_expression(
                multiply(numerator, denominator_divisor),
                multiply(numerator_divisor, denominator),
            )
        numerator, denominator = self._polynomial(value)
        return rational_expression(numerator, denominator)


def expression_node_count(value: Expression, domain: str, dimension: int) -> int:
    if isinstance(value, RationalPolynomial):
        indices = {index: Fraction(1) for index in (*value.numerator, *value.denominator)}
        return 2 * bernstein_node_count(indices, domain, dimension)
    return bernstein_node_count(value, domain, dimension)


def expression_physical_jacobian(
    coordinates: tuple[Expression, ...], kind: str, dimension: int
) -> tuple[tuple[Expression, ...], ...]:
    if kind != "pyramid":
        affine = _affine_polynomial_jacobian(coordinates, dimension)
        if affine is not None:
            return affine
    rows = tuple(
        tuple(expression_derivative(value, axis) for axis in range(dimension))
        for value in coordinates
    )
    if kind != "pyramid":
        return rows
    u, v, w = axes(3)
    collapse = add(constant(1, 3), scale(w, -1))
    reciprocal = RationalPolynomial(constant(1, 3), collapse)
    result = []
    for first, second, height in rows:
        x, y = (
            expression_multiply(first, reciprocal),
            expression_multiply(second, reciprocal),
        )
        z = expression_sum(
            (
                height,
                expression_multiply(add(u, constant(Fraction(-1, 2), 3)), x),
                expression_multiply(add(v, constant(Fraction(-1, 2), 3)), y),
            )
        )
        result.append((x, y, z))
    return tuple(result)


def expression_reference_evaluate(
    value: Expression, point: tuple[Fraction, ...], domain: str
) -> Fraction:
    """Evaluate including a proved continuous rational chart corner.

    The denominator must be homogeneous about the singular corner. Bounded
    rational Bernstein controls exclude unresolved transverse poles; the
    lowest homogeneous numerator must equal one constant times the denominator.
    Higher homogeneous terms then tend to zero on the occupied reference cone.
    """
    if not isinstance(value, RationalPolynomial) or evaluate(value.denominator, point):
        return expression_evaluate(value, point)
    expression_bernstein_coefficients(value, domain, len(point))
    arguments = tuple(
        add(constant(x, len(point)), variable)
        for x, variable in zip(point, axes(len(point)), strict=True)
    )
    numerator, denominator = (
        compose(value.numerator, arguments),
        compose(value.denominator, arguments),
    )
    degrees = {sum(index) for index in denominator}
    if len(degrees) != 1:
        raise RationalEnclosureError(
            "Rational corner limit requires an exact homogeneous denominator."
        )
    degree = next(iter(degrees))
    if any(sum(index) < degree for index in numerator):
        raise RationalEnclosureError("Rational corner limit has a nonremovable pole.")
    leading = {
        index: coefficient
        for index, coefficient in numerator.items()
        if sum(index) == degree
    }
    pivot = next(iter(denominator))
    limit = leading.get(pivot, Fraction(0)) / denominator[pivot]
    if leading != scale(denominator, limit):
        raise RationalEnclosureError(
            "Rational corner image depends on the reference approach."
        )
    return limit


def expression_control_points(
    coordinates: tuple[Expression, ...],
    domain: str,
    dimension: int,
    maximum_nodes: int,
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Bound a rational image by its common-denominator projective control net."""
    denominators: list[Polynomial] = []
    for value in coordinates:
        if (
            isinstance(value, RationalPolynomial)
            and value.denominator not in denominators
        ):
            denominators.append(value.denominator)
    denominator = math_product_polynomials(tuple(denominators), dimension)
    numerators = []
    for value in coordinates:
        if isinstance(value, RationalPolynomial):
            factor = math_product_polynomials(
                tuple(d for d in denominators if d != value.denominator), dimension
            )
            numerators.append(multiply(value.numerator, factor))
        else:
            numerators.append(multiply(value, denominator))
    indices = tuple(
        index for polynomial in (*numerators, denominator) for index in polynomial
    )
    if domain == "simplex":
        degrees = (max((sum(index) for index in indices), default=0),)
    elif domain == "prism":
        degrees = (
            max((sum(index[:2]) for index in indices), default=0),
            max((index[2] for index in indices), default=0),
        )
    else:
        degrees = tuple(
            max((index[axis] for index in indices), default=0)
            for axis in range(dimension)
        )
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
    weights = _expression_control_net(denominator, domain, degrees, dimension)
    controls = tuple(
        _expression_control_net(value, domain, degrees, dimension) for value in numerators
    )
    if max(weights) < 0:
        weights = tuple(-value for value in weights)
        controls = tuple(tuple(-value for value in row) for row in controls)
    if min(weights) < 0 or not any(weights):
        return None
    points = []
    for weight, row in zip(weights, zip(*controls, strict=True), strict=True):
        if weight:
            points.append(tuple(value / weight for value in row))
        elif any(row):
            return None
    return tuple(points)


def _has_cartesian_reference_chain(element: CellGeometryElement, /) -> bool:
    """Distinguish Cartesian pullbacks from full coefficient-action ancestry."""
    from ._cell_geometry import (
        BarycentricCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
    )

    current = element
    while isinstance(
        current,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        current = current.source_element
    return not isinstance(current, BarycentricCellGeometryElement)


def coordinate_reference_chain(
    element: CellGeometryElement,
    *,
    ancestor: CellGeometryElement | None = None,
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
        element, ancestor=ancestor, prove_partition_unity=False
    )


def coordinate_partition_unity_reference_chain(
    element: CellGeometryElement,
    *,
    ancestor: CellGeometryElement | None = None,
) -> tuple[
    FiniteElementSpec
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement,
    tuple[Expression, ...],
]:
    """Recover source references only after proving the complete P1 unity law."""
    return _prepare_coordinate_reference_chain(
        element, ancestor=ancestor, prove_partition_unity=True
    )


def _prepare_coordinate_reference_chain(
    element: CellGeometryElement,
    *,
    ancestor: CellGeometryElement | None,
    prove_partition_unity: bool,
) -> tuple[
    FiniteElementSpec
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement,
    tuple[Expression, ...],
]:
    """Exact physical-root reference map of affine and polynomial pullbacks.

    Every stored binary64 affine coefficient and every authored rational chart
    coefficient participates in exact arithmetic; no chain is collapsed through
    floating matrix products or interpolated target corners.
    """
    from ._cell_geometry import (
        _require_scalar_coordinate_element,
        BarycentricCellGeometryElement,
        LayerColumnCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )

    current = _require_scalar_coordinate_element(
        element, "Exact coordinate reference chain"
    )
    arguments = chart_arguments(current.cell_kind, current.topological_dimension)
    while isinstance(
        current,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        if ancestor is not None and current.element_id == ancestor.element_id:
            if coordinate_source_signature(current) != coordinate_source_signature(
                ancestor
            ):
                raise ValueError(
                    "The declared coordinate ancestor has changed live source semantics."
                )
            return current, arguments
        if isinstance(current, RestrictedCellGeometryElement):
            chart = affine_arguments(
                np.asarray(current.offset), np.asarray(current.matrix)
            )
        else:
            chart = reference_composition_arguments(current)
        if isinstance(current, RationalComposedCellGeometryElement):
            height = arguments[2]
            divisor = expression_add(
                constant(1, current.topological_dimension), expression_scale(height, -1)
            )
            a, b = expression_parts(divisor, current.topological_dimension)
            reciprocal = rational_expression(b, a)
            collapsed = (
                expression_multiply(
                    expression_add(
                        arguments[0], expression_scale(height, Fraction(-1, 2))
                    ),
                    reciprocal,
                ),
                expression_multiply(
                    expression_add(
                        arguments[1], expression_scale(height, Fraction(-1, 2))
                    ),
                    reciprocal,
                ),
                height,
            )
            composition = ExpressionComposition(collapsed)
        else:
            composition = ExpressionComposition(arguments)
        arguments = tuple(composition(value) for value in chart)
        current = current.source_element
    if isinstance(current, BarycentricCellGeometryElement):
        if not prove_partition_unity:
            raise ValueError(
                "Full barycentric coefficient actions do not define a Cartesian source-reference chain."
            )
        root = current
        while isinstance(root, BarycentricCellGeometryElement):
            root = root.source_element
        basis = source_basis(current)
        if basis is None or len(basis) != root.topological_dimension + 1:
            raise ValueError(
                "A source-reference action requires its complete original P1 cardinal basis."
            )
        unity = constant(1, root.topological_dimension)
        if sum_polynomials(basis) != unity or basis[0] != add(
            unity, scale(sum_polynomials(basis[1:]), -1)
        ):
            raise ValueError(
                "Complete P1 source-reference action does not preserve exact partition of unity."
            )
        composition = ExpressionComposition(arguments)
        arguments = tuple(composition(value) for value in basis[1:])
        current = root
    if ancestor is not None:
        if current.element_id != ancestor.element_id or coordinate_source_signature(
            current
        ) != coordinate_source_signature(ancestor):
            raise ValueError(
                "The requested coordinate source is not a current authored ancestor."
            )
        return current, arguments
    if not isinstance(
        current,
        (FiniteElementSpec, SplineCellGeometryElement, LayerColumnCellGeometryElement),
    ):
        raise TypeError(
            "A composed coordinate reference chain requires its canonical scalar source root."
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
) -> tuple[CellGeometryElement, tuple[Polynomial, ...], tuple[Expression, ...]]:
    """Prove a packet-bound full P1 UV map and retain its whole XYZ remainder.

    All source cardinal columns contribute even when their sum is not one.
    Only the original UV triangle frame is inverted; no target XYZ, RNE
    carrier or field samples are fitted. The caller owns root namespace/SCI
    and source-bound admission and must decide the returned whole remainder.
    """
    from ..linalg._small_batched import prepare_exact_small_linear_actions
    from ._cell_geometry import (
        _require_p1_cardinal_source,
        BarycentricCellGeometryElement,
    )

    root = element
    while isinstance(root, BarycentricCellGeometryElement):
        root = root.source_element
    _require_p1_cardinal_source(root)
    if root.cell_kind != "triangle" or coordinate_source_signature(
        root
    ) != coordinate_source_signature(root_element):
        raise ValueError(
            "Full P1 packet must retain its actual original triangular source basis."
        )
    ids, retained = np.asarray(root_corner_ids), np.asarray(retained_corner_ids)
    if (
        ids.dtype.kind not in "iu"
        or retained.dtype.kind not in "iu"
        or ids.shape != (3,)
        or not np.array_equal(ids, retained)
        or len(set(ids.tolist())) != 3
    ):
        raise ValueError("Full P1 packet changes ordered original source corner SCI.")
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
            "Full P1 packet requires its finite original UV and XYZ coefficient banks."
        )
    if not np.array_equal(xyz.view(np.uint64), controls.view(np.uint64)):
        raise ValueError("Full P1 packet changes the original physical coefficient bank.")
    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        raise RuntimeError(
            "Packet source-reference proof requires its original active coordinate allowance."
        )
    uv_source = coordinate_polynomials(element, uv)
    physical = coordinate_expressions(element, controls)
    if uv_source is None or physical is None or len(original_root_expressions) != 3:
        raise ValueError(
            "Full P1 packet lacks complete actual UV/physical source expressions."
        )
    original_uv = tuple(tuple(Fraction(float(value)) for value in row) for row in uv)
    matrix = tuple(
        tuple(original_uv[column + 1][axis] - original_uv[0][axis] for column in range(2))
        for axis in range(2)
    )
    prepared = prepare_exact_small_linear_actions(
        matrix,
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
        coordinate_budget=ledger,
    )
    if prepared.actions is None:
        raise ValueError("Original source UV triangle has deficient exact rank.")
    differences = tuple(
        add(value, constant(-original_uv[0][axis], 2))
        for axis, value in enumerate(uv_source)
    )
    arguments = linear_combinations(differences, prepared.actions)
    composition = ExpressionComposition(arguments)
    restricted = tuple(composition(value) for value in original_root_expressions)
    remainder = tuple(
        expression_add(actual, expression_scale(expected, -1))
        for actual, expected in zip(physical, restricted, strict=True)
    )
    return root, arguments, remainder
