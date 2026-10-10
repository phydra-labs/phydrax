#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact analytic and rational curve and surface carriers of native B-Reps.

Evaluation is pure JAX so that realized geometry stays differentiable at a fixed
epoch. Parameter-domain validation, seam periods, collapsed (pole) isolines,
Bezier extraction and conservative bounding boxes are host-side preparation
data: bounding boxes use exact trigonometric extrema, interval arithmetic and
the convex-hull property of positively weighted rational Bernstein controls,
widened by a small outward rounding margin.
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, comb, floor, isfinite, isqrt, pi
from typing import Iterator, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import (
    bspline_jet_stencil,
    RationalSplineJet,
    TensorBSplineJetPlan,
)
from ..._interpolation._bspline import (
    bezier_refinement,
    exact_bezier_refinement,
    restrict_bernstein_bounds,
)
from ..._strict import StrictModule
from ...linalg._small_batched import prepare_exact_small_linear_actions
from ...typing import ConvertibleToArray


if TYPE_CHECKING:
    from ...discretization._coordinate_enclosure import CoordinateEnclosureBudget


def _coordinate_budget() -> CoordinateEnclosureBudget | None:
    from ...discretization._coordinate_enclosure import _COORDINATE_BUDGET

    return _COORDINATE_BUDGET.get()


def _rational_enclosure_error(message: str, /) -> Exception:
    from ...discretization._coordinate_enclosure import RationalEnclosureError

    return RationalEnclosureError(message)


_TWO_PI = 2.0 * pi
_FRAME_TOLERANCE = 1.0e-10


def _host_vector(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size not in (2, 3) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite two- or three-dimensional vector.")
    return vector


def _host_scalar(value: ConvertibleToArray, name: str, /) -> float:
    scalar = np.asarray(value, dtype=np.float64)
    if scalar.size != 1 or not np.isfinite(scalar.reshape(-1)[0]):
        raise ValueError(f"{name} must be one finite scalar.")
    return float(scalar.reshape(-1)[0])


def _orthonormal_pair(
    first: np.ndarray, second: np.ndarray, name: str, /
) -> tuple[np.ndarray, np.ndarray]:
    if (
        abs(np.linalg.norm(first) - 1.0) > _FRAME_TOLERANCE
        or abs(np.linalg.norm(second) - 1.0) > _FRAME_TOLERANCE
        or abs(float(first @ second)) > _FRAME_TOLERANCE
    ):
        raise ValueError(f"{name} axes must be orthonormal.")
    return first, second


def _check_frame(first: Array, second: Array, axis: Array, /) -> None:
    a, b, c = (np.asarray(vector, dtype=np.float64) for vector in (first, second, axis))
    if any(
        vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in (a, b, c)
    ):
        raise ValueError("Analytic patch frames require finite three-dimensional axes.")
    _orthonormal_pair(a, b, "Analytic patch")
    if (
        abs(np.linalg.norm(c) - 1) > _FRAME_TOLERANCE
        or abs(float(a @ c)) > _FRAME_TOLERANCE
        or abs(float(b @ c)) > _FRAME_TOLERANCE
    ):
        raise ValueError("Analytic patch axis must complete an orthonormal frame.")


def _crosses_c0_knot(knots: Array, degree: int, lower: float, upper: float, /) -> bool:
    values = np.asarray(knots)
    for knot in np.unique(values[degree + 1 : -degree - 1]):
        if lower <= knot <= upper and np.count_nonzero(values == knot) >= degree:
            return True
    return False


def _padded(box: np.ndarray, /) -> np.ndarray:
    """Widen a box outward by a floating-point evaluation margin."""
    margin = 64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(box))))
    return np.stack((box[0] - margin, box[1] + margin))


def _trig_ranges(
    first: float, last: float, /
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Exact ``(cos, sin)`` ranges over ``[first, last]`` (extrema at quarter turns)."""
    if last - first >= _TWO_PI:
        return (-1.0, 1.0), (-1.0, 1.0)
    quarter = 0.5 * pi
    angles = np.asarray(
        [first, last]
        + [
            index * quarter
            for index in range(ceil(first / quarter), floor(last / quarter) + 1)
        ],
        dtype=np.float64,
    )
    cosine, sine = np.clip(np.cos(angles), -1.0, 1.0), np.clip(np.sin(angles), -1.0, 1.0)
    return (
        (float(np.min(cosine)), float(np.max(cosine))),
        (float(np.min(sine)), float(np.max(sine))),
    )


def _product(
    first: tuple[float, float], second: tuple[float, float], /
) -> tuple[float, float]:
    values = [a * b for a in first for b in second]
    return min(values), max(values)


def _interval_box(
    offset: np.ndarray, terms: tuple[tuple[np.ndarray, tuple[float, float]], ...], /
) -> np.ndarray:
    """Box of ``offset + sum(vector * interval)`` by interval arithmetic."""
    lower = offset.astype(np.float64).copy()
    upper = offset.astype(np.float64).copy()
    for vector, (first, last) in terms:
        low, high = vector * first, vector * last
        lower += np.minimum(low, high)
        upper += np.maximum(low, high)
    return _padded(np.stack((lower, upper)))


def _range(first: ConvertibleToArray, last: ConvertibleToArray, /) -> tuple[float, float]:
    first_, last_ = _host_scalar(first, "first"), _host_scalar(last, "last")
    if not first_ < last_:
        raise ValueError("A parameter range must satisfy first < last.")
    return first_, last_


def _parameter_box(parameter_box: ConvertibleToArray, /) -> np.ndarray:
    box = np.asarray(parameter_box, dtype=np.float64)
    if box.shape != (2, 2) or not np.all(np.isfinite(box)):
        raise ValueError("A parameter box must be a finite (2, 2) array.")
    if np.any(box[1] < box[0]):
        raise ValueError("A closed parameter query box requires lower <= upper.")
    return box


def _check_knots(knots: np.ndarray, degree: int, /) -> None:
    """Nondecreasing knots with legal end and interior multiplicities."""
    if not np.all(np.isfinite(knots)) or np.any(np.diff(knots) < 0.0):
        raise ValueError("B-spline knots must be finite and nondecreasing.")
    lower, upper = knots[degree], knots[-degree - 1]
    if not upper > lower:
        raise ValueError("B-spline knots must define a nonempty parameter domain.")
    values, counts = np.unique(knots, return_counts=True)
    interior = (values > lower) & (values < upper)
    if np.any(counts > degree + 1) or np.any(counts[interior] > degree):
        raise ValueError("B-spline knot multiplicities exceed the degree.")


@dataclass(frozen=True, slots=True)
class RationalBezierPiece:
    """Homogeneous Bernstein controls ``(x * w, w)`` of one nonempty spline span.

    Curves hold ``(degree + 1, dimension + 1)`` controls, surfaces
    ``(u_degree + 1, v_degree + 1, 4)``. ``parameter_bounds`` holds one
    ``(lower, upper)`` interval per parameter axis and ``span_index`` the span
    position in the refined knot vectors.
    """

    homogeneous_controls: np.ndarray
    parameter_bounds: tuple[tuple[float, float], ...]
    span_index: tuple[int, ...]
    homogeneous_lower: np.ndarray
    homogeneous_upper: np.ndarray


def _piece(
    controls: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    index: tuple[int, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> RationalBezierPiece:
    arrays = tuple(
        np.ascontiguousarray(value, dtype=np.float64)
        for value in (controls, lower, upper)
    )
    for value in arrays:
        value.setflags(write=False)
    return RationalBezierPiece(arrays[0], bounds, index, arrays[1], arrays[2])


def _homogeneous_bounds(
    points: np.ndarray, weights: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    product = points * weights[..., None]
    nominal = np.concatenate((product, weights[..., None]), -1)
    lower = np.concatenate((np.nextafter(product, -np.inf), weights[..., None]), -1)
    upper = np.concatenate((np.nextafter(product, np.inf), weights[..., None]), -1)
    return nominal, lower, upper


def _piece_hull(
    piece: RationalBezierPiece,
    parameter_lower: tuple[float, ...],
    parameter_upper: tuple[float, ...],
    /,
) -> np.ndarray:
    lower, upper = piece.homogeneous_lower, piece.homogeneous_upper
    for axis, (start, end) in enumerate(piece.parameter_bounds):
        first = max(start, parameter_lower[axis])
        last = min(end, parameter_upper[axis])
        width = Fraction(end) - Fraction(start)
        local_first = max(
            0.0, np.nextafter(float((Fraction(first) - Fraction(start)) / width), -np.inf)
        )
        local_last = min(
            1.0, np.nextafter(float((Fraction(last) - Fraction(start)) / width), np.inf)
        )
        lower, upper = restrict_bernstein_bounds(
            lower, upper, local_first, local_last, axis
        )
    if np.any(lower[..., -1] <= 0.0):
        raise ValueError(
            "Conservative rational bounds require a positive denominator enclosure."
        )
    quotients = np.stack(
        (
            lower[..., :-1] / lower[..., -1, None],
            lower[..., :-1] / upper[..., -1, None],
            upper[..., :-1] / lower[..., -1, None],
            upper[..., :-1] / upper[..., -1, None],
        )
    )
    dimension = lower.shape[-1] - 1
    lo = np.nextafter(np.min(quotients.reshape((-1, dimension)), axis=0), -np.inf)
    hi = np.nextafter(np.max(quotients.reshape((-1, dimension)), axis=0), np.inf)
    return np.stack((lo, hi))


def _rational_piece_jet_bounds(
    piece: RationalBezierPiece,
    first: float,
    last: float,
    /,
    *,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Outward homogeneous Bernstein hull and exact rational quotient jets.

    A positive denominator is proved by its restricted control hull, not by an
    expanded basis sum. Differentiated homogeneous controls enclose exact source
    derivatives. The quotient recurrence retains coefficient construction
    enclosures and directed rounding at every subtraction/product/division.
    Point queries and one-sided closed spline spans are supported.
    """
    if order not in (0, 1, 2) or isinstance(order, bool):
        raise ValueError("Rational curve jets support orders zero, one and two.")
    return _rational_piece_jets(piece, first, last, order)[order]


def _rational_piece_jets(
    piece: RationalBezierPiece, first: float, last: float, order: int, /
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Every rational quotient jet of orders ``0..order`` (at most three)."""
    from .._interval_enclosure import (
        interval_divide,
        interval_multiply,
        interval_subtract,
    )

    if len(piece.parameter_bounds) != 1 or piece.homogeneous_controls.ndim != 2:
        raise ValueError("Rational curve jets require one Bernstein curve span.")
    if order not in (0, 1, 2, 3) or isinstance(order, bool):
        raise ValueError("Rational curve jets support orders zero to three.")
    ((start, end),) = piece.parameter_bounds
    if not isfinite(first) or not isfinite(last) or not start <= first <= last <= end:
        raise ValueError("A rational jet query must stay inside its closed source span.")
    width = Fraction(end) - Fraction(start)
    local_first = max(
        0.0, np.nextafter(float((Fraction(first) - Fraction(start)) / width), -np.inf)
    )
    local_last = min(
        1.0, np.nextafter(float((Fraction(last) - Fraction(start)) / width), np.inf)
    )
    controls = (piece.homogeneous_lower, piece.homogeneous_upper)
    homogeneous = []
    for derivative in range(order + 1):
        restricted = restrict_bernstein_bounds(
            controls[0], controls[1], local_first, local_last, 0
        )
        homogeneous.append((np.min(restricted[0], axis=0), np.max(restricted[1], axis=0)))
        if derivative < order:
            degree = controls[0].shape[0] - 1
            if degree == 0:
                controls = (np.zeros_like(controls[0]), np.zeros_like(controls[1]))
            else:
                exact_scale = Fraction(degree) / width
                scale = float(exact_scale)
                represented = Fraction(scale)
                factor = (
                    np.asarray(
                        scale
                        if represented <= exact_scale
                        else np.nextafter(scale, -np.inf)
                    ),
                    np.asarray(
                        scale
                        if represented >= exact_scale
                        else np.nextafter(scale, np.inf)
                    ),
                )
                difference = interval_subtract(
                    (controls[0][1:], controls[1][1:]),
                    (controls[0][:-1], controls[1][:-1]),
                )
                controls = interval_multiply(difference, factor)
    denominator = (homogeneous[0][0][-1], homogeneous[0][1][-1])
    if denominator[0] <= 0.0:
        raise _rational_enclosure_error(
            "Rational source jets require a positive denominator enclosure."
        )
    jets = [
        interval_divide((homogeneous[0][0][:-1], homogeneous[0][1][:-1]), denominator)
    ]
    # Leibniz on (w x)^(n) = P^(n): x^(n) = (P^(n) - sum_k C(n,k) w^(k) x^(n-k)) / w.
    for derivative in range(1, order + 1):
        remainder = interval_subtract(
            (homogeneous[derivative][0][:-1], homogeneous[derivative][1][:-1]),
            interval_multiply(
                (homogeneous[derivative][0][-1], homogeneous[derivative][1][-1]),
                jets[0],
            ),
        )
        for lower in range(1, derivative):
            weight = (
                homogeneous[derivative - lower][0][-1],
                homogeneous[derivative - lower][1][-1],
            )
            coefficient = float(comb(derivative, lower))
            remainder = interval_subtract(
                remainder,
                interval_multiply(
                    interval_multiply(
                        (np.asarray(coefficient), np.asarray(coefficient)), weight
                    ),
                    jets[lower],
                ),
            )
        jets.append(interval_divide(remainder, denominator))
    return tuple(jets)


class _BernsteinRestrictionBank:
    """Immutable exact-identity restrictions owned by one live source scope."""

    def __init__(self, budget: CoordinateEnclosureBudget | None) -> None:
        self.budget = budget
        self.entries: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.prefixes: dict[tuple, tuple] = {}
        self.storage = None if budget is None else budget.live_storage()
        self.live = None if self.storage is None else self.storage.__enter__()
        self.bytes_upper = 0

    def close(self) -> None:
        self.entries.clear()
        self.prefixes.clear()
        if self.storage is not None:
            self.storage.__exit__(None, None, None)

    def prefix(self, piece: RationalBezierPiece, order: int) -> tuple:
        # Cache authority is actual coefficient bytes, not a source label or
        # rounded box. Preparing the descriptor occurs once per source batch.
        arrays = (
            piece.homogeneous_controls,
            piece.homogeneous_lower,
            piece.homogeneous_upper,
        )
        size = 1024 + sum(value.nbytes for value in arrays)
        if self.budget is not None:
            if self.live is None:
                raise RuntimeError(
                    "A budgeted restriction bank lost its live storage owner."
                )
            self.live.set_bound(
                self.bytes_upper + size,
                work=sum(value.size for value in arrays),
            )
        key = (
            piece.span_index,
            piece.parameter_bounds,
            order,
            tuple((value.dtype.str, value.shape, value.tobytes()) for value in arrays),
        )
        cached = self.prefixes.get(key)
        if cached is None:
            self.prefixes[key] = key
            self.bytes_upper += size
            return key
        del key
        if self.live is not None:
            self.live.set_bound(self.bytes_upper)
        return cached

    def restrict(
        self,
        key: tuple,
        lower: np.ndarray,
        upper: np.ndarray,
        first: float,
        last: float,
        axis: int,
        /,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.budget is not None:
            self.budget.reserve(1, 256)
        cache_prefix = axis == 0
        full_key = (key, axis, np.float64(first).tobytes(), np.float64(last).tobytes())
        if cache_prefix:
            found = self.entries.get(full_key)
            if found is not None:
                return found
        if self.budget is not None:
            # A cache hit consumes only the identity lookup above. Coefficient
            # visits and de Casteljau work are charged exactly once on a miss.
            self.budget.reserve(2 * lower.size)
            # The first-axis prefix survives for exact reuse; the completed
            # two-axis restriction is consumed inside the caller's temporary
            # row scope and must not accumulate across the whole batch.
            self.budget.reserve(
                2 * lower.size * max(0, lower.shape[axis] - 1),
                4096 + 32 * (lower.nbytes + upper.nbytes),
            )
        result = restrict_bernstein_bounds(lower, upper, first, last, axis)
        if not cache_prefix:
            return result
        retained = 1024 + 4 * (lower.nbytes + upper.nbytes)
        if self.live is not None:
            self.live.set_bound(self.bytes_upper + retained)
        for array in result:
            array.setflags(write=False)
        self.entries[full_key] = result
        self.bytes_upper += retained
        return result


_BERNSTEIN_RESTRICTION_BANK: ContextVar[_BernsteinRestrictionBank | None] = ContextVar(
    "source_bernstein_restriction_bank",
    default=None,
)


@contextmanager
def source_bernstein_restriction_scope() -> Iterator[None]:
    """Borrow immutable restrictions only within the current owner lifetime."""
    previous = _BERNSTEIN_RESTRICTION_BANK.get()
    budget = _coordinate_budget()
    if previous is not None and previous.budget is budget:
        yield
        return
    bank = _BernsteinRestrictionBank(budget)
    token = _BERNSTEIN_RESTRICTION_BANK.set(bank)
    try:
        yield
    finally:
        _BERNSTEIN_RESTRICTION_BANK.reset(token)
        bank.close()


def _prepare_tensor_piece_controls(
    piece: RationalBezierPiece,
    order: int,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Prepare differentiated source coefficients once for a bound batch."""
    from .._interval_enclosure import interval_multiply, interval_subtract

    indices: tuple[tuple[int, int], ...] = ((1, 0), (0, 1))
    if order == 2:
        indices += ((2, 0), (1, 1), (0, 2))
    if budget is not None:
        terms = sum(
            max(1, piece.homogeneous_controls.shape[0] - derivative[0])
            * max(1, piece.homogeneous_controls.shape[1] - derivative[1])
            * piece.homogeneous_controls.shape[-1]
            for derivative in indices
        )
        budget.reserve(4 * terms, 4096 + 96 * terms)
    controls: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {
        (0, 0): (piece.homogeneous_lower, piece.homogeneous_upper),
    }
    for derivative in indices:
        axis = 0 if derivative[0] else 1
        parent = (
            derivative[0] - int(axis == 0),
            derivative[1] - int(axis == 1),
        )
        low, high = controls[parent]
        degree = low.shape[axis] - 1
        if degree == 0:
            controls[derivative] = (np.zeros_like(low), np.zeros_like(high))
        else:
            head = [slice(None)] * low.ndim
            tail = head.copy()
            head[axis], tail[axis] = slice(1, None), slice(None, -1)
            difference = interval_subtract(
                (low[tuple(head)], high[tuple(head)]),
                (low[tuple(tail)], high[tuple(tail)]),
            )
            start, end = piece.parameter_bounds[axis]
            exact_scale = Fraction(degree) / (Fraction(end) - Fraction(start))
            scale = float(exact_scale)
            represented = Fraction(scale)
            factor = (
                np.asarray(
                    scale if represented <= exact_scale else np.nextafter(scale, -np.inf)
                ),
                np.asarray(
                    scale if represented >= exact_scale else np.nextafter(scale, np.inf)
                ),
            )
            controls[derivative] = interval_multiply(difference, factor)
    return controls


def _rational_tensor_piece_jet_bounds(
    piece: RationalBezierPiece,
    box: np.ndarray,
    order: int,
    budget: CoordinateEnclosureBudget | None,
    /,
    *,
    controls: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    source_key: tuple,
    retain_first_jet: bool,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
    """Tensor source quotient jets from certified homogeneous control hulls."""
    from .._interval_enclosure import (
        interval_divide,
        interval_multiply,
        interval_subtract,
    )

    if len(piece.parameter_bounds) != 2 or piece.homogeneous_controls.ndim != 3:
        raise ValueError("Tensor source jets require one two-axis Bernstein span.")
    if order not in (1, 2) or isinstance(order, bool):
        raise ValueError("Tensor source jets support orders one and two.")
    indices: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1))
    if order == 2:
        indices += ((2, 0), (1, 1), (0, 2))
    rows = piece.homogeneous_controls.shape[:2]
    columns = piece.homogeneous_controls.shape[-1]
    terms = [
        max(1, rows[0] - derivative[0]) * max(1, rows[1] - derivative[1]) * columns
        for derivative in indices
    ]
    if budget is not None:
        work = 2 * sum(terms)
        work += columns * sum(4 + 10 * (sum(derivative) + 1) for derivative in indices)
        budget.reserve(work, 4096 + 8 * (12 * sum(terms) + 16 * len(indices) * columns))
    local_bounds = []
    clipped_bounds = []
    for axis, (start, end) in enumerate(piece.parameter_bounds):
        first, last = max(start, float(box[0, axis])), min(end, float(box[1, axis]))
        if not isfinite(first) or not isfinite(last) or not start <= first <= last <= end:
            raise ValueError(
                "Tensor source jets must stay inside their closed native span."
            )
        clipped_bounds.append((np.float64(first).tobytes(), np.float64(last).tobytes()))
        width = Fraction(end) - Fraction(start)
        local_bounds.append(
            (
                max(
                    0.0,
                    np.nextafter(
                        float((Fraction(first) - Fraction(start)) / width), -np.inf
                    ),
                ),
                min(
                    1.0,
                    np.nextafter(
                        float((Fraction(last) - Fraction(start)) / width), np.inf
                    ),
                ),
            )
        )
    homogeneous: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for derivative in indices:
        restricted = controls[derivative]
        key = (source_key, derivative)
        bank = _BERNSTEIN_RESTRICTION_BANK.get()
        for axis, (first, last) in enumerate(local_bounds):
            key = (
                key,
                axis,
                clipped_bounds[axis],
                np.float64(first).tobytes(),
                np.float64(last).tobytes(),
            )
            if bank is None:
                if budget is not None:
                    budget.reserve(
                        2 * restricted[0].size * max(0, restricted[0].shape[axis] - 1)
                    )
                restricted = restrict_bernstein_bounds(*restricted, first, last, axis)
            else:
                restricted = bank.restrict(key, *restricted, first, last, axis)
        homogeneous[derivative] = (
            np.min(restricted[0], axis=(0, 1)),
            np.max(restricted[1], axis=(0, 1)),
        )
    denominator = (homogeneous[(0, 0)][0][-1], homogeneous[(0, 0)][1][-1])
    if denominator[0] <= 0.0:
        raise _rational_enclosure_error(
            "Tensor source jets require a positive denominator enclosure."
        )
    jets: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for derivative in indices:
        low, high = homogeneous[derivative]
        remainder = (low[:-1], high[:-1])
        for weight_derivative in indices[1:]:
            if any(
                value > limit
                for value, limit in zip(weight_derivative, derivative, strict=True)
            ):
                continue
            parent = (
                derivative[0] - weight_derivative[0],
                derivative[1] - weight_derivative[1],
            )
            weight_low, weight_high = homogeneous[weight_derivative]
            multiplicity = (
                2 if derivative in ((2, 0), (0, 2)) and sum(weight_derivative) == 1 else 1
            )
            weight = (weight_low[-1], weight_high[-1])
            if multiplicity == 2:
                weight = interval_multiply(weight, (np.asarray(2.0), np.asarray(2.0)))
            remainder = interval_subtract(
                remainder, interval_multiply(weight, jets[parent])
            )
        jets[derivative] = interval_divide(remainder, denominator)
    if order == 1:
        return (
            (
                np.stack((jets[(1, 0)][0], jets[(0, 1)][0]), axis=-1),
                np.stack((jets[(1, 0)][1], jets[(0, 1)][1]), axis=-1),
            ),
            None,
        )
    first_jet = None
    if budget is not None and retain_first_jet:
        budget.reserve(12, 1024 + 12 * np.dtype(np.float64).itemsize)
        first_jet = (
            np.stack((jets[(1, 0)][0], jets[(0, 1)][0]), axis=-1),
            np.stack((jets[(1, 0)][1], jets[(0, 1)][1]), axis=-1),
        )
    return (
        (
            np.stack(
                (
                    np.stack((jets[(2, 0)][0], jets[(1, 1)][0]), axis=-1),
                    np.stack((jets[(1, 1)][0], jets[(0, 2)][0]), axis=-1),
                ),
                axis=-2,
            ),
            np.stack(
                (
                    np.stack((jets[(2, 0)][1], jets[(1, 1)][1]), axis=-1),
                    np.stack((jets[(1, 1)][1], jets[(0, 2)][1]), axis=-1),
                ),
                axis=-2,
            ),
        ),
        first_jet,
    )


# ------------------------------------------------------------------ curves


class AbstractCurve(StrictModule):
    """Exact parametric curve in two (p-curve) or three (edge) dimensions."""

    @property
    @abstractmethod
    def ambient_dimension(self) -> int:
        raise NotImplementedError

    @property
    def period(self) -> float | None:
        """Parameter period of a closed periodic carrier, else ``None``."""
        return None

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        """Finite admissible parameter domain, or ``None`` when unbounded."""
        return None

    @abstractmethod
    def evaluate(self, parameters: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        """Conservative ``(2, dimension)`` box of the curve over ``[first, last]``."""
        raise NotImplementedError

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        """Validate a trimmed parameter range against the carrier domain."""
        first_, last_ = _range(first, last)
        period = self.period
        if period is not None and last_ - first_ > period * (1.0 + 1.0e-12):
            raise ValueError("A periodic curve range cannot exceed one period.")
        domain = self.parameter_domain
        if domain is not None:
            slack = 1.0e-12 * max(1.0, abs(domain[0]), abs(domain[1]))
            if first_ < domain[0] - slack or last_ > domain[1] + slack:
                raise ValueError("A curve range lies outside the carrier domain.")
        return first_, last_

    def validate_query_range(self, first: float, last: float, /) -> tuple[float, float]:
        if first < last:
            return self.validate_range(first, last)
        if first != last or not np.isfinite(first):
            raise ValueError("A closed curve query requires finite lower <= upper.")
        domain = self.parameter_domain
        if domain is not None and not domain[0] <= first <= domain[1]:
            raise ValueError("A point query lies outside the carrier domain.")
        return float(first), float(last)

    def is_c1_on(self, first: float, last: float, /) -> bool:
        self.validate_query_range(first, last)
        return not isinstance(self, BSplineCurve) or not _crosses_c0_knot(
            self.knots, self.degree, first, last
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Source-faithful interval derivative enclosure on a parameter range."""
        if order not in (1, 2) or isinstance(order, bool):
            raise ValueError("Curve derivative bounds support orders one and two.")
        if order == 2 and not self.is_c1_on(first, last):
            return np.full(self.ambient_dimension, -np.inf), np.full(
                self.ambient_dimension, np.inf
            )
        if isinstance(self, BSplineCurve):
            self.validate_query_range(first, last)
            budget = _coordinate_budget()
            cache_key: tuple[str, int, bytes] | None = None
            if budget is not None:
                source_key = canonical_fingerprint(
                    {
                        "kind": "source-bspline-curve-derivative-bounds",
                        "type": f"{type(self).__module__}.{type(self).__qualname__}",
                        "degree": self.degree,
                        "coefficients": array_tree_fingerprint(self),
                    }
                )
                range_key = np.asarray((first, last), dtype=np.float64).tobytes()
                cache_key = source_key, order, range_key
                cached = budget.source_derivative_bounds_cache.get(cache_key)
                if cached is not None:
                    budget.reserve(
                        2 * cached[0].size,
                        256 + cached[0].nbytes + cached[1].nbytes,
                    )
                    return cached[0].copy(), cached[1].copy()
            pieces = (
                self.bezier_pieces()
                if budget is None
                else _prepare_bspline_curve_spans(self, budget)
            )
            jets = []
            for piece in pieces:
                ((start, end),) = piece.parameter_bounds
                lower, upper = max(first, start), min(last, end)
                # A closed knot owns both one-sided jets, including the
                # right-span JVP when the queried arc ends at a C0 knot.
                if lower <= upper:
                    jets.append(
                        _rational_piece_jet_bounds(piece, lower, upper, order=order)
                        if budget is None
                        else _bspline_span_jet_bounds(piece, lower, upper, order, budget)
                    )
            result = (
                np.min([jet[0] for jet in jets], axis=0),
                np.max([jet[1] for jet in jets], axis=0),
            )
            if budget is not None:
                if cache_key is None:
                    raise RuntimeError(
                        "Budgeted curve derivative bounds lost their cache identity."
                    )
                with budget.temporary_scope():
                    budget.reserve(
                        2 * result[0].size,
                        1024 + result[0].nbytes + result[1].nbytes,
                    )
                    cached = result[0].copy(), result[1].copy()
                    cached[0].setflags(write=False)
                    cached[1].setflags(write=False)
                    budget.retain_basis((cache_key, *cached))
                    budget.source_derivative_bounds_cache[cache_key] = cached
            return result
        from .._interval_enclosure import prepare_interval_function
        from ._intersection_curve import coefficient_enclosures, curve_pieces_for_interval

        lower_bounds, upper_bounds = [], []
        for piece in curve_pieces_for_interval(self, first, last):
            derivative = piece.evaluator.evaluate
            for _ in range(order):
                derivative = jax.jacfwd(derivative)
            prepared = prepare_interval_function(
                lambda parameters: derivative(parameters[0]),
                1,
                batch_capacity=1,
                constant_bounds=coefficient_enclosures(piece.evaluator),
            )
            lower, upper = prepared.evaluate(
                np.asarray([[piece.lower]]), np.asarray([[piece.upper]])
            )
            lower_bounds.append(lower[0])
            upper_bounds.append(upper[0])
        return np.min(lower_bounds, axis=0), np.max(upper_bounds, axis=0)


class LineCurve(AbstractCurve):
    """``origin + t * direction``."""

    origin: Array
    direction: Array

    def __init__(self, origin: ConvertibleToArray, direction: ConvertibleToArray) -> None:
        origin_ = _host_vector(origin, "origin")
        direction_ = _host_vector(direction, "direction")
        if origin_.shape != direction_.shape:
            raise ValueError("Line origin and direction dimensions must agree.")
        if not np.linalg.norm(direction_) > 0.0:
            raise ValueError("A line direction must be nonzero.")
        self.origin = jnp.asarray(origin_)
        self.direction = jnp.asarray(direction_)

    @property
    def ambient_dimension(self) -> int:
        return self.origin.shape[0]

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.origin.dtype)
        return self.origin + parameters_[..., None] * self.direction

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        first_, last_ = self.validate_query_range(first, last)
        return _interval_box(
            np.asarray(self.origin), ((np.asarray(self.direction), (first_, last_)),)
        )


class CircleCurve(AbstractCurve):
    """``center + radius * (cos t * first_axis + sin t * second_axis)``."""

    center: Array
    first_axis: Array
    second_axis: Array
    radius: Array

    def __init__(
        self,
        center: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        radius: ConvertibleToArray,
    ) -> None:
        center_ = _host_vector(center, "center")
        first, second = _orthonormal_pair(
            _host_vector(first_axis, "first_axis"),
            _host_vector(second_axis, "second_axis"),
            "Circle",
        )
        radius_ = _host_scalar(radius, "radius")
        if not (center_.shape == first.shape == second.shape):
            raise ValueError("Circle center and axes dimensions must agree.")
        if not radius_ > 0.0:
            raise ValueError("A circle radius must be positive.")
        self.center = jnp.asarray(center_)
        self.first_axis = jnp.asarray(first)
        self.second_axis = jnp.asarray(second)
        self.radius = jnp.asarray(radius_, dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return self.center.shape[0]

    @property
    def period(self) -> float | None:
        return _TWO_PI

    def evaluate(self, parameters: Array, /) -> Array:
        angle = jnp.asarray(parameters, dtype=self.center.dtype)[..., None]
        return self.center + self.radius * (
            jnp.cos(angle) * self.first_axis + jnp.sin(angle) * self.second_axis
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        cosine, sine = _trig_ranges(*self.validate_query_range(first, last))
        radius = float(self.radius)
        return _interval_box(
            np.asarray(self.center),
            (
                (radius * np.asarray(self.first_axis), cosine),
                (radius * np.asarray(self.second_axis), sine),
            ),
        )


class EllipseCurve(AbstractCurve):
    """``center + a cos t * first_axis + b sin t * second_axis``."""

    center: Array
    first_axis: Array
    second_axis: Array
    first_radius: Array
    second_radius: Array

    def __init__(
        self,
        center: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        first_radius: ConvertibleToArray,
        second_radius: ConvertibleToArray,
    ) -> None:
        center_ = _host_vector(center, "center")
        first, second = _orthonormal_pair(
            _host_vector(first_axis, "first_axis"),
            _host_vector(second_axis, "second_axis"),
            "Ellipse",
        )
        radii = (
            _host_scalar(first_radius, "first_radius"),
            _host_scalar(second_radius, "second_radius"),
        )
        if not (center_.shape == first.shape == second.shape):
            raise ValueError("Ellipse center and axes dimensions must agree.")
        if min(radii) <= 0.0:
            raise ValueError("Ellipse radii must be positive.")
        self.center = jnp.asarray(center_)
        self.first_axis = jnp.asarray(first)
        self.second_axis = jnp.asarray(second)
        self.first_radius = jnp.asarray(radii[0], dtype=jnp.float64)
        self.second_radius = jnp.asarray(radii[1], dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return self.center.shape[0]

    @property
    def period(self) -> float | None:
        return _TWO_PI

    def evaluate(self, parameters: Array, /) -> Array:
        angle = jnp.asarray(parameters, dtype=self.center.dtype)[..., None]
        return (
            self.center
            + self.first_radius * jnp.cos(angle) * self.first_axis
            + self.second_radius * jnp.sin(angle) * self.second_axis
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        cosine, sine = _trig_ranges(*self.validate_query_range(first, last))
        return _interval_box(
            np.asarray(self.center),
            (
                (float(self.first_radius) * np.asarray(self.first_axis), cosine),
                (float(self.second_radius) * np.asarray(self.second_axis), sine),
            ),
        )


class ParabolaCurve(AbstractCurve):
    """Exact vertex-frame parabola ``vertex + t²/(4f) x + t y``.

    The parameter and focal length carry the source length unit.
    """

    vertex: Array
    first_axis: Array
    second_axis: Array
    focal_length: Array

    def __init__(
        self,
        vertex: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        focal_length: ConvertibleToArray,
    ) -> None:
        vertex_ = _host_vector(vertex, "vertex")
        first, second = _orthonormal_pair(
            _host_vector(first_axis, "first_axis"),
            _host_vector(second_axis, "second_axis"),
            "Parabola",
        )
        focal = _host_scalar(focal_length, "focal_length")
        if not vertex_.shape == first.shape == second.shape or focal <= 0.0:
            raise ValueError(
                "A parabola requires matching frame dimensions and positive focal length."
            )
        self.vertex = jnp.asarray(vertex_, dtype=jnp.float64)
        self.first_axis = jnp.asarray(first, dtype=jnp.float64)
        self.second_axis = jnp.asarray(second, dtype=jnp.float64)
        self.focal_length = jnp.asarray(focal, dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return self.vertex.shape[0]

    def evaluate(self, parameters: Array, /) -> Array:
        value = jnp.asarray(parameters, dtype=jnp.float64)[..., None]
        return (
            self.vertex
            + value**2 / (4.0 * self.focal_length) * self.first_axis
            + value * self.second_axis
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        first, last = self.validate_query_range(first, last)
        square = (
            0.0 if first <= 0.0 <= last else min(first**2, last**2),
            max(first**2, last**2),
        )
        return _interval_box(
            np.asarray(self.vertex),
            (
                (np.asarray(self.first_axis) / (4.0 * float(self.focal_length)), square),
                (np.asarray(self.second_axis), (first, last)),
            ),
        )


class HyperbolaCurve(AbstractCurve):
    """Exact positive branch ``center + a cosh(t) x + b sinh(t) y``."""

    center: Array
    first_axis: Array
    second_axis: Array
    first_radius: Array
    second_radius: Array

    def __init__(
        self,
        center: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        first_radius: ConvertibleToArray,
        second_radius: ConvertibleToArray,
    ) -> None:
        center_ = _host_vector(center, "center")
        first, second = _orthonormal_pair(
            _host_vector(first_axis, "first_axis"),
            _host_vector(second_axis, "second_axis"),
            "Hyperbola",
        )
        radii = (
            _host_scalar(first_radius, "first_radius"),
            _host_scalar(second_radius, "second_radius"),
        )
        if not center_.shape == first.shape == second.shape or min(radii) <= 0.0:
            raise ValueError(
                "A hyperbola requires matching frame dimensions and positive radii."
            )
        self.center = jnp.asarray(center_, dtype=jnp.float64)
        self.first_axis = jnp.asarray(first, dtype=jnp.float64)
        self.second_axis = jnp.asarray(second, dtype=jnp.float64)
        self.first_radius = jnp.asarray(radii[0], dtype=jnp.float64)
        self.second_radius = jnp.asarray(radii[1], dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return self.center.shape[0]

    def evaluate(self, parameters: Array, /) -> Array:
        value = jnp.asarray(parameters, dtype=jnp.float64)[..., None]
        positive, negative = jnp.exp(value), jnp.exp(-value)
        return (
            self.center
            + 0.5 * self.first_radius * (positive + negative) * self.first_axis
            + 0.5 * self.second_radius * (positive - negative) * self.second_axis
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        first, last = self.validate_query_range(first, last)
        cosine = (
            1.0 if first <= 0.0 <= last else float(min(np.cosh(first), np.cosh(last))),
            float(max(np.cosh(first), np.cosh(last))),
        )
        sine = (float(np.sinh(first)), float(np.sinh(last)))
        box = _interval_box(
            np.asarray(self.center),
            (
                (float(self.first_radius) * np.asarray(self.first_axis), cosine),
                (float(self.second_radius) * np.asarray(self.second_axis), sine),
            ),
        )
        if not np.all(np.isfinite(box)):
            raise ValueError("A hyperbola source range overflows finite coordinates.")
        return box


class OffsetCurve(AbstractCurve):
    """Exact signed right-normal curve offset retaining its basis parameter.

    In three dimensions the normal is ``tangent cross direction``; the
    two-dimensional convention is ``(dy, -dx)``.
    """

    base: AbstractCurve
    distance: Array
    direction: Array | None

    def __init__(
        self,
        base: AbstractCurve,
        distance: ConvertibleToArray,
        direction: ConvertibleToArray | None = None,
    ) -> None:
        if not isinstance(base, AbstractCurve):
            raise TypeError("A curve offset requires a native basis curve.")
        distance_ = _host_scalar(distance, "distance")
        if base.ambient_dimension == 3:
            if direction is None:
                raise ValueError(
                    "A three-dimensional curve offset requires its source direction."
                )
            direction_ = _host_vector(direction, "direction")
            if (
                direction_.shape != (3,)
                or abs(np.linalg.norm(direction_) - 1.0) > _FRAME_TOLERANCE
            ):
                raise ValueError(
                    "A curve offset direction must be a unit three-dimensional vector."
                )
            self.direction = jnp.asarray(direction_, dtype=jnp.float64)
        elif base.ambient_dimension == 2 and direction is None:
            self.direction = None
        else:
            raise ValueError("A planar curve offset has no three-dimensional direction.")
        self.base = base
        self.distance = jnp.asarray(distance_, dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return self.base.ambient_dimension

    @property
    def period(self) -> float | None:
        return self.base.period

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.base.parameter_domain

    def _evaluate_one(self, parameter: Array) -> Array:
        value, tangent = jax.jvp(
            self.base.evaluate, (parameter,), (jnp.ones_like(parameter),)
        )
        normal = (
            jnp.stack((tangent[1], -tangent[0]))
            if self.direction is None
            else jnp.cross(tangent, self.direction)
        )
        length = jnp.linalg.norm(normal)
        return value + self.distance * normal / length

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        if values.ndim == 0:
            return self._evaluate_one(values)
        return jax.vmap(self._evaluate_one)(values.reshape(-1)).reshape(
            (*values.shape, self.ambient_dimension)
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        first, last = self.validate_query_range(first, last)
        box = self.base.bounding_box(first, last)
        distance = abs(float(self.distance))
        return np.stack(
            (
                np.nextafter(box[0] - distance, -np.inf),
                np.nextafter(box[1] + distance, np.inf),
            )
        )

    def is_c1_on(self, first: float, last: float, /) -> bool:
        if not self.base.is_c1_on(first, last):
            return False
        if isinstance(self.base, BSplineCurve):
            knots = np.asarray(self.base.knots)
            domain = self.base.parameter_domain
            return not any(
                first <= knot <= last
                and domain[0] < knot < domain[1]
                and np.count_nonzero(knots == knot) >= self.base.degree - 1
                for knot in np.unique(knots)
            )
        return True


def _rational_coordinate_value(
    plan: TensorBSplineJetPlan,
    weights: Array,
    controls: Array,
    /,
) -> Array:
    """Apply the canonical local rational basis as an affine coordinate action.

    Its source partition of unity permits subtracting the locally dominant
    control and adding it back after contraction. A one-hot endpoint basis then
    reproduces either authored boundary control exactly, while interior
    evaluation minimizes cancellation without changing the source controls,
    weights, knots or address.
    """
    rational = RationalSplineJet(plan, weights)
    local = plan.gather(controls)
    anchor = local[jnp.argmax(jnp.abs(rational.values))]
    return anchor + rational.values @ (local - anchor)


class BSplineCurve(AbstractCurve):
    """Rational B-spline with an expanded knot vector on its finite active domain."""

    control_points: Array
    weights: Array
    knots: Array
    degree: int = eqx.field(static=True)

    def __init__(
        self,
        control_points: ConvertibleToArray,
        weights: ConvertibleToArray,
        knots: ConvertibleToArray,
        degree: int,
    ) -> None:
        control_points_ = jnp.asarray(control_points, dtype=jnp.float64)
        weights_ = jnp.asarray(weights, dtype=jnp.float64).reshape((-1,))
        knots_ = jnp.asarray(knots, dtype=jnp.float64).reshape((-1,))
        if control_points_.ndim != 2 or control_points_.shape[0] != weights_.shape[0]:
            raise ValueError("Curve control points and weights are inconsistent.")
        if int(degree) < 1:
            raise ValueError("B-spline curve degree must be positive.")
        if control_points_.shape[0] <= int(degree):
            raise ValueError("B-spline curve control-point count must exceed its degree.")
        if knots_.shape[0] != control_points_.shape[0] + int(degree) + 1:
            raise ValueError("Curve knot vector length is inconsistent.")
        _check_knots(np.asarray(knots_), int(degree))
        self.control_points = control_points_
        self.weights = weights_
        self.knots = knots_
        self.degree = int(degree)

    @classmethod
    def bezier(
        cls, control_points: ConvertibleToArray, weights: ConvertibleToArray
    ) -> BSplineCurve:
        """Rational Bezier curve on ``[0, 1]`` (one clamped span)."""
        count = np.asarray(control_points).shape[0]
        degree = count - 1
        knots = np.concatenate((np.zeros(degree + 1), np.ones(degree + 1)))
        return cls(control_points, weights, knots, degree)

    @property
    def ambient_dimension(self) -> int:
        return self.control_points.shape[1]

    @property
    def parameter_domain(self) -> tuple[float, float]:
        knots = np.asarray(self.knots)
        return float(knots[self.degree]), float(knots[-self.degree - 1])

    def _evaluate_one(self, parameter: Array, span: int | None = None) -> Array:
        # A pinned span evaluates that span's polynomial (Bernstein-piece
        # semantics, one-sided at C0 knots); otherwise the span is searched.
        stencil = (
            bspline_jet_stencil(
                self.knots, parameter, degree=self.degree, maximum_order=0
            )
            if span is None
            else bspline_jet_stencil(
                self.knots,
                parameter,
                degree=self.degree,
                maximum_order=0,
                spans=jnp.asarray(span, dtype=jnp.int32),
                bounds="extrapolate",
            )
        )
        plan = TensorBSplineJetPlan((stencil,), maximum_order=0)
        return _rational_coordinate_value(plan, self.weights, self.control_points)

    def _evaluate(self, parameters: Array, span: int | None, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.control_points.dtype)
        if parameters_.ndim == 0:
            return self._evaluate_one(parameters_, span)
        return jax.vmap(lambda value: self._evaluate_one(value, span))(
            parameters_.reshape((-1,))
        ).reshape((*parameters_.shape, self.control_points.shape[1]))

    def evaluate(self, parameters: Array, /) -> Array:
        return self._evaluate(parameters, None)

    def bezier_pieces(self) -> tuple[RationalBezierPiece, ...]:
        """Numerical Bernstein pieces with source-faithful coefficient enclosures."""
        points = np.asarray(self.control_points, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        homogeneous, homogeneous_lower, homogeneous_upper = _homogeneous_bounds(
            points, weights
        )
        pieces, bounds = bezier_refinement(
            homogeneous, np.asarray(self.knots), self.degree, 0
        )
        piece_lower, piece_upper, _ = bezier_refinement(
            homogeneous_lower,
            np.asarray(self.knots),
            self.degree,
            0,
            control_upper=homogeneous_upper,
        )
        return tuple(
            _piece(piece, ((float(lower), float(upper)),), (index,), lo, hi)
            for index, (piece, lo, hi, (lower, upper)) in enumerate(
                zip(pieces, piece_lower, piece_upper, bounds, strict=True)
            )
        )

    def exact_bezier_pieces(self) -> tuple[RationalBezierPiece, ...]:
        """Host Fraction coefficient pieces of the exact binary-rational source."""
        points, weights = np.asarray(self.control_points), np.asarray(self.weights)
        homogeneous = np.empty((points.shape[0], points.shape[1] + 1), dtype=object)
        for index in range(points.shape[0]):
            weight = Fraction(float(weights[index]))
            for coordinate in range(points.shape[1]):
                homogeneous[index, coordinate] = (
                    Fraction(float(points[index, coordinate])) * weight
                )
            homogeneous[index, -1] = weight
        pieces, bounds = exact_bezier_refinement(
            homogeneous, np.asarray(self.knots), self.degree, 0
        )
        result = []
        for index, (controls, (lower, upper)) in enumerate(
            zip(pieces, bounds, strict=True)
        ):
            controls.setflags(write=False)
            result.append(
                RationalBezierPiece(
                    controls,
                    ((float(lower), float(upper)),),
                    (index,),
                    controls,
                    controls,
                )
            )
        return tuple(result)

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        first_, last_ = self.validate_query_range(first, last)
        budget = _coordinate_budget()
        pieces = (
            self.bezier_pieces()
            if budget is None
            else _prepare_bspline_curve_spans(self, budget)
        )
        if budget is not None:
            # Admit every selected restriction and the output reduction before
            # allocating; preparation itself retains the source bank.
            selected_count = 0
            for piece in pieces:
                if (
                    piece.parameter_bounds[0][1] < first_
                    or piece.parameter_bounds[0][0] > last_
                ):
                    continue
                selected_count += 1
                rows, columns = piece.homogeneous_controls.shape
                budget.reserve(
                    2 * rows * max(1, rows - 1) * columns + 8 * rows * columns,
                    4096 + 8 * (32 * rows * columns + 8 * columns),
                )
            budget.reserve(
                selected_count * self.ambient_dimension * 4,
                128 + 64 * selected_count * self.ambient_dimension,
            )
        selected = [
            _piece_hull(piece, (first_,), (last_,))
            for piece in pieces
            if piece.parameter_bounds[0][1] >= first_
            and piece.parameter_bounds[0][0] <= last_
        ]
        boxes = np.stack(selected)
        return np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))


def _retain_bspline_spans(
    key: str,
    pieces: tuple[RationalBezierPiece, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[RationalBezierPiece, ...]:
    """Retain actual source-bank allocations on the canonical span ledger."""
    retained: list[object] = []
    for piece in pieces:
        for value in (piece, piece.parameter_bounds, piece.span_index):
            budget.reserve(1, 32)
            retained.append(value)
        for array in (
            piece.homogeneous_controls,
            piece.homogeneous_lower,
            piece.homogeneous_upper,
        ):
            owner: np.ndarray | None = array
            while owner is not None:
                budget.reserve(1, 32)
                retained.append(owner)
                owner = owner.base if isinstance(owner.base, np.ndarray) else None
    budget.reserve(0, 64 + 8 * len(retained))
    budget.reserve(0, 256)
    budget.retain_basis((key, pieces, tuple(retained)))
    budget.bspline_span_cache[key] = pieces
    return pieces


def _prepare_bspline_curve_spans(
    curve: BSplineCurve,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[RationalBezierPiece, ...]:
    """Prepare source Bernstein spans on the original coefficient ledger.

    Extraction scratch is admitted before host transfer or native refinement.
    The returned pieces and their actual NumPy backing allocations are retained
    on this ledger; a view never stands in for its uncounted backing array.
    """
    if not isinstance(curve, BSplineCurve):
        raise TypeError("Source span preparation requires a BSplineCurve.")
    count, dimension = curve.control_points.shape
    degree = curve.degree
    knot_count = curve.knots.shape[0]
    columns = dimension + 1
    with budget.temporary_scope():
        # Cache authority includes every numerical source leaf and its dtype,
        # shape and path, plus the native static degree. Knot/range changes
        # therefore cannot reuse an ancestor source's span bank.
        source_terms = curve.control_points.size + curve.weights.size + knot_count
        budget.reserve(source_terms, 8192 + 24 * source_terms)
        key = canonical_fingerprint(
            {
                "kind": "source-bspline-spans",
                "type": f"{type(curve).__module__}.{type(curve).__qualname__}",
                "degree": degree,
                "coefficients": array_tree_fingerprint(curve),
            }
        )
        cached = budget.bspline_span_cache.get(key)
        if cached is not None:
            return cached
        # Source transfer and multiplicity indexing precede extraction sizing.
        budget.reserve(
            knot_count,
            1024 + 8 * (count * dimension + count + 5 * knot_count),
        )
        knots = np.asarray(curve.knots, dtype=np.float64)
        lower, upper = knots[degree], knots[-degree - 1]
        values, multiplicities = np.unique(knots, return_counts=True)
        active = (values >= lower) & (values <= upper)
        active_values, active_counts = values[active], multiplicities[active]
        spans = active_values.size - 1
        endpoint_insertions = sum(
            max(0, degree + 1 - int(multiplicity))
            for multiplicity in (active_counts[0], active_counts[-1])
        )
        interior_insertions = sum(
            max(0, degree - int(multiplicity)) for multiplicity in active_counts[1:-1]
        )
        insertions = endpoint_insertions + interior_insertions
        refined_count = count + insertions
        piece_terms = spans * (degree + 1) * columns
        # Three homogeneous streams (nominal/lower/upper), native Boehm
        # destination coefficients, and the materialized Bernstein streams.
        coefficient_work = count * (dimension + 3 * columns) + 3 * piece_terms
        current_count, current_knots = count, knot_count
        for _ in range(endpoint_insertions):
            coefficient_work += 3 * (current_count + 1) * columns + 2 * (
                current_knots + 1
            )
            current_count += 1
            current_knots += 1
        current_count = spans * degree + 1 - interior_insertions
        current_knots = current_count + degree + 1
        for _ in range(interior_insertions):
            coefficient_work += 3 * (current_count + 1) * columns + 2 * (
                current_knots + 1
            )
            current_count += 1
            current_knots += 1
        # Both old/new refinement streams coexist with the nominal output.
        # Boehm interpolation has eight product lanes per homogeneous row.
        scratch_terms = (
            count * (dimension + 3 * columns)
            + 6 * refined_count * columns
            + 3 * piece_terms
            + 4 * (knot_count + insertions)
            + 4 * spans
            + 32 * columns
        )
        budget.reserve(coefficient_work, 2048 + 8 * scratch_terms + 1024 * spans)
        pieces = curve.bezier_pieces()
        return _retain_bspline_spans(key, pieces, budget)


def _bspline_span_jet_bounds(
    piece: RationalBezierPiece,
    first: float,
    last: float,
    order: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound one closed native source span, including either one-sided endpoint.

    The caller owns the temporary scope containing the returned arrays. Public
    whole-curve C0 derivative semantics are deliberately not changed.
    """
    if order not in (0, 1, 2) or isinstance(order, bool):
        raise ValueError("Source span jets support orders zero, one and two.")
    return _bspline_span_jets(piece, first, last, order, budget)[order]


def _bspline_span_jets(
    piece: RationalBezierPiece,
    first: float,
    last: float,
    order: int,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Every closed-span jet of orders ``0..order`` on the source ledger."""
    if len(piece.parameter_bounds) != 1 or piece.homogeneous_controls.ndim != 2:
        raise ValueError("Source span jets require one Bernstein curve span.")
    if order not in (0, 1, 2, 3) or isinstance(order, bool):
        raise ValueError("Source span jets support orders zero to three.")
    ((start, end),) = piece.parameter_bounds
    if not isfinite(first) or not isfinite(last) or not start <= first <= last <= end:
        raise ValueError("Source span jets must remain inside their closed native span.")
    rows, columns = piece.homogeneous_controls.shape
    # Each derivative restricts both coefficient streams by at most two
    # complete de Casteljau triangles, then visits their homogeneous hulls.
    coefficient_work = sum(
        2 * max(1, rows - derivative) * columns
        + 2 * max(1, rows - derivative) * max(0, rows - derivative - 1) * columns
        for derivative in range(order + 1)
    )
    coefficient_work += order * 2 * rows * columns + (order + 1) * columns
    # Restriction products, two split output streams, derivative controls and
    # quotient recurrence arrays coexist. Scalar Fraction chart normalization
    # uses bounded binary64 numerator/denominator metadata, not sampled values.
    if budget is not None:
        budget.reserve(
            coefficient_work,
            4096 + 8 * (32 * rows * columns + 8 * (order + 1) * columns),
        )
    return _rational_piece_jets(piece, first, last, order)


class BSplineSpanCurve(AbstractCurve):
    """One knot span of a live rational B-spline curve: its Bernstein-piece map.

    ``span`` is immutable host addressing of an active knot span. Control
    points, weights and knots remain the live leaves of ``source``; values and
    every derivative (including knot tangents) come from the canonical span
    evaluation, extended polynomially exactly like the span's Bernstein piece.
    """

    source: BSplineCurve
    span: int = eqx.field(static=True)

    def __init__(self, source: BSplineCurve, span: int) -> None:
        if not isinstance(source, BSplineCurve):
            raise TypeError("A pinned span requires a BSplineCurve source.")
        if isinstance(span, bool) or not isinstance(span, int):
            raise TypeError("A pinned knot span must be an integer.")
        if not source.degree <= span < source.control_points.shape[0]:
            raise ValueError("A pinned knot span must address an active span.")
        self.source = source
        self.span = span

    @property
    def ambient_dimension(self) -> int:
        return self.source.ambient_dimension

    @property
    def parameter_domain(self) -> tuple[float, float]:
        knots = np.asarray(self.source.knots)
        return float(knots[self.span]), float(knots[self.span + 1])

    def evaluate(self, parameters: Array, /) -> Array:
        return self.source._evaluate(parameters, self.span)

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        lower, upper = self.parameter_domain
        first_, last_ = self.validate_query_range(first, last)
        if first_ < lower or last_ > upper:
            raise ValueError("A pinned span box must lie inside its knot span.")
        return self.source.bounding_box(first_, last_)


# ---------------------------------------------------------------- surfaces


class AbstractSurfacePatch(StrictModule):
    """Pure-JAX parametric surface patch in native CAD coordinates."""

    @abstractmethod
    def evaluate(self, parameters: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        """Conservative ``(2, 3)`` box of the patch over a ``(2, 2)`` parameter box."""
        raise NotImplementedError

    @property
    def periods(self) -> tuple[float | None, float | None]:
        """Parameter periods of the ``u`` and ``v`` directions (``None``: open)."""
        return None, None

    def degenerate_isolines(
        self, parameter_box: ConvertibleToArray, /
    ) -> tuple[tuple[int, float], ...]:
        """``(axis, value)`` isolines ``parameters[axis] == value`` collapsing to a point.

        The collapsed isoline (a pole or apex) is where the chart is singular;
        only isolines inside the closed parameter box are reported.
        """
        del parameter_box
        return ()

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        """Validate a face parameter box against periods and carrier domains."""
        box = _parameter_box(parameter_box)
        for axis, period in enumerate(self.periods):
            if period is not None and box[1, axis] - box[0, axis] > period * (
                1.0 + 1.0e-12
            ):
                raise ValueError("A periodic parameter extent cannot exceed its period.")
        return box

    def is_c1_on(self, parameter_box: ConvertibleToArray, /) -> bool:
        box = self.validate_parameter_box(parameter_box)
        if isinstance(self, BSplineSurfacePatch):
            return not (
                _crosses_c0_knot(self.u_knots, self.u_degree, box[0, 0], box[1, 0])
                or _crosses_c0_knot(self.v_knots, self.v_degree, box[0, 1], box[1, 1])
            )
        if isinstance(self, ExtrusionSurface):
            return self.curve.is_c1_on(box[0, 0], box[1, 0])
        if isinstance(self, RevolutionSurface):
            return self.curve.is_c1_on(box[0, 1], box[1, 1])
        if isinstance(self, RuledSurface):
            return self.first.is_c1_on(box[0, 0], box[1, 0]) and self.second.is_c1_on(
                box[0, 0], box[1, 0]
            )
        return True

    def derivative_bounds(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        order: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Source interval jets, shaped (3,2) or (3,2,2), not sampled bounds.

        Finite derivative bounds do not prove chart regularity. Pole/apex
        metadata and a positive bound on the tangent cross product remain
        required for normal-turn or inverse-chart certificates.
        """
        lower, upper = self.derivative_bounds_batch(
            np.asarray(parameter_box, dtype=np.float64)[None], order=order
        )
        return lower[0], upper[0]

    @source_bernstein_restriction_scope()
    def derivative_bounds_batch(
        self,
        parameter_boxes: ConvertibleToArray,
        /,
        *,
        order: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Source interval jets of ``(B, 2, 2)`` boxes, shaped (B,3,2) or (B,3,2,2).

        Every box is enclosed independently by its own source pieces. Tensor
        splines use retained homogeneous Bernstein quotient jets; other pieces
        with identical source coefficients share one interval program per call.
        """
        if order not in (1, 2) or isinstance(order, bool):
            raise ValueError("Surface derivative bounds support orders one and two.")
        boxes = np.asarray(parameter_boxes, dtype=np.float64)
        if boxes.ndim != 3 or boxes.shape[1:] != (2, 2):
            raise ValueError(
                "Surface derivative boxes must have shape (num_boxes, 2, 2)."
            )
        if isinstance(self, BSplineSurfacePatch):
            budget = _coordinate_budget()
            shape = (boxes.shape[0], 3, 2) + ((2,) if order == 2 else ())
            size = boxes.shape[0] * 3 * 2**order
            if budget is not None:
                budget.reserve(2 * size, 128 + 16 * size)
            lower, upper = np.full(shape, np.inf), np.full(shape, -np.inf)
            if not boxes.shape[0]:
                return lower, upper
            if not np.all(np.isfinite(boxes)):
                raise ValueError("A parameter box must be a finite (2, 2) array.")
            if np.any(boxes[:, 1] < boxes[:, 0]):
                raise ValueError("A closed parameter query box requires lower <= upper.")
            c0 = np.zeros((boxes.shape[0],), dtype=np.bool_)
            for axis, (source_knots, degree) in enumerate(
                ((self.u_knots, self.u_degree), (self.v_knots, self.v_degree))
            ):
                if budget is not None:
                    budget.reserve(
                        source_knots.size + 4 * boxes.shape[0],
                        1024 + 48 * source_knots.size + 8 * boxes.shape[0],
                    )
                knots = np.asarray(source_knots)
                first, last = float(knots[degree]), float(knots[-degree - 1])
                slack = 1.0e-12 * max(1.0, abs(first), abs(last))
                if np.any(boxes[:, 0, axis] < first - slack) or np.any(
                    boxes[:, 1, axis] > last + slack
                ):
                    raise ValueError("A parameter box lies outside the B-spline domain.")
                if order == 2:
                    for knot in np.unique(knots[degree + 1 : -degree - 1]):
                        if np.count_nonzero(knots == knot) >= degree:
                            if budget is not None:
                                budget.reserve(3 * boxes.shape[0])
                            c0 |= (boxes[:, 0, axis] <= knot) & (
                                knot <= boxes[:, 1, axis]
                            )
            pieces: tuple[RationalBezierPiece, ...] | None = None
            piece_controls: dict[
                tuple[int, ...],
                dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
            ] = {}
            with budget.temporary_scope() if budget is not None else nullcontext():
                if budget is not None:
                    source_terms = (
                        self.control_points.size
                        + self.weights.size
                        + self.u_knots.size
                        + self.v_knots.size
                    )
                    budget.reserve(source_terms, 8192 + 24 * source_terms)
                source_cache_key = canonical_fingerprint(
                    {
                        "kind": "source-bspline-surface-derivative-bounds",
                        "type": f"{type(self).__module__}.{type(self).__qualname__}",
                        "degrees": (self.u_degree, self.v_degree),
                        "coefficients": array_tree_fingerprint(self),
                    }
                )
                # The batch owns one immutable source and derivative order.
                # Repeated exact binary boxes therefore have identical jets;
                # retain only row references into the already admitted output.
                prepared_rows: dict[bytes, int] = {}
                try:
                    for row in range(boxes.shape[0]):
                        box = np.empty((2, 2), dtype=np.float64)
                        box[...] = boxes[row]
                        if budget is not None:
                            budget.reserve(box.size, 256)
                        box_key = box.tobytes()
                        prepared_row = prepared_rows.get(box_key)
                        if prepared_row is not None:
                            if budget is not None:
                                budget.reserve(2 * lower[row].size)
                            lower[row] = lower[prepared_row]
                            upper[row] = upper[prepared_row]
                            continue
                        prepared_rows[box_key] = row
                        cache_key = source_cache_key, order, box_key
                        cached = (
                            None
                            if budget is None
                            else budget.source_derivative_bounds_cache.get(cache_key)
                        )
                        if cached is not None:
                            if budget is not None:
                                budget.reserve(2 * lower[row].size)
                            lower[row], upper[row] = cached
                            continue
                        if c0[row]:
                            lower[row], upper[row] = -np.inf, np.inf
                            continue
                        if pieces is None:
                            pieces = (
                                self.bezier_pieces()
                                if budget is None
                                else _prepare_bspline_surface_spans(self, budget)
                            )
                            bank = _BERNSTEIN_RESTRICTION_BANK.get()
                            if bank is None:
                                raise RuntimeError(
                                    "Spline bounds lost their restriction bank."
                                )
                            source_keys = {
                                piece.span_index: bank.prefix(piece, order)
                                for piece in pieces
                            }
                            piece_controls = {
                                piece.span_index: _prepare_tensor_piece_controls(
                                    piece, order, budget
                                )
                                for piece in pieces
                            }
                        # Batch output owns copies; cache storage has its own
                        # live owner and survives only its enclosing scope.
                        first_jet_bounds = None
                        if (
                            budget is not None
                            and order == 2
                            and (source_cache_key, 1, box_key)
                            not in budget.source_derivative_bounds_cache
                        ):
                            budget.reserve(12, 1024 + 12 * np.dtype(np.float64).itemsize)
                            first_jet_bounds = (
                                np.full((3, 2), np.inf),
                                np.full((3, 2), -np.inf),
                            )
                        with (
                            budget.temporary_scope()
                            if budget is not None
                            else nullcontext()
                        ):
                            for piece in pieces:
                                if any(
                                    piece.parameter_bounds[axis][1] < box[0, axis]
                                    or piece.parameter_bounds[axis][0] > box[1, axis]
                                    for axis in (0, 1)
                                ):
                                    continue
                                (low, high), first_jet = (
                                    _rational_tensor_piece_jet_bounds(
                                        piece,
                                        box,
                                        order,
                                        budget,
                                        controls=piece_controls[piece.span_index],
                                        source_key=source_keys[piece.span_index],
                                        retain_first_jet=first_jet_bounds is not None,
                                    )
                                )
                                if budget is not None:
                                    budget.reserve(2 * low.size)
                                np.minimum(lower[row], low, out=lower[row])
                                np.maximum(upper[row], high, out=upper[row])
                                if first_jet_bounds is not None and first_jet is not None:
                                    if budget is not None:
                                        budget.reserve(12)
                                    np.minimum(
                                        first_jet_bounds[0],
                                        first_jet[0],
                                        out=first_jet_bounds[0],
                                    )
                                    np.maximum(
                                        first_jet_bounds[1],
                                        first_jet[1],
                                        out=first_jet_bounds[1],
                                    )
                                del low, high
                        if budget is not None:
                            with budget.temporary_scope():
                                budget.reserve(
                                    2 * lower[row].size,
                                    1024 + lower[row].nbytes + upper[row].nbytes,
                                )
                                cached_lower, cached_upper = (
                                    lower[row].copy(),
                                    upper[row].copy(),
                                )
                                cached_lower.setflags(write=False)
                                cached_upper.setflags(write=False)
                                budget.retain_basis(
                                    (cache_key, cached_lower, cached_upper)
                                )
                                budget.source_derivative_bounds_cache[cache_key] = (
                                    cached_lower,
                                    cached_upper,
                                )
                            if first_jet_bounds is not None:
                                first_key = source_cache_key, 1, box_key
                                first_jet_bounds[0].setflags(write=False)
                                first_jet_bounds[1].setflags(write=False)
                                budget.retain_basis((first_key, *first_jet_bounds))
                                budget.source_derivative_bounds_cache[first_key] = (
                                    first_jet_bounds
                                )
                finally:
                    # Derivative coefficients cease to be a batch owner here;
                    # any cached view backing is retained by the live bank.
                    piece_controls.clear()
                    prepared_rows.clear()
            return lower, upper
        if isinstance(self, ExtrusionSurface):
            # The native swept source is affine in v. Its profile already owns
            # authoritative rational quotient jets; expanding that quotient
            # again through generic interval AD loses denominator dependency,
            # even on a fixed-u extrusion ruling.
            shape = (boxes.shape[0], 3, 2) + ((2,) if order == 2 else ())
            lower, upper = np.zeros(shape), np.zeros(shape)
            for row in range(boxes.shape[0]):
                box = boxes[row : row + 1].reshape((2, 2))
                self.validate_parameter_box(box)
                profile_low, profile_high = self.curve.derivative_bounds(
                    float(box[0, 0]),
                    float(box[1, 0]),
                    order=order,
                )
                if order == 1:
                    lower[row, :, 0], upper[row, :, 0] = profile_low, profile_high
                    lower[row, :, 1] = upper[row, :, 1] = np.asarray(self.direction)
                else:
                    lower[row, :, 0, 0], upper[row, :, 0, 0] = profile_low, profile_high
            return lower, upper
        if isinstance(self, RevolutionSurface):
            # ``o + R(u)(c(v) - o)``: rotate the profile's authoritative jets.
            # Generic interval AD of the composed rotation and rational
            # quotient loses every dependency between them and does not even
            # separate the normal cone of a small interior box.
            spans = _meridian_spans(self.curve, _coordinate_budget())
            if spans is not None:
                return _revolution_jet_bounds(self, spans, boxes, order)
            origin = np.asarray(self.axis_origin, dtype=np.float64)
            hull = [
                (np.zeros((boxes.shape[0], 3)), np.zeros((boxes.shape[0], 3)))
                for _ in range(order + 1)
            ]
            rough = np.zeros((boxes.shape[0],), dtype=np.bool_)
            for row in range(boxes.shape[0]):
                box = boxes[row : row + 1].reshape((2, 2))
                self.validate_parameter_box(box)
                if order == 2 and not self.is_c1_on(box):
                    rough[row] = True
                    continue
                first, last = float(box[0, 1]), float(box[1, 1])
                profile = self.curve.bounding_box(
                    *_curve_box_range(self.curve, first, last)
                )
                hull[0][0][row] = np.nextafter(profile[0] - origin, -np.inf)
                hull[0][1][row] = np.nextafter(profile[1] - origin, np.inf)
                for derivative in range(1, order + 1):
                    hull[derivative][0][row], hull[derivative][1][row] = (
                        self.curve.derivative_bounds(first, last, order=derivative)
                    )
            lower, upper = _rotated_jets(
                np.asarray(self.axis_direction, dtype=np.float64), boxes, hull, order
            )
            lower[rough], upper[rough] = -np.inf, np.inf
            return lower, upper
        if isinstance(self, OffsetSurface) and isinstance(self.base, RevolutionSurface):
            # An offset of a planar-meridian revolution is the revolution of
            # the exact parallel meridian; generic interval AD of its
            # normalized normal loses all dependency and does not converge.
            frame = _meridian_frame(self.base, _coordinate_budget())
            if frame is not None:
                return _offset_meridian_jet_bounds(self, frame, boxes, order)
        from .._interval_enclosure import prepare_interval_function
        from ._intersection_curve import (
            coefficient_enclosures,
            surface_pieces_for_box,
            SurfaceEvaluator,
        )

        shape = (boxes.shape[0], 3, 2) + ((2,) if order == 2 else ())
        lower = np.full(shape, np.inf)
        upper = np.full(shape, -np.inf)
        groups: dict[
            tuple[object, ...],
            tuple[SurfaceEvaluator, list[tuple[int, np.ndarray, np.ndarray]]],
        ] = {}
        for row, box in enumerate(boxes):
            if order == 2 and not self.is_c1_on(box):
                lower[row], upper[row] = -np.inf, np.inf
                continue
            pieces = surface_pieces_for_box(self, box)
            if not pieces:
                raise ValueError("A surface derivative box selects no source piece.")
            for piece in pieces:
                leaves, structure = jax.tree_util.tree_flatten(piece.evaluator)
                key = (
                    structure,
                    *(
                        (value.dtype.str, value.shape, value.tobytes())
                        for value in (np.asarray(leaf) for leaf in leaves)
                    ),
                )
                groups.setdefault(key, (piece.evaluator, []))[1].append(
                    (row, piece.lower, piece.upper)
                )
        for evaluator, members in groups.values():
            derivative = evaluator.evaluate
            for _ in range(order):
                derivative = jax.jacfwd(derivative)
            prepared = prepare_interval_function(
                derivative,
                2,
                batch_capacity=min(len(members), 1024),
                constant_bounds=coefficient_enclosures(evaluator),
            )
            rows = np.asarray([row for row, _, _ in members], dtype=np.int64)
            piece_lower, piece_upper = prepared.evaluate(
                np.stack([value for _, value, _ in members]),
                np.stack([value for _, _, value in members]),
            )
            np.minimum.at(lower, rows, piece_lower)
            np.maximum.at(upper, rows, piece_upper)
        return lower, upper


def _isolines_at(
    axis: int, values: tuple[float, ...], box: np.ndarray, /
) -> tuple[tuple[int, float], ...]:
    slack = 1.0e-12 * max(1.0, float(np.max(np.abs(box[:, axis]))))
    return tuple(
        (axis, value)
        for value in values
        if box[0, axis] - slack <= value <= box[1, axis] + slack
    )


class PlanePatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array

    def __init__(
        self,
        origin: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
    ) -> None:
        self.origin = jnp.asarray(origin, dtype=jnp.float64).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=jnp.float64).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=jnp.float64).reshape((3,))

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.origin.dtype)
        return (
            self.origin
            + parameters_[..., :1] * self.first_axis
            + parameters_[..., 1:2] * self.second_axis
        )

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        return _interval_box(
            np.asarray(self.origin),
            (
                (np.asarray(self.first_axis), (box[0, 0], box[1, 0])),
                (np.asarray(self.second_axis), (box[0, 1], box[1, 1])),
            ),
        )


class CylinderPatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    radius: Array

    def __init__(
        self,
        origin: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        axis: ConvertibleToArray,
        radius: ConvertibleToArray,
    ) -> None:
        self.origin = jnp.asarray(origin, dtype=jnp.float64).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=jnp.float64).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=jnp.float64).reshape((3,))
        self.axis = jnp.asarray(axis, dtype=jnp.float64).reshape((3,))
        self.radius = jnp.asarray(radius, dtype=jnp.float64).reshape(())
        _check_frame(self.first_axis, self.second_axis, self.axis)
        if (
            not np.all(np.isfinite(self.origin))
            or not np.isfinite(float(self.radius))
            or float(self.radius) <= 0
        ):
            raise ValueError("A cylinder requires a finite origin and positive radius.")

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return _TWO_PI, None

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.origin.dtype)
        angle = parameters_[..., 0]
        height = parameters_[..., 1]
        radial = (
            jnp.cos(angle)[..., None] * self.first_axis
            + jnp.sin(angle)[..., None] * self.second_axis
        )
        return self.origin + self.radius * radial + height[..., None] * self.axis

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        cosine, sine = _trig_ranges(box[0, 0], box[1, 0])
        radius = float(self.radius)
        return _interval_box(
            np.asarray(self.origin),
            (
                (radius * np.asarray(self.first_axis), cosine),
                (radius * np.asarray(self.second_axis), sine),
                (np.asarray(self.axis), (box[0, 1], box[1, 1])),
            ),
        )


class ConePatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    reference_radius: Array
    semi_angle: Array

    def __init__(
        self,
        origin: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        axis: ConvertibleToArray,
        reference_radius: ConvertibleToArray,
        semi_angle: ConvertibleToArray,
    ) -> None:
        self.origin = jnp.asarray(origin, dtype=jnp.float64).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=jnp.float64).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=jnp.float64).reshape((3,))
        self.axis = jnp.asarray(axis, dtype=jnp.float64).reshape((3,))
        self.reference_radius = jnp.asarray(reference_radius, dtype=jnp.float64).reshape(
            ()
        )
        self.semi_angle = jnp.asarray(semi_angle, dtype=jnp.float64).reshape(())
        _check_frame(self.first_axis, self.second_axis, self.axis)
        if (
            not np.all(np.isfinite(self.origin))
            or not np.isfinite(float(self.reference_radius))
            or float(self.reference_radius) < 0
            or not np.isfinite(float(self.semi_angle))
            or abs(float(self.semi_angle)) >= pi / 2
        ):
            raise ValueError(
                "A cone requires a finite origin, nonnegative reference radius and finite nonsingular semi-angle."
            )

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return _TWO_PI, None

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.origin.dtype)
        angle = parameters_[..., 0]
        axial = parameters_[..., 1]
        radius = self.reference_radius + axial * jnp.tan(self.semi_angle)
        radial = (
            jnp.cos(angle)[..., None] * self.first_axis
            + jnp.sin(angle)[..., None] * self.second_axis
        )
        return self.origin + axial[..., None] * self.axis + radius[..., None] * radial

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        cosine, sine = _trig_ranges(box[0, 0], box[1, 0])
        slope = float(np.tan(float(self.semi_angle)))
        reference = float(self.reference_radius)
        ends = (reference + box[0, 1] * slope, reference + box[1, 1] * slope)
        radius = (min(ends), max(ends))
        return _interval_box(
            np.asarray(self.origin),
            (
                (np.asarray(self.first_axis), _product(radius, cosine)),
                (np.asarray(self.second_axis), _product(radius, sine)),
                (np.asarray(self.axis), (box[0, 1], box[1, 1])),
            ),
        )

    def degenerate_isolines(
        self, parameter_box: ConvertibleToArray, /
    ) -> tuple[tuple[int, float], ...]:
        box = _parameter_box(parameter_box)
        slope = float(np.tan(float(self.semi_angle)))
        if slope == 0.0:
            return ()
        return _isolines_at(1, (-float(self.reference_radius) / slope,), box)


class SpherePatch(AbstractSurfacePatch):
    center: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    radius: Array

    def __init__(
        self,
        center: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        axis: ConvertibleToArray,
        radius: ConvertibleToArray,
    ) -> None:
        self.center = jnp.asarray(center, dtype=jnp.float64).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=jnp.float64).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=jnp.float64).reshape((3,))
        self.axis = jnp.asarray(axis, dtype=jnp.float64).reshape((3,))
        self.radius = jnp.asarray(radius, dtype=jnp.float64).reshape(())
        _check_frame(self.first_axis, self.second_axis, self.axis)
        if (
            not np.all(np.isfinite(self.center))
            or not np.isfinite(float(self.radius))
            or float(self.radius) <= 0
        ):
            raise ValueError("A sphere requires a finite center and positive radius.")

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return _TWO_PI, None

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.center.dtype)
        longitude = parameters_[..., 0]
        latitude = parameters_[..., 1]
        equatorial = (
            jnp.cos(longitude)[..., None] * self.first_axis
            + jnp.sin(longitude)[..., None] * self.second_axis
        )
        direction = (
            jnp.cos(latitude)[..., None] * equatorial
            + jnp.sin(latitude)[..., None] * self.axis
        )
        return self.center + self.radius * direction

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = super().validate_parameter_box(parameter_box)
        slack = 1.0e-12
        if box[0, 1] < -0.5 * pi - slack or box[1, 1] > 0.5 * pi + slack:
            raise ValueError("Sphere latitude must lie within [-pi/2, pi/2].")
        return box

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        cos_u, sin_u = _trig_ranges(box[0, 0], box[1, 0])
        cos_v, sin_v = _trig_ranges(box[0, 1], box[1, 1])
        radius = float(self.radius)
        return _interval_box(
            np.asarray(self.center),
            (
                (radius * np.asarray(self.first_axis), _product(cos_v, cos_u)),
                (radius * np.asarray(self.second_axis), _product(cos_v, sin_u)),
                (radius * np.asarray(self.axis), sin_v),
            ),
        )

    def degenerate_isolines(
        self, parameter_box: ConvertibleToArray, /
    ) -> tuple[tuple[int, float], ...]:
        return _isolines_at(1, (-0.5 * pi, 0.5 * pi), _parameter_box(parameter_box))


class TorusPatch(AbstractSurfacePatch):
    center: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    major_radius: Array
    minor_radius: Array

    def __init__(
        self,
        center: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        axis: ConvertibleToArray,
        major_radius: ConvertibleToArray,
        minor_radius: ConvertibleToArray,
    ) -> None:
        self.center = jnp.asarray(center, dtype=jnp.float64).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=jnp.float64).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=jnp.float64).reshape((3,))
        self.axis = jnp.asarray(axis, dtype=jnp.float64).reshape((3,))
        self.major_radius = jnp.asarray(major_radius, dtype=jnp.float64).reshape(())
        self.minor_radius = jnp.asarray(minor_radius, dtype=jnp.float64).reshape(())
        _check_frame(self.first_axis, self.second_axis, self.axis)
        if (
            not np.all(np.isfinite(self.center))
            or not np.isfinite(float(self.major_radius))
            or not np.isfinite(float(self.minor_radius))
            or float(self.major_radius) <= 0
            or float(self.minor_radius) <= 0
        ):
            raise ValueError("A torus requires a finite center and positive radii.")

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return _TWO_PI, _TWO_PI

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.center.dtype)
        longitude = parameters_[..., 0]
        tube_angle = parameters_[..., 1]
        radial = (
            jnp.cos(longitude)[..., None] * self.first_axis
            + jnp.sin(longitude)[..., None] * self.second_axis
        )
        ring_radius = self.major_radius + self.minor_radius * jnp.cos(tube_angle)
        return (
            self.center
            + ring_radius[..., None] * radial
            + self.minor_radius * jnp.sin(tube_angle)[..., None] * self.axis
        )

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        cos_u, sin_u = _trig_ranges(box[0, 0], box[1, 0])
        cos_v, sin_v = _trig_ranges(box[0, 1], box[1, 1])
        major, minor = float(self.major_radius), float(self.minor_radius)
        ring = _product((minor, minor), cos_v)
        ring = (major + ring[0], major + ring[1])
        return _interval_box(
            np.asarray(self.center),
            (
                (np.asarray(self.first_axis), _product(ring, cos_u)),
                (np.asarray(self.second_axis), _product(ring, sin_u)),
                (np.asarray(self.axis), _product((minor, minor), sin_v)),
            ),
        )


class BSplineSurfacePatch(AbstractSurfacePatch):
    """Tensor-product rational B-spline surface with expanded knot vectors."""

    control_points: Array
    weights: Array
    u_knots: Array
    v_knots: Array
    u_degree: int = eqx.field(static=True)
    v_degree: int = eqx.field(static=True)

    def __init__(
        self,
        control_points: ConvertibleToArray,
        weights: ConvertibleToArray,
        u_knots: ConvertibleToArray,
        v_knots: ConvertibleToArray,
        u_degree: int,
        v_degree: int,
    ) -> None:
        control_points_ = jnp.asarray(control_points, dtype=jnp.float64)
        weights_ = jnp.asarray(weights, dtype=jnp.float64)
        u_knots_ = jnp.asarray(u_knots, dtype=jnp.float64).reshape((-1,))
        v_knots_ = jnp.asarray(v_knots, dtype=jnp.float64).reshape((-1,))
        if control_points_.ndim != 3 or control_points_.shape[-1] != 3:
            raise ValueError("control_points must have shape (num_u, num_v, 3).")
        if weights_.shape != control_points_.shape[:2]:
            raise ValueError("weights must match the control-point grid.")
        if int(u_degree) < 1 or int(v_degree) < 1:
            raise ValueError("B-spline degrees must be positive.")
        if control_points_.shape[0] <= int(u_degree) or control_points_.shape[1] <= int(
            v_degree
        ):
            raise ValueError("Each B-spline control-point axis must exceed its degree.")
        if u_knots_.shape[0] != control_points_.shape[0] + int(u_degree) + 1:
            raise ValueError(
                "u_knots length is inconsistent with control points and degree."
            )
        if v_knots_.shape[0] != control_points_.shape[1] + int(v_degree) + 1:
            raise ValueError(
                "v_knots length is inconsistent with control points and degree."
            )
        _check_knots(np.asarray(u_knots_), int(u_degree))
        _check_knots(np.asarray(v_knots_), int(v_degree))
        self.control_points = control_points_
        self.weights = weights_
        self.u_knots = u_knots_
        self.v_knots = v_knots_
        self.u_degree = int(u_degree)
        self.v_degree = int(v_degree)

    def _evaluate_one(
        self,
        parameters: Array,
        spans: tuple[int, int] | None = None,
    ) -> Array:
        # Pinned spans evaluate that tensor span's polynomial (Bernstein-piece
        # semantics, one-sided at C0 knots); otherwise spans are searched.
        axes = ((self.u_knots, self.u_degree), (self.v_knots, self.v_degree))
        u_stencil, v_stencil = (
            bspline_jet_stencil(
                knots,
                parameters[axis],
                degree=degree,
                maximum_order=0,
                bounds="clip",
            )
            if spans is None
            else bspline_jet_stencil(
                knots,
                parameters[axis],
                degree=degree,
                maximum_order=0,
                spans=jnp.asarray(spans[axis], dtype=jnp.int32),
                bounds="extrapolate",
            )
            for axis, (knots, degree) in enumerate(axes)
        )
        plan = TensorBSplineJetPlan((u_stencil, v_stencil), maximum_order=0)
        return _rational_coordinate_value(plan, self.weights, self.control_points)

    def _evaluate(self, parameters: Array, spans: tuple[int, int] | None, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.control_points.dtype)
        if parameters_.ndim == 1:
            return self._evaluate_one(parameters_, spans)
        leading = parameters_.shape[:-1]
        values = jax.vmap(lambda value: self._evaluate_one(value, spans))(
            parameters_.reshape((-1, 2))
        )
        return values.reshape((*leading, 3))

    def evaluate(self, parameters: Array, /) -> Array:
        return self._evaluate(parameters, None)

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = super().validate_parameter_box(parameter_box)
        for axis, (knots, degree) in enumerate(
            ((self.u_knots, self.u_degree), (self.v_knots, self.v_degree))
        ):
            host = np.asarray(knots)
            lower, upper = host[degree], host[-degree - 1]
            slack = 1.0e-12 * max(1.0, abs(lower), abs(upper))
            if box[0, axis] < lower - slack or box[1, axis] > upper + slack:
                raise ValueError("A parameter box lies outside the B-spline domain.")
        return box

    def bezier_pieces(self) -> tuple[RationalBezierPiece, ...]:
        """Numerical Bernstein patches with source-faithful coefficient enclosures."""
        points = np.asarray(self.control_points, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        homogeneous, homogeneous_lower, homogeneous_upper = _homogeneous_bounds(
            points, weights
        )
        u_pieces, u_bounds = bezier_refinement(
            homogeneous, np.asarray(self.u_knots), self.u_degree, 0
        )
        u_lower_pieces, u_upper_pieces, _ = bezier_refinement(
            homogeneous_lower,
            np.asarray(self.u_knots),
            self.u_degree,
            0,
            control_upper=homogeneous_upper,
        )
        pieces: list[RationalBezierPiece] = []
        for u_index, (strip, strip_lower, strip_upper, (u_lower, u_upper)) in enumerate(
            zip(u_pieces, u_lower_pieces, u_upper_pieces, u_bounds, strict=True)
        ):
            v_pieces, v_bounds = bezier_refinement(
                strip, np.asarray(self.v_knots), self.v_degree, 1
            )
            v_lower_pieces, v_upper_pieces, _ = bezier_refinement(
                strip_lower,
                np.asarray(self.v_knots),
                self.v_degree,
                1,
                control_upper=strip_upper,
            )
            pieces.extend(
                _piece(
                    piece,
                    (
                        (float(u_lower), float(u_upper)),
                        (float(v_lower), float(v_upper)),
                    ),
                    (u_index, v_index),
                    lo,
                    hi,
                )
                for v_index, (piece, lo, hi, (v_lower, v_upper)) in enumerate(
                    zip(v_pieces, v_lower_pieces, v_upper_pieces, v_bounds, strict=True)
                )
            )
        return tuple(pieces)

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        budget = _coordinate_budget()
        pieces = (
            self.bezier_pieces()
            if budget is None
            else _prepare_bspline_surface_spans(self, budget)
        )
        if budget is not None:
            selected_count = 0
            for piece in pieces:
                if any(
                    piece.parameter_bounds[axis][1] < box[0, axis]
                    or piece.parameter_bounds[axis][0] > box[1, axis]
                    for axis in (0, 1)
                ):
                    continue
                selected_count += 1
                terms = piece.homogeneous_controls.size
                degree_sum = sum(
                    value - 1 for value in piece.homogeneous_controls.shape[:2]
                )
                budget.reserve(
                    2 * terms * (degree_sum + 1) + 8 * terms, 4096 + 256 * terms
                )
            budget.reserve(12 * selected_count, 128 + 192 * selected_count)
        selected = [
            _piece_hull(piece, tuple(box[0]), tuple(box[1]))
            for piece in pieces
            if all(
                piece.parameter_bounds[axis][1] >= box[0, axis]
                and piece.parameter_bounds[axis][0] <= box[1, axis]
                for axis in (0, 1)
            )
        ]
        boxes = np.stack(selected)
        return np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))


def _prepare_bspline_surface_spans(
    surface: BSplineSurfacePatch,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[RationalBezierPiece, ...]:
    """Borrow tensor source spans from the same original spline-bank owner."""
    counts = surface.control_points.shape[:2]
    columns = surface.control_points.shape[-1] + 1
    degrees = (surface.u_degree, surface.v_degree)
    knot_sources = (surface.u_knots, surface.v_knots)
    with budget.temporary_scope():
        source_terms = (
            surface.control_points.size
            + surface.weights.size
            + sum(knots.size for knots in knot_sources)
        )
        budget.reserve(source_terms, 8192 + 24 * source_terms)
        key = canonical_fingerprint(
            {
                "kind": "source-bspline-surface-spans",
                "type": f"{type(surface).__module__}.{type(surface).__qualname__}",
                "degrees": degrees,
                "coefficients": array_tree_fingerprint(surface),
            }
        )
        cached = budget.bspline_span_cache.get(key)
        if cached is not None:
            return cached
        plans = []
        for count, degree, source_knots in zip(
            counts, degrees, knot_sources, strict=True
        ):
            budget.reserve(source_knots.size, 1024 + 48 * source_knots.size)
            knots = np.asarray(source_knots, dtype=np.float64)
            lower, upper = knots[degree], knots[-degree - 1]
            values, multiplicities = np.unique(knots, return_counts=True)
            active = (values >= lower) & (values <= upper)
            active_counts = multiplicities[active]
            spans = active_counts.size - 1
            endpoint_insertions = sum(
                max(0, degree + 1 - int(value))
                for value in (active_counts[0], active_counts[-1])
            )
            interior_insertions = sum(
                max(0, degree - int(value)) for value in active_counts[1:-1]
            )
            visits = 0
            rows, knot_count = count, knots.size
            knot_visits = 2 * knot_count
            for _ in range(endpoint_insertions):
                visits += 3 * (rows + 1)
                knot_visits += 2 * (knot_count + 1)
                rows += 1
                knot_count += 1
            active_rows = int(np.sum(active_counts)) + endpoint_insertions - degree - 1
            active_knots = active_rows + degree + 1
            for _ in range(interior_insertions):
                visits += 3 * (active_rows + 1)
                knot_visits += 2 * (active_knots + 1)
                active_rows += 1
                active_knots += 1
            plans.append(
                (
                    spans,
                    visits,
                    max(rows, active_rows),
                    knot_count + active_knots,
                    spans * (degree + 1),
                    knot_visits,
                )
            )
        u_spans, u_visits, u_refined, u_knots, u_rows, u_knot_visits = plans[0]
        v_spans, v_visits, v_refined, v_knots, v_rows, v_knot_visits = plans[1]
        source_rows = counts[0] * counts[1]
        u_terms = u_rows * counts[1] * columns
        piece_terms = u_rows * v_rows * columns
        extraction_work = (
            source_rows * (surface.control_points.shape[-1] + 3 * columns)
            + u_visits * counts[1] * columns
            + u_spans * v_visits * (degrees[0] + 1) * columns
            + 3 * (u_terms + piece_terms)
            + u_knot_visits
            + u_spans * v_knot_visits
        )
        scratch_terms = (
            3 * source_rows * columns
            + 3 * u_terms
            + 3 * piece_terms
            + 6 * max(u_refined * counts[1], v_refined * (degrees[0] + 1)) * columns
            + 4 * (u_knots + u_spans * v_knots)
            + 64 * columns
        )
        budget.reserve(
            extraction_work, 4096 + 8 * scratch_terms + 1024 * u_spans * v_spans
        )
        return _retain_bspline_spans(key, surface.bezier_pieces(), budget)


class BSplineSpanSurface(AbstractSurfacePatch):
    """One tensor knot span of a live rational B-spline surface.

    ``spans`` is immutable host addressing of an active ``(u, v)`` span. All
    numerical source leaves stay those of ``source``; evaluation is the
    canonical span polynomial, extended exactly like the span's Bernstein piece.
    """

    source: BSplineSurfacePatch
    spans: tuple[int, int] = eqx.field(static=True)

    def __init__(self, source: BSplineSurfacePatch, spans: tuple[int, int]) -> None:
        if not isinstance(source, BSplineSurfacePatch):
            raise TypeError("Pinned spans require a BSplineSurfacePatch source.")
        spans_ = tuple(spans)
        if len(spans_) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in spans_
        ):
            raise TypeError("Pinned surface spans must be two integers.")
        counts = source.control_points.shape[:2]
        degrees = (source.u_degree, source.v_degree)
        if not all(degrees[axis] <= spans_[axis] < counts[axis] for axis in (0, 1)):
            raise ValueError("Pinned surface spans must address active spans.")
        self.source = source
        self.spans = (spans_[0], spans_[1])

    def _span_box(self) -> np.ndarray:
        knots = (np.asarray(self.source.u_knots), np.asarray(self.source.v_knots))
        return np.asarray(
            [
                [knots[axis][self.spans[axis]] for axis in (0, 1)],
                [knots[axis][self.spans[axis] + 1] for axis in (0, 1)],
            ],
            dtype=np.float64,
        )

    def evaluate(self, parameters: Array, /) -> Array:
        return self.source._evaluate(parameters, self.spans)

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        span = self._span_box()
        if np.any(box[0] < span[0]) or np.any(box[1] > span[1]):
            raise ValueError("A pinned span box must lie inside its knot span.")
        return self.source.bounding_box(box)


def _domain_clamp(
    parameters: Array, lower: Array | float, upper: Array | float, /
) -> Array:
    """Clamp to a closed domain whose end values keep their one-sided jets.

    ``jnp.clip`` splits a tie between its operand and bound, halving the
    derivative exactly at a domain or span end; interval extensions of that
    tie admit every factor in ``[0, 1]`` on any box touching the end. Strict
    comparisons select the operand on the closed domain, so values are
    unchanged and every derivative there is the source's own one-sided jet.
    """
    return jnp.where(
        parameters < lower, lower, jnp.where(parameters > upper, upper, parameters)
    )


def _curve_parameter(curve: AbstractCurve, parameters: Array, /) -> Array:
    """Evaluate a carrier curve after clamping to its finite domain."""
    if isinstance(curve, BSplineCurve):
        return curve.evaluate(
            _domain_clamp(
                parameters, curve.knots[curve.degree], curve.knots[-curve.degree - 1]
            )
        )
    if isinstance(curve, BSplineSpanCurve):
        # The pinned span's Bernstein-piece domain, from the live knots.
        knots = curve.source.knots
        return curve.evaluate(
            _domain_clamp(parameters, knots[curve.span], knots[curve.span + 1])
        )
    domain = curve.parameter_domain
    if domain is None:
        return curve.evaluate(parameters)
    return curve.evaluate(_domain_clamp(parameters, domain[0], domain[1]))


def _require_space_curve(curve: AbstractCurve, name: str, /) -> AbstractCurve:
    if not isinstance(curve, AbstractCurve):
        raise TypeError(f"{name} must be an AbstractCurve.")
    if curve.ambient_dimension != 3:
        raise ValueError(f"{name} must be a three-dimensional curve.")
    return curve


def _curve_box_range(
    curve: AbstractCurve, first: float, last: float, /
) -> tuple[float, float]:
    """Clip a parameter range to the curve domain for bounding."""
    domain = curve.parameter_domain
    if domain is None:
        return first, last
    return max(first, domain[0]), min(last, domain[1])


class ExtrusionSurface(AbstractSurfacePatch):
    """Linear extrusion ``curve(u) + v * direction`` of a space curve."""

    curve: AbstractCurve
    direction: Array

    def __init__(self, curve: AbstractCurve, direction: ConvertibleToArray) -> None:
        curve_ = _require_space_curve(curve, "curve")
        direction_ = _host_vector(direction, "direction")
        if direction_.size != 3 or not np.linalg.norm(direction_) > 0.0:
            raise ValueError("An extrusion direction must be a nonzero 3D vector.")
        self.curve = curve_
        self.direction = jnp.asarray(direction_)

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.curve.period, None

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.direction.dtype)
        return (
            _curve_parameter(self.curve, parameters_[..., 0])
            + parameters_[..., 1:2] * self.direction
        )

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = super().validate_parameter_box(parameter_box)
        self.curve.validate_query_range(box[0, 0], box[1, 0])
        return box

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        base = self.curve.bounding_box(
            *_curve_box_range(self.curve, box[0, 0], box[1, 0])
        )
        sweep = _interval_box(
            np.zeros(3), ((np.asarray(self.direction), (box[0, 1], box[1, 1])),)
        )
        return np.stack((base[0] + sweep[0], base[1] + sweep[1]))


def _axis_frame(axis: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal completion of a unit axis."""
    seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = seed - (seed @ axis) * axis
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def _rotation_jet_bounds(
    axis: np.ndarray,
    vector: tuple[np.ndarray, np.ndarray],
    angle: tuple[np.ndarray | float, np.ndarray | float],
    derivative: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose ``d^n/du^n R(u, axis) w`` for ``w`` in a box and ``u`` in an interval.

    The evaluated rotation ``w cos u + (k x w) sin u + k (k.w)(1 - cos u)``
    differentiates to the same terms with ``cos``/``sin`` shifted by
    ``n pi / 2`` and no constant axial term, by outward-rounded interval
    arithmetic over the exact unit-axis coefficients. Vectors ``(..., 3)``
    pair with angle intervals of the leading shape ``(...)``.
    """
    from .._interval_enclosure import (
        _interval_cos,
        _interval_sin,
        interval_add,
        interval_multiply,
        interval_subtract,
    )

    bounds = (np.asarray(angle[0]), np.asarray(angle[1]))
    cosine_lower, cosine_upper = _interval_cos(bounds)
    sine_lower, sine_upper = _interval_sin(bounds)
    cosine = (cosine_lower[..., None], cosine_upper[..., None])
    sine = (sine_lower[..., None], sine_upper[..., None])
    match derivative:
        case 0:
            even, odd = cosine, sine
            axial = interval_subtract((np.asarray(1.0), np.asarray(1.0)), cosine)
        case 1:
            even, odd = (-sine[1], -sine[0]), cosine
            axial = sine
        case 2:
            even, odd = (-cosine[1], -cosine[0]), (-sine[1], -sine[0])
            axial = cosine
        case _:
            raise ValueError("Rotation jets support derivative orders zero to two.")
    lower, upper = (np.asarray(value, dtype=np.float64) for value in vector)
    lead, trail = np.asarray((1, 2, 0)), np.asarray((2, 0, 1))
    cross = interval_subtract(
        interval_multiply(
            (axis[lead], axis[lead]), (lower[..., trail], upper[..., trail])
        ),
        interval_multiply(
            (axis[trail], axis[trail]), (lower[..., lead], upper[..., lead])
        ),
    )
    products = interval_multiply((axis, axis), (lower, upper))
    dot = interval_add(
        interval_add(
            (products[0][..., 0], products[1][..., 0]),
            (products[0][..., 1], products[1][..., 1]),
        ),
        (products[0][..., 2], products[1][..., 2]),
    )
    projection = interval_multiply((dot[0][..., None], dot[1][..., None]), axial)
    return interval_add(
        interval_add(
            interval_multiply((lower, upper), even), interval_multiply(cross, odd)
        ),
        interval_multiply((axis, axis), projection),
    )


def _fraction_down(value: Fraction, /) -> float:
    represented = float(value)
    return (
        represented
        if Fraction(represented) <= value
        else float(np.nextafter(represented, -np.inf))
    )


def _fraction_up(value: Fraction, /) -> float:
    represented = float(value)
    return (
        represented
        if Fraction(represented) >= value
        else float(np.nextafter(represented, np.inf))
    )


@dataclass(frozen=True, slots=True)
class _MeridianSpans:
    """Prepared rational hodographs of every span of a B-spline meridian.

    On a span with local parameter ``t`` and homogeneous numerator ``P`` and
    weight ``w`` of degree ``p``, the ``k``-th parameter derivative is the
    rational Bezier ``D_k / w^(k+1)`` of degree ``(k+1) p``, with
    ``D_0 = P`` and ``D_(k+1) = D_k' w - (k+1) D_k w'`` (degree-elevated once)
    scaled exactly by the span width. ``hodographs[k]`` holds outward
    enclosures of the exact coefficients ``(D_k, w^(k+1))``, shaped
    ``(span, (k+1) p + 1, 4)``. Positive weights make every denominator
    coefficient positive, so each derivative lies in the hull of its restricted
    control quotients, with no quotient-recurrence dependency loss. ``bending``
    likewise encloses ``(D_1 x D_1', w^4)``, the rational Bezier ``c' x c''``:
    ``D_1 x D_2 = w (D_1 x D_1')`` because ``D_1 x D_1 = 0``.
    Interior ``junctions`` carry exact one-sided proofs: ``g1`` (equal points,
    tangents in one direction), ``c1`` (equal first derivatives) and
    ``parallel_c1`` (C1 with an equal curvature vector normal part, so every
    parallel curve within reach is C1 there too).
    """

    starts: np.ndarray
    ends: np.ndarray
    hodographs: tuple[tuple[np.ndarray, np.ndarray], ...]
    bending: tuple[np.ndarray, np.ndarray]
    junctions: np.ndarray
    g1: np.ndarray
    c1: np.ndarray
    parallel_c1: np.ndarray


@dataclass(frozen=True, slots=True)
class _MeridianFrame:
    """Exact planar meridian of a revolution about an exactly unit axis.

    Every control point lies in the plane through the axis with normal
    ``normal`` (an enclosure of the unit normal), so positive weights keep the
    whole profile in it.
    """

    axis: np.ndarray
    origin: np.ndarray
    normal: tuple[np.ndarray, np.ndarray]
    spans: _MeridianSpans


@dataclass(frozen=True, slots=True)
class _MeridianJets:
    """Closed span overlaps of a batch of meridian ranges and their jets.

    Overlaps are ordered by row; ``offsets`` starts each row's overlaps.
    ``jets[k]`` encloses the ``k``-th derivative over each overlap, followed
    by ``c' x c''`` when bending was requested.
    """

    rows: np.ndarray
    offsets: np.ndarray
    widths: np.ndarray
    jets: tuple[tuple[np.ndarray, np.ndarray], ...]

    def hull(self, order: int, /) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = self.jets[order]
        return (
            np.minimum.reduceat(lower, self.offsets, axis=0),
            np.maximum.reduceat(upper, self.offsets, axis=0),
        )


# Third meridian jets give second jets of an exact parallel meridian.
_MERIDIAN_ORDER = 3
# Hodograph key of the bending numerator ``c' x c''`` after the jet orders.
_BENDING = _MERIDIAN_ORDER + 1
# Overlaps per admitted restriction batch; bounds transient ledger scratch.
_MERIDIAN_CHUNK = 4096


@dataclass(frozen=True, slots=True)
class _MeridianOverlaps:
    """Closed span overlaps of the distinct ranges of one batch, and their jets.

    A sweep's meridian jets depend only on a row's meridian range, and rows
    of one batch repeat few distinct ranges, so overlaps are formed and
    enclosed per distinct range. ``index`` gathers each row's overlaps (in
    row order, starting at ``offsets``) from the distinct ones. ``jets``
    holds each hodograph key enclosed so far over the distinct overlaps:
    orders ``0..3`` and ``_BENDING``; each is enclosed and charged once.
    """

    spans: _MeridianSpans
    first: np.ndarray
    last: np.ndarray
    span: np.ndarray
    local_first: np.ndarray
    local_last: np.ndarray
    rows: np.ndarray
    offsets: np.ndarray
    widths: np.ndarray
    index: np.ndarray
    jets: dict[int, tuple[np.ndarray, np.ndarray]]


_SHARED_MERIDIAN_JETS: ContextVar[list[_MeridianOverlaps] | None] = ContextVar(
    "shared_meridian_jets", default=None
)


@contextmanager
def shared_meridian_jets() -> Iterator[None]:
    """Share meridian jets among consumers of the same chart rows.

    Interpolation and normal-turn bounds of one batch of chart triangles
    restrict the same closed span overlaps. Within this scope every batch's
    jets are kept for its lifetime, so a later consumer of identical ranges
    encloses only the hodographs not yet enclosed.
    """
    token = _SHARED_MERIDIAN_JETS.set([])
    try:
        yield
    finally:
        _SHARED_MERIDIAN_JETS.reset(token)


def _exact_cross(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...], /
) -> tuple[Fraction, Fraction, Fraction]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


type _ExactBernstein = list[tuple[Fraction, ...]]


def _bernstein_derivative(coefficients: _ExactBernstein, /) -> _ExactBernstein:
    degree = len(coefficients) - 1
    return [
        tuple(degree * (b - a) for a, b in zip(left, right, strict=True))
        for left, right in zip(coefficients[:-1], coefficients[1:], strict=True)
    ]


def _bernstein_product(
    first: _ExactBernstein, second: list[Fraction], /
) -> _ExactBernstein:
    """Exact Bernstein coefficients of a vector times a scalar polynomial."""
    left, right = len(first) - 1, len(second) - 1
    result = [[Fraction()] * len(first[0]) for _ in range(left + right + 1)]
    for i, vector in enumerate(first):
        for j, scalar in enumerate(second):
            factor = Fraction(comb(left, i) * comb(right, j), comb(left + right, i + j))
            row = result[i + j]
            for axis, entry in enumerate(vector):
                row[axis] += factor * scalar * entry
    return [tuple(row) for row in result]


def _bernstein_cross(
    first: _ExactBernstein, second: _ExactBernstein, /
) -> _ExactBernstein:
    """Exact Bernstein coefficients of the cross product of two 3D polynomials."""
    left, right = len(first) - 1, len(second) - 1
    result = [[Fraction()] * 3 for _ in range(left + right + 1)]
    for i, vector in enumerate(first):
        for j, other in enumerate(second):
            factor = Fraction(comb(left, i) * comb(right, j), comb(left + right, i + j))
            row = result[i + j]
            for axis, entry in enumerate(_exact_cross(vector, other)):
                row[axis] += factor * entry
    return [tuple(row) for row in result]


def _exact_hodographs(
    piece: RationalBezierPiece, /
) -> list[tuple[_ExactBernstein, list[Fraction]]]:
    """Exact ``(D_k, w^(k+1))`` of one span for ``k = 0.._MERIDIAN_ORDER``."""
    controls = piece.homogeneous_controls
    rows = [tuple(controls[index]) for index in range(controls.shape[0])]
    degree = len(rows) - 1
    ((start, end),) = piece.parameter_bounds
    scale = 1 / (Fraction(end) - Fraction(start))
    weights = [row[-1] for row in rows]
    slopes = [degree * (b - a) for a, b in zip(weights[:-1], weights[1:], strict=True)]
    numerator: _ExactBernstein = [row[:-1] for row in rows]
    denominator = list(weights)
    levels = []
    for derivative in range(_MERIDIAN_ORDER + 1):
        factor = scale**derivative
        levels.append(
            ([tuple(factor * entry for entry in row) for row in numerator], denominator)
        )
        if derivative == _MERIDIAN_ORDER:
            break
        changing = _bernstein_product(_bernstein_derivative(numerator), weights)
        moving = _bernstein_product(
            numerator, [(derivative + 1) * slope for slope in slopes]
        )
        numerator = _bernstein_product(
            [
                tuple(a - b for a, b in zip(first, second, strict=True))
                for first, second in zip(changing, moving, strict=True)
            ],
            [Fraction(1), Fraction(1)],
        )
        denominator = [
            entry
            for (entry,) in _bernstein_product(
                [(value,) for value in denominator], weights
            )
        ]
    return levels


def _junction_flags(
    left: list[tuple[_ExactBernstein, list[Fraction]]],
    right: list[tuple[_ExactBernstein, list[Fraction]]],
    /,
) -> tuple[bool, bool, bool]:
    """Exact G1, C1 and parallel-C1 proofs from endpoint-interpolating hodographs."""
    point, incoming, bending = (
        tuple(entry / denominator[-1] for entry in numerator[-1])
        for numerator, denominator in left[:3]
    )
    joined, outgoing, turning = (
        tuple(entry / denominator[0] for entry in numerator[0])
        for numerator, denominator in right[:3]
    )
    g1 = (
        point == joined
        and any(incoming)
        and not any(_exact_cross(incoming, outgoing))
        and sum(a * b for a, b in zip(incoming, outgoing, strict=True)) > 0
    )
    c1 = point == joined and incoming == outgoing and any(incoming)
    parallel = c1 and _exact_cross(incoming, bending) == _exact_cross(outgoing, turning)
    return g1, c1, parallel


def _outward_spans(
    spans: list[tuple[_ExactBernstein, list[Fraction]]], /
) -> tuple[np.ndarray, np.ndarray]:
    """Outward ``(span, coefficient, 4)`` enclosures of exact rational Beziers."""
    rows = [
        [(*vector, weight) for vector, weight in zip(numerator, denominator, strict=True)]
        for numerator, denominator in spans
    ]
    return (
        np.asarray(
            [[[_fraction_down(entry) for entry in row] for row in span] for span in rows]
        ),
        np.asarray(
            [[[_fraction_up(entry) for entry in row] for row in span] for span in rows]
        ),
    )


def _prepare_meridian_spans(curve: BSplineCurve, /) -> _MeridianSpans:
    """Exact host hodograph preparation, rounded outward once per source."""
    pieces = curve.exact_bezier_pieces()
    exact = [_exact_hodographs(piece) for piece in pieces]
    bending: list[tuple[_ExactBernstein, list[Fraction]]] = []
    for piece, levels in zip(pieces, exact, strict=True):
        ((start, end),) = piece.parameter_bounds
        scale = 1 / (Fraction(end) - Fraction(start))
        velocity, square = levels[1]
        # c' x c'' = (D_1 x D_1') / w^4; the numerator, one degree lower than
        # w^4, is degree-elevated once onto the same Bernstein basis.
        turning = [
            tuple(scale * entry for entry in row)
            for row in _bernstein_derivative(velocity)
        ]
        bending.append(
            (
                _bernstein_product(
                    _bernstein_cross(velocity, turning), [Fraction(1), Fraction(1)]
                ),
                [
                    entry
                    for (entry,) in _bernstein_product(
                        [(value,) for value in square], square
                    )
                ],
            )
        )
    flags = np.asarray(
        [
            _junction_flags(left, right)
            for left, right in zip(exact[:-1], exact[1:], strict=True)
        ],
        dtype=np.bool_,
    ).reshape((-1, 3))
    starts = np.asarray([piece.parameter_bounds[0][0] for piece in pieces])
    ends = np.asarray([piece.parameter_bounds[0][1] for piece in pieces])
    return _MeridianSpans(
        starts,
        ends,
        tuple(
            _outward_spans([levels[derivative] for levels in exact])
            for derivative in range(_MERIDIAN_ORDER + 1)
        ),
        _outward_spans(bending),
        ends[:-1],
        flags[:, 0],
        flags[:, 1],
        flags[:, 2],
    )


def _meridian_spans(
    curve: AbstractCurve, budget: CoordinateEnclosureBudget | None, /
) -> _MeridianSpans | None:
    """Prepared hodographs of a positively weighted spatial B-spline, or None.

    A budget retains one preparation per exact source identity, so batches
    and single-box queries reuse it.
    """
    if (
        not isinstance(curve, BSplineCurve)
        or curve.ambient_dimension != 3
        or np.any(np.asarray(curve.weights) <= 0)
    ):
        return None
    if budget is None:
        return _prepare_meridian_spans(curve)
    source_terms = curve.control_points.size + curve.weights.size + curve.knots.shape[0]
    with budget.temporary_scope():
        budget.reserve(source_terms, 8192 + 24 * source_terms)
        key = canonical_fingerprint(
            {
                "kind": "source-meridian-hodographs",
                "type": f"{type(curve).__module__}.{type(curve).__qualname__}",
                "degree": curve.degree,
                "coefficients": array_tree_fingerprint(curve),
            }
        )
    cached = budget.meridian_span_cache.get(key)
    if cached is not None:
        return cached
    # Exact Boehm extraction and hodograph products: the degree-(k+1)p
    # coefficients of every derivative order are quadratic in their degree.
    terms = sum(
        ((order + 1) * curve.degree + 1) ** 2 for order in range(_MERIDIAN_ORDER + 1)
    )
    with budget.temporary_scope():
        budget.reserve(4 * source_terms * terms, 8192 + 512 * source_terms * terms)
        spans = _prepare_meridian_spans(curve)
    arrays = (
        spans.starts,
        spans.ends,
        spans.junctions,
        spans.g1,
        spans.c1,
        spans.parallel_c1,
        *(array for pair in (*spans.hodographs, spans.bending) for array in pair),
    )
    budget.retain_basis((key, spans, arrays))
    budget.meridian_span_cache[key] = spans
    return spans


def _closure_tangent_continuous(
    patch: AbstractSurfacePatch,
    axis: int,
    lower: float,
    upper: float,
    /,
    *,
    tangent: bool = True,
) -> bool:
    """Exact proof that a chart's two sides on ``axis`` glue into one seam.

    A proved native ``2 pi`` period glues them. Otherwise the swept B-spline
    profile of a revolution (``axis`` one) or extrusion (``axis`` zero) must
    close exactly over its whole domain; with ``tangent`` the closure is also
    tangent-continuous (G1), which keeps every normal offset glued.
    """
    from ._intersection_curve import _native_period_symbols

    if _native_period_symbols(patch)[axis] == "two_pi" and upper - lower == _TWO_PI:
        return True
    match patch:
        case RevolutionSurface() if axis == 1:
            curve = patch.curve
        case ExtrusionSurface() if axis == 0:
            curve = patch.curve
        case _:
            return False
    if not isinstance(curve, BSplineCurve) or curve.parameter_domain != (lower, upper):
        return False
    controls = [
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(curve.control_points)
    ]
    if controls[0] != controls[-1]:
        return False
    if not tangent:
        return True
    # Clamped positive-weight end derivatives are positive multiples of the
    # end control legs, so the closure is tangent-continuous iff these legs
    # are exactly positively parallel.
    first = tuple(b - a for a, b in zip(controls[0], controls[1], strict=True))
    last = tuple(b - a for a, b in zip(controls[-2], controls[-1], strict=True))
    return not any(_exact_cross(first, last)) and (
        sum((a * b for a, b in zip(first, last, strict=True)), Fraction()) > 0
    )


def _exact_unit(axis: np.ndarray, /) -> bool:
    return sum(Fraction(float(value)) ** 2 for value in axis) == 1


def _meridian_frame(
    surface: RevolutionSurface, budget: CoordinateEnclosureBudget | None, /
) -> _MeridianFrame | None:
    """Prove an exactly planar meridian containing an exactly unit axis, or None."""
    from ._sphere_membership import _square_root_interval

    curve = surface.curve
    axis = np.asarray(surface.axis_direction, dtype=np.float64)
    origin = np.asarray(surface.axis_origin, dtype=np.float64)
    if not isinstance(curve, BSplineCurve) or not _exact_unit(axis):
        return None
    unit = tuple(Fraction(float(value)) for value in axis)
    base = tuple(Fraction(float(value)) for value in origin)
    relative = [
        tuple(
            Fraction(float(value)) - shift for value, shift in zip(row, base, strict=True)
        )
        for row in np.asarray(curve.control_points, dtype=np.float64)
    ]
    normal = next(
        (cross for cross in (_exact_cross(unit, row) for row in relative) if any(cross)),
        None,
    )
    if normal is None or any(
        sum(a * b for a, b in zip(row, normal, strict=True)) for row in relative
    ):
        return None
    spans = _meridian_spans(curve, budget)
    if spans is None:
        return None
    norm_lower, norm_upper = _square_root_interval(
        sum((value * value for value in normal), Fraction())
    )
    lower = np.asarray(
        [
            _fraction_down(value / Fraction(norm_upper if value >= 0 else norm_lower))
            for value in normal
        ]
    )
    upper = np.asarray(
        [
            _fraction_up(value / Fraction(norm_lower if value >= 0 else norm_upper))
            for value in normal
        ]
    )
    return _MeridianFrame(axis, origin, (lower, upper), spans)


def _revolution_ranges(
    surface: RevolutionSurface, boxes: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Validate chart boxes of a B-spline meridian revolution; clipped v ranges."""
    curve = surface.curve
    if not isinstance(curve, BSplineCurve):
        raise TypeError("Batched revolution ranges require a B-spline meridian.")
    if not np.all(np.isfinite(boxes)):
        raise ValueError("A parameter box must be a finite (2, 2) array.")
    if np.any(boxes[:, 1] < boxes[:, 0]):
        raise ValueError("A closed parameter query box requires lower <= upper.")
    if np.any(boxes[:, 1, 0] - boxes[:, 0, 0] > _TWO_PI * (1.0 + 1.0e-12)):
        raise ValueError("A periodic parameter extent cannot exceed its period.")
    first, last = boxes[:, 0, 1], boxes[:, 1, 1]
    lower, upper = curve.parameter_domain
    slack = 1.0e-12 * max(1.0, abs(lower), abs(upper))
    ranged = first < last
    if np.any(ranged & ((first < lower - slack) | (last > upper + slack))):
        raise ValueError("A curve range lies outside the carrier domain.")
    if np.any(~ranged & ((first < lower) | (first > upper))):
        raise ValueError("A point query lies outside the carrier domain.")
    return np.maximum(first, lower), np.minimum(last, upper)


def _split_rows(
    lower: np.ndarray, upper: np.ndarray, parameter: np.ndarray, /
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Row-wise outward de Casteljau split of ``(row, control, column)`` boxes."""
    scale = parameter[:, None, None]
    complement = (
        np.maximum(0.0, np.nextafter(1.0 - scale, -np.inf)),
        np.minimum(1.0, np.nextafter(1.0 - scale, np.inf)),
    )
    left_lower, left_upper = [lower[:, 0]], [upper[:, 0]]
    right_lower, right_upper = [lower[:, -1]], [upper[:, -1]]
    while lower.shape[1] > 1:
        products = np.stack(
            (
                complement[0] * lower[:, :-1],
                complement[1] * lower[:, :-1],
                complement[0] * upper[:, :-1],
                complement[1] * upper[:, :-1],
            )
        )
        lower = np.nextafter(
            np.nextafter(np.min(products, axis=0), -np.inf)
            + np.nextafter(scale * lower[:, 1:], -np.inf),
            -np.inf,
        )
        upper = np.nextafter(
            np.nextafter(np.max(products, axis=0), np.inf)
            + np.nextafter(scale * upper[:, 1:], np.inf),
            np.inf,
        )
        left_lower.append(lower[:, 0])
        left_upper.append(upper[:, 0])
        right_lower.append(lower[:, -1])
        right_upper.append(upper[:, -1])
    return (
        (np.stack(left_lower, axis=1), np.stack(left_upper, axis=1)),
        (np.stack(right_lower[::-1], axis=1), np.stack(right_upper[::-1], axis=1)),
    )


def _restrict_rows(
    lower: np.ndarray, upper: np.ndarray, first: np.ndarray, last: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise ``restrict_bernstein_bounds`` to local ``[first, last]`` in [0, 1].

    The same outward splits; the rescaled second split parameter is rounded
    upward in binary64 rather than through an exact host ratio.
    """
    at_end = first >= 1.0
    _, right = _split_rows(lower, upper, np.where(at_end, 0.0, first))
    split = ((first > 0.0) & ~at_end)[:, None, None]
    restricted_lower = np.where(split, right[0], lower)
    restricted_upper = np.where(split, right[1], upper)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.minimum(
            1.0,
            np.nextafter(
                np.nextafter(last - first, np.inf) / np.nextafter(1.0 - first, -np.inf),
                np.inf,
            ),
        )
    trim = (last < 1.0) & ~at_end
    left, _ = _split_rows(restricted_lower, restricted_upper, np.where(trim, ratio, 1.0))
    restricted_lower = np.where(trim[:, None, None], left[0], restricted_lower)
    restricted_upper = np.where(trim[:, None, None], left[1], restricted_upper)
    end = at_end[:, None, None]
    return (
        np.where(end, lower[:, -1:], restricted_lower),
        np.where(end, upper[:, -1:], restricted_upper),
    )


def _span_jets(
    hodographs: list[tuple[np.ndarray, np.ndarray]],
    span: np.ndarray,
    first: np.ndarray,
    last: np.ndarray,
    /,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Enclosures of rational Beziers over local span ranges, row-wise.

    Each is positively weighted, so it lies in the outward hull of its
    restricted control quotients.
    """
    jets = []
    for lower, upper in hodographs:
        restricted_lower, restricted_upper = _restrict_rows(
            lower[span], upper[span], first, last
        )
        weights = (restricted_lower[..., -1:], restricted_upper[..., -1:])
        if np.any(weights[0] <= 0.0):
            raise _rational_enclosure_error(
                "Rational source jets require a positive denominator enclosure."
            )
        quotients = np.stack(
            (
                restricted_lower[..., :-1] / weights[0],
                restricted_lower[..., :-1] / weights[1],
                restricted_upper[..., :-1] / weights[0],
                restricted_upper[..., :-1] / weights[1],
            )
        )
        jets.append(
            (
                np.nextafter(np.min(quotients, axis=(0, 2)), -np.inf),
                np.nextafter(np.max(quotients, axis=(0, 2)), np.inf),
            )
        )
    return jets


def _meridian_overlaps(
    spans: _MeridianSpans,
    first: np.ndarray,
    last: np.ndarray,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> _MeridianOverlaps:
    """Closed span overlaps of distinct ranges, with outward local coordinates."""
    if budget is not None:
        # One lexicographic sort of the range pairs finds the distinct ranges.
        budget.reserve(2 * first.size * max(1, first.size.bit_length()))
    ranges, inverse = np.unique(
        np.stack((first, last), axis=1), axis=0, return_inverse=True
    )
    inverse = inverse.reshape((-1,))
    distinct, span = np.nonzero(
        (ranges[:, 0, None] <= spans.ends) & (ranges[:, 1, None] >= spans.starts)
    )
    starts = np.flatnonzero(np.diff(distinct, prepend=-1))
    if starts.size != ranges.shape[0]:
        raise ValueError("A meridian range selects no source span.")
    low = np.maximum(ranges[distinct, 0], spans.starts[span])
    high = np.minimum(ranges[distinct, 1], spans.ends[span])
    width = spans.ends[span] - spans.starts[span]
    local_first = np.maximum(
        0.0,
        np.nextafter(
            np.nextafter(low - spans.starts[span], -np.inf) / np.nextafter(width, np.inf),
            -np.inf,
        ),
    )
    local_last = np.minimum(
        1.0,
        np.nextafter(
            np.nextafter(high - spans.starts[span], np.inf)
            / np.nextafter(width, -np.inf),
            np.inf,
        ),
    )
    # Each row repeats the consecutive overlaps of its distinct range.
    counts = np.diff(np.append(starts, distinct.size))[inverse]
    offsets = np.cumsum(counts) - counts
    index = np.repeat(starts[inverse] - offsets, counts) + np.arange(np.sum(counts))
    return _MeridianOverlaps(
        spans,
        first,
        last,
        span,
        local_first,
        local_last,
        np.repeat(np.arange(first.size), counts),
        offsets,
        (high - low)[index],
        index,
        {},
    )


def _meridian_jets(
    spans: _MeridianSpans,
    first: np.ndarray,
    last: np.ndarray,
    order: int,
    budget: CoordinateEnclosureBudget | None,
    /,
    *,
    bending: bool = False,
) -> _MeridianJets:
    """Closed-span overlap jets ``0..order`` of every range ``[first, last]``.

    Every overlap of every distinct range is enclosed in admitted batches of
    at most ``_MERIDIAN_CHUNK`` restrictions, then gathered to the rows; no
    per-row host program is formed. ``bending`` appends ``c' x c''``. Inside
    ``shared_meridian_jets`` the hodographs already enclosed for identical
    ranges are reused, not redone.
    """
    if order not in range(_MERIDIAN_ORDER + 1):
        raise ValueError("Meridian jets support orders zero to three.")
    shared = _SHARED_MERIDIAN_JETS.get()
    overlaps = None
    for candidate in shared or ():
        if budget is not None:
            # Each identity lookup compares both range endpoints once.
            budget.reserve(2 * first.size)
        if (
            candidate.spans is spans
            and np.array_equal(candidate.first, first)
            and np.array_equal(candidate.last, last)
        ):
            overlaps = candidate
            break
    if overlaps is None:
        overlaps = _meridian_overlaps(spans, first, last, budget)
        if shared is not None:
            shared.append(overlaps)
    keys = [*range(order + 1), *((_BENDING,) if bending else ())]
    missing = [key for key in keys if key not in overlaps.jets]
    columns = spans.hodographs[0][0].shape[2]
    if missing:
        hodographs = [
            spans.bending if key == _BENDING else spans.hodographs[key] for key in missing
        ]
        count = overlaps.span.size
        sizes = [lower.shape[1] for lower, _ in hodographs]
        # Two complete de Casteljau triangles and the four-corner control
        # quotient hull of every enclosed rational Bezier, per overlap.
        work = sum(2 * size * (size + 2) * columns for size in sizes)
        scratch = 8 * sum(32 * size * columns for size in sizes)
        jets = [
            (np.empty((count, columns - 1)), np.empty((count, columns - 1)))
            for _ in hodographs
        ]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            for begin in range(0, count, _MERIDIAN_CHUNK):
                part = slice(begin, min(count, begin + _MERIDIAN_CHUNK))
                chunk = part.stop - part.start
                with budget.temporary_scope() if budget is not None else nullcontext():
                    if budget is not None:
                        budget.reserve(work * chunk, 4096 + scratch * chunk)
                    for target, (lower, upper) in zip(
                        jets,
                        _span_jets(
                            hodographs,
                            overlaps.span[part],
                            overlaps.local_first[part],
                            overlaps.local_last[part],
                        ),
                        strict=True,
                    ):
                        target[0][part], target[1][part] = lower, upper
        overlaps.jets.update(zip(missing, jets, strict=True))
    if budget is not None:
        # Gathering both bounds of every requested jet to each row overlap.
        budget.reserve(2 * overlaps.index.size * (columns - 1) * len(keys))
    return _MeridianJets(
        overlaps.rows,
        overlaps.offsets,
        overlaps.widths,
        tuple(
            (overlaps.jets[key][0][overlaps.index], overlaps.jets[key][1][overlaps.index])
            for key in keys
        ),
    )


def _junction_rows(
    spans: _MeridianSpans,
    first: np.ndarray,
    last: np.ndarray,
    admitted: np.ndarray,
    /,
    *,
    closed: bool,
) -> np.ndarray:
    """Rows whose range meets (or strictly straddles) an unadmitted junction."""
    knots = spans.junctions[~admitted]
    if closed:
        hits = (first[:, None] <= knots) & (knots <= last[:, None])
    else:
        hits = (first[:, None] < knots) & (knots < last[:, None])
    return np.any(hits, axis=1)


def _interval_cross(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray], /
) -> tuple[np.ndarray, np.ndarray]:
    from .._interval_enclosure import interval_multiply, interval_subtract

    lead, trail = np.asarray((1, 2, 0)), np.asarray((2, 0, 1))
    return interval_subtract(
        interval_multiply(
            (first[0][..., lead], first[1][..., lead]),
            (second[0][..., trail], second[1][..., trail]),
        ),
        interval_multiply(
            (first[0][..., trail], first[1][..., trail]),
            (second[0][..., lead], second[1][..., lead]),
        ),
    )


def _interval_dot(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray], /
) -> tuple[np.ndarray, np.ndarray]:
    """Outward dot product over the last axis."""
    from .._interval_enclosure import interval_add, interval_multiply

    products = interval_multiply(first, second)
    total = (products[0][..., 0], products[1][..., 0])
    for axis in range(1, products[0].shape[-1]):
        total = interval_add(total, (products[0][..., axis], products[1][..., axis]))
    return np.asarray(total[0]), np.asarray(total[1])


def _directed_sum_of_squares(values: np.ndarray, direction: float, /) -> np.ndarray:
    total = np.zeros(values.shape[:-1])
    for axis in range(values.shape[-1]):
        total = np.nextafter(
            total + np.nextafter(values[..., axis] * values[..., axis], direction),
            direction,
        )
    return np.maximum(total, 0.0)


def _interval_norm(
    value: tuple[np.ndarray, np.ndarray], /
) -> tuple[np.ndarray, np.ndarray]:
    """Outward ``[min, max]`` Euclidean norms of vector boxes over the last axis."""
    separated = np.maximum(np.maximum(value[0], -value[1]), 0.0)
    magnitude = np.maximum(np.abs(value[0]), np.abs(value[1]))
    with np.errstate(over="ignore", invalid="ignore"):
        lower = np.maximum(
            np.nextafter(np.sqrt(_directed_sum_of_squares(separated, -np.inf)), -np.inf),
            0.0,
        )
        upper = np.nextafter(np.sqrt(_directed_sum_of_squares(magnitude, np.inf)), np.inf)
    return lower, np.where(np.all(np.isfinite(magnitude), axis=-1), upper, np.inf)


def _parallel_meridian_jets(
    frame: _MeridianFrame,
    jets: tuple[tuple[np.ndarray, np.ndarray], ...],
    distance: float,
    /,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], np.ndarray]:
    """Jets ``q - o, q', q''`` of the parallel meridian ``q = c + d n``, row-wise.

    Off the axis, the revolution's unit normal is ``R(u) n`` with
    ``n = s (m x t)``, unit tangent ``t = c' / |c'|`` and constant side ``s``;
    ``t' = (c'' - t s') / |c'|`` and ``t'' = (c''' - 2 t' s' - t s'') / |c'|``
    with ``s' = t.c''`` and ``s'' = t'.c'' + t.c'''``. Rows whose side or
    speed is not separated from zero are returned as not regular.
    """
    from .._interval_enclosure import (
        interval_add,
        interval_divide,
        interval_multiply,
        interval_subtract,
    )

    origin = (frame.origin, frame.origin)
    value, first = jets[0], jets[1]
    side = _interval_dot(
        _interval_cross((frame.axis, frame.axis), interval_subtract(value, origin)),
        frame.normal,
    )
    positive = (side[0] > 0)[:, None]
    oriented = (
        np.where(positive, frame.normal[0], -frame.normal[1]),
        np.where(positive, frame.normal[1], -frame.normal[0]),
    )
    speed_lower, speed_upper = _interval_norm(first)
    regular = ((side[0] > 0) | (side[1] < 0)) & (speed_lower > 0)
    speed = (speed_lower[:, None], speed_upper[:, None])
    scale = (np.asarray(distance), np.asarray(distance))
    tangent = interval_divide(first, speed)
    result = [
        interval_subtract(
            interval_add(
                value, interval_multiply(scale, _interval_cross(oriented, tangent))
            ),
            origin,
        )
    ]
    if len(jets) < 3:
        return tuple(result), regular
    second = jets[2]
    rate_lower, rate_upper = _interval_dot(tangent, second)
    rate = (rate_lower[:, None], rate_upper[:, None])
    turning = interval_divide(
        interval_subtract(second, interval_multiply(tangent, rate)), speed
    )
    result.append(
        interval_add(first, interval_multiply(scale, _interval_cross(oriented, turning)))
    )
    if len(jets) < 4:
        return tuple(result), regular
    third = jets[3]
    acceleration_lower, acceleration_upper = interval_add(
        _interval_dot(turning, second), _interval_dot(tangent, third)
    )
    acceleration = (acceleration_lower[:, None], acceleration_upper[:, None])
    bending = interval_divide(
        interval_subtract(
            interval_subtract(
                third,
                interval_multiply(
                    turning,
                    interval_multiply((np.asarray(2.0), np.asarray(2.0)), rate),
                ),
            ),
            interval_multiply(tangent, acceleration),
        ),
        speed,
    )
    result.append(
        interval_add(second, interval_multiply(scale, _interval_cross(oriented, bending)))
    )
    return tuple(result), regular


def _rotated_jets(
    axis: np.ndarray,
    boxes: np.ndarray,
    hull: list[tuple[np.ndarray, np.ndarray]],
    order: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Surface jets of ``o + R(u) w(v)`` from row hulls of ``w - o, w', w''``."""
    angle = (boxes[:, 0, 0], boxes[:, 1, 0])
    shape = (boxes.shape[0], 3, 2) + ((2,) if order == 2 else ())
    lower, upper = np.empty(shape), np.empty(shape)
    if order == 1:
        lower[:, :, 0], upper[:, :, 0] = _rotation_jet_bounds(axis, hull[0], angle, 1)
        lower[:, :, 1], upper[:, :, 1] = _rotation_jet_bounds(axis, hull[1], angle, 0)
        return lower, upper
    lower[:, :, 0, 0], upper[:, :, 0, 0] = _rotation_jet_bounds(axis, hull[0], angle, 2)
    mixed = _rotation_jet_bounds(axis, hull[1], angle, 1)
    lower[:, :, 0, 1], upper[:, :, 0, 1] = mixed
    lower[:, :, 1, 0], upper[:, :, 1, 0] = mixed
    lower[:, :, 1, 1], upper[:, :, 1, 1] = _rotation_jet_bounds(axis, hull[2], angle, 0)
    return lower, upper


def _revolution_jet_bounds(
    surface: RevolutionSurface,
    spans: _MeridianSpans,
    boxes: np.ndarray,
    order: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation jets of a B-spline meridian's batched closed-span jets.

    Order two is unbounded on rows meeting a junction without an exact C1
    proof; across a proved C1 junction the one-sided second jets bound the
    Lipschitz first derivative.
    """
    first, last = _revolution_ranges(surface, boxes)
    if not boxes.shape[0]:
        shape = (0, 3, 2) + ((2,) if order == 2 else ())
        return np.zeros(shape), np.zeros(shape)
    origin = np.asarray(surface.axis_origin, dtype=np.float64)
    jets = _meridian_jets(spans, first, last, order, _coordinate_budget())
    position = jets.hull(0)
    hull = [
        (
            np.nextafter(position[0] - origin, -np.inf),
            np.nextafter(position[1] - origin, np.inf),
        ),
        *(jets.hull(derivative) for derivative in range(1, order + 1)),
    ]
    axis = np.asarray(surface.axis_direction, dtype=np.float64)
    lower, upper = _rotated_jets(axis, boxes, hull, order)
    if order == 2:
        rough = _junction_rows(spans, first, last, spans.c1, closed=True)
        lower[rough], upper[rough] = -np.inf, np.inf
    return lower, upper


def _offset_meridian_jet_bounds(
    surface: OffsetSurface,
    frame: _MeridianFrame,
    boxes: np.ndarray,
    order: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation jets of the exact parallel meridian of an offset revolution.

    ``o + R(u)(c - o) + d R(u) n = o + R(u)(q - o)``. Rows whose span hull
    straddles an unproved G1 junction stay unbounded: the offset itself is
    discontinuous there. Order two also requires every met junction to keep
    each parallel curve C1.
    """
    base = surface.base
    if not isinstance(base, RevolutionSurface):
        raise TypeError("A parallel meridian requires a revolution base.")
    first, last = _revolution_ranges(base, boxes)
    shape = (boxes.shape[0], 3, 2) + ((2,) if order == 2 else ())
    if not boxes.shape[0]:
        return np.zeros(shape), np.zeros(shape)
    jets = _meridian_jets(frame.spans, first, last, order + 1, _coordinate_budget())
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        parallel, regular = _parallel_meridian_jets(
            frame, jets.jets, float(surface.distance)
        )
        hull = [
            (
                np.minimum.reduceat(lower, jets.offsets, axis=0),
                np.maximum.reduceat(upper, jets.offsets, axis=0),
            )
            for lower, upper in parallel
        ]
        lower, upper = _rotated_jets(frame.axis, boxes, hull, order)
    usable = np.logical_and.reduceat(regular, jets.offsets) & ~_junction_rows(
        frame.spans, first, last, frame.spans.g1, closed=False
    )
    if order == 2:
        usable &= ~_junction_rows(
            frame.spans, first, last, frame.spans.parallel_c1, closed=True
        )
    lower[~usable], upper[~usable] = -np.inf, np.inf
    return lower, upper


def _revolution_meridian(
    surface: AbstractSurfacePatch, /
) -> tuple[RevolutionSurface, float] | None:
    """A revolution, or a normal offset of one, with its signed distance."""
    if isinstance(surface, RevolutionSurface):
        return surface, 0.0
    if isinstance(surface, OffsetSurface) and isinstance(surface.base, RevolutionSurface):
        return surface.base, float(surface.distance)
    return None


def _meridian_second_jets(
    surface: AbstractSurfacePatch, boxes: np.ndarray, /
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray] | None:
    """Row hulls of ``w - o, w', w''`` of the swept meridian ``w`` of chart boxes.

    ``w`` is a revolution's B-spline meridian, or an offset's exact parallel
    meridian, about an exactly unit axis, so ``|x_uu| = |k x (w - o)|``,
    ``|x_uv| = |k x w'|`` and ``|x_vv| = |w''|`` at every chart point. Returns
    the hulls, the unit axis and the rows whose second jets bound a C1 sweep;
    None when the surface has no such batched meridian structure.
    """
    resolved = _revolution_meridian(surface)
    if resolved is None:
        return None
    revolution, distance = resolved
    budget = _coordinate_budget()
    axis = np.asarray(revolution.axis_direction, dtype=np.float64)
    origin = np.asarray(revolution.axis_origin, dtype=np.float64)
    if isinstance(surface, OffsetSurface):
        frame = _meridian_frame(revolution, budget)
        if frame is None:
            return None
        spans = frame.spans
    else:
        frame = None
        spans = _meridian_spans(revolution.curve, budget)
        if spans is None or not _exact_unit(axis):
            return None
    first, last = _revolution_ranges(revolution, boxes)
    if not boxes.shape[0]:
        empty = (np.zeros((0, 3)), np.zeros((0, 3)))
        return [empty, empty, empty], axis, np.zeros((0,), dtype=np.bool_)
    if frame is None:
        jets = _meridian_jets(spans, first, last, 2, budget)
        position = jets.hull(0)
        hull = [
            (
                np.nextafter(position[0] - origin, -np.inf),
                np.nextafter(position[1] - origin, np.inf),
            ),
            jets.hull(1),
            jets.hull(2),
        ]
        return hull, axis, ~_junction_rows(spans, first, last, spans.c1, closed=True)
    jets = _meridian_jets(spans, first, last, 3, budget)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        parallel, regular = _parallel_meridian_jets(frame, jets.jets, distance)
    hull = [
        (
            np.minimum.reduceat(lower, jets.offsets, axis=0),
            np.maximum.reduceat(upper, jets.offsets, axis=0),
        )
        for lower, upper in parallel
    ]
    smooth = (
        np.logical_and.reduceat(regular, jets.offsets)
        & ~_junction_rows(spans, first, last, spans.g1, closed=False)
        & ~_junction_rows(spans, first, last, spans.parallel_c1, closed=True)
    )
    return hull, axis, smooth


# Equal closed pieces per range of an offset's meridian value enclosure.
_VALUE_PIECES = 16


def _meridian_value_bounds(
    surface: AbstractSurfacePatch, boxes: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray] | None:
    """Batched value boxes of a B-spline-meridian revolution or its offset.

    ``o + R(u)(w - o)`` with ``w`` the meridian, or an offset's exact parallel
    meridian, rotated by the same interval rotation the source evaluates.
    Offset rows whose parallel jets are not regular, or that straddle an
    unproved G1 junction, stay unbounded. None without this structure.
    """
    from .._interval_enclosure import interval_add

    resolved = _revolution_meridian(surface)
    if resolved is None:
        return None
    revolution, distance = resolved
    budget = _coordinate_budget()
    origin = np.asarray(revolution.axis_origin, dtype=np.float64)
    if isinstance(surface, OffsetSurface):
        frame = _meridian_frame(revolution, budget)
        if frame is None:
            return None
        spans, axis = frame.spans, frame.axis
    else:
        frame = None
        spans = _meridian_spans(revolution.curve, budget)
        if spans is None:
            return None
        axis = np.asarray(revolution.axis_direction, dtype=np.float64)
    first, last = _revolution_ranges(revolution, boxes)
    if not boxes.shape[0]:
        return np.zeros((0, 3)), np.zeros((0, 3))
    usable = np.ones((boxes.shape[0],), dtype=np.bool_)
    if frame is None:
        jets = _meridian_jets(spans, first, last, 0, budget)
        position = jets.hull(0)
        relative = (
            np.nextafter(position[0] - origin, -np.inf),
            np.nextafter(position[1] - origin, np.inf),
        )
    else:
        # The unit normal of a whole-span hull is not separated from zero;
        # equal closed pieces covering each range keep every tangent box off
        # the origin. Pieces share computed endpoints and end at ``last``.
        fractions = np.arange(_VALUE_PIECES + 1) / _VALUE_PIECES
        cuts = np.clip(
            first[:, None] + (last - first)[:, None] * fractions,
            first[:, None],
            last[:, None],
        )
        cuts[:, -1] = last
        jets = _meridian_jets(
            spans, cuts[:, :-1].reshape(-1), cuts[:, 1:].reshape(-1), 1, budget
        )
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            parallel, regular = _parallel_meridian_jets(frame, jets.jets, distance)
        pieces = (boxes.shape[0], _VALUE_PIECES, 3)
        relative = (
            np.min(
                np.minimum.reduceat(parallel[0][0], jets.offsets, axis=0).reshape(pieces),
                axis=1,
            ),
            np.max(
                np.maximum.reduceat(parallel[0][1], jets.offsets, axis=0).reshape(pieces),
                axis=1,
            ),
        )
        usable = np.all(
            np.logical_and.reduceat(regular, jets.offsets).reshape(pieces[:2]), axis=1
        ) & ~_junction_rows(spans, first, last, spans.g1, closed=False)
    with np.errstate(invalid="ignore", over="ignore"):
        lower, upper = interval_add(
            (origin, origin),
            _rotation_jet_bounds(axis, relative, (boxes[:, 0, 0], boxes[:, 1, 0]), 0),
        )
    lower[~usable], upper[~usable] = -np.inf, np.inf
    return lower, upper


# Equal parameter ranges covering a meridian for its convex-offset proof.
_CONVEXITY_RANGES = 256


@dataclass(frozen=True, slots=True)
class _ConvexOffsetEvidence:
    """Premises of the convex-meridian offset embedding proof, each proved or not.

    ``closed_g1``: every meridian junction is exactly G1 and the chart is glued
    in both directions. ``one_turn``: the signed curvature enclosure has one
    strict sign over the cover and the enclosed total turning is below
    ``4 pi``, so the closed locally convex meridian turns exactly once
    (``turning_sign`` is its orientation) and is simple and convex.
    ``forward``: ``q_t' . c' > 0`` over the cover for every ``t`` in [0, 1]
    (``q_t'`` is affine in ``t``; ``t = 1`` is enclosed), so each ``q_t`` keeps
    ``c``'s strictly monotone tangent angle and is simple and convex; inward
    this is ``|d| kappa_max < 1``. ``layered``: within reach
    ``dist(q_t, c) = t |d|``, so distinct layers are nested and disjoint;
    it follows from the previous premises. ``off_axis``: ``c`` and ``q_1``
    lie strictly on one side of the axis, hence every ``q_t`` does, and the
    rotation embeds each layer.
    """

    closed_g1: bool
    one_turn: bool
    turning_sign: int
    forward: bool
    layered: bool
    off_axis: bool

    @property
    def proved(self) -> bool:
        return (
            self.closed_g1
            and self.one_turn
            and self.forward
            and self.layered
            and self.off_axis
        )


def _convex_meridian_offset(
    surface: OffsetSurface, glued: tuple[bool, bool], /
) -> _ConvexOffsetEvidence:
    """Prove every ``F_t``, ``t`` in [0, 1], embeds as a revolved convex layer.

    See ``_ConvexOffsetEvidence``; any unproved premise leaves the caller's
    box route in charge. All enclosures come from the batched exact meridian
    hodographs over ``_CONVEXITY_RANGES`` closed ranges covering the domain.
    """
    from .._interval_enclosure import interval_subtract

    unproved = _ConvexOffsetEvidence(False, False, 0, False, False, False)
    base = surface.base
    if not isinstance(base, RevolutionSurface):
        return unproved
    budget = _coordinate_budget()
    frame = _meridian_frame(base, budget)
    if frame is None:
        return unproved
    spans = frame.spans
    closed_g1 = bool(np.all(spans.g1)) and glued == (True, True)
    edges = np.linspace(spans.starts[0], spans.ends[-1], _CONVEXITY_RANGES + 1)
    jets = _meridian_jets(spans, edges[:-1], edges[1:], 2, budget, bending=True)
    value, velocity, _, bending = jets.jets
    origin, axis = (frame.origin, frame.origin), (frame.axis, frame.axis)
    eps = np.finfo(np.float64).eps
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        signed = _interval_dot(bending, frame.normal)
        speed = _interval_norm(velocity)[0]
        rate = np.nextafter(
            _interval_norm(bending)[1] / np.nextafter(speed * speed, -np.inf), np.inf
        )
        # Upper total turning: per-range tangent-angle rates times widths,
        # with a margin for the binary64 width differences and summation.
        turning = np.nextafter(
            np.sum(np.nextafter(rate * jets.widths, np.inf))
            * (1 + 4 * _CONVEXITY_RANGES * eps),
            np.inf,
        )
        parallel, regular = _parallel_meridian_jets(
            frame, jets.jets[:3], float(surface.distance)
        )
        forward = _interval_dot(parallel[1], velocity)[0]
        sides = [
            _interval_dot(_interval_cross(axis, relative), frame.normal)
            for relative in (interval_subtract(value, origin), parallel[0])
        ]
    sign = 1 if np.all(signed[0] > 0) else -1 if np.all(signed[1] < 0) else 0
    one_turn = sign != 0 and bool(np.all(speed > 0)) and bool(turning < 4 * np.pi)
    forward_proved = bool(np.all(regular)) and bool(np.all(forward > 0))
    off_axis = all(bool(np.all(side[0] > 0)) for side in sides) or all(
        bool(np.all(side[1] < 0)) for side in sides
    )
    return _ConvexOffsetEvidence(
        closed_g1,
        one_turn,
        sign,
        forward_proved,
        closed_g1 and one_turn and forward_proved,
        off_axis,
    )


def _cone_diameters(lower: np.ndarray, upper: np.ndarray, /) -> np.ndarray:
    """Outward angular diameters of the direction sets of vector boxes ``(..., 3)``.

    A box lies in the ball about its center of radius half its diagonal, and
    every point is at least the box's origin distance away; the ball subtends
    at most ``2 asin(radius / distance)``. Unseparated or unbounded boxes are
    infinite, as in the per-box Gauss-set diameter.
    """
    separated = np.maximum(np.maximum(lower, -upper), 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        distance = _directed_sum_of_squares(separated, -np.inf)
        diameter = _directed_sum_of_squares(np.nextafter(upper - lower, np.inf), np.inf)
        ratio = np.nextafter(diameter / np.nextafter(4.0 * distance, -np.inf), np.inf)
        angle = 2.0 * np.arcsin(
            np.minimum(1.0, np.nextafter(np.sqrt(np.minimum(ratio, 1.0)), np.inf))
        )
    angle = np.nextafter(angle * (1 + 32 * np.finfo(np.float64).eps), np.inf)
    angle = np.where(ratio >= 1.0, np.nextafter(np.pi, np.inf), angle)
    finite = np.all(np.isfinite(lower) & np.isfinite(upper), axis=-1)
    return np.where(finite & (distance > 0), angle, np.inf)


def _meridian_normal_turns(
    revolution: RevolutionSurface, distance: float, charts: np.ndarray, /
) -> np.ndarray | None:
    """Gauss-set diameters over chart triangles of a planar-meridian revolution.

    Off the axis the unit normal is ``R(u) n(v)`` with ``n`` the in-plane
    meridian normal. Its chart derivatives ``R(u) (k x n)`` and ``R(u) n'``
    are orthogonal, with ``|k x n| = |t.k| <= sup|c'.k| / inf|c'|`` and
    ``|n'| = |c' x c''| / |c'|^2``. Across proved G1 junctions the Gauss image
    of a chart segment is therefore no longer than ``sqrt((B du)^2 + (A dv)^2)``
    with those suprema, a norm of the segment, so a triangle's diameter is at
    most its longest edge's. Independently it is at most the tangent turn (the
    angular diameter of the one-sided first-jet hull, or across G1 junctions
    the integral of ``A``) plus ``B du``. An offset within reach (``r > |d|``
    and ``|d| sup|c' x c''| / inf|c'|^3 < 1``) keeps the same oriented Gauss
    map. NaN marks rows this structure cannot certify; None, a profile without
    an exactly planar meridian frame.
    """
    from .._interval_enclosure import interval_subtract

    budget = _coordinate_budget()
    frame = _meridian_frame(revolution, budget)
    if frame is None:
        return None
    boxes = np.stack((np.min(charts, axis=1), np.max(charts, axis=1)), axis=1)
    first, last = _revolution_ranges(revolution, boxes)
    result = np.full((boxes.shape[0],), np.nan)
    if not boxes.shape[0]:
        return result
    # The exact bending hodograph encloses c' x c'' without the c'' jet.
    jets = _meridian_jets(frame.spans, first, last, 1, budget, bending=True)
    value, velocity, cross = jets.jets
    axis = (frame.axis, frame.axis)
    eps = np.finfo(np.float64).eps
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        side = _interval_dot(
            _interval_cross(axis, interval_subtract(value, (frame.origin, frame.origin))),
            frame.normal,
        )
        radius = np.where(side[0] > 0, side[0], -side[1])
        speed = _interval_norm(velocity)[0]
        regular = ((side[0] > 0) | (side[1] < 0)) & (speed > 0)
        square = np.nextafter(speed * speed, -np.inf)
        bending = _interval_norm(cross)[1]
        rate = np.nextafter(bending / square, np.inf)
        axial = _interval_dot(velocity, axis)
        spin = np.minimum(
            1.0,
            np.nextafter(np.maximum(np.abs(axial[0]), np.abs(axial[1])) / speed, np.inf),
        )
        curvature = np.nextafter(bending / np.nextafter(square * speed, -np.inf), np.inf)
        turn = np.nextafter(rate * jets.widths, np.inf)
        offsets = jets.offsets
        overlaps = np.diff(np.append(offsets, jets.rows.size))
        integral = np.nextafter(
            np.add.reduceat(turn, offsets) * (1 + overlaps * eps), np.inf
        )
        smooth = ~_junction_rows(frame.spans, first, last, frame.spans.g1, closed=False)
        cone = _cone_diameters(*jets.hull(1))
        tangent = np.where(smooth, np.minimum(cone, integral), cone)
        rotation = np.maximum.reduceat(spin, offsets)
        total = np.nextafter(
            tangent + np.nextafter(rotation * (boxes[:, 1, 0] - boxes[:, 0, 0]), np.inf),
            np.inf,
        )
        edges = charts - np.roll(charts, 1, axis=1)
        steepest = np.maximum.reduceat(rate, offsets)
        path = np.max(
            np.hypot(
                steepest[:, None] * edges[..., 1], rotation[:, None] * edges[..., 0]
            ),
            axis=1,
        )
        path = np.nextafter(path * (1 + 16 * eps), np.inf)
    total = np.where(smooth, np.minimum(total, path), total)
    admitted = np.logical_and.reduceat(regular, offsets)
    if distance:
        admitted &= (
            smooth
            & (np.minimum.reduceat(radius, offsets) > abs(distance))
            & (abs(distance) * np.maximum.reduceat(curvature, offsets) < 1.0)
        )
    result[admitted] = total[admitted]
    return result


class RevolutionSurface(AbstractSurfacePatch):
    """Revolution of a space curve: ``origin + R(u, axis) (curve(v) - origin)``."""

    curve: AbstractCurve
    axis_origin: Array
    axis_direction: Array

    def __init__(
        self,
        curve: AbstractCurve,
        axis_origin: ConvertibleToArray,
        axis_direction: ConvertibleToArray,
    ) -> None:
        curve_ = _require_space_curve(curve, "curve")
        origin = _host_vector(axis_origin, "axis_origin")
        direction = _host_vector(axis_direction, "axis_direction")
        if origin.size != 3 or direction.size != 3:
            raise ValueError("A revolution axis must be three-dimensional.")
        if abs(np.linalg.norm(direction) - 1.0) > _FRAME_TOLERANCE:
            raise ValueError("A revolution axis direction must be a unit vector.")
        self.curve = curve_
        self.axis_origin = jnp.asarray(origin)
        self.axis_direction = jnp.asarray(direction)

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return _TWO_PI, self.curve.period

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.axis_origin.dtype)
        angle = parameters_[..., 0:1]
        relative = _curve_parameter(self.curve, parameters_[..., 1]) - self.axis_origin
        axis = jnp.broadcast_to(self.axis_direction, relative.shape)
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        axial = jnp.sum(relative * axis, axis=-1, keepdims=True)
        rotated = (
            relative * cosine
            + jnp.cross(axis, relative) * sine
            + axis * axial * (1.0 - cosine)
        )
        return self.axis_origin + rotated

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = super().validate_parameter_box(parameter_box)
        self.curve.validate_query_range(box[0, 1], box[1, 1])
        return box

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        profile = self.curve.bounding_box(
            *_curve_box_range(self.curve, box[0, 1], box[1, 1])
        )
        origin = np.asarray(self.axis_origin)
        axis = np.asarray(self.axis_direction)
        corners = (
            np.stack(np.meshgrid(*profile.T, indexing="ij"), axis=-1).reshape(-1, 3)
            - origin
        )
        axial = corners @ axis
        # The radius is convex over the profile box: its maximum is a corner.
        radius = float(np.max(np.linalg.norm(corners - axial[:, None] * axis, axis=1)))
        first, second = _axis_frame(axis)
        return _interval_box(
            origin,
            (
                (axis, (float(np.min(axial)), float(np.max(axial)))),
                (first, (-radius, radius)),
                (second, (-radius, radius)),
            ),
        )

    def degenerate_isolines(
        self, parameter_box: ConvertibleToArray, /
    ) -> tuple[tuple[int, float], ...]:
        box = _parameter_box(parameter_box)
        ends = np.asarray(
            _curve_parameter(self.curve, jnp.asarray(box[:, 1])), dtype=np.float64
        )
        relative = ends - np.asarray(self.axis_origin)
        axis = np.asarray(self.axis_direction)
        radial = np.linalg.norm(relative - (relative @ axis)[:, None] * axis, axis=1)
        scale = max(1.0, float(np.max(np.abs(ends))))
        on_axis = radial <= 1.0e-12 * scale
        return tuple((1, float(box[row, 1])) for row in range(2) if bool(on_axis[row]))


class RuledSurface(AbstractSurfacePatch):
    """Ruled surface ``(1 - v) * first(u) + v * second(u)`` with ``v`` in ``[0, 1]``."""

    first: AbstractCurve
    second: AbstractCurve

    def __init__(self, first: AbstractCurve, second: AbstractCurve) -> None:
        self.first = _require_space_curve(first, "first")
        self.second = _require_space_curve(second, "second")

    @property
    def periods(self) -> tuple[float | None, float | None]:
        period = self.first.period
        return (period if period == self.second.period else None), None

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=jnp.float64)
        blend = parameters_[..., 1:2]
        return (1.0 - blend) * _curve_parameter(
            self.first, parameters_[..., 0]
        ) + blend * _curve_parameter(self.second, parameters_[..., 0])

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = super().validate_parameter_box(parameter_box)
        if box[0, 1] < 0.0 or box[1, 1] > 1.0:
            raise ValueError("A ruled surface blend parameter must lie in [0, 1].")
        self.first.validate_query_range(box[0, 0], box[1, 0])
        self.second.validate_query_range(box[0, 0], box[1, 0])
        return box

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        # Each point is a convex combination of the two rails.
        first = self.first.bounding_box(box[0, 0], box[1, 0])
        second = self.second.bounding_box(box[0, 0], box[1, 0])
        return np.stack(
            (np.minimum(first[0], second[0]), np.maximum(first[1], second[1]))
        )


class OffsetSurface(AbstractSurfacePatch):
    """Exact signed normal-offset operation tree with bounded numerical jets."""

    base: AbstractSurfacePatch
    distance: Array
    sphere_normal_depth: int = eqx.field(static=True)

    def __init__(self, base: AbstractSurfacePatch, distance: ConvertibleToArray) -> None:
        if not isinstance(base, AbstractSurfacePatch):
            raise TypeError("An offset requires a native surface patch.")
        self.base = base
        self.distance = jnp.asarray(_host_scalar(distance, "distance"), dtype=jnp.float64)
        equivalence = sphere_source_equivalence(base)
        depth = 0
        if equivalence is not None and equivalence[1] != 0:
            original = base
            while isinstance(original, OffsetSurface):
                depth += 1
                original = original.base
        self.sphere_normal_depth = depth

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.base.periods

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.base.validate_parameter_box(parameter_box)
        sphere = sphere_source_equivalence(self)
        if sphere is not None and sphere[1] == 0:
            raise ValueError("The normal-offset source tree collapses the sphere rank.")
        if (
            float(self.distance) != 0.0
            and isinstance(self.base, ConePatch)
            and self.base.degenerate_isolines(box)
        ):
            raise ValueError("A normal offset is not uniquely defined at a cone apex.")
        base = self.base
        if float(self.distance) != 0 and isinstance(base, TorusPatch):
            cosine, _ = _trig_ranges(box[0, 1], box[1, 1])
            radius = (
                float(base.major_radius) + float(base.minor_radius) * cosine[0],
                float(base.major_radius) + float(base.minor_radius) * cosine[1],
            )
            if radius[0] <= 0 <= radius[1]:
                raise ValueError(
                    "A torus normal-offset chart meets a base horn/spindle singularity."
                )
        if isinstance(base, (SpherePatch, CylinderPatch, TorusPatch)):
            axes = [
                [Fraction(float(value)) for value in np.asarray(axis)]
                for axis in (base.first_axis, base.second_axis, base.axis)
            ]
            unit = all(
                sum(a * b for a, b in zip(axes[i], axes[j], strict=True))
                == (1 if i == j else 0)
                for i in range(3)
                for j in range(3)
            )
            if unit:
                orientation = Fraction(
                    float(
                        np.dot(
                            np.cross(
                                np.asarray(base.first_axis), np.asarray(base.second_axis)
                            ),
                            np.asarray(base.axis),
                        )
                    )
                )
                original = (
                    float(base.minor_radius)
                    if isinstance(base, TorusPatch)
                    else float(base.radius)
                )
                radius = Fraction(original) + orientation * Fraction(float(self.distance))
                if radius == 0:
                    raise ValueError(
                        "The normal offset collapses the source surface rank."
                    )
                if isinstance(base, TorusPatch):
                    cosine, _ = _trig_ranges(box[0, 1], box[1, 1])
                    values = (
                        Fraction(float(base.major_radius)) + radius * Fraction(cosine[0]),
                        Fraction(float(base.major_radius)) + radius * Fraction(cosine[1]),
                    )
                    if min(values) <= 0 <= max(values):
                        raise ValueError(
                            "The normal-offset torus chart meets a new horn/spindle singularity."
                        )
        return box

    def degenerate_isolines(
        self, parameter_box: ConvertibleToArray, /
    ) -> tuple[tuple[int, float], ...]:
        return self.base.degenerate_isolines(parameter_box)

    def is_c1_on(self, parameter_box: ConvertibleToArray, /) -> bool:
        return self.base.is_c1_on(parameter_box)

    def _normal(self, parameters: Array) -> Array:
        base = self.base
        for _ in range(self.sphere_normal_depth):
            if not isinstance(base, OffsetSurface):
                raise ValueError(
                    "A source sphere-normal proof has changed operation-tree structure."
                )
            base = base.base
        if isinstance(base, PlanePatch):
            normal = jnp.cross(base.first_axis, base.second_axis)
            return normal / jnp.linalg.norm(normal)
        if isinstance(base, (CylinderPatch, SpherePatch, TorusPatch)):
            first = jnp.cross(base.second_axis, base.axis)
            second = jnp.cross(base.axis, base.first_axis)
            if isinstance(base, CylinderPatch):
                coordinates = jnp.stack((jnp.cos(parameters[0]), jnp.sin(parameters[0])))
                vectors = jnp.stack((first, second), axis=1)
            else:
                third = jnp.cross(base.first_axis, base.second_axis)
                angle = parameters[1]
                coordinates = jnp.stack(
                    (
                        jnp.cos(angle) * jnp.cos(parameters[0]),
                        jnp.cos(angle) * jnp.sin(parameters[0]),
                        jnp.sin(angle),
                    )
                )
                vectors = jnp.stack((first, second, third), axis=1)
            normal = vectors @ coordinates
            gram = vectors.T @ vectors
            # Unit angular-coordinate identities avoid spurious zero lower
            # bounds in a full-angle normal denominator. The stored frame is
            # still enclosed; tolerance admission is not exact orthogonality.
            norm_squared = gram[0, 0]
            for index in range(1, coordinates.shape[0]):
                norm_squared = (
                    norm_squared
                    + (gram[index, index] - gram[0, 0]) * coordinates[index] ** 2
                )
            for i in range(coordinates.shape[0]):
                for j in range(i + 1, coordinates.shape[0]):
                    norm_squared = (
                        norm_squared + 2 * gram[i, j] * coordinates[i] * coordinates[j]
                    )
            if isinstance(base, TorusPatch):
                normal = normal * jnp.sign(
                    base.major_radius + base.minor_radius * jnp.cos(parameters[1])
                )
            return normal / jnp.sqrt(norm_squared)
        frame = jax.jacfwd(base.evaluate)(parameters)
        normal = jnp.cross(frame[:, 0], frame[:, 1])
        return normal / jnp.linalg.norm(normal)

    def _evaluate_one(self, parameters: Array) -> Array:
        value = self.base.evaluate(parameters)
        return jnp.where(
            self.distance == 0, value, value + self.distance * self._normal(parameters)
        )

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        if values.ndim == 1:
            return self._evaluate_one(values)
        return jax.vmap(self._evaluate_one)(values.reshape((-1, 2))).reshape(
            (*values.shape[:-1], 3)
        )

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = self.validate_parameter_box(parameter_box)
        from .._interval_enclosure import (
            interval_add,
            interval_multiply,
            interval_subtract,
        )

        sphere = sphere_source_equivalence(self)
        if sphere is not None:
            original, exact_radius = sphere
            source_box = original.bounding_box(box)
            center = np.asarray(original.center)
            relative = interval_subtract((source_box[0], source_box[1]), (center, center))
            ratio = exact_radius / Fraction(float(original.radius))
            represented = float(ratio)
            lower = (
                float(np.nextafter(represented, -np.inf))
                if Fraction(represented) > ratio
                else represented
            )
            upper = (
                float(np.nextafter(represented, np.inf))
                if Fraction(represented) < ratio
                else represented
            )
            image = interval_add(
                interval_multiply(relative, (np.asarray(lower), np.asarray(upper))),
                (center, center),
            )
            return np.stack(image)
        equivalent = self.analytic_equivalent()
        if equivalent is not None:
            return equivalent.bounding_box(box)
        # A revolution offset rotates its exact parallel meridian; interval AD
        # of the normalized normal is unbounded over a full rotation.
        swept = _meridian_value_bounds(self, box[None])
        if swept is not None:
            return np.stack((swept[0][0], swept[1][0]))
        from .._interval_enclosure import prepare_interval_function
        from ._intersection_curve import coefficient_enclosures, surface_pieces_for_box

        lower_bounds, upper_bounds = [], []
        for piece in surface_pieces_for_box(self, box):
            prepared = prepare_interval_function(
                piece.evaluator.evaluate,
                2,
                batch_capacity=1,
                constant_bounds=coefficient_enclosures(piece.evaluator),
            )
            lower, upper = prepared.evaluate(piece.lower[None], piece.upper[None])
            lower_bounds.append(lower[0])
            upper_bounds.append(upper[0])
        return np.stack((np.min(lower_bounds, axis=0), np.max(upper_bounds, axis=0)))

    def analytic_equivalent(self) -> AbstractSurfacePatch | None:
        """Lower only when the existing float carrier represents the SAME map."""
        base = self.base
        distance = Fraction(float(self.distance))
        sphere = sphere_source_equivalence(self)
        if sphere is not None:
            original, exact_radius = sphere
            represented = float(exact_radius)
            if exact_radius > 0 and Fraction(represented) == exact_radius:
                return SpherePatch(
                    original.center,
                    original.first_axis,
                    original.second_axis,
                    original.axis,
                    represented,
                )
            return None
        if distance == 0:
            return base
        if isinstance(base, (CylinderPatch, SpherePatch, TorusPatch)):
            vectors = [
                np.asarray(base.first_axis),
                np.asarray(base.second_axis),
                np.asarray(base.axis),
            ]
            rational = [
                [Fraction(float(value)) for value in vector] for vector in vectors
            ]
            if any(
                sum(a * b for a, b in zip(rational[i], rational[j], strict=True))
                != (1 if i == j else 0)
                for i in range(3)
                for j in range(3)
            ):
                return None
            u, v, w = rational
            orientation = (
                u[0] * (v[1] * w[2] - v[2] * w[1])
                - u[1] * (v[0] * w[2] - v[2] * w[0])
                + u[2] * (v[0] * w[1] - v[1] * w[0])
            )
            if isinstance(base, TorusPatch):
                # A ring torus keeps one normal sense, so its offset is the
                # same longitude/tube map with an exactly shifted tube radius;
                # the offset must itself remain a ring torus.
                major = Fraction(float(base.major_radius))
                minor = Fraction(float(base.minor_radius))
                radius = minor + orientation * distance
                if major <= minor or major <= radius:
                    return None
            else:
                radius = Fraction(float(base.radius)) + orientation * distance
            represented = float(radius)
            if radius <= 0 or Fraction(represented) != radius:
                return None
            if isinstance(base, TorusPatch):
                return TorusPatch(
                    base.center,
                    base.first_axis,
                    base.second_axis,
                    base.axis,
                    base.major_radius,
                    represented,
                )
            if isinstance(base, CylinderPatch):
                return CylinderPatch(
                    base.origin, base.first_axis, base.second_axis, base.axis, represented
                )
            return SpherePatch(
                base.center, base.first_axis, base.second_axis, base.axis, represented
            )
        if isinstance(base, PlanePatch):
            a, b = [
                [Fraction(float(value)) for value in np.asarray(vector)]
                for vector in (base.first_axis, base.second_axis)
            ]
            normal = (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )
            norm_squared = sum(value * value for value in normal)
            numerator, denominator = (
                isqrt(norm_squared.numerator),
                isqrt(norm_squared.denominator),
            )
            if (
                numerator * numerator != norm_squared.numerator
                or denominator * denominator != norm_squared.denominator
            ):
                return None
            norm = Fraction(numerator, denominator)
            if norm == 0:
                return None
            origin = [
                Fraction(float(value)) + distance * component / norm
                for value, component in zip(np.asarray(base.origin), normal, strict=True)
            ]
            represented = np.asarray([float(value) for value in origin])
            if any(
                Fraction(float(value)) != exact
                for value, exact in zip(represented, origin, strict=True)
            ):
                return None
            return PlanePatch(represented, base.first_axis, base.second_axis)
        return None


def sphere_source_equivalence(
    source: AbstractSurfacePatch,
    /,
) -> tuple[SpherePatch, Fraction] | None:
    """Prove an original sphere/normal-offset tree's signed radial expression.

    A nonzero offset is reduced only for an exactly orthogonal binary source
    frame; tolerance admission is never treated as exact orthogonality. Radius
    sums remain Fractions even when no binary64 primitive could represent them.
    No caller is licensed to replace the original operation tree or its IDs.
    """
    offsets = []
    while isinstance(source, OffsetSurface):
        offsets.append(Fraction(float(source.distance)))
        source = source.base
    if not isinstance(source, SpherePatch):
        return None
    radius = Fraction(float(source.radius))
    if not any(offsets):
        return source, radius
    columns = tuple(
        tuple(Fraction(float(value)) for value in np.asarray(axis))
        for axis in (source.first_axis, source.second_axis, source.axis)
    )
    if any(
        sum(
            (
                first * second
                for first, second in zip(columns[i], columns[j], strict=True)
            ),
            Fraction(0),
        )
        != int(i == j)
        for i in range(3)
        for j in range(3)
    ):
        return None
    matrix = tuple(tuple(columns[column][row] for column in range(3)) for row in range(3))
    orientation = prepare_exact_small_linear_actions(matrix, matrix)
    if not orientation.successful:
        return None
    for offset in reversed(offsets):
        if radius == 0 and offset:
            return None
        radius += orientation.determinant * offset
    return source, radius


def sphere_source_radius_terms(
    source: AbstractSurfacePatch,
    /,
) -> tuple[SpherePatch, tuple[Fraction, ...]] | None:
    """Original radial coefficient operations, without rounding their exact sum."""
    equivalence = sphere_source_equivalence(source)
    if equivalence is None:
        return None
    original, _ = equivalence
    offsets = []
    while isinstance(source, OffsetSurface):
        offsets.append(Fraction(float(source.distance)))
        source = source.base
    terms = [Fraction(float(original.radius))]
    if any(offsets):
        matrix = tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in np.column_stack(
                (
                    np.asarray(original.first_axis),
                    np.asarray(original.second_axis),
                    np.asarray(original.axis),
                )
            )
        )
        proof = prepare_exact_small_linear_actions(matrix, matrix)
        if not proof.successful:
            return None
        terms.extend(proof.determinant * offset for offset in reversed(offsets))
    return original, tuple(terms)


def sphere_source_pole_normal_limits(
    source: AbstractSurfacePatch,
    parameters: ConvertibleToArray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical Gauss limits at the source's actual declared spherical poles."""
    charts = np.asarray(parameters, dtype=np.float64).reshape((-1, 2))
    normals = np.zeros((charts.shape[0], 3), dtype=np.float64)
    supported = np.zeros(charts.shape[0], dtype=np.bool_)
    equivalence = sphere_source_equivalence(source)
    if equivalence is None or equivalence[1] == 0:
        return normals, supported
    original, _ = equivalence
    supported = (charts[:, 1] == -0.5 * np.pi) | (charts[:, 1] == 0.5 * np.pi)
    normal = np.cross(np.asarray(original.first_axis), np.asarray(original.second_axis))
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length == 0:
        return normals, np.zeros(charts.shape[0], dtype=np.bool_)
    normals[supported] = np.sign(charts[supported, 1])[:, None] * normal[None] / length
    return normals, supported


class SurfaceIsoparametricCurve(AbstractCurve):
    """Exact source seam/isoline: one surface coordinate fixed, the other is t."""

    surface: AbstractSurfacePatch
    fixed_axis: int = eqx.field(static=True)
    fixed_value: float = eqx.field(static=True)
    parameter_range: tuple[float, float] | None = eqx.field(static=True)

    def __init__(
        self,
        surface: AbstractSurfacePatch,
        fixed_axis: int,
        fixed_value: float,
        /,
        *,
        parameter_range: tuple[float, float] | None = None,
    ) -> None:
        if (
            not isinstance(surface, AbstractSurfacePatch)
            or type(fixed_axis) is not int
            or fixed_axis not in (0, 1)
            or not np.isfinite(fixed_value)
        ):
            raise ValueError(
                "An isoline requires a native surface, axis zero/one and finite fixed coordinate."
            )
        if parameter_range is not None:
            parameter_range = _range(*parameter_range)
        self.surface, self.fixed_axis, self.fixed_value = (
            surface,
            fixed_axis,
            float(fixed_value),
        )
        self.parameter_range = parameter_range

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def period(self) -> float | None:
        return self.surface.periods[1 - self.fixed_axis]

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        if self.parameter_range is not None:
            return self.parameter_range
        if self.period is not None:
            return 0.0, self.period
        surface = self.surface
        while isinstance(surface, OffsetSurface):
            surface = surface.base
        if isinstance(surface, BSplineSurfacePatch):
            knots, degree = (
                (surface.u_knots, surface.u_degree)
                if self.fixed_axis == 1
                else (surface.v_knots, surface.v_degree)
            )
            values = np.asarray(knots)
            return float(values[degree]), float(values[-degree - 1])
        if isinstance(surface, SpherePatch) and self.fixed_axis == 0:
            return -pi / 2, pi / 2
        return None

    def p_curve(self) -> LineCurve:
        origin, direction = np.zeros(2), np.zeros(2)
        origin[self.fixed_axis] = self.fixed_value
        direction[1 - self.fixed_axis] = 1.0
        return LineCurve(origin, direction)

    def parameter_box(self, first: float, last: float, /) -> np.ndarray:
        lower, upper = np.empty(2), np.empty(2)
        lower[self.fixed_axis] = upper[self.fixed_axis] = self.fixed_value
        lower[1 - self.fixed_axis], upper[1 - self.fixed_axis] = first, last
        return self.surface.validate_parameter_box(np.stack((lower, upper)))

    def _evaluate_one(self, parameter: Array) -> Array:
        coordinates = (
            jnp.stack((jnp.asarray(self.fixed_value), parameter))
            if self.fixed_axis == 0
            else jnp.stack((parameter, jnp.asarray(self.fixed_value)))
        )
        return self.surface.evaluate(coordinates)

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        if values.ndim == 0:
            return self._evaluate_one(values)
        return jax.vmap(self._evaluate_one)(values.reshape(-1)).reshape(
            (*values.shape, 3)
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        self.validate_query_range(first, last)
        return self.surface.bounding_box(self.parameter_box(first, last))

    def is_c1_on(self, first: float, last: float, /) -> bool:
        return self.surface.is_c1_on(self.parameter_box(first, last))

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Isoline jets are the free-axis surface jets over its degenerate box.

        A swept planar meridian (a revolution or its offset) encloses them with
        its batched exact meridian jets; generic interval AD of an offset's
        normalized normal loses all dependency and does not converge.
        """
        if (
            isinstance(order, bool)
            or order not in (1, 2)
            or _revolution_meridian(self.surface) is None
        ):
            return super().derivative_bounds(first, last, order=order)
        box = self.parameter_box(first, last)
        lower, upper = self.surface.derivative_bounds_batch(box[None], order=order)
        free = 1 - self.fixed_axis
        if order == 1:
            return lower[0, :, free], upper[0, :, free]
        return lower[0, :, free, free], upper[0, :, free, free]


SurfacePatch: TypeAlias = (
    PlanePatch
    | CylinderPatch
    | ConePatch
    | SpherePatch
    | TorusPatch
    | BSplineSurfacePatch
    | ExtrusionSurface
    | RevolutionSurface
    | RuledSurface
    | OffsetSurface
)
Curve: TypeAlias = (
    LineCurve
    | CircleCurve
    | EllipseCurve
    | ParabolaCurve
    | HyperbolaCurve
    | OffsetCurve
    | BSplineCurve
    | SurfaceIsoparametricCurve
)


def surface_differential(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    parameters_ = jnp.asarray(parameters, dtype=jnp.float64)
    leading = parameters_.shape[:-1]
    values = jax.vmap(jax.jacfwd(lambda uv: patch.evaluate(uv)))(
        parameters_.reshape((-1, 2))
    )
    return values.reshape((*leading, 3, 2))


def surface_jacobian(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    differential = surface_differential(patch, parameters)
    return jnp.linalg.norm(
        jnp.cross(differential[..., :, 0], differential[..., :, 1]), axis=-1
    )


def surface_normal(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    differential = surface_differential(patch, parameters)
    normal = jnp.cross(differential[..., :, 0], differential[..., :, 1])
    return normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)


__all__ = [
    "AbstractCurve",
    "AbstractSurfacePatch",
    "BSplineCurve",
    "BSplineSurfacePatch",
    "CircleCurve",
    "ConePatch",
    "Curve",
    "CylinderPatch",
    "EllipseCurve",
    "ExtrusionSurface",
    "LineCurve",
    "PlanePatch",
    "OffsetSurface",
    "SurfaceIsoparametricCurve",
    "RationalBezierPiece",
    "RevolutionSurface",
    "RuledSurface",
    "SpherePatch",
    "SurfacePatch",
    "TorusPatch",
    "surface_differential",
    "surface_jacobian",
    "surface_normal",
]
