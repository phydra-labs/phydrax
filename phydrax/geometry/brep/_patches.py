#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic and rational curve and surface carriers of native B-Reps.

Evaluation is pure JAX so that realized geometry stays differentiable at a fixed
epoch. Parameter-domain validation, seam periods, collapsed (pole) isolines,
Bezier extraction and conservative bounding boxes are host-side preparation
data: bounding boxes use trigonometric extrema, interval arithmetic and
the convex-hull property of positively weighted rational Bernstein controls,
widened by a small outward rounding margin.
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, comb, floor, isfinite, isqrt, pi
from typing import Iterator, Literal, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

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
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    RationalEnclosureError,
)
from ...linalg._small_batched import prepare_exact_small_linear_actions
from ...typing import checked, ConvertibleToArray
from .._interval_enclosure import (
    device_interval_evaluation_error,
    Interval,
    interval_add,
    interval_cos,
    interval_divide,
    interval_multiply,
    interval_sin,
    interval_subtract,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._rational import fraction_down, fraction_up, rational_cross, rational_vector
from ._carrier_pieces import (
    child_coefficient_enclosures,
    CoefficientEnclosureSource,
    CurveCoordinateResult,
    CurvePiece,
    RootCurveCoordinateSource,
    SurfacePiece,
)
from ._coordinate_request import coordinate_source_request
from ._exact import square_root_interval
from ._interval_derivatives import prepare_derivative_interval
from ._request_preparations import TrimRequestPreparations
from ._spline_box import device_bspline_surface_bounds


_TWO_PI = 2.0 * pi
_FRAME_TOLERANCE = 1.0e-10


def _array_interval_bounds(value: Interval, /) -> tuple[np.ndarray, np.ndarray]:
    """Materialize numeric array-bound endpoints as float64 without changing rank.

    Existing float64 arrays remain the same arrays. This boundary is used for
    coordinate/coefficient tensors, not the interval interpreter's scalar or
    boolean leaves.
    """
    return (
        np.asarray(value[0], dtype=np.float64),
        np.asarray(value[1], dtype=np.float64),
    )


def _host_vector(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size not in (2, 3) or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"carrier.vector must be finite with two or three components; got {name}={vector.tolist()}."
        )
    return vector


def _host_scalar(value: ConvertibleToArray, name: str, /) -> float:
    scalar = np.asarray(value, dtype=np.float64)
    if scalar.size != 1 or not np.isfinite(scalar.reshape(-1)[0]):
        raise ValueError(
            f"carrier.scalar must contain one finite value; got {name} with shape {scalar.shape} and values {scalar}."
        )
    return float(scalar.reshape(-1)[0])


def _orthonormal_pair(
    first: np.ndarray, second: np.ndarray, name: str, /
) -> tuple[np.ndarray, np.ndarray]:
    if (
        abs(np.linalg.norm(first) - 1.0) > _FRAME_TOLERANCE
        or abs(np.linalg.norm(second) - 1.0) > _FRAME_TOLERANCE
        or abs(float(first @ second)) > _FRAME_TOLERANCE
    ):
        raise ValueError(
            f"carrier.frame must contain orthonormal axes within {_FRAME_TOLERANCE}; got {name} axes {first.tolist()} and {second.tolist()}."
        )
    return first, second


def _check_frame(first: Array, second: Array, axis: Array, /) -> None:
    a, b, c = (np.asarray(vector, dtype=np.float64) for vector in (first, second, axis))
    if any(
        vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in (a, b, c)
    ):
        raise ValueError(
            f"patch.frame must contain finite three-dimensional axes; got shapes {tuple(vector.shape for vector in (a, b, c))} and values {(a, b, c)}."
        )
    _orthonormal_pair(a, b, "Analytic patch")
    if (
        abs(np.linalg.norm(c) - 1) > _FRAME_TOLERANCE
        or abs(float(a @ c)) > _FRAME_TOLERANCE
        or abs(float(b @ c)) > _FRAME_TOLERANCE
    ):
        raise ValueError(
            f"patch.axis must be unit length and orthogonal to both frame axes within {_FRAME_TOLERANCE}; got {c.tolist()}."
        )


def _crosses_c0_knot(knots: Array, degree: int, lower: float, upper: float, /) -> bool:
    values = np.asarray(knots)
    for knot in np.unique(values[degree + 1 : -degree - 1]):
        if lower <= knot <= upper and np.count_nonzero(values == knot) >= degree:
            return True
    return False


def _padded(box: np.ndarray, arithmetic_magnitude: np.ndarray, /) -> np.ndarray:
    """Round each coordinate outward using only its own arithmetic operands."""
    margin = 64.0 * np.finfo(np.float64).eps * arithmetic_magnitude
    return np.stack(
        (
            np.nextafter(box[0] - margin, -np.inf),
            np.nextafter(box[1] + margin, np.inf),
        )
    )


def _trig_ranges(
    first: float, last: float, /
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bound ``(cos, sin)`` on ``[first, last]`` including quarter-turn extrema."""
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
    magnitude = np.abs(offset.astype(np.float64))
    for vector, (first, last) in terms:
        low, high = vector * first, vector * last
        lower += np.minimum(low, high)
        upper += np.maximum(low, high)
        magnitude = np.nextafter(
            magnitude + np.maximum(np.abs(low), np.abs(high)), np.inf
        )
    return _padded(np.stack((lower, upper)), magnitude)


def _range(first: ConvertibleToArray, last: ConvertibleToArray, /) -> tuple[float, float]:
    first_, last_ = _host_scalar(first, "first"), _host_scalar(last, "last")
    if not first_ < last_:
        raise ValueError(
            f"curve.parameter_range must satisfy first < last; got {(first_, last_)}."
        )
    return first_, last_


def _parameter_box(parameter_box: ConvertibleToArray, /) -> np.ndarray:
    box = np.asarray(parameter_box, dtype=np.float64)
    if box.shape != (2, 2) or not np.all(np.isfinite(box)):
        raise ValueError(
            f"patch.parameter_box must be finite with shape (2, 2); got shape {box.shape} and values {box}."
        )
    if np.any(box[1] < box[0]):
        raise ValueError(f"patch.parameter_box must satisfy lower <= upper; got {box}.")
    return box


def _check_knots(knots: np.ndarray, degree: int, /) -> None:
    """Nondecreasing knots with legal end and interior multiplicities."""
    if not np.all(np.isfinite(knots)) or np.any(np.diff(knots) < 0.0):
        raise ValueError(f"spline.knots must be finite and nondecreasing; got {knots}.")
    lower, upper = knots[degree], knots[-degree - 1]
    if not upper > lower:
        raise ValueError(
            f"spline.knots must define a nonempty active parameter domain; got {(lower, upper)}."
        )
    values, counts = np.unique(knots, return_counts=True)
    interior = (values > lower) & (values < upper)
    if np.any(counts > degree + 1) or np.any(counts[interior] > degree):
        raise ValueError(
            f"spline.knot_multiplicities must not exceed degree internally or degree + 1 at the ends; got degree {degree} and multiplicities {counts.tolist()}."
        )


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


@checked
def piece_hull(
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
        # De Casteljau executes no split on the full range or a constant axis.
        # Its copied bank is identical to the already owned source controls.
        if lower.shape[axis] == 1 or (local_first == 0.0 and local_last == 1.0):
            continue
        lower, upper = restrict_bernstein_bounds(
            lower, upper, local_first, local_last, axis
        )
    if np.any(lower[..., -1] <= 0.0):
        raise ValueError(
            f"rational.denominator must have a positive lower bound; got {lower[..., -1]}."
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


@checked
def rational_piece_jet_bounds(
    piece: RationalBezierPiece,
    first: float,
    last: float,
    /,
    *,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose rational quotient jets with outward homogeneous Bernstein hulls.

    A positive denominator is proved by its restricted control hull, not by an
    expanded basis sum. Differentiated homogeneous controls enclose source
    derivatives. The quotient recurrence retains coefficient construction
    enclosures and directed rounding at every subtraction/product/division.
    Point queries and one-sided closed spline spans are supported.
    """
    if order not in (0, 1, 2) or isinstance(order, bool):
        raise ValueError(
            f"rational.order must be zero, one, or two and not boolean; got {order!r}."
        )
    return _rational_piece_jets(piece, first, last, order)[order]


def _rational_piece_jets(
    piece: RationalBezierPiece, first: float, last: float, order: int, /
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Every rational quotient jet of orders ``0..order`` (at most three)."""

    if len(piece.parameter_bounds) != 1 or piece.homogeneous_controls.ndim != 2:
        raise ValueError(
            f"rational.piece must describe one Bernstein curve span; got bounds {piece.parameter_bounds} and controls shape {piece.homogeneous_controls.shape}."
        )
    if order not in (0, 1, 2, 3) or isinstance(order, bool):
        raise ValueError(
            f"rational.order must be zero through three and not boolean; got {order!r}."
        )
    ((start, end),) = piece.parameter_bounds
    if not isfinite(first) or not isfinite(last) or not start <= first <= last <= end:
        raise ValueError(
            f"rational.parameter_range must be finite and lie inside its closed source span {(start, end)}; got {(first, last)}."
        )
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
                controls = _array_interval_bounds(interval_multiply(difference, factor))
    denominator = (homogeneous[0][0][-1], homogeneous[0][1][-1])
    if denominator[0] <= 0.0:
        raise RationalEnclosureError(
            f"rational.denominator must have a positive lower bound; got {denominator[0]}."
        )
    jets = [
        _array_interval_bounds(
            interval_divide((homogeneous[0][0][:-1], homogeneous[0][1][:-1]), denominator)
        )
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
        jets.append(_array_interval_bounds(interval_divide(remainder, denominator)))
    return tuple(jets)


class BernsteinRestrictionBank:
    """Retain immutable Bernstein restrictions keyed by coefficient bytes per scope."""

    @checked
    def __init__(self, *, budget: CoordinateEnclosureBudget) -> None:
        self.budget = budget
        self.entries: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.prefixes: dict[tuple, tuple] = {}
        self.jet_bounds: dict[
            tuple,
            tuple[
                tuple[np.ndarray, np.ndarray],
                tuple[np.ndarray, np.ndarray] | None,
            ],
        ] = {}
        self.storage = budget.live_storage()
        self.live = self.storage.__enter__()
        self.bytes_upper = 0

    def close(self) -> None:
        self.entries.clear()
        self.prefixes.clear()
        self.jet_bounds.clear()
        self.storage.__exit__(None, None, None)

    def prefix(self, piece: RationalBezierPiece, order: int) -> tuple:
        arrays = (
            piece.homogeneous_controls,
            piece.homogeneous_lower,
            piece.homogeneous_upper,
        )
        size = 1024 + sum((value.nbytes for value in arrays))
        self.live.set_bound(
            self.bytes_upper + size, work=sum((value.size for value in arrays))
        )
        key = (
            piece.span_index,
            piece.parameter_bounds,
            order,
            tuple(((value.dtype.str, value.shape, value.tobytes()) for value in arrays)),
        )
        cached = self.prefixes.get(key)
        if cached is None:
            self.prefixes[key] = key
            self.bytes_upper += size
            return key
        del key
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
        self.budget.reserve(1, 256)
        cache_prefix = axis == 0
        full_key = (key, axis, np.float64(first).tobytes(), np.float64(last).tobytes())
        if cache_prefix:
            found = self.entries.get(full_key)
            if found is not None:
                return found
        self.budget.reserve(2 * lower.size)
        self.budget.reserve(
            2 * lower.size * max(0, lower.shape[axis] - 1),
            4096 + 32 * (lower.nbytes + upper.nbytes),
        )
        result = restrict_bernstein_bounds(lower, upper, first, last, axis)
        if not cache_prefix:
            return result
        retained = 1024 + 4 * (lower.nbytes + upper.nbytes)
        self.live.set_bound(self.bytes_upper + retained)
        for array in result:
            array.setflags(write=False)
        self.entries[full_key] = result
        self.bytes_upper += retained
        return result


@contextmanager
@checked
def source_bernstein_restriction_scope(
    *,
    budget: CoordinateEnclosureBudget,
    restriction_bank: BernsteinRestrictionBank | None = None,
) -> Iterator[BernsteinRestrictionBank]:
    """Borrow the supplied restriction bank or own one for this request."""
    if restriction_bank is not None:
        if restriction_bank.budget is not budget:
            raise ValueError(
                "restriction_bank.budget must match the request budget; got a different owner."
            )
        yield restriction_bank
        return
    bank = BernsteinRestrictionBank(budget=budget)
    try:
        yield bank
    finally:
        bank.close()


def _prepare_tensor_piece_controls(
    piece: RationalBezierPiece, order: int, /, *, budget: CoordinateEnclosureBudget
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Prepare differentiated source coefficients once for a bound batch."""

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
            controls[derivative] = _array_interval_bounds(
                interval_multiply(difference, factor)
            )
    return controls


def _rational_tensor_piece_jet_bounds(
    piece: RationalBezierPiece,
    box: np.ndarray,
    order: int,
    /,
    *,
    controls: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    source_key: tuple,
    retain_first_jet: bool,
    budget: CoordinateEnclosureBudget,
    restriction_bank: BernsteinRestrictionBank,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
    """Tensor source quotient jets from certified homogeneous control hulls."""

    if len(piece.parameter_bounds) != 2 or piece.homogeneous_controls.ndim != 3:
        raise ValueError(
            f"tensor.piece must describe one two-axis Bernstein span; got bounds {piece.parameter_bounds} and controls shape {piece.homogeneous_controls.shape}."
        )
    if order not in (1, 2) or isinstance(order, bool):
        raise ValueError(
            f"tensor.order must be one or two and not boolean; got {order!r}."
        )
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
                f"tensor.parameter_range must be finite and lie inside its closed native span {(start, end)}; got axis {axis} range {(first, last)}."
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
        bank = restriction_bank
        for axis, (first, last) in enumerate(local_bounds):
            key = (
                key,
                axis,
                clipped_bounds[axis],
                np.float64(first).tobytes(),
                np.float64(last).tobytes(),
            )
            if restricted[0].shape[axis] == 1 or (first == 0.0 and last == 1.0):
                continue
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
        raise RationalEnclosureError(
            f"tensor.denominator must have a positive lower bound; got {denominator[0]}."
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
        jets[derivative] = _array_interval_bounds(interval_divide(remainder, denominator))
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


class AbstractCurve(StrictModule, CoefficientEnclosureSource):
    """Parametric curve in two (p-curve) or three (edge) dimensions."""

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
    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Conservative ``(2, dimension)`` box of the curve over ``[first, last]``."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        raise NotImplementedError

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        """Validate a trimmed parameter range against the carrier domain."""
        first_, last_ = _range(first, last)
        period = self.period
        if period is not None and last_ - first_ > period * (1.0 + 1.0e-12):
            raise ValueError(
                f"curve.parameter_range must cover at most one period {period}; got {(first_, last_)}."
            )
        domain = self.parameter_domain
        if domain is not None:
            slack = 1.0e-12 * max(1.0, abs(domain[0]), abs(domain[1]))
            if first_ < domain[0] - slack or last_ > domain[1] + slack:
                raise ValueError(
                    f"curve.parameter_range must lie inside its carrier domain {domain} within slack {slack}; got {(first_, last_)}."
                )
        return first_, last_

    def validate_query_range(self, first: float, last: float, /) -> tuple[float, float]:
        if first < last:
            return self.validate_range(first, last)
        if first != last or not np.isfinite(first):
            raise ValueError(
                f"curve.parameter_range must have finite lower <= upper; got {(first, last)}."
            )
        domain = self.parameter_domain
        if domain is not None and not domain[0] <= first <= domain[1]:
            raise ValueError(
                f"curve.parameter_point must lie inside its carrier domain {domain}; got {first}."
            )
        return float(first), float(last)

    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> bool:
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
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Source-faithful interval derivative enclosure on a parameter range."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.derivative_bounds(
                    first,
                    last,
                    order=order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
        if order not in (1, 2) or isinstance(order, bool):
            raise ValueError(
                f"curve.derivative_order must be one or two and not boolean; got {order!r}."
            )
        if order == 2 and not self.is_c1_on(
            first, last, preparations=preparations, budget=budget
        ):
            return np.full(self.ambient_dimension, -np.inf), np.full(
                self.ambient_dimension, np.inf
            )
        if isinstance(self, BSplineCurve):
            self.validate_query_range(first, last)
            source_key = canonical_fingerprint(
                {
                    "kind": "source-bspline-curve-derivative-bounds",
                    "type": f"{type(self).__module__}.{type(self).__qualname__}",
                    "degree": self.degree,
                    "coefficients": array_tree_fingerprint(self),
                }
            )
            range_key = np.asarray((first, last), dtype=np.float64).tobytes()
            cache_key = (source_key, order, range_key)
            cached = budget.source_derivative_bounds_cache.get(cache_key)
            if cached is not None:
                budget.reserve(
                    2 * cached[0].size, 256 + cached[0].nbytes + cached[1].nbytes
                )
                return (cached[0].copy(), cached[1].copy())
            pieces = prepare_bspline_curve_spans(self, budget=budget)
            jets = []
            for piece in pieces:
                ((start, end),) = piece.parameter_bounds
                lower, upper = (max(first, start), min(last, end))
                if lower <= upper:
                    jets.append(
                        bspline_span_jet_bounds(piece, lower, upper, order, budget=budget)
                    )
            result = (
                np.min([jet[0] for jet in jets], axis=0),
                np.max([jet[1] for jet in jets], axis=0),
            )
            with budget.temporary_scope():
                budget.reserve(
                    2 * result[0].size, 1024 + result[0].nbytes + result[1].nbytes
                )
                cached = (result[0].copy(), result[1].copy())
                cached[0].setflags(write=False)
                cached[1].setflags(write=False)
                budget.retain_basis((cache_key, *cached))
                budget.source_derivative_bounds_cache[cache_key] = cached
            return result

        lower_bounds, upper_bounds = [], []
        for piece in self.source_pieces(first, last, coordinate_budget=budget):
            prepared = prepare_derivative_interval(
                piece.evaluator,
                1,
                order,
                batch_capacity=1,
                constant_bounds=coefficient_enclosures(piece.evaluator),
            )
            lower, upper = prepared.evaluate(
                np.asarray([[piece.lower]]), np.asarray([[piece.upper]])
            )
            lower_bounds.append(lower[0])
            upper_bounds.append(upper[0])
        return np.min(lower_bounds, axis=0), np.max(upper_bounds, axis=0)

    def source_pieces(
        self,
        first: float,
        last: float,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> tuple[CurvePiece[AbstractCurve], ...]:
        """Restrict a closed interval to this arithmetic source evaluator."""
        _validate_piece_interval(self, first, last)
        return (CurvePiece(self, first, last, 0),)

    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Collect construction bounds from descendant coefficient sources."""
        return child_coefficient_enclosures(self)

    def native_period_symbol(self) -> str | None:
        """Return a proved mathematical period symbol, or None."""
        return None


class LineCurve(AbstractCurve):
    """``origin + t * direction``."""

    origin: Array
    direction: Array

    def __init__(self, origin: ConvertibleToArray, direction: ConvertibleToArray) -> None:
        origin_ = _host_vector(origin, "origin")
        direction_ = _host_vector(direction, "direction")
        if origin_.shape != direction_.shape:
            raise ValueError(
                f"LineCurve.direction must match the origin dimension; got origin shape {origin_.shape} and direction shape {direction_.shape}."
            )
        if not np.linalg.norm(direction_) > 0.0:
            raise ValueError(
                f"LineCurve.direction must be nonzero; got {direction_.tolist()}."
            )
        self.origin = jax.device_put(origin_)
        self.direction = jax.device_put(direction_)

    @property
    def ambient_dimension(self) -> int:
        return self.origin.shape[0]

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.origin.dtype)
        return self.origin + parameters_[..., None] * self.direction

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
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
            raise ValueError(
                f"CircleCurve.frame must match the center dimension; got center shape {center_.shape} and axis shapes {(first.shape, second.shape)}."
            )
        if not radius_ > 0.0:
            raise ValueError(f"CircleCurve.radius must be positive; got {radius_}.")
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

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        cosine, sine = _trig_ranges(*self.validate_query_range(first, last))
        radius = float(self.radius)
        return _interval_box(
            np.asarray(self.center),
            (
                (radius * np.asarray(self.first_axis), cosine),
                (radius * np.asarray(self.second_axis), sine),
            ),
        )

    def native_period_symbol(self) -> str | None:
        """Prove the mathematical two-pi period of this native analytic carrier."""
        return "two_pi" if type(self) is CircleCurve else None


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
            raise ValueError(
                f"EllipseCurve.frame must match the center dimension; got center shape {center_.shape} and axis shapes {(first.shape, second.shape)}."
            )
        if min(radii) <= 0.0:
            raise ValueError(f"EllipseCurve.radii must be positive; got {radii}.")
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

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        cosine, sine = _trig_ranges(*self.validate_query_range(first, last))
        return _interval_box(
            np.asarray(self.center),
            (
                (float(self.first_radius) * np.asarray(self.first_axis), cosine),
                (float(self.second_radius) * np.asarray(self.second_axis), sine),
            ),
        )

    def native_period_symbol(self) -> str | None:
        """Prove the mathematical two-pi period of this native analytic carrier."""
        return "two_pi" if type(self) is EllipseCurve else None


class ParabolaCurve(AbstractCurve):
    """Vertex-frame parabola ``vertex + t²/(4f) x + t y``.

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
                f"ParabolaCurve.frame_and_focal_length must have matching dimensions and positive focal length; got vertex shape {vertex_.shape}, axis shapes {(first.shape, second.shape)}, and focal length {focal}."
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

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
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
    """Positive hyperbola branch ``center + a cosh(t) x + b sinh(t) y``."""

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
                f"HyperbolaCurve.frame_and_radii must have matching dimensions and positive radii; got center shape {center_.shape}, axis shapes {(first.shape, second.shape)}, and radii {radii}."
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

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
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
            raise ValueError(
                f"HyperbolaCurve.parameter_range must produce finite coordinate bounds; got range {(first, last)} and bounds {box}."
            )
        return box


class OffsetCurve(AbstractCurve):
    """Signed right-normal curve offset retaining its basis parameter.

    In three dimensions the normal is ``tangent cross direction``; the
    two-dimensional convention is ``(dy, -dx)``.
    """

    base: AbstractCurve
    distance: Array
    direction: Array | None

    @checked
    def __init__(
        self,
        base: AbstractCurve,
        distance: ConvertibleToArray,
        direction: ConvertibleToArray | None = None,
    ) -> None:
        distance_ = _host_scalar(distance, "distance")
        if base.ambient_dimension == 3:
            if direction is None:
                raise ValueError(
                    "OffsetCurve.direction must be supplied for a three-dimensional basis curve; got None."
                )
            direction_ = _host_vector(direction, "direction")
            if (
                direction_.shape != (3,)
                or abs(np.linalg.norm(direction_) - 1.0) > _FRAME_TOLERANCE
            ):
                raise ValueError(
                    f"OffsetCurve.direction must be a unit three-dimensional vector within {_FRAME_TOLERANCE}; got {direction_.tolist()}."
                )
            self.direction = jnp.asarray(direction_, dtype=jnp.float64)
        elif base.ambient_dimension == 2 and direction is None:
            self.direction = None
        else:
            raise ValueError(
                f"OffsetCurve.direction must be absent for a planar basis curve; got basis dimension {base.ambient_dimension} and direction {direction}."
            )
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

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        first, last = self.validate_query_range(first, last)
        box = self.base.bounding_box(
            first, last, preparations=preparations, budget=budget
        )
        distance = abs(float(self.distance))
        return np.stack(
            (
                np.nextafter(box[0] - distance, -np.inf),
                np.nextafter(box[1] + distance, np.inf),
            )
        )

    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.is_c1_on(
                    first, last, preparations=preparations, budget=budget
                )
        if not self.base.is_c1_on(first, last, preparations=preparations, budget=budget):
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

    def source_pieces(
        self,
        first: float,
        last: float,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> tuple[CurvePiece[AbstractCurve], ...]:
        """Offset each basis restriction without changing source parameters."""
        return tuple(
            CurvePiece(
                OffsetCurve(piece.evaluator, self.distance, self.direction),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in self.base.source_pieces(
                first, last, coordinate_budget=coordinate_budget
            )
        )


def _rational_coordinate_value(
    plan: TensorBSplineJetPlan,
    weights: Array,
    controls: Array,
    /,
) -> Array:
    """Apply the local rational basis as an affine coordinate action.

    Its source partition of unity permits subtracting the locally dominant
    control and adding it back after contraction. A one-hot endpoint basis then
    reproduces either authored boundary control exactly, while interior
    evaluation minimizes cancellation without changing the source controls,
    weights, knots or address.
    """
    rational = RationalSplineJet(plan, weights)
    local = plan.gather(controls)
    anchor = local[jnp.argmax(jnp.abs(rational.values))]
    return anchor + rational.values @ (local - anchor[None, ...])


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
            raise ValueError(
                f"BSplineCurve.control_points must have rank two and one row per weight; got control shape {control_points_.shape} and weight shape {weights_.shape}."
            )
        if int(degree) < 1:
            raise ValueError(f"BSplineCurve.degree must be positive; got {degree}.")
        if control_points_.shape[0] <= int(degree):
            raise ValueError(
                f"BSplineCurve.control_points must have more rows than the degree; got {control_points_.shape[0]} rows and degree {degree}."
            )
        if knots_.shape[0] != control_points_.shape[0] + int(degree) + 1:
            raise ValueError(
                f"BSplineCurve.knots must have control_count + degree + 1 entries; got {knots_.shape[0]} entries, {control_points_.shape[0]} controls, and degree {degree}."
            )
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
        """Return host Fraction Bernstein coefficients of the binary-rational source."""
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

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        first_, last_ = self.validate_query_range(first, last)
        source_key = _bspline_source_key(
            self, kind="source-bspline-curve-value-bounds", budget=budget
        )
        budget.reserve(2, 256)
        range_key = np.asarray((first_, last_), dtype=np.float64).tobytes()
        cache_key = (source_key, 0, range_key)
        cached = budget.source_derivative_bounds_cache.get(cache_key)
        if cached is not None:
            budget.reserve(2 * cached[0].size, 256 + cached[0].nbytes + cached[1].nbytes)
            return np.stack(cached)
        pieces = (
            self.bezier_pieces()
            if budget is None
            else prepare_bspline_curve_spans(self, budget=budget)
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
            piece_hull(piece, (first_,), (last_,))
            for piece in pieces
            if piece.parameter_bounds[0][1] >= first_
            and piece.parameter_bounds[0][0] <= last_
        ]
        boxes = np.stack(selected)
        result = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
        _retain_source_derivative_bounds(
            result[0], result[1], None, source_key, range_key, 0, budget=budget
        )
        return result

    def source_pieces(
        self,
        first: float,
        last: float,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> tuple[CurvePiece[AbstractCurve], ...]:
        """Extract closed Bernstein span restrictions on the source budget."""
        _validate_piece_interval(self, first, last)
        source_pieces = prepare_bspline_curve_spans(self, budget=coordinate_budget)
        pieces = []
        for index, piece in enumerate(source_pieces):
            ((lower, upper),) = piece.parameter_bounds
            piece_lower = max(lower, first)
            piece_upper = min(upper, last)
            if piece_upper > piece_lower or (
                first == last and piece_upper == piece_lower
            ):
                terms = piece.homogeneous_controls.size
                coordinate_budget.reserve(3 * terms + 2, 1024 + 8 * (3 * terms + 2))
                evaluator = BernsteinCurvePiece(
                    piece.homogeneous_controls,
                    lower,
                    upper,
                    controls_lower=piece.homogeneous_lower,
                    controls_upper=piece.homogeneous_upper,
                )
                pieces.append(CurvePiece(evaluator, piece_lower, piece_upper, index))
        return tuple(pieces)


def _retain_bspline_spans(
    key: str,
    pieces: tuple[RationalBezierPiece, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[RationalBezierPiece, ...]:
    """Retain source-bank allocations on the span ledger."""
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


@checked
def prepare_bspline_curve_spans(
    curve: BSplineCurve, /, *, budget: CoordinateEnclosureBudget
) -> tuple[RationalBezierPiece, ...]:
    """Prepare source Bernstein spans on the coefficient ledger.

    Extraction scratch is admitted before host transfer or native refinement.
    The returned pieces and their NumPy backing allocations are retained
    on this ledger; a view never stands in for its uncounted backing array.
    """
    count, dimension = curve.control_points.shape
    degree = curve.degree
    knot_count = curve.knots.shape[0]
    columns = dimension + 1
    with budget.temporary_scope():
        # The span cache key includes every numerical source leaf and its dtype,
        # shape and path, plus the native static degree. Knot/range changes
        # therefore cannot reuse an ancestor source's span bank.
        key = _bspline_source_key(curve, kind="source-bspline-spans", budget=budget)
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
        return _retain_bspline_spans(key, pieces, budget=budget)


@checked
def bspline_span_jet_bounds(
    piece: RationalBezierPiece,
    first: float,
    last: float,
    order: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound one closed native source span, including either one-sided endpoint.

    The caller owns the temporary scope containing the returned arrays. Public
    whole-curve C0 derivative semantics are deliberately not changed.
    """
    if order not in (0, 1, 2) or isinstance(order, bool):
        raise ValueError(
            f"spline.jet_order must be zero, one, or two and not boolean; got {order!r}."
        )
    return _bspline_span_jets(piece, first, last, order, budget=budget)[order]


def _bspline_span_jets(
    piece: RationalBezierPiece,
    first: float,
    last: float,
    order: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Every closed-span jet of orders ``0..order`` on the source ledger."""
    if len(piece.parameter_bounds) != 1 or piece.homogeneous_controls.ndim != 2:
        raise ValueError(
            f"spline.piece must describe one Bernstein curve span; got bounds {piece.parameter_bounds} and controls shape {piece.homogeneous_controls.shape}."
        )
    if order not in (0, 1, 2, 3) or isinstance(order, bool):
        raise ValueError(
            f"spline.jet_order must be zero through three and not boolean; got {order!r}."
        )
    ((start, end),) = piece.parameter_bounds
    if not isfinite(first) or not isfinite(last) or not start <= first <= last <= end:
        raise ValueError(
            f"spline.parameter_range must be finite and lie inside its closed native span {(start, end)}; got {(first, last)}."
        )
    rows, columns = piece.homogeneous_controls.shape
    # Each derivative restricts both coefficient streams by at most two
    # full de Casteljau triangles, then visits their homogeneous hulls.
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
    every derivative (including knot tangents) come from the source span
    evaluation, extended polynomially exactly like the span's Bernstein piece.
    """

    source: BSplineCurve
    span: int = eqx.field(static=True)

    @checked
    def __init__(self, source: BSplineCurve, span: int) -> None:
        if isinstance(span, bool) or not isinstance(span, int):
            raise TypeError(
                f"BSplineSpanCurve.span must be an integer and not boolean; got {span!r}."
            )
        if not source.degree <= span < source.control_points.shape[0]:
            raise ValueError(
                f"BSplineSpanCurve.span must address an active span in [{source.degree}, {source.control_points.shape[0]}); got {span}."
            )
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

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        lower, upper = self.parameter_domain
        first_, last_ = self.validate_query_range(first, last)
        if first_ < lower or last_ > upper:
            raise ValueError(
                f"BSplineSpanCurve.parameter_range must lie inside its knot span {(lower, upper)}; got {(first_, last_)}."
            )
        return self.source.bounding_box(
            first_, last_, preparations=preparations, budget=budget
        )


# ---------------------------------------------------------------- surfaces


class AbstractSurfacePatch(StrictModule, CoefficientEnclosureSource):
    """Pure-JAX parametric surface patch in native CAD coordinates."""

    @abstractmethod
    def evaluate(self, parameters: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Conservative ``(2, 3)`` box of the patch over a ``(2, 2)`` parameter box."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
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

    def validate_parameter_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Validate a face parameter box against periods and carrier domains."""
        box = _parameter_box(parameter_box)
        for axis, period in enumerate(self.periods):
            if period is not None and box[1, axis] - box[0, axis] > period * (
                1.0 + 1.0e-12
            ):
                raise ValueError(
                    f"patch.parameter_box must cover at most one period {period} on axis {axis}; got range {(box[0, axis], box[1, axis])}."
                )
        return box

    def is_c1_on(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.is_c1_on(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
        if isinstance(self, BSplineSurfacePatch):
            return not (
                _crosses_c0_knot(self.u_knots, self.u_degree, box[0, 0], box[1, 0])
                or _crosses_c0_knot(self.v_knots, self.v_degree, box[0, 1], box[1, 1])
            )
        if isinstance(self, ExtrusionSurface):
            return self.curve.is_c1_on(
                box[0, 0], box[1, 0], preparations=preparations, budget=budget
            )
        if isinstance(self, RevolutionSurface):
            return self.curve.is_c1_on(
                box[0, 1], box[1, 1], preparations=preparations, budget=budget
            )
        if isinstance(self, RuledSurface):
            return self.first.is_c1_on(
                box[0, 0], box[1, 0], preparations=preparations, budget=budget
            ) and self.second.is_c1_on(
                box[0, 0], box[1, 0], preparations=preparations, budget=budget
            )
        return True

    def derivative_bounds(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        order: int = 1,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Source interval jets, shaped (3,2) or (3,2,2), not sampled bounds.

        Finite derivative bounds do not prove chart regularity. Pole/apex
        metadata and a positive bound on the tangent cross product remain
        required for normal-turn or inverse-chart certificates.
        """
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.derivative_bounds(
                    parameter_box,
                    order=order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
        lower, upper = self.derivative_bounds_batch(
            np.asarray(parameter_box, dtype=np.float64)[None],
            order=order,
            preparations=preparations,
            budget=budget,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        return lower[0], upper[0]

    def derivative_bounds_batch(
        self,
        parameter_boxes: ConvertibleToArray,
        /,
        *,
        order: int = 1,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Source interval jets of ``(B, 2, 2)`` boxes, shaped (B,3,2) or (B,3,2,2).

        Every box is enclosed independently by its own source pieces. Tensor
        splines use retained homogeneous Bernstein quotient jets; other pieces
        with identical source coefficients share one interval program per call.
        """
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_boxes,
                )
            ) as budget:
                return self.derivative_bounds_batch(
                    parameter_boxes,
                    order=order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
        if restriction_bank is None:
            with source_bernstein_restriction_scope(budget=budget) as restriction_bank:
                return self.derivative_bounds_batch(
                    parameter_boxes,
                    order=order,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                    preparations=preparations,
                )
        if order not in (1, 2) or isinstance(order, bool):
            raise ValueError(
                f"surface.derivative_order must be one or two and not boolean; got {order!r}."
            )
        boxes = np.asarray(parameter_boxes, dtype=np.float64)
        if boxes.ndim != 3 or boxes.shape[1:] != (2, 2):
            raise ValueError(
                f"surface.derivative_boxes must have shape (num_boxes, 2, 2); got {boxes.shape}."
            )
        match self:
            case BSplineSurfacePatch():
                return _bspline_surface_derivative_batch(
                    self,
                    boxes,
                    order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
            case ExtrusionSurface():
                return _extrusion_derivative_batch(
                    self,
                    boxes,
                    order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
            case RevolutionSurface():
                return _revolution_derivative_batch(
                    self,
                    boxes,
                    order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
            case OffsetSurface(base=RevolutionSurface() as base):
                frame = meridian_frame(base, budget=budget)
                if frame is not None:
                    return _offset_meridian_jet_bounds(
                        self,
                        frame,
                        boxes,
                        order,
                        budget=budget,
                        meridian_jets=meridian_jets,
                    )
                return _interval_surface_derivative_batch(
                    self,
                    boxes,
                    order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
            case AbstractSurfacePatch():
                return _interval_surface_derivative_batch(
                    self,
                    boxes,
                    order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
            case _:
                raise TypeError(
                    f"surface.patch must be a native surface patch; got {type(self).__name__}."
                )

    def source_pieces(
        self, parameter_box: ConvertibleToArray, /, *, budget: CoordinateEnclosureBudget
    ) -> tuple[SurfacePiece[AbstractSurfacePatch], ...]:
        """Restrict a closed box to this arithmetic source evaluator."""
        box = np.asarray(
            self.validate_parameter_box(parameter_box, budget=budget), dtype=np.float64
        )
        lower, upper = box
        return (SurfacePiece(self, lower, upper, 0),)

    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Collect construction bounds from descendant coefficient sources."""
        return child_coefficient_enclosures(self)

    def native_period_symbols(self) -> tuple[str | None, str | None]:
        """Return proved mathematical period symbols for the two chart axes."""
        return None, None


class _TensorDerivativeSource(StrictModule):
    """Retain one source's Bernstein pieces, restriction identities and jets."""

    pieces: tuple[RationalBezierPiece, ...]
    source_keys: dict[tuple[int, ...], tuple]
    controls: dict[tuple[int, ...], dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]


def _prepare_tensor_derivative_source(
    patch: BSplineSurfacePatch,
    order: int,
    restriction_bank: BernsteinRestrictionBank,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    source_key: str,
) -> _TensorDerivativeSource:
    cache_key = (source_key, order)
    cached = budget.tensor_derivative_source_cache.get(cache_key)
    if cached is not None:
        budget.reserve(1)
        return cached
    pieces = _prepare_bspline_surface_spans(patch, budget=budget)
    source_keys = {
        piece.span_index: restriction_bank.prefix(piece, order) for piece in pieces
    }
    controls = {
        piece.span_index: _prepare_tensor_piece_controls(piece, order, budget=budget)
        for piece in pieces
    }
    prepared = _TensorDerivativeSource(
        pieces=pieces, source_keys=source_keys, controls=controls
    )
    budget.retain_owner(prepared)
    budget.retain_basis((cache_key,))
    budget.tensor_derivative_source_cache[cache_key] = prepared
    return prepared


def _bspline_source_key(
    source: BSplineCurve | BSplineSurfacePatch,
    /,
    *,
    kind: Literal[
        "source-bspline-spans",
        "source-bspline-curve-value-bounds",
        "source-bspline-surface-derivative-bounds",
    ],
    budget: CoordinateEnclosureBudget,
) -> str:
    leaves: tuple[Array, ...]
    degrees: tuple[int, ...]
    static_value: int | tuple[int, ...]
    if isinstance(source, BSplineCurve):
        leaves = (source.control_points, source.weights, source.knots)
        degrees = (source.degree,)
        static_name, static_value = "degree", source.degree
        exact = type(source) is BSplineCurve
    else:
        leaves = (source.control_points, source.weights, source.u_knots, source.v_knots)
        degrees = (source.u_degree, source.v_degree)
        static_name, static_value = "degrees", degrees
        exact = type(source) is BSplineSurfacePatch
    identity = None
    if exact and all(
        isinstance(leaf, Array) and not leaf.is_deleted() for leaf in leaves
    ):
        budget.reserve(len(leaves) + 1, 1024)
        identity = (kind, degrees, tuple(map(id, leaves)))
        cached = budget.immutable_bspline_fingerprint_cache.get(identity)
        if cached is not None:
            return cached
    source_terms = sum(leaf.size for leaf in leaves)
    metadata = 1024 if kind == "source-bspline-curve-value-bounds" else 8192
    budget.reserve(source_terms, metadata + 24 * source_terms)
    key = canonical_fingerprint(
        {
            "kind": kind,
            "type": f"{type(source).__module__}.{type(source).__qualname__}",
            static_name: static_value,
            "coefficients": array_tree_fingerprint(source),
        }
    )
    if identity is not None:
        budget.retain_owner(leaves)
        budget.retain_basis((identity, key))
        budget.immutable_bspline_fingerprint_cache[identity] = key
    return key


def _bspline_surface_derivative_batch(
    self: BSplineSurfacePatch,
    boxes: np.ndarray,
    order: int,
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    restriction_bank: BernsteinRestrictionBank,
    meridian_jets: list[MeridianOverlaps] | None,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (boxes.shape[0], 3, 2) + ((2,) if order == 2 else ())
    size = boxes.shape[0] * 3 * 2**order
    budget.reserve(0, 128 + 16 * size)
    lower, upper = (np.empty(shape), np.empty(shape))
    if not boxes.shape[0]:
        return (lower, upper)
    pieces: tuple[RationalBezierPiece, ...] | None = None
    piece_controls: dict[
        tuple[int, ...], dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    ] = {}
    with budget.temporary_scope():
        source_cache_key = _bspline_source_key(
            self, kind="source-bspline-surface-derivative-bounds", budget=budget
        )
        c0 = _prepared_tensor_derivative_boxes(
            self, boxes, order, source_cache_key, budget=budget
        )
        prepared_rows: dict[bytes, int] = {}
        try:
            for row in range(boxes.shape[0]):
                box = np.empty((2, 2), dtype=np.float64)
                box[...] = boxes[row]
                budget.reserve(box.size, 256)
                box_key = box.tobytes()
                prepared_row = prepared_rows.get(box_key)
                if prepared_row is not None:
                    budget.reserve(2 * lower[row].size)
                    lower[row] = lower[prepared_row]
                    upper[row] = upper[prepared_row]
                    continue
                prepared_rows[box_key] = row
                cache_key = (source_cache_key, order, box_key)
                cached = budget.source_derivative_bounds_cache.get(cache_key)
                if cached is not None:
                    budget.reserve(2 * lower[row].size)
                    lower[row], upper[row] = cached
                    continue
                if c0[row]:
                    budget.reserve(2 * lower[row].size)
                    lower[row], upper[row] = (-np.inf, np.inf)
                    continue
                if pieces is None:
                    source = _prepare_tensor_derivative_source(
                        self,
                        order,
                        restriction_bank,
                        budget=budget,
                        source_key=source_cache_key,
                    )
                    pieces = source.pieces
                    source_keys = source.source_keys
                    piece_controls = source.controls
                budget.reserve(2 * lower[row].size)
                lower[row].fill(np.inf)
                upper[row].fill(-np.inf)
                first_jet_bounds = None
                if (
                    order == 2
                    and (source_cache_key, 1, box_key)
                    not in budget.source_derivative_bounds_cache
                ):
                    budget.reserve(12, 1024 + 12 * np.dtype(np.float64).itemsize)
                    first_jet_bounds = (np.full((3, 2), np.inf), np.full((3, 2), -np.inf))
                with budget.temporary_scope():
                    _enclose_tensor_derivative_row(
                        pieces,
                        box,
                        order,
                        lower[row],
                        upper[row],
                        first_jet_bounds,
                        piece_controls=piece_controls,
                        source_keys=source_keys,
                        budget=budget,
                        restriction_bank=restriction_bank,
                    )
                _retain_source_derivative_bounds(
                    lower[row],
                    upper[row],
                    first_jet_bounds,
                    source_cache_key,
                    box_key,
                    order,
                    budget=budget,
                )
        finally:
            prepared_rows.clear()
    return (lower, upper)


def _prepared_tensor_derivative_boxes(
    source: BSplineSurfacePatch,
    boxes: np.ndarray,
    order: int,
    source_key: str,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Reuse ordered float64 boxes under their source fingerprint and order."""
    # The public batch boundary already fixes the trailing shape to (2, 2)
    # and dtype to float64, so byte length also fixes the leading box count.
    budget.reserve(boxes.size + 1, 512 + boxes.nbytes)
    # C0-knot boxes are masked only for second derivatives; first-order
    # enclosures can stay finite for the same source and parameter boxes.
    key = (source_key, order, boxes.tobytes())
    cached = budget.source_derivative_box_validation_cache.get(key)
    if cached is not None:
        budget.reserve(cached.size, 128 + cached.nbytes)
        return cached.copy()
    result = _validate_tensor_derivative_boxes(source, boxes, order, budget=budget)
    budget.reserve(result.size, 128 + result.nbytes)
    stored = result.copy()
    stored.setflags(write=False)
    budget.retain_owner(stored)
    budget.retain_basis((key,))
    budget.source_derivative_box_validation_cache[key] = stored
    return result


def _validate_tensor_derivative_boxes(
    self: BSplineSurfacePatch,
    boxes: np.ndarray,
    order: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    if not np.all(np.isfinite(boxes)):
        raise ValueError(f"surface.derivative_boxes must be finite; got {boxes}.")
    if np.any(boxes[:, 1] < boxes[:, 0]):
        raise ValueError(
            f"surface.derivative_boxes must satisfy lower <= upper; got {boxes}."
        )
    c0 = np.zeros((boxes.shape[0],), dtype=np.bool_)
    for axis, (source_knots, degree) in enumerate(
        ((self.u_knots, self.u_degree), (self.v_knots, self.v_degree))
    ):
        budget.reserve(
            source_knots.size + 4 * boxes.shape[0],
            1024 + 48 * source_knots.size + 8 * boxes.shape[0],
        )
        knots = np.asarray(source_knots)
        first, last = (float(knots[degree]), float(knots[-degree - 1]))
        slack = 1e-12 * max(1.0, abs(first), abs(last))
        if np.any(boxes[:, 0, axis] < first - slack) or np.any(
            boxes[:, 1, axis] > last + slack
        ):
            raise ValueError(
                f"BSplineSurfacePatch.parameter_boxes must lie inside the active domain {(first, last)} on axis {axis} within slack {slack}; got {boxes[:, :, axis]}."
            )
        if order == 2:
            for knot in np.unique(knots[degree + 1 : -degree - 1]):
                if np.count_nonzero(knots == knot) >= degree:
                    budget.reserve(3 * boxes.shape[0])
                    c0 |= (boxes[:, 0, axis] <= knot) & (knot <= boxes[:, 1, axis])
    return c0


def _enclose_tensor_derivative_row(
    pieces: tuple[RationalBezierPiece, ...],
    box: np.ndarray,
    order: int,
    lower: np.ndarray,
    upper: np.ndarray,
    first_jet_bounds: tuple[np.ndarray, np.ndarray] | None,
    /,
    *,
    piece_controls: dict[
        tuple[int, ...], dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    ],
    source_keys: dict[tuple[int, ...], tuple],
    budget: CoordinateEnclosureBudget,
    restriction_bank: BernsteinRestrictionBank,
) -> None:
    for piece in pieces:
        if any(
            (
                piece.parameter_bounds[axis][1] < box[0, axis]
                or piece.parameter_bounds[axis][0] > box[1, axis]
                for axis in (0, 1)
            )
        ):
            continue
        cache_key = None
        if len(pieces) > 1:
            # Only multi-span queries can share clipped piece ranges without
            # sharing the whole row already retained by the request budget.
            budget.reserve(4, 256)
            clipped = tuple(
                (
                    np.float64(max(start, box[0, axis])).tobytes(),
                    np.float64(min(end, box[1, axis])).tobytes(),
                )
                for axis, (start, end) in enumerate(piece.parameter_bounds)
            )
            cache_key = (
                source_keys[piece.span_index],
                clipped,
                first_jet_bounds is not None,
            )
        cached = None if cache_key is None else restriction_bank.jet_bounds.get(cache_key)
        if cached is None:
            cached = _rational_tensor_piece_jet_bounds(
                piece,
                box,
                order,
                controls=piece_controls[piece.span_index],
                source_key=source_keys[piece.span_index],
                retain_first_jet=first_jet_bounds is not None,
                budget=budget,
                restriction_bank=restriction_bank,
            )
            if cache_key is not None:
                arrays = cached[0] + (() if cached[1] is None else cached[1])
                retained = 1024 + sum(array.nbytes for array in arrays)
                restriction_bank.live.set_bound(restriction_bank.bytes_upper + retained)
                for array in arrays:
                    array.setflags(write=False)
                restriction_bank.jet_bounds[cache_key] = cached
                restriction_bank.bytes_upper += retained
        (low, high), first_jet = cached
        budget.reserve(2 * low.size)
        np.minimum(lower, low, out=lower)
        np.maximum(upper, high, out=upper)
        if first_jet_bounds is not None and first_jet is not None:
            budget.reserve(12)
            np.minimum(first_jet_bounds[0], first_jet[0], out=first_jet_bounds[0])
            np.maximum(first_jet_bounds[1], first_jet[1], out=first_jet_bounds[1])
        del low, high


def _retain_source_derivative_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    first_jet_bounds: tuple[np.ndarray, np.ndarray] | None,
    source_cache_key: str,
    box_key: bytes,
    order: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    cache_key = (source_cache_key, order, box_key)
    with budget.temporary_scope():
        budget.reserve(2 * lower.size, 1024 + lower.nbytes + upper.nbytes)
        cached_lower, cached_upper = (lower.copy(), upper.copy())
        cached_lower.setflags(write=False)
        cached_upper.setflags(write=False)
        budget.retain_basis((cache_key, cached_lower, cached_upper))
        budget.source_derivative_bounds_cache[cache_key] = (cached_lower, cached_upper)
    if first_jet_bounds is not None:
        first_key = (source_cache_key, 1, box_key)
        first_jet_bounds[0].setflags(write=False)
        first_jet_bounds[1].setflags(write=False)
        budget.retain_basis((first_key, *first_jet_bounds))
        budget.source_derivative_bounds_cache[first_key] = first_jet_bounds


def _extrusion_derivative_batch(
    self: ExtrusionSurface,
    boxes: np.ndarray,
    order: int,
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    restriction_bank: BernsteinRestrictionBank,
    meridian_jets: list[MeridianOverlaps] | None,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (boxes.shape[0], 3, 2) + ((2,) if order == 2 else ())
    lower, upper = np.zeros(shape), np.zeros(shape)
    for row in range(boxes.shape[0]):
        box = boxes[row : row + 1].reshape((2, 2))
        self.validate_parameter_box(box, budget=budget)
        profile_low, profile_high = self.curve.derivative_bounds(
            float(box[0, 0]),
            float(box[1, 0]),
            order=order,
            preparations=preparations,
            budget=budget,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        if order == 1:
            lower[row, :, 0], upper[row, :, 0] = profile_low, profile_high
            lower[row, :, 1] = upper[row, :, 1] = np.asarray(self.direction)
        else:
            lower[row, :, 0, 0], upper[row, :, 0, 0] = profile_low, profile_high
    return lower, upper


def _revolution_derivative_batch(
    self: RevolutionSurface,
    boxes: np.ndarray,
    order: int,
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    restriction_bank: BernsteinRestrictionBank,
    meridian_jets: list[MeridianOverlaps] | None,
) -> tuple[np.ndarray, np.ndarray]:
    spans = prepare_meridian_spans(self.curve, budget=budget)
    if spans is not None:
        return _revolution_jet_bounds(
            self, spans, boxes, order, budget=budget, meridian_jets=meridian_jets
        )
    origin = np.asarray(self.axis_origin, dtype=np.float64)
    hull = [
        (np.zeros((boxes.shape[0], 3)), np.zeros((boxes.shape[0], 3)))
        for _ in range(order + 1)
    ]
    rough = np.zeros((boxes.shape[0],), dtype=np.bool_)
    for row in range(boxes.shape[0]):
        box = boxes[row : row + 1].reshape((2, 2))
        self.validate_parameter_box(box, budget=budget)
        if order == 2 and not self.is_c1_on(
            box, preparations=preparations, budget=budget
        ):
            rough[row] = True
            continue
        first, last = float(box[0, 1]), float(box[1, 1])
        profile = self.curve.bounding_box(
            *_curve_box_range(self.curve, first, last),
            preparations=preparations,
            budget=budget,
        )
        hull[0][0][row] = np.nextafter(profile[0] - origin, -np.inf)
        hull[0][1][row] = np.nextafter(profile[1] - origin, np.inf)
        for derivative in range(1, order + 1):
            hull[derivative][0][row], hull[derivative][1][row] = (
                self.curve.derivative_bounds(
                    first,
                    last,
                    order=derivative,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
            )
    lower, upper = _rotated_jets(
        np.asarray(self.axis_direction, dtype=np.float64), boxes, hull, order
    )
    lower[rough], upper[rough] = -np.inf, np.inf
    return lower, upper


def _interval_surface_derivative_batch(
    self: AbstractSurfacePatch,
    boxes: np.ndarray,
    order: int,
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    restriction_bank: BernsteinRestrictionBank,
    meridian_jets: list[MeridianOverlaps] | None,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (boxes.shape[0], 3, 2) + ((2,) if order == 2 else ())
    lower = np.full(shape, np.inf)
    upper = np.full(shape, -np.inf)
    groups: dict[
        tuple[object, ...],
        tuple[
            AbstractSurfacePatch,
            list[tuple[int, np.ndarray, np.ndarray]],
        ],
    ] = {}
    for row, box in enumerate(boxes):
        if order == 2 and not self.is_c1_on(
            box, preparations=preparations, budget=budget
        ):
            lower[row], upper[row] = -np.inf, np.inf
            continue
        pieces = self.source_pieces(box, budget=budget)
        if not pieces:
            raise ValueError(
                f"surface.derivative_box must select a source piece; got {box}."
            )
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
        prepared = prepare_derivative_interval(
            evaluator,
            2,
            order,
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

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
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
            raise ValueError(
                f"CylinderPatch.origin_and_radius must have finite origin and positive finite radius; got origin {self.origin} and radius {self.radius}."
            )

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

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
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

    def native_period_symbols(self) -> tuple[str | None, str | None]:
        """Prove mathematical angular periods for this native analytic patch."""
        return ("two_pi", None) if type(self) is CylinderPatch else (None, None)


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
                f"ConePatch.origin_radius_and_angle must have finite origin, nonnegative finite reference radius, and finite semi-angle with absolute value below pi/2; got origin {self.origin}, radius {self.reference_radius}, and angle {self.semi_angle}."
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

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
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

    def native_period_symbols(self) -> tuple[str | None, str | None]:
        """Prove mathematical angular periods for this native analytic patch."""
        return ("two_pi", None) if type(self) is ConePatch else (None, None)


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
            raise ValueError(
                f"SpherePatch.center_and_radius must have finite center and positive finite radius; got center {self.center} and radius {self.radius}."
            )

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

    def validate_parameter_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.validate_parameter_box(parameter_box, budget=budget)
        box = super().validate_parameter_box(parameter_box, budget=budget)
        slack = 1.0e-12
        if box[0, 1] < -0.5 * pi - slack or box[1, 1] > 0.5 * pi + slack:
            raise ValueError(
                f"SpherePatch.latitude must lie within [-pi/2, pi/2] with slack {slack}; got {(box[0, 1], box[1, 1])}."
            )
        return box

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
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

    def native_period_symbols(self) -> tuple[str | None, str | None]:
        """Prove mathematical angular periods for this native analytic patch."""
        return ("two_pi", None) if type(self) is SpherePatch else (None, None)


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
            raise ValueError(
                f"TorusPatch.center_and_radii must have finite center and positive finite radii; got center {self.center} and radii {(self.major_radius, self.minor_radius)}."
            )

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

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
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

    def native_period_symbols(self) -> tuple[str | None, str | None]:
        """Prove mathematical angular periods for this native analytic patch."""
        return ("two_pi", "two_pi") if type(self) is TorusPatch else (None, None)


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
            raise ValueError(
                f"BSplineSurfacePatch.control_points must have shape (num_u, num_v, 3); got {control_points_.shape}."
            )
        if weights_.shape != control_points_.shape[:2]:
            raise ValueError(
                f"BSplineSurfacePatch.weights must match the control-point grid; got weight shape {weights_.shape} and control shape {control_points_.shape}."
            )
        if int(u_degree) < 1 or int(v_degree) < 1:
            raise ValueError(
                f"BSplineSurfacePatch.degrees must be positive; got {(u_degree, v_degree)}."
            )
        if control_points_.shape[0] <= int(u_degree) or control_points_.shape[1] <= int(
            v_degree
        ):
            raise ValueError(
                f"BSplineSurfacePatch.control_points must have more rows on each axis than its degree; got grid {control_points_.shape[:2]} and degrees {(u_degree, v_degree)}."
            )
        if u_knots_.shape[0] != control_points_.shape[0] + int(u_degree) + 1:
            raise ValueError(
                f"BSplineSurfacePatch.u_knots must have u_control_count + u_degree + 1 entries; got {u_knots_.shape[0]} entries, {control_points_.shape[0]} controls, and degree {u_degree}."
            )
        if v_knots_.shape[0] != control_points_.shape[1] + int(v_degree) + 1:
            raise ValueError(
                f"BSplineSurfacePatch.v_knots must have v_control_count + v_degree + 1 entries; got {v_knots_.shape[0]} entries, {control_points_.shape[1]} controls, and degree {v_degree}."
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

    def validate_parameter_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.validate_parameter_box(parameter_box, budget=budget)
        box = super().validate_parameter_box(parameter_box, budget=budget)
        for axis, (knots, degree) in enumerate(
            ((self.u_knots, self.u_degree), (self.v_knots, self.v_degree))
        ):
            host = np.asarray(knots)
            lower, upper = host[degree], host[-degree - 1]
            slack = 1.0e-12 * max(1.0, abs(lower), abs(upper))
            if box[0, axis] < lower - slack or box[1, axis] > upper + slack:
                raise ValueError(
                    f"BSplineSurfacePatch.parameter_box must lie inside the active domain {(lower, upper)} on axis {axis} within slack {slack}; got {(box[0, axis], box[1, axis])}."
                )
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

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
        source_terms = (
            self.control_points.size
            + self.weights.size
            + self.u_knots.size
            + self.v_knots.size
        )
        budget.reserve(source_terms + box.size, 1024 + 24 * source_terms)
        source_key = canonical_fingerprint(
            {
                "kind": "source-bspline-surface-value-bounds",
                "type": f"{type(self).__module__}.{type(self).__qualname__}",
                "degrees": (self.u_degree, self.v_degree),
                "coefficients": array_tree_fingerprint(self),
            }
        )
        cache_key = (source_key, 0, box.tobytes())
        cached = budget.source_derivative_bounds_cache.get(cache_key)
        if cached is not None:
            budget.reserve(12, 256 + cached[0].nbytes + cached[1].nbytes)
            return np.stack(cached)
        pieces = (
            self.bezier_pieces()
            if budget is None
            else _prepare_bspline_surface_spans(self, budget=budget)
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
            piece_hull(piece, tuple(box[0]), tuple(box[1]))
            for piece in pieces
            if all(
                piece.parameter_bounds[axis][1] >= box[0, axis]
                and piece.parameter_bounds[axis][0] <= box[1, axis]
                for axis in (0, 1)
            )
        ]
        boxes = np.stack(selected)
        result = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
        _retain_source_derivative_bounds(
            result[0], result[1], None, source_key, box.tobytes(), 0, budget=budget
        )
        return result

    def source_pieces(
        self, parameter_box: ConvertibleToArray, /, *, budget: CoordinateEnclosureBudget
    ) -> tuple[SurfacePiece[AbstractSurfacePatch], ...]:
        """Extract tensor Bernstein restrictions, retaining closed degenerate boxes."""
        box = np.asarray(
            self.validate_parameter_box(parameter_box, budget=budget), dtype=np.float64
        )
        lower, upper = box
        degenerate = bool(np.any(lower == upper))
        pieces = []
        for index, piece in enumerate(
            _prepare_bspline_surface_spans(self, budget=budget)
        ):
            bounds = np.asarray(piece.parameter_bounds, dtype=np.float64)
            piece_lower = np.maximum(bounds[:, 0], lower)
            piece_upper = np.minimum(bounds[:, 1], upper)
            if np.all(piece_upper >= piece_lower) and (
                degenerate or np.all(piece_upper > piece_lower)
            ):
                evaluator = BernsteinSurfacePiece(
                    piece.homogeneous_controls,
                    bounds[:, 0],
                    bounds[:, 1],
                    controls_lower=piece.homogeneous_lower,
                    controls_upper=piece.homogeneous_upper,
                )
                pieces.append(SurfacePiece(evaluator, piece_lower, piece_upper, index))
        return tuple(pieces)


def _prepare_bspline_surface_spans(
    surface: BSplineSurfacePatch, /, *, budget: CoordinateEnclosureBudget
) -> tuple[RationalBezierPiece, ...]:
    """Borrow tensor source spans from the spline-bank owner."""
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

        return _retain_bspline_spans(key, surface.bezier_pieces(), budget=budget)


class BSplineSpanSurface(AbstractSurfacePatch):
    """One tensor knot span of a live rational B-spline surface.

    ``spans`` is immutable host addressing of an active ``(u, v)`` span. All
    numerical source leaves stay those of ``source``; evaluation is the
    source span polynomial, extended exactly like the span's Bernstein piece.
    """

    source: BSplineSurfacePatch
    spans: tuple[int, int] = eqx.field(static=True)

    @checked
    def __init__(self, source: BSplineSurfacePatch, spans: tuple[int, int]) -> None:
        spans_ = tuple(spans)
        if len(spans_) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in spans_
        ):
            raise TypeError(
                f"BSplineSpanSurface.spans must contain two integers and not booleans; got {spans_!r}."
            )
        counts = source.control_points.shape[:2]
        degrees = (source.u_degree, source.v_degree)
        if not all(degrees[axis] <= spans_[axis] < counts[axis] for axis in (0, 1)):
            raise ValueError(
                f"BSplineSpanSurface.spans must address active spans between degrees {degrees} and control counts {counts}; got {spans_}."
            )
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

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
        span = self._span_box()
        if np.any(box[0] < span[0]) or np.any(box[1] > span[1]):
            raise ValueError(
                f"BSplineSpanSurface.parameter_box must lie inside its knot span {span}; got {box}."
            )
        return self.source.bounding_box(box, preparations=preparations, budget=budget)


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
    if curve.ambient_dimension != 3:
        raise ValueError(
            f"surface.profile must be three-dimensional; got {name} with dimension {curve.ambient_dimension}."
        )
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

    @checked
    def __init__(self, curve: AbstractCurve, direction: ConvertibleToArray) -> None:
        curve_ = _require_space_curve(curve, "curve")
        direction_ = _host_vector(direction, "direction")
        if direction_.size != 3 or not np.linalg.norm(direction_) > 0.0:
            raise ValueError(
                f"ExtrusionSurface.direction must be a nonzero three-dimensional vector; got {direction_.tolist()}."
            )
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

    def validate_parameter_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.validate_parameter_box(parameter_box, budget=budget)
        box = super().validate_parameter_box(parameter_box, budget=budget)
        self.curve.validate_query_range(box[0, 0], box[1, 0])
        return box

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
        base = self.curve.bounding_box(
            *_curve_box_range(self.curve, box[0, 0], box[1, 0]),
            preparations=preparations,
            budget=budget,
        )
        sweep = _interval_box(
            np.zeros(3), ((np.asarray(self.direction), (box[0, 1], box[1, 1])),)
        )
        return np.stack((base[0] + sweep[0], base[1] + sweep[1]))

    def source_pieces(
        self, parameter_box: ConvertibleToArray, /, *, budget: CoordinateEnclosureBudget
    ) -> tuple[SurfacePiece[AbstractSurfacePatch], ...]:
        """Sweep each profile restriction and retain its authored parameter interval."""
        box = np.asarray(
            self.validate_parameter_box(parameter_box, budget=budget), dtype=np.float64
        )
        lower, upper = box
        pieces = []
        for index, piece in enumerate(
            self.curve.source_pieces(lower[0], upper[0], coordinate_budget=budget)
        ):
            evaluator = ExtrusionSurface(piece.evaluator, self.direction)
            piece_lower, piece_upper = lower.copy(), upper.copy()
            piece_lower[0], piece_upper[0] = piece.lower, piece.upper
            pieces.append(SurfacePiece(evaluator, piece_lower, piece_upper, index))
        return tuple(pieces)

    def native_period_symbols(self) -> tuple[str | None, str | None]:
        """Return period proofs from this sweep and its source profile."""
        return (
            (self.curve.native_period_symbol(), None)
            if type(self) is ExtrusionSurface
            else (None, None)
        )


def axis_frame(axis: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
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
    arithmetic over the stored axis coefficients. Vectors ``(..., 3)``
    pair with angle intervals of the leading shape ``(...)``.
    """

    bounds = (np.asarray(angle[0]), np.asarray(angle[1]))
    cosine_lower, cosine_upper = _array_interval_bounds(interval_cos(bounds))
    sine_lower, sine_upper = _array_interval_bounds(interval_sin(bounds))
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
            raise ValueError(
                f"rotation.derivative_order must be zero through two; got {derivative}."
            )
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
    products = _array_interval_bounds(interval_multiply((axis, axis), (lower, upper)))
    dot = _array_interval_bounds(
        interval_add(
            interval_add(
                (products[0][..., 0], products[1][..., 0]),
                (products[0][..., 1], products[1][..., 1]),
            ),
            (products[0][..., 2], products[1][..., 2]),
        )
    )
    projection = interval_multiply((dot[0][..., None], dot[1][..., None]), axial)
    return _array_interval_bounds(
        interval_add(
            interval_add(
                interval_multiply((lower, upper), even), interval_multiply(cross, odd)
            ),
            interval_multiply((axis, axis), projection),
        )
    )


@dataclass(frozen=True, slots=True)
class MeridianSpans:
    """Prepared rational hodographs of every span of a B-spline meridian.

    On a span with local parameter ``t`` and homogeneous numerator ``P`` and
    weight ``w`` of degree ``p``, the ``k``-th parameter derivative is the
    rational Bezier ``D_k / w^(k+1)`` of degree ``(k+1) p``, with
    ``D_0 = P`` and ``D_(k+1) = D_k' w - (k+1) D_k w'`` (degree-elevated once)
    scaled exactly by the span width. ``hodographs[k]`` holds outward
    enclosures of the rational coefficients ``(D_k, w^(k+1))``, shaped
    ``(span, (k+1) p + 1, 4)``. Positive weights make every denominator
    coefficient positive, so each derivative lies in the hull of its restricted
    control quotients, with no quotient-recurrence dependency loss. ``bending``
    likewise encloses ``(D_1 x D_1', w^4)``, the rational Bezier ``c' x c''``:
    ``D_1 x D_2 = w (D_1 x D_1')`` because ``D_1 x D_1 = 0``.
    Interior ``junctions`` carry one-sided equality proofs: ``g1`` (equal points,
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
class MeridianFrame:
    """Represent a planar meridian proved about an axis with squared norm one.

    Every control point lies in the plane through the axis with normal
    ``normal`` (an enclosure of the unit normal), so positive weights keep the
    whole profile in it.
    """

    axis: np.ndarray
    origin: np.ndarray
    normal: tuple[np.ndarray, np.ndarray]
    spans: MeridianSpans


@dataclass(frozen=True, slots=True)
class MeridianJets:
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


# Third meridian jets give second jets of a parallel meridian.
_MERIDIAN_ORDER = 3
# Hodograph key of the bending numerator ``c' x c''`` after the jet orders.
_BENDING = _MERIDIAN_ORDER + 1
# Overlaps per admitted restriction batch; bounds transient ledger scratch.
_MERIDIAN_CHUNK = 4096


@dataclass(frozen=True, slots=True)
class MeridianOverlaps:
    """Closed span overlaps of the distinct ranges of one batch, and their jets.

    A sweep's meridian jets depend only on a row's meridian range, and rows
    of one batch repeat few distinct ranges, so overlaps are formed and
    enclosed per distinct range. ``index`` gathers each row's overlaps (in
    row order, starting at ``offsets``) from the distinct ones. ``jets``
    holds each hodograph key enclosed so far over the distinct overlaps:
    orders ``0..3`` and ``_BENDING``; each is enclosed and charged once.
    """

    spans: MeridianSpans
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


@contextmanager
def shared_meridian_jets(
    *,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> Iterator[list[MeridianOverlaps]]:
    """Borrow overlap jets or retain them for this explicit chart-row request."""
    if meridian_jets is not None:
        yield meridian_jets
        return
    jets: list[MeridianOverlaps] = []
    try:
        yield jets
    finally:
        jets.clear()


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
    """Compute rational Bernstein coefficients of a vector times a scalar polynomial."""
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
    """Compute rational Bernstein coefficients of two 3D polynomials' cross product."""
    left, right = len(first) - 1, len(second) - 1
    result = [[Fraction()] * 3 for _ in range(left + right + 1)]
    for i, vector in enumerate(first):
        for j, other in enumerate(second):
            factor = Fraction(comb(left, i) * comb(right, j), comb(left + right, i + j))
            row = result[i + j]
            for axis, entry in enumerate(rational_cross(vector, other)):
                row[axis] += factor * entry
    return [tuple(row) for row in result]


def _exact_hodographs(
    piece: RationalBezierPiece, /
) -> list[tuple[_ExactBernstein, list[Fraction]]]:
    """Return Fraction ``(D_k, w^(k+1))`` coefficients for ``k = 0.._MERIDIAN_ORDER``."""
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
    """Prove G1, C1 and parallel-C1 by rational endpoint-hodograph comparisons."""
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
        and not any(rational_cross(incoming, outgoing))
        and sum(a * b for a, b in zip(incoming, outgoing, strict=True)) > 0
    )
    c1 = point == joined and incoming == outgoing and any(incoming)
    parallel = c1 and rational_cross(incoming, bending) == rational_cross(
        outgoing, turning
    )
    return g1, c1, parallel


def _outward_spans(
    spans: list[tuple[_ExactBernstein, list[Fraction]]], /
) -> tuple[np.ndarray, np.ndarray]:
    """Return outward ``(span, coefficient, 4)`` enclosures of rational Beziers."""
    rows = [
        [(*vector, weight) for vector, weight in zip(numerator, denominator, strict=True)]
        for numerator, denominator in spans
    ]
    return (
        np.asarray(
            [[[fraction_down(entry) for entry in row] for row in span] for span in rows]
        ),
        np.asarray(
            [[[fraction_up(entry) for entry in row] for row in span] for span in rows]
        ),
    )


def _prepare_meridian_spans(curve: BSplineCurve, /) -> MeridianSpans:
    """Prepare rational hodograph coefficients, then round outward once per source."""
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
    return MeridianSpans(
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


@checked
def prepare_meridian_spans(
    curve: AbstractCurve, /, *, budget: CoordinateEnclosureBudget
) -> MeridianSpans | None:
    """Prepared hodographs of a positively weighted spatial B-spline, or None.

    A budget retains one preparation per coefficient-derived source identity, so batches
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
    # Rational Boehm extraction and hodograph products: the degree-(k+1)p
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


@checked
def closure_tangent_continuous(
    patch: AbstractSurfacePatch,
    axis: int,
    lower: float,
    upper: float,
    /,
    *,
    tangent: bool = True,
) -> bool:
    """Prove that a chart's two sides on ``axis`` glue into one seam.

    A proved native ``2 pi`` period glues them. Otherwise the swept B-spline
    profile of a revolution (``axis`` one) or extrusion (``axis`` zero) must
    close exactly over its whole domain; with ``tangent`` the closure is also
    tangent-continuous (G1), which keeps every normal offset glued.
    """

    if patch.native_period_symbols()[axis] == "two_pi" and upper - lower == _TWO_PI:
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
    return not any(rational_cross(first, last)) and (
        sum((a * b for a, b in zip(first, last, strict=True)), Fraction()) > 0
    )


def exact_unit(axis: np.ndarray, /) -> bool:
    return sum(Fraction(float(value)) ** 2 for value in axis) == 1


@checked
def meridian_frame(
    surface: RevolutionSurface, /, *, budget: CoordinateEnclosureBudget
) -> MeridianFrame | None:
    """Prove an exactly planar meridian containing an exactly unit axis, or None."""

    curve = surface.curve
    axis = np.asarray(surface.axis_direction, dtype=np.float64)
    origin = np.asarray(surface.axis_origin, dtype=np.float64)
    if not isinstance(curve, BSplineCurve) or not exact_unit(axis):
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
        (
            cross
            for cross in (rational_cross(unit, row) for row in relative)
            if any(cross)
        ),
        None,
    )
    if normal is None or any(
        sum(a * b for a, b in zip(row, normal, strict=True)) for row in relative
    ):
        return None
    spans = prepare_meridian_spans(curve, budget=budget)
    if spans is None:
        return None
    norm_lower, norm_upper = square_root_interval(
        sum((value * value for value in normal), Fraction())
    )
    lower = np.asarray(
        [
            fraction_down(value / Fraction(norm_upper if value >= 0 else norm_lower))
            for value in normal
        ]
    )
    upper = np.asarray(
        [
            fraction_up(value / Fraction(norm_lower if value >= 0 else norm_upper))
            for value in normal
        ]
    )
    return MeridianFrame(axis, origin, (lower, upper), spans)


def _revolution_ranges(
    surface: RevolutionSurface, boxes: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Validate chart boxes of a B-spline meridian revolution; clipped v ranges."""
    curve = surface.curve
    if not isinstance(curve, BSplineCurve):
        raise TypeError(
            f"RevolutionSurface.curve must be a B-spline meridian for batched ranges; got {type(curve).__name__}."
        )
    if not np.all(np.isfinite(boxes)):
        raise ValueError(
            f"RevolutionSurface.parameter_boxes must be finite; got {boxes}."
        )
    if np.any(boxes[:, 1] < boxes[:, 0]):
        raise ValueError(
            f"RevolutionSurface.parameter_boxes must satisfy lower <= upper; got {boxes}."
        )
    if np.any(boxes[:, 1, 0] - boxes[:, 0, 0] > _TWO_PI * (1.0 + 1.0e-12)):
        raise ValueError(
            f"RevolutionSurface.parameter_boxes must cover at most one turn on axis zero; got {boxes[:, :, 0]}."
        )
    first, last = boxes[:, 0, 1], boxes[:, 1, 1]
    lower, upper = curve.parameter_domain
    slack = 1.0e-12 * max(1.0, abs(lower), abs(upper))
    ranged = first < last
    if np.any(ranged & ((first < lower - slack) | (last > upper + slack))):
        raise ValueError(
            f"RevolutionSurface.meridian_ranges must lie inside the carrier domain {(lower, upper)} within slack {slack}; got ranges {(first, last)}."
        )
    if np.any(~ranged & ((first < lower) | (first > upper))):
        raise ValueError(
            f"RevolutionSurface.meridian_points must lie inside the carrier domain {(lower, upper)}; got {first[~ranged]}."
        )
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
    upward in binary64 rather than through a host Fraction ratio.
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
            raise RationalEnclosureError(
                f"meridian.denominator must have positive restricted weight lower bounds; got {weights[0]}."
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
    spans: MeridianSpans,
    first: np.ndarray,
    last: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> MeridianOverlaps:
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
        raise ValueError(
            f"meridian.parameter_ranges must each select a source span; got {ranges}."
        )
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
    return MeridianOverlaps(
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


def evaluate_meridian_jets(
    spans: MeridianSpans,
    first: np.ndarray,
    last: np.ndarray,
    order: int,
    /,
    *,
    bending: bool = False,
    budget: CoordinateEnclosureBudget,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> MeridianJets:
    """Closed-span overlap jets ``0..order`` of every range ``[first, last]``.

    Every overlap of every distinct range is enclosed in admitted batches of
    at most ``_MERIDIAN_CHUNK`` restrictions, then gathered to the rows; no
    per-row host program is formed. ``bending`` appends ``c' x c''``. Inside
    ``shared_meridian_jets`` the hodographs already enclosed for identical
    ranges are reused, not redone.
    """
    if order not in range(_MERIDIAN_ORDER + 1):
        raise ValueError(
            f"meridian.jet_order must be zero through {_MERIDIAN_ORDER}; got {order!r}."
        )
    shared = meridian_jets
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
        overlaps = _meridian_overlaps(spans, first, last, budget=budget)
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
        # Two full de Casteljau triangles and the four-corner control
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
    return MeridianJets(
        overlaps.rows,
        overlaps.offsets,
        overlaps.widths,
        tuple(
            (overlaps.jets[key][0][overlaps.index], overlaps.jets[key][1][overlaps.index])
            for key in keys
        ),
    )


def junction_rows(
    spans: MeridianSpans,
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


def interval_cross(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray], /
) -> tuple[np.ndarray, np.ndarray]:

    lead, trail = np.asarray((1, 2, 0)), np.asarray((2, 0, 1))
    return _array_interval_bounds(
        interval_subtract(
            interval_multiply(
                (first[0][..., lead], first[1][..., lead]),
                (second[0][..., trail], second[1][..., trail]),
            ),
            interval_multiply(
                (first[0][..., trail], first[1][..., trail]),
                (second[0][..., lead], second[1][..., lead]),
            ),
        )
    )


def interval_dot(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray], /
) -> tuple[np.ndarray, np.ndarray]:
    """Outward dot product over the last axis."""

    products = _array_interval_bounds(interval_multiply(first, second))
    total = (products[0][..., 0], products[1][..., 0])
    for axis in range(1, products[0].shape[-1]):
        total = interval_add(total, (products[0][..., axis], products[1][..., axis]))
    return np.asarray(total[0]), np.asarray(total[1])


def directed_sum_of_squares(values: np.ndarray, direction: float, /) -> np.ndarray:
    total = np.zeros(values.shape[:-1])
    for axis in range(values.shape[-1]):
        total = np.nextafter(
            total + np.nextafter(values[..., axis] * values[..., axis], direction),
            direction,
        )
    return np.maximum(total, 0.0)


def interval_norm(
    value: tuple[np.ndarray, np.ndarray], /
) -> tuple[np.ndarray, np.ndarray]:
    """Outward ``[min, max]`` Euclidean norms of vector boxes over the last axis."""
    separated = np.maximum(np.maximum(value[0], -value[1]), 0.0)
    magnitude = np.maximum(np.abs(value[0]), np.abs(value[1]))
    with np.errstate(over="ignore", invalid="ignore"):
        lower = np.maximum(
            np.nextafter(np.sqrt(directed_sum_of_squares(separated, -np.inf)), -np.inf),
            0.0,
        )
        upper = np.nextafter(np.sqrt(directed_sum_of_squares(magnitude, np.inf)), np.inf)
    return lower, np.where(np.all(np.isfinite(magnitude), axis=-1), upper, np.inf)


def parallel_meridian_jets(
    frame: MeridianFrame,
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

    origin = (frame.origin, frame.origin)
    value, first = jets[0], jets[1]
    side = interval_dot(
        interval_cross(
            (frame.axis, frame.axis),
            _array_interval_bounds(interval_subtract(value, origin)),
        ),
        frame.normal,
    )
    positive = (side[0] > 0)[:, None]
    oriented = (
        np.where(positive, frame.normal[0], -frame.normal[1]),
        np.where(positive, frame.normal[1], -frame.normal[0]),
    )
    speed_lower, speed_upper = interval_norm(first)
    regular = ((side[0] > 0) | (side[1] < 0)) & (speed_lower > 0)
    speed = (speed_lower[:, None], speed_upper[:, None])
    scale = (np.asarray(distance), np.asarray(distance))
    tangent = _array_interval_bounds(interval_divide(first, speed))
    result = [
        _array_interval_bounds(
            interval_subtract(
                interval_add(
                    value, interval_multiply(scale, interval_cross(oriented, tangent))
                ),
                origin,
            )
        )
    ]
    if len(jets) < 3:
        return tuple(result), regular
    second = jets[2]
    rate_lower, rate_upper = interval_dot(tangent, second)
    rate = (rate_lower[:, None], rate_upper[:, None])
    turning = _array_interval_bounds(
        interval_divide(
            interval_subtract(second, interval_multiply(tangent, rate)), speed
        )
    )
    result.append(
        _array_interval_bounds(
            interval_add(
                first, interval_multiply(scale, interval_cross(oriented, turning))
            )
        )
    )
    if len(jets) < 4:
        return tuple(result), regular
    third = jets[3]
    acceleration_lower, acceleration_upper = _array_interval_bounds(
        interval_add(interval_dot(turning, second), interval_dot(tangent, third))
    )
    acceleration = (acceleration_lower[:, None], acceleration_upper[:, None])
    bending = _array_interval_bounds(
        interval_divide(
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
    )
    result.append(
        _array_interval_bounds(
            interval_add(
                second, interval_multiply(scale, interval_cross(oriented, bending))
            )
        )
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
    spans: MeridianSpans,
    boxes: np.ndarray,
    order: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation jets of a B-spline meridian's batched closed-span jets.

    Order two is unbounded on rows meeting a junction without a C1 equality
    proof; across a proved C1 junction the one-sided second jets bound the
    Lipschitz first derivative.
    """
    first, last = _revolution_ranges(surface, boxes)
    if not boxes.shape[0]:
        shape = (0, 3, 2) + ((2,) if order == 2 else ())
        return np.zeros(shape), np.zeros(shape)
    origin = np.asarray(surface.axis_origin, dtype=np.float64)
    jets = evaluate_meridian_jets(
        spans, first, last, order, budget=budget, meridian_jets=meridian_jets
    )
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
        rough = junction_rows(spans, first, last, spans.c1, closed=True)
        lower[rough], upper[rough] = -np.inf, np.inf
    return lower, upper


def _offset_meridian_jet_bounds(
    surface: OffsetSurface,
    frame: MeridianFrame,
    boxes: np.ndarray,
    order: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose rotation jets of the parallel meridian of an offset revolution.

    ``o + R(u)(c - o) + d R(u) n = o + R(u)(q - o)``. Rows whose span hull
    straddles an unproved G1 junction stay unbounded: the offset itself is
    discontinuous there. Order two also requires every met junction to keep
    each parallel curve C1.
    """
    base = surface.base
    if not isinstance(base, RevolutionSurface):
        raise TypeError(
            f"OffsetSurface.base must be a revolution for parallel-meridian jets; got {type(base).__name__}."
        )
    first, last = _revolution_ranges(base, boxes)
    shape = (boxes.shape[0], 3, 2) + ((2,) if order == 2 else ())
    if not boxes.shape[0]:
        return np.zeros(shape), np.zeros(shape)
    jets = evaluate_meridian_jets(
        frame.spans, first, last, order + 1, budget=budget, meridian_jets=meridian_jets
    )
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        parallel, regular = parallel_meridian_jets(
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
    usable = np.logical_and.reduceat(regular, jets.offsets) & ~junction_rows(
        frame.spans, first, last, frame.spans.g1, closed=False
    )
    if order == 2:
        usable &= ~junction_rows(
            frame.spans, first, last, frame.spans.parallel_c1, closed=True
        )
    lower[~usable], upper[~usable] = -np.inf, np.inf
    return lower, upper


@checked
def revolution_meridian(
    surface: AbstractSurfacePatch, /
) -> tuple[RevolutionSurface, float] | None:
    """A revolution, or a normal offset of one, with its signed distance."""
    if isinstance(surface, RevolutionSurface):
        return surface, 0.0
    if isinstance(surface, OffsetSurface) and isinstance(surface.base, RevolutionSurface):
        return surface.base, float(surface.distance)
    return None


@checked
def meridian_second_jets(
    surface: AbstractSurfacePatch,
    boxes: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray] | None:
    """Row hulls of ``w - o, w', w''`` of the swept meridian ``w`` of chart boxes.

    ``w`` is a revolution's B-spline meridian, or an offset's parallel
    meridian, about an exactly unit axis, so ``|x_uu| = |k x (w - o)|``,
    ``|x_uv| = |k x w'|`` and ``|x_vv| = |w''|`` at every chart point. Returns
    the hulls, the unit axis and the rows whose second jets bound a C1 sweep;
    None when the surface has no such batched meridian structure.
    """
    resolved = revolution_meridian(surface)
    if resolved is None:
        return None
    revolution, distance = resolved
    axis = np.asarray(revolution.axis_direction, dtype=np.float64)
    origin = np.asarray(revolution.axis_origin, dtype=np.float64)
    if isinstance(surface, OffsetSurface):
        frame = meridian_frame(revolution, budget=budget)
        if frame is None:
            return None
        spans = frame.spans
    else:
        frame = None
        spans = prepare_meridian_spans(revolution.curve, budget=budget)
        if spans is None or not exact_unit(axis):
            return None
    first, last = _revolution_ranges(revolution, boxes)
    if not boxes.shape[0]:
        empty = (np.zeros((0, 3)), np.zeros((0, 3)))
        return [empty, empty, empty], axis, np.zeros((0,), dtype=np.bool_)
    if frame is None:
        jets = evaluate_meridian_jets(
            spans, first, last, 2, budget=budget, meridian_jets=meridian_jets
        )
        position = jets.hull(0)
        hull = [
            (
                np.nextafter(position[0] - origin, -np.inf),
                np.nextafter(position[1] - origin, np.inf),
            ),
            jets.hull(1),
            jets.hull(2),
        ]
        return hull, axis, ~junction_rows(spans, first, last, spans.c1, closed=True)
    jets = evaluate_meridian_jets(
        spans, first, last, 3, budget=budget, meridian_jets=meridian_jets
    )
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        parallel, regular = parallel_meridian_jets(frame, jets.jets, distance)
    hull = [
        (
            np.minimum.reduceat(lower, jets.offsets, axis=0),
            np.maximum.reduceat(upper, jets.offsets, axis=0),
        )
        for lower, upper in parallel
    ]
    smooth = (
        np.logical_and.reduceat(regular, jets.offsets)
        & ~junction_rows(spans, first, last, spans.g1, closed=False)
        & ~junction_rows(spans, first, last, spans.parallel_c1, closed=True)
    )
    return hull, axis, smooth


# Equal closed pieces per range of an offset's meridian value enclosure.
_VALUE_PIECES = 16


@checked
def meridian_value_bounds(
    surface: AbstractSurfacePatch,
    boxes: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Batched value boxes of a B-spline-meridian revolution or its offset.

    ``o + R(u)(w - o)`` with ``w`` the meridian, or an offset's parallel
    meridian, rotated by the same interval rotation the source evaluates.
    Offset rows whose parallel jets are not regular, or that straddle an
    unproved G1 junction, stay unbounded. None without this structure.
    """

    resolved = revolution_meridian(surface)
    if resolved is None:
        return None
    revolution, distance = resolved
    origin = np.asarray(revolution.axis_origin, dtype=np.float64)
    if isinstance(surface, OffsetSurface):
        frame = meridian_frame(revolution, budget=budget)
        if frame is None:
            return None
        spans, axis = frame.spans, frame.axis
    else:
        frame = None
        spans = prepare_meridian_spans(revolution.curve, budget=budget)
        if spans is None:
            return None
        axis = np.asarray(revolution.axis_direction, dtype=np.float64)
    first, last = _revolution_ranges(revolution, boxes)
    if not boxes.shape[0]:
        return np.zeros((0, 3)), np.zeros((0, 3))
    usable = np.ones((boxes.shape[0],), dtype=np.bool_)
    if frame is None:
        jets = evaluate_meridian_jets(
            spans, first, last, 0, budget=budget, meridian_jets=meridian_jets
        )
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
        jets = evaluate_meridian_jets(
            spans,
            cuts[:, :-1].reshape(-1),
            cuts[:, 1:].reshape(-1),
            1,
            budget=budget,
            meridian_jets=meridian_jets,
        )
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            parallel, regular = parallel_meridian_jets(frame, jets.jets, distance)
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
        ) & ~junction_rows(spans, first, last, spans.g1, closed=False)
    with np.errstate(invalid="ignore", over="ignore"):
        lower, upper = _array_interval_bounds(
            interval_add(
                (origin, origin),
                _rotation_jet_bounds(axis, relative, (boxes[:, 0, 0], boxes[:, 1, 0]), 0),
            )
        )
    lower[~usable], upper[~usable] = -np.inf, np.inf
    return lower, upper


# Equal parameter ranges covering a meridian for its convex-offset proof.
_CONVEXITY_RANGES = 256


@dataclass(frozen=True, slots=True)
class ConvexOffsetEvidence:
    """Report convex-layer embedding and the meridian's signed turning direction."""

    proved: bool
    turning_sign: int


@checked
def convex_meridian_offset(
    surface: OffsetSurface,
    glued: tuple[bool, bool],
    /,
    *,
    budget: CoordinateEnclosureBudget,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> ConvexOffsetEvidence:
    """Prove that every intermediate revolved parallel meridian embeds.

    The closed G1 meridian must turn once with strictly signed curvature.
    Forward parallel tangents preserve that convexity; separation from the
    rotation axis then embeds the nested, disjoint parallel layers.
    """

    unproved = ConvexOffsetEvidence(proved=False, turning_sign=0)
    base = surface.base
    if not isinstance(base, RevolutionSurface):
        return unproved
    frame = meridian_frame(base, budget=budget)
    if frame is None:
        return unproved
    spans = frame.spans
    closed_g1 = bool(np.all(spans.g1)) and glued == (True, True)
    edges = np.linspace(spans.starts[0], spans.ends[-1], _CONVEXITY_RANGES + 1)
    jets = evaluate_meridian_jets(
        spans,
        edges[:-1],
        edges[1:],
        2,
        bending=True,
        budget=budget,
        meridian_jets=meridian_jets,
    )
    value, velocity, _, bending = jets.jets
    origin, axis = (frame.origin, frame.origin), (frame.axis, frame.axis)
    eps = np.finfo(np.float64).eps
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        signed = interval_dot(bending, frame.normal)
        speed = interval_norm(velocity)[0]
        rate = np.nextafter(
            interval_norm(bending)[1] / np.nextafter(speed * speed, -np.inf), np.inf
        )
        # Upper total turning: per-range tangent-angle rates times widths,
        # with a margin for the binary64 width differences and summation.
        turning = np.nextafter(
            np.sum(np.nextafter(rate * jets.widths, np.inf))
            * (1 + 4 * _CONVEXITY_RANGES * eps),
            np.inf,
        )
        parallel, regular = parallel_meridian_jets(
            frame, jets.jets[:3], float(surface.distance)
        )
        forward = interval_dot(parallel[1], velocity)[0]
        sides = [
            interval_dot(interval_cross(axis, relative), frame.normal)
            for relative in (
                _array_interval_bounds(interval_subtract(value, origin)),
                parallel[0],
            )
        ]
    sign = 1 if np.all(signed[0] > 0) else -1 if np.all(signed[1] < 0) else 0
    one_turn = sign != 0 and bool(np.all(speed > 0)) and bool(turning < 4 * np.pi)
    forward_proved = bool(np.all(regular)) and bool(np.all(forward > 0))
    off_axis = all(bool(np.all(side[0] > 0)) for side in sides) or all(
        bool(np.all(side[1] < 0)) for side in sides
    )
    return ConvexOffsetEvidence(
        proved=closed_g1 and one_turn and forward_proved and off_axis,
        turning_sign=sign,
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
        distance = directed_sum_of_squares(separated, -np.inf)
        diameter = directed_sum_of_squares(np.nextafter(upper - lower, np.inf), np.inf)
        ratio = np.nextafter(diameter / np.nextafter(4.0 * distance, -np.inf), np.inf)
        angle = 2.0 * np.arcsin(
            np.minimum(1.0, np.nextafter(np.sqrt(np.minimum(ratio, 1.0)), np.inf))
        )
    angle = np.nextafter(angle * (1 + 32 * np.finfo(np.float64).eps), np.inf)
    angle = np.where(ratio >= 1.0, np.nextafter(np.pi, np.inf), angle)
    finite = np.all(np.isfinite(lower) & np.isfinite(upper), axis=-1)
    return np.where(finite & (distance > 0), angle, np.inf)


@checked
def meridian_normal_turns(
    revolution: RevolutionSurface,
    distance: float,
    charts: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    meridian_jets: list[MeridianOverlaps] | None = None,
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

    frame = meridian_frame(revolution, budget=budget)
    if frame is None:
        return None
    boxes = np.stack((np.min(charts, axis=1), np.max(charts, axis=1)), axis=1)
    first, last = _revolution_ranges(revolution, boxes)
    result = np.full((boxes.shape[0],), np.nan)
    if not boxes.shape[0]:
        return result
    # The bending hodograph encloses c' x c'' without the c'' jet.
    jets = evaluate_meridian_jets(
        frame.spans,
        first,
        last,
        1,
        bending=True,
        budget=budget,
        meridian_jets=meridian_jets,
    )
    value, velocity, cross = jets.jets
    axis = (frame.axis, frame.axis)
    eps = np.finfo(np.float64).eps
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        side = interval_dot(
            interval_cross(
                axis,
                _array_interval_bounds(
                    interval_subtract(value, (frame.origin, frame.origin))
                ),
            ),
            frame.normal,
        )
        radius = np.where(side[0] > 0, side[0], -side[1])
        speed = interval_norm(velocity)[0]
        regular = ((side[0] > 0) | (side[1] < 0)) & (speed > 0)
        square = np.nextafter(speed * speed, -np.inf)
        bending = interval_norm(cross)[1]
        rate = np.nextafter(bending / square, np.inf)
        axial = interval_dot(velocity, axis)
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
        smooth = ~junction_rows(frame.spans, first, last, frame.spans.g1, closed=False)
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

    @checked
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
            raise ValueError(
                f"RevolutionSurface.axis must have three-dimensional origin and direction; got shapes {(origin.shape, direction.shape)}."
            )
        if abs(np.linalg.norm(direction) - 1.0) > _FRAME_TOLERANCE:
            raise ValueError(
                f"RevolutionSurface.axis_direction must be a unit vector within {_FRAME_TOLERANCE}; got {direction.tolist()}."
            )
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

    def validate_parameter_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.validate_parameter_box(parameter_box, budget=budget)
        box = super().validate_parameter_box(parameter_box, budget=budget)
        self.curve.validate_query_range(box[0, 1], box[1, 1])
        return box

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
        profile = self.curve.bounding_box(
            *_curve_box_range(self.curve, box[0, 1], box[1, 1]),
            preparations=preparations,
            budget=budget,
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
        first, second = axis_frame(axis)
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

    def source_pieces(
        self, parameter_box: ConvertibleToArray, /, *, budget: CoordinateEnclosureBudget
    ) -> tuple[SurfacePiece[AbstractSurfacePatch], ...]:
        """Sweep each profile restriction and retain its authored parameter interval."""
        box = np.asarray(
            self.validate_parameter_box(parameter_box, budget=budget), dtype=np.float64
        )
        lower, upper = box
        pieces = []
        for index, piece in enumerate(
            self.curve.source_pieces(lower[1], upper[1], coordinate_budget=budget)
        ):
            evaluator = RevolutionSurface(
                piece.evaluator, self.axis_origin, self.axis_direction
            )
            piece_lower, piece_upper = lower.copy(), upper.copy()
            piece_lower[1], piece_upper[1] = piece.lower, piece.upper
            pieces.append(SurfacePiece(evaluator, piece_lower, piece_upper, index))
        return tuple(pieces)

    def native_period_symbols(self) -> tuple[str | None, str | None]:
        """Return period proofs from this sweep and its source profile."""
        return (
            ("two_pi", self.curve.native_period_symbol())
            if type(self) is RevolutionSurface
            else (None, None)
        )


class RuledSurface(AbstractSurfacePatch):
    """Ruled surface ``(1 - v) * first(u) + v * second(u)`` with ``v`` in ``[0, 1]``."""

    first: AbstractCurve
    second: AbstractCurve

    @checked
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

    def validate_parameter_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.validate_parameter_box(parameter_box, budget=budget)
        box = super().validate_parameter_box(parameter_box, budget=budget)
        if box[0, 1] < 0.0 or box[1, 1] > 1.0:
            raise ValueError(
                f"RuledSurface.blend_range must lie inside [0, 1]; got {(box[0, 1], box[1, 1])}."
            )
        self.first.validate_query_range(box[0, 0], box[1, 0])
        self.second.validate_query_range(box[0, 0], box[1, 0])
        return box

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
        # Each point is a convex combination of the two rails.
        first = self.first.bounding_box(
            box[0, 0], box[1, 0], preparations=preparations, budget=budget
        )
        second = self.second.bounding_box(
            box[0, 0], box[1, 0], preparations=preparations, budget=budget
        )
        return np.stack(
            (np.minimum(first[0], second[0]), np.maximum(first[1], second[1]))
        )

    def source_pieces(
        self, parameter_box: ConvertibleToArray, /, *, budget: CoordinateEnclosureBudget
    ) -> tuple[SurfacePiece[AbstractSurfacePatch], ...]:
        """Intersect both profile partitions before assembling each ruled restriction."""
        box = np.asarray(
            self.validate_parameter_box(parameter_box, budget=budget), dtype=np.float64
        )
        lower, upper = box
        first = self.first.source_pieces(lower[0], upper[0], coordinate_budget=budget)
        second = self.second.source_pieces(lower[0], upper[0], coordinate_budget=budget)
        if lower[0] == upper[0]:
            return tuple(
                SurfacePiece(RuledSurface(a.evaluator, b.evaluator), lower, upper, index)
                for index, (a, b) in enumerate((a, b) for a in first for b in second)
            )
        breaks = sorted(
            {
                lower[0],
                upper[0],
                *(piece.lower for piece in (*first, *second)),
                *(piece.upper for piece in (*first, *second)),
            }
        )
        pieces = []
        for index, (start, end) in enumerate(zip(breaks[:-1], breaks[1:], strict=True)):
            midpoint = 0.5 * (start + end)
            a = next(piece for piece in first if piece.lower <= midpoint <= piece.upper)
            b = next(piece for piece in second if piece.lower <= midpoint <= piece.upper)
            evaluator = RuledSurface(a.evaluator, b.evaluator)
            piece_lower, piece_upper = lower.copy(), upper.copy()
            piece_lower[0], piece_upper[0] = start, end
            pieces.append(SurfacePiece(evaluator, piece_lower, piece_upper, index))
        return tuple(pieces)

    def native_period_symbols(self) -> tuple[str | None, str | None]:
        """Admit the ruling period only when both source profiles prove the same one."""
        if type(self) is not RuledSurface:
            return None, None
        first, second = (
            self.first.native_period_symbol(),
            self.second.native_period_symbol(),
        )
        return (first if first == second else None), None


class OffsetSurface(AbstractSurfacePatch):
    """Signed normal-offset operation tree with bounded numerical jets."""

    base: AbstractSurfacePatch
    distance: Array
    sphere_normal_depth: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        base: AbstractSurfacePatch,
        distance: ConvertibleToArray,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                base,
                distance,
            ),
            budget=budget,
        ) as budget:
            self.base = base
            self.distance = jnp.asarray(
                _host_scalar(distance, "distance"), dtype=jnp.float64
            )
            equivalence = sphere_source_equivalence(base, budget=budget)
            depth = 0
            if equivalence is not None and equivalence[1] != 0:
                depth = len(offset_chain(base)[1])
            self.sphere_normal_depth = depth

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.base.periods

    def validate_parameter_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.validate_parameter_box(parameter_box, budget=budget)
        box = self.base.validate_parameter_box(parameter_box, budget=budget)
        sphere = sphere_source_equivalence(self, budget=budget)
        if sphere is not None and sphere[1] == 0:
            raise ValueError(
                f"OffsetSurface.sphere_radius must remain nonzero; got radius {sphere[1]}."
            )
        if (
            float(self.distance) != 0.0
            and isinstance(self.base, ConePatch)
            and self.base.degenerate_isolines(box)
        ):
            raise ValueError(
                f"OffsetSurface.parameter_box must avoid the cone apex for a nonzero offset; got {box} and distance {self.distance}."
            )
        base = self.base
        if float(self.distance) != 0 and isinstance(base, TorusPatch):
            cosine, _ = _trig_ranges(box[0, 1], box[1, 1])
            radius = (
                float(base.major_radius) + float(base.minor_radius) * cosine[0],
                float(base.major_radius) + float(base.minor_radius) * cosine[1],
            )
            if radius[0] <= 0 <= radius[1]:
                raise ValueError(
                    f"OffsetSurface.base_radial_range must exclude zero to define the torus normal; got {radius}."
                )
        if isinstance(base, (SpherePatch, CylinderPatch, TorusPatch)):
            orientation = exact_frame_orientation(
                base.first_axis, base.second_axis, base.axis, budget=budget
            )
            if orientation is not None:
                original = (
                    float(base.minor_radius)
                    if isinstance(base, TorusPatch)
                    else float(base.radius)
                )
                radius = Fraction(original) + orientation * Fraction(float(self.distance))
                if radius == 0:
                    raise ValueError(
                        f"OffsetSurface.radial_coefficient must remain nonzero; got {radius}."
                    )
                if isinstance(base, TorusPatch):
                    cosine, _ = _trig_ranges(box[0, 1], box[1, 1])
                    values = (
                        Fraction(float(base.major_radius)) + radius * Fraction(cosine[0]),
                        Fraction(float(base.major_radius)) + radius * Fraction(cosine[1]),
                    )
                    if min(values) <= 0 <= max(values):
                        raise ValueError(
                            f"OffsetSurface.radial_range must exclude zero to preserve torus rank; got {values}."
                        )
        return box

    def degenerate_isolines(
        self, parameter_box: ConvertibleToArray, /
    ) -> tuple[tuple[int, float], ...]:
        return self.base.degenerate_isolines(parameter_box)

    def is_c1_on(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.is_c1_on(
                    parameter_box, preparations=preparations, budget=budget
                )
        return self.base.is_c1_on(parameter_box, preparations=preparations, budget=budget)

    def _normal(self, parameters: Array) -> Array:
        base = self.base
        for _ in range(self.sphere_normal_depth):
            if not isinstance(base, OffsetSurface):
                raise ValueError(
                    f"OffsetSurface.sphere_normal_depth must match an offset-chain prefix; got depth {self.sphere_normal_depth} and basis {type(base).__name__}."
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
            # still enclosed separately from the frame's orthogonality evidence.
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

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)

        sphere = sphere_source_equivalence(self, budget=budget)
        if sphere is not None:
            original, exact_radius = sphere
            source_box = original.bounding_box(
                box, preparations=preparations, budget=budget
            )
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
        equivalent = self.analytic_equivalent(budget=budget)
        if equivalent is not None:
            return equivalent.bounding_box(box, preparations=preparations, budget=budget)
        # A revolution offset rotates its parallel meridian; interval AD
        # of the normalized normal is unbounded over a full rotation.
        swept = meridian_value_bounds(self, box[None], budget=budget)
        if swept is not None:
            return np.stack((swept[0][0], swept[1][0]))

        lower_bounds, upper_bounds = [], []
        for piece in self.source_pieces(box, budget=budget):
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

    def analytic_equivalent(
        self, *, budget: CoordinateEnclosureBudget | None = None
    ) -> AbstractSurfacePatch | None:
        """Lower only when the existing float carrier represents the SAME map."""
        if budget is None:
            with coordinate_source_request((self,)) as budget:
                return self.analytic_equivalent(budget=budget)
        base = self.base
        distance = Fraction(float(self.distance))
        sphere = sphere_source_equivalence(self, budget=budget)
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
        match base:
            case CylinderPatch() | SpherePatch() | TorusPatch():
                return _radial_offset_equivalent(base, distance, budget=budget)
            case PlanePatch():
                return _plane_offset_equivalent(base, distance)
            case AbstractSurfacePatch():
                return None
            case _:
                raise TypeError(
                    f"OffsetSurface.base must be a native surface patch; got {type(base).__name__}."
                )

    def source_pieces(
        self, parameter_box: ConvertibleToArray, /, *, budget: CoordinateEnclosureBudget
    ) -> tuple[SurfacePiece[AbstractSurfacePatch], ...]:
        """Offset each validated basis restriction with unchanged box indices."""
        box = np.asarray(
            self.validate_parameter_box(parameter_box, budget=budget), dtype=np.float64
        )
        return tuple(
            SurfacePiece(
                OffsetSurface(piece.evaluator, self.distance, budget=budget),
                piece.lower,
                piece.upper,
                piece.index,
            )
            for piece in self.base.source_pieces(box, budget=budget)
        )

    def native_period_symbols(self) -> tuple[str | None, str | None]:
        """Retain the basis period proof for this offset operation."""
        return (
            self.base.native_period_symbols()
            if type(self) is OffsetSurface
            else (None, None)
        )


def _radial_offset_equivalent(
    base: CylinderPatch | SpherePatch | TorusPatch,
    distance: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> AbstractSurfacePatch | None:
    orientation = exact_frame_orientation(
        base.first_axis, base.second_axis, base.axis, budget=budget
    )
    if orientation is None:
        return None
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


def _plane_offset_equivalent(
    base: PlanePatch, distance: Fraction, /
) -> PlanePatch | None:
    normal = rational_cross(
        rational_vector(np.asarray(base.first_axis)),
        rational_vector(np.asarray(base.second_axis)),
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


@checked
def offset_chain(
    source: AbstractSurfacePatch, /
) -> tuple[AbstractSurfacePatch, tuple[OffsetSurface, ...]]:
    """Return the basis patch and outermost-first offset nodes without reading leaves."""
    offsets: list[OffsetSurface] = []
    while isinstance(source, OffsetSurface):
        offsets.append(source)
        source = source.base
    return source, tuple(offsets)


@checked
def exact_frame_orientation(
    first_axis: ConvertibleToArray,
    second_axis: ConvertibleToArray,
    axis: ConvertibleToArray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> Fraction | None:
    """Return the determinant when the rational frame Gram matrix is identity, or None."""
    if budget is None:
        with coordinate_source_request((first_axis, second_axis, axis)) as budget:
            return exact_frame_orientation(first_axis, second_axis, axis, budget=budget)
    columns = tuple(
        rational_vector(np.asarray(vector)) for vector in (first_axis, second_axis, axis)
    )
    if any(
        sum((a * b for a, b in zip(columns[i], columns[j], strict=True)), Fraction(0))
        != int(i == j)
        for i in range(3)
        for j in range(3)
    ):
        return None
    matrix = tuple(tuple(columns[column][row] for column in range(3)) for row in range(3))
    proof = prepare_exact_small_linear_actions(matrix, ((), (), ()), budget=budget)
    return proof.determinant if proof.successful else None


def _sphere_radius_proof(
    source: AbstractSurfacePatch, /, *, budget: CoordinateEnclosureBudget
) -> tuple[SpherePatch, Fraction, tuple[Fraction, ...]] | None:
    base, nodes = offset_chain(source)
    if not isinstance(base, SpherePatch):
        return None
    offsets = tuple(Fraction(float(node.distance)) for node in nodes)
    radius = Fraction(float(base.radius))
    terms = [radius]
    if not any(offsets):
        return base, radius, tuple(terms)
    orientation = exact_frame_orientation(
        base.first_axis, base.second_axis, base.axis, budget=budget
    )
    if orientation is None:
        return None
    for offset in reversed(offsets):
        if radius == 0 and offset:
            return None
        term = orientation * offset
        radius += term
        terms.append(term)
    return base, radius, tuple(terms)


@checked
def sphere_source_equivalence(
    source: AbstractSurfacePatch, /, *, budget: CoordinateEnclosureBudget | None = None
) -> tuple[SpherePatch, Fraction] | None:
    """Prove a sphere/normal-offset tree's signed radial expression.

    A nonzero offset is reduced only for an exactly orthogonal binary source
    frame; orthogonality is established by rational Gram-matrix equality. Radius
    sums remain Fractions even when no binary64 primitive could represent them.
    The proof retains the source operation tree and its IDs.
    """
    if budget is None:
        with coordinate_source_request((source,)) as budget:
            return sphere_source_equivalence(source, budget=budget)
    proof = _sphere_radius_proof(source, budget=budget)
    return None if proof is None else (proof[0], proof[1])


@checked
def sphere_source_radius_terms(
    source: AbstractSurfacePatch, /, *, budget: CoordinateEnclosureBudget | None = None
) -> tuple[SpherePatch, tuple[Fraction, ...]] | None:
    """Return radial coefficient terms without rounding their rational sum."""
    if budget is None:
        with coordinate_source_request((source,)) as budget:
            return sphere_source_radius_terms(source, budget=budget)
    proof = _sphere_radius_proof(source, budget=budget)
    return None if proof is None else (proof[0], proof[2])


@checked
def sphere_source_pole_normal_limits(
    source: AbstractSurfacePatch,
    parameters: ConvertibleToArray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss limits at the source's declared spherical poles."""
    if budget is None:
        with coordinate_source_request(
            (
                source,
                parameters,
            )
        ) as budget:
            return sphere_source_pole_normal_limits(source, parameters, budget=budget)
    charts = np.asarray(parameters, dtype=np.float64).reshape((-1, 2))
    normals = np.zeros((charts.shape[0], 3), dtype=np.float64)
    supported = np.zeros(charts.shape[0], dtype=np.bool_)
    equivalence = sphere_source_equivalence(source, budget=budget)
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
    """Source seam/isoline: one surface coordinate fixed, the other is t."""

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
                f"SurfaceIsoparametricCurve.definition must have a native surface, axis zero or one, and finite fixed coordinate; got surface {type(surface).__name__}, axis {fixed_axis!r}, and coordinate {fixed_value!r}."
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
        surface, _ = offset_chain(self.surface)
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

    def parameter_box(
        self,
        first: float,
        last: float,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.parameter_box(first, last, budget=budget)
        lower, upper = np.empty(2), np.empty(2)
        lower[self.fixed_axis] = upper[self.fixed_axis] = self.fixed_value
        lower[1 - self.fixed_axis], upper[1 - self.fixed_axis] = first, last
        return self.surface.validate_parameter_box(
            np.stack((lower, upper)), budget=budget
        )

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

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        self.validate_query_range(first, last)
        return self.surface.bounding_box(
            self.parameter_box(first, last, budget=budget),
            preparations=preparations,
            budget=budget,
        )

    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> bool:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.is_c1_on(
                    first, last, preparations=preparations, budget=budget
                )
        return self.surface.is_c1_on(
            self.parameter_box(first, last, budget=budget),
            preparations=preparations,
            budget=budget,
        )

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Isoline jets are the free-axis surface jets over its degenerate box.

        A swept planar meridian (a revolution or its offset) encloses them with
        its batched rational meridian jets; generic interval AD of an offset's
        normalized normal loses all dependency and does not converge.
        """
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.derivative_bounds(
                    first,
                    last,
                    order=order,
                    preparations=preparations,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
        if (
            isinstance(order, bool)
            or order not in (1, 2)
            or revolution_meridian(self.surface) is None
        ):
            return super().derivative_bounds(
                first,
                last,
                order=order,
                preparations=preparations,
                budget=budget,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            )
        box = self.parameter_box(first, last, budget=budget)
        lower, upper = self.surface.derivative_bounds_batch(
            box[None],
            order=order,
            preparations=preparations,
            budget=budget,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        free = 1 - self.fixed_axis
        if order == 1:
            return lower[0, :, free], upper[0, :, free]
        return lower[0, :, free, free], upper[0, :, free, free]

    def source_pieces(
        self,
        first: float,
        last: float,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> tuple[CurvePiece[AbstractCurve], ...]:
        """Restrict the source surface and retain its free-axis parameters."""
        return tuple(
            CurvePiece(
                SurfaceIsoparametricCurve(
                    piece.evaluator,
                    self.fixed_axis,
                    self.fixed_value,
                    parameter_range=(first, last) if first < last else None,
                ),
                float(piece.lower[1 - self.fixed_axis]),
                float(piece.upper[1 - self.fixed_axis]),
                piece.index,
            )
            for piece in self.surface.source_pieces(
                self.parameter_box(first, last, budget=coordinate_budget),
                budget=coordinate_budget,
            )
        )

    def native_period_symbol(self) -> str | None:
        """Return the source surface's period on the free chart axis."""
        return self.surface.native_period_symbols()[1 - self.fixed_axis]


def _validate_piece_interval(curve: AbstractCurve, first: float, last: float, /) -> None:
    if not (isfinite(first) and isfinite(last) and first <= last):
        raise ValueError(
            f"CurveRange.interval must be finite and ordered; got {(first, last)!r}"
        )
    if first < last:
        curve.validate_range(first, last)
    else:
        domain = curve.parameter_domain
        if domain is not None and not domain[0] <= first <= domain[1]:
            raise ValueError(
                f"CurveRange.parameter must lie inside the source curve domain; got {first!r}"
            )


@checked
def coefficient_enclosures(
    system: StrictModule, /
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Merge coefficient constructions that share a represented jaxpr constant.

    Distinct rational constructions can round to one constant array. Each
    matching construction contributes to its lower/upper hull, in tree order.
    """
    pieces = jax.tree_util.tree_leaves(
        system, is_leaf=lambda item: isinstance(item, CoefficientEnclosureSource)
    )
    bounds = []
    for piece in pieces:
        if isinstance(piece, CoefficientEnclosureSource):
            bounds.extend(piece.coefficient_enclosures())
    merged: dict[
        tuple[str, tuple[int, ...], bytes], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for nominal, lower, upper in bounds:
        key = (nominal.dtype.str, nominal.shape, nominal.tobytes())
        if key in merged:
            old_nominal, old_lower, old_upper = merged[key]
            merged[key] = (
                old_nominal,
                np.minimum(old_lower, lower),
                np.maximum(old_upper, upper),
            )
        else:
            merged[key] = nominal, lower, upper
    return tuple(merged.values())


def _bernstein(parameter: Array, degree: int, /) -> Array:
    binomial = jnp.asarray(
        [comb(degree, index) for index in range(degree + 1)], dtype=jnp.float64
    )
    return binomial * jnp.stack(
        [
            parameter**index * (1.0 - parameter) ** (degree - index)
            for index in range(degree + 1)
        ],
        axis=-1,
    )


def _bernstein_array(
    value: ArrayLike, /, *, shape: tuple[int, ...] | None = None
) -> Array:
    """Admit host coefficient banks without changing traced coefficient paths."""
    if isinstance(value, np.ndarray):
        host = np.asarray(value, dtype=np.float64)
        if shape is not None:
            host = host.reshape(shape)
        return jax.device_put(host)
    array = jnp.asarray(value, dtype=jnp.float64)
    return array if shape is None else array.reshape(shape)


class BernsteinSurfacePiece(AbstractSurfacePatch):
    """Rational Bernstein restriction carrying source construction enclosures."""

    controls: Array
    lower: Array
    upper: Array
    controls_lower: Array
    controls_upper: Array

    def __init__(
        self,
        controls: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        *,
        controls_lower: ArrayLike,
        controls_upper: ArrayLike,
    ) -> None:
        controls_ = _bernstein_array(controls)
        if controls_.ndim != 3 or controls_.shape[-1] != 4:
            raise ValueError(
                f"BernsteinSurfacePiece.controls must have shape (du+1, dv+1, 4); got {controls_.shape!r}"
            )
        self.controls = controls_
        self.lower = _bernstein_array(lower, shape=(2,))
        self.upper = _bernstein_array(upper, shape=(2,))
        self.controls_lower = _bernstein_array(controls_lower)
        self.controls_upper = _bernstein_array(controls_upper)

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameter_box,
                )
            ) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        box = self.validate_parameter_box(parameter_box, budget=budget)
        piece = RationalBezierPiece(
            np.asarray(self.controls),
            tuple(zip(np.asarray(self.lower), np.asarray(self.upper), strict=True)),
            (0, 0),
            np.asarray(self.controls_lower),
            np.asarray(self.controls_upper),
        )
        return piece_hull(piece, tuple(box[0]), tuple(box[1]))

    def evaluate(self, parameters: Array, /) -> Array:
        local = (parameters - self.lower) / (self.upper - self.lower)
        first = _bernstein(local[..., 0], self.controls.shape[0] - 1)
        second = _bernstein(local[..., 1], self.controls.shape[1] - 1)
        rows = first @ self.controls.reshape((self.controls.shape[0], -1))
        if parameters.ndim == 1:
            homogeneous = second @ rows.reshape((self.controls.shape[1], 4))
            return homogeneous[:3] / homogeneous[3]
        rows = rows.reshape((*parameters.shape[:-1], self.controls.shape[1], 4))
        homogeneous = (second[..., None, :] @ rows)[..., 0, :]
        return homogeneous[..., :3] / homogeneous[..., 3, None]

    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Propagate the retained construction enclosure of homogeneous controls."""
        return (
            (
                np.asarray(self.controls),
                np.asarray(self.controls_lower),
                np.asarray(self.controls_upper),
            ),
        )


class BernsteinCurvePiece(AbstractCurve):
    """Rational Bernstein restriction carrying source construction enclosures."""

    controls: Array
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    controls_lower: Array
    controls_upper: Array

    def __init__(
        self,
        controls: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        *,
        controls_lower: ArrayLike,
        controls_upper: ArrayLike,
    ) -> None:
        controls_ = jnp.asarray(controls, dtype=jnp.float64)
        if controls_.ndim != 2 or controls_.shape[-1] not in (3, 4):
            raise ValueError(
                f"BernsteinCurvePiece.controls must have shape (d+1, dim+1); got {controls_.shape!r}"
            )
        self.controls = controls_
        lower_, upper_ = float(np.asarray(lower)), float(np.asarray(upper))
        if not (isfinite(lower_) and isfinite(upper_) and lower_ < upper_):
            raise ValueError(
                f"BernsteinCurvePiece.interval must be finite and ordered; got {(lower_, upper_)!r}"
            )
        self.lower, self.upper = lower_, upper_
        self.controls_lower = jnp.asarray(controls_lower, dtype=jnp.float64)
        self.controls_upper = jnp.asarray(controls_upper, dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return self.controls.shape[-1] - 1

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return float(self.lower), float(self.upper)

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        first_, last_ = self.validate_range(first, last)
        piece = RationalBezierPiece(
            np.asarray(self.controls),
            (self.parameter_domain,),
            (0,),
            np.asarray(self.controls_lower),
            np.asarray(self.controls_upper),
        )
        return piece_hull(piece, (first_,), (last_,))

    def evaluate(self, parameters: Array, /) -> Array:
        local = (parameters - self.lower) / (self.upper - self.lower)
        homogeneous = _bernstein(local, self.controls.shape[0] - 1) @ self.controls
        return homogeneous[..., :-1] / homogeneous[..., -1, None]

    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Propagate the retained construction enclosure of homogeneous controls."""
        return (
            (
                np.asarray(self.controls),
                np.asarray(self.controls_lower),
                np.asarray(self.controls_upper),
            ),
        )


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


class CurveEvaluator(Protocol):
    """Evaluate a scalar parameter to coordinates or a coupled coordinate result."""

    def evaluate(self, parameters: Array, /) -> Array | CurveCoordinateResult: ...


def _curve_coordinate_value(curve: CurveEvaluator, parameter: Array, /) -> Array:
    value = curve.evaluate(parameter)
    match value:
        case CurveCoordinateResult():
            return value.point
        case _:
            return value


def _curve_value_and_first_jet(
    curve: CurveEvaluator, parameter: Array, /
) -> tuple[Array, Array]:
    """Compute the coordinate value and its scalar jet from one primal evaluation."""

    def value_with_aux(point: Array) -> tuple[Array, Array]:
        value = _curve_coordinate_value(curve, point)
        return value, value

    derivative, value = jax.jacfwd(value_with_aux, has_aux=True)(parameter)
    return value, derivative


def _surface_value_and_first_jets(
    patch: AbstractSurfacePatch, parameter: Array, /
) -> tuple[Array, Array]:
    """Compute the surface value and full differential from one primal evaluation."""

    def value_with_aux(point: Array) -> tuple[Array, Array]:
        value = patch.evaluate(point)
        return value, value

    differential, value = jax.jacfwd(value_with_aux, has_aux=True)(parameter)
    return value, differential


@eqx.filter_jit
def _evaluate_root_curve_coordinates(
    curve: CurveEvaluator,
    parameters: Array,
    /,
    *,
    with_derivative: bool,
    batch_capacity: int,
) -> Array:
    """Execute native-root coordinates in bounded device batches."""

    def evaluate(parameter: Array) -> Array:
        if with_derivative:
            value, derivative = _curve_value_and_first_jet(curve, parameter)
            return jnp.stack((value, derivative), axis=-1)
        return _curve_coordinate_value(curve, parameter)

    return jax.lax.map(evaluate, parameters[:, 0], batch_size=min(batch_capacity, 64))


@dataclass(frozen=True, slots=True)
class PreparedRootCurveHost:
    """Retain native-root execution without transferring inside its iterations.

    The source producer admits its coupled result or raises its convergence
    refusal before coordinates are extracted. The host boundary transfers
    the bounded compiled execution's requested outputs once.
    """

    curve: CurveEvaluator
    with_derivative: bool
    batch_capacity: int

    def evaluate_host_points(self, points: ConvertibleToArray, /) -> np.ndarray:
        parameters = np.asarray(points, dtype=np.float64)
        if parameters.ndim != 2 or parameters.shape[1] != 1:
            raise ValueError(
                f"curve.points must have shape (num_points, 1); got {parameters.shape}."
            )
        values = _evaluate_root_curve_coordinates(
            self.curve,
            jnp.asarray(parameters, dtype=jnp.float64),
            with_derivative=self.with_derivative,
            batch_capacity=self.batch_capacity,
        )
        return np.asarray(jax.device_get(values))


def prepare_curve_host(
    curve: CurveEvaluator,
    /,
    *,
    with_derivative: bool = False,
    batch_capacity: int = 64,
    budget: CoordinateEnclosureBudget | None = None,
) -> PreparedIntervalFunction | PreparedRootCurveHost:
    """Prepare bounded curve execution, optionally including the first jet.

    Arithmetic maps use the NumPy semantic plan. Curves containing a nominal
    root source retain compiled native-root execution with one host transfer.
    ``evaluate_host_points`` accepts ``(N, 1)`` parameter rows and returns
    ``(N, ambient)`` values, or ``(N, ambient, 2)`` with value and first jet
    on the last axis when a derivative is requested.
    A supplied budget owns host preparation only; neither numerical execution
    route captures the resource view.
    """
    if type(with_derivative) is not bool:
        raise TypeError(
            f"curve.with_derivative must be a boolean; got {type(with_derivative).__name__}."
        )
    if batch_capacity < 1:
        raise ValueError(f"curve.batch_capacity must be positive; got {batch_capacity}.")
    with coordinate_source_request(
        (curve, with_derivative, batch_capacity), budget=budget
    ):
        leaves = jax.tree_util.tree_leaves(
            curve, is_leaf=lambda item: isinstance(item, RootCurveCoordinateSource)
        )
        if any(isinstance(item, RootCurveCoordinateSource) for item in leaves):
            return PreparedRootCurveHost(
                curve=curve,
                with_derivative=with_derivative,
                batch_capacity=batch_capacity,
            )

        def evaluate(parameter: Array) -> Array:
            if with_derivative:
                value, derivative = _curve_value_and_first_jet(curve, parameter[0])
                return jnp.stack((value, derivative), axis=-1)
            return _curve_coordinate_value(curve, parameter[0])

        return prepare_interval_function(evaluate, 1, batch_capacity=batch_capacity)


@checked
def prepare_surface_host(
    patch: AbstractSurfacePatch,
    /,
    *,
    with_differential: bool = False,
    batch_capacity: int = 64,
) -> PreparedIntervalFunction:
    """Prepare bounded NumPy surface execution, optionally including first jets.

    ``evaluate_host_points`` accepts ``(N, 2)`` parameter rows. Without a
    differential it returns ``(N, 3)`` values; with a differential its
    ``(N, 3, 3)`` last axis contains the value and the two coordinate jets.
    """
    if type(with_differential) is not bool:
        raise TypeError(
            f"surface.with_differential must be a boolean; got {type(with_differential).__name__}."
        )

    def evaluate(parameter: Array) -> Array:
        if with_differential:
            value, differential = _surface_value_and_first_jets(patch, parameter)
            return jnp.concatenate((value[:, None], differential), axis=-1)
        return patch.evaluate(parameter)

    return prepare_interval_function(evaluate, 2, batch_capacity=batch_capacity)


def curve_evaluate_host(
    curve: CurveEvaluator,
    parameters: ConvertibleToArray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> np.ndarray:
    """Evaluate coordinates after host preparation under the caller's budget."""
    points = np.asarray(parameters, dtype=np.float64)

    prepared = prepare_curve_host(
        curve, batch_capacity=min(max(points.size, 1), 1024), budget=budget
    )
    values = prepared.evaluate_host_points(points.reshape((-1, 1)))
    return values.reshape((*points.shape, values.shape[-1]))


@checked
def surface_evaluate_host(
    patch: AbstractSurfacePatch, parameters: ConvertibleToArray, /
) -> np.ndarray:
    """Evaluate surface coordinates with the authored map's NumPy execution plan."""
    points = np.asarray(parameters, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError(
            f"surface.parameters must have trailing dimension two; got {points.shape}."
        )
    flat = points.reshape((-1, 2))
    prepared = prepare_surface_host(
        patch, batch_capacity=min(max(flat.shape[0], 1), 1024)
    )
    return prepared.evaluate_host_points(flat).reshape((*points.shape[:-1], 3))


def curve_value_and_derivative_host(
    curve: CurveEvaluator, parameters: ConvertibleToArray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate curve values and first jets on their prepared execution route."""
    points = np.asarray(parameters, dtype=np.float64)

    prepared = prepare_curve_host(
        curve, with_derivative=True, batch_capacity=min(max(points.size, 1), 1024)
    )
    result = prepared.evaluate_host_points(points.reshape((-1, 1)))
    result = result.reshape((*points.shape, result.shape[-2], 2))
    return result[..., 0], result[..., 1]


@checked
def surface_value_and_differential_host(
    patch: AbstractSurfacePatch, parameters: ConvertibleToArray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate surface values and first jets using one NumPy execution plan."""
    points = np.asarray(parameters, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError(
            f"surface.parameters must have trailing dimension two; got {points.shape}."
        )
    flat = points.reshape((-1, 2))

    prepared = prepare_surface_host(
        patch, with_differential=True, batch_capacity=min(max(flat.shape[0], 1), 1024)
    )
    result = prepared.evaluate_host_points(flat).reshape((*points.shape[:-1], 3, 3))
    return result[..., 0], result[..., 1:]


@eqx.filter_jit
@checked
def curve_value_and_derivative(
    curve: CurveEvaluator, parameters: Array, /
) -> tuple[Array, Array]:
    """Evaluate authored curve values and first jets in one compiled boundary."""
    points = parameters.reshape((-1,))
    values, derivatives = jax.vmap(
        lambda point: _curve_value_and_first_jet(curve, point)
    )(points)
    shape = (*parameters.shape, values.shape[-1])
    return values.reshape(shape), derivatives.reshape(shape)


@eqx.filter_jit
@checked
def surface_value_and_differential(
    patch: AbstractSurfacePatch, parameters: Array, /
) -> tuple[Array, Array]:
    """Evaluate surface values and first jets in one compiled boundary."""
    values, differentials = jax.vmap(
        lambda point: _surface_value_and_first_jets(patch, point)
    )(parameters.reshape((-1, 2)))
    return (
        values.reshape((*parameters.shape[:-1], 3)),
        differentials.reshape((*parameters.shape[:-1], 3, 2)),
    )


@checked
def device_bspline_surface_box(
    patch: BSplineSurfacePatch, lower: Array, upper: Array, /
) -> tuple[Array, Array]:
    """Return device source boxes with certified rational denominators."""
    low, high, denominator_low, denominator_high = device_bspline_surface_bounds(
        patch.control_points,
        patch.weights,
        patch.u_knots,
        patch.v_knots,
        lower,
        upper,
        u_degree=patch.u_degree,
        v_degree=patch.v_degree,
    )
    scale = jax.lax.stop_gradient(jnp.max(jnp.abs(patch.weights)))
    low = eqx.error_if(
        low,
        ~jnp.isfinite(scale) | (scale == 0.0),
        "Rational spline weights must contain a finite nonzero value.",
    )
    tolerance = jnp.finfo(low.dtype).eps * jnp.maximum(
        1.0, jnp.maximum(jnp.abs(denominator_low), jnp.abs(denominator_high))
    )
    low = eqx.error_if(
        low,
        jnp.any(
            ~jnp.isfinite(denominator_low)
            | ~jnp.isfinite(denominator_high)
            | ((denominator_low <= tolerance) & (denominator_high >= -tolerance))
        ),
        "Rational spline weights produced a zero or non-finite denominator.",
    )
    low = eqx.error_if(
        low,
        jnp.any(~jnp.isfinite(low) | ~jnp.isfinite(high)),
        "BSplineSurfacePatch.device_box must enclose finite source coordinates; "
        "got a nonfinite box",
    )
    return low, high


@checked
def patch_evaluation_error(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    """Bound coordinatewise roundoff of patch evaluation at each parameter point.

    The bound propagates singleton intervals through the authored evaluation
    map, including dynamic coefficients and placed-occurrence arithmetic.
    It uses the interval substrate's IEEE and four-ulp elementary-function
    assumptions; unsupported primitives refuse during tracing.
    No geometric tolerance or model-unit floor is added.
    """
    if parameters.ndim == 0 or parameters.shape[-1] != 2:
        raise ValueError(
            f"patch.parameters must have trailing dimension two; got {parameters.shape}."
        )

    def error(point: Array) -> Array:
        return device_interval_evaluation_error(patch.evaluate, point)

    errors = jax.vmap(error)(parameters.reshape((-1, 2)))
    return errors.reshape((*parameters.shape[:-1], 3))


@checked
def surface_differential(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    parameters_ = jnp.asarray(parameters, dtype=jnp.float64)
    leading = parameters_.shape[:-1]
    values = jax.vmap(jax.jacfwd(lambda uv: patch.evaluate(uv)))(
        parameters_.reshape((-1, 2))
    )
    return values.reshape((*leading, 3, 2))


@checked
def surface_jacobian(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    differential = surface_differential(patch, parameters)
    return jnp.linalg.norm(
        jnp.cross(differential[..., :, 0], differential[..., :, 1]), axis=-1
    )


@checked
def surface_normal(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    differential = surface_differential(patch, parameters)
    normal = jnp.cross(differential[..., :, 0], differential[..., :, 1])
    return normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)


__all__ = [
    "AbstractCurve",
    "AbstractSurfacePatch",
    "PreparedRootCurveHost",
    "BernsteinCurvePiece",
    "BernsteinSurfacePiece",
    "BSplineCurve",
    "BSplineSurfacePatch",
    "CircleCurve",
    "ConePatch",
    "coefficient_enclosures",
    "ConvexOffsetEvidence",
    "CurveEvaluator",
    "Curve",
    "CylinderPatch",
    "EllipseCurve",
    "ExtrusionSurface",
    "HyperbolaCurve",
    "BernsteinRestrictionBank",
    "MeridianFrame",
    "MeridianJets",
    "MeridianOverlaps",
    "MeridianSpans",
    "LineCurve",
    "PlanePatch",
    "ParabolaCurve",
    "OffsetCurve",
    "OffsetSurface",
    "SurfaceIsoparametricCurve",
    "RationalBezierPiece",
    "RevolutionSurface",
    "RuledSurface",
    "SpherePatch",
    "SurfacePatch",
    "TorusPatch",
    "axis_frame",
    "bspline_span_jet_bounds",
    "closure_tangent_continuous",
    "convex_meridian_offset",
    "curve_evaluate_host",
    "curve_value_and_derivative",
    "curve_value_and_derivative_host",
    "device_bspline_surface_box",
    "directed_sum_of_squares",
    "evaluate_meridian_jets",
    "exact_frame_orientation",
    "exact_unit",
    "interval_cross",
    "interval_dot",
    "interval_norm",
    "junction_rows",
    "meridian_frame",
    "meridian_normal_turns",
    "meridian_second_jets",
    "meridian_value_bounds",
    "offset_chain",
    "parallel_meridian_jets",
    "patch_evaluation_error",
    "piece_hull",
    "prepare_bspline_curve_spans",
    "prepare_curve_host",
    "prepare_surface_host",
    "prepare_meridian_spans",
    "rational_piece_jet_bounds",
    "revolution_meridian",
    "shared_meridian_jets",
    "source_bernstein_restriction_scope",
    "sphere_source_equivalence",
    "sphere_source_pole_normal_limits",
    "sphere_source_radius_terms",
    "surface_evaluate_host",
    "surface_value_and_differential",
    "surface_value_and_differential_host",
    "surface_differential",
    "surface_jacobian",
    "surface_normal",
]
