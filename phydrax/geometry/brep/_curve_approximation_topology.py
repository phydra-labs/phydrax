#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Continuous, bounded source/fit embedding proofs, not sample certificates.

The three straight homotopies share the authored atlas parameter. Every cell
and every neighboring pair has a common strictly monotone coordinate for the
source and fit derivatives. All other pairs have disjoint whole-homotopy image
boxes. These premises prove an oriented piecewise-regular embedding throughout
the homotopy; they do not assert C1 continuity across independently parameterized
source charts. Root references, not numerical endpoint agreement, establish
source incidence.

A nonzero native-period winding is an OPEN lifted UV trace here. A binary
polynomial does not acquire exact irrational-period closure from a small seam
gap. Exporters must inspect ``trim_closure_kinds`` before admitting a closed
face trim. This module never changes source roots or invents a seam correction.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from ..._interpolation._bspline import restrict_bernstein_bounds
from .._interval_enclosure import (
    interval_add,
    interval_divide,
    interval_multiply,
    interval_subtract,
)
from ._intersection_curve import IntersectionCurve
from ._patches import _piece_hull, BSplineCurve, RationalBezierPiece


class BranchApproximationTopologyResourceError(ValueError):
    """A continuous topology premise remains unresolved within explicit limits."""

    def __init__(self, cause: str, *, cells: int, depth: int, bytes_used: int) -> None:
        self.cause = cause
        self.cells = cells
        self.maximum_depth_used = depth
        self.bytes_used = bytes_used
        super().__init__(
            f"Branch approximation topology: {cause} "
            f"(work={cells}, depth={depth}, bytes={bytes_used})."
        )


@dataclass(frozen=True, slots=True)
class BranchApproximationTopologyEvidence:
    """Immutable evidence for exactly the supplied carriers and source atlas.

    ``homotopy_boxes`` and ``derivative_intervals`` have shape (cells,2,7),
    columns XYZ, first UV, second UV. ``local_axes`` and ``adjacent_axes``
    encode signed one-based coordinate axes in each of the three spaces.
    Nonlocal rows hold cell indices and three conservative max-norm separation
    lower bounds. Those bounds concern the recorded nonlocal pairs, not the
    minimum distance between arbitrary distinct points of a continuous curve.
    ``bytes_used`` is the conservative peak reservation for certificate-owned
    storage and enclosure workspaces, not process RSS or a native-runtime cache.
    """

    cells: int
    maximum_depth_used: int
    bytes_used: int
    subdivision_cells: int
    pair_checks: int
    source_branch_id: str
    source_identity: str
    approximation_identity: str
    homotopy_authority: str
    orientation_authority: str
    closed: bool
    trim_closure_kinds: tuple[str, str]
    parameter_cells: np.ndarray
    chart_indices: np.ndarray
    homotopy_boxes: np.ndarray
    derivative_intervals: np.ndarray
    local_axes: np.ndarray
    adjacent_axes: np.ndarray
    nonlocal_pairs: np.ndarray
    tube_separation_lower_bound: float
    trim_separation_lower_bounds: tuple[float, float]
    root_references: np.ndarray
    root_period_shifts: np.ndarray
    chart_gauge_shifts: np.ndarray
    winding_shifts: np.ndarray
    period_intervals: np.ndarray
    source_seam_shift_interval: np.ndarray
    fit_endpoint_difference_interval: np.ndarray


@dataclass(frozen=True, slots=True)
class BranchTrimSeparationEvidence:
    topology_identity: str
    trim_identity: str
    side: str
    cells: int
    maximum_depth_used: int
    bytes_used: int
    separation_lower_bound: float
    interval_evidence: np.ndarray


def _readonly(value: Any, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    # Immutable backing prevents callers from re-enabling writeability.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class _Budget:
    def __init__(self, cells: int, depth: int, memory: int) -> None:
        for name, value, minimum in (
            ("maximum_cells", cells, 1),
            ("maximum_depth", depth, 0),
            ("maximum_bytes", memory, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self.limit, self.depth_limit, self.memory_limit = cells, depth, memory
        self.work = self.depth = self.memory = self.peak = 0
        self.subdivisions = self.pairs = 0

    def fail(self, cause: str) -> None:
        raise BranchApproximationTopologyResourceError(
            cause,
            cells=self.work,
            depth=self.depth,
            bytes_used=self.peak,
        )

    def reserve(self, amount: int) -> None:
        if amount < 0 or self.memory + amount > self.memory_limit:
            self.fail("maximum_bytes exhausted before growing interval storage")
        self.memory += amount
        self.peak = max(self.peak, self.memory)

    def release(self, amount: int) -> None:
        self.memory -= amount

    def charge(self, depth: int, *, pair: bool = False) -> None:
        if depth > self.depth_limit:
            self.fail("maximum_depth exhausted with an unresolved continuous premise")
        if self.work >= self.limit:
            self.fail("maximum_cells exhausted with an unresolved continuous premise")
        self.work += 1
        self.depth = max(self.depth, depth)
        if pair:
            self.pairs += 1
        else:
            self.subdivisions += 1


def _identity(value: Any) -> str:
    from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint

    if isinstance(value, IntersectionCurve):
        return canonical_fingerprint(value.payload())
    carriers = value if isinstance(value, tuple) else (value,)
    return canonical_fingerprint(
        {
            "types": [type(carrier).__name__ for carrier in carriers],
            "degrees": [getattr(carrier, "degree", None) for carrier in carriers],
            "arrays": array_tree_fingerprint(value),
        }
    )


def _finite_box(value: Any, dimension: int) -> np.ndarray:
    box = np.asarray(value, dtype=np.float64)
    if (
        box.shape != (2, dimension)
        or not np.all(np.isfinite(box))
        or np.any(box[0] > box[1])
    ):
        raise ValueError(
            "The native enclosure substrate returned no finite ordered interval."
        )
    return box


def _axis(bounds: np.ndarray, columns: slice) -> int:
    lo, hi = bounds[:, columns]
    for axis in range(lo.size):
        if lo[axis] > 0:
            return axis + 1
        if hi[axis] < 0:
            return -(axis + 1)
    return 0


_SPACES = (slice(0, 3), slice(3, 5), slice(5, 7))


def _separation(first: np.ndarray, second: np.ndarray) -> float:
    forward = interval_subtract((second[0], second[0]), (first[1], first[1]))[0]
    backward = interval_subtract((first[0], first[0]), (second[1], second[1]))[0]
    return max(0.0, float(np.max(np.maximum(forward, backward))))


def _spline_pieces(
    curve: BSplineCurve, budget: _Budget
) -> tuple[RationalBezierPiece, ...]:
    points = np.asarray(curve.control_points)
    weights = np.asarray(curve.weights)
    knots = np.asarray(curve.knots)
    if (
        not np.all(np.isfinite(points))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "Spline embedding requires finite controls and strictly positive weights."
        )
    spans = int(np.count_nonzero(np.diff(knots) > 0))
    coefficients = spans * (curve.degree + 1) * (curve.ambient_dimension + 1)
    # Extraction holds nominal, lower, upper, insertion workspace and final
    # pieces concurrently. Reserve before invoking the native Bernstein path.
    workspace = 12 * 8 * coefficients + 4096
    budget.reserve(workspace)
    budget.charge(0)
    pieces = curve.bezier_pieces()
    retained = sum(
        sys.getsizeof(piece)
        + sum(
            sys.getsizeof(a) + a.nbytes
            for a in (
                piece.homogeneous_controls,
                piece.homogeneous_lower,
                piece.homogeneous_upper,
            )
        )
        for piece in pieces
    )
    if retained > workspace:
        budget.reserve(retained - workspace)
    else:
        budget.release(workspace - retained)
    return pieces


def _spline_bounds(
    pieces: tuple[RationalBezierPiece, ...], first: float, last: float
) -> tuple[np.ndarray, np.ndarray]:
    boxes, jets = [], []
    for piece in pieces:
        start, end = piece.parameter_bounds[0]
        if end < first or start > last:
            continue
        boxes.append(_piece_hull(piece, (first,), (last,)))
        lo, hi = piece.homogeneous_lower, piece.homogeneous_upper
        width = interval_subtract(
            (np.asarray(end), np.asarray(end)), (np.asarray(start), np.asarray(start))
        )
        degree = lo.shape[0] - 1
        difference = interval_subtract((lo[1:], hi[1:]), (lo[:-1], hi[:-1]))
        derivative = interval_divide(
            interval_multiply(difference, (degree, degree)), width
        )
        exact_width = Fraction(end) - Fraction(start)
        a = max(
            0.0,
            float(
                np.nextafter(
                    float((Fraction(max(first, start)) - Fraction(start)) / exact_width),
                    -np.inf,
                )
            ),
        )
        b = min(
            1.0,
            float(
                np.nextafter(
                    float((Fraction(min(last, end)) - Fraction(start)) / exact_width),
                    np.inf,
                )
            ),
        )
        values = restrict_bernstein_bounds(lo, hi, a, b, 0)
        derivatives = restrict_bernstein_bounds(*derivative, a, b, 0)
        h = (np.min(values[0], axis=0), np.max(values[1], axis=0))
        dh = (np.min(derivatives[0], axis=0), np.max(derivatives[1], axis=0))
        numerator = interval_subtract(
            interval_multiply((dh[0][:-1], dh[1][:-1]), (h[0][-1], h[1][-1])),
            interval_multiply((h[0][:-1], h[1][:-1]), (dh[0][-1], dh[1][-1])),
        )
        denominator = interval_multiply((h[0][-1], h[1][-1]), (h[0][-1], h[1][-1]))
        if denominator[0] <= 0:
            raise ValueError(
                "The rational spline derivative denominator is not strictly positive."
            )
        jets.append(np.stack(interval_divide(numerator, denominator)))
    return (
        np.stack(
            (np.min(np.stack(boxes)[:, 0], axis=0), np.max(np.stack(boxes)[:, 1], axis=0))
        ),
        np.stack(
            (np.min(np.stack(jets)[:, 0], axis=0), np.max(np.stack(jets)[:, 1], axis=0))
        ),
    )


@dataclass(slots=True)
class _Cell:
    first: float
    last: float
    chart: int
    depth: int
    box: np.ndarray
    jet: np.ndarray
    axes: tuple[int, int, int]


_CELL_BYTES = 1024
_PAIR_BYTES = 512


def certify_branch_approximation_topology(
    source: IntersectionCurve,
    curve: BSplineCurve,
    first_pcurve: BSplineCurve,
    second_pcurve: BSplineCurve,
    *,
    maximum_cells: int,
    maximum_depth: int,
    maximum_bytes: int,
) -> BranchApproximationTopologyEvidence:
    """Prove coupled physical/UV straight homotopies with bounded adaptive work.

    Positive native-period lifts prove trace embedding only, not exact quotient
    closure of the emitted UV polynomial. The source's exact winding and root
    gauge remain separately recorded. No supplied carrier or source is mutated.
    ``cells`` counts all attempted interval cells and pair checks, including
    failed/refined attempts, rather than only the final evidence partition.
    """
    budget = _Budget(maximum_cells, maximum_depth, maximum_bytes)
    if not isinstance(source, IntersectionCurve) or not source.fully_certified:
        raise ValueError(
            "Topology requires a fully certified source chart/shared-root atlas."
        )
    fits = (curve, first_pcurve, second_pcurve)
    domain = source.parameter_domain
    for fit, dimension in zip(fits, (3, 2, 2), strict=True):
        if (
            not isinstance(fit, BSplineCurve)
            or fit.ambient_dimension != dimension
            or fit.parameter_domain != domain
        ):
            raise ValueError(
                "Each fitted carrier must use the source atlas parameter and its native dimension."
            )
        knots = np.asarray(fit.knots)
        if (
            np.count_nonzero(knots == domain[0]) != fit.degree + 1
            or np.count_nonzero(knots == domain[1]) != fit.degree + 1
        ):
            raise ValueError(
                "Root-incidence evidence requires clamped fitted endpoint coefficients."
            )
    # Source/root bookkeeping, serialization and native enclosure temporaries.
    # Include generating coefficient trees: a large rational surface must not
    # bypass the byte guard merely because its intersection has few charts.
    import jax

    source_workspace = 0
    for patch in (source.first.patch, source.second.patch):
        coefficient_bytes = sum(
            int(leaf.size) * np.dtype(leaf.dtype).itemsize
            for leaf in jax.tree_util.tree_leaves(patch)
            if hasattr(leaf, "size") and hasattr(leaf, "dtype")
        )
        expansion = (getattr(patch, "u_degree", 0) + 1) * (
            getattr(patch, "v_degree", 0) + 1
        )
        source_workspace += 64 * expansion * coefficient_bytes
    budget.reserve(32768 + source.num_charts * 4096 + source_workspace)
    gauge = np.zeros((source.num_charts, 4), dtype=np.int64)
    periods = (*source.first.patch.periods, *source.second.patch.periods)
    for chart in range(1, source.num_charts):
        gauge[chart] = gauge[chart - 1]
        for axis, shift in enumerate(source.transition_shifts[chart - 1]):
            if shift:
                period = periods[axis]
                if period is None:
                    raise ValueError("A source transition has no authored native period.")
                integer = int(round(float(shift) / period))
                if float(shift) != integer * period:
                    raise ValueError(
                        "A source transition lacks exact integer period incidence."
                    )
                gauge[chart, axis] -= integer
    winding = np.asarray(source.node_period_shifts[-1], dtype=np.int64) + gauge[-1]
    closures: list[str] = []
    if source.closed:
        if int(source.node_references[-1]) != int(source.node_references[0]):
            raise ValueError(
                "The closed source does not reference one authored seam root."
            )
        if not np.array_equal(
            np.asarray(curve.control_points)[0], np.asarray(curve.control_points)[-1]
        ):
            raise ValueError(
                "The fitted physical curve lacks algebraically identical closed endpoints."
            )
    for side, fit in enumerate(fits[1:]):
        shift = winding[2 * side : 2 * side + 2]
        if source.closed and not np.any(shift):
            if not np.array_equal(
                np.asarray(fit.control_points)[0], np.asarray(fit.control_points)[-1]
            ):
                raise ValueError(
                    "A zero-winding fitted trim lacks algebraically identical closed endpoints."
                )
            closures.append("closed")
        elif source.closed:
            closures.append("open-native-period-lift")
        else:
            closures.append("open")
    pieces = tuple(_spline_pieces(fit, budget) for fit in fits)
    # One reusable reservation covers restriction/quotient-rule temporaries
    # and overlapping-span box/jet lists before any cell queries allocate them.
    budget.reserve(
        sum(
            len(items) * (1024 + 256 * fit.degree * (fit.ambient_dimension + 1))
            for fit, items in zip(fits, pieces, strict=True)
        )
    )
    pcurves = (source.p_curve("first"), source.p_curve("second"))

    def make(first: float, last: float, chart: int, depth: int) -> _Cell:
        budget.charge(depth)
        budget.reserve(_CELL_BYTES)
        source_boxes = [_finite_box(source.bounding_box(first, last), 3)]
        coupled = source.parameter_enclosures(
            first, last, minimum_chart=chart, maximum_chart=chart
        )[0]
        delta = source._shift_bounds(gauge[chart])
        lifted = np.stack(interval_add((coupled[0], coupled[1]), delta))
        source_boxes.extend((lifted[:, :2], lifted[:, 2:]))
        source_jets = [np.stack(source.derivative_bounds(first, last))]
        source_jets.extend(
            np.stack(pcurve.derivative_bounds(first, last)) for pcurve in pcurves
        )
        fitted = [_spline_bounds(item, first, last) for item in pieces]
        source_box = np.concatenate(source_boxes, axis=1)
        fit_box = np.concatenate([item[0] for item in fitted], axis=1)
        source_jet = np.concatenate(source_jets, axis=1)
        fit_jet = np.concatenate([item[1] for item in fitted], axis=1)
        box = _finite_box(
            np.stack(
                (
                    np.minimum(source_box[0], fit_box[0]),
                    np.maximum(source_box[1], fit_box[1]),
                )
            ),
            7,
        )
        jet = np.stack(
            (np.minimum(source_jet[0], fit_jet[0]), np.maximum(source_jet[1], fit_jet[1]))
        )
        axes = (_axis(jet, _SPACES[0]), _axis(jet, _SPACES[1]), _axis(jet, _SPACES[2]))
        return _Cell(first, last, chart, depth, box, jet, axes)

    def split(cell: _Cell) -> tuple[_Cell, _Cell]:
        if cell.depth >= maximum_depth:
            budget.fail(
                "maximum_depth exhausted: regularity, local injectivity or homotopy separation unresolved"
            )
        midpoint = cell.first + (cell.last - cell.first) / 2
        if not cell.first < midpoint < cell.last:
            budget.fail("binary parameter subdivision cannot advance")
        children = (
            make(cell.first, midpoint, cell.chart, cell.depth + 1),
            make(midpoint, cell.last, cell.chart, cell.depth + 1),
        )
        budget.release(_CELL_BYTES)
        return children

    leaves = [
        make(float(chart), float(chart + 1), chart, 0)
        for chart in range(source.num_charts)
    ]
    while True:
        offender = next(
            (index for index, cell in enumerate(leaves) if not all(cell.axes)), None
        )
        adjacent_rows: list[tuple[int, ...]] = []
        pair_rows: list[tuple[float, ...]] = []
        pair_memory = 0
        if offender is None:
            for i, first in enumerate(leaves):
                for j in range(i + 1, len(leaves)):
                    second = leaves[j]
                    budget.charge(max(first.depth, second.depth), pair=True)
                    adjacent = j == i + 1
                    seam = source.closed and i == 0 and j == len(leaves) - 1
                    bounds = np.stack(
                        (
                            np.minimum(first.jet[0], second.jet[0]),
                            np.maximum(first.jet[1], second.jet[1]),
                        )
                    )
                    axes = tuple(_axis(bounds, space) for space in _SPACES)
                    distances = tuple(
                        _separation(first.box[:, space], second.box[:, space])
                        for space in _SPACES
                    )
                    # At the physical seam and zero-winding UV seams, the last
                    # cell followed by the first has the same oriented axis.
                    local = tuple(
                        adjacent
                        or (seam and (space == 0 or closures[space - 1] == "closed"))
                        for space in range(3)
                    )
                    if any(
                        (not axes[k]) if local[k] else distances[k] <= 0 for k in range(3)
                    ):
                        offender = (
                            i
                            if first.last - first.first >= second.last - second.first
                            else j
                        )
                        break
                    budget.reserve(_PAIR_BYTES)
                    pair_memory += _PAIR_BYTES
                    if adjacent or seam:
                        adjacent_rows.append(
                            (i, j, *[axes[k] if local[k] else 0 for k in range(3)])
                        )
                    pair_rows.append(
                        (
                            i,
                            j,
                            *[math.inf if local[k] else distances[k] for k in range(3)],
                        )
                    )
                if offender is not None:
                    break
        if offender is None:
            break
        budget.release(pair_memory)
        leaves[offender : offender + 1] = split(leaves[offender])
    # Reserve final immutable copies while the working partition is still live.
    budget.reserve(len(leaves) * 768 + len(pair_rows) * 80 + source.num_charts * 256)
    margins = tuple(
        min((row[2 + k] for row in pair_rows), default=math.inf) for k in range(3)
    )
    period_intervals = np.zeros((2, 4))
    for axis, period in enumerate(periods):
        if period is not None:
            unit = np.zeros(4, dtype=np.int64)
            unit[axis] = 1
            lower, upper = source._shift_bounds(unit)
            period_intervals[:, axis] = (lower[axis], upper[axis])
    differences = []
    for fit in fits[1:]:
        points = np.asarray(fit.control_points)
        differences.append(
            np.stack(interval_subtract((points[-1], points[-1]), (points[0], points[0])))
        )
    return BranchApproximationTopologyEvidence(
        budget.work,
        budget.depth,
        budget.peak,
        budget.subdivisions,
        budget.pairs,
        source.branch_id,
        _identity(source),
        _identity(fits),
        "continuous-straight-homotopy/interval-bernstein-source-atlas",
        "shared-strict-monotone-coordinate/authored-root-chart-incidence",
        source.closed,
        (closures[0], closures[1]),
        _readonly([(cell.first, cell.last) for cell in leaves]),
        _readonly([cell.chart for cell in leaves], np.int64),
        _readonly([cell.box for cell in leaves]),
        _readonly([cell.jet for cell in leaves]),
        _readonly([cell.axes for cell in leaves], np.int64),
        _readonly(adjacent_rows, np.int64).reshape((-1, 5)),
        _readonly(pair_rows).reshape((-1, 5)),
        margins[0],
        (margins[1], margins[2]),
        _readonly(source.node_references, np.int64),
        _readonly(source.node_period_shifts, np.int64),
        _readonly(gauge, np.int64),
        _readonly(winding, np.int64),
        _readonly(period_intervals),
        _readonly(np.stack(source._shift_bounds(winding))),
        _readonly(np.concatenate(differences, axis=1)),
    )


def certify_branch_trim_separation(
    evidence: BranchApproximationTopologyEvidence,
    trim: Any,
    *,
    side: str,
    maximum_cells: int,
    maximum_depth: int,
    maximum_bytes: int,
    first: float | None = None,
    last: float | None = None,
) -> BranchTrimSeparationEvidence:
    """Prove separation of a whole branch homotopy from another unaffected trim.

    Every potentially intersecting native-period translate is checked. A trim
    sharing an intentional root must instead be admitted by the model's exact
    incidence algorithm; this function does not remove endpoint neighborhoods.
    """
    if side not in ("first", "second"):
        raise ValueError("side must be 'first' or 'second'.")
    budget = _Budget(maximum_cells, maximum_depth, maximum_bytes)
    offset = 0 if side == "first" else 2
    columns = slice(3 + offset, 5 + offset)
    domain = getattr(trim, "parameter_domain", None) or getattr(
        trim, "parameter_interval", None
    )
    if first is None or last is None:
        if domain is None:
            raise ValueError(
                "An unaffected trim requires an explicit finite parameter interval."
            )
        first = float(domain[0]) if first is None else first
        last = float(domain[1]) if last is None else last
    if not math.isfinite(first) or not math.isfinite(last) or not first < last:
        raise ValueError("Unaffected-trim separation requires a finite nonempty range.")
    enclosure = getattr(trim, "bounding_box", None) or getattr(trim, "enclosure", None)
    if enclosure is None:
        raise ValueError(
            "The unaffected trim has no native interval enclosure substrate."
        )
    import jax

    coefficient_bytes = sum(
        int(leaf.size) * np.dtype(leaf.dtype).itemsize
        for leaf in jax.tree_util.tree_leaves(trim)
        if hasattr(leaf, "size") and hasattr(leaf, "dtype")
    )
    budget.reserve(8192 + 64 * (getattr(trim, "degree", 0) + 1) * coefficient_bytes)
    workspace_bytes = getattr(trim, "enclosure_workspace_bytes", 0)
    if (
        isinstance(workspace_bytes, bool)
        or not isinstance(workspace_bytes, int)
        or workspace_bytes < 0
    ):
        raise ValueError("Trim enclosure workspace must be a nonnegative byte count.")
    budget.reserve(workspace_bytes)
    rows: list[tuple[float, ...]] = []
    minimum = math.inf
    for cell, tube in enumerate(evidence.homotopy_boxes[:, :, columns]):
        budget.charge(0, pair=True)
        whole = _finite_box(enclosure(first, last), 2)
        ranges = []
        for axis in range(2):
            period = evidence.period_intervals[:, offset + axis]
            if period[0] > 0:
                numerator = interval_subtract(
                    (tube[0, axis], tube[1, axis]), (whole[0, axis], whole[1, axis])
                )
                possible = interval_divide(numerator, (period[0], period[1]))
                ranges.append(
                    range(
                        math.floor(float(possible[0])) - 1,
                        math.ceil(float(possible[1])) + 2,
                    )
                )
            else:
                ranges.append(range(0, 1))
        copies = len(ranges[0]) * len(ranges[1])
        if copies > budget.limit - budget.work:
            budget.fail("native-period translate count exceeds maximum_cells")
        for ku in ranges[0]:
            for kv in ranges[1]:
                shift = interval_multiply(
                    (
                        evidence.period_intervals[0, offset : offset + 2],
                        evidence.period_intervals[1, offset : offset + 2],
                    ),
                    (np.asarray((ku, kv)), np.asarray((ku, kv))),
                )
                stack = [(float(first), float(last), 0)]
                budget.reserve(256)
                while stack:
                    a, b, depth = stack.pop()
                    budget.release(256)
                    budget.charge(depth, pair=True)
                    box = _finite_box(enclosure(a, b), 2)
                    moved = np.stack(interval_add((box[0], box[1]), shift))
                    distance = _separation(tube, moved)
                    if distance > 0:
                        budget.reserve(_PAIR_BYTES)
                        rows.append((cell, a, b, ku, kv, distance))
                        minimum = min(minimum, distance)
                        continue
                    if depth >= maximum_depth:
                        budget.fail(
                            "unaffected trim separation unresolved at maximum_depth"
                        )
                    midpoint = a + (b - a) / 2
                    if not a < midpoint < b:
                        budget.fail("unaffected trim subdivision cannot advance")
                    budget.reserve(512)
                    stack.extend(((a, midpoint, depth + 1), (midpoint, b, depth + 1)))
    budget.reserve(len(rows) * 96)
    return BranchTrimSeparationEvidence(
        evidence.approximation_identity,
        _identity(trim),
        side,
        budget.work,
        budget.depth,
        budget.peak,
        minimum,
        _readonly(rows).reshape((-1, 6)),
    )
