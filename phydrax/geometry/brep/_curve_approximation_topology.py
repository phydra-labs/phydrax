#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certify continuous source/fit embedding and separation of trim homotopies.

The physical and two chart-space straight homotopies share the source atlas
parameter. Local cells share an oriented monotone coordinate; nonlocal cells
have disjoint image boxes. Root references establish endpoint incidence.
Native-period winding certifies an open lifted trace: exporters inspect
``trim_closure_kinds`` before admitting a closed face trim.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import assert_never, final, Literal, NoReturn, TypedDict

import equinox as eqx
import jax
import numpy as np
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._resource_ticket import HostBatchTicket, ManagedBytesContributor
from ..._strict import StrictModule
from ..._validation import nonnegative_integer, positive_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, parse
from .._atlas import AbstractTrimCurve
from .._interval_enclosure import (
    interval_add,
    interval_divide,
    interval_multiply,
    interval_subtract,
)
from ._coordinate_request import coordinate_source_request
from ._intersection_curve import IntersectionCurve, IntersectionCurveSide
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    piece_hull,
    rational_piece_jet_bounds,
    RationalBezierPiece,
)
from ._request_preparations import TrimRequestPreparations


type TrimClosureKind = Literal["closed", "open-native-period-lift", "open"]


class BranchApproximationTopologyResourceError(ValueError):
    """Report unresolved continuous topology and consumed resource evidence."""

    def __init__(self, cause: str, *, cells: int, depth: int, bytes_used: int) -> None:
        self.cause = cause
        self.cells = cells
        self.maximum_depth_used = depth
        self.bytes_used = bytes_used
        super().__init__(
            f"BranchApproximationTopology.resource must cover continuous topology; "
            f"got {cause} (work={cells}, depth={depth}, bytes={bytes_used})."
        )


@final
class BranchApproximationTopologyEvidence(StrictModule):
    """Record continuous embedding bounds for the supplied carriers and atlas.

    ``homotopy_boxes`` and ``derivative_intervals`` have shape (cells,2,7),
    columns XYZ, first UV, second UV. ``local_axes`` and ``adjacent_axes``
    encode signed one-based coordinate axes in each of the three spaces.
    Nonlocal rows hold cell indices and three conservative max-norm separation
    lower bounds. Those bounds concern the recorded nonlocal pairs, not the
    minimum distance between arbitrary distinct points of a continuous curve.
    ``bytes_used`` is the conservative peak reservation for certificate-owned
    storage and enclosure workspaces, not process RSS or a native-runtime cache.
    """

    cells: int = eqx.field(static=True)
    maximum_depth_used: int = eqx.field(static=True)
    bytes_used: int = eqx.field(static=True)
    subdivision_cells: int = eqx.field(static=True)
    pair_checks: int = eqx.field(static=True)
    source_branch_id: str = eqx.field(static=True)
    source_identity: str = eqx.field(static=True)
    approximation_identity: str = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    trim_closure_kinds: tuple[TrimClosureKind, TrimClosureKind] = eqx.field(static=True)
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


@final
class BranchTrimSeparationEvidence(StrictModule):
    """Record interval separation from every potentially intersecting trim lift."""

    topology_identity: str = eqx.field(static=True)
    trim_identity: str = eqx.field(static=True)
    side: IntersectionCurveSide = eqx.field(static=True)
    cells: int = eqx.field(static=True)
    maximum_depth_used: int = eqx.field(static=True)
    bytes_used: int = eqx.field(static=True)
    separation_lower_bound: float
    interval_evidence: np.ndarray


def _readonly(value: ArrayLike, dtype: DTypeLike = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    # Immutable backing prevents callers from re-enabling writeability.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class _TopologyBudget:
    """Retain topology depth/categories over the shared host resource ledger."""

    def __init__(
        self, ticket: HostBatchTicket, storage: ManagedBytesContributor, depth: int
    ) -> None:
        self.ticket = ticket
        self.storage = storage
        self.depth_limit = depth
        self.depth = self.subdivisions = self.pairs = 0

    @property
    def work(self) -> int:
        return self.ticket.work_units

    @property
    def peak(self) -> int:
        return self.ticket.managed_bytes

    def fail(self, cause: str) -> NoReturn:
        raise BranchApproximationTopologyResourceError(
            cause, cells=self.work, depth=self.depth, bytes_used=self.peak
        )

    def reserve(self, amount: int) -> None:
        if amount < 0 or self.storage.bound + amount > self.ticket.bytes_upper:
            self.fail("maximum_bytes exhausted before growing interval storage")
        self.storage.set_bound(self.storage.bound + amount)

    def release(self, amount: int) -> None:
        self.storage.set_bound(self.storage.bound - amount)

    def charge(self, depth: int, *, pair: bool = False) -> None:
        if depth > self.depth_limit:
            self.fail("maximum_depth exhausted with unresolved continuous topology")
        if self.ticket.remaining_work_units <= 0:
            self.fail("maximum_cells exhausted with unresolved continuous topology")
        self.ticket.consume(1)
        self.depth = max(self.depth, depth)
        if pair:
            self.pairs += 1
        else:
            self.subdivisions += 1


@contextmanager
def _topology_budget(cells: int, depth: int, memory: int) -> Iterator[_TopologyBudget]:
    positive_integer(cells, "maximum_cells")
    nonnegative_integer(depth, "maximum_depth")
    positive_integer(memory, "maximum_bytes")
    with HostBatchTicket(cells, bytes_upper=memory) as ticket:
        with ticket.managed_storage() as storage:
            yield _TopologyBudget(ticket, storage, depth)


class _CarrierIdentityPayload(TypedDict):
    types: list[str]
    degrees: list[int | None]
    arrays: dict[str, str | list[dict[str, str | list[int]]]]


def _identity(value: IntersectionCurve | TrimCarrier | tuple[BSplineCurve, ...]) -> str:
    match value:
        case IntersectionCurve():
            return canonical_fingerprint(value.payload())
        case tuple():
            carriers = value
        case _:
            carriers = (value,)
    payload: _CarrierIdentityPayload = {
        "types": [type(carrier).__name__ for carrier in carriers],
        "degrees": [_carrier_degree(carrier) for carrier in carriers],
        "arrays": array_tree_fingerprint(value),
    }
    return canonical_fingerprint(payload)


def _carrier_degree(carrier: TrimCarrier) -> int | None:
    match carrier:
        case BSplineCurve():
            return carrier.degree
        case _:
            return None


def _finite_box(value: ArrayLike, dimension: int) -> np.ndarray:
    box = np.asarray(value, dtype=np.float64)
    if (
        box.shape != (2, dimension)
        or not np.all(np.isfinite(box))
        or np.any(box[0] > box[1])
    ):
        raise ValueError(
            f"BranchApproximationTopology.enclosure must be finite ordered (2, {dimension}); "
            f"got shape={box.shape}."
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
    curve: BSplineCurve, budget: _TopologyBudget
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
            "BranchApproximationTopology.fit must have finite controls and positive weights; "
            "got nonfinite controls or nonpositive weights."
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


def _fitted_interval(
    pieces: tuple[RationalBezierPiece, ...], first: float, last: float
) -> tuple[np.ndarray, np.ndarray]:
    boxes: list[np.ndarray] = []
    jets: list[np.ndarray] = []
    for piece in pieces:
        start, end = piece.parameter_bounds[0]
        if end < first or start > last:
            continue
        boxes.append(piece_hull(piece, (first,), (last,)))
        jets.append(
            np.stack(rational_piece_jet_bounds(piece, max(first, start), min(last, end)))
        )
    return (
        np.stack(
            (np.min(np.stack(boxes)[:, 0], axis=0), np.max(np.stack(boxes)[:, 1], axis=0))
        ),
        np.stack(
            (np.min(np.stack(jets)[:, 0], axis=0), np.max(np.stack(jets)[:, 1], axis=0))
        ),
    )


@final
class _Cell(StrictModule):
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    chart: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    box: np.ndarray
    jet: np.ndarray
    axes: tuple[int, int, int]


@final
class _FittedAtlas(StrictModule):
    source: IntersectionCurve
    fits: tuple[BSplineCurve, BSplineCurve, BSplineCurve]
    pieces: tuple[tuple[RationalBezierPiece, ...], ...]
    pcurves: tuple[AbstractTrimCurve, AbstractTrimCurve]
    gauge: np.ndarray
    winding: np.ndarray
    periods: tuple[float | None, ...] = eqx.field(static=True)
    closures: tuple[TrimClosureKind, TrimClosureKind] = eqx.field(static=True)


@final
class _PairPartition(StrictModule):
    offender: int | None = eqx.field(static=True)
    adjacent_rows: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    pair_rows: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    memory: int = eqx.field(static=True)


@final
class _PairRelation(StrictModule):
    adjacent: bool = eqx.field(static=True)
    seam: bool = eqx.field(static=True)
    axes: tuple[int, ...] = eqx.field(static=True)
    distances: tuple[float, ...] = eqx.field(static=True)
    local: tuple[bool, ...] = eqx.field(static=True)
    resolved: bool = eqx.field(static=True)


@final
class BranchTrimHomotopyTube(StrictModule):
    """Enclose the chart-space homotopy over a represented source range."""

    evidence: BranchApproximationTopologyEvidence
    side: IntersectionCurveSide = eqx.field(static=True)
    parameter_domain: tuple[float, float] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        evidence: BranchApproximationTopologyEvidence,
        side: IntersectionCurveSide,
        parameter_domain: tuple[float, float],
    ) -> None:
        side = parse(side, IntersectionCurveSide, "BranchTrimHomotopyTube.side")
        first, last = parameter_domain
        if not math.isfinite(first) or not math.isfinite(last) or not first < last:
            raise ValueError(
                "BranchTrimHomotopyTube.parameter_domain must be finite and nonempty; "
                f"got {parameter_domain}."
            )
        self.evidence = evidence
        self.side = side
        self.parameter_domain = parameter_domain

    @property
    def enclosure_workspace_bytes(self) -> int:
        return 64 * self.evidence.parameter_cells.shape[0]

    def enclosure(self, first: float, last: float, /) -> np.ndarray:
        cells = self.evidence.parameter_cells
        if not self.parameter_domain[0] <= first <= last <= self.parameter_domain[1]:
            raise ValueError(
                "BranchTrimHomotopyTube.range must retain its represented source range; "
                f"got ({first}, {last})."
            )
        active = (cells[:, 0] <= last) & (cells[:, 1] >= first)
        if not np.any(active):
            raise ValueError(
                "BranchTrimHomotopyTube.range must overlap certified cells; "
                f"got ({first}, {last})."
            )
        columns = slice(3, 5) if self.side == "first" else slice(5, 7)
        boxes = self.evidence.homotopy_boxes[active, :, columns]
        return np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))


type TrimCarrier = AbstractCurve | AbstractTrimCurve | BranchTrimHomotopyTube


_CELL_BYTES = 1024
_PAIR_BYTES = 512


def _coefficient_bytes(value: AbstractSurfacePatch | TrimCarrier) -> int:
    # A tube borrows certificate storage; its exporter accounts for that storage
    # separately. Only its active-cell enclosure workspace is newly reserved here.
    match value:
        case BranchTrimHomotopyTube():
            return 0
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        match leaf:
            case jax.Array() | np.ndarray():
                total += leaf.size * np.dtype(leaf.dtype).itemsize
    return total


def _validate_fits(
    source: IntersectionCurve, fits: tuple[BSplineCurve, BSplineCurve, BSplineCurve]
) -> None:
    if not source.fully_certified:
        raise ValueError(
            "BranchApproximationTopology.source must have certified charts and shared roots; "
            "got an uncertified source atlas."
        )
    domain = source.parameter_domain
    for fit, dimension in zip(fits, (3, 2, 2), strict=True):
        if fit.ambient_dimension != dimension or fit.parameter_domain != domain:
            raise ValueError(
                "BranchApproximationTopology.fit must use the source domain and space dimension; "
                f"got domain={fit.parameter_domain}, dimension={fit.ambient_dimension}."
            )
        knots = np.asarray(fit.knots)
        if (
            np.count_nonzero(knots == domain[0]) != fit.degree + 1
            or np.count_nonzero(knots == domain[1]) != fit.degree + 1
        ):
            raise ValueError(
                "BranchApproximationTopology.fit.knots must clamp both source endpoints; "
                f"got degree={fit.degree}, domain={domain}."
            )


@final
class _RootGauge(StrictModule):
    gauge: np.ndarray
    winding: np.ndarray
    periods: tuple[float | None, ...] = eqx.field(static=True)


def _prepare_root_gauge(source: IntersectionCurve) -> _RootGauge:
    gauge = np.zeros((source.num_charts, 4), dtype=np.int64)
    periods = (*source.first.patch.periods, *source.second.patch.periods)
    for chart in range(1, source.num_charts):
        gauge[chart] = gauge[chart - 1]
        for axis, shift in enumerate(source.transition_shifts[chart - 1]):
            if shift:
                period = periods[axis]
                if period is None:
                    raise ValueError(
                        "BranchApproximationTopology.source.transition must have a declared period; "
                        f"got chart={chart}, axis={axis}, shift={shift}."
                    )
                integer = int(round(float(shift) / period))
                if float(shift) != integer * period:
                    raise ValueError(
                        "BranchApproximationTopology.source.transition must be an integer period shift; "
                        f"got shift={shift}, period={period}."
                    )
                gauge[chart, axis] -= integer
    return _RootGauge(
        gauge=gauge,
        winding=np.asarray(source.node_period_shifts[-1], dtype=np.int64) + gauge[-1],
        periods=periods,
    )


def _trim_closures(
    source: IntersectionCurve,
    fits: tuple[BSplineCurve, BSplineCurve, BSplineCurve],
    winding: np.ndarray,
) -> tuple[TrimClosureKind, TrimClosureKind]:
    if source.closed:
        if int(source.node_references[-1]) != int(source.node_references[0]):
            raise ValueError(
                "BranchApproximationTopology.source.seam must reference one shared root; "
                "got different endpoint root references."
            )
        if not np.array_equal(
            np.asarray(fits[0].control_points)[0], np.asarray(fits[0].control_points)[-1]
        ):
            raise ValueError(
                "BranchApproximationTopology.fit.endpoints must coincide algebraically for a closed source; "
                "got different physical endpoint controls."
            )
    closures: list[TrimClosureKind] = []
    for side, fit in enumerate(fits[1:]):
        shift = winding[2 * side : 2 * side + 2]
        if source.closed and not np.any(shift):
            if not np.array_equal(
                np.asarray(fit.control_points)[0], np.asarray(fit.control_points)[-1]
            ):
                raise ValueError(
                    "BranchApproximationTopology.fit.trim_endpoints must coincide for zero winding; "
                    f"got different endpoint controls on side={side}."
                )
            closures.append("closed")
        elif source.closed:
            closures.append("open-native-period-lift")
        else:
            closures.append("open")
    return closures[0], closures[1]


def _prepare_fitted_atlas(
    source: IntersectionCurve,
    fits: tuple[BSplineCurve, BSplineCurve, BSplineCurve],
    budget: _TopologyBudget,
) -> _FittedAtlas:
    _validate_fits(source, fits)
    # Generating coefficients contribute even when the intersection has few charts.
    source_workspace = 0
    for patch in (source.first.patch, source.second.patch):
        match patch:
            case BSplineSurfacePatch():
                expansion = (patch.u_degree + 1) * (patch.v_degree + 1)
            case _:
                expansion = 1
        source_workspace += 64 * expansion * _coefficient_bytes(patch)
    budget.reserve(32768 + source.num_charts * 4096 + source_workspace)
    roots = _prepare_root_gauge(source)
    closures = _trim_closures(source, fits, roots.winding)
    pieces = tuple(_spline_pieces(fit, budget) for fit in fits)
    # This reusable workspace precedes every restriction/quotient interval query.
    budget.reserve(
        sum(
            len(items) * (1024 + 256 * fit.degree * (fit.ambient_dimension + 1))
            for fit, items in zip(fits, pieces, strict=True)
        )
    )
    return _FittedAtlas(
        source=source,
        fits=fits,
        pieces=pieces,
        pcurves=(source.p_curve("first"), source.p_curve("second")),
        gauge=roots.gauge,
        winding=roots.winding,
        periods=roots.periods,
        closures=closures,
    )


def _make_cell(
    atlas: _FittedAtlas,
    interval: tuple[float, float],
    chart_depth: tuple[int, int],
    budget: _TopologyBudget,
    preparations: TrimRequestPreparations,
    coordinate_budget: CoordinateEnclosureBudget,
) -> _Cell:
    first, last = interval
    chart, depth = chart_depth
    source = atlas.source
    budget.charge(depth)
    budget.reserve(_CELL_BYTES)
    source_boxes = [
        _finite_box(
            source.bounding_box(
                first, last, preparations=preparations, budget=coordinate_budget
            ),
            3,
        )
    ]
    coupled = source.parameter_enclosures(
        first,
        last,
        minimum_chart=chart,
        maximum_chart=chart,
        preparations=preparations,
        budget=coordinate_budget,
    )[0]
    delta = source.shift_parameter_bounds(atlas.gauge[chart])
    lifted = np.stack(interval_add((coupled[0], coupled[1]), delta))
    source_boxes.extend((lifted[:, :2], lifted[:, 2:]))
    source_jets = [
        np.stack(
            source.derivative_bounds(
                first, last, preparations=preparations, budget=coordinate_budget
            )
        )
    ]
    source_jets.extend(
        np.stack(
            pcurve.derivative_bounds(
                first, last, preparations=preparations, budget=coordinate_budget
            )
        )
        for pcurve in atlas.pcurves
    )
    fitted = [_fitted_interval(item, first, last) for item in atlas.pieces]
    source_box = np.concatenate(source_boxes, axis=1)
    fit_box = np.concatenate([item[0] for item in fitted], axis=1)
    source_jet = np.concatenate(source_jets, axis=1)
    fit_jet = np.concatenate([item[1] for item in fitted], axis=1)
    box = _finite_box(
        np.stack(
            (np.minimum(source_box[0], fit_box[0]), np.maximum(source_box[1], fit_box[1]))
        ),
        7,
    )
    jet = np.stack(
        (np.minimum(source_jet[0], fit_jet[0]), np.maximum(source_jet[1], fit_jet[1]))
    )
    return _Cell(
        first=first,
        last=last,
        chart=chart,
        depth=depth,
        box=box,
        jet=jet,
        axes=(_axis(jet, _SPACES[0]), _axis(jet, _SPACES[1]), _axis(jet, _SPACES[2])),
    )


def _split_cell(
    cell: _Cell,
    atlas: _FittedAtlas,
    budget: _TopologyBudget,
    preparations: TrimRequestPreparations,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[_Cell, _Cell]:
    if cell.depth >= budget.depth_limit:
        budget.fail(
            "maximum_depth exhausted: regularity, local injectivity or homotopy separation unresolved"
        )
    midpoint = cell.first + (cell.last - cell.first) / 2
    if not cell.first < midpoint < cell.last:
        budget.fail("binary parameter subdivision cannot advance")
    children = (
        _make_cell(
            atlas,
            (cell.first, midpoint),
            (cell.chart, cell.depth + 1),
            budget,
            preparations,
            coordinate_budget,
        ),
        _make_cell(
            atlas,
            (midpoint, cell.last),
            (cell.chart, cell.depth + 1),
            budget,
            preparations,
            coordinate_budget,
        ),
    )
    budget.release(_CELL_BYTES)
    return children


def _pair_relation(
    atlas: _FittedAtlas, first: _Cell, second: _Cell, i: int, j: int, count: int
) -> _PairRelation:
    adjacent = j == i + 1
    seam = atlas.source.closed and i == 0 and j == count - 1
    bounds = np.stack(
        (np.minimum(first.jet[0], second.jet[0]), np.maximum(first.jet[1], second.jet[1]))
    )
    axes = tuple(_axis(bounds, space) for space in _SPACES)
    distances = tuple(
        _separation(first.box[:, space], second.box[:, space]) for space in _SPACES
    )
    # Physical and zero-winding UV seams retain one oriented axis.
    local = tuple(
        adjacent or (seam and (space == 0 or atlas.closures[space - 1] == "closed"))
        for space in range(3)
    )
    resolved = not any((not axes[k]) if local[k] else distances[k] <= 0 for k in range(3))
    return _PairRelation(
        adjacent=adjacent,
        seam=seam,
        axes=axes,
        distances=distances,
        local=local,
        resolved=resolved,
    )


def _check_partition(
    atlas: _FittedAtlas, leaves: list[_Cell], budget: _TopologyBudget
) -> _PairPartition:
    offender = next(
        (index for index, cell in enumerate(leaves) if not all(cell.axes)), None
    )
    if offender is not None:
        return _PairPartition(offender=offender, adjacent_rows=(), pair_rows=(), memory=0)
    adjacent_rows: list[tuple[int, ...]] = []
    pair_rows: list[tuple[float, ...]] = []
    pair_memory = 0
    for i, first in enumerate(leaves):
        for j in range(i + 1, len(leaves)):
            second = leaves[j]
            budget.charge(max(first.depth, second.depth), pair=True)
            relation = _pair_relation(atlas, first, second, i, j, len(leaves))
            if not relation.resolved:
                offender = (
                    i if first.last - first.first >= second.last - second.first else j
                )
                return _PairPartition(
                    offender=offender,
                    adjacent_rows=tuple(adjacent_rows),
                    pair_rows=tuple(pair_rows),
                    memory=pair_memory,
                )
            budget.reserve(_PAIR_BYTES)
            pair_memory += _PAIR_BYTES
            if relation.adjacent or relation.seam:
                adjacent_rows.append(
                    (
                        i,
                        j,
                        *[relation.axes[k] if relation.local[k] else 0 for k in range(3)],
                    )
                )
            pair_rows.append(
                (
                    i,
                    j,
                    *[
                        math.inf if relation.local[k] else relation.distances[k]
                        for k in range(3)
                    ],
                )
            )
    return _PairPartition(
        offender=None,
        adjacent_rows=tuple(adjacent_rows),
        pair_rows=tuple(pair_rows),
        memory=pair_memory,
    )


def _topology_evidence(
    atlas: _FittedAtlas,
    leaves: list[_Cell],
    partition: _PairPartition,
    budget: _TopologyBudget,
) -> BranchApproximationTopologyEvidence:
    source = atlas.source
    # Immutable copies coexist with the working partition until construction ends.
    budget.reserve(
        len(leaves) * 768 + len(partition.pair_rows) * 80 + source.num_charts * 256
    )
    margins = tuple(
        min((row[2 + k] for row in partition.pair_rows), default=math.inf)
        for k in range(3)
    )
    period_intervals = np.zeros((2, 4))
    for axis, period in enumerate(atlas.periods):
        if period is not None:
            unit = np.zeros(4, dtype=np.int64)
            unit[axis] = 1
            lower, upper = source.shift_parameter_bounds(unit)
            period_intervals[:, axis] = (lower[axis], upper[axis])
    differences: list[np.ndarray] = []
    for fit in atlas.fits[1:]:
        points = np.asarray(fit.control_points)
        differences.append(
            np.stack(interval_subtract((points[-1], points[-1]), (points[0], points[0])))
        )
    return BranchApproximationTopologyEvidence(
        cells=budget.work,
        maximum_depth_used=budget.depth,
        bytes_used=budget.peak,
        subdivision_cells=budget.subdivisions,
        pair_checks=budget.pairs,
        source_branch_id=source.branch_id,
        source_identity=_identity(source),
        approximation_identity=_identity(atlas.fits),
        closed=source.closed,
        trim_closure_kinds=atlas.closures,
        parameter_cells=_readonly(
            np.asarray([(cell.first, cell.last) for cell in leaves])
        ),
        chart_indices=_readonly(np.asarray([cell.chart for cell in leaves]), np.int64),
        homotopy_boxes=_readonly(np.asarray([cell.box for cell in leaves])),
        derivative_intervals=_readonly(np.asarray([cell.jet for cell in leaves])),
        local_axes=_readonly(np.asarray([cell.axes for cell in leaves]), np.int64),
        adjacent_axes=_readonly(np.asarray(partition.adjacent_rows), np.int64).reshape(
            (-1, 5)
        ),
        nonlocal_pairs=_readonly(np.asarray(partition.pair_rows)).reshape((-1, 5)),
        tube_separation_lower_bound=margins[0],
        trim_separation_lower_bounds=(margins[1], margins[2]),
        root_references=_readonly(source.node_references, np.int64),
        root_period_shifts=_readonly(source.node_period_shifts, np.int64),
        chart_gauge_shifts=_readonly(atlas.gauge, np.int64),
        winding_shifts=_readonly(atlas.winding, np.int64),
        period_intervals=_readonly(period_intervals),
        source_seam_shift_interval=_readonly(
            np.stack(source.shift_parameter_bounds(atlas.winding))
        ),
        fit_endpoint_difference_interval=_readonly(np.concatenate(differences, axis=1)),
    )


@checked
def certify_branch_approximation_topology(
    source: IntersectionCurve,
    curve: BSplineCurve,
    first_pcurve: BSplineCurve,
    second_pcurve: BSplineCurve,
    *,
    maximum_cells: int,
    maximum_depth: int,
    maximum_bytes: int,
    preparations: TrimRequestPreparations | None = None,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> BranchApproximationTopologyEvidence:
    """Certify physical and chart-space homotopy embedding within explicit limits.

    Native-period lifts establish trace embedding, not closure of the fitted
    polynomial in the quotient. Work counts attempted interval cells and pairs.
    """
    preparations = preparations if preparations is not None else TrimRequestPreparations()
    fits = (curve, first_pcurve, second_pcurve)
    request = (
        coordinate_source_request((source, *fits))
        if coordinate_budget is None
        else nullcontext(coordinate_budget)
    )
    with request as coordinate_budget:
        with _topology_budget(maximum_cells, maximum_depth, maximum_bytes) as budget:
            atlas = _prepare_fitted_atlas(source, fits, budget)
            leaves = [
                _make_cell(
                    atlas,
                    (float(chart), float(chart + 1)),
                    (chart, 0),
                    budget,
                    preparations,
                    coordinate_budget,
                )
                for chart in range(source.num_charts)
            ]
            while True:
                partition = _check_partition(atlas, leaves, budget)
                if partition.offender is None:
                    break
                budget.release(partition.memory)
                offender = partition.offender
                leaves[offender : offender + 1] = _split_cell(
                    leaves[offender], atlas, budget, preparations, coordinate_budget
                )
            return _topology_evidence(atlas, leaves, partition, budget)


@final
class _TrimRange(StrictModule):
    trim: TrimCarrier
    first: float
    last: float
    offset: int


def _prepare_trim_range(
    trim: TrimCarrier,
    side: IntersectionCurveSide,
    first: float | None,
    last: float | None,
    budget: _TopologyBudget,
) -> _TrimRange:
    domain = trim.parameter_domain
    if first is None or last is None:
        if domain is None:
            raise ValueError(
                "BranchTrimSeparation.range must have explicit finite endpoints for an unbounded trim; "
                f"got first={first}, last={last}."
            )
        first = float(domain[0]) if first is None else first
        last = float(domain[1]) if last is None else last
    if not math.isfinite(first) or not math.isfinite(last) or not first < last:
        raise ValueError(
            "BranchTrimSeparation.range must be finite and nonempty; "
            f"got ({first}, {last})."
        )
    match trim:
        case BSplineCurve():
            degree = trim.degree
        case _:
            degree = 0
    budget.reserve(8192 + 64 * (degree + 1) * _coefficient_bytes(trim))
    match trim:
        case BranchTrimHomotopyTube():
            workspace = trim.enclosure_workspace_bytes
        case _:
            workspace = 0
    budget.reserve(workspace)
    return _TrimRange(
        trim=trim, first=first, last=last, offset=0 if side == "first" else 2
    )


def _trim_box(
    trim: TrimCarrier,
    first: float,
    last: float,
    preparations: TrimRequestPreparations,
    coordinate_budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    match trim:
        case AbstractCurve() | AbstractTrimCurve():
            box = trim.bounding_box(
                first, last, preparations=preparations, budget=coordinate_budget
            )
        case BranchTrimHomotopyTube():
            box = trim.enclosure(first, last)
        case _ as invalid:
            assert_never(invalid)
    return _finite_box(box, 2)


def _period_translates(
    tube: np.ndarray, whole: np.ndarray, periods: np.ndarray
) -> tuple[range, range]:
    ranges: list[range] = []
    for axis in range(2):
        period = periods[:, axis]
        if period[0] > 0:
            numerator = interval_subtract(
                (tube[0, axis], tube[1, axis]), (whole[0, axis], whole[1, axis])
            )
            possible = interval_divide(numerator, (period[0], period[1]))
            ranges.append(
                range(
                    math.floor(float(possible[0])) - 1, math.ceil(float(possible[1])) + 2
                )
            )
        else:
            ranges.append(range(0, 1))
    return ranges[0], ranges[1]


@final
class _TrimCellSeparation(StrictModule):
    rows: tuple[tuple[float, ...], ...]
    minimum: float


def _separate_trim_translate(
    cell: int,
    tube: np.ndarray,
    trim_range: _TrimRange,
    translation: tuple[int, int],
    shift: tuple[np.ndarray, np.ndarray],
    budget: _TopologyBudget,
    preparations: TrimRequestPreparations,
    coordinate_budget: CoordinateEnclosureBudget,
) -> _TrimCellSeparation:
    ku, kv = translation
    stack = [(trim_range.first, trim_range.last, 0)]
    rows: list[tuple[float, ...]] = []
    minimum = math.inf
    budget.reserve(256)
    while stack:
        a, b, depth = stack.pop()
        budget.release(256)
        budget.charge(depth, pair=True)
        box = _trim_box(trim_range.trim, a, b, preparations, coordinate_budget)
        moved = np.stack(interval_add((box[0], box[1]), shift))
        distance = _separation(tube, moved)
        if distance > 0:
            budget.reserve(_PAIR_BYTES)
            rows.append((cell, a, b, ku, kv, distance))
            minimum = min(minimum, distance)
            continue
        if depth >= budget.depth_limit:
            budget.fail("unaffected trim separation unresolved at maximum_depth")
        midpoint = a + (b - a) / 2
        if not a < midpoint < b:
            budget.fail("unaffected trim subdivision cannot advance")
        budget.reserve(512)
        stack.extend(((a, midpoint, depth + 1), (midpoint, b, depth + 1)))
    return _TrimCellSeparation(rows=tuple(rows), minimum=minimum)


@checked
def certify_branch_trim_separation(
    evidence: BranchApproximationTopologyEvidence,
    trim: TrimCarrier,
    *,
    side: IntersectionCurveSide,
    maximum_cells: int,
    maximum_depth: int,
    maximum_bytes: int,
    first: float | None = None,
    last: float | None = None,
    preparations: TrimRequestPreparations | None = None,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> BranchTrimSeparationEvidence:
    """Certify branch-homotopy separation from every overlapping trim translate.

    Shared intentional endpoints require the model's root-incidence algorithm;
    this proof retains endpoint neighborhoods.
    """
    preparations = preparations if preparations is not None else TrimRequestPreparations()
    request = (
        coordinate_source_request((evidence, trim))
        if coordinate_budget is None
        else nullcontext(coordinate_budget)
    )
    with request as coordinate_budget:
        side = parse(side, IntersectionCurveSide, "BranchTrimSeparation.side")
        with _topology_budget(maximum_cells, maximum_depth, maximum_bytes) as budget:
            trim_range = _prepare_trim_range(trim, side, first, last, budget)
            offset = trim_range.offset
            columns = slice(3 + offset, 5 + offset)
            periods = evidence.period_intervals[:, offset : offset + 2]
            rows: list[tuple[float, ...]] = []
            minimum = math.inf
            for cell, tube in enumerate(evidence.homotopy_boxes[:, :, columns]):
                budget.charge(0, pair=True)
                whole = _trim_box(
                    trim,
                    trim_range.first,
                    trim_range.last,
                    preparations,
                    coordinate_budget,
                )
                ranges = _period_translates(tube, whole, periods)
                copies = len(ranges[0]) * len(ranges[1])
                if copies > budget.ticket.remaining_work_units:
                    budget.fail("native-period translate count exceeds maximum_cells")
                for ku in ranges[0]:
                    for kv in ranges[1]:
                        shift_lower, shift_upper = interval_multiply(
                            (periods[0], periods[1]),
                            (np.asarray((ku, kv)), np.asarray((ku, kv))),
                        )
                        shift = (
                            np.asarray(shift_lower, dtype=np.float64),
                            np.asarray(shift_upper, dtype=np.float64),
                        )
                        separated = _separate_trim_translate(
                            cell,
                            tube,
                            trim_range,
                            (ku, kv),
                            shift,
                            budget,
                            preparations,
                            coordinate_budget,
                        )
                        rows.extend(separated.rows)
                        minimum = min(minimum, separated.minimum)
            budget.reserve(len(rows) * 96)
            return BranchTrimSeparationEvidence(
                topology_identity=evidence.approximation_identity,
                trim_identity=_identity(trim),
                side=side,
                cells=budget.work,
                maximum_depth_used=budget.depth,
                bytes_used=budget.peak,
                separation_lower_bound=minimum,
                interval_evidence=_readonly(np.asarray(rows)).reshape((-1, 6)),
            )
