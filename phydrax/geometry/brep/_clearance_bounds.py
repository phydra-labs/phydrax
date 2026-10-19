"""Source-chart interval clearance with a best-first face hierarchy."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import final

import equinox as eqx
import numpy as np

from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from .._atlas import TrimDomain
from .._interval_enclosure import prepare_interval_function, PreparedIntervalFunction
from .._rational import rational_vector
from ._constructors import brep_trim_domain
from ._exact import square_root_interval
from ._intersection_curve import SurfaceRegion
from ._model import BRepModel
from ._patches import coefficient_enclosures
from ._query_membership import full_rectangle
from ._request_preparations import TrimRequestPreparations


@final
class ClearanceFace(StrictModule):
    """A face's native piece cover and optional certified source trim."""

    region: SurfaceRegion
    charts: tuple[PreparedIntervalFunction, ...]
    domains: tuple[np.ndarray, ...]
    trim: TrimDomain | None

    def enclose(self, box: np.ndarray) -> np.ndarray:
        bounds = []
        for chart, domain in zip(self.charts, self.domains, strict=True):
            lower, upper = np.maximum(box[0], domain[0]), np.minimum(box[1], domain[1])
            if np.all(lower <= upper):
                lo, hi = chart.evaluate(lower[None, :], upper[None, :])
                bounds.append(np.stack((lo[0], hi[0])))
        if not bounds:
            raise ValueError(
                "clearance.chart_cover must cover the query; got no source piece"
            )
        return np.stack(
            (
                np.min([b[0] for b in bounds], axis=0),
                np.max([b[1] for b in bounds], axis=0),
            )
        )

    def outside(
        self,
        box: np.ndarray,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations,
    ) -> bool:
        if self.trim is None:
            return False
        classified = self.trim.classify_boxes(
            box[None, :], maximum_refinements=0, budget=budget, preparations=preparations
        )
        return bool(classified.resolved[0] and not classified.inside[0])

    def witness(
        self,
        box: np.ndarray,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations,
    ) -> np.ndarray | None:
        # A sampled point is only a candidate upper bound, never a minimum decision.
        point = box[0] / 2 + box[1] / 2
        if self.trim is not None:
            classified = self.trim.classify(
                point[None, :], maximum_depth=0, budget=budget, preparations=preparations
            )
            if not classified.resolved[0] or not (
                classified.inside[0] or classified.boundary[0]
            ):
                return None
        return self.enclose(np.stack((point, point)))


def prepare_clearance_faces(
    operand: BRepModel | SurfaceRegion, budget: CoordinateEnclosureBudget
) -> tuple[ClearanceFace, ...] | None:
    match operand:
        case SurfaceRegion():
            regions = (operand,)
        case BRepModel() if operand.geometry is not None:
            regions = tuple(
                SurfaceRegion(
                    patch, np.asarray(operand.parameter_bounds)[face], budget=budget
                )
                for face, patch in enumerate(operand.patches)
            )
        case _:
            return None
    faces = []
    for face, region in enumerate(regions):
        box = region.parameter_box
        pieces = region.patch.source_pieces(box, budget=budget)
        charts = tuple(
            prepare_interval_function(
                piece.evaluator.evaluate,
                2,
                batch_capacity=1,
                constant_bounds=coefficient_enclosures(piece.evaluator),
            )
            for piece in pieces
        )
        trim = None
        if isinstance(operand, BRepModel):
            geometry = operand.geometry
            if geometry is None:
                return None
            if not full_rectangle(geometry, face, region.patch, box):
                # This cover diameter is preparation only. All membership decisions
                # still use the source classifier, including its unresolved bands.
                diameter = float(np.sum(np.abs(box[1] - box[0])))
                trim = brep_trim_domain(
                    geometry,
                    face,
                    region.patch,
                    tolerance=np.nextafter(diameter, np.inf),
                    relative_closure_tolerance=0.0,
                    budget=budget,
                )
        faces.append(
            ClearanceFace(
                region,
                charts,
                tuple(np.stack((piece.lower, piece.upper)) for piece in pieces),
                trim,
            )
        )
    return tuple(faces)


def box_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Directed Euclidean distance bounds from binary endpoint arithmetic."""
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        return 0.0, np.inf
    a, b, c, d = (rational_vector(row) for row in (*first, *second))
    gaps = tuple(max(0, x - v, u - y) for x, y, u, v in zip(a, b, c, d, strict=True))
    far = tuple(
        max(abs(x - v), abs(y - u)) for x, y, u, v in zip(a, b, c, d, strict=True)
    )
    return square_root_interval(sum(x * x for x in gaps))[0], square_root_interval(
        sum(x * x for x in far)
    )[1]


@final
class _FaceNode(StrictModule):
    """A deterministic median-split hierarchy over certified face boxes."""

    box: np.ndarray
    indices: tuple[int, ...] = eqx.field(static=True)
    children: tuple[_FaceNode, ...]


def _face_tree(boxes: tuple[np.ndarray, ...], indices: tuple[int, ...]) -> _FaceNode:
    box = np.stack(
        (
            np.min([boxes[i][0] for i in indices], axis=0),
            np.max([boxes[i][1] for i in indices], axis=0),
        )
    )
    if len(indices) == 1:
        return _FaceNode(box, indices, ())
    axis = int(np.argmax(box[1] - box[0]))
    ordered = tuple(
        sorted(indices, key=lambda i: (boxes[i][0, axis] / 2 + boxes[i][1, axis] / 2, i))
    )
    midpoint = len(ordered) // 2
    return _FaceNode(
        box,
        indices,
        (_face_tree(boxes, ordered[:midpoint]), _face_tree(boxes, ordered[midpoint:])),
    )


@final
class _PairCell(StrictModule):
    """One four-parameter rectangle, or one unexpanded face-node pair."""

    first: _FaceNode | None
    second: _FaceNode | None
    pair: tuple[int, int] = eqx.field(static=True)
    boxes: tuple[np.ndarray, ...]


@final
class ClearanceBounds(StrictModule):
    """Bounded result retaining every potentially minimizing face pair."""

    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    witness_lower: np.ndarray
    witness_upper: np.ndarray
    unresolved_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    boxes_processed: int = eqx.field(static=True)


def _node_children(cell: _PairCell) -> tuple[_PairCell, ...]:
    a, b = cell.first, cell.second
    if a is None or b is None:
        raise ValueError(
            "clearance.face_nodes must be present before hierarchy expansion; got absent nodes"
        )
    if not a.children and not b.children:
        return (_PairCell(None, None, (a.indices[0], b.indices[0]), ()),)
    if a.children and (not b.children or len(a.indices) >= len(b.indices)):
        return tuple(_PairCell(child, b, (-1, -1), ()) for child in a.children)
    return tuple(_PairCell(a, child, (-1, -1), ()) for child in b.children)


def _split_boxes(boxes: tuple[np.ndarray, ...]) -> tuple[tuple[np.ndarray, ...], ...]:
    widths = np.concatenate([b[1] - b[0] for b in boxes])
    axis = int(np.argmax(widths))
    side, coordinate = divmod(axis, 2)
    box = boxes[side]
    midpoint = box[0, coordinate] / 2 + box[1, coordinate] / 2
    if not box[0, coordinate] < midpoint < box[1, coordinate]:
        return ()
    first, second = box.copy(), box.copy()
    first[1, coordinate], second[0, coordinate] = midpoint, midpoint
    return tuple(
        tuple(child if k == side else value for k, value in enumerate(boxes))
        for child in (first, second)
    )


def _pending_pairs(cell: _PairCell) -> tuple[tuple[int, int], ...]:
    if cell.first is None or cell.second is None:
        return (cell.pair,)
    return tuple((i, j) for i in cell.first.indices for j in cell.second.indices)


def _pair_witness(
    first: ClearanceFace,
    second: ClearanceFace,
    boxes: tuple[np.ndarray, ...],
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
    upper: float,
    witness: np.ndarray,
    witness_hi: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    a = first.witness(boxes[0], budget, preparations)
    b = second.witness(boxes[1], budget, preparations)
    if a is not None and b is not None:
        candidate = box_distance(a, b)[1]
        if candidate < upper:
            return candidate, np.stack((a[0], b[0])), np.stack((a[1], b[1]))
    return upper, witness, witness_hi


def _finish_bounds(
    queue: list[tuple[float, int, _PairCell]],
    stalled: list[tuple[float, _PairCell]],
    upper: float,
    witness: np.ndarray,
    witness_hi: np.ndarray,
    processed: int,
    solved_lower: list[float],
) -> ClearanceBounds:
    pending = [(lo, cell) for lo, _, cell in queue if lo < upper] + [
        (lo, cell) for lo, cell in stalled if lo < upper
    ]
    lower = min(
        min((lo for lo, _ in pending), default=upper), min(solved_lower, default=upper)
    )
    pairs = tuple(sorted({pair for _, cell in pending for pair in _pending_pairs(cell)}))
    return ClearanceBounds(lower, upper, witness, witness_hi, pairs, processed)


def branch_clearance(
    first: tuple[ClearanceFace, ...],
    second: tuple[ClearanceFace, ...],
    maximum_boxes: int,
    budget: CoordinateEnclosureBudget,
    closed_pair: Callable[[ClearanceFace, ClearanceFace], ClearanceBounds | None],
) -> ClearanceBounds:
    """Refine the least lower bound; discard cells only by a certified upper bound."""
    budget.reserve(
        128 * (maximum_boxes + len(first) + len(second) + 1),
        8192 * (maximum_boxes + len(first) + len(second) + 1),
    )
    preparations = TrimRequestPreparations()
    trees = tuple(
        _face_tree(
            tuple(f.enclose(f.region.parameter_box) for f in faces),
            tuple(range(len(faces))),
        )
        for faces in (first, second)
    )
    queue: list[tuple[float, int, _PairCell]] = []
    serial = 0
    upper = np.inf
    witness = np.empty((0, 3), dtype=np.float64)
    witness_hi = witness
    stalled: list[tuple[float, _PairCell]] = []
    solved_lower: list[float] = []

    def enqueue(cell: _PairCell) -> None:
        nonlocal serial, upper, witness, witness_hi
        if cell.first is not None and cell.second is not None:
            lower = box_distance(cell.first.box, cell.second.box)[0]
        else:
            i, j = cell.pair
            if not cell.boxes:
                closed = closed_pair(first[i], second[j])
                if closed is not None:
                    solved_lower.append(closed.lower)
                    if closed.upper < upper:
                        upper, witness, witness_hi = (
                            closed.upper,
                            closed.witness_lower,
                            closed.witness_upper,
                        )
                    return
            boxes = cell.boxes or (
                first[i].region.parameter_box,
                second[j].region.parameter_box,
            )
            if first[i].outside(boxes[0], budget, preparations) or second[j].outside(
                boxes[1], budget, preparations
            ):
                return
            lower = box_distance(first[i].enclose(boxes[0]), second[j].enclose(boxes[1]))[
                0
            ]
            cell = _PairCell(None, None, cell.pair, boxes)
        if lower < upper:
            heapq.heappush(queue, (lower, serial, cell))
            serial += 1

    enqueue(_PairCell(trees[0], trees[1], (-1, -1), ()))
    processed = 0
    while queue and processed < maximum_boxes:
        lower, _, cell = heapq.heappop(queue)
        if lower >= upper:
            continue
        processed += 1
        if cell.first is not None:
            for child in _node_children(cell):
                enqueue(child)
            continue
        i, j = cell.pair
        upper, witness, witness_hi = _pair_witness(
            first[i],
            second[j],
            cell.boxes,
            budget,
            preparations,
            upper,
            witness,
            witness_hi,
        )
        children = _split_boxes(cell.boxes)
        if not children:
            stalled.append((lower, cell))
        for boxes in children:
            enqueue(_PairCell(None, None, cell.pair, boxes))
    return _finish_bounds(
        queue, stalled, upper, witness, witness_hi, processed, solved_lower
    )
