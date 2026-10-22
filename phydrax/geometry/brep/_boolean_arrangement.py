#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Construct source-oriented curve boundary arrangements for CAD Booleans."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._atlas import AbstractTrimCurve, CurveTrimLoop, PolygonTrimLoop, TrimDomain
from .._interval_enclosure import (
    interval_add,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
)
from ._boolean_coincidence import CoincidentFaceArc, SourceEdgeFacePairCoverage
from ._boolean_core import (
    boolean_face_chart_box,
    BRepBooleanFailure,
    BRepBooleanPolicy,
    MaterialSelection,
)
from ._carrier_pieces import carrier_payload
from ._constructors import assemble_brep_model
from ._intersection import (
    BranchRootEndpoint,
    CurveIntersectionResult,
    CurveSurfaceIntersectionRoot,
    CurveSurfaceSystem,
    intersect_curve_region,
    intersect_trim_curves,
    IntersectionCurvePointRoot,
    joint_root_parameter_image,
    NativePeriodEndpoint,
    ParametricIntersectionPolicy,
    prepare,
    RootEndpoint,
    source_pcurve_image,
    SurfaceIntersectionResult,
    TrimIntersectionRoot,
    TrimRootEndpoint,
    TripleSurfaceIntersectionRoot,
)
from ._intersection_curve import (
    curve_pieces,
    CurveRange,
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionCurveSide,
    IntersectionPCurve,
    prepare_trim_first_jets,
    surface_pieces,
    SurfaceRegion,
)
from ._intersection_kernel import krawczyk
from ._interval_source import evaluate_interval
from ._model import (
    BRepCurve,
    BRepGeometry,
    BRepImportReport,
    BRepModel,
    BRepPCurve,
)
from ._patches import AbstractCurve, AbstractSurfacePatch, coefficient_enclosures
from ._projection_contracts import brep_entity_id
from ._query import prepare_brep_query, PreparedBRepQuery
from ._request_preparations import TrimRequestPreparations
from ._root_bindings import BRepCurveSurfaceLift, BRepRootSupport, BRepVertexRoot
from ._sewing import shells


type FaceOwner = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Arc:
    token: str
    owner: FaceOwner
    trim: AbstractTrimCurve
    curve: BRepCurve | None
    pcurve: BRepPCurve
    offset: float
    scale: float
    first_node: str
    last_node: str
    first_vertex: str
    last_vertex: str
    source_edge: int | None
    branch: IntersectionCurve | None
    parent_token: str | None = None


@dataclass(frozen=True, slots=True)
class Cut:
    """Local directed trim scalar with a native raw-carrier root expression.

    RootEndpoint owns segment unwrapping; its enclosure is not in the local
    ``parameter/lower/upper`` units and must not be affined a second time.
    """

    parameter: float
    lower: float
    upper: float
    node: str
    vertex: str
    endpoint: RootEndpoint | None
    vertex_root: BRepVertexRoot | None
    point: np.ndarray


@dataclass(frozen=True, slots=True)
class Piece:
    arc: Arc
    first: Cut
    last: Cut
    sense: int


@dataclass
class Arrangement:
    """Accumulate source topology and scientific box-work evidence.

    The root-search cap is independent of the request's work/byte ticket.
    Interval and coordinate operations use CoordinateEnclosureBudget for
    resource accounting; boxes_used retains discovery and material evidence.
    """

    models: tuple[BRepModel, ...]
    selection: MaterialSelection
    policy: BRepBooleanPolicy
    queries: tuple[PreparedBRepQuery, ...]
    arcs: dict[FaceOwner, list[Arc]] = field(default_factory=dict)
    domains: dict[FaceOwner, TrimDomain] = field(default_factory=dict)
    cuts: dict[str, list[Cut]] = field(default_factory=dict)
    spatial_roots: dict[
        tuple[int, int, int, int], tuple[CurveSurfaceIntersectionRoot, ...]
    ] = field(default_factory=dict)
    vertex_aliases: dict[str, list[tuple[AbstractSurfacePatch, TrimIntersectionRoot]]] = (
        field(default_factory=dict)
    )
    vertex_spatial: dict[str, CurveSurfaceIntersectionRoot] = field(default_factory=dict)
    vertex_joint: dict[str, TripleSurfaceIntersectionRoot] = field(default_factory=dict)
    vertex_lifts: dict[str, tuple[BRepCurveSurfaceLift, ...]] = field(
        default_factory=dict
    )
    branches: dict[str, tuple[IntersectionCurve, FaceOwner, FaceOwner]] = field(
        default_factory=dict
    )
    branch_owners: dict[str, tuple[FaceOwner, FaceOwner]] = field(default_factory=dict)
    # Branch sides proved (source lift + certified chart uniqueness) to be a
    # present source edge of that face (its own, or a coincident partner's):
    # branch id -> {face: (source model, source edge)}.
    branch_source_edges: dict[str, dict[FaceOwner, tuple[int, int]]] = field(
        default_factory=dict
    )
    source_pair_coverage: dict[
        tuple[FaceOwner, FaceOwner], SourceEdgeFacePairCoverage
    ] = field(default_factory=dict)
    source_adjacencies: dict[tuple[FaceOwner, FaceOwner], tuple[int, ...]] = field(
        default_factory=dict
    )
    roots: dict[str, TrimIntersectionRoot] = field(default_factory=dict)
    vertex_bindings: dict[str, BRepVertexRoot | None] = field(default_factory=dict)
    vertex_branch_points: dict[str, tuple[IntersectionCurve, float]] = field(
        default_factory=dict
    )
    spatial_pairs: set[tuple[str, str]] = field(default_factory=set)
    # Request-lifetime source face charts; see ``boolean_face_chart_box``.
    face_charts: dict[FaceOwner, np.ndarray] = field(default_factory=dict)
    boxes_used: int = 0

    def consume(self, boxes: int) -> None:
        self.boxes_used += boxes
        if self.boxes_used > self.policy.intersection.maximum_boxes:
            raise BRepBooleanFailure(
                "BRepBoolean.intersection.work must fit maximum_boxes; got coupled intersection work budget exhausted"
            )

    @property
    def remaining_boxes(self) -> int:
        return max(0, self.policy.intersection.maximum_boxes - self.boxes_used)

    def face_chart(
        self,
        owner: FaceOwner,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> np.ndarray:
        """Source face box in the chart shared by every Boolean region of the face."""
        box = self.face_charts.get(owner)
        if box is None:
            box = boolean_face_chart_box(
                self.models[owner[0]],
                owner[1],
                preparations=preparations,
                budget=budget,
            )
            self.face_charts[owner] = box
        return box.copy()

    def face_region(
        self,
        owner: FaceOwner,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> SurfaceRegion:
        return SurfaceRegion(
            self.models[owner[0]].patches[owner[1]],
            self.face_chart(owner, preparations=preparations, budget=budget),
            preparations=preparations,
            budget=budget,
        )


@checked
def remaining_intersection_policy(state: Arrangement) -> ParametricIntersectionPolicy:
    remaining = state.remaining_boxes
    if remaining <= 0:
        raise BRepBooleanFailure(
            "BRepBoolean.root_work must have remaining boxes; got exhausted root work budget"
        )
    policy = state.policy.intersection
    return ParametricIntersectionPolicy(
        maximum_boxes=remaining,
        relative_resolution=policy.relative_resolution,
        relative_probe=policy.relative_probe,
        relative_coincidence=policy.relative_coincidence,
        relative_tangency=policy.relative_tangency,
        relative_step=policy.relative_step,
        relative_minimum_step=policy.relative_minimum_step,
        maximum_turn=policy.maximum_turn,
        maximum_march_steps=policy.maximum_march_steps,
        batch=policy.batch,
        relative_junction=policy.relative_junction,
    )


def _face_id(model: BRepModel, face: int) -> str:
    return f"{model.source_revision}:face:{face}"


def overlap(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.all(first[0] <= second[1]) and np.all(second[0] <= first[1]))


class ArcPairIntersection(StrictModule):
    result: CurveIntersectionResult | None


@checked
def intersect_arc_pair(
    first: Arc,
    second: Arc,
    policy: ParametricIntersectionPolicy | Arrangement,
    *,
    intersection: Callable[
        [AbstractTrimCurve, AbstractTrimCurve], CurveIntersectionResult
    ]
    | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> ArcPairIntersection:
    """Intersect overlapping arc enclosures without charging caller work counters."""
    first_box = first.trim.enclosure(
        *first.trim.parameter_interval, preparations=preparations, budget=budget
    )
    second_box = second.trim.enclosure(
        *second.trim.parameter_interval, preparations=preparations, budget=budget
    )
    if not overlap(first_box, second_box):
        return ArcPairIntersection(None)
    if intersection is not None:
        return ArcPairIntersection(intersection(first.trim, second.trim))
    match policy:
        case Arrangement():
            policy_ = remaining_intersection_policy(policy)
        case ParametricIntersectionPolicy():
            policy_ = policy
        case _:
            raise TypeError(
                "intersect_arc_pair.policy must be an intersection policy or arrangement; "
                f"got {type(policy).__name__}"
            )
    result = intersect_trim_curves(
        first.trim,
        second.trim,
        policy=policy_,
        preparations=preparations,
        budget=budget,
    )
    return ArcPairIntersection(result)


@checked
def source_arcs(
    state: Arrangement,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> None:
    for model_index, model in enumerate(state.models):
        geometry = model.geometry
        if geometry is None:
            raise BRepBooleanFailure(
                "BRepBoolean.source.geometry must contain curve/face topology; got None",
                (model.model_id,),
            )
        points = np.asarray(geometry.vertex_points)
        ranges = np.asarray(geometry.edge_ranges)
        for face, loops in enumerate(geometry.face_loops):
            owner = model_index, face
            arcs = []
            for loop_index, loop in enumerate(loops):
                for ordinal, coedge in enumerate(loop):
                    edge = geometry.coedge_edges[coedge]
                    sense = geometry.coedge_senses[coedge]
                    first, last = (float(value) for value in ranges[edge])
                    pcurve = geometry.pcurves[coedge]
                    first_root, last_root = geometry.coedge_endpoint_roots[coedge]
                    trim = CurveTrimSegment(
                        pcurve,
                        first,
                        last,
                        reversed=sense < 0,
                        first_root=first_root,
                        last_root=last_root,
                        preparations=preparations,
                        budget=budget,
                    )
                    start, end = geometry.edge_vertices[edge]
                    if sense < 0:
                        start, end = end, start
                        first_root, last_root = last_root, first_root
                    curve_index = geometry.edge_curves[edge]
                    curve = None if curve_index < 0 else geometry.curves[curve_index]
                    token = f"source:{model_index}:coedge:{coedge}"
                    first_node = (
                        f"face:{model_index}:{face}:loop:{loop_index}:corner:{ordinal}"
                    )
                    last_node = f"face:{model_index}:{face}:loop:{loop_index}:corner:{(ordinal + 1) % len(loop)}"
                    first_vertex = f"source:{model_index}:vertex:{start}"
                    last_vertex = f"source:{model_index}:vertex:{end}"
                    arc = Arc(
                        token,
                        owner,
                        trim,
                        curve,
                        pcurve,
                        first if sense > 0 else last,
                        (last - first) * sense,
                        first_node,
                        last_node,
                        first_vertex,
                        last_vertex,
                        edge,
                        None,
                    )
                    arcs.append(arc)
                    state.cuts[token] = [
                        Cut(
                            0.0,
                            0.0,
                            0.0,
                            first_node,
                            first_vertex,
                            first_root,
                            geometry.vertex_roots[start],
                            points[start],
                        ),
                        Cut(
                            1.0,
                            1.0,
                            1.0,
                            last_node,
                            last_vertex,
                            last_root,
                            geometry.vertex_roots[end],
                            points[end],
                        ),
                    ]
            state.arcs[owner] = arcs
            state.domains[owner] = state.queries[model_index].faces[face].trim_domain


@checked
def check_surface_result(
    state: Arrangement,
    result: SurfaceIntersectionResult,
    first: FaceOwner,
    second: FaceOwner,
) -> None:
    state.consume(result.work.boxes_processed + result.work.march_steps)
    sources = tuple(
        _face_id(state.models[index], face) for index, face in (first, second)
    )
    if not result.complete:
        reasons = ",".join(sorted({region.reason for region in result.unresolved}))
        raise BRepBooleanFailure(
            f"BRepBoolean.surface_intersection must resolve discovery; got {reasons}",
            sources,
        )
    if result.coincident:
        raise BRepBooleanFailure(
            "BRepBoolean.coincident_faces must have certified parameter correspondence; got unresolved correspondence",
            sources,
        )


def periodic_transition(
    curve: IntersectionCurve,
    transition: int,
    side: IntersectionCurveSide | None = None,
) -> bool:
    periods = (*curve.first.patch.periods, *curve.second.patch.periods)
    columns = range(4) if side is None else (range(2) if side == "first" else range(2, 4))
    return any(
        period is not None
        and round(float(curve.transition_shifts[transition, column]) / period) != 0
        for column in columns
        for period in (periods[column],)
    )


@checked
def branch_arc(
    state: Arrangement,
    curve: IntersectionCurve,
    owner: FaceOwner,
    side: IntersectionCurveSide,
    first: int,
    last: int,
    *,
    owners: tuple[FaceOwner, FaceOwner],
) -> None:
    pcurve = curve.p_curve(side)
    local = IntersectionPCurve(curve, side, first=float(first), last=float(last))
    first_vertex = f"branch:{curve.branch_id}:node:{first}"
    final = 0 if curve.closed and last == curve.num_charts else last
    last_vertex = f"branch:{curve.branch_id}:node:{final}"
    first_seam = periodic_transition(curve, (first - 1) % curve.num_charts, side)
    last_seam = periodic_transition(curve, (last - 1) % curve.num_charts, side)
    first_sheet = "after" if first_seam else "common"
    last_sheet = "before" if last_seam else "common"
    first_node = f"{first_vertex}:face:{owner}:sheet:{first_sheet}"
    last_node = f"{last_vertex}:face:{owner}:sheet:{last_sheet}"
    token = f"branch:{curve.branch_id}:owners:{owners}:face:{owner}:span:{first}:{last}"
    state.branch_owners[token] = owners
    arc = Arc(
        token,
        owner,
        local,
        curve,
        pcurve,
        0.0,
        1.0,
        first_node,
        last_node,
        first_vertex,
        last_vertex,
        None,
        curve,
    )
    state.arcs[owner].append(arc)
    state.vertex_branch_points[first_vertex] = curve, float(first)
    state.vertex_branch_points[last_vertex] = curve, float(final)
    first_binding = BRepVertexRoot.from_curve_point(curve, float(first))
    last_binding = BRepVertexRoot.from_curve_point(curve, float(last))
    if not isinstance(
        first_binding.primary, IntersectionCurvePointRoot
    ) or not isinstance(
        last_binding.primary,
        IntersectionCurvePointRoot,
    ):
        raise RuntimeError(
            "BRepBoolean.branch.endpoints must retain fixed parameter roots; got another endpoint family"
        )
    if final == 0 and first == 0 and first_binding.root_id != last_binding.root_id:
        raise BRepBooleanFailure(
            "BRepBoolean.closed_branch.endpoints must share their vertex identity; got different identities",
            (curve.branch_id,),
        )
    first_endpoint = TrimRootEndpoint(first_binding.primary, "first")
    last_endpoint = TrimRootEndpoint(last_binding.primary, "first")
    state.vertex_bindings.setdefault(first_vertex, first_binding)
    state.vertex_bindings.setdefault(last_vertex, last_binding)
    state.cuts[token] = [
        Cut(
            float(first),
            float(first),
            float(first),
            first_node,
            first_vertex,
            first_endpoint,
            first_binding,
            first_binding.evaluate()[0],
        ),
        Cut(
            float(last),
            float(last),
            float(last),
            last_node,
            last_vertex,
            last_endpoint,
            last_binding,
            last_binding.evaluate()[0],
        ),
    ]


def arc_area_bounds(
    curve: AbstractTrimCurve,
    first: float,
    last: float,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:
    box = curve.enclosure(first, last, preparations=preparations, budget=budget)
    derivative = curve.derivative_bounds(
        first, last, preparations=preparations, budget=budget
    )
    forward = interval_multiply(
        (box[0, 0], box[1, 0]), (derivative[0][1], derivative[1][1])
    )
    backward = interval_multiply(
        (box[0, 1], box[1, 1]), (derivative[0][0], derivative[1][0])
    )
    integrand = interval_subtract(forward, backward)
    width = np.asarray(0.5 * (last - first), dtype=np.float64)
    result = interval_multiply(
        integrand, (np.nextafter(width, -np.inf), np.nextafter(width, np.inf))
    )
    return float(result[0]), float(result[1])


def arc_area_bounds_batch(
    cells: Sequence[tuple[AbstractTrimCurve, float, float]],
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> list[tuple[float, float]]:
    """Evaluate the same Green-integral bounds with batched chart preparation."""
    intervals = []
    for start in range(0, len(cells), 64):
        budget.ticket.checkpoint()
        batch = cells[start : start + 64]
        prepare_trim_first_jets(batch, preparations=preparations, budget=budget)
        intervals.extend(
            arc_area_bounds(curve, first, last, preparations=preparations, budget=budget)
            for curve, first, last in batch
        )
    return intervals


def _loop_sign(
    loop: CurveTrimLoop,
    maximum_cells: int,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> int:
    if preparations is None:
        preparations = TrimRequestPreparations()
    cells = [
        (loop.curves[int(index)], float(first), float(last))
        for index, first, last in zip(
            loop.arc_curves, loop.arc_first, loop.arc_last, strict=True
        )
    ]
    if len(cells) > maximum_cells:
        raise BRepBooleanFailure(
            "BRepBoolean.trim.cover must fit its area-sign work limit; got an oversized cover"
        )
    while True:
        intervals = arc_area_bounds_batch(
            cells, preparations=preparations, budget=budget
        )
        lower, upper = np.asarray(0.0, np.float64), np.asarray(0.0, np.float64)
        for a, b in intervals:
            lower, upper = interval_add((lower, upper), (np.asarray(a), np.asarray(b)))
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        if lower > 0.0:
            return 1
        if upper < 0.0:
            return -1
        if len(cells) * 2 > maximum_cells:
            raise BRepBooleanFailure(
                "BRepBoolean.trim.area_sign must fit its work limit; got area-sign budget exhausted"
            )
        following = []
        for curve, first, last in cells:
            middle = 0.5 * (first + last)
            if not first < middle < last:
                raise BRepBooleanFailure(
                    "BRepBoolean.trim.area_sign must separate from zero; got an unresolved sign"
                )
            following.extend(((curve, first, middle), (curve, middle, last)))
        cells = following


@checked
def trim_box_status(
    domain: TrimDomain,
    box: np.ndarray,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> int:
    for loop in domain.loops:
        match loop:
            case CurveTrimLoop():
                lower = np.concatenate((loop.arc_lower, loop.junction_lower))
                upper = np.concatenate((loop.arc_upper, loop.junction_upper))
            case PolygonTrimLoop():
                points = np.asarray(loop.vertices)
                following = np.roll(points, -1, axis=0)
                lower, upper = (
                    np.minimum(points, following),
                    np.maximum(points, following),
                )
            case _:
                raise TypeError(
                    "BRepBoolean.trim.loops must use supported polygon or curve loops; got another representation"
                )
        if np.any(np.all(lower <= box[1], axis=1) & np.all(upper >= box[0], axis=1)):
            return 0
    decision = domain.classify(
        (0.5 * (box[0] + box[1]))[None], preparations=preparations, budget=budget
    )
    if not decision.resolved[0]:
        return 0
    return 1 if decision.inside[0] else -1


def _shell_volume_sign(
    patches: tuple[AbstractSurfacePatch, ...],
    bounds: np.ndarray,
    domains: tuple[TrimDomain, ...],
    signs: np.ndarray,
    faces: tuple[int, ...],
    maximum_cells: int,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> int:
    functions = []
    cell_faces = []
    cells = []
    for face in faces:
        for piece in surface_pieces(
            SurfaceRegion(
                patches[face], bounds[face], budget=budget, preparations=preparations
            ),
            budget=budget,
        ):
            if len(functions) >= maximum_cells:
                raise BRepBooleanFailure(
                    "BRepBoolean.shell.cover must fit maximum_cells; got an oversized initial cover"
                )
            evaluator = piece.evaluator

            def integrand(
                uv: jax.Array,
                evaluator: AbstractSurfacePatch = evaluator,
            ) -> jax.Array:
                point = evaluator.evaluate(uv)
                frame = jax.jacfwd(evaluator.evaluate)(uv)
                return (point @ jnp.cross(frame[:, 0], frame[:, 1])) / 3.0

            index = len(functions)
            functions.append(
                prepare_interval_function(
                    integrand,
                    2,
                    batch_capacity=1,
                    constant_bounds=coefficient_enclosures(evaluator),
                )
            )
            cell_faces.append(face)
            cells.append((index, np.stack((piece.lower, piece.upper))))

    def cell_contribution(
        index: int,
        box: np.ndarray,
        *,
        preparations: TrimRequestPreparations | None = preparations,
        budget: CoordinateEnclosureBudget = budget,
    ) -> tuple[np.ndarray, np.ndarray]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        face = cell_faces[index]
        status = trim_box_status(
            domains[face], box, preparations=preparations, budget=budget
        )
        if status < 0:
            return np.asarray(0.0, np.float64), np.asarray(0.0, np.float64)
        lower, upper = evaluate_interval(
            functions[index],
            box[0][None],
            box[1][None],
            budget=budget,
            preparations=preparations,
        )
        integrand_bounds = lower[0], upper[0]
        if signs[face] < 0:
            integrand_bounds = -integrand_bounds[1], -integrand_bounds[0]
        if status == 0:
            integrand_bounds = (
                np.minimum(integrand_bounds[0], 0.0),
                np.maximum(integrand_bounds[1], 0.0),
            )
        area = interval_multiply(
            (
                np.nextafter(box[1, 0] - box[0, 0], -np.inf),
                np.nextafter(box[1, 0] - box[0, 0], np.inf),
            ),
            (
                np.nextafter(box[1, 1] - box[0, 1], -np.inf),
                np.nextafter(box[1, 1] - box[0, 1], np.inf),
            ),
        )
        contribution = interval_multiply(integrand_bounds, area)
        return (
            np.asarray(contribution[0], dtype=np.float64),
            np.asarray(contribution[1], dtype=np.float64),
        )

    bounded_cells = [
        (
            index,
            box,
            cell_contribution(index, box, preparations=preparations, budget=budget),
        )
        for index, box in cells
    ]
    while True:
        total = np.asarray(0.0, np.float64), np.asarray(0.0, np.float64)
        widths = []
        for _, _, contribution in bounded_cells:
            total = interval_add(total, contribution)
            widths.append(float(contribution[1] - contribution[0]))
        total_lower = np.asarray(total[0], dtype=np.float64)
        total_upper = np.asarray(total[1], dtype=np.float64)
        if total_lower > 0.0:
            return 1
        if total_upper < 0.0:
            return -1
        if len(bounded_cells) >= maximum_cells:
            raise BRepBooleanFailure(
                "BRepBoolean.shell.volume_sign must fit its work limit; got volume-sign budget exhausted"
            )
        worst = int(np.argmax(widths))
        index, box, _ = bounded_cells.pop(worst)
        axis = int(np.argmax(box[1] - box[0]))
        middle = 0.5 * (box[0, axis] + box[1, axis])
        if not box[0, axis] < middle < box[1, axis]:
            raise BRepBooleanFailure(
                "BRepBoolean.shell.volume_sign must separate from zero; got an unresolved sign"
            )
        first, second = box.copy(), box.copy()
        first[1, axis], second[0, axis] = middle, middle
        bounded_cells.append(
            (
                index,
                first,
                cell_contribution(index, first, preparations=preparations, budget=budget),
            )
        )
        bounded_cells.append(
            (
                index,
                second,
                cell_contribution(
                    index, second, preparations=preparations, budget=budget
                ),
            )
        )


def _edge_spatial_roots(
    state: Arrangement,
    source: FaceOwner,
    edge: int,
    opposite: FaceOwner,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[CurveSurfaceIntersectionRoot, ...]:
    key = source[0], edge, opposite[0], opposite[1]
    if key in state.spatial_roots:
        return state.spatial_roots[key]
    model = state.models[source[0]]
    geometry = model.geometry
    if geometry is None:
        raise BRepBooleanFailure(
            "BRepBoolean.source_edge.geometry must be defined; got None"
        )
    curve_index = geometry.edge_curves[edge]
    if curve_index < 0:
        raise BRepBooleanFailure(
            "BRepBoolean.source_edge.curve must be regular at the branch; got a degenerate edge"
        )
    curve = geometry.curves[curve_index]
    if isinstance(curve, IntersectionCurve):
        raise BRepBooleanFailure(
            "BRepBoolean.source_edge.curve must support coupled spatial range isolation; got an intersection carrier",
            (f"{model.source_revision}:edge:{edge}",),
        )
    first, last = np.asarray(geometry.edge_ranges)[edge]
    region = state.face_region(opposite, preparations=preparations, budget=budget)
    result = intersect_curve_region(
        CurveRange(curve, float(first), float(last)),
        region,
        policy=remaining_intersection_policy(state),
        preparations=preparations,
        coordinate_budget=budget,
    )
    state.consume(result.work.boxes_processed)
    if not result.complete or result.coincident:
        raise BRepBooleanFailure(
            "BRepBoolean.source_edge.roots must be isolated and resolved; got unresolved or coincident roots",
            (
                f"{model.source_revision}:edge:{edge}",
                _face_id(state.models[opposite[0]], opposite[1]),
            ),
        )
    points = sorted(
        (point for point in result.points if point.kind == "crossing"),
        key=lambda point: float(point.parameters[0]),
    )
    roots = tuple(
        CurveSurfaceIntersectionRoot(
            curve,
            region,
            parameter_lower=point.parameter_lower,
            parameter_upper=point.parameter_upper,
            budget=budget,
            preparations=preparations,
        )
        for point in points
    )
    if any(
        roots[index].parameter_upper[0] >= roots[index + 1].parameter_lower[0]
        for index in range(len(roots) - 1)
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.source_edge.roots must have disjoint ordered intervals; got overlapping intervals"
        )
    state.spatial_roots[key] = roots
    return roots


def _normalize_periodic_box(
    lower: np.ndarray,
    upper: np.ndarray,
    reference: np.ndarray,
    region: SurfaceRegion,
) -> tuple[np.ndarray, np.ndarray]:
    lower_, upper_ = lower.copy(), upper.copy()
    for axis, periodic in enumerate(region.periodic):
        if periodic:
            period = region.upper[axis] - region.lower[axis]
            shift = round(
                float((0.5 * (lower[axis] + upper[axis]) - reference[axis]) / period)
            )
            offset = np.asarray(shift * period, np.float64)
            lower_[axis], upper_[axis] = interval_subtract(
                (np.asarray(lower_[axis]), np.asarray(upper_[axis])),
                (offset, offset),
            )
    return lower_, upper_


def _prove_edge_root_alias(
    state: Arrangement,
    root: TrimIntersectionRoot,
    source_arc: Arc,
    branch_arc: Arc,
    source_operand: int,
    branch_operand: int,
    spatial: CurveSurfaceIntersectionRoot,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> None:

    source_endpoint = TrimRootEndpoint(root, "first" if source_operand == 0 else "second")
    t_lower, t_upper = source_endpoint.parameter_enclosure(
        preparations=preparations, budget=budget
    )
    branch = branch_arc.branch
    if branch is None:
        raise BRepBooleanFailure(
            "BRepBoolean.spatial_alias.branch must be defined; got None"
        )
    tau_lower = max(0.0, float(root.parameter_lower[branch_operand]))
    tau_upper = min(float(branch.num_charts), float(root.parameter_upper[branch_operand]))
    coupled = branch.parameter_enclosures(
        tau_lower, tau_upper, preparations=preparations, budget=budget
    )
    first, second = state.branch_owners[branch_arc.token]
    columns = slice(2, 4) if branch_arc.owner == first else slice(0, 2)
    uv_lower = np.min(coupled[:, 0, columns], axis=0)
    uv_upper = np.max(coupled[:, 1, columns], axis=0)
    uv_lower, uv_upper = _normalize_periodic_box(
        uv_lower,
        uv_upper,
        spatial.parameters[1:],
        spatial.surface,
    )
    lower = np.minimum(
        spatial.parameter_lower,
        np.concatenate((np.asarray((t_lower,), np.float64), uv_lower)),
    )
    upper = np.maximum(
        spatial.parameter_upper,
        np.concatenate((np.asarray((t_upper,), np.float64), uv_upper)),
    )
    margin = (
        64.0 * np.finfo(np.float64).eps * (1.0 + np.maximum(np.abs(lower), np.abs(upper)))
    )
    lower, upper = (
        np.nextafter(lower - margin, -np.inf),
        np.nextafter(upper + margin, np.inf),
    )
    if isinstance(spatial.curve, IntersectionCurve):
        evaluator = spatial.curve
    else:
        pieces = curve_pieces(
            CurveRange(spatial.curve, float(lower[0]), float(upper[0])), budget=budget
        )
        if len(pieces) != 1:
            raise BRepBooleanFailure(
                "BRepBoolean.spatial_alias.curve must lie in one source knot piece; got multiple pieces"
            )
        evaluator = pieces[0].evaluator
    pieces = [
        piece
        for piece in surface_pieces(spatial.surface, budget=budget)
        if np.all(lower[1:] >= piece.lower) and np.all(upper[1:] <= piece.upper)
    ]
    if len(pieces) != 1:
        raise BRepBooleanFailure(
            "BRepBoolean.spatial_alias.surface must lie in one source chart piece; got multiple pieces"
        )
    prepared = prepare(
        CurveSurfaceSystem(evaluator, pieces[0].evaluator),
        3,
        1,
        preparations=preparations,
    )
    state.consume(1)
    inclusion = krawczyk(
        prepared, lower[None], upper[None], budget=budget, preparations=preparations
    )
    if not bool(inclusion.certified[0]):
        raise BRepBooleanFailure(
            "BRepBoolean.spatial_alias.roots must share a unique spatial certificate; got failed certification",
            (source_arc.token, branch_arc.token),
        )


def event_vertex(
    state: Arrangement,
    root: TrimIntersectionRoot,
    first: Arc,
    second: Arc,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> str:
    source_arc = first if first.source_edge is not None else second
    branch_arc = second if source_arc is first else first
    if source_arc.source_edge is not None and branch_arc.branch is not None:
        source_operand, branch_operand = (0, 1) if source_arc is first else (1, 0)
        owner_a, owner_b = state.branch_owners[branch_arc.token]
        opposite = owner_b if source_arc.owner == owner_a else owner_a
        roots = _edge_spatial_roots(
            state,
            source_arc.owner,
            source_arc.source_edge,
            opposite,
            preparations=preparations,
            budget=budget,
        )
        endpoint = TrimRootEndpoint(root, "first" if source_operand == 0 else "second")
        lower, upper = endpoint.parameter_enclosure(
            preparations=preparations, budget=budget
        )
        matches = [
            spatial
            for spatial in roots
            if lower <= spatial.parameter_upper[0] and upper >= spatial.parameter_lower[0]
        ]
        if len(matches) != 1:
            raise BRepBooleanFailure(
                "BRepBoolean.trim_root must correspond to one spatial root; got another candidate count",
                (source_arc.token, branch_arc.token),
            )
        spatial = matches[0]
        _prove_edge_root_alias(
            state,
            root,
            source_arc,
            branch_arc,
            source_operand,
            branch_operand,
            spatial,
            preparations=preparations,
            budget=budget,
        )
        vertex = f"spatial-root:{spatial.root_id}"
        state.vertex_spatial[vertex] = spatial
    elif first.branch is not None and second.branch is not None:
        first_owners = state.branch_owners[first.token]
        second_owners = state.branch_owners[second.token]
        other_a = first_owners[1] if first.owner == first_owners[0] else first_owners[0]
        other_b = (
            second_owners[1] if second.owner == second_owners[0] else second_owners[0]
        )
        if other_a[0] != other_b[0]:
            raise BRepBooleanFailure(
                "BRepBoolean.triple_junction.sources must agree on source ownership; got incompatible owners"
            )
        model = state.models[other_a[0]]
        common = set(model.topology.face_edges[other_a[1]]) & set(
            model.topology.face_edges[other_b[1]]
        )
        candidates = [
            (edge, spatial)
            for edge in sorted(common)
            for spatial in _edge_spatial_roots(
                state,
                other_a,
                edge,
                first.owner,
                preparations=preparations,
                budget=budget,
            )
        ]
        point_box = np.asarray(
            state.models[first.owner[0]]
            .patches[first.owner[1]]
            .bounding_box(
                root.point_enclosure(preparations=preparations, budget=budget),
                preparations=preparations,
                budget=budget,
            )
        )
        matches = [
            (edge, spatial)
            for edge, spatial in candidates
            if overlap(
                point_box,
                spatial.point_enclosure(preparations=preparations, budget=budget),
            )
        ]
        if len(matches) != 1:
            raise BRepBooleanFailure(
                "BRepBoolean.triple_junction.source_edge must have one supporting root; got another root count"
            )
        edge, spatial = matches[0]
        joint, lifts = _triple_vertex_certificate(
            state,
            root,
            first.owner,
            other_a,
            other_b,
            edge,
            spatial,
            preparations=preparations,
            budget=budget,
        )
        vertex = f"spatial-root:{spatial.root_id}"
        state.vertex_spatial[vertex] = spatial
        state.vertex_joint[vertex] = joint
        state.vertex_lifts[vertex] = lifts
    else:
        raise BRepBooleanFailure(
            "BRepBoolean.source_boundary must declare every self-intersection; got an undeclared intersection"
        )
    patch = state.models[first.owner[0]].patches[first.owner[1]]
    state.vertex_aliases.setdefault(vertex, []).append((patch, root))
    return vertex


def add_root_cut(
    state: Arrangement,
    arc: Arc,
    root: TrimIntersectionRoot,
    operand: int,
    vertex: str,
) -> None:
    parameter = float(root.parameters[operand])
    lower, upper = (
        float(root.parameter_lower[operand]),
        float(root.parameter_upper[operand]),
    )
    endpoint = TrimRootEndpoint(root, "first" if operand == 0 else "second")
    uv, _, _ = root.evaluate()
    patch = state.models[arc.owner[0]].patches[arc.owner[1]]
    point = np.asarray(patch.evaluate(jnp.asarray(uv, jnp.float64)))
    node = f"root:{root.root_id}:face:{arc.owner}"
    first, last = arc.trim.parameter_interval
    if lower <= first <= upper:
        node = arc.first_node
    elif lower <= last <= upper:
        node = arc.last_node
    state.cuts[arc.token].append(
        Cut(parameter, lower, upper, node, vertex, endpoint, None, point)
    )


def piece_trim(
    piece: Piece,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> CurveTrimSegment:
    arc = piece.arc
    a = arc.offset + arc.scale * piece.first.parameter
    b = arc.offset + arc.scale * piece.last.parameter
    first, last = (a, b) if a < b else (b, a)
    first_root, last_root = (
        (piece.first.endpoint, piece.last.endpoint)
        if arc.scale > 0.0
        else (piece.last.endpoint, piece.first.endpoint)
    )
    return CurveTrimSegment(
        arc.pcurve,
        first,
        last,
        reversed=piece.sense * arc.scale < 0.0,
        first_root=first_root,
        last_root=last_root,
        preparations=preparations,
        budget=budget,
    )


def ordered_cuts(state: Arrangement, arc: Arc) -> tuple[Cut, ...]:
    cuts = sorted(state.cuts[arc.token], key=lambda cut: cut.parameter)
    result = []
    for cut in cuts:
        if result and result[-1].upper >= cut.lower:
            previous = result[-1]
            if previous.vertex != cut.vertex:
                raise BRepBooleanFailure(
                    "BRepBoolean.arc.cuts must have ordered intervals or common vertex identity; got incompatible overlapping cuts",
                    (arc.token,),
                )
            if previous.endpoint is None and cut.endpoint is not None:
                result[-1] = cut
            continue
        result.append(cut)
    return tuple(result)


def propagate_branch_cuts(state: Arrangement) -> None:
    branch_cuts: dict[str, dict[str, tuple[Cut, Arc]]] = {}
    for arcs in state.arcs.values():
        for arc in arcs:
            if arc.branch is None:
                continue
            for cut in state.cuts[arc.token]:
                if cut.endpoint is not None:
                    branch_cuts.setdefault(arc.branch.branch_id, {}).setdefault(
                        cut.vertex, (cut, arc)
                    )
    for owner, arcs in state.arcs.items():
        for arc in arcs:
            if arc.branch is None:
                continue
            first, last = arc.trim.parameter_interval
            local = {
                cut.vertex for cut in state.cuts[arc.token] if cut.endpoint is not None
            }
            for vertex, (cut, original) in branch_cuts.get(
                arc.branch.branch_id, {}
            ).items():
                if vertex in local or not first < cut.parameter < last:
                    continue
                state.cuts[arc.token].append(
                    Cut(
                        cut.parameter,
                        cut.lower,
                        cut.upper,
                        f"transported:{vertex}:face:{owner}",
                        vertex,
                        cut.endpoint,
                        cut.vertex_root,
                        cut.point,
                    )
                )
            if len(state.cuts[arc.token]) > state.policy.sewing.maximum_coedges:
                raise BRepBooleanFailure(
                    "BRepBoolean.branch.cuts must fit maximum_coedges; got branch cut capacity exhausted"
                )


def piece_nodes(piece: Piece) -> tuple[str, str]:
    return (
        (piece.first.node, piece.last.node)
        if piece.sense > 0
        else (piece.last.node, piece.first.node)
    )


@dataclass(frozen=True, slots=True)
class FaceFragment:
    owner: FaceOwner
    loops: tuple[tuple[Piece, ...], ...]
    domain: TrimDomain
    orientation: int


class BuiltBoundary(StrictModule):
    geometry: BRepGeometry
    patches: tuple[AbstractSurfacePatch, ...]
    bounds: np.ndarray
    orientation: np.ndarray
    domains: tuple[TrimDomain, ...]
    sources: tuple[FaceOwner, ...] = eqx.field(static=True)
    tags: tuple[str, ...] = eqx.field(static=True)


def _vertex_binding(
    state: Arrangement,
    vertex: str,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> BRepVertexRoot | None:
    if vertex in state.vertex_bindings:
        return state.vertex_bindings[vertex]
    if vertex in state.vertex_spatial:
        aliases = tuple(
            BRepRootSupport(patch, root, budget=budget, preparations=preparations)
            for patch, root in state.vertex_aliases[vertex]
        )
        binding = BRepVertexRoot(
            state.vertex_spatial[vertex],
            aliases=aliases,
            joint_root=state.vertex_joint.get(vertex),
            source_edge_lifts=state.vertex_lifts.get(vertex, ()),
            budget=budget,
            preparations=preparations,
        )
    elif vertex in state.vertex_branch_points:
        curve, parameter = state.vertex_branch_points[vertex]
        binding = BRepVertexRoot.from_curve_point(curve, parameter)
    else:
        model_text, index_text = vertex.removeprefix("source:").split(":vertex:", 1)
        geometry = state.models[int(model_text)].geometry
        if geometry is None:
            raise BRepBooleanFailure(
                "BRepBoolean.source_vertex.geometry must be defined; got None"
            )
        binding = geometry.vertex_roots[int(index_text)]
    state.vertex_bindings[vertex] = binding
    return binding


def _edge_key(
    piece: Piece, source: CoincidentFaceArc | None = None
) -> tuple[str, str, str]:
    first, last = piece.first.vertex, piece.last.vertex
    if piece.arc.scale < 0.0:
        first, last = last, first
    if source is not None:
        entity = source.edge_id
        carrier = brep_entity_id(
            entity.source_revision,
            1,
            entity.index,
            occurrence_path=entity.occurrence_path,
        )
    elif piece.arc.branch is not None:
        carrier = f"branch:{piece.arc.branch.branch_id}"
    else:
        carrier = f"source:{piece.arc.owner[0]}:edge:{piece.arc.source_edge}"
    return carrier, first, last


def _endpoint_on_carrier(
    endpoint: RootEndpoint | None, carrier: BRepCurve | BRepPCurve
) -> RootEndpoint | None:
    """Bind an authored native scalar to the retained source lift.

    Coincident chart transport changes the UV carrier, not its raw parameter.
    The new carrier retains the source source expression and chart map.
    Isolated roots remain with their existing certified support definitions.
    """
    if not isinstance(endpoint, NativePeriodEndpoint) or (
        canonical_fingerprint(carrier_payload(endpoint.carrier))
        == canonical_fingerprint(carrier_payload(carrier))
    ):
        return endpoint
    return NativePeriodEndpoint(
        carrier,
        endpoint.patch,
        endpoint.axis,
        rational=endpoint.rational,
        turns=endpoint.turns,
    )


class _BoundVertex(StrictModule):
    point: np.ndarray
    binding: BRepVertexRoot | None


def _certify_boundary_vertex(
    state: Arrangement,
    arc: Arc,
    cut: Cut,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> _BoundVertex:
    binding = _vertex_binding(state, cut.vertex, preparations=preparations, budget=budget)
    if binding is None:
        return _BoundVertex(cut.point, binding)
    point, _, certified = binding.evaluate()
    box = binding.point_enclosure(preparations=preparations, budget=budget)
    radius = np.nextafter(
        np.maximum(np.abs(box[0] - point), np.abs(box[1] - point)), np.inf
    )
    squared = interval_multiply((radius, radius), (radius, radius))
    squared_lower = np.asarray(squared[0], dtype=np.float64)
    squared_upper = np.asarray(squared[1], dtype=np.float64)
    total = (np.asarray(0.0, np.float64), np.asarray(0.0, np.float64))
    for axis in range(3):
        total = interval_add(total, (squared_lower[axis], squared_upper[axis]))
    total_upper = np.asarray(total[1], dtype=np.float64)
    bound = np.nextafter(np.sqrt(total_upper), np.inf)
    if (
        not certified
        or not np.isfinite(bound)
        or bound > state.policy.construction_tolerance
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.vertex must have a certified positional bound within construction_tolerance; got uncertified, nonfinite, or excessive error",
            (arc.token, cut.vertex),
        )
    return _BoundVertex(point, binding)


class _BoundaryTopology(StrictModule):
    vertices: list[np.ndarray] = eqx.field(default_factory=list)
    vertex_roots: list[BRepVertexRoot | None] = eqx.field(default_factory=list)
    vertex_ids: dict[str, int] = eqx.field(default_factory=dict)
    curves: list[BRepCurve] = eqx.field(default_factory=list)
    edge_curves: list[int] = eqx.field(default_factory=list)
    edge_vertices: list[tuple[int, int]] = eqx.field(default_factory=list)
    edge_ranges: list[tuple[float, float]] = eqx.field(default_factory=list)
    edge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = eqx.field(
        default_factory=list
    )
    edge_ids: dict[tuple[str, str, str], int] = eqx.field(default_factory=dict)
    pcurves: list[BRepPCurve] = eqx.field(default_factory=list)
    coedge_edges: list[int] = eqx.field(default_factory=list)
    coedge_senses: list[int] = eqx.field(default_factory=list)
    coedge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = eqx.field(
        default_factory=list
    )
    face_loops: list[tuple[tuple[int, ...], ...]] = eqx.field(default_factory=list)
    scalar_boxes: dict[tuple[int, int], list[tuple[float, float]]] = eqx.field(
        default_factory=dict
    )

    def append_piece(
        self,
        state: Arrangement,
        piece: Piece,
        overlay_sources: Mapping[str, CoincidentFaceArc] | None,
        *,
        preparations: TrimRequestPreparations | None,
        budget: CoordinateEnclosureBudget,
    ) -> int:
        arc = piece.arc
        endpoints = (
            (piece.first, piece.last) if arc.scale > 0.0 else (piece.last, piece.first)
        )
        ids = []
        for cut in endpoints:
            if cut.vertex not in self.vertex_ids:
                prepared = _certify_boundary_vertex(
                    state, arc, cut, preparations=preparations, budget=budget
                )
                self.vertex_ids[cut.vertex] = len(self.vertices)
                self.vertices.append(prepared.point)
                self.vertex_roots.append(prepared.binding)
            ids.append(self.vertex_ids[cut.vertex])
        source = None if overlay_sources is None else overlay_sources.get(arc.token)
        key = _edge_key(piece, source)
        if key not in self.edge_ids:
            edge = len(self.edge_vertices)
            self.edge_ids[key] = edge
            if arc.curve is None:
                self.edge_curves.append(-1)
            else:
                self.edge_curves.append(len(self.curves))
                self.curves.append(arc.curve)
            self.edge_vertices.append((ids[0], ids[1]))
            self.edge_ranges.append(
                (
                    arc.offset + arc.scale * endpoints[0].parameter,
                    arc.offset + arc.scale * endpoints[1].parameter,
                )
            )
            self.edge_roots.append(
                (
                    endpoints[0].endpoint
                    if arc.curve is None
                    else _endpoint_on_carrier(endpoints[0].endpoint, arc.curve),
                    endpoints[1].endpoint
                    if arc.curve is None
                    else _endpoint_on_carrier(endpoints[1].endpoint, arc.curve),
                )
            )
        edge = self.edge_ids[key]
        for endpoint, cut in enumerate(endpoints):
            if cut.endpoint is not None:
                self.scalar_boxes.setdefault((edge, endpoint), []).append(
                    cut.endpoint.parameter_enclosure(
                        preparations=preparations, budget=budget
                    )
                )
        pcurve = arc.pcurve
        if isinstance(pcurve, IntersectionPCurve):
            first, last = self.edge_ranges[edge]
            pcurve = IntersectionPCurve(
                pcurve.curve, pcurve.side, first=float(first), last=float(last)
            )
        self.pcurves.append(pcurve)
        coedge = len(self.coedge_edges)
        self.coedge_edges.append(edge)
        self.coedge_senses.append(piece.sense * (1 if arc.scale > 0.0 else -1))
        self.coedge_roots.append(
            (
                _endpoint_on_carrier(endpoints[0].endpoint, pcurve),
                _endpoint_on_carrier(endpoints[1].endpoint, pcurve),
            )
        )
        return coedge

    def geometry(
        self,
        state: Arrangement,
        *,
        preparations: TrimRequestPreparations | None,
        budget: CoordinateEnclosureBudget,
    ) -> BRepGeometry:
        for (edge, endpoint), boxes in self.scalar_boxes.items():
            lower, upper = max(box[0] for box in boxes), min(box[1] for box in boxes)
            if lower > upper:
                raise BRepBooleanFailure(
                    "BRepBoolean.edge.endpoint_enclosures must have a common scalar interval; got disjoint intervals"
                )
            values = list(self.edge_ranges[edge])
            values[endpoint] = 0.5 * (lower + upper)
            self.edge_ranges[edge] = (values[0], values[1])
        if len(self.coedge_edges) > state.policy.sewing.maximum_coedges:
            raise BRepBooleanFailure(
                "BRepBoolean.boundary.coedges must fit maximum_coedges; got coedge allocation budget exhausted"
            )
        return BRepGeometry(
            vertex_points=np.asarray(self.vertices, np.float64).reshape((-1, 3)),
            curves=tuple(self.curves),
            edge_curves=tuple(self.edge_curves),
            edge_ranges=np.asarray(self.edge_ranges, np.float64).reshape((-1, 2)),
            edge_vertices=tuple(self.edge_vertices),
            pcurves=tuple(self.pcurves),
            coedge_edges=tuple(self.coedge_edges),
            coedge_senses=tuple(self.coedge_senses),
            face_loops=tuple(self.face_loops),
            shell_faces=(),
            shell_orientations=(),
            solid_shells=(),
            vertex_roots=tuple(self.vertex_roots),
            edge_endpoint_roots=tuple(self.edge_roots),
            coedge_endpoint_roots=tuple(self.coedge_roots),
            budget=budget,
            preparations=preparations,
        )


@checked
def build_boundary(
    state: Arrangement,
    fragments: tuple[FaceFragment, ...],
    *,
    overlay_sources: Mapping[str, CoincidentFaceArc] | None = None,
    vertex_bindings: Mapping[str, BRepVertexRoot | None] | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> BuiltBoundary:
    if vertex_bindings is not None:
        state.vertex_bindings.update(vertex_bindings)
    topology = _BoundaryTopology()
    for fragment in fragments:
        loops = []
        for loop in fragment.loops:
            coedges = []
            for piece in loop:
                coedges.append(
                    topology.append_piece(
                        state,
                        piece,
                        overlay_sources,
                        preparations=preparations,
                        budget=budget,
                    )
                )
            loops.append(tuple(coedges))
        topology.face_loops.append(tuple(loops))
    geometry = topology.geometry(state, preparations=preparations, budget=budget)
    return BuiltBoundary(
        geometry,
        tuple(
            state.models[fragment.owner[0]].patches[fragment.owner[1]]
            for fragment in fragments
        ),
        np.asarray(
            [
                np.asarray(state.models[fragment.owner[0]].parameter_bounds)[
                    fragment.owner[1]
                ]
                for fragment in fragments
            ],
            np.float64,
        ).reshape((-1, 2, 2)),
        np.asarray([fragment.orientation for fragment in fragments], np.float64),
        tuple(fragment.domain for fragment in fragments),
        tuple(fragment.owner for fragment in fragments),
        tuple(
            state.models[fragment.owner[0]].physical_tags[fragment.owner[1]]
            for fragment in fragments
        ),
    )


def _with_shells(
    geometry: BRepGeometry,
    shells: tuple[tuple[int, ...], ...],
    shell_orientations: tuple[tuple[int, ...], ...],
    solids: tuple[tuple[int, ...], ...],
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> BRepGeometry:
    return BRepGeometry(
        vertex_points=geometry.vertex_points,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=shells,
        shell_orientations=shell_orientations,
        solid_shells=solids,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        preparations=preparations,
        budget=budget,
    )


def _shell_query_model(
    state: Arrangement,
    built: BuiltBoundary,
    faces: tuple[int, ...],
    outward: np.ndarray,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> BRepModel:
    orientation = np.asarray(built.orientation)
    geometry = _with_shells(
        built.geometry,
        (faces,),
        (tuple(int(outward[face] * orientation[face]) for face in faces),),
        ((0,),),
        preparations=preparations,
        budget=budget,
    )
    digest = canonical_fingerprint(
        {"kind": "boolean-shell-query", "geometry": geometry.geometry_id}
    )
    report = BRepImportReport(
        source_id=f"native-boolean-shell:{digest}",
        source_digest=digest,
        source_format="native",
        coordinate_contract=state.models[0].coordinate_contract,
        import_policy_id=digest,
        num_solids=1,
        num_faces=len(built.patches),
        num_edges=len(geometry.edge_curves),
        num_vertices=geometry.vertex_points.shape[0],
        num_triangles=0,
        linear_deflection=state.policy.tessellation.linear_deflection,
        angular_deflection=state.policy.tessellation.angular_deflection,
        trim_samples_per_edge=state.policy.tessellation.trim_samples_per_edge,
        converted_surface_count=0,
    )
    return BRepModel(
        patches=built.patches,
        parameter_bounds=built.bounds,
        orientation=built.orientation,
        trim_domains=built.domains,
        topology=geometry.topology(),
        coordinate_contract=state.models[0].coordinate_contract,
        mesh_vertices=np.empty((0, 3), np.float64),
        mesh_faces=np.empty((0, 3), np.int32),
        triangle_face_ids=np.empty((0,), np.int32),
        triangle_parameters=np.empty((0, 3, 2), np.float64),
        physical_tags=built.tags,
        report=report,
        geometry=geometry,
        tessellation_deviation_bounds=np.empty((0,), np.float64),
        tessellation_normal_bounds=np.empty((0,), np.float64),
        mesh_vertex_source_dimensions=np.empty((0,), np.int32),
        mesh_vertex_source_indices=np.empty((0,), np.int64),
        mesh_vertex_parameters=np.empty((0, 2), np.float64),
        mesh_chart_restriction_vertices=np.empty((0,), np.int64),
        mesh_chart_restriction_edges=np.empty((0, 2), np.int64),
        mesh_chart_restriction_endpoint_parameters=np.empty((0, 2, 2), np.float64),
        mesh_chart_restriction_parameters=np.empty((0, 2), np.int64),
        triangle_occurrence_ids=np.empty((0,), np.int32),
        vertex_occurrence_ids=np.empty((0,), np.int32),
        preparations=preparations,
        budget=budget,
    )


def _face_seed(
    domain: TrimDomain,
    bounds: np.ndarray,
    maximum_cells: int,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    count = 1
    while count * count <= maximum_cells:
        fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
        points = np.asarray([(x, y) for x in fractions for y in fractions], np.float64)
        points = bounds[0] + points * (bounds[1] - bounds[0])
        membership = domain.classify(points, preparations=preparations, budget=budget)
        candidates = np.flatnonzero(membership.inside & membership.resolved)
        if candidates.size:
            return points[candidates[0]]
        count *= 2
    raise BRepBooleanFailure(
        "BRepBoolean.trim.interior_seed must fit its search limit; got interior seed budget exhausted"
    )


def _oriented_solids(
    state: Arrangement,
    built: BuiltBoundary,
    faces: tuple[int, ...],
    outward: np.ndarray,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Closed shells of one material set and their outer/cavity solid nesting.

    ``outward`` is each face's material-outward sign relative to its patch
    normal. Returned solids index the returned shells, outer shell first.
    """
    shells_bank = shells(built.geometry, faces, outward)
    signs = tuple(
        _shell_volume_sign(
            built.patches,
            built.bounds,
            built.domains,
            outward,
            shell,
            state.policy.maximum_cells,
            preparations=preparations,
            budget=budget,
        )
        for shell in shells_bank
    )
    outer = [index for index, sign in enumerate(signs) if sign > 0]
    if not outer:
        raise BRepBooleanFailure(
            "BRepBoolean.shells must contain a positively oriented material shell; got none"
        )
    if len(outer) == len(shells_bank):
        return shells_bank, tuple((index,) for index in outer)
    queries = {
        index: prepare_brep_query(
            _shell_query_model(
                state,
                built,
                shells_bank[index],
                outward,
                preparations=preparations,
                budget=budget,
            ),
            preparations=preparations,
            budget=budget,
        )
        for index in outer
    }
    parent_contains: dict[tuple[int, int], bool] = {}
    for child in outer:
        face = shells_bank[child][0]
        uv = _face_seed(
            built.domains[face],
            built.bounds[face],
            state.policy.maximum_cells,
            preparations=preparations,
            budget=budget,
        )
        point = np.asarray(built.patches[face].evaluate(jnp.asarray(uv, jnp.float64)))[
            None
        ]
        for parent, query in queries.items():
            if child == parent:
                continue
            membership = query.contains(
                point, preparations=preparations, coordinate_budget=budget
            )
            if membership.unresolved[0]:
                raise BRepBooleanFailure(
                    "BRepBoolean.shell.containment must be resolved; got an unresolved positive-shell query"
                )
            parent_contains[parent, child] = bool(membership.inside[0])
    children: dict[int, list[int]] = {index: [] for index in outer}
    for index, sign in enumerate(signs):
        if sign > 0:
            continue
        face = shells_bank[index][0]
        uv = _face_seed(
            built.domains[face],
            built.bounds[face],
            state.policy.maximum_cells,
            preparations=preparations,
            budget=budget,
        )
        point = np.asarray(built.patches[face].evaluate(jnp.asarray(uv, jnp.float64)))[
            None
        ]
        candidates = []
        for parent, query in queries.items():
            membership = query.contains(
                point, preparations=preparations, coordinate_budget=budget
            )
            if membership.unresolved[0]:
                raise BRepBooleanFailure(
                    "BRepBoolean.cavity.containment must be resolved; got an unresolved nesting query"
                )
            if membership.inside[0]:
                candidates.append(parent)
        immediate = [
            child
            for child in candidates
            if all(
                parent == child or parent_contains[parent, child] for parent in candidates
            )
        ]
        if len(immediate) != 1:
            raise BRepBooleanFailure(
                "BRepBoolean.cavity.owner must identify one immediate containing shell; got another owner count"
            )
        children[immediate[0]].append(index)
    return shells_bank, tuple((index, *children[index]) for index in outer)


@checked
def reconstruct_solids(
    state: Arrangement,
    built: BuiltBoundary,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> BRepGeometry:
    if not built.patches:
        return built.geometry
    faces = tuple(range(len(built.patches)))
    shells, solids = _oriented_solids(
        state,
        built,
        faces,
        np.asarray(built.orientation),
        preparations=preparations,
        budget=budget,
    )
    return _with_shells(
        built.geometry,
        shells,
        tuple(tuple(1 for _ in shell) for shell in shells),
        solids,
        preparations=preparations,
        budget=budget,
    )


@checked
def reconstruct_region_solids(
    state: Arrangement,
    built: BuiltBoundary,
    sides: tuple[tuple[str | None, str | None], ...],
    regions: tuple[str, ...],
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[BRepGeometry, tuple[str, ...]]:
    """Shared-face solids of every owned region, in declared region order.

    ``sides`` holds each built face's negative/positive material owner. A face
    bounds both owners: outward along its patch normal for the negative-side
    owner and against it for the positive-side owner.
    """
    orientation = np.asarray(built.orientation)
    shells: list[tuple[int, ...]] = []
    shell_orientations: list[tuple[int, ...]] = []
    solids: list[tuple[int, ...]] = []
    owners: list[str] = []
    for region in regions:
        faces = tuple(face for face, owners_ in enumerate(sides) if region in owners_)
        if not faces:
            continue
        outward = orientation.copy()
        for face in faces:
            outward[face] = 1.0 if sides[face][0] == region else -1.0
        region_shells, region_solids = _oriented_solids(
            state, built, faces, outward, preparations=preparations, budget=budget
        )
        offset = len(shells)
        shells.extend(region_shells)
        shell_orientations.extend(
            tuple(int(outward[face] * orientation[face]) for face in shell)
            for shell in region_shells
        )
        solids.extend(tuple(offset + shell for shell in solid) for solid in region_solids)
        owners.extend(region for _ in region_solids)
    return _with_shells(
        built.geometry,
        tuple(shells),
        tuple(shell_orientations),
        tuple(solids),
        preparations=preparations,
        budget=budget,
    ), tuple(owners)


@checked
def publish_curved(
    state: Arrangement,
    built: BuiltBoundary,
    geometry: BRepGeometry,
    certificate: str,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> BRepModel:
    return assemble_brep_model(
        geometry,
        built.patches,
        built.bounds,
        built.orientation,
        built.tags,
        coordinate_contract=state.models[0].coordinate_contract,
        source_id=f"native-boolean:{certificate}",
        source_format="native",
        source_digest=certificate,
        import_policy_id=certificate,
        tessellation=state.policy.tessellation,
        preparations=preparations,
        budget=budget,
    )


def _triple_vertex_certificate(
    state: Arrangement,
    root: TrimIntersectionRoot,
    owner: FaceOwner,
    other_a: FaceOwner,
    other_b: FaceOwner,
    edge: int,
    spatial: CurveSurfaceIntersectionRoot,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[TripleSurfaceIntersectionRoot, tuple[BRepCurveSurfaceLift, ...]]:

    def source_region(face_owner: FaceOwner) -> SurfaceRegion:
        return state.face_region(face_owner, preparations=preparations, budget=budget)

    regions = source_region(owner), source_region(other_a), source_region(other_b)
    image = joint_root_parameter_image(
        root, regions, preparations=preparations, budget=budget
    )
    if image is None:
        raise BRepBooleanFailure(
            "BRepBoolean.triple_root must retain all generating surface parameter images; got a missing image"
        )
    source = state.models[other_a[0]]
    geometry = source.geometry
    if geometry is None:
        raise BRepBooleanFailure(
            "BRepBoolean.triple_lift.geometry must be defined; got None"
        )
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
    t_lower, t_upper = (
        float(spatial.parameter_lower[0]),
        float(spatial.parameter_upper[0]),
    )
    lifts = []
    spatial_boxes = [np.stack((spatial.parameter_lower[1:], spatial.parameter_upper[1:]))]
    for _, face in (other_a, other_b):
        coedges = [
            coedge
            for loop in geometry.face_loops[face]
            for coedge in loop
            if geometry.coedge_edges[coedge] == edge
        ]
        if len(coedges) != 1:
            raise BRepBooleanFailure(
                "BRepBoolean.triple_lift.edge must have one coedge per incident face; got another coedge count"
            )
        pcurve = geometry.pcurves[coedges[0]]
        region = state.face_region(
            (other_a[0], face), preparations=preparations, budget=budget
        )
        lift = BRepCurveSurfaceLift(
            geometry.curves[geometry.edge_curves[edge]],
            pcurve,
            region,
            first,
            last,
            budget=budget,
            preparations=preparations,
        )
        lifts.append(lift)
        spatial_boxes.append(
            np.asarray(
                pcurve.bounding_box(
                    t_lower, t_upper, preparations=preparations, budget=budget
                )
            )
        )
    spatial_image = np.concatenate(spatial_boxes, axis=1)
    lower, upper = (
        np.minimum(image[0], spatial_image[0]),
        np.maximum(image[1], spatial_image[1]),
    )
    margin = (
        256.0
        * np.finfo(np.float64).eps
        * (1.0 + np.maximum(np.abs(lower), np.abs(upper)))
    )
    lower, upper = (
        np.nextafter(lower - margin, -np.inf),
        np.nextafter(upper + margin, np.inf),
    )
    proof_regions = (
        SurfaceRegion(
            regions[0].patch,
            np.stack((lower[0:2], upper[0:2])),
            budget=budget,
            preparations=preparations,
        ),
        SurfaceRegion(
            regions[1].patch,
            np.stack((lower[2:4], upper[2:4])),
            budget=budget,
            preparations=preparations,
        ),
        SurfaceRegion(
            regions[2].patch,
            np.stack((lower[4:6], upper[4:6])),
            budget=budget,
            preparations=preparations,
        ),
    )
    joint = TripleSurfaceIntersectionRoot(
        *proof_regions,
        parameter_lower=lower,
        parameter_upper=upper,
        budget=budget,
        preparations=preparations,
    )
    if not joint.certifies_root(root, preparations=preparations, budget=budget):
        raise BRepBooleanFailure(
            "BRepBoolean.triple_event must certify a shared unique source-system root; got failed certification"
        )
    return joint, tuple(lifts)


class _SpatialSourceArc(StrictModule):
    arc: Arc
    edge: int = eqx.field(static=True)


class _SpatialBranchArc(StrictModule):
    arc: Arc
    branch: IntersectionCurve


class _FixedBranchCut(StrictModule):
    cut: Cut
    root: IntersectionCurvePointRoot


class _SpatialArcPair(StrictModule):
    branch: IntersectionCurve
    side: IntersectionCurveSide = eqx.field(static=True)
    whole_first: float = eqx.field(static=True)
    whole_last: float = eqx.field(static=True)
    span_first: float = eqx.field(static=True)
    span_last: float = eqx.field(static=True)


def _append_spatial_root_cuts(
    state: Arrangement,
    owner: FaceOwner,
    source_arc: Arc,
    source_pcurve: AbstractCurve,
    branch_arc: Arc,
    pair: _SpatialArcPair,
    spatial: CurveSurfaceIntersectionRoot,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> None:
    branch, side = pair.branch, pair.side
    span_first, span_last = pair.span_first, pair.span_last
    source_box = source_pcurve_image(
        source_pcurve,
        float(spatial.parameter_lower[0]),
        float(spatial.parameter_upper[0]),
        preparations=preparations,
        budget=budget,
    )
    opposite_box = np.stack((spatial.parameter_lower[1:], spatial.parameter_upper[1:]))
    image = np.concatenate(
        (source_box, opposite_box) if side == "first" else (opposite_box, source_box),
        axis=1,
    )
    charts = [
        chart
        for chart in range(int(span_first), int(span_last))
        if np.all(image[0] >= branch.box_lower[chart])
        and np.all(image[1] <= branch.box_upper[chart])
    ]
    if not charts:
        return
    chart = charts[0]
    branch_endpoint = BranchRootEndpoint(
        spatial,
        branch,
        chart,
        source_pcurve=source_pcurve,
        source_side=side,
        source_first=pair.whole_first,
        source_last=pair.whole_last,
        budget=budget,
        preparations=preparations,
    )
    lower, upper = branch_endpoint.parameter_enclosure(
        preparations=preparations, budget=budget
    )
    if upper < span_first or lower > span_last:
        return
    parameter = min(span_last, max(span_first, branch_endpoint.parameter))
    if not lower <= parameter <= upper:
        raise BRepBooleanFailure(
            "BRepBoolean.source_root.parameter must lie inside its enclosure; got an outside representative"
        )
    source_endpoint = TrimRootEndpoint(
        spatial,
        "first",
        source_pcurve=source_pcurve,
        source_surface=state.face_region(
            owner, preparations=preparations, budget=budget
        ),
    )
    source_local = source_endpoint.affine(
        1.0 / source_arc.scale, -source_arc.offset / source_arc.scale
    )
    source_lower, source_upper = source_local.parameter_enclosure(
        preparations=preparations, budget=budget
    )
    if not 0.0 < source_lower < source_upper < 1.0:
        raise BRepBooleanFailure(
            "BRepBoolean.source_root.interval must lie strictly inside the source edge; got a corner root without a vertex alias",
            (source_arc.token,),
        )
    vertex = f"spatial-root:{spatial.root_id}"
    binding = BRepVertexRoot(spatial, budget=budget, preparations=preparations)
    point = binding.evaluate()[0]
    state.vertex_spatial[vertex] = spatial
    state.vertex_aliases.setdefault(vertex, [])
    state.vertex_bindings[vertex] = binding
    node = f"spatial:{spatial.root_id}:face:{owner}"
    alias_node = _bind_fixed_branch_alias(
        state,
        branch_arc,
        branch_endpoint,
        vertex,
        binding,
        point,
        preparations=preparations,
        budget=budget,
    )
    if alias_node is not None:
        node = alias_node
    state.cuts[source_arc.token].append(
        Cut(
            source_local.parameter,
            source_lower,
            source_upper,
            node,
            vertex,
            source_endpoint,
            binding,
            point,
        )
    )
    state.cuts[branch_arc.token].append(
        Cut(parameter, lower, upper, node, vertex, branch_endpoint, binding, point)
    )


@checked
def boundary_spatial_events(
    state: Arrangement,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> None:
    for owner, arcs in sorted(state.arcs.items()):
        source_arcs = tuple(
            _SpatialSourceArc(arc=arc, edge=arc.source_edge)
            for arc in arcs
            if arc.source_edge is not None and arc.curve is not None
        )
        branch_arcs = tuple(
            _SpatialBranchArc(arc=arc, branch=arc.branch)
            for arc in arcs
            if arc.branch is not None
        )
        for source in source_arcs:
            source_arc = source.arc
            if not isinstance(source_arc.pcurve, AbstractCurve):
                raise BRepBooleanFailure(
                    "BRepBoolean.source_root.pcurve must be a native curve; got another pcurve family",
                    (source_arc.token,),
                )
            source_pcurve = source_arc.pcurve
            for candidate in branch_arcs:
                branch_arc, branch = candidate.arc, candidate.branch
                if (
                    branch.start_kind != "boundary"
                    and branch.end_kind != "boundary"
                    and not any(
                        periodic_transition(branch, index)
                        for index in range(branch.num_charts)
                    )
                ):
                    continue
                first_owner, second_owner = state.branch_owners[branch_arc.token]
                side: IntersectionCurveSide = (
                    "first" if owner == first_owner else "second"
                )
                opposite = second_owner if side == "first" else first_owner
                roots = _edge_spatial_roots(
                    state,
                    owner,
                    source.edge,
                    opposite,
                    preparations=preparations,
                    budget=budget,
                )
                geometry = state.models[owner[0]].geometry
                if geometry is None:
                    raise RuntimeError(
                        "BRepBoolean.spatial_arc.geometry must be defined; got None"
                    )
                whole_first, whole_last = (
                    float(value)
                    for value in np.asarray(geometry.edge_ranges)[source.edge]
                )
                span_first, span_last = branch_arc.trim.parameter_interval
                pair = _SpatialArcPair(
                    branch, side, whole_first, whole_last, span_first, span_last
                )
                for spatial in roots:
                    _append_spatial_root_cuts(
                        state,
                        owner,
                        source_arc,
                        source_pcurve,
                        branch_arc,
                        pair,
                        spatial,
                        preparations=preparations,
                        budget=budget,
                    )
                first_token, last_token = sorted((source_arc.token, branch_arc.token))
                state.spatial_pairs.add((first_token, last_token))


class _SourceNodeGauge(StrictModule):
    shifts: np.ndarray


def _branch_node_gauge(branch: IntersectionCurve, node: int, chart: int) -> np.ndarray:
    shifts = branch.node_period_shifts[node].astype(np.int64).copy()
    if node == chart and chart:
        for axis, period in enumerate(
            (*branch.first.patch.periods, *branch.second.patch.periods)
        ):
            if period is not None:
                shifts[axis] += int(
                    round(branch.transition_shifts[chart - 1, axis] / period)
                )
    return shifts


def _certify_source_node_gauge(
    endpoint: BranchRootEndpoint,
    matched_reference: int,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> _SourceNodeGauge:
    branch = endpoint.curve
    # Closed-loop nodes can occur in distinct source parameter gauges.
    source_image = endpoint.source_image(
        maximum_steps=16,
        include_periodic_shifts=False,
        preparations=preparations,
        budget=budget,
    )
    source_gauge = None
    for node in (endpoint.chart, endpoint.chart + 1):
        reference = int(branch.node_references[node])
        if reference != matched_reference:
            continue
        shifts = _branch_node_gauge(branch, node, endpoint.chart)
        difference = endpoint.source_gauges() - shifts
        image = source_image
        if np.any(difference):
            lower_shift, upper_shift = branch.shift_parameter_bounds(difference)
            image = np.stack(
                interval_add(
                    (source_image[0], source_image[1]), (lower_shift, upper_shift)
                )
            )
        axis = int(branch.node_axes[reference])
        if image[0, axis] == branch.node_values[reference] == image[1, axis]:
            source_gauge = shifts
            break
    if source_gauge is None:
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias must retain a source-node gauge; got no matching gauge"
        )
    return _SourceNodeGauge(source_gauge)


def _transport_fixed_branch_cut(
    other: Arc,
    cut: Cut,
    endpoint: BranchRootEndpoint,
    source_gauge: _SourceNodeGauge,
    vertex: str,
    binding: BRepVertexRoot,
    point: np.ndarray,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> Cut:
    if (
        other.branch is None
        or not isinstance(cut.endpoint, TrimRootEndpoint)
        or not isinstance(cut.endpoint.root, IntersectionCurvePointRoot)
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias.incident_endpoints must be fixed branch point roots; got an unrelated endpoint"
        )
    first, last = other.trim.parameter_interval
    chart = min(int(last) - 1, max(int(first), int(np.floor(cut.parameter))))
    target_gauge = _branch_node_gauge(
        endpoint.curve, int(cut.endpoint.root.parameter), chart
    )
    shifts = (
        np.asarray(endpoint.periodic_shifts, dtype=np.int64)
        + target_gauge
        - source_gauge.shifts
    )
    transported = BranchRootEndpoint(
        endpoint.root,
        endpoint.curve,
        chart,
        source_pcurve=endpoint.source_pcurve,
        source_side=endpoint.source_side,
        source_first=endpoint.source_first,
        source_last=endpoint.source_last,
        periodic_shifts=(int(shifts[0]), int(shifts[1]), int(shifts[2]), int(shifts[3])),
        budget=budget,
        preparations=preparations,
    )
    if not transported.certifies_point_root(
        cut.endpoint.root, preparations=preparations, budget=budget
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias.incident_endpoint must certify the same fixed node; got failed point-root certification"
        )
    lower, upper = transported.parameter_enclosure(
        preparations=preparations, budget=budget
    )
    if not lower <= cut.parameter <= upper:
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias.cut.parameter must lie in the transported enclosure; got an outside representative"
        )
    return replace(
        cut,
        lower=lower,
        upper=upper,
        vertex=vertex,
        endpoint=transported,
        vertex_root=binding,
        point=point,
    )


def _bind_fixed_branch_alias(
    state: Arrangement,
    arc: Arc,
    endpoint: BranchRootEndpoint,
    vertex: str,
    binding: BRepVertexRoot,
    point: np.ndarray,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> str | None:

    first_owner, _ = state.branch_owners[arc.token]
    columns = slice(0, 2) if arc.owner == first_owner else slice(2, 4)
    patch = state.models[arc.owner[0]].patches[arc.owner[1]]
    for cut in state.cuts[arc.token]:
        existing = cut.endpoint
        if (
            isinstance(existing, BranchRootEndpoint)
            and existing.root_id == endpoint.root_id
            and np.array_equal(
                existing.source_gauges()[columns], endpoint.source_gauges()[columns]
            )
            and isinstance(arc.pcurve, (AbstractCurve, IntersectionPCurve))
            and binding.same_uv_endpoint(
                patch,
                arc.pcurve,
                endpoint,
                arc.pcurve,
                existing,
                preparations=preparations,
                budget=budget,
            )
        ):
            return cut.node

    matches: list[_FixedBranchCut] = []
    for cut in state.cuts[arc.token]:
        match cut.endpoint:
            case TrimRootEndpoint(root=IntersectionCurvePointRoot() as root):
                if endpoint.certifies_point_root(
                    root, preparations=preparations, budget=budget
                ):
                    matches.append(_FixedBranchCut(cut=cut, root=root))
            case _:
                continue
    if not matches:
        return None
    branch = endpoint.curve
    lower, upper = endpoint.parameter_enclosure(preparations=preparations, budget=budget)
    matched_cut = next(
        (match for match in matches if lower <= match.cut.parameter <= upper), None
    )
    if matched_cut is None:
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias must have a matched cut inside its source chart; got no matching cut"
        )
    matched_reference = int(branch.node_references[int(matched_cut.root.parameter)])

    source_gauge = _certify_source_node_gauge(
        endpoint, matched_reference, preparations=preparations, budget=budget
    )
    old_vertices = {match.cut.vertex for match in matches}
    changes = []
    for arcs in state.arcs.values():
        for other in arcs:
            for index, cut in enumerate(state.cuts[other.token]):
                if cut.vertex not in old_vertices:
                    continue
                transported_cut = _transport_fixed_branch_cut(
                    other,
                    cut,
                    endpoint,
                    source_gauge,
                    vertex,
                    binding,
                    point,
                    preparations=preparations,
                    budget=budget,
                )
                changes.append((other.token, index, transported_cut))
    for token, index, cut in changes:
        state.cuts[token][index] = cut
    return matched_cut.cut.node
