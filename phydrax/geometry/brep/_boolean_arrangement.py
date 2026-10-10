#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-oriented exact-curve boundary arrangements for native CAD Booleans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from .._atlas import AbstractTrimCurve, CurveTrimLoop, PolygonTrimLoop, TrimDomain
from .._interval_enclosure import (
    interval_add,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
)
from ._boolean import (
    _MaterialSelection,
    BRepBooleanFailure,
    BRepBooleanPolicy,
)
from ._boolean_coincidence import CoincidentFaceArc, SourceEdgeFacePairCoverage
from ._constructors import assemble_brep_model
from ._intersection import (
    BranchRootEndpoint,
    CurveSurfaceIntersectionRoot,
    intersect_curve_region,
    IntersectionCurvePointRoot,
    NativePeriodEndpoint,
    ParametricIntersectionPolicy,
    RootEndpoint,
    SurfaceIntersectionResult,
    TrimIntersectionRoot,
    TrimRootEndpoint,
    TripleSurfaceIntersectionRoot,
)
from ._intersection_curve import (
    coefficient_enclosures,
    curve_pieces,
    CurveRange,
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionCurveSide,
    IntersectionPCurve,
    surface_pieces,
    SurfaceEvaluator,
    SurfaceRegion,
)
from ._model import (
    _carrier_payload,
    BRepCurve,
    BRepGeometry,
    BRepImportReport,
    BRepModel,
    BRepPCurve,
)
from ._patches import AbstractSurfacePatch
from ._projection_contracts import brep_entity_id
from ._query import prepare_brep_query, PreparedBRepQuery
from ._root_bindings import BRepCurveSurfaceLift, BRepRootSupport, BRepVertexRoot
from ._sewing import _shells


type _Face = tuple[int, int]


@dataclass(frozen=True, slots=True)
class _Arc:
    token: str
    owner: _Face
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


@dataclass(frozen=True, slots=True)
class _Cut:
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
class _Piece:
    arc: _Arc
    first: _Cut
    last: _Cut
    sense: int


@dataclass
class _Arrangement:
    models: tuple[BRepModel, ...]
    selection: _MaterialSelection
    policy: BRepBooleanPolicy
    queries: tuple[PreparedBRepQuery, ...]
    arcs: dict[_Face, list[_Arc]] = field(default_factory=dict)
    domains: dict[_Face, TrimDomain] = field(default_factory=dict)
    cuts: dict[str, list[_Cut]] = field(default_factory=dict)
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
    branches: dict[str, tuple[IntersectionCurve, _Face, _Face]] = field(
        default_factory=dict
    )
    branch_owners: dict[str, tuple[_Face, _Face]] = field(default_factory=dict)
    # Branch sides proved (exact lift + certified chart uniqueness) to be a
    # present source edge of that face (its own, or a coincident partner's):
    # branch id -> {face: (source model, source edge)}.
    branch_source_edges: dict[str, dict[_Face, tuple[int, int]]] = field(
        default_factory=dict
    )
    source_pair_coverage: dict[tuple[_Face, _Face], SourceEdgeFacePairCoverage] = field(
        default_factory=dict
    )
    source_adjacencies: dict[tuple[_Face, _Face], tuple[int, ...]] = field(
        default_factory=dict
    )
    roots: dict[str, TrimIntersectionRoot] = field(default_factory=dict)
    vertex_bindings: dict[str, BRepVertexRoot | None] = field(default_factory=dict)
    vertex_branch_points: dict[str, tuple[IntersectionCurve, float]] = field(
        default_factory=dict
    )
    spatial_pairs: set[tuple[str, str]] = field(default_factory=set)
    boxes_used: int = 0

    def consume(self, boxes: int) -> None:
        self.boxes_used += boxes
        if self.boxes_used > self.policy.intersection.maximum_boxes:
            raise BRepBooleanFailure("coupled intersection work budget exhausted")

    @property
    def remaining_boxes(self) -> int:
        return max(0, self.policy.intersection.maximum_boxes - self.boxes_used)


def _remaining_intersection_policy(state: _Arrangement) -> ParametricIntersectionPolicy:
    remaining = state.remaining_boxes
    if remaining <= 0:
        raise BRepBooleanFailure("curved Boolean exhausted its native root work budget")
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


def _overlap(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.all(first[0] <= second[1]) and np.all(second[0] <= first[1]))


def _source_arcs(state: _Arrangement) -> None:
    for model_index, model in enumerate(state.models):
        geometry = model.geometry
        if geometry is None:
            raise BRepBooleanFailure(
                "source has no authoritative curve/face topology", (model.model_id,)
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
                    arc = _Arc(
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
                        _Cut(
                            0.0,
                            0.0,
                            0.0,
                            first_node,
                            first_vertex,
                            first_root,
                            geometry.vertex_roots[start],
                            points[start],
                        ),
                        _Cut(
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


def _check_surface_result(
    state: _Arrangement,
    result: SurfaceIntersectionResult,
    first: _Face,
    second: _Face,
) -> None:
    state.consume(result.work.boxes_processed + result.work.march_steps)
    sources = tuple(
        _face_id(state.models[index], face) for index, face in (first, second)
    )
    if not result.complete:
        reasons = ",".join(sorted({region.reason for region in result.unresolved}))
        raise BRepBooleanFailure(
            f"surface intersection discovery unresolved ({reasons})", sources
        )
    if result.coincident:
        raise BRepBooleanFailure(
            "coincident curved face arrangement requires a certified parameter correspondence",
            sources,
        )


def _periodic_transition(
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


def _branch_arc(
    state: _Arrangement,
    curve: IntersectionCurve,
    owner: _Face,
    side: IntersectionCurveSide,
    first: int,
    last: int,
    *,
    owners: tuple[_Face, _Face],
) -> None:
    pcurve = curve.p_curve(side)
    local = IntersectionPCurve(curve, side, first=float(first), last=float(last))
    first_vertex = f"branch:{curve.branch_id}:node:{first}"
    final = 0 if curve.closed and last == curve.num_charts else last
    last_vertex = f"branch:{curve.branch_id}:node:{final}"
    first_seam = _periodic_transition(curve, (first - 1) % curve.num_charts, side)
    last_seam = _periodic_transition(curve, (last - 1) % curve.num_charts, side)
    first_sheet = "after" if first_seam else "common"
    last_sheet = "before" if last_seam else "common"
    first_node = f"{first_vertex}:face:{owner}:sheet:{first_sheet}"
    last_node = f"{last_vertex}:face:{owner}:sheet:{last_sheet}"
    token = f"branch:{curve.branch_id}:owners:{owners}:face:{owner}:span:{first}:{last}"
    state.branch_owners[token] = owners
    arc = _Arc(
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
        raise RuntimeError("Fixed branch factories must return fixed parameter roots.")
    if final == 0 and first == 0 and first_binding.root_id != last_binding.root_id:
        raise BRepBooleanFailure(
            "closed branch endpoint identities disagree", (curve.branch_id,)
        )
    first_endpoint = TrimRootEndpoint(first_binding.primary, "first")
    last_endpoint = TrimRootEndpoint(last_binding.primary, "first")
    state.vertex_bindings.setdefault(first_vertex, first_binding)
    state.vertex_bindings.setdefault(last_vertex, last_binding)
    state.cuts[token] = [
        _Cut(
            float(first),
            float(first),
            float(first),
            first_node,
            first_vertex,
            first_endpoint,
            first_binding,
            first_binding.evaluate()[0],
        ),
        _Cut(
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


def _arc_area_bounds(
    curve: AbstractTrimCurve, first: float, last: float
) -> tuple[float, float]:
    box = curve.enclosure(first, last)
    derivative = curve.derivative_bounds(first, last)
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


def _loop_sign(loop: CurveTrimLoop, maximum_cells: int) -> int:
    cells = [
        (loop.curves[int(index)], float(first), float(last))
        for index, first, last in zip(
            loop.arc_curves, loop.arc_first, loop.arc_last, strict=True
        )
    ]
    if len(cells) > maximum_cells:
        raise BRepBooleanFailure("oriented trim cover exceeds its area-sign budget")
    while True:
        intervals = [_arc_area_bounds(curve, first, last) for curve, first, last in cells]
        lower, upper = np.asarray(0.0, np.float64), np.asarray(0.0, np.float64)
        for a, b in intervals:
            lower, upper = interval_add((lower, upper), (np.asarray(a), np.asarray(b)))
        if lower > 0.0:
            return 1
        if upper < 0.0:
            return -1
        if len(cells) * 2 > maximum_cells:
            raise BRepBooleanFailure("oriented trim area sign budget exhausted")
        following = []
        for curve, first, last in cells:
            middle = 0.5 * (first + last)
            if not first < middle < last:
                raise BRepBooleanFailure("oriented trim area sign is unresolvable")
            following.extend(((curve, first, middle), (curve, middle, last)))
        cells = following


def _trim_box_status(domain: TrimDomain, box: np.ndarray) -> int:
    for loop in domain.loops:
        if isinstance(loop, CurveTrimLoop):
            lower = np.concatenate((loop.arc_lower, loop.junction_lower))
            upper = np.concatenate((loop.arc_upper, loop.junction_upper))
        elif isinstance(loop, PolygonTrimLoop):
            points = np.asarray(loop.vertices)
            following = np.roll(points, -1, axis=0)
            lower, upper = np.minimum(points, following), np.maximum(points, following)
        else:
            raise TypeError("Unknown exact trim-loop representation.")
        if np.any(np.all(lower <= box[1], axis=1) & np.all(upper >= box[0], axis=1)):
            return 0
    decision = domain.classify((0.5 * (box[0] + box[1]))[None])
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
) -> int:
    functions = []
    cell_faces = []
    cells = []
    for face in faces:
        for piece in surface_pieces(SurfaceRegion(patches[face], bounds[face])):
            if len(functions) >= maximum_cells:
                raise BRepBooleanFailure(
                    "oriented shell initial cover exceeds its cell budget"
                )
            evaluator = piece.evaluator

            def integrand(
                uv: jax.Array,
                evaluator: SurfaceEvaluator = evaluator,
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

    def cell_contribution(index: int, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        face = cell_faces[index]
        status = _trim_box_status(domains[face], box)
        if status < 0:
            return np.asarray(0.0, np.float64), np.asarray(0.0, np.float64)
        lower, upper = functions[index].evaluate(box[0][None], box[1][None])
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
        return interval_multiply(integrand_bounds, area)

    bounded_cells = [(index, box, cell_contribution(index, box)) for index, box in cells]
    while True:
        total = np.asarray(0.0, np.float64), np.asarray(0.0, np.float64)
        widths = []
        for _, _, contribution in bounded_cells:
            total = interval_add(total, contribution)
            widths.append(float(contribution[1] - contribution[0]))
        if total[0] > 0.0:
            return 1
        if total[1] < 0.0:
            return -1
        if len(bounded_cells) >= maximum_cells:
            raise BRepBooleanFailure("oriented shell volume sign budget exhausted")
        worst = int(np.argmax(widths))
        index, box, _ = bounded_cells.pop(worst)
        axis = int(np.argmax(box[1] - box[0]))
        middle = 0.5 * (box[0, axis] + box[1, axis])
        if not box[0, axis] < middle < box[1, axis]:
            raise BRepBooleanFailure("oriented shell volume sign cannot be resolved")
        first, second = box.copy(), box.copy()
        first[1, axis], second[0, axis] = middle, middle
        bounded_cells.append((index, first, cell_contribution(index, first)))
        bounded_cells.append((index, second, cell_contribution(index, second)))


def _edge_spatial_roots(
    state: _Arrangement,
    source: _Face,
    edge: int,
    opposite: _Face,
) -> tuple[CurveSurfaceIntersectionRoot, ...]:
    key = source[0], edge, opposite[0], opposite[1]
    if key in state.spatial_roots:
        return state.spatial_roots[key]
    model = state.models[source[0]]
    geometry = model.geometry
    if geometry is None:
        raise BRepBooleanFailure("source edge has no exact geometry")
    curve_index = geometry.edge_curves[edge]
    if curve_index < 0:
        raise BRepBooleanFailure(
            "a degenerate source edge meets a regular intersection branch"
        )
    curve = geometry.curves[curve_index]
    if isinstance(curve, IntersectionCurve):
        raise BRepBooleanFailure(
            "source intersection-edge discovery requires coupled spatial range isolation",
            (f"{model.source_revision}:edge:{edge}",),
        )
    first, last = np.asarray(geometry.edge_ranges)[edge]
    other = state.models[opposite[0]]
    region = SurfaceRegion(
        other.patches[opposite[1]], np.asarray(other.parameter_bounds)[opposite[1]]
    )
    result = intersect_curve_region(
        CurveRange(curve, float(first), float(last)),
        region,
        policy=_remaining_intersection_policy(state),
    )
    state.consume(result.work.boxes_processed)
    if not result.complete or result.coincident:
        raise BRepBooleanFailure(
            "source-edge spatial roots are unresolved or coincident",
            (f"{model.source_revision}:edge:{edge}", _face_id(other, opposite[1])),
        )
    points = sorted(
        (point for point in result.points if point.kind == "transversal"),
        key=lambda point: float(point.parameters[0]),
    )
    roots = tuple(
        CurveSurfaceIntersectionRoot(
            curve,
            region,
            parameter_lower=point.parameter_lower,
            parameter_upper=point.parameter_upper,
        )
        for point in points
    )
    if any(
        roots[index].parameter_upper[0] >= roots[index + 1].parameter_lower[0]
        for index in range(len(roots) - 1)
    ):
        raise BRepBooleanFailure("source-edge spatial root order is unresolved")
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
    state: _Arrangement,
    root: TrimIntersectionRoot,
    source_arc: _Arc,
    branch_arc: _Arc,
    source_operand: int,
    branch_operand: int,
    spatial: CurveSurfaceIntersectionRoot,
) -> None:
    from ._intersection import _CurveSurfaceSystem, _krawczyk, _prepare

    source_endpoint = TrimRootEndpoint(root, "first" if source_operand == 0 else "second")
    t_lower, t_upper = source_endpoint.parameter_enclosure()
    branch = branch_arc.branch
    if branch is None:
        raise BRepBooleanFailure("spatial alias requires an authoritative branch")
    tau_lower = max(0.0, float(root.parameter_lower[branch_operand]))
    tau_upper = min(float(branch.num_charts), float(root.parameter_upper[branch_operand]))
    coupled = branch.parameter_enclosures(tau_lower, tau_upper)
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
        pieces = curve_pieces(CurveRange(spatial.curve, float(lower[0]), float(upper[0])))
        if len(pieces) != 1:
            raise BRepBooleanFailure("spatial alias crosses an unresolved source knot")
        evaluator = pieces[0].evaluator
    pieces = [
        piece
        for piece in surface_pieces(spatial.surface)
        if np.all(lower[1:] >= piece.lower) and np.all(upper[1:] <= piece.upper)
    ]
    if len(pieces) != 1:
        raise BRepBooleanFailure(
            "spatial alias crosses an unresolved source chart boundary"
        )
    prepared = _prepare(_CurveSurfaceSystem(evaluator, pieces[0].evaluator), 3, 1)
    state.consume(1)
    inclusion = _krawczyk(prepared, lower[None], upper[None])
    if not bool(inclusion.certified[0]):
        raise BRepBooleanFailure(
            "adjacent UV roots have no unique shared spatial root certificate",
            (source_arc.token, branch_arc.token),
        )


def _event_vertex(
    state: _Arrangement,
    root: TrimIntersectionRoot,
    first: _Arc,
    second: _Arc,
) -> str:
    source_arc = first if first.source_edge is not None else second
    branch_arc = second if source_arc is first else first
    if source_arc.source_edge is not None and branch_arc.branch is not None:
        source_operand, branch_operand = (0, 1) if source_arc is first else (1, 0)
        owner_a, owner_b = state.branch_owners[branch_arc.token]
        opposite = owner_b if source_arc.owner == owner_a else owner_a
        roots = _edge_spatial_roots(
            state, source_arc.owner, source_arc.source_edge, opposite
        )
        endpoint = TrimRootEndpoint(root, "first" if source_operand == 0 else "second")
        lower, upper = endpoint.parameter_enclosure()
        matches = [
            spatial
            for spatial in roots
            if lower <= spatial.parameter_upper[0] and upper >= spatial.parameter_lower[0]
        ]
        if len(matches) != 1:
            raise BRepBooleanFailure(
                "trim and spatial root correspondence is not unique",
                (source_arc.token, branch_arc.token),
            )
        spatial = matches[0]
        _prove_edge_root_alias(
            state, root, source_arc, branch_arc, source_operand, branch_operand, spatial
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
            raise BRepBooleanFailure("triple junction source ownership is inconsistent")
        model = state.models[other_a[0]]
        common = set(model.topology.face_edges[other_a[1]]) & set(
            model.topology.face_edges[other_b[1]]
        )
        candidates = [
            (edge, spatial)
            for edge in sorted(common)
            for spatial in _edge_spatial_roots(state, other_a, edge, first.owner)
        ]
        point_box = np.asarray(
            state.models[first.owner[0]]
            .patches[first.owner[1]]
            .bounding_box(root.point_enclosure())
        )
        matches = [
            (edge, spatial)
            for edge, spatial in candidates
            if _overlap(point_box, spatial.point_enclosure())
        ]
        if len(matches) != 1:
            raise BRepBooleanFailure("triple point has no unique source-edge witness")
        edge, spatial = matches[0]
        joint, lifts = _triple_vertex_certificate(
            state,
            root,
            first.owner,
            other_a,
            other_b,
            edge,
            spatial,
        )
        vertex = f"spatial-root:{spatial.root_id}"
        state.vertex_spatial[vertex] = spatial
        state.vertex_joint[vertex] = joint
        state.vertex_lifts[vertex] = lifts
    else:
        raise BRepBooleanFailure("a source boundary self-intersection was not declared")
    patch = state.models[first.owner[0]].patches[first.owner[1]]
    state.vertex_aliases.setdefault(vertex, []).append((patch, root))
    return vertex


def _add_root_cut(
    state: _Arrangement,
    arc: _Arc,
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
        _Cut(parameter, lower, upper, node, vertex, endpoint, None, point)
    )


def _piece_trim(piece: _Piece) -> CurveTrimSegment:
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
    )


def _ordered_cuts(state: _Arrangement, arc: _Arc) -> tuple[_Cut, ...]:
    cuts = sorted(state.cuts[arc.token], key=lambda cut: cut.parameter)
    result = []
    for cut in cuts:
        if result and result[-1].upper >= cut.lower:
            previous = result[-1]
            if previous.vertex != cut.vertex:
                raise BRepBooleanFailure(
                    "root ordering or boundary endpoint identity is unresolved",
                    (arc.token,),
                )
            if previous.endpoint is None and cut.endpoint is not None:
                result[-1] = cut
            continue
        result.append(cut)
    return tuple(result)


def _propagate_branch_cuts(state: _Arrangement) -> None:
    branch_cuts: dict[str, dict[str, tuple[_Cut, _Arc]]] = {}
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
                    _Cut(
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
                raise BRepBooleanFailure("coupled branch cut capacity exhausted")


def _piece_nodes(piece: _Piece) -> tuple[str, str]:
    return (
        (piece.first.node, piece.last.node)
        if piece.sense > 0
        else (piece.last.node, piece.first.node)
    )


@dataclass(frozen=True, slots=True)
class _FaceFragment:
    owner: _Face
    loops: tuple[tuple[_Piece, ...], ...]
    domain: TrimDomain
    orientation: int


@dataclass(frozen=True, slots=True)
class _BuiltBoundary:
    geometry: BRepGeometry
    patches: tuple[AbstractSurfacePatch, ...]
    bounds: np.ndarray
    orientation: np.ndarray
    domains: tuple[TrimDomain, ...]
    sources: tuple[_Face, ...]
    tags: tuple[str, ...]


def _vertex_binding(state: _Arrangement, vertex: str) -> BRepVertexRoot | None:
    if vertex in state.vertex_bindings:
        return state.vertex_bindings[vertex]
    if vertex in state.vertex_spatial:
        aliases = tuple(
            BRepRootSupport(patch, root) for patch, root in state.vertex_aliases[vertex]
        )
        binding = BRepVertexRoot(
            state.vertex_spatial[vertex],
            aliases=aliases,
            joint_root=state.vertex_joint.get(vertex),
            source_edge_lifts=state.vertex_lifts.get(vertex, ()),
        )
    elif vertex in state.vertex_branch_points:
        curve, parameter = state.vertex_branch_points[vertex]
        binding = BRepVertexRoot.from_curve_point(curve, parameter)
    else:
        model_text, index_text = vertex.removeprefix("source:").split(":vertex:", 1)
        geometry = state.models[int(model_text)].geometry
        if geometry is None:
            raise BRepBooleanFailure("source vertex lost its authoritative geometry")
        binding = geometry.vertex_roots[int(index_text)]
    state.vertex_bindings[vertex] = binding
    return binding


def _edge_key(
    piece: _Piece, source: CoincidentFaceArc | None = None
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
    """Bind an authored native scalar to the retained exact source lift.

    Coincident chart transport changes the UV carrier, not its raw parameter.
    The new carrier retains the original source expression and chart map.
    Isolated roots remain with their existing certified support definitions.
    """
    if not isinstance(endpoint, NativePeriodEndpoint) or (
        canonical_fingerprint(_carrier_payload(endpoint.carrier))
        == canonical_fingerprint(_carrier_payload(carrier))
    ):
        return endpoint
    return NativePeriodEndpoint(
        carrier,
        endpoint.patch,
        endpoint.axis,
        rational=endpoint.rational,
        turns=endpoint.turns,
    )


def _build_boundary(
    state: _Arrangement,
    fragments: tuple[_FaceFragment, ...],
    *,
    overlay_sources: Mapping[str, CoincidentFaceArc] | None = None,
    vertex_bindings: Mapping[str, BRepVertexRoot | None] | None = None,
) -> _BuiltBoundary:
    if vertex_bindings is not None:
        state.vertex_bindings.update(vertex_bindings)
    vertices = []
    vertex_roots = []
    vertex_ids: dict[str, int] = {}
    curves = []
    edge_curves = []
    edge_vertices = []
    edge_ranges = []
    edge_roots = []
    edge_ids: dict[tuple[str, str, str], int] = {}
    pcurves = []
    coedge_edges = []
    coedge_senses = []
    coedge_roots = []
    face_loops = []
    scalar_boxes: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for fragment in fragments:
        loops = []
        for loop in fragment.loops:
            coedges = []
            for piece in loop:
                arc = piece.arc
                endpoints = (
                    (piece.first, piece.last)
                    if arc.scale > 0.0
                    else (piece.last, piece.first)
                )
                ids = []
                for cut in endpoints:
                    if cut.vertex not in vertex_ids:
                        binding = _vertex_binding(state, cut.vertex)
                        if binding is None:
                            point = cut.point
                        else:
                            point, _, certified = binding.evaluate()
                            box = binding.point_enclosure()
                            radius = np.nextafter(
                                np.maximum(
                                    np.abs(box[0] - point), np.abs(box[1] - point)
                                ),
                                np.inf,
                            )
                            squared = interval_multiply(
                                (radius, radius), (radius, radius)
                            )
                            total = (
                                np.asarray(0.0, np.float64),
                                np.asarray(0.0, np.float64),
                            )
                            for axis in range(3):
                                total = interval_add(
                                    total, (squared[0][axis], squared[1][axis])
                                )
                            bound = np.nextafter(np.sqrt(total[1]), np.inf)
                            if (
                                not certified
                                or not np.isfinite(bound)
                                or bound > state.policy.construction_tolerance
                            ):
                                raise BRepBooleanFailure(
                                    "constructed vertex misses its declared positional bound",
                                    (arc.token, cut.vertex),
                                )
                        vertex_ids[cut.vertex] = len(vertices)
                        vertices.append(point)
                        vertex_roots.append(binding)
                    ids.append(vertex_ids[cut.vertex])
                source = (
                    None if overlay_sources is None else overlay_sources.get(arc.token)
                )
                key = _edge_key(piece, source)
                if key not in edge_ids:
                    edge = len(edge_vertices)
                    edge_ids[key] = edge
                    if arc.curve is None:
                        edge_curves.append(-1)
                    else:
                        edge_curves.append(len(curves))
                        curves.append(arc.curve)
                    edge_vertices.append((ids[0], ids[1]))
                    edge_ranges.append(
                        tuple(arc.offset + arc.scale * cut.parameter for cut in endpoints)
                    )
                    edge_roots.append(
                        (
                            endpoints[0].endpoint
                            if arc.curve is None
                            else _endpoint_on_carrier(endpoints[0].endpoint, arc.curve),
                            endpoints[1].endpoint
                            if arc.curve is None
                            else _endpoint_on_carrier(endpoints[1].endpoint, arc.curve),
                        )
                    )
                edge = edge_ids[key]
                for endpoint, cut in enumerate(endpoints):
                    if cut.endpoint is not None:
                        scalar_boxes.setdefault((edge, endpoint), []).append(
                            cut.endpoint.parameter_enclosure()
                        )
                pcurve = arc.pcurve
                if isinstance(pcurve, IntersectionPCurve):
                    first, last = edge_ranges[edge]
                    pcurve = IntersectionPCurve(
                        pcurve.curve, pcurve.side, first=float(first), last=float(last)
                    )
                pcurves.append(pcurve)
                coedges.append(len(coedge_edges))
                coedge_edges.append(edge)
                coedge_senses.append(piece.sense * (1 if arc.scale > 0.0 else -1))
                coedge_roots.append(
                    (
                        _endpoint_on_carrier(endpoints[0].endpoint, pcurve),
                        _endpoint_on_carrier(endpoints[1].endpoint, pcurve),
                    )
                )
            loops.append(tuple(coedges))
        face_loops.append(tuple(loops))
    for (edge, endpoint), boxes in scalar_boxes.items():
        lower, upper = max(box[0] for box in boxes), min(box[1] for box in boxes)
        if lower > upper:
            raise BRepBooleanFailure(
                "shared coedge scalar source enclosures are inconsistent"
            )
        values = list(edge_ranges[edge])
        values[endpoint] = 0.5 * (lower + upper)
        edge_ranges[edge] = tuple(values)
    if len(coedge_edges) > state.policy.sewing.maximum_coedges:
        raise BRepBooleanFailure("curved boundary coedge allocation budget exhausted")
    geometry = BRepGeometry(
        vertex_points=np.asarray(vertices, np.float64).reshape((-1, 3)),
        curves=tuple(curves),
        edge_curves=tuple(edge_curves),
        edge_ranges=np.asarray(edge_ranges, np.float64).reshape((-1, 2)),
        edge_vertices=tuple(edge_vertices),
        pcurves=tuple(pcurves),
        coedge_edges=tuple(coedge_edges),
        coedge_senses=tuple(coedge_senses),
        face_loops=tuple(face_loops),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
        vertex_roots=tuple(vertex_roots),
        edge_endpoint_roots=tuple(edge_roots),
        coedge_endpoint_roots=tuple(coedge_roots),
    )
    return _BuiltBoundary(
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
    )


def _shell_query_model(
    state: _Arrangement,
    built: _BuiltBoundary,
    faces: tuple[int, ...],
    outward: np.ndarray,
) -> BRepModel:
    orientation = np.asarray(built.orientation)
    geometry = _with_shells(
        built.geometry,
        (faces,),
        (tuple(int(outward[face] * orientation[face]) for face in faces),),
        ((0,),),
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
    )


def _face_seed(domain: TrimDomain, bounds: np.ndarray, maximum_cells: int) -> np.ndarray:
    count = 1
    while count * count <= maximum_cells:
        fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
        points = np.asarray([(x, y) for x in fractions for y in fractions], np.float64)
        points = bounds[0] + points * (bounds[1] - bounds[0])
        membership = domain.classify(points)
        candidates = np.flatnonzero(membership.inside & membership.resolved)
        if candidates.size:
            return points[candidates[0]]
        count *= 2
    raise BRepBooleanFailure("trim fragment interior seed budget exhausted")


def _oriented_solids(
    state: _Arrangement,
    built: _BuiltBoundary,
    faces: tuple[int, ...],
    outward: np.ndarray,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Closed shells of one material set and their outer/cavity solid nesting.

    ``outward`` is each face's material-outward sign relative to its patch
    normal. Returned solids index the returned shells, outer shell first.
    """
    shells = _shells(built.geometry, faces, outward)
    signs = tuple(
        _shell_volume_sign(
            built.patches,
            built.bounds,
            built.domains,
            outward,
            shell,
            state.policy.maximum_cells,
        )
        for shell in shells
    )
    outer = [index for index, sign in enumerate(signs) if sign > 0]
    if not outer:
        raise BRepBooleanFailure("no positively oriented material shell survives")
    if len(outer) == len(shells):
        return shells, tuple((index,) for index in outer)
    queries = {
        index: prepare_brep_query(
            _shell_query_model(state, built, shells[index], outward)
        )
        for index in outer
    }
    parent_contains: dict[tuple[int, int], bool] = {}
    for child in outer:
        face = shells[child][0]
        uv = _face_seed(
            built.domains[face], built.bounds[face], state.policy.maximum_cells
        )
        point = np.asarray(built.patches[face].evaluate(jnp.asarray(uv, jnp.float64)))[
            None
        ]
        for parent, query in queries.items():
            if child == parent:
                continue
            membership = query.contains(point)
            if membership.unresolved[0]:
                raise BRepBooleanFailure("positive shell containment order is unresolved")
            parent_contains[parent, child] = bool(membership.inside[0])
    children: dict[int, list[int]] = {index: [] for index in outer}
    for index, sign in enumerate(signs):
        if sign > 0:
            continue
        face = shells[index][0]
        uv = _face_seed(
            built.domains[face], built.bounds[face], state.policy.maximum_cells
        )
        point = np.asarray(built.patches[face].evaluate(jnp.asarray(uv, jnp.float64)))[
            None
        ]
        candidates = []
        for parent, query in queries.items():
            membership = query.contains(point)
            if membership.unresolved[0]:
                raise BRepBooleanFailure("cavity shell nesting is unresolved")
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
                "cavity shell has no unique immediate material owner"
            )
        children[immediate[0]].append(index)
    return shells, tuple((index, *children[index]) for index in outer)


def _reconstruct_solids(state: _Arrangement, built: _BuiltBoundary) -> BRepGeometry:
    if not built.patches:
        return built.geometry
    faces = tuple(range(len(built.patches)))
    shells, solids = _oriented_solids(state, built, faces, np.asarray(built.orientation))
    return _with_shells(
        built.geometry,
        shells,
        tuple(tuple(1 for _ in shell) for shell in shells),
        solids,
    )


def _reconstruct_region_solids(
    state: _Arrangement,
    built: _BuiltBoundary,
    sides: tuple[tuple[str | None, str | None], ...],
    regions: tuple[str, ...],
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
        region_shells, region_solids = _oriented_solids(state, built, faces, outward)
        offset = len(shells)
        shells.extend(region_shells)
        shell_orientations.extend(
            tuple(int(outward[face] * orientation[face]) for face in shell)
            for shell in region_shells
        )
        solids.extend(tuple(offset + shell for shell in solid) for solid in region_solids)
        owners.extend(region for _ in region_solids)
    return _with_shells(
        built.geometry, tuple(shells), tuple(shell_orientations), tuple(solids)
    ), tuple(owners)


def _publish_curved(
    state: _Arrangement,
    built: _BuiltBoundary,
    geometry: BRepGeometry,
    certificate: str,
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
    )


def _triple_vertex_certificate(
    state: _Arrangement,
    root: TrimIntersectionRoot,
    owner: _Face,
    other_a: _Face,
    other_b: _Face,
    edge: int,
    spatial: CurveSurfaceIntersectionRoot,
) -> tuple[TripleSurfaceIntersectionRoot, tuple[BRepCurveSurfaceLift, ...]]:
    from ._intersection import _joint_root_parameter_image

    def source_region(face_owner: _Face) -> SurfaceRegion:
        index, face = face_owner
        return SurfaceRegion(
            state.models[index].patches[face],
            np.asarray(state.models[index].parameter_bounds)[face],
        )

    regions = source_region(owner), source_region(other_a), source_region(other_b)
    image = _joint_root_parameter_image(root, regions)
    if image is None:
        raise BRepBooleanFailure(
            "triple UV root does not retain its generating source surfaces"
        )
    source = state.models[other_a[0]]
    geometry = source.geometry
    if geometry is None:
        raise BRepBooleanFailure("source-edge triple lift lost its exact geometry")
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
            raise BRepBooleanFailure("triple source edge has no unique coedge lift")
        pcurve = geometry.pcurves[coedges[0]]
        patch = source.patches[face]
        region = SurfaceRegion(patch, np.asarray(source.parameter_bounds)[face])
        lift = BRepCurveSurfaceLift(
            geometry.curves[geometry.edge_curves[edge]], pcurve, region, first, last
        )
        lifts.append(lift)
        spatial_boxes.append(np.asarray(pcurve.bounding_box(t_lower, t_upper)))
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
        SurfaceRegion(regions[0].patch, np.stack((lower[0:2], upper[0:2]))),
        SurfaceRegion(regions[1].patch, np.stack((lower[2:4], upper[2:4]))),
        SurfaceRegion(regions[2].patch, np.stack((lower[4:6], upper[4:6]))),
    )
    joint = TripleSurfaceIntersectionRoot(
        *proof_regions, parameter_lower=lower, parameter_upper=upper
    )
    if not joint.certifies_root(root):
        raise BRepBooleanFailure(
            "triple UV event has no shared unique source-system certificate"
        )
    return joint, tuple(lifts)


def _boundary_spatial_events(state: _Arrangement) -> None:
    from ._intersection import _source_pcurve_image
    from ._patches import AbstractCurve

    for owner, arcs in sorted(state.arcs.items()):
        source_arcs = tuple(
            arc for arc in arcs if arc.source_edge is not None and arc.curve is not None
        )
        branch_arcs = tuple(arc for arc in arcs if arc.branch is not None)
        for source_arc in source_arcs:
            if not isinstance(source_arc.pcurve, AbstractCurve):
                raise BRepBooleanFailure(
                    "source-root boundary transport requires an owned native pcurve lift",
                    (source_arc.token,),
                )
            for branch_arc in branch_arcs:
                branch = branch_arc.branch
                if branch is None or source_arc.source_edge is None:
                    raise RuntimeError(
                        "Prepared branch/source arcs lost their owning identities."
                    )
                if (
                    branch.start_kind != "boundary"
                    and branch.end_kind != "boundary"
                    and not any(
                        _periodic_transition(branch, index)
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
                    state, owner, source_arc.source_edge, opposite
                )
                geometry = state.models[owner[0]].geometry
                if geometry is None:
                    raise RuntimeError("Prepared source edge lost its exact geometry.")
                whole_first, whole_last = (
                    float(value)
                    for value in np.asarray(geometry.edge_ranges)[source_arc.source_edge]
                )
                span_first, span_last = branch_arc.trim.parameter_interval
                for spatial in roots:
                    source_box = _source_pcurve_image(
                        source_arc.pcurve,
                        float(spatial.parameter_lower[0]),
                        float(spatial.parameter_upper[0]),
                    )
                    opposite_box = np.stack(
                        (spatial.parameter_lower[1:], spatial.parameter_upper[1:])
                    )
                    image = np.concatenate(
                        (source_box, opposite_box)
                        if side == "first"
                        else (opposite_box, source_box),
                        axis=1,
                    )
                    charts = [
                        chart
                        for chart in range(int(span_first), int(span_last))
                        if np.all(image[0] >= branch.box_lower[chart])
                        and np.all(image[1] <= branch.box_upper[chart])
                    ]
                    if not charts:
                        continue
                    chart = charts[0]
                    branch_endpoint = BranchRootEndpoint(
                        spatial,
                        branch,
                        chart,
                        source_pcurve=source_arc.pcurve,
                        source_side=side,
                        source_first=whole_first,
                        source_last=whole_last,
                    )
                    lower, upper = branch_endpoint.parameter_enclosure()
                    if upper < span_first or lower > span_last:
                        continue
                    parameter = min(span_last, max(span_first, branch_endpoint.parameter))
                    if not lower <= parameter <= upper:
                        raise BRepBooleanFailure(
                            "source-root chart endpoint has no legal bounded representative"
                        )
                    source_endpoint = TrimRootEndpoint(
                        spatial,
                        "first",
                        source_pcurve=source_arc.pcurve,
                        source_surface=SurfaceRegion(
                            state.models[owner[0]].patches[owner[1]],
                            np.asarray(state.models[owner[0]].parameter_bounds)[owner[1]],
                        ),
                    )
                    source_local = source_endpoint.affine(
                        1.0 / source_arc.scale, -source_arc.offset / source_arc.scale
                    )
                    source_lower, source_upper = source_local.parameter_enclosure()
                    if not 0.0 < source_lower < source_upper < 1.0:
                        raise BRepBooleanFailure(
                            "source corner root identity requires an exact vertex alias",
                            (source_arc.token,),
                        )
                    vertex = f"spatial-root:{spatial.root_id}"
                    binding = BRepVertexRoot(spatial)
                    point = binding.evaluate()[0]
                    state.vertex_spatial[vertex] = spatial
                    state.vertex_aliases.setdefault(vertex, [])
                    state.vertex_bindings[vertex] = binding
                    node = f"spatial:{spatial.root_id}:face:{owner}"
                    alias_node = _bind_fixed_branch_alias(
                        state, branch_arc, branch_endpoint, vertex, binding, point
                    )
                    if alias_node is not None:
                        node = alias_node
                    state.cuts[source_arc.token].append(
                        _Cut(
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
                        _Cut(
                            parameter,
                            lower,
                            upper,
                            node,
                            vertex,
                            branch_endpoint,
                            binding,
                            point,
                        )
                    )
                first_token, last_token = sorted((source_arc.token, branch_arc.token))
                state.spatial_pairs.add((first_token, last_token))


def _bind_fixed_branch_alias(
    state: _Arrangement,
    arc: _Arc,
    endpoint: BranchRootEndpoint,
    vertex: str,
    binding: BRepVertexRoot,
    point: np.ndarray,
) -> str | None:
    from dataclasses import replace

    from ._patches import AbstractCurve

    first_owner, _ = state.branch_owners[arc.token]
    columns = slice(0, 2) if arc.owner == first_owner else slice(2, 4)
    patch = state.models[arc.owner[0]].patches[arc.owner[1]]
    for cut in state.cuts[arc.token]:
        existing = cut.endpoint
        if (
            isinstance(existing, BranchRootEndpoint)
            and existing.root_id == endpoint.root_id
            and np.array_equal(
                existing._source_gauges()[columns], endpoint._source_gauges()[columns]
            )
            and isinstance(arc.pcurve, (AbstractCurve, IntersectionPCurve))
            and binding.same_uv_endpoint(
                patch, arc.pcurve, endpoint, arc.pcurve, existing
            )
        ):
            return cut.node

    matches = [
        cut
        for cut in state.cuts[arc.token]
        if isinstance(cut.endpoint, TrimRootEndpoint)
        and isinstance(cut.endpoint.root, IntersectionCurvePointRoot)
        and endpoint.certifies_point_root(cut.endpoint.root)
    ]
    if not matches:
        return None
    branch = endpoint.curve
    lower, upper = endpoint.parameter_enclosure()
    matched_cut = next((cut for cut in matches if lower <= cut.parameter <= upper), None)
    if matched_cut is None:
        raise BRepBooleanFailure(
            "fixed branch alias has no node in its certified source chart"
        )
    matched_endpoint = matched_cut.endpoint
    if not isinstance(matched_endpoint, TrimRootEndpoint) or not isinstance(
        matched_endpoint.root,
        IntersectionCurvePointRoot,
    ):
        raise BRepBooleanFailure("fixed branch alias lost its source-node definition")
    matched_reference = int(branch.node_references[int(matched_endpoint.root.parameter)])

    def node_gauge(node: int, chart: int) -> np.ndarray:
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

    # Identify the source node in its own certified chart before changing its
    # sheet. A closed-loop vertex can occur in distinct source parameter gauges.
    source_image = endpoint._source_image(maximum_steps=16, include_periodic_shifts=False)
    source_gauge = None
    for node in (endpoint.chart, endpoint.chart + 1):
        reference = int(branch.node_references[node])
        if reference != matched_reference:
            continue
        shifts = node_gauge(node, endpoint.chart)
        difference = endpoint._source_gauges() - shifts
        image = source_image
        if np.any(difference):
            lower_shift, upper_shift = branch._shift_bounds(difference)
            from .._interval_enclosure import interval_add

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
        raise BRepBooleanFailure("fixed branch alias lost its exact source-node gauge")
    old_vertices = {cut.vertex for cut in matches}
    changes = []
    for arcs in state.arcs.values():
        for other in arcs:
            for index, cut in enumerate(state.cuts[other.token]):
                if cut.vertex not in old_vertices:
                    continue
                if (
                    other.branch is None
                    or not isinstance(cut.endpoint, TrimRootEndpoint)
                    or not isinstance(
                        cut.endpoint.root,
                        IntersectionCurvePointRoot,
                    )
                ):
                    raise BRepBooleanFailure(
                        "fixed branch alias has an unrelated incident source endpoint"
                    )
                first, last = other.trim.parameter_interval
                chart = min(int(last) - 1, max(int(first), int(np.floor(cut.parameter))))
                target_gauge = node_gauge(int(cut.endpoint.root.parameter), chart)
                shifts = (
                    np.asarray(endpoint.periodic_shifts, dtype=np.int64)
                    + target_gauge
                    - source_gauge
                )
                transported = BranchRootEndpoint(
                    endpoint.root,
                    endpoint.curve,
                    chart,
                    source_pcurve=endpoint.source_pcurve,
                    source_side=endpoint.source_side,
                    source_first=endpoint.source_first,
                    source_last=endpoint.source_last,
                    periodic_shifts=(
                        int(shifts[0]),
                        int(shifts[1]),
                        int(shifts[2]),
                        int(shifts[3]),
                    ),
                )
                if not transported.certifies_point_root(cut.endpoint.root):
                    raise BRepBooleanFailure(
                        "incident branch endpoint has no exact common node proof"
                    )
                lower, upper = transported.parameter_enclosure()
                if not lower <= cut.parameter <= upper:
                    raise BRepBooleanFailure(
                        "fixed-node numerical representative leaves its source expression"
                    )
                changes.append(
                    (
                        other.token,
                        index,
                        replace(
                            cut,
                            lower=lower,
                            upper=upper,
                            vertex=vertex,
                            endpoint=transported,
                            vertex_root=binding,
                            point=point,
                        ),
                    )
                )
    for token, index, cut in changes:
        state.cuts[token][index] = cut
    return matched_cut.node
