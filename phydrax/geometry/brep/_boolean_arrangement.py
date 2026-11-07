#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Construct source-oriented curve boundary arrangements for CAD Booleans."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...typing import checked
from .._atlas import TrimDomain, TrimEnclosures
from .._interval_enclosure import (
    interval_add,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
)
from ._boolean_coincidence import CoincidentFaceArc, SourceEdgeFacePairCoverage
from ._boolean_core import (
    affine_root_endpoint,
    boolean_face_chart_box,
    BRepBooleanFailure,
    BRepBooleanPolicy,
    MaterialSelection,
)
from ._carrier_pieces import carrier_payload
from ._constructors import _Builder, assemble_brep_model
from ._intersection import (
    _same_parameter_source,
    BranchRootEndpoint,
    ConicParameterEndpoint,
    CurveIntersectionResult,
    CurveRange,
    CurveSurfaceIntersectionRoot,
    ExactLineCylinderTangency,
    intersect_curve_region,
    intersect_trim_curves,
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
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
    IntersectionSide,
    orthogonal_cylinder_family,
    SurfaceRegion,
)
from ._model import BRepEntityId, BRepGeometry, BRepModel
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
)
from ._projection import NativeBRepProjection, prepare_brep_projection
from ._projection_contracts import brep_entity_id, BRepProjectionStatus
from ._root_bindings import (
    _curve_surface_root_shares_trim_support,
    BRepBranchNodeRoots,
    BRepCurveSurfaceLift,
    BRepRootSupport,
    BRepVertexRoot,
)
from ._sewing import shells


type FaceOwner = tuple[int, int]
type SourceEntity = tuple[int, BRepEntityId]


def _source_root_endpoint(value: object, /) -> RootEndpoint | None:
    if value is None or isinstance(value, RootEndpoint):
        return value
    raise BRepBooleanFailure(
        "BRepBoolean.source_endpoint must retain a recognized exact root definition"
    )


def _source_vertex_root(value: object, /) -> BRepVertexRoot | None:
    if value is None or isinstance(value, BRepVertexRoot):
        return value
    raise BRepBooleanFailure(
        "BRepBoolean.source_vertex must retain a recognized exact root binding"
    )


@dataclass(frozen=True, slots=True)
class Arc:
    token: str
    owner: FaceOwner
    trim: AbstractCurve
    curve: AbstractCurve | None
    pcurve: AbstractCurve
    offset: float
    scale: float
    first_node: str
    last_node: str
    first_vertex: str
    last_vertex: str
    source_edge: int | None
    branch: IntersectionCurve | None
    parent_token: str | None = None
    # A source edge of another face lying on this face: ``(model, edge)``.
    lifted_edge: tuple[int, int] | None = None


def _arc_parameter_interval(arc: Arc, /) -> tuple[float, float]:
    interval = arc.trim.parameter_domain
    if (
        interval is None
        or not np.all(np.isfinite(interval))
        or interval[0] >= interval[1]
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.arc.trim must retain a finite increasing parameter domain",
            (arc.token,),
        )
    return interval


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
    """Accumulate source topology and immutable geometry correspondence evidence."""

    models: tuple[BRepModel, ...]
    selection: MaterialSelection
    policy: BRepBooleanPolicy
    projections: tuple[NativeBRepProjection, ...]
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
    # Enclosure trees of each source face trim, prepared once per operation.
    trim_enclosures: dict[FaceOwner, TrimEnclosures] = field(default_factory=dict)
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

    def face_chart(
        self,
        owner: FaceOwner,
        /,
    ) -> np.ndarray:
        """Source face box in the chart shared by every Boolean region of the face."""
        box = self.face_charts.get(owner)
        if box is None:
            box = boolean_face_chart_box(
                self.models[owner[0]],
                owner[1],
            )
            self.face_charts[owner] = box
        return box.copy()

    def face_region(
        self,
        owner: FaceOwner,
        /,
    ) -> SurfaceRegion:
        return SurfaceRegion(
            self.models[owner[0]].patches[owner[1]],
            self.face_chart(owner),
        )


def _face_id(model: BRepModel, face: int) -> str:
    return f"{model.source_revision}:face:{face}"


def overlap(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.all(first[0] <= second[1]) and np.all(second[0] <= first[1]))


class ArcPairIntersection(StrictModule):
    result: CurveIntersectionResult | None


def _shared_conic_endpoint_root(
    first: Arc,
    second: Arc,
    source_region: SurfaceRegion,
    /,
) -> bool:
    if not isinstance(first.trim, CurveTrimSegment) or not isinstance(
        second.trim, CurveTrimSegment
    ):
        return False

    def endpoint_vertices(arc: Arc, /) -> tuple[tuple[TrimRootEndpoint, str], ...]:
        trim = arc.trim
        assert isinstance(trim, CurveTrimSegment)
        first_vertex, last_vertex = (
            (arc.last_vertex, arc.first_vertex)
            if trim.reversed
            else (arc.first_vertex, arc.last_vertex)
        )
        return tuple(
            (endpoint, vertex)
            for endpoint, vertex in (
                (trim.first_root, first_vertex),
                (trim.last_root, last_vertex),
            )
            if isinstance(endpoint, TrimRootEndpoint)
        )

    for first_endpoint, first_vertex in endpoint_vertices(first):
        first_root = first_endpoint.root
        if (
            not isinstance(first_root, TrimIntersectionRoot)
            or first_root.endpoint_witness is None
        ):
            continue
        witness = first_root.endpoint_witness
        source_box = np.asarray(source_region.parameter_box, dtype=np.float64)
        witness_box = np.asarray(witness.uv_enclosure, dtype=np.float64)
        if (
            not _same_parameter_source(source_region.patch, witness.source_region.patch)
            or np.any(source_box[0] > witness_box[0])
            or np.any(source_box[1] < witness_box[1])
        ):
            continue
        if not witness.binds_curves(first_root.first, first_root.second):
            continue
        for second_endpoint, second_vertex in endpoint_vertices(second):
            second_root = second_endpoint.root
            if (
                first_vertex != second_vertex
                or not isinstance(second_root, TrimIntersectionRoot)
                or second_root.root_id != first_root.root_id
                or second_root.endpoint_witness is None
                or second_endpoint.operand == first_endpoint.operand
            ):
                continue
            endpoints = (first_endpoint, second_endpoint)
            arcs = (first, second)
            valid = True
            for endpoint, arc in zip(endpoints, arcs, strict=True):
                source_curve = (
                    first_root.first if endpoint.operand == "first" else first_root.second
                )
                trim = arc.trim
                assert isinstance(trim, CurveTrimSegment)
                endpoint_region_box = (
                    None
                    if endpoint.source_surface is None
                    else np.asarray(endpoint.source_surface.parameter_box)
                )
                endpoint_uv = np.asarray(witness.uv_enclosure)
                if (
                    carrier_payload(trim.curve) != carrier_payload(source_curve)
                    or endpoint.source_pcurve is None
                    or carrier_payload(endpoint.source_pcurve)
                    != carrier_payload(source_curve)
                    or endpoint.source_surface is None
                    or endpoint_region_box is None
                    or not _same_parameter_source(
                        endpoint.source_surface.patch,
                        witness.source_region.patch,
                    )
                    or np.any(endpoint_region_box[0] > endpoint_uv[0])
                    or np.any(endpoint_region_box[1] < endpoint_uv[1])
                ):
                    valid = False
                    break
            if valid:
                return True
    return False


@checked
def intersect_arc_pair(
    first: Arc,
    second: Arc,
    policy: ParametricIntersectionPolicy | Arrangement,
    *,
    intersection: Callable[[AbstractCurve, AbstractCurve], CurveIntersectionResult]
    | None = None,
) -> ArcPairIntersection:
    """Intersect overlapping arc enclosures with the declared geometric policy."""
    first_box = source_pcurve_image(first.trim, *_arc_parameter_interval(first))
    second_box = source_pcurve_image(second.trim, *_arc_parameter_interval(second))
    if not overlap(first_box, second_box):
        return ArcPairIntersection(None)
    if intersection is not None:
        return ArcPairIntersection(intersection(first.trim, second.trim))
    source_region = (
        policy.face_region(first.owner)
        if isinstance(policy, Arrangement) and first.owner == second.owner
        else None
    )
    if source_region is not None and _shared_conic_endpoint_root(
        first, second, source_region
    ):
        return ArcPairIntersection(None)
    match policy:
        case Arrangement():
            policy_ = policy.policy.intersection
        case ParametricIntersectionPolicy():
            policy_ = policy
        case _:
            raise TypeError(
                "intersect_arc_pair.policy must be an intersection policy or arrangement; "
                f"got {type(policy).__name__}"
            )
    source_patch = None if source_region is None else source_region.patch
    result = intersect_trim_curves(
        first.trim,
        second.trim,
        policy=policy_,
        source_patch=source_patch,
    )
    return ArcPairIntersection(result)


@checked
def source_arcs(
    state: Arrangement,
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
                    first_root_raw, last_root_raw = geometry.coedge_endpoint_roots[coedge]
                    first_root = _source_root_endpoint(first_root_raw)
                    last_root = _source_root_endpoint(last_root_raw)
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
                    first_vertex_root = _source_vertex_root(geometry.vertex_roots[start])
                    last_vertex_root = _source_vertex_root(geometry.vertex_roots[end])
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
                            first_vertex_root,
                            points[start],
                        ),
                        Cut(
                            1.0,
                            1.0,
                            1.0,
                            last_node,
                            last_vertex,
                            last_root,
                            last_vertex_root,
                            points[end],
                        ),
                    ]
            state.arcs[owner] = arcs
            domain = native_chart_trim_domain(model, face)
            if domain is None:
                raise BRepBooleanFailure(
                    "BRepBoolean.source_face.trim_domain must be present for every bounded source face",
                    (_face_id(model, face),),
                )
            state.domains[owner] = domain


@checked
def check_surface_result(
    state: Arrangement,
    result: SurfaceIntersectionResult,
    first: FaceOwner,
    second: FaceOwner,
) -> None:
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
    side: IntersectionSide | None = None,
) -> bool:
    if transition == curve.num_charts - 1 and not curve.closed:
        return False
    if transition < 0 or transition >= curve.num_charts:
        raise ValueError(
            "periodic_transition.transition must select an intersection chart."
        )
    node = transition + 1
    if node == curve.num_charts:
        # The closing node is the start root seen through the branch's
        # closure deck; express its incoming gauge in the start-node frame.
        incoming = (
            np.asarray(curve.node_period_shifts[node, 0], dtype=np.int64)
            + curve.closure_period_shifts()
        )
        outgoing = np.asarray(curve.node_period_shifts[0, 1], dtype=np.int64)
    else:
        incoming = np.asarray(curve.node_period_shifts[node, 0], dtype=np.int64)
        outgoing = np.asarray(curve.node_period_shifts[node, 1], dtype=np.int64)
    difference = incoming - outgoing
    columns = range(4) if side is None else (range(2) if side == "first" else range(2, 4))
    return any(difference[column] != 0 for column in columns)


@checked
def branch_arc(
    state: Arrangement,
    curve: IntersectionCurve,
    owner: FaceOwner,
    side: IntersectionSide,
    first: int,
    last: int,
    *,
    owners: tuple[FaceOwner, FaceOwner],
    source_vertices: Mapping[int, tuple[str, str]] | None = None,
) -> None:
    """Add one branch span to ``owner``.

    ``source_vertices`` maps an end node proved to be a source vertex to that
    vertex's face-graph node and global vertex, so the span ends on it.
    """
    vertices = {} if source_vertices is None else source_vertices
    local = IntersectionPCurve(curve, side, first=float(first), last=float(last))
    # A span between periodic transitions owns its source sheet; the whole
    # branch p-curve would evaluate a terminal node in the next span's sheet.
    pcurve = local
    first_vertex = f"branch:{curve.branch_id}:owners:{owners}:node:{first}"
    final = 0 if curve.closed and last == curve.num_charts else last
    last_vertex = f"branch:{curve.branch_id}:owners:{owners}:node:{final}"
    first_seam = periodic_transition(curve, (first - 1) % curve.num_charts, side)
    last_seam = periodic_transition(curve, (last - 1) % curve.num_charts, side)
    first_sheet = "after" if first_seam else "common"
    last_sheet = "before" if last_seam else "common"
    first_node = f"{first_vertex}:face:{owner}:sheet:{first_sheet}"
    last_node = f"{last_vertex}:face:{owner}:sheet:{last_sheet}"
    if first in vertices:
        first_node, first_vertex = vertices[first]
    if last in vertices:
        last_node, last_vertex = vertices[last]
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
    first_binding = BRepVertexRoot.from_curve_point(curve, float(first))
    last_binding = BRepVertexRoot.from_curve_point(curve, float(final))
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
    # A closed branch returns to its start root one branch period later; the
    # terminal trim scalar retains that exact deck translation.
    last_endpoint = TrimRootEndpoint(
        last_binding.primary,
        "first",
        affine_parameter_offset=float(last - final),
    )
    for node, vertex, parameter, binding in (
        (first, first_vertex, float(first), first_binding),
        (final, last_vertex, float(final), last_binding),
    ):
        if node not in vertices:
            state.vertex_branch_points[vertex] = curve, parameter
            state.vertex_bindings.setdefault(vertex, binding)
    state.cuts[token] = [
        Cut(
            float(first),
            float(first),
            float(first),
            first_node,
            first_vertex,
            # A source-vertex end is that exact vertex; the node parameter is
            # exact on the branch, so no root stands in for it.
            None if first in vertices else first_endpoint,
            None if first in vertices else first_binding,
            np.asarray(
                curve.evaluate(jnp.asarray(first, dtype=jnp.float64)), dtype=np.float64
            ),
        ),
        Cut(
            float(last),
            float(last),
            float(last),
            last_node,
            last_vertex,
            None if last in vertices else last_endpoint,
            None if last in vertices else last_binding,
            np.asarray(
                curve.evaluate(jnp.asarray(last, dtype=jnp.float64)), dtype=np.float64
            ),
        ),
    ]


def source_vertex_node(
    state: Arrangement, owner: FaceOwner, vertex: tuple[int, int], /
) -> tuple[str, str]:
    """The face-graph node and global vertex of a source vertex on ``owner``."""
    model_index, index = vertex
    name = f"source:{model_index}:vertex:{index}"
    geometry = state.models[owner[0]].geometry
    if owner[0] == model_index and geometry is not None:
        for loop_index, loop in enumerate(geometry.face_loops[owner[1]]):
            for ordinal, coedge in enumerate(loop):
                start, end = geometry.edge_vertices[geometry.coedge_edges[coedge]]
                if (start if geometry.coedge_senses[coedge] > 0 else end) == index:
                    return (
                        f"face:{model_index}:{owner[1]}:loop:{loop_index}:corner:{ordinal}",
                        name,
                    )
    return f"{name}:face:{owner}", name


def lift_source_edge_arc(
    state: Arrangement,
    owner: FaceOwner,
    edge: tuple[int, int],
    pcurve: AbstractCurve,
    /,
) -> bool:
    """Add a source edge lying on ``owner`` as one arc of ``owner``'s graph.

    ``pcurve`` is the edge's proved exact lift onto ``owner``. The arc ends on
    the edge's source vertices and publishes that same edge, so every face of
    the edge and ``owner`` share it. Edges with root-defined ends keep their
    branch instead. Returns whether the edge is (now) an arc of ``owner``.
    """
    model_index, index = edge
    token = f"lift:{model_index}:edge:{index}:face:{owner}"
    if token in state.cuts:
        return True
    geometry = state.models[model_index].geometry
    if geometry is None or geometry.edge_curves[index] < 0:
        return False
    start, end = geometry.edge_vertices[index]
    if (
        any(root is not None for root in geometry.edge_endpoint_roots[index])
        or geometry.vertex_roots[start] is not None
        or geometry.vertex_roots[end] is not None
    ):
        return False
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[index])
    first_node, first_vertex = source_vertex_node(state, owner, (model_index, start))
    last_node, last_vertex = source_vertex_node(state, owner, (model_index, end))
    state.arcs[owner].append(
        Arc(
            token,
            owner,
            CurveTrimSegment(pcurve, first, last),
            geometry.curves[geometry.edge_curves[index]],
            pcurve,
            first,
            last - first,
            first_node,
            last_node,
            first_vertex,
            last_vertex,
            None,
            None,
            lifted_edge=edge,
        )
    )
    points = np.asarray(geometry.vertex_points)
    state.cuts[token] = [
        Cut(0.0, 0.0, 0.0, first_node, first_vertex, None, None, points[start]),
        Cut(1.0, 1.0, 1.0, last_node, last_vertex, None, None, points[end]),
    ]
    return True


def arc_area_bounds(
    curve: AbstractCurve, first: float, last: float
) -> tuple[float, float]:
    box = source_pcurve_image(curve, first, last)
    try:
        derivative_lower, derivative_upper = curve.derivative_bounds(first, last)
    except NotImplementedError:
        derivative = prepare_interval_function(jax.jacfwd(curve.evaluate), 1)
        lower, upper = derivative.evaluate(np.asarray([[first]]), np.asarray([[last]]))
        derivative_lower = lower[0, :, 0]
        derivative_upper = upper[0, :, 0]
    derivative_bounds = np.stack((derivative_lower, derivative_upper))
    forward = interval_multiply(
        (box[0, 0], box[1, 0]), (derivative_bounds[0, 1], derivative_bounds[1, 1])
    )
    backward = interval_multiply(
        (box[0, 1], box[1, 1]), (derivative_bounds[0, 0], derivative_bounds[1, 0])
    )
    integrand = interval_subtract(forward, backward)
    width = np.asarray(0.5 * (last - first), dtype=np.float64)
    result = interval_multiply(
        integrand, (np.nextafter(width, -np.inf), np.nextafter(width, np.inf))
    )
    return float(result[0]), float(result[1])


def arc_area_bounds_batch(
    cells: Sequence[tuple[AbstractCurve, float, float]],
) -> list[tuple[float, float]]:
    return [arc_area_bounds(curve, first, last) for curve, first, last in cells]


def owner_enclosures(state: Arrangement, owner: FaceOwner, /) -> TrimEnclosures:
    """The operation's enclosure trees of a source face's trim domain."""
    enclosures = state.trim_enclosures.get(owner)
    if enclosures is None:
        enclosures = state.domains[owner].prepare_enclosures()
        state.trim_enclosures[owner] = enclosures
    return enclosures


@checked
def trim_box_status(
    domain: TrimDomain,
    box: np.ndarray,
    *,
    enclosures: TrimEnclosures | None = None,
) -> int:
    decision = domain.classify_boxes(
        np.asarray(box, dtype=np.float64)[None], enclosures=enclosures
    )
    if not decision.resolved[0] or decision.boundary[0]:
        return 0
    return 1 if decision.inside[0] else -1


def _spatial_root_vertex(
    source: FaceOwner,
    edge: int,
    opposite: FaceOwner,
    root: CurveSurfaceIntersectionRoot,
    /,
) -> str:
    """Qualify a geometric root by its authored edge and opposite face."""
    return f"spatial-root:{source[0]}:{edge}:{opposite[0]}:{opposite[1]}:{root.root_id}"


def _edge_spatial_roots(
    state: Arrangement,
    source: FaceOwner,
    edge: int,
    opposite: FaceOwner,
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
    region = state.face_region(opposite)
    result = intersect_curve_region(
        CurveRange(curve, float(first), float(last)),
        region,
        policy=state.policy.intersection,
    )
    if not result.complete or result.coincident:
        raise BRepBooleanFailure(
            "BRepBoolean.source_edge.roots must be isolated and resolved; got unresolved or coincident roots",
            (
                f"{model.source_revision}:edge:{edge}",
                _face_id(state.models[opposite[0]], opposite[1]),
            ),
        )
    # A crossing is a Krawczyk root; an analytic tangent point is an exact
    # double root whose multiplicity witness replaces the contraction proof.
    if any(
        point.kind != "crossing"
        and not (
            point.kind == "tangent_point" and point.certificate == "analytic-contact"
        )
        for point in result.points
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.source_edge.junction must be transversal; got a singular contact",
            (
                f"{model.source_revision}:edge:{edge}",
                _face_id(state.models[opposite[0]], opposite[1]),
            ),
        )
    points = sorted(result.points, key=lambda point: float(point.parameters[0]))
    roots = tuple(
        CurveSurfaceIntersectionRoot(
            curve,
            region,
            parameter_lower=point.parameter_lower,
            parameter_upper=point.parameter_upper,
            tangent_witness=(
                None
                if point.kind == "crossing"
                else ExactLineCylinderTangency(curve, region)
            ),
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
    for axis, period in enumerate(region.patch.periods):
        if period is None or region.patch.native_period_symbols()[axis] != "two_pi":
            continue
        shift = round(
            float((0.5 * (lower[axis] + upper[axis]) - reference[axis]) / float(period))
        )
        offset = np.asarray(shift * float(period), np.float64)
        lower_[axis], upper_[axis] = interval_subtract(
            (np.asarray(lower_[axis]), np.asarray(upper_[axis])),
            (offset, offset),
        )
    return lower_, upper_


def source_pcurve_image(curve: AbstractCurve, first: float, last: float, /) -> np.ndarray:
    return np.asarray(curve.bounding_box(first, last), dtype=np.float64)


def joint_root_parameter_image(
    root: TrimIntersectionRoot,
    regions: tuple[SurfaceRegion, SurfaceRegion, SurfaceRegion],
    *,
    source_owner: SurfaceRegion | None = None,
) -> np.ndarray | None:
    """Map exact trim-root operands through coupled branch p-curves to source UV boxes."""
    lower_parameters = np.asarray(root.parameter_lower, dtype=np.float64)
    upper_parameters = np.asarray(root.parameter_upper, dtype=np.float64)
    images: dict[str, list[np.ndarray]] = {}

    def add(patch: AbstractSurfacePatch, box: np.ndarray) -> None:
        key = canonical_fingerprint(carrier_payload(patch))
        images.setdefault(key, []).append(np.asarray(box, dtype=np.float64))

    for operand, carrier in enumerate((root.first, root.second)):
        lo, hi = float(lower_parameters[operand]), float(upper_parameters[operand])
        source = carrier
        if isinstance(source, CurveTrimSegment):
            endpoints = np.asarray(
                source.source_parameters(jnp.asarray((lo, hi), dtype=jnp.float64)),
                dtype=np.float64,
            )
            lo, hi = float(np.min(endpoints)), float(np.max(endpoints))
            source = source.curve
        if isinstance(source, IntersectionPCurve):
            coupled = source.curve.parameter_enclosures(lo, hi)
            for region, columns in (
                (source.curve.first, slice(0, 2)),
                (source.curve.second, slice(2, 4)),
            ):
                add(
                    region.patch,
                    np.stack(
                        (
                            np.min(coupled[:, 0, columns], axis=0),
                            np.max(coupled[:, 1, columns], axis=0),
                        )
                    ),
                )
        elif source_owner is not None:
            add(
                source_owner.patch,
                source_pcurve_image(
                    carrier,
                    float(lower_parameters[operand]),
                    float(upper_parameters[operand]),
                ),
            )
        else:
            return None

    result_lower, result_upper = [], []
    for region in regions:
        candidates = images.get(canonical_fingerprint(carrier_payload(region.patch)), ())
        if not candidates:
            return None
        lower = np.max(np.stack([box[0] for box in candidates]), axis=0)
        upper = np.min(np.stack([box[1] for box in candidates]), axis=0)
        if np.any(lower > upper):
            return None
        result_lower.extend(lower)
        result_upper.extend(upper)
    return np.stack((np.asarray(result_lower), np.asarray(result_upper)))


def _prove_edge_root_alias(
    root: TrimIntersectionRoot,
    source_arc: Arc,
    branch_arc: Arc,
    source_operand: int,
    branch_operand: int,
    spatial: CurveSurfaceIntersectionRoot,
    source_lift: BRepCurveSurfaceLift,
    support: BRepRootSupport,
    /,
) -> bool:
    if (
        source_arc.curve is None
        or not _same_parameter_source(
            (root.first, root.second)[source_operand], source_arc.trim
        )
        or not _same_parameter_source(
            (root.first, root.second)[branch_operand], branch_arc.trim
        )
        or not _same_parameter_source(source_arc.curve, source_lift.curve)
    ):
        return False
    return _curve_surface_root_shares_trim_support(spatial, support, (source_lift,))


def event_vertex(
    state: Arrangement,
    root: TrimIntersectionRoot,
    first: Arc,
    second: Arc,
) -> str:
    source_arc = first if first.source_edge is not None else second
    branch_arc = second if source_arc is first else first
    if source_arc.source_edge is not None and branch_arc.branch is not None:
        source_operand, branch_operand = (0, 1) if source_arc is first else (1, 0)
        source_edge = source_arc.source_edge
        source_curve = source_arc.curve
        source_model = state.models[source_arc.owner[0]]
        source_geometry = source_model.geometry
        if source_curve is None or source_geometry is None:
            raise BRepBooleanFailure(
                "BRepBoolean.source_edge_lift must retain its source curve and geometry",
                (source_arc.token,),
            )
        source_region = state.face_region(source_arc.owner)
        edge_first, edge_last = (
            float(value) for value in np.asarray(source_geometry.edge_ranges)[source_edge]
        )
        try:
            source_lift = BRepCurveSurfaceLift(
                source_curve,
                source_region.patch,
                source_arc.pcurve,
                edge_first,
                edge_last,
                source_region=source_region,
            )
        except ValueError as error:
            raise BRepBooleanFailure(
                "BRepBoolean.source_edge_lift must prove its actual source face",
                (source_arc.token,),
            ) from error
        try:
            support = BRepRootSupport(source_region.patch, root)
        except ValueError as error:
            raise BRepBooleanFailure(
                "BRepBoolean.trim_root must retain an exact source-face support",
                (source_arc.token, branch_arc.token),
            ) from error
        owner_a, owner_b = state.branch_owners[branch_arc.token]
        opposite = owner_b if source_arc.owner == owner_a else owner_a
        roots = _edge_spatial_roots(
            state,
            source_arc.owner,
            source_edge,
            opposite,
        )
        matches = [
            spatial
            for spatial in roots
            if _prove_edge_root_alias(
                root,
                source_arc,
                branch_arc,
                source_operand,
                branch_operand,
                spatial,
                source_lift,
                support,
            )
        ]
        if len(matches) != 1:
            raise BRepBooleanFailure(
                "BRepBoolean.trim_root must correspond to one spatial root; got another candidate count",
                (source_arc.token, branch_arc.token),
            )
        spatial = matches[0]
        vertex = _spatial_root_vertex(source_arc.owner, source_edge, opposite, spatial)
        state.vertex_spatial[vertex] = spatial
        prior_binding = state.vertex_bindings.get(vertex)
        aliases = {
            alias.root_id: alias
            for alias in (() if prior_binding is None else prior_binding.aliases)
        }
        aliases[support.root_id] = support
        lift_records = {
            lift.root_id: lift
            for lift in (
                *state.vertex_lifts.get(vertex, ()),
                *(() if prior_binding is None else prior_binding.source_edge_lifts),
            )
        }
        lift_records[source_lift.root_id] = source_lift
        vertex_lifts = tuple(lift_records[key] for key in sorted(lift_records))
        state.vertex_lifts[vertex] = vertex_lifts
        if prior_binding is not None:
            state.vertex_bindings[vertex] = BRepVertexRoot(
                prior_binding.primary,
                aliases=tuple(aliases[key] for key in sorted(aliases)),
                spatial_root=prior_binding.spatial_root or spatial,
                joint_root=prior_binding.joint_root,
                source_edge_lifts=vertex_lifts,
                branch_node=prior_binding.branch_node,
            )
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
            )
        ]
        certified: list[
            tuple[
                int,
                CurveSurfaceIntersectionRoot,
                TripleSurfaceIntersectionRoot,
                tuple[BRepCurveSurfaceLift, ...],
            ]
        ] = []
        for edge, spatial in candidates:
            try:
                joint, lifts = _triple_vertex_certificate(
                    state,
                    root,
                    first.owner,
                    other_a,
                    other_b,
                    edge,
                    spatial,
                )
            except BRepBooleanFailure:
                continue
            certified.append((edge, spatial, joint, lifts))
        if len(certified) != 1:
            raise BRepBooleanFailure(
                "BRepBoolean.triple_junction.source_edge must have one source-lift certificate; got another root count"
            )
        edge, spatial, joint, lifts = certified[0]
        vertex = _spatial_root_vertex(other_a, edge, first.owner, spatial)
        state.vertex_spatial[vertex] = spatial
        state.vertex_joint[vertex] = joint
        state.vertex_lifts[vertex] = lifts
    else:
        raise BRepBooleanFailure(
            "BRepBoolean.source_boundary must declare every self-intersection; got an undeclared intersection"
        )
    patch = state.models[first.owner[0]].patches[first.owner[1]]
    aliases = state.vertex_aliases.setdefault(vertex, [])
    if all(old_root.root_id != root.root_id for _, old_root in aliases):
        aliases.append((patch, root))
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
    uv = np.asarray(root.first.evaluate(root.parameters[0]), dtype=np.float64)
    patch = state.models[arc.owner[0]].patches[arc.owner[1]]
    point = np.asarray(patch.evaluate(jnp.asarray(uv, jnp.float64)))
    node = f"root:{root.root_id}:vertex:{vertex}:face:{arc.owner}"
    first, last = _arc_parameter_interval(arc)
    if lower <= first <= upper:
        node = arc.first_node
    elif lower <= last <= upper:
        node = arc.last_node
    state.cuts[arc.token].append(
        Cut(parameter, lower, upper, node, vertex, endpoint, None, point)
    )


def piece_trim(
    piece: Piece,
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
    branch_cuts: dict[tuple[str, tuple[FaceOwner, FaceOwner]], dict[str, Cut]] = {}
    for arcs in state.arcs.values():
        for arc in arcs:
            if arc.branch is None:
                continue
            for cut in state.cuts[arc.token]:
                if cut.endpoint is not None:
                    key = arc.branch.branch_id, state.branch_owners[arc.token]
                    branch_cuts.setdefault(key, {}).setdefault(cut.vertex, cut)
    for owner, arcs in state.arcs.items():
        for arc in arcs:
            if arc.branch is None:
                continue
            first, last = _arc_parameter_interval(arc)
            local = {
                cut.vertex for cut in state.cuts[arc.token] if cut.endpoint is not None
            }
            key = arc.branch.branch_id, state.branch_owners[arc.token]
            for vertex, cut in branch_cuts.get(key, {}).items():
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
    edge_sources: tuple[tuple[SourceEntity, ...], ...] = eqx.field(static=True)
    vertex_sources: tuple[tuple[SourceEntity, ...], ...] = eqx.field(static=True)


def _vertex_binding(
    state: Arrangement,
    vertex: str,
    source_entity: SourceEntity | None,
) -> BRepVertexRoot | None:
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
    elif source_entity is not None:
        source_model, entity = source_entity
        if entity.kind != "vertex":
            raise BRepBooleanFailure(
                "BRepBoolean.lineage.source_vertex must reference a vertex entity"
            )
        geometry = state.models[source_model].geometry
        if geometry is None:
            raise BRepBooleanFailure("BRepBoolean.source_vertex.geometry must be defined")
        binding = _source_vertex_root(geometry.vertex_roots[entity.index])
    else:
        raise BRepBooleanFailure(
            "BRepBoolean.vertex must retain its exact source-root or branch binding",
            (vertex,),
        )
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
            entity.source_id,
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
    endpoint: RootEndpoint | None,
    carrier: AbstractCurve,
    parameter: Fraction,
    patch: AbstractSurfacePatch,
    *,
    coedge: bool,
) -> RootEndpoint | None:
    """Bind an authored native scalar or conic root to the retained source lift.

    Coincident chart transport changes the UV carrier, not its raw parameter.
    The new carrier retains the source source expression and chart map. A conic
    root reached on a coincident carrier binds that carrier at the retained raw
    edge parameter; the model's exact conic endpoint check proves the point.
    A native period root names its period through the publishing face patch:
    a transported coedge root keeps its raw expression and rebinds the
    coincident patch's identical declared period on the same axis; an edge
    root rebinds only when its publishing patch declares that period.
    Isolated roots remain with their existing certified support definitions.
    """
    if isinstance(endpoint, ConicParameterEndpoint):
        if carrier_payload(endpoint.carrier) == carrier_payload(carrier):
            return endpoint
        return ConicParameterEndpoint(
            endpoint.root, carrier, parameter, endpoint.point_exact
        )
    if not isinstance(endpoint, NativePeriodEndpoint):
        return endpoint
    same_curve = canonical_fingerprint(
        carrier_payload(endpoint.curve)
    ) == canonical_fingerprint(carrier_payload(carrier))
    period_patch = endpoint.patch
    if period_patch is not None and carrier_payload(period_patch) != carrier_payload(
        patch
    ):
        axis = endpoint.axis
        target_period = None if axis is None else patch.periods[axis]
        source_period = None if axis is None else period_patch.periods[axis]
        if (
            axis is not None
            and patch.native_period_symbols()[axis] == "two_pi"
            and target_period is not None
            and source_period is not None
            and float(target_period) == float(source_period)
        ):
            period_patch = patch
        elif coedge:
            raise BRepBooleanFailure(
                "BRepBoolean.native_period_endpoint must rebind an identical declared period on its publishing face patch"
            )
        elif same_curve:
            # An edge root may keep the period of another incident face.
            return endpoint
    elif same_curve:
        return endpoint
    return NativePeriodEndpoint(
        carrier,
        period_patch,
        endpoint.axis,
        rational=endpoint.rational,
        turns=endpoint.turns,
    )


class _BoundVertex(StrictModule):
    point: np.ndarray
    binding: BRepVertexRoot | None
    exact_point: tuple[Fraction, Fraction, Fraction] | None


def _certify_boundary_vertex(
    state: Arrangement,
    arc: Arc,
    cut: Cut,
    source_entities: tuple[SourceEntity, ...],
) -> _BoundVertex:
    source_entity = source_entities[0] if source_entities else None
    binding = _vertex_binding(state, cut.vertex, source_entity)
    exact_points = []
    for source_model, entity in source_entities:
        geometry = state.models[source_model].geometry
        if geometry is None:
            raise BRepBooleanFailure(
                "BRepBoolean.source_vertex.geometry must be defined; got None"
            )
        exact_points.append(geometry.vertex_exact_points[entity.index])
    if exact_points and any(point != exact_points[0] for point in exact_points[1:]):
        raise BRepBooleanFailure(
            "BRepBoolean.lineage.source_vertex must bind equal exact source points"
        )
    exact_point = exact_points[0] if exact_points else None
    if binding is None:
        # A branch end proved to be a source vertex realizes the vertex's own
        # exact point, not the branch's numerical evaluation there.
        point = (
            cut.point
            if exact_point is None
            else np.asarray([float(value) for value in exact_point], dtype=np.float64)
        )
        return _BoundVertex(point, binding, exact_point)
    box = binding.point_enclosure()
    if box.shape != (2, 3) or not np.all(np.isfinite(box)) or np.any(box[0] > box[1]):
        raise BRepBooleanFailure(
            "BRepBoolean.vertex must have a finite ordered source-root enclosure"
        )
    point = box[0] + 0.5 * (box[1] - box[0])
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
    if not np.isfinite(bound) or bound > state.policy.construction_tolerance:
        raise BRepBooleanFailure(
            "BRepBoolean.vertex must have a certified positional bound within construction_tolerance; got nonfinite or excessive error",
            (arc.token, cut.vertex),
        )
    return _BoundVertex(point, binding, exact_point)


class _BoundaryTopology(StrictModule):
    vertices: list[np.ndarray] = eqx.field(default_factory=list)
    vertex_roots: list[BRepVertexRoot | None] = eqx.field(default_factory=list)
    vertex_exact_points: list[tuple[Fraction, Fraction, Fraction] | None] = eqx.field(
        default_factory=list
    )
    vertex_ids: dict[str, int] = eqx.field(default_factory=dict)
    curves: list[AbstractCurve] = eqx.field(default_factory=list)
    edge_curves: list[int] = eqx.field(default_factory=list)
    edge_vertices: list[tuple[int, int]] = eqx.field(default_factory=list)
    edge_ranges: list[tuple[float, float]] = eqx.field(default_factory=list)
    edge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = eqx.field(
        default_factory=list
    )
    edge_ids: dict[tuple[str, str, str], int] = eqx.field(default_factory=dict)
    pcurves: list[AbstractCurve] = eqx.field(default_factory=list)
    coedge_edges: list[int] = eqx.field(default_factory=list)
    coedge_senses: list[int] = eqx.field(default_factory=list)
    coedge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = eqx.field(
        default_factory=list
    )
    face_loops: list[tuple[tuple[int, ...], ...]] = eqx.field(default_factory=list)
    scalar_boxes: dict[tuple[int, int], list[tuple[float, float]]] = eqx.field(
        default_factory=dict
    )
    edge_sources: list[set[SourceEntity]] = eqx.field(default_factory=list)
    vertex_sources: list[set[SourceEntity]] = eqx.field(default_factory=list)
    entity_cache: dict[int, dict[tuple[str, int], tuple[BRepEntityId, ...]]] = eqx.field(
        default_factory=dict
    )

    def _entity_index_for_model(
        self, state: Arrangement, model_index: int, /
    ) -> dict[tuple[str, int], tuple[BRepEntityId, ...]]:
        if model_index not in self.entity_cache:
            model = state.models[model_index]
            grouped: dict[tuple[str, int], set[BRepEntityId]] = {}
            for record in model.geometry.qualified_entity_incidence(
                model.source_id, model.source_revision
            ):
                if record.container == record.member:
                    grouped.setdefault(
                        (record.member.kind, record.member.index), set()
                    ).add(record.member)
            self.entity_cache[model_index] = {
                key: tuple(sorted(values)) for key, values in grouped.items()
            }
        return self.entity_cache[model_index]

    def _entities_on_face(
        self,
        state: Arrangement,
        owner: FaceOwner,
        kind: str,
        index: int,
        /,
    ) -> tuple[BRepEntityId, ...]:
        model_index, face = owner
        entity_index = self._entity_index_for_model(state, model_index)
        face_paths = {
            entity.occurrence_path for entity in entity_index.get(("face", face), ())
        }
        return tuple(
            entity
            for entity in entity_index.get((kind, index), ())
            if entity.occurrence_path in face_paths
        )

    def _vertex_occurrences(
        self,
        state: Arrangement,
        owner: FaceOwner,
        source_entities: tuple[SourceEntity, ...],
        /,
    ) -> tuple[SourceEntity, ...]:
        """Bind model-level source vertices to their exact operand occurrences.

        A vertex of the publishing face's operand keeps the occurrences carrying
        that face; a coincident operand's vertex must have one occurrence.
        """
        result: list[SourceEntity] = []
        for operand, entity in source_entities:
            if entity.occurrence_path != ():
                result.append((operand, entity))
                continue
            occurrences = (
                self._entities_on_face(state, owner, "vertex", entity.index)
                if operand == owner[0]
                else self._entity_index_for_model(state, operand).get(
                    ("vertex", entity.index), ()
                )
            )
            if len(occurrences) != 1 and operand != owner[0]:
                raise BRepBooleanFailure(
                    "BRepBoolean.lineage.coincident_vertex must name one exact operand occurrence",
                    (str(entity),),
                )
            if not occurrences:
                raise BRepBooleanFailure(
                    "BRepBoolean.lineage.source_vertex must lie on its publishing face occurrence",
                    (str(entity),),
                )
            result.extend((operand, occurrence) for occurrence in occurrences)
        return tuple(result)

    def append_piece(
        self,
        state: Arrangement,
        piece: Piece,
        overlay_sources: Mapping[str, CoincidentFaceArc] | None,
        source_vertex_entities: Mapping[str, tuple[SourceEntity, ...]],
    ) -> int:
        arc = piece.arc
        endpoints = (
            (piece.first, piece.last) if arc.scale > 0.0 else (piece.last, piece.first)
        )
        ids = []
        for cut in endpoints:
            source_entities = source_vertex_entities.get(cut.vertex, ())
            if any(entity.kind != "vertex" for _, entity in source_entities):
                raise BRepBooleanFailure(
                    "BRepBoolean.lineage.source_vertex must reference a vertex entity"
                )
            if cut.vertex not in self.vertex_ids:
                prepared = _certify_boundary_vertex(state, arc, cut, source_entities)
                self.vertex_ids[cut.vertex] = len(self.vertices)
                self.vertices.append(prepared.point)
                self.vertex_roots.append(prepared.binding)
                self.vertex_exact_points.append(prepared.exact_point)
                self.vertex_sources.append(set())
            vertex = self.vertex_ids[cut.vertex]
            self.vertex_sources[vertex].update(
                self._vertex_occurrences(state, arc.owner, source_entities)
            )
            ids.append(vertex)
        source = None if overlay_sources is None else overlay_sources.get(arc.token)
        key = _edge_key(piece, source)
        raw = tuple(
            Fraction(arc.offset) + Fraction(arc.scale) * Fraction(cut.parameter)
            for cut in endpoints
        )
        patch = state.models[arc.owner[0]].patches[arc.owner[1]]
        if key not in self.edge_ids:
            edge = len(self.edge_vertices)
            self.edge_ids[key] = edge
            if arc.curve is None:
                self.edge_curves.append(-1)
            else:
                self.edge_curves.append(len(self.curves))
                self.curves.append(arc.curve)
            self.edge_vertices.append((ids[0], ids[1]))
            self.edge_sources.append(set())
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
                    else _endpoint_on_carrier(
                        endpoints[0].endpoint, arc.curve, raw[0], patch, coedge=False
                    ),
                    endpoints[1].endpoint
                    if arc.curve is None
                    else _endpoint_on_carrier(
                        endpoints[1].endpoint, arc.curve, raw[1], patch, coedge=False
                    ),
                )
            )
        edge = self.edge_ids[key]
        if source is None:
            source_edge = arc.source_edge
            if source_edge is not None:
                source_entities = self._entities_on_face(
                    state, arc.owner, "edge", source_edge
                )
                if len(source_entities) != 1:
                    raise BRepBooleanFailure(
                        "BRepBoolean.lineage.source_edge must have one exact face occurrence"
                    )
                self.edge_sources[edge].add((arc.owner[0], source_entities[0]))
        else:
            if source.edge_id.kind != "edge":
                raise BRepBooleanFailure(
                    "BRepBoolean.lineage.coincident_edge must reference an edge entity"
                )
            # A model-level edge id names every occurrence of that edge; keep
            # exactly the occurrences that carry the overlaid source face.
            coincident_entities: set[tuple[int, BRepEntityId]] = set()
            for model_index, model in enumerate(state.models):
                if (
                    model.source_id != source.edge_id.source_id
                    or model.source_revision != source.edge_id.source_revision
                ):
                    continue
                index = self._entity_index_for_model(state, model_index)
                face_paths = {
                    entity.occurrence_path
                    for entity in index.get(("face", source.face_id.index), ())
                }
                for entity in index.get(("edge", source.edge_id.index), ()):
                    if entity == source.edge_id or (
                        source.edge_id.occurrence_path == ()
                        and entity.occurrence_path in face_paths
                    ):
                        coincident_entities.add((model_index, entity))
            if not coincident_entities:
                raise BRepBooleanFailure(
                    "BRepBoolean.lineage.coincident_edge must retain its exact source owner, revision, and occurrence"
                )
            self.edge_sources[edge].update(coincident_entities)
        for endpoint, cut in enumerate(endpoints):
            if cut.endpoint is not None:
                self.scalar_boxes.setdefault((edge, endpoint), []).append(
                    cut.endpoint.parameter_enclosure()
                )
        pcurve = arc.pcurve
        # Keep the certified branch-span carrier. The edge range contains
        # root representatives, not necessarily their full isolating boxes;
        # restricting the p-curve to those representatives would exclude the
        # very endpoint roots the coedge must retain.
        self.pcurves.append(pcurve)
        coedge = len(self.coedge_edges)
        self.coedge_edges.append(edge)
        self.coedge_senses.append(piece.sense * (1 if arc.scale > 0.0 else -1))
        self.coedge_roots.append(
            (
                _endpoint_on_carrier(
                    endpoints[0].endpoint, pcurve, raw[0], patch, coedge=True
                ),
                _endpoint_on_carrier(
                    endpoints[1].endpoint, pcurve, raw[1], patch, coedge=True
                ),
            )
        )
        return coedge

    def geometry(
        self,
        state: Arrangement,
        fragments: tuple[FaceFragment, ...],
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
        builder = _Builder()
        for point, exact_point, root in zip(
            self.vertices, self.vertex_exact_points, self.vertex_roots, strict=True
        ):
            builder.vertex(point, exact_point=exact_point, root=root)
        for edge, (start, end) in enumerate(self.edge_vertices):
            first, last = self.edge_ranges[edge]
            roots = self.edge_roots[edge]
            curve_index = self.edge_curves[edge]
            if curve_index < 0:
                if start != end:
                    raise BRepBooleanFailure(
                        "BRepBoolean.boundary.degenerate_edge must have one source vertex; got distinct endpoints"
                    )
                builder.degenerate(start, first, last, endpoint_roots=roots)
            else:
                builder.edge(
                    self.curves[curve_index],
                    first,
                    last,
                    start,
                    end,
                    endpoint_roots=roots,
                )
        for edge, sense, pcurve, roots in zip(
            self.coedge_edges,
            self.coedge_senses,
            self.pcurves,
            self.coedge_roots,
            strict=True,
        ):
            builder.coedge(edge, sense, pcurve, endpoint_roots=roots)
        for face, fragment in enumerate(fragments):
            model = state.models[fragment.owner[0]]
            builder.face(
                model.patches[fragment.owner[1]],
                np.asarray(model.parameter_bounds)[fragment.owner[1]],
                fragment.orientation,
                model.physical_tags[fragment.owner[1]],
                [list(loop) for loop in self.face_loops[face]],
            )
        geometry, _, _, _, _, _ = builder.geometry(solid=False)
        return geometry


def _face_curve_trim_domain(geometry: BRepGeometry, face: int, /) -> TrimDomain:
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    curve_loops = []
    for loop in geometry.face_loops[face]:
        segments = []
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            first, last = (float(value) for value in ranges[edge])
            first_root, last_root = geometry.coedge_endpoint_roots[coedge]
            segments.append(
                CurveTrimSegment(
                    geometry.pcurves[coedge],
                    first,
                    last,
                    reversed=geometry.coedge_senses[coedge] < 0,
                    first_root=first_root,
                    last_root=last_root,
                )
            )
        curve_loops.append(tuple(segments))
    return TrimDomain.from_curve_loops(tuple(curve_loops))


def _trim_domains_from_geometry(geometry: BRepGeometry) -> tuple[TrimDomain, ...]:
    return tuple(
        _face_curve_trim_domain(geometry, face)
        for face in range(len(geometry.face_loops))
    )


def native_chart_trim_domain(model: BRepModel, face: int, /) -> TrimDomain | None:
    """Return the face trim in native patch-chart coordinates.

    Polygon trims are published normalized to the face parameter bounds;
    arrangement and classification boxes live in the native chart, so those
    faces are re-expressed by their exact source p-curve loops.
    """
    domain = model.trim_domains[face]
    if domain is None or domain.curve_loops:
        return domain
    geometry = model.geometry
    if geometry is None:
        return None
    return _face_curve_trim_domain(geometry, face)


@checked
def build_boundary(
    state: Arrangement,
    fragments: tuple[FaceFragment, ...],
    *,
    source_vertex_entities: Mapping[str, tuple[SourceEntity, ...]],
    overlay_sources: Mapping[str, CoincidentFaceArc] | None = None,
    vertex_bindings: Mapping[str, BRepVertexRoot | None] | None = None,
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
                        source_vertex_entities,
                    )
                )
            loops.append(tuple(coedges))
        topology.face_loops.append(tuple(loops))
    geometry = topology.geometry(state, fragments)
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
        _trim_domains_from_geometry(geometry),
        tuple(fragment.owner for fragment in fragments),
        tuple(
            state.models[fragment.owner[0]].physical_tags[fragment.owner[1]]
            for fragment in fragments
        ),
        tuple(
            tuple(sorted(values, key=lambda item: (item[0], item[1])))
            for values in topology.edge_sources
        ),
        tuple(
            tuple(sorted(values, key=lambda item: (item[0], item[1])))
            for values in topology.vertex_sources
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
        vertex_exact_points=geometry.vertex_exact_points,
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
    state: Arrangement,
    built: BuiltBoundary,
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
    return assemble_brep_model(
        geometry,
        built.patches,
        built.bounds,
        orientation,
        built.tags,
        coordinate_contract=state.models[0].coordinate_contract,
        source_id=f"native-boolean-shell:{digest}",
        source_format="native",
        source_digest=digest,
        import_policy_id=digest,
        tessellation=None,
        trim_domains=built.domains,
    )


def _face_seed(
    domain: TrimDomain,
    bounds: np.ndarray,
    relative_resolution: float,
) -> np.ndarray:
    pending = [np.asarray(bounds, dtype=np.float64)]
    while pending:
        box = pending.pop()
        center = box[0] + 0.5 * (box[1] - box[0])
        membership = domain.classify(center[None])
        if membership.resolved[0] and membership.inside[0]:
            return center
        widths = box[1] - box[0]
        scale = np.maximum(1.0, np.maximum(np.abs(box[0]), np.abs(box[1])))
        resolution = np.maximum(
            relative_resolution * np.maximum(widths, np.finfo(np.float64).tiny),
            64.0 * np.finfo(np.float64).eps * scale,
        )
        if np.any(widths <= resolution):
            continue
        axis = int(np.argmax(widths / resolution))
        middle = box[0, axis] + 0.5 * widths[axis]
        if not box[0, axis] < middle < box[1, axis]:
            continue
        lower, upper = box.copy(), box.copy()
        lower[1, axis] = middle
        upper[0, axis] = middle
        pending.append(upper)
        pending.append(lower)
    raise BRepBooleanFailure(
        "BRepBoolean.trim.interior_seed remains unresolved at the declared parameter resolution"
    )


def _contains_shell(projection: NativeBRepProjection, point: np.ndarray, /) -> bool:
    result = projection.locate_solids(point)
    status = BRepProjectionStatus(int(np.asarray(result.status)[0]))
    if status not in (BRepProjectionStatus.UNIQUE, BRepProjectionStatus.FAILED):
        raise BRepBooleanFailure(
            "BRepBoolean.shell.containment remains unresolved in the canonical source projection"
        )
    return (
        status == BRepProjectionStatus.UNIQUE and int(np.asarray(result.indices)[0]) == 0
    )


def _oriented_solids(
    state: Arrangement,
    built: BuiltBoundary,
    faces: tuple[int, ...],
    outward: np.ndarray,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Closed shells of one material set and their outer/cavity solid nesting.

    ``outward`` is each face's material-outward sign relative to its patch
    normal. Returned solids index the returned shells, outer shell first.
    """
    from ._measure import BRepMeasureRefusal, source_shell_volume_sign

    shells_bank = shells(built.geometry, faces, outward)
    if len(shells_bank) == 1:
        # Operand solids are bounded, so the material is bounded: the only
        # closed shell bounds it from outside and is positively oriented.
        return shells_bank, ((0,),)
    try:
        signs = tuple(
            source_shell_volume_sign(
                built.geometry,
                built.patches,
                built.bounds,
                built.domains,
                outward,
                shell,
            )
            for shell in shells_bank
        )
    except BRepMeasureRefusal as error:
        raise BRepBooleanFailure(
            f"BRepBoolean.shell.volume_sign remains unresolved: {error}"
        ) from error
    outer = [index for index, sign in enumerate(signs) if sign > 0]
    if not outer:
        raise BRepBooleanFailure(
            "BRepBoolean.shells must contain a positively oriented material shell; got none"
        )
    if len(outer) == len(shells_bank):
        return shells_bank, tuple((index,) for index in outer)
    queries = {
        index: prepare_brep_projection(
            _shell_query_model(
                state,
                built,
                shells_bank[index],
                outward,
            ),
        )
        for index in outer
    }
    parent_contains: dict[tuple[int, int], bool] = {}
    for child in outer:
        face = shells_bank[child][0]
        uv = _face_seed(
            built.domains[face],
            built.bounds[face],
            state.policy.intersection.relative_resolution,
        )
        point = np.asarray(built.patches[face].evaluate(jnp.asarray(uv, jnp.float64)))[
            None
        ]
        for parent, projection in queries.items():
            if child == parent:
                continue
            parent_contains[parent, child] = _contains_shell(projection, point)
    children: dict[int, list[int]] = {index: [] for index in outer}
    for index, sign in enumerate(signs):
        if sign > 0:
            continue
        face = shells_bank[index][0]
        uv = _face_seed(
            built.domains[face],
            built.bounds[face],
            state.policy.intersection.relative_resolution,
        )
        point = np.asarray(built.patches[face].evaluate(jnp.asarray(uv, jnp.float64)))[
            None
        ]
        candidates = []
        for parent, projection in queries.items():
            if _contains_shell(projection, point):
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
) -> BRepGeometry:
    if not built.patches:
        return built.geometry
    faces = tuple(range(len(built.patches)))
    shells, solids = _oriented_solids(
        state,
        built,
        faces,
        np.asarray(built.orientation),
    )
    return _with_shells(
        built.geometry,
        shells,
        tuple(tuple(1 for _ in shell) for shell in shells),
        solids,
    )


@checked
def reconstruct_region_solids(
    state: Arrangement,
    built: BuiltBoundary,
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
        built.geometry,
        tuple(shells),
        tuple(shell_orientations),
        tuple(solids),
    ), tuple(owners)


@checked
def publish_curved(
    state: Arrangement,
    built: BuiltBoundary,
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
        trim_domains=built.domains,
    )


def _triple_vertex_certificate(
    state: Arrangement,
    root: TrimIntersectionRoot,
    owner: FaceOwner,
    other_a: FaceOwner,
    other_b: FaceOwner,
    edge: int,
    spatial: CurveSurfaceIntersectionRoot,
) -> tuple[TripleSurfaceIntersectionRoot, tuple[BRepCurveSurfaceLift, ...]]:

    regions = (
        state.face_region(owner),
        state.face_region(other_a),
        state.face_region(other_b),
    )
    image = joint_root_parameter_image(root, regions, source_owner=regions[0])
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
    curve_index = geometry.edge_curves[edge]
    if curve_index < 0:
        raise BRepBooleanFailure(
            "BRepBoolean.triple_lift.edge must retain an exact source curve"
        )
    source_curve = geometry.curves[curve_index]
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
    t_lower, t_upper = (
        float(spatial.parameter_lower[0]),
        float(spatial.parameter_upper[0]),
    )
    boxes: dict[str, list[np.ndarray]] = {}

    def add(patch: AbstractSurfacePatch, box: np.ndarray) -> None:
        key = canonical_fingerprint(carrier_payload(patch))
        boxes.setdefault(key, []).append(np.asarray(box, dtype=np.float64))

    add(
        spatial.surface.patch,
        np.stack((spatial.parameter_lower[1:], spatial.parameter_upper[1:])),
    )
    lifts = []
    for face_owner in (other_a, other_b):
        if face_owner[0] != other_a[0]:
            continue
        face = face_owner[1]
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
        region = state.face_region(face_owner)
        lift = BRepCurveSurfaceLift(
            source_curve, region.patch, pcurve, first, last, source_region=region
        )
        lifts.append(lift)
        add(region.patch, source_pcurve_image(pcurve, t_lower, t_upper))

    lower, upper = image[0].copy(), image[1].copy()
    for index, region in enumerate(regions):
        candidates = boxes.get(canonical_fingerprint(carrier_payload(region.patch)), ())
        for candidate in candidates:
            columns = slice(2 * index, 2 * index + 2)
            lower[columns] = np.minimum(lower[columns], candidate[0])
            upper[columns] = np.maximum(upper[columns], candidate[1])
    margin = (
        256.0
        * np.finfo(np.float64).eps
        * (1.0 + np.maximum(np.abs(lower), np.abs(upper)))
    )
    lower, upper = (
        np.nextafter(lower - margin, -np.inf),
        np.nextafter(upper + margin, np.inf),
    )
    joint = TripleSurfaceIntersectionRoot(
        *regions, parameter_lower=lower, parameter_upper=upper
    )
    if not joint.certifies_root(
        root, source_edge_lifts=tuple(lifts)
    ) or not joint.certifies_root(spatial, source_edge_lifts=tuple(lifts)):
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
    side: IntersectionSide = eqx.field(static=True)
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
) -> None:
    branch, side = pair.branch, pair.side
    span_first, span_last = pair.span_first, pair.span_last
    source_box = source_pcurve_image(
        source_pcurve,
        float(spatial.parameter_lower[0]),
        float(spatial.parameter_upper[0]),
    )
    opposite_box = np.stack((spatial.parameter_lower[1:], spatial.parameter_upper[1:]))
    image = np.concatenate(
        (source_box, opposite_box) if side == "first" else (opposite_box, source_box),
        axis=1,
    )
    # A source root may sit on a chart node; its isolation box then meets both
    # adjacent charts. Select the first meeting chart whose endpoint certifies
    # the branch parameter (inside the chart or as the identified node). A root
    # on the opposite face's seam is seen in one native sheet; the chart may
    # use another, so each exact opposite deck turn is tried, untouched first.
    opposite_patch = (branch.second if side == "first" else branch.first).patch
    opposite_offset = 2 if side == "first" else 0
    periodic_axes = tuple(
        opposite_offset + local
        for local, (period, symbol) in enumerate(
            zip(
                opposite_patch.periods,
                opposite_patch.native_period_symbols(),
                strict=True,
            )
        )
        if period is not None and symbol == "two_pi"
    )
    gauges = []
    for turns in itertools.product((0, 1, -1), repeat=len(periodic_axes)):
        gauge = [0, 0, 0, 0]
        for axis, turn in zip(periodic_axes, turns, strict=True):
            gauge[axis] = turn
        gauges.append((gauge[0], gauge[1], gauge[2], gauge[3]))
    branch_endpoint: BranchRootEndpoint | None = None
    enclosure: tuple[float, float] | None = None
    for gauge in gauges:
        shift_lower, shift_upper = branch.shift_parameter_bounds(
            np.asarray(gauge, dtype=np.int64)
        )
        gauged = np.stack(
            (
                np.nextafter(image[0] + shift_lower, -np.inf),
                np.nextafter(image[1] + shift_upper, np.inf),
            )
        )
        for chart in range(int(span_first), int(span_last)):
            if np.any(gauged[1] < branch.chart_lower[chart]) or np.any(
                gauged[0] > branch.chart_upper[chart]
            ):
                continue
            candidate = BranchRootEndpoint(
                spatial,
                branch,
                chart,
                source_pcurve=source_pcurve,
                source_side=side,
                source_first=pair.whole_first,
                source_last=pair.whole_last,
                periodic_shifts=gauge,
            )
            try:
                enclosure = candidate.parameter_enclosure()
            except ValueError:
                continue
            branch_endpoint = candidate
            break
        if branch_endpoint is not None:
            break
    if branch_endpoint is None or enclosure is None:
        return
    lower, upper = enclosure
    if upper < span_first or lower > span_last:
        return
    parameter = min(span_last, max(span_first, lower + 0.5 * (upper - lower)))
    if not lower <= parameter <= upper:
        raise BRepBooleanFailure(
            "BRepBoolean.source_root.parameter must lie inside its enclosure; got an outside representative"
        )
    source_endpoint = TrimRootEndpoint(
        spatial,
        "first",
        source_pcurve=source_pcurve,
        source_surface=state.face_region(owner),
    )
    source_local = affine_root_endpoint(
        source_endpoint, 1.0 / source_arc.scale, -source_arc.offset / source_arc.scale
    )
    source_lower, source_upper = source_local.parameter_enclosure()
    # An exact double root encloses its edge parameter as one float.
    if not 0.0 < source_lower <= source_upper < 1.0:
        raise BRepBooleanFailure(
            "BRepBoolean.source_root.interval must lie strictly inside the source edge; got a corner root without a vertex alias",
            (source_arc.token,),
        )
    source_parameter = source_lower + 0.5 * (source_upper - source_lower)
    first_owner, second_owner = state.branch_owners[branch_arc.token]
    opposite = second_owner if owner == first_owner else first_owner
    assert source_arc.source_edge is not None
    vertex = _spatial_root_vertex(owner, source_arc.source_edge, opposite, spatial)
    binding = BRepVertexRoot(spatial)
    root_box = binding.point_enclosure()
    point = root_box[0] + 0.5 * (root_box[1] - root_box[0])
    node = f"{vertex}:face:{owner}"
    alias = _bind_fixed_branch_alias(
        state,
        branch_arc,
        branch_endpoint,
        vertex,
        binding,
        point,
    )
    if alias is not None and alias[1] != vertex:
        # Another source root already certified as this branch node: one vertex.
        node, vertex = alias
        binding = _bind_second_source_root(state, vertex, branch_arc, branch_endpoint)
    else:
        state.vertex_spatial[vertex] = spatial
        state.vertex_aliases.setdefault(vertex, [])
        state.vertex_bindings[vertex] = binding
        if alias is not None:
            node = alias[0]
    state.cuts[source_arc.token].append(
        Cut(
            source_parameter,
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


def _bind_second_source_root(
    state: Arrangement, vertex: str, arc: Arc, endpoint: BranchRootEndpoint
) -> BRepVertexRoot:
    """Bind a second source-edge root certified as an existing branch-node vertex.

    Both roots are proved to be the same isolated branch node, as where two
    periodic seams cross on the branch; the vertex records that proof.
    """
    existing = state.vertex_bindings[vertex]
    if existing is None or not isinstance(existing.primary, CurveSurfaceIntersectionRoot):
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias must share a source-root vertex binding",
            (vertex,),
        )
    if existing.primary.root_id == endpoint.root.root_id or (
        existing.branch_node is not None and existing.branch_node.binds(endpoint.root)
    ):
        return existing
    lower, upper = endpoint.parameter_enclosure()
    primary_endpoints = tuple(
        cut.endpoint
        for cut in state.cuts[arc.token]
        if cut.vertex == vertex
        and isinstance(cut.endpoint, BranchRootEndpoint)
        and cut.endpoint.root.root_id == existing.primary.root_id
    )
    if lower != upper or not float(lower).is_integer() or not primary_endpoints:
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias must retain both certified node endpoints",
            (vertex,),
        )
    curve = endpoint.curve
    parameter = 0.0 if curve.closed and lower == curve.num_charts else float(lower)
    endpoints = (
        (primary_endpoints[0], endpoint)
        if existing.branch_node is None
        else (*existing.branch_node.endpoints, endpoint)
    )
    try:
        branch_node = BRepBranchNodeRoots(
            IntersectionCurvePointRoot(curve, parameter), endpoints
        )
    except ValueError as error:
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias must certify every source root as the branch node",
            (vertex,),
        ) from error
    binding = BRepVertexRoot(
        existing.primary,
        aliases=existing.aliases,
        spatial_root=existing.spatial_root,
        joint_root=existing.joint_root,
        source_edge_lifts=existing.source_edge_lifts,
        branch_node=branch_node,
    )
    state.vertex_bindings[vertex] = binding
    for token, cuts in state.cuts.items():
        state.cuts[token] = [
            replace(cut, vertex_root=binding) if cut.vertex == vertex else cut
            for cut in cuts
        ]
    return binding


@checked
def boundary_spatial_events(
    state: Arrangement,
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
                analytic_family = (
                    orthogonal_cylinder_family(branch.first, branch.second) is not None
                )
                if (
                    branch.start_kind != "boundary"
                    and branch.end_kind != "boundary"
                    and not any(
                        periodic_transition(branch, index)
                        for index in range(branch.num_charts)
                    )
                    and not analytic_family
                ):
                    continue
                first_owner, second_owner = state.branch_owners[branch_arc.token]
                side: IntersectionSide = "first" if owner == first_owner else "second"
                opposite = second_owner if side == "first" else first_owner
                geometry = state.models[owner[0]].geometry
                if geometry is None:
                    raise RuntimeError(
                        "BRepBoolean.spatial_arc.geometry must be defined; got None"
                    )
                whole_first, whole_last = (
                    float(value)
                    for value in np.asarray(geometry.edge_ranges)[source.edge]
                )
                span_first, span_last = _arc_parameter_interval(branch_arc)
                source_curve = source_arc.curve
                if (
                    analytic_family
                    and isinstance(source_curve, AbstractCurve)
                    and source_curve.ambient_dimension == 3
                ):
                    try:
                        source_box = np.asarray(
                            source_curve.bounding_box(whole_first, whole_last),
                            dtype=np.float64,
                        )
                        branch_box = np.asarray(
                            branch.bounding_box(span_first, span_last),
                            dtype=np.float64,
                        )
                    except NotImplementedError:
                        pass
                    else:
                        if (
                            source_box.shape == (2, 3)
                            and branch_box.shape == (2, 3)
                            and np.all(np.isfinite(source_box))
                            and np.all(np.isfinite(branch_box))
                            and (
                                np.any(source_box[1] < branch_box[0])
                                or np.any(branch_box[1] < source_box[0])
                            )
                        ):
                            first_token, last_token = sorted(
                                (source_arc.token, branch_arc.token)
                            )
                            state.spatial_pairs.add((first_token, last_token))
                            continue
                roots = _edge_spatial_roots(
                    state,
                    owner,
                    source.edge,
                    opposite,
                )
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
                    )
                first_token, last_token = sorted((source_arc.token, branch_arc.token))
                state.spatial_pairs.add((first_token, last_token))


class _SourceNodeGauge(StrictModule):
    shifts: np.ndarray
    closing: bool


def _branch_node_gauge(branch: IntersectionCurve, node: int, chart: int) -> np.ndarray:
    direction = 1 if node == chart else 0
    return np.asarray(branch.node_period_shifts[node, direction], dtype=np.int64)


def _certify_source_node_gauge(
    endpoint: BranchRootEndpoint,
    matched_root: IntersectionCurvePointRoot,
) -> _SourceNodeGauge:
    branch = endpoint.curve
    if (
        matched_root.curve.branch_id != branch.branch_id
        or not matched_root.parameter.is_integer()
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias must retain an isolated integer branch node root"
        )
    matched_node = int(matched_root.parameter)
    for node in (endpoint.chart, endpoint.chart + 1):
        same_node = matched_node == node or (
            branch.closed and {matched_node, node} == {0, branch.num_charts}
        )
        if not same_node:
            continue
        shifts = _branch_node_gauge(branch, node, endpoint.chart)
        closing = branch.closed and node == branch.num_charts
        # The endpoint's own exact deck already places its root in this
        # chart's frame; the node gauge on this chart side relates that frame
        # to the node's.
        if endpoint.certifies_point_root(matched_root):
            return _SourceNodeGauge(shifts=shifts, closing=closing)
        candidate = BranchRootEndpoint(
            endpoint.root,
            branch,
            endpoint.chart,
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
            affine_transforms=endpoint.affine_transforms,
        )
        if candidate.certifies_point_root(matched_root):
            return _SourceNodeGauge(shifts=shifts, closing=closing)
    raise BRepBooleanFailure(
        "BRepBoolean.branch_alias must retain a source-node gauge; got no matching gauge"
    )


def _transport_fixed_branch_cut(
    other: Arc,
    cut: Cut,
    endpoint: BranchRootEndpoint,
    source_gauge: _SourceNodeGauge,
    vertex: str,
    binding: BRepVertexRoot,
    point: np.ndarray,
) -> Cut:
    if (
        other.branch is None
        or not isinstance(cut.endpoint, TrimRootEndpoint)
        or not isinstance(cut.endpoint.root, IntersectionCurvePointRoot)
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias.incident_endpoints must be fixed branch point roots; got an unrelated endpoint"
        )
    first, last = _arc_parameter_interval(other)
    chart = min(int(last) - 1, max(int(first), int(np.floor(cut.parameter))))
    point_parameter = cut.endpoint.root.parameter
    if not point_parameter.is_integer():
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias.incident_endpoint must be an isolated integer branch node"
        )
    curve = endpoint.curve
    # A closed branch's terminal node is its start root on the closure sheet;
    # gauges are expressed in the frame of the node as seen from this chart.
    target_node = (
        int(cut.parameter)
        if float(cut.parameter).is_integer() and int(cut.parameter) in (chart, chart + 1)
        else int(point_parameter)
    )
    target_gauge = _branch_node_gauge(curve, target_node, chart)
    # Raw shifts exclude the source p-curve's own period gauge, which the
    # transported endpoint re-derives from the same p-curve.
    shifts = (
        np.asarray(endpoint.periodic_shifts, dtype=np.int64)
        + target_gauge
        - source_gauge.shifts
    )
    if curve.closed:
        target_closing = target_node == curve.num_charts
        shifts = shifts + curve.closure_period_shifts() * (
            int(target_closing) - int(source_gauge.closing)
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
    )
    if not transported.certifies_point_root(cut.endpoint.root):
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias.incident_endpoint must certify the same fixed node; got failed point-root certification"
        )
    lower, upper = transported.parameter_enclosure()
    if lower != upper or cut.parameter != lower:
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias.cut.parameter must equal the unique certified fixed branch node"
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
) -> tuple[str, str] | None:
    """Return the node and vertex a source root shares with a fixed branch node."""

    first_owner, _ = state.branch_owners[arc.token]
    columns = slice(0, 2) if arc.owner == first_owner else slice(2, 4)
    patch = state.models[arc.owner[0]].patches[arc.owner[1]]
    for cut in state.cuts[arc.token]:
        existing = cut.endpoint
        if (
            isinstance(existing, BranchRootEndpoint)
            and existing.root.root_id == endpoint.root.root_id
            and existing.curve.branch_id == endpoint.curve.branch_id
            and existing.chart == endpoint.chart
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
            )
        ):
            return cut.node, cut.vertex
    lower, upper = endpoint.parameter_enclosure()
    if lower == upper and float(lower).is_integer():
        for cut in state.cuts[arc.token]:
            existing = cut.endpoint
            # A different source root already certified as the same isolated
            # branch node on this chart sheet names the same vertex.
            if (
                isinstance(existing, BranchRootEndpoint)
                and existing.root.root_id != endpoint.root.root_id
                and existing.curve.branch_id == endpoint.curve.branch_id
                and existing.chart == endpoint.chart
                and cut.lower == cut.upper == cut.parameter == lower
                and np.array_equal(
                    existing.source_gauges()[columns],
                    endpoint.source_gauges()[columns],
                )
            ):
                return cut.node, cut.vertex

    matches: list[_FixedBranchCut] = []
    for cut in state.cuts[arc.token]:
        match cut.endpoint:
            case TrimRootEndpoint(root=IntersectionCurvePointRoot() as root):
                if endpoint.certifies_point_root(root):
                    matches.append(_FixedBranchCut(cut=cut, root=root))
            case _:
                continue
    if not matches:
        return None
    lower, upper = endpoint.parameter_enclosure()
    matched_cut = next(
        (match for match in matches if lower == upper == match.cut.parameter), None
    )
    if matched_cut is None:
        raise BRepBooleanFailure(
            "BRepBoolean.branch_alias must have a matched cut inside its source chart; got no matching cut"
        )
    source_gauge = _certify_source_node_gauge(endpoint, matched_cut.root)
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
                )
                changes.append((other.token, index, transported_cut))
    for token, index, cut in changes:
        state.cuts[token][index] = cut
    return matched_cut.cut.node, vertex
