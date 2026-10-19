#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native standalone curve route: size-driven arc-length interval placement.

Each selected curve ``gamma(t)``, ``t`` in the unit reference interval, receives
the size density ``w(t) = |gamma'(t)| / h(t)``, where ``h`` combines the uniform
and curvature size controls applying to the curve. ``n = ceil(integral w)``
intervals are placed at equal increments of the cumulative density: a
composite Gauss-Legendre table split at source span/chart joins
brackets every target. A native bracketed scalar root
(`phydrax.nonlinear.scalar_root`) solves each cell in its affine coordinate
``[0, 1]``. Every residual is freshly evaluated at the mapped source parameter
against partial/full cell quadrature, without cumulative-prefix subtraction.
A protected curve feature then bisects, in density, every interval whose chord
deviation bound exceeds its bound until the bound or the work budget is
reached. Declared junctions share one vertex; nothing is joined by proximity.

Uniform-only density uses the source's first derivative, not a redundant
curvature/third-derivative program. Placement runs through a stable compiled
native scalar-root callable with dynamic coefficients, bounded single-chart
worksets and charged padding. Heterogeneous source branches do not enter an
unrelated chart's placement program; the quadrature evidence and root status are retained.

Fidelity is certified per interval ``[a, b]`` of the chart parameter: the chord
is the linear interpolant ``L`` of ``gamma`` at ``a`` and ``b``, and
``|gamma(t) - L(t)| <= (t - a)(b - t) / 2 max |gamma''| <= (b - a)^2 / 8 M``
for an outward-rounded interval enclosure ``M`` of ``|gamma''|`` over the
interval, computed by interval evaluation of the chart's second-derivative
program. The bound holds in both directions (every curve point is near its
chord point and conversely), plus the rounding of the published vertices.
Charts whose programs have no interval rule keep sampled evidence.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.extend import core as jax_core

import phydrax.discretization as cell_mesh_owner
import phydrax.discretization as reference_cell_types_owner
import phydrax.geometry as atlas_owner
import phydrax.geometry as regions_owner
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
)
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    CurveMeshingSpec,
    MeshingLimits,
    MeshingProviderInfo,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing.providers._native_route_policy import publication_policy

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._physical import SpatialCoordinateContract
from ..._resource_ticket import BatchTicket
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...geometry._atlas import AbstractBoundaryMap
from ...geometry._domain_numerics import (
    bucketed,
    carrier_breakpoints,
    source_batch_bucket,
)
from ...geometry._domain_preparation import certify_brep_gap_bounds
from ...geometry._meshing_domain import DomainCurveMap, PatchCurveUse, PhysicalCurveMap
from ...geometry.brep._model import BRepGeometry, BRepModel
from ...geometry.implicit._enclosure import (
    evaluate,
    Exact,
    Interval,
    interval,
    unsupported_primitives,
)
from ...nonlinear import (
    NonlinearStatus,
    NonlinearTermination,
    scalar_root,
    ScalarRootProblem,
)
from ...typing import checked
from .._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from .._canonical import canonicalize_cell_mesh
from .._controls import FeatureKind
from .._measurements import measure_phase, NativeMeshingPhaseRecorder
from .._organization_records import MeshLabel, MeshPatch
from .._result import CellMeshingResult, MeshingComplianceReport
from .._sizing import (
    CurvatureSizeControl,
    ProximitySizeControl,
    SizeControlStrength,
    UniformSizeControl,
)
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from . import _native_sources as sources_owner
from ._native_publication import (
    coordinate_budget,
    coordinate_phase,
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from ._native_route_checks import (
    check_deadline,
    edge_size_evidence,
    simplex_entity_limits,
    uniform_size_compliance,
)
from ._native_route_metadata import mesh_entity_scope
from ._native_schedules import NativeCurveSchedule


_ROOT_STEPS = 64
# Junction endpoints must coincide up to the rounding of chart evaluation.
_JUNCTION_ROUNDING = 256.0
# Subintervals per piece enclosed separately to limit interval overestimation.
_ENCLOSURE_SUBDIVISIONS = 4
# Padded piece counts bound the number of compiled enclosure programs.
_ENCLOSURE_BUCKETS = (256, 4096)

# Bounded root-workset widths; padding is included in execution work accounting.
_PLACEMENT_BUCKETS = (1, 2, 4, 8, 16, 32, 128, 512)


@final
class _AccelerationEnclosure(StrictModule, NonTrainableState):
    """Interval evaluation of ``gamma''`` of the selected curves over intervals.

    ``rigorous`` is false when the chart program uses a primitive without an
    outward-rounded interval rule; its enclosures are then not produced.
    """

    consts: tuple[Array, ...]
    jaxpr: jax_core.Jaxpr = eqx.field(static=True)
    rigorous: bool = eqx.field(static=True)

    def __init__(self, charts: CurveCharts, /) -> None:
        def acceleration(row: Array, parameter: Array) -> Array:
            return charts.jet(row, parameter)[2]

        closed = jax.make_jaxpr(acceleration)(
            jax.ShapeDtypeStruct((), np.int32), jax.ShapeDtypeStruct((), np.float64)
        )
        self.consts = jax.device_put(tuple(np.asarray(value) for value in closed.consts))
        self.jaxpr = closed.jaxpr
        self.rigorous = not unsupported_primitives(closed.jaxpr)

    def enclose(self, row: Array, lower: Array, upper: Array, /) -> Array:
        """Upper bound of ``|gamma''|`` of curve ``row`` over ``[lower, upper]``."""

        seeds = jnp.ones((1,), dtype=jnp.float64)
        (value,) = evaluate(
            self.jaxpr,
            self.consts,
            [Exact(row), Interval(lower, upper, seeds, seeds)],
            1,
        )
        result = interval(value, 1)
        magnitude = jnp.maximum(jnp.abs(result.lower), jnp.abs(result.upper))
        # Outward rounding of the norm: one ulp per operation suffices.
        return jnp.nextafter(jnp.sqrt(jnp.sum(magnitude * magnitude)), jnp.inf)


@eqx.filter_jit
def _enclose_accelerations(
    enclosure: _AccelerationEnclosure, rows: Array, lower: Array, upper: Array
) -> Array:
    return jax.vmap(enclosure.enclose)(rows, lower, upper)


@final
class CurveCharts(StrictModule, NonTrainableState):
    """Batched evaluation of the selected source curves on ``[0, 1]``."""

    mapping: AbstractBoundaryMap | None
    segments: Array | None
    chart_rows: Array

    @checked
    def __init__(
        self,
        curves: atlas_owner.BoundaryAtlas | regions_owner.SegmentMesh,
        chart_rows: Array,
    ) -> None:
        match curves:
            case atlas_owner.BoundaryAtlas():
                self.mapping = curves.mapping
                self.segments = None
            case regions_owner.SegmentMesh():
                self.mapping = None
                self.segments = curves.segments
            case _:
                raise TypeError(
                    f"CurveCharts.curves must be BoundaryAtlas or SegmentMesh; got {type(curves).__name__}"
                )
        self.chart_rows = jax.device_put(
            np.asarray(jax.device_get(chart_rows), dtype=np.int32)
        )

    def point(self, curve: Array, parameter: Array, /) -> Array:
        rows = self.chart_rows[curve]
        if self.mapping is not None:
            return self.mapping.map(rows, parameter[..., None])
        if self.segments is None:
            raise RuntimeError(
                "CurveCharts.source must provide a mapping or segments; got neither"
            )
        segment = self.segments[rows]
        return segment[..., 0, :] + parameter[..., None] * (
            segment[..., 1, :] - segment[..., 0, :]
        )

    def jet(self, curve: Array, parameter: Array, /) -> tuple[Array, Array, Array]:
        """Points with first and second parameter derivatives, elementwise."""

        ones = jnp.ones_like(parameter)

        def first(value: Array, /) -> Array:
            return jax.jvp(lambda t: self.point(curve, t), (value,), (ones,))[1]

        point, velocity = jax.jvp(lambda t: self.point(curve, t), (parameter,), (ones,))
        _, acceleration = jax.jvp(first, (parameter,), (ones,))
        return point, velocity, acceleration

    def velocity(self, curve: Array, parameter: Array, /) -> Array:
        return jax.jvp(
            lambda value: self.point(curve, value),
            (parameter,),
            (jnp.ones_like(parameter),),
        )[1]

    def breakpoints(self, row: int, /) -> np.ndarray:
        """Source span/chart cuts in the normalized curve coordinate."""
        source_row = int(np.asarray(jax.device_get(self.chart_rows))[row])
        mapping = self.mapping
        match mapping:
            case PhysicalCurveMap():
                carrier = mapping.curves[source_row]
                first, last = np.asarray(jax.device_get(mapping.ranges))[source_row]
            case DomainCurveMap():
                carrier = mapping.pcurves[source_row]
                first, last = (
                    float(np.asarray(jax.device_get(mapping.firsts))[source_row]),
                    float(np.asarray(jax.device_get(mapping.lasts))[source_row]),
                )
            case AbstractBoundaryMap() | None:
                return np.empty((0,), dtype=np.float64)
            case _:
                raise TypeError(
                    "CurveCharts.mapping must be a boundary map or None; got an unsupported evaluator"
                )
        cuts = carrier_breakpoints(
            carrier, float(min(first, last)), float(max(first, last))
        )
        return np.sort((cuts - first) / (last - first))

    def select(self, row: int, /) -> CurveCharts:
        """Select a homogeneous chart with host-prepared row metadata."""
        host_rows = np.asarray(jax.device_get(self.chart_rows))
        source_row = int(host_rows[row])
        match self.mapping:
            case DomainCurveMap() | PhysicalCurveMap():
                return eqx.tree_at(
                    lambda value: (value.mapping, value.chart_rows),
                    self,
                    (
                        self.mapping.select_chart(source_row),
                        jax.device_put(np.zeros((1,), dtype=np.int32)),
                    ),
                )
            case AbstractBoundaryMap():
                return eqx.tree_at(
                    lambda value: value.chart_rows,
                    self,
                    jax.device_put(host_rows[row : row + 1]),
                )
            case None:
                segments = self.segments
                if segments is None:
                    raise RuntimeError(
                        "CurveCharts.source must provide a mapping or segments; got neither"
                    )
                return eqx.tree_at(
                    lambda value: (value.segments, value.chart_rows),
                    self,
                    (
                        jax.device_put(
                            np.asarray(jax.device_get(segments))[
                                source_row : source_row + 1
                            ]
                        ),
                        jax.device_put(np.zeros((1,), dtype=np.int32)),
                    ),
                )
            case _:
                raise TypeError(
                    "CurveCharts.mapping must be a boundary map or None; got an unsupported evaluator"
                )


@final
class CurveSizing(StrictModule, NonTrainableState):
    """Per-curve size parameters combined as ``max(lower, min(u, clip(a/k)))``."""

    uniform: Array
    normal_angle: Array
    curvature_minimum: Array
    curvature_maximum: Array
    lower: Array
    requires_curvature: bool = eqx.field(static=True)

    def __init__(
        self,
        uniform: np.ndarray,
        normal_angle: np.ndarray,
        curvature_minimum: np.ndarray,
        curvature_maximum: np.ndarray,
        lower: np.ndarray,
    ) -> None:
        self.uniform = jax.device_put(np.asarray(uniform, dtype=np.float64))
        self.normal_angle = jax.device_put(np.asarray(normal_angle, dtype=np.float64))
        self.curvature_minimum = jax.device_put(
            np.asarray(curvature_minimum, dtype=np.float64)
        )
        self.curvature_maximum = jax.device_put(
            np.asarray(curvature_maximum, dtype=np.float64)
        )
        self.lower = jax.device_put(np.asarray(lower, dtype=np.float64))
        self.requires_curvature = bool(np.any(np.isfinite(normal_angle)))

    def select(self, row: int, /) -> CurveSizing:
        """Select one host-prepared row and transfer its numerical controls once."""
        controls = jax.device_get(
            (
                self.uniform,
                self.normal_angle,
                self.curvature_minimum,
                self.curvature_maximum,
                self.lower,
            )
        )
        selected = tuple(
            jax.device_put(np.asarray(value)[row : row + 1]) for value in controls
        )
        return eqx.tree_at(
            lambda value: (
                value.uniform,
                value.normal_angle,
                value.curvature_minimum,
                value.curvature_maximum,
                value.lower,
            ),
            self,
            selected,
        )

    def density(
        self, charts: CurveCharts, curve: Array, parameter: Array, /
    ) -> tuple[Array, Array]:
        """Size density ``|gamma'| / h`` and the speed ``|gamma'|``."""

        if not self.requires_curvature:
            speed = jnp.linalg.norm(charts.velocity(curve, parameter), axis=-1)
            size = jnp.maximum(self.lower[curve], self.uniform[curve])
            return speed / size, speed
        _, velocity, acceleration = charts.jet(curve, parameter)
        speed = jnp.linalg.norm(velocity, axis=-1)
        cross = jnp.sqrt(
            jnp.maximum(
                jnp.sum(velocity * velocity, axis=-1)
                * jnp.sum(acceleration * acceleration, axis=-1)
                - jnp.sum(velocity * acceleration, axis=-1) ** 2,
                0.0,
            )
        )
        curvature = cross / jnp.maximum(speed**3, jnp.finfo(jnp.float64).tiny)
        curvature_size = jnp.clip(
            self.normal_angle[curve] / curvature,
            self.curvature_minimum[curve],
            self.curvature_maximum[curve],
        )
        size = jnp.maximum(
            self.lower[curve], jnp.minimum(self.uniform[curve], curvature_size)
        )
        return speed / size, speed


@checked
def curve_support_issues(
    source: sources_owner.NativeCurveSource, specification: CurveMeshingSpec, /
) -> list[str]:
    """Physical requests of one curve specification this route cannot enforce."""

    unsupported: list[str] = []
    target = specification.target
    if target.geometry_order != 1:
        unsupported.append("affine interval geometry")
    if target.ambient_dimension != source.ambient_dimension:
        unsupported.append("the ambient dimension of the source curves")
    curves = np.asarray(specification.scope.entity_ids, dtype=np.int64)
    available = _curve_ids(source)
    if np.setdiff1d(curves, available).size:
        unsupported.append("curves that exist in the source")
    covered = np.zeros(curves.shape, dtype=np.bool_)
    for control in specification.size_controls:
        match control:
            case UniformSizeControl() | CurvatureSizeControl():
                covered |= np.isin(curves, np.asarray(control.scope.entity_ids))
            case _:
                unsupported.append("proximity size controls on curves")
    if not np.all(covered):
        unsupported.append("a size control on every meshed curve")
    for feature in specification.protected_features:
        if feature.feature_kind is not FeatureKind.CURVE:
            unsupported.append(f"{feature.feature_kind.value} protected features")
        elif np.setdiff1d(np.asarray(feature.scope.entity_ids), curves).size:
            unsupported.append("protected curves outside the meshing scope")
    return unsupported


def _curve_ids(source: sources_owner.NativeCurveSource, /) -> np.ndarray:
    """Curve identities: atlas chart indices or segment edge rows."""

    match source.curves:
        case atlas_owner.BoundaryAtlas():
            return np.arange(source.curves.num_charts, dtype=np.int64)
        case regions_owner.SegmentMesh():
            return np.arange(source.curves.edges.shape[0], dtype=np.int64)
        case _:
            raise TypeError(
                "source.curves must be BoundaryAtlas or SegmentMesh; got an unsupported evaluator"
            )


def _curve_control(
    control: UniformSizeControl | CurvatureSizeControl | ProximitySizeControl, /
) -> UniformSizeControl | CurvatureSizeControl:
    match control:
        case UniformSizeControl() | CurvatureSizeControl():
            return control
        case _:
            raise TypeError(
                "specification.size_controls must contain uniform or curvature controls; got an unsupported control"
            )


def _sizing(
    specification: CurveMeshingSpec, curves: np.ndarray, /, *, ticket: BatchTicket
) -> CurveSizing:
    count = curves.size
    ticket.consume(count)
    uniform = np.full((count,), np.inf)
    angle = np.full((count,), np.inf)
    curvature_minimum = np.zeros((count,))
    curvature_maximum = np.full((count,), np.inf)
    lower = np.zeros((count,))
    for control in map(_curve_control, specification.size_controls):
        ticket.consume(count)
        applies = np.isin(curves, np.asarray(control.scope.entity_ids))
        match control:
            case UniformSizeControl():
                size = control.target_size
                if control.maximum_size is not None:
                    size = min(size, control.maximum_size)
                uniform[applies] = np.minimum(uniform[applies], size)
            case CurvatureSizeControl():
                angle[applies] = np.minimum(angle[applies], control.normal_angle)
                if control.minimum_size is not None:
                    curvature_minimum[applies] = np.maximum(
                        curvature_minimum[applies], control.minimum_size
                    )
                if control.maximum_size is not None:
                    curvature_maximum[applies] = np.minimum(
                        curvature_maximum[applies], control.maximum_size
                    )
            case _:
                raise TypeError(
                    "specification.size_controls must contain uniform or curvature controls; got an unsupported control"
                )
        if control.minimum_size is not None:
            lower[applies] = np.maximum(lower[applies], control.minimum_size)
    return CurveSizing(uniform, angle, curvature_minimum, curvature_maximum, lower)


def _endpoint_vertices(
    specification: CurveMeshingSpec, curves: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Junction index of each curve start and end (-1 for a free end)."""

    position = {int(curve): row for row, curve in enumerate(curves.tolist())}
    endpoints = np.full((curves.size, 2), -1, dtype=np.int64)
    for index, junction in enumerate(specification.junctions):
        for curve, end in junction.endpoints:
            endpoints[position[curve], 0 if end == "start" else 1] = index
    closed = (endpoints[:, 0] >= 0) & (endpoints[:, 0] == endpoints[:, 1])
    # Only parallel curves sharing the same two junctions need an interior
    # vertex to keep their edges distinct. A unique junction pair needs one
    # interval; its density integral determines any additional intervals.
    paired = np.all(endpoints >= 0, axis=1) & ~closed
    pairs = np.sort(endpoints[paired], axis=1)
    _, inverse, multiplicity = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True
    )
    minimum = np.ones((curves.size,), dtype=np.int64)
    minimum[paired] = np.where(multiplicity[inverse] > 1, 2, 1)
    # A closed curve needs three intervals to form a nondegenerate loop.
    minimum[closed] = 3
    return endpoints, minimum


@checked
def curve_preparation_upper(specification: CurveMeshingSpec, /) -> tuple[int, int]:
    """Return structural work and source-query bounds for preparation.

    Each selected curve evaluates two endpoints in the source's padded bucket;
    every executed lane counts as work. Acceleration-enclosure program
    preparation counts as one work unit. Sizing counts one initialization row
    per curve and one curve/control visit for every size control. Source-query
    evidence retains the analytic preparation policy of zero queries.
    """
    count = len(specification.scope.entity_ids)
    work = count * (source_batch_bucket(2) + 1 + len(specification.size_controls)) + 1
    return work, 0


def _certified_junction_gaps(
    source: sources_owner.NativeCurveSource,
    specification: CurveMeshingSpec,
    curves: np.ndarray,
    endpoints: np.ndarray,
    /,
    *,
    ticket: BatchTicket,
    budget: CoordinateEnclosureBudget | None,
) -> np.ndarray | None:
    """Transport proved endpoint gaps through explicit native corner incidence."""
    domain = source.gap_domain
    if domain is None:
        return None
    match domain.brep_authority:
        case BRepGeometry() as geometry:
            pass
        case BRepModel() as model:
            geometry = model.require_geometry()
        case _:
            raise ValueError(
                "NativeCurveSource.gap_domain must retain a native BRep source; got missing source"
            )
    if budget is None:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=ticket.work_upper,
            maximum_memory_bytes=ticket.bytes_upper,
        )
    certify_brep_gap_bounds(
        geometry, tuple(patch.surface for patch in domain.patches), budget=budget
    )
    budget.reserve(
        endpoints.size + len(specification.junctions),
        32 * endpoints.size + 16 * len(specification.junctions),
    )
    errors = np.zeros(endpoints.shape, dtype=np.float64)
    corners = np.empty(endpoints.shape, dtype=np.int64)
    for row, chart in enumerate(curves):
        edge = domain.source_indices[1][chart]
        patch, loop, position = domain.curve_owners[chart]
        use = domain.patches[patch].loops[loop][position]
        if not isinstance(use, PatchCurveUse):
            raise ValueError(
                "NativeCurveSource.gap_domain must map charts to curve uses; got pole use"
            )
        bounds = (
            (0.0, 0.0)
            if geometry.edge_vertex_gap_bounds is None
            else geometry.edge_vertex_gap_bounds[edge]
        )
        errors[row] = bounds if use.first < use.last else bounds[::-1]
        curve = domain.curves[chart]
        corners[row] = (curve.start, curve.end)
    result = np.zeros((len(specification.junctions),), dtype=np.float64)
    for index in range(result.size):
        members = errors[endpoints == index]
        vertices = corners[endpoints == index]
        if np.any(vertices != vertices[0]):
            raise ValueError(
                "NativeCurveSource.gap_domain must bind each junction to one corner; got distinct native vertices"
            )
        upper = max((members[0] + bound for bound in members[1:]), default=0.0)
        result[index] = np.nextafter(upper, np.inf) if upper > 0.0 else 0.0
    return result


@final
class PreparedCurveNetwork(StrictModule, NonTrainableState):
    """Admitted curves, size parameters, and legal declared junctions.

    The caller reserves ``curve_preparation_upper(specification)`` before
    construction and settles its explicit ticket, including on failure.
    Preparation evidence counts endpoint evaluation lanes, enclosure-program
    preparation, and sizing rows under that structural work policy.
    """

    charts: CurveCharts
    sizing: CurveSizing
    curves: np.ndarray
    endpoints: np.ndarray
    minimum_pieces: np.ndarray
    junction_points: np.ndarray
    junction_gap: float = eqx.field(static=True)
    junction_gap_bounds: np.ndarray | None
    acceleration: _AccelerationEnclosure
    schedule: NativeCurveSchedule
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativeCurveSource,
        specification: CurveMeshingSpec,
        schedule: NativeCurveSchedule,
        /,
        *,
        ticket: BatchTicket,
        gap_budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        curves = np.asarray(specification.scope.entity_ids, dtype=np.int64)
        # Curve identities are chart/edge rows, so the scope selects rows.
        charts = CurveCharts(
            source.curves, jax.device_put(np.asarray(curves, dtype=np.int32))
        )
        endpoints, minimum = _endpoint_vertices(specification, curves)
        certified_gaps = _certified_junction_gaps(
            source, specification, curves, endpoints, ticket=ticket, budget=gap_budget
        )
        endpoint_points = []
        for row in range(curves.size):
            ticket.consume(source_batch_bucket(2))
            endpoint_points.append(curve_points(charts, row, np.asarray((0.0, 1.0))))
        ends = np.stack(endpoint_points)
        scale = max(float(np.max(np.abs(ends))), 1.0)
        tolerance = _JUNCTION_ROUNDING * np.finfo(np.float64).eps * scale
        junction_points = np.zeros((len(specification.junctions), ends.shape[2]))
        gap = 0.0
        for index in range(len(specification.junctions)):
            members = ends[endpoints == index]
            junction_points[index] = members[0]
            spread = float(np.max(np.linalg.norm(members - members[0], axis=1)))
            certified_gap = None if certified_gaps is None else certified_gaps[index]
            gap = max(gap, spread if certified_gap is None else float(certified_gap))
            allowed = (
                tolerance
                if certified_gap is None or certified_gap == 0.0
                else float(np.nextafter(tolerance + certified_gap, np.inf))
            )
            if spread > allowed:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    f"specification.junctions must join coincident source endpoints; got "
                    f"{specification.junctions[index].name!r} gap {spread} > {allowed}",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                    entity_ids=tuple(
                        int(curve)
                        for curve, _ in specification.junctions[index].endpoints
                    ),
                    requested=(("junction_gap", allowed),),
                    achieved=(("junction_gap", spread),),
                )
        ticket.consume(1)
        acceleration = _AccelerationEnclosure(charts)
        sizing = _sizing(specification, curves, ticket=ticket)
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-curve-network",
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
                "curves": array_tree_fingerprint(curves),
                "endpoints": array_tree_fingerprint(endpoints),
                "junction_points": array_tree_fingerprint(junction_points),
                **(
                    {
                        "gap_authority": source.gap_domain.domain_id,
                        "junction_gap_bounds": array_tree_fingerprint(certified_gaps),
                    }
                    if source.gap_domain is not None and certified_gaps is not None
                    else {}
                ),
            }
        )
        self.charts = charts
        self.acceleration = acceleration
        self.sizing = sizing
        self.curves = curves
        self.endpoints = endpoints
        self.minimum_pieces = minimum
        self.junction_points = junction_points
        self.junction_gap = gap
        self.junction_gap_bounds = certified_gaps
        self.schedule = schedule
        self.prepared_id = prepared_id


class QueryLedger:
    """Host accounting of source-geometry queries and native work units."""

    __slots__ = ("limits", "queries", "work", "budget")

    def __init__(
        self, limits: MeshingLimits, /, *, budget: CoordinateEnclosureBudget
    ) -> None:
        self.limits = limits
        self.queries = 0
        self.work = 0
        self.budget = budget

    def _record(self, queries: int, work: int, stage: MeshingStageKind, /) -> None:
        if (
            self.queries + queries > self.limits.maximum_geometry_queries
            or self.work + work > self.limits.maximum_work_units
        ):
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "curve.discretization must respect limits.maximum_geometry_queries and limits.maximum_work_units; got excess work",
                stage=stage.value,
                requested=(
                    ("maximum_geometry_queries", self.limits.maximum_geometry_queries),
                    ("maximum_work_units", self.limits.maximum_work_units),
                ),
                achieved=(
                    ("geometry_queries", self.queries + queries),
                    ("work_units", self.work + work),
                ),
            )
        self.queries += queries
        self.work += work

    def reserve(self, queries: int, work: int, stage: MeshingStageKind, /) -> None:
        """Record source-query and work proposals under the scientific policy."""
        self._record(queries, work, stage)

    def observe_exact(self, work: int, stage: MeshingStageKind, /) -> None:
        """Record scientific work already counted by the coordinate view."""
        self._record(0, work, stage)

    def observe_native(self, queries: int, work: int, stage: MeshingStageKind, /) -> None:
        """Record scientific counters already charged by native execution."""
        self._record(queries, work, stage)


@eqx.filter_jit
def _source_density(
    charts: CurveCharts, sizing: CurveSizing, curves: Array, parameters: Array, /
) -> tuple[Array, Array]:
    """Stable batched source velocity/jet lowering with dynamic numerical leaves."""
    return sizing.density(charts, curves, parameters)


@eqx.filter_jit
def _source_points(charts: CurveCharts, parameters: Array, /) -> Array:
    """Source points of a one-row chart selection, coordinates kept dynamic."""
    return charts.point(
        jax.device_put(np.zeros(parameters.shape, dtype=np.int32)), parameters
    )


@checked
def curve_points(charts: CurveCharts, row: int, parameters: np.ndarray, /) -> np.ndarray:
    """Nonempty source points of one chart row through a bounded batch bucket.

    One compiled program serves each chart structure, trailing parameter shape
    and power-of-two count of leading rows; padding rows repeat the first
    admitted row and are sliced off.
    """
    padded = bucketed(np.asarray(parameters, dtype=np.float64))
    return np.asarray(
        jax.device_get(_source_points(charts.select(row), padded)), dtype=np.float64
    )[: parameters.shape[0]]


def _chart_points(
    prepared: PreparedCurveNetwork,
    rows: np.ndarray,
    parameters: np.ndarray,
    /,
) -> np.ndarray:
    """Evaluate each source row without tracing unrelated chart branches."""
    points = np.empty(
        parameters.shape + (prepared.junction_points.shape[-1],), dtype=np.float64
    )
    for source_row in np.unique(rows):
        selected = np.flatnonzero(rows == source_row)
        points[selected] = curve_points(
            prepared.charts, int(source_row), parameters[selected]
        )
    return points


def _density_table(
    prepared: PreparedCurveNetwork, ledger: QueryLedger, /
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cumulative density at the breakpoints and the table error estimate."""

    schedule = prepared.schedule
    nodes, weights = np.polynomial.legendre.leggauss(schedule.quadrature_order)
    fine_breaks = []
    fine_tables = []
    error = 0.0
    for row in range(prepared.curves.size):
        charts, sizing = prepared.charts.select(row), prepared.sizing.select(row)
        cuts = prepared.charts.breakpoints(row)

        def integrals(intervals: int, /) -> tuple[np.ndarray, np.ndarray]:
            # A source C0 join changes parameter speed. Integrating across it
            # makes moving Gauss nodes create jumps in the placement residual.
            breaks = np.unique(
                np.concatenate((np.linspace(0.0, 1.0, intervals + 1), cuts))
            )
            ledger.reserve(
                (breaks.size - 1) * nodes.size, 0, MeshingStageKind.CURVE_MESHING
            )
            half = 0.5 * np.diff(breaks)
            parameters = breaks[:-1, None] + half[:, None] * (nodes[None, :] + 1.0)
            values = np.asarray(
                jax.device_get(
                    _source_density(
                        charts,
                        sizing,
                        jax.device_put(np.zeros(parameters.shape, dtype=np.int32)),
                        jax.device_put(np.asarray(parameters, dtype=np.float64)),
                    )[0]
                )
            )
            pieces = np.sum(values * weights, axis=1) * half
            return breaks, np.concatenate(
                (np.zeros((1,), dtype=np.float64), np.cumsum(pieces))
            )

        _, coarse = integrals(schedule.quadrature_intervals)
        breaks, fine = integrals(2 * schedule.quadrature_intervals)
        error = max(error, abs(float(fine[-1] - coarse[-1])))
        fine_breaks.append(breaks)
        fine_tables.append(fine)
    width = max(breaks.size for breaks in fine_breaks)
    return (
        np.stack(
            [
                np.pad(breaks, (0, width - breaks.size), mode="edge")
                for breaks in fine_breaks
            ]
        ),
        np.stack(
            [np.pad(table, (0, width - table.size), mode="edge") for table in fine_tables]
        ),
        error,
    )


def _placement_residual(
    coordinate: Array,
    args: tuple[
        CurveCharts, CurveSizing, Array, Array, Array, Array, Array, Array, Array
    ],
    /,
) -> Array:
    """Evaluate the density residual at the mapped global source parameter."""
    charts, sizing, chart, start, stop, mass, fraction, nodes, weights = args
    parameter = start + (stop - start) * coordinate
    half = 0.5 * (parameter - start)
    points = start + half * (nodes + 1.0)
    values, _ = sizing.density(charts, jnp.broadcast_to(chart, points.shape), points)
    return half * jnp.sum(weights * values) - fraction * mass


@eqx.filter_jit
def placement_roots(
    charts: CurveCharts,
    sizing: CurveSizing,
    lower: Array,
    upper: Array,
    fractions: Array,
    nodes: Array,
    weights: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Solve cell coordinates and return global parameters, native statuses and residuals."""
    termination = NonlinearTermination(
        absolute_residual=1.0e-12, maximum_steps=_ROOT_STEPS
    )

    def solve(start: Array, stop: Array, fraction: Array) -> tuple[Array, Array, Array]:
        half = 0.5 * (stop - start)
        points = start + half * (nodes + 1.0)
        chart = jnp.zeros((), dtype=jnp.int32)
        values, _ = sizing.density(charts, jnp.broadcast_to(chart, points.shape), points)
        mass = half * jnp.sum(weights * values)
        # The step threshold acts on this cell's coordinate, not its global
        # parameter origin; the same residual criterion certifies source placement.
        result = scalar_root(
            ScalarRootProblem(
                _placement_residual,
                bracket=(
                    jnp.zeros((), dtype=jnp.float64),
                    jnp.ones((), dtype=jnp.float64),
                ),
                problem_id="native-curve-density-placement",
            ),
            termination=termination,
            args=(charts, sizing, chart, start, stop, mass, fraction, nodes, weights),
        )
        return start + (stop - start) * result.root, result.status, result.value

    return jax.vmap(solve)(lower, upper, fractions)


def _place(
    prepared: PreparedCurveNetwork,
    breaks: np.ndarray,
    table: np.ndarray,
    curve: np.ndarray,
    targets: np.ndarray,
    ledger: QueryLedger,
    /,
) -> np.ndarray:
    """Parameters at which each curve's cumulative density reaches its target."""

    if curve.size == 0:
        return np.zeros((0,), dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(prepared.schedule.quadrature_order)
    # Each compiled workset contains one source chart. Padding contributes real
    # quadrature/root work and is charged before its arrays are allocated.
    # Cumulative density tables are nondecreasing, so the bracketing interval
    # is the count of breakpoints at or below the target (bounded K + 1 work).
    interval = np.clip(
        np.sum(table[curve] <= targets[:, None], axis=1, dtype=np.int64) - 1,
        0,
        breaks.shape[1] - 2,
    )
    roots = np.empty(targets.shape, dtype=np.float64)
    statuses = np.empty(targets.shape, dtype=np.int32)
    values = np.empty(targets.shape, dtype=np.float64)
    nodes_, weights_ = jax.device_put(nodes), jax.device_put(weights)
    host_sizing = jax.device_get(
        (
            prepared.sizing.uniform,
            prepared.sizing.normal_angle,
            prepared.sizing.curvature_minimum,
            prepared.sizing.curvature_maximum,
            prepared.sizing.lower,
        )
    )
    for source_row in np.unique(curve):
        selected = np.flatnonzero(curve == source_row)
        charts = prepared.charts.select(int(source_row))
        row = slice(int(source_row), int(source_row) + 1)
        sizing = CurveSizing(
            np.asarray(host_sizing[0][row]),
            np.asarray(host_sizing[1][row]),
            np.asarray(host_sizing[2][row]),
            np.asarray(host_sizing[3][row]),
            np.asarray(host_sizing[4][row]),
        )
        for start in range(0, selected.size, _PLACEMENT_BUCKETS[-1]):
            positions = selected[start : start + _PLACEMENT_BUCKETS[-1]]
            width = next(value for value in _PLACEMENT_BUCKETS if positions.size <= value)
            # TOMS748 evaluates twice per cycle, plus both bracket endpoints,
            # the final certificate, and one fresh full-cell integral.
            ledger.reserve(
                width * (2 * _ROOT_STEPS + 4) * nodes.size,
                width,
                MeshingStageKind.CURVE_MESHING,
            )
            padding = (0, width - positions.size)
            interval_rows = interval[positions]
            prefixes = table[curve[positions], interval_rows]
            masses = table[curve[positions], interval_rows + 1] - prefixes
            fractions = (targets[positions] - prefixes) / masses
            output = placement_roots(
                charts,
                sizing,
                jax.device_put(
                    np.pad(breaks[curve[positions], interval_rows], padding, mode="edge")
                ),
                jax.device_put(
                    np.pad(
                        breaks[curve[positions], interval_rows + 1], padding, mode="edge"
                    )
                ),
                jax.device_put(np.pad(fractions, padding, mode="edge")),
                nodes_,
                weights_,
            )
            roots[positions], statuses[positions], values[positions] = (
                np.asarray(value)[: positions.size] for value in jax.device_get(output)
            )
    successful_ = statuses == int(NonlinearStatus.SUCCESS)
    if not np.all(successful_):
        failed = np.unique(prepared.curves[curve[~successful_]])
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "curve.placement_roots must converge; got failed roots",
            stage=MeshingStageKind.CURVE_MESHING.value,
            entity_ids=tuple(int(value) for value in failed),
            achieved=(
                ("maximum_placement_residual", float(np.max(np.abs(np.asarray(values))))),
            )
            + tuple(
                (
                    f"placement_status_{NonlinearStatus(int(status)).name}",
                    float(np.sum(statuses == status)),
                )
                for status in np.unique(statuses[~successful_])
            ),
        )
    return np.asarray(roots, dtype=np.float64)


def _chord_deviation(
    prepared: PreparedCurveNetwork,
    curve: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    ledger: QueryLedger,
    /,
) -> np.ndarray:
    """Largest sampled distance of each curve piece from its chord (witnessed)."""

    samples = prepared.schedule.fidelity_samples
    ledger.reserve(curve.size * (samples + 2), 0, MeshingStageKind.CURVE_MESHING)
    fractions = np.arange(1, samples + 1, dtype=np.float64) / (samples + 1)
    parameters = np.concatenate(
        (
            start[:, None],
            start[:, None] + fractions[None, :] * (end - start)[:, None],
            end[:, None],
        ),
        axis=1,
    )
    points = _chart_points(prepared, curve, parameters)
    first = points[:, :1]
    chord = points[:, -1:] - first
    offsets = points[:, 1:-1] - first
    length = np.maximum(np.sum(chord * chord, axis=2), np.finfo(np.float64).tiny)
    along = np.clip(np.sum(offsets * chord, axis=2) / length, 0.0, 1.0)
    distance = np.linalg.norm(offsets - along[..., None] * chord, axis=2)
    return np.max(distance, axis=1)


def _chord_bound(
    prepared: PreparedCurveNetwork,
    curve: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    ledger: QueryLedger,
    /,
) -> tuple[np.ndarray, bool]:
    """Two-sided chord deviation of each piece: certified bound or sampled value.

    Returns the per-piece deviation and whether it is a certified upper bound.
    The certified bound adds the published-vertex rounding: junction vertices
    are one member endpoint, the others lie within the admitted junction gap.
    """

    if not prepared.acceleration.rigorous or curve.size == 0:
        return _chord_deviation(prepared, curve, start, end, ledger), False
    count = curve.size * _ENCLOSURE_SUBDIVISIONS
    ledger.reserve(count, 0, MeshingStageKind.CURVE_MESHING)
    fractions = np.arange(_ENCLOSURE_SUBDIVISIONS + 1) / _ENCLOSURE_SUBDIVISIONS
    breaks = start[:, None] + fractions[None, :] * (end - start)[:, None]
    # Subinterval ends are widened to cover the rounding of the breakpoints.
    lower = np.nextafter(breaks[:, :-1], -np.inf).reshape(-1)
    upper = np.nextafter(breaks[:, 1:], np.inf).reshape(-1)
    rows = np.repeat(curve, _ENCLOSURE_SUBDIVISIONS)
    bucket = next(
        (value for value in _ENCLOSURE_BUCKETS if count <= value),
        _ENCLOSURE_BUCKETS[-1] * -(-count // _ENCLOSURE_BUCKETS[-1]),
    )
    padding = (0, bucket - count)
    enclosed = np.asarray(
        jax.device_get(
            _enclose_accelerations(
                prepared.acceleration,
                jax.device_put(np.pad(rows, padding, mode="edge").astype(np.int32)),
                jax.device_put(np.pad(lower, padding, mode="edge")),
                jax.device_put(np.pad(upper, padding, mode="edge")),
            )
        ),
        dtype=np.float64,
    )[:count]
    maximum = np.max(enclosed.reshape(curve.size, _ENCLOSURE_SUBDIVISIONS), axis=1)
    if not np.all(np.isfinite(maximum)):
        return _chord_deviation(prepared, curve, start, end, ledger), False
    scale = max(float(np.max(np.abs(prepared.junction_points), initial=1.0)), 1.0)
    rounding = prepared.junction_gap + _JUNCTION_ROUNDING * np.finfo(np.float64).eps * (
        scale
    )
    width = end - start
    bound = 0.125 * width * width * maximum * (1.0 + 8.0 * np.finfo(np.float64).eps)
    return bound + rounding, True


def _deviation_bounds(
    specification: CurveMeshingSpec, curves: np.ndarray, scale: float, /
) -> np.ndarray:
    bounds = np.full((curves.size,), np.inf)
    rounding = _JUNCTION_ROUNDING * np.finfo(np.float64).eps * scale
    for feature in specification.protected_features:
        applies = np.isin(curves, np.asarray(feature.scope.entity_ids))
        bounds[applies] = np.minimum(
            bounds[applies], feature.maximum_deviation + rounding
        )
    return bounds


def discretize(
    prepared: PreparedCurveNetwork,
    specification: CurveMeshingSpec,
    ledger: QueryLedger,
    started: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Parameters, their curve rows, cumulative densities, and the table error."""

    breaks, table, error = _density_table(prepared, ledger)
    totals = table[:, -1]
    counts = np.maximum(
        np.ceil(totals * (1.0 - 64.0 * np.finfo(np.float64).eps)).astype(np.int64),
        prepared.minimum_pieces,
    )
    curve = np.repeat(np.arange(prepared.curves.size), counts - 1)
    offsets = np.concatenate(([0], np.cumsum(counts - 1)))
    step = np.arange(curve.size) - offsets[curve] + 1
    targets = totals[curve] * step / counts[curve]
    interior = _place(prepared, breaks, table, curve, targets, ledger)
    rows = np.concatenate((np.arange(prepared.curves.size),) * 2 + (curve,))
    parameters = np.concatenate(
        (np.zeros(prepared.curves.size), np.ones(prepared.curves.size), interior)
    )
    density = np.concatenate((np.zeros(prepared.curves.size), totals, targets))
    scale = max(float(np.max(np.abs(prepared.junction_points), initial=1.0)), 1.0)
    bounds = _deviation_bounds(specification, prepared.curves, scale)
    while True:
        check_deadline(started, specification.limits, MeshingStageKind.CURVE_MESHING)
        order = np.lexsort((parameters, rows))
        rows, parameters, density = rows[order], parameters[order], density[order]
        same = rows[1:] == rows[:-1]
        piece_rows = rows[:-1][same]
        guarded = np.isfinite(bounds[piece_rows])
        if not np.any(guarded):
            break
        start = parameters[:-1][same][guarded]
        end = parameters[1:][same][guarded]
        deviation, _ = _chord_bound(prepared, piece_rows[guarded], start, end, ledger)
        violated = deviation > bounds[piece_rows[guarded]]
        if not np.any(violated):
            break
        lower_density = density[:-1][same][guarded][violated]
        upper_density = density[1:][same][guarded][violated]
        split_rows = piece_rows[guarded][violated]
        midpoint = 0.5 * (lower_density + upper_density)
        inserted = _place(prepared, breaks, table, split_rows, midpoint, ledger)
        rows = np.concatenate((rows, split_rows))
        parameters = np.concatenate((parameters, inserted))
        density = np.concatenate((density, midpoint))
    return rows, parameters, density, error


def _assemble(
    prepared: PreparedCurveNetwork,
    rows: np.ndarray,
    parameters: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vertices, intervals, interval curve rows, and interval mid-parameters."""

    points = _chart_points(prepared, rows, parameters)
    first = np.concatenate(([True], rows[1:] != rows[:-1]))
    last = np.concatenate((rows[1:] != rows[:-1], [True]))
    junction = np.full(rows.shape, -1, dtype=np.int64)
    junction[first] = prepared.endpoints[rows.astype(np.int64)[first], 0]
    junction[last] = prepared.endpoints[rows[last], 1]
    vertex = np.empty(rows.shape, dtype=np.int64)
    coordinates = []
    assigned: dict[int, int] = {}
    for position, owner in enumerate(junction.tolist()):
        if owner >= 0 and owner in assigned:
            vertex[position] = assigned[owner]
            continue
        vertex[position] = len(coordinates)
        if owner >= 0:
            assigned[owner] = vertex[position]
            coordinates.append(prepared.junction_points[owner])
        else:
            coordinates.append(points[position])
    same = rows[1:] == rows[:-1]
    cells = np.stack((vertex[:-1][same], vertex[1:][same]), axis=1).astype(np.int32)
    middle = 0.5 * (parameters[:-1][same] + parameters[1:][same])
    return np.asarray(coordinates, dtype=np.float64), cells, rows[:-1][same], middle


def _curve_compliance(
    specification: CurveMeshingSpec,
    prepared: PreparedCurveNetwork,
    vertices: np.ndarray,
    cells: np.ndarray,
    cell_rows: np.ndarray,
    fidelity: tuple[np.ndarray, np.ndarray, bool],
    table_error: float,
    ledger: QueryLedger,
    /,
) -> MeshingComplianceReport:
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = [
        ("density_table_error", table_error),
        ("junction_gap", prepared.junction_gap),
        ("geometry_queries", float(ledger.queries)),
        ("work_units", float(ledger.work)),
    ]
    issues: list[str] = []
    policy = specification.size_compliance
    for control in map(_curve_control, specification.size_controls):
        selected = np.isin(
            prepared.curves[cell_rows], np.asarray(control.scope.entity_ids)
        )
        match control:
            case UniformSizeControl():
                lengths, growth = edge_size_evidence(vertices, cells[selected])
                control_requested, control_achieved, control_issues = (
                    uniform_size_compliance(control, policy, lengths, growth)
                )
                requested.extend(control_requested)
                achieved.extend(control_achieved)
                issues.extend(control_issues)
            case CurvatureSizeControl():
                turning = _turning_angle(vertices, cells[selected])
                key = f"size:{control.control_id}"
                requested.append((f"{key}:normal_angle", control.normal_angle))
                achieved.append((f"{key}:maximum_chord_turning", turning))
                tolerance = policy.tolerance(control.normal_angle)
                if (
                    control.strength is SizeControlStrength.HARD
                    and turning > control.normal_angle + tolerance
                ):
                    issues.append(f"normal_angle:{control.control_id}")
            case _:
                raise TypeError(
                    "specification.size_controls must contain uniform or curvature controls; got an unsupported control"
                )
    deviation, witnessed, certified = fidelity
    for feature in specification.protected_features:
        selected = np.isin(
            prepared.curves[cell_rows], np.asarray(feature.scope.entity_ids)
        )
        key = f"protected:{feature.feature_id}"
        requested.append((f"{key}:maximum_deviation", feature.maximum_deviation))
        achieved.extend(
            (
                (
                    f"{key}:chord_deviation_upper",
                    float(np.max(deviation[selected], initial=0.0)),
                ),
                (
                    f"{key}:chord_deviation_witnessed",
                    float(np.max(witnessed[selected], initial=0.0)),
                ),
                (f"{key}:chord_deviation_certified", float(certified)),
            )
        )
        # A hard fidelity request demands a certified bound, not samples.
        if feature.hard and not certified:
            issues.append(f"uncertified_fidelity:{feature.feature_id}")
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _turning_angle(vertices: np.ndarray, cells: np.ndarray, /) -> float:
    """Largest angle between consecutive interval chords sharing a vertex."""

    if cells.shape[0] < 2:
        return 0.0
    chords = vertices[cells[:, 1]] - vertices[cells[:, 0]]
    following = {int(start): row for row, start in enumerate(cells[:, 0].tolist())}
    pairs = np.asarray(
        [
            (row, following[int(end)])
            for row, end in enumerate(cells[:, 1].tolist())
            if int(end) in following
        ],
        dtype=np.int64,
    ).reshape((-1, 2))
    if pairs.shape[0] == 0:
        return 0.0
    first = chords[pairs[:, 0]]
    second = chords[pairs[:, 1]]
    cosine = np.sum(first * second, axis=1) / (
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    )
    return float(np.max(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _organization_and_association(
    mesh: cell_mesh_owner.CellMesh,
    source: sources_owner.NativeCurveSource,
    specification: CurveMeshingSpec,
    prepared: PreparedCurveNetwork,
    cells: np.ndarray,
    cell_rows: np.ndarray,
    middle: np.ndarray,
    deviation: np.ndarray,
    /,
) -> tuple[tuple[MeshPatch, ...], tuple[MeshLabel, ...], GeometryAssociation]:
    revision = source.source_revision
    cell_set = mesh.entity_set(1)
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)

    patches = tuple(
        MeshPatch(
            f"curve:{curve}",
            mesh_entity_scope(mesh, 1, cell_ids[cell_rows == row]),
        )
        for row, curve in enumerate(prepared.curves.tolist())
    )
    junction_vertices = []
    for index, junction in enumerate(specification.junctions):
        curve, end = junction.endpoints[0]
        row = int(np.flatnonzero(prepared.curves == curve)[0])
        owned = np.flatnonzero(cell_rows == row)
        cell = owned[0] if end == "start" else owned[-1]
        vertex = cells[cell, 0 if end == "start" else 1]
        junction_vertices.append((junction.name, vertex_ids[vertex], index))
    labels = tuple(
        MeshLabel(name, mesh_entity_scope(mesh, 0, np.asarray((vertex,))))
        for name, vertex, _ in junction_vertices
    )
    association = GeometryAssociation(
        GeometryAssociationKind.CURVE,
        source.source_id,
        revision,
        cell_set.entity_set_id,
        cell_ids,
        tuple(
            sources_owner.source_entity_id(revision, "curve", int(prepared.curves[row]))
            for row in cell_rows.tolist()
        ),
        deviation,
        exact=False,
        parameters=np.stack((middle, np.full(middle.shape, np.nan)), axis=1),
        orientations=np.ones((cell_ids.size,), dtype=np.int8),
    )
    return patches, labels, association


@checked
def execute_curve_route(
    source: sources_owner.NativeCurveSource,
    specification: CurveMeshingSpec,
    prepared: PreparedCurveNetwork,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    """Place, assemble, associate, audit, and publish one curve network."""
    budget = coordinate_budget(specification.limits)
    with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
        started = scope.operation_started
        limits = specification.limits
        ledger = QueryLedger(limits, budget=budget)
        with measure_phase(record_phase, "refinement"):
            with coordinate_phase(scope, specification.limits, budget):
                rows, parameters, _, table_error = discretize(
                    prepared, specification, ledger, started
                )
        with measure_phase(record_phase, "construction"):
            vertices, cells, cell_rows, middle = _assemble(prepared, rows, parameters)
            simplex_entity_limits(
                vertices,
                cells,
                limits,
                MeshingStageKind.CURVE_MESHING,
                cell_kind="interval",
            )
            starts = parameters[:-1][rows[1:] == rows[:-1]]
            ends = parameters[1:][rows[1:] == rows[:-1]]
            with coordinate_phase(scope, specification.limits, budget):
                deviation, certified = _chord_bound(
                    prepared, cell_rows, starts, ends, ledger
                )
            with coordinate_phase(scope, specification.limits, budget):
                witnessed = _chord_deviation(prepared, cell_rows, starts, ends, ledger)
            with coordinate_phase(scope, specification.limits, budget):
                mesh = canonicalize_cell_mesh(
                    cell_mesh_owner.CellMesh(
                        vertices,
                        (
                            reference_cell_types_owner.CellBlock(
                                "intervals", "interval", cells
                            ),
                        ),
                        numeric_version=source.source_revision,
                    ),
                    budget=budget,
                )
        with measure_phase(record_phase, "geometry_association"):
            patches, labels, association = _organization_and_association(
                mesh, source, specification, prepared, cells, cell_rows, middle, deviation
            )
        with measure_phase(record_phase, "compliance"):
            with coordinate_phase(scope, specification.limits, budget):
                compliance = _curve_compliance(
                    specification,
                    prepared,
                    vertices,
                    cells,
                    cell_rows,
                    (deviation, witnessed, certified),
                    table_error,
                    ledger,
                )
        check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
        construction = (
            MeshingStageReport(
                MeshingStageKind.SOURCE_INSPECTION,
                MeshingStageStatus.PASSED,
                input_ids=(source.source_revision,),
                output_ids=(prepared.prepared_id,),
            ),
            MeshingStageReport(
                MeshingStageKind.CURVE_MESHING,
                MeshingStageStatus.PASSED,
                input_ids=(prepared.prepared_id,),
                output_ids=(mesh.mesh_id,),
                created_count=cells.shape[0],
            ),
        )
        return publish_native_result(
            mesh,
            coordinate_contract,
            compliance,
            construction,
            provider,
            {
                "kind": "native-curve-cell-mesh",
                "route": "curve_arc_length",
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "plan": plan_id,
                "specification": specification.specification_id,
                "fidelity": "certified-chord-bound"
                if certified
                else "sampled-chord-deviation",
            },
            NativeCertificationRequest(
                MeshCertificationSchedule("curve"),
                source.source_id,
                source.source_revision,
                limits,
                junction_vertices=np.concatenate(
                    [
                        np.asarray(label.scope.entity_ids, dtype=np.int64)
                        for label in labels
                    ]
                )
                if labels
                else None,
            ),
            policy=publication_policy(
                "curve",
                audit_policy=CellMeshAuditPolicy(
                    require_complete_association=True,
                    nonmanifold=CellMeshAuditDisposition.RECORD
                    if any(
                        (len(value.endpoints) > 2 for value in specification.junctions)
                    )
                    else CellMeshAuditDisposition.REJECT,
                ),
            ),
            organization=NativePublicationOrganization(
                patches=patches, labels=labels, associations=(association,)
            ),
            execution=NativePublicationExecution(scope, workspace, record_phase, budget),
        )


__all__ = [
    "PreparedCurveNetwork",
    "curve_support_issues",
    "execute_curve_route",
]
