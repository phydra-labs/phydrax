#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native curve meshing by physical size-density and source-parameter inversion."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.extend import core as jax_core

import phydrax.discretization as cell_mesh_owner
import phydrax.meshing._association as association_owner
from phydrax.geometry import BoundaryAtlas
from phydrax.geometry.simplicial import SegmentMesh
from phydrax.meshing._contracts import (
    CurveMeshingSpec,
    MeshingFailure,
    MeshingFailureCategory,
)
from phydrax.meshing._result import MeshingComplianceReport
from phydrax.meshing._sizing import (
    CurvatureSizeControl,
    resolve_size_controls,
    SizeCombinationPolicy,
    SizeControlStrength,
    SizeFieldDomain,
    UniformSizeControl,
)

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellGeometrySpec
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...geometry._atlas import AbstractBoundaryMap
from ...geometry._curve_boundary import (
    ParametricCurveBoundarySource,
    SegmentChartCurve,
)
from ...geometry._interval_enclosure import (
    IntervalExtensionError,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from ...geometry._rational import binary64_difference_magnitude, binary64_norm_upper
from ...geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
)
from ...geometry.brep._patches import AbstractCurve
from ...typing import checked
from .._certification_schedule import MeshCertificationSchedule
from .._controls import FeatureKind
from .._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from .._organization import MeshLabel, MeshPatch
from .._scope import MeshingEntityKind, MeshingScope
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from . import _native_sources as sources_owner
from ._native_options import NativeMeshingOptions
from ._native_schedules import NativeCurveSchedule


@final
class _AtlasCurveCarrier(AbstractCurve):
    """One source parameter interval evaluated through its original atlas chart."""

    mapping: AbstractBoundaryMap
    chart_row: int = eqx.field(static=True)
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mapping: AbstractBoundaryMap,
        chart_row: int,
        first: float,
        last: float,
        /,
    ) -> None:
        self.mapping = mapping
        self.chart_row = chart_row
        self.first = first
        self.last = last

    @property
    def ambient_dimension(self) -> int:
        return self.mapping.ambient_dimension

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        local = (values - self.first) / (self.last - self.first)
        rows = jnp.full(values.shape, self.chart_row, dtype=jnp.int32)
        return self.mapping.map(rows, local[..., None])

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        """Enclose the atlas chart program, which is this carrier's source map."""
        AbstractCurve.is_c1_on(self, first, last)
        prepared = prepare_interval_function(
            lambda parameters: self.evaluate(parameters[0]), 1
        )
        lower, upper = prepared.evaluate(
            np.asarray(((first,),), dtype=np.float64),
            np.asarray(((last,),), dtype=np.float64),
        )
        return np.stack((lower[0], upper[0]))


def _curve_source_query(
    prepared: PreparedCurveNetwork,
    selected_entity_ids: np.ndarray | None = None,
    /,
) -> ParametricCurveBoundarySource:
    """Bind continuous queries to exact authored charts, ranges, and endpoints."""
    source = prepared.source
    selected_rows = (
        np.arange(prepared.scope_entity_ids.size, dtype=np.int64)
        if selected_entity_ids is None
        else np.flatnonzero(
            np.isin(
                prepared.scope_entity_ids,
                np.asarray(selected_entity_ids, dtype=np.int64),
            )
        )
    )
    if selected_rows.size == 0:
        raise ValueError(
            "A curve source query requires at least one selected authored chart."
        )
    chart_rows = np.asarray(prepared.chart_rows, dtype=np.int64)[selected_rows]
    scope_ids = np.asarray(prepared.scope_entity_ids, dtype=np.int64)[selected_rows]
    source_ranges = np.asarray(prepared.source_ranges, dtype=np.float64)[selected_rows]
    chart_ids = tuple(prepared.chart_ids[int(row)] for row in selected_rows.tolist())
    ranges = tuple((float(first), float(last)) for first, last in source_ranges.tolist())
    if isinstance(source.curves, BoundaryAtlas):
        carriers = tuple(
            _AtlasCurveCarrier(source.curves.mapping, int(chart), first, last)
            for chart, (first, last) in zip(chart_rows.tolist(), ranges, strict=True)
        )
    else:
        source_vertices = np.asarray(source.curves.vertices, dtype=np.float64)
        source_edges = np.asarray(source.curves.edges, dtype=np.int64)
        carriers = tuple(
            SegmentChartCurve(
                source_vertices[source_edges[chart, 0]],
                source_vertices[source_edges[chart, 1]],
                first,
                last,
            )
            for chart, (first, last) in zip(chart_rows.tolist(), ranges, strict=True)
        )
    source_keys = tuple(
        source.source_entity_key_for_scope(1, int(identifier))
        for identifier in scope_ids.tolist()
    )
    endpoint_source_keys = []
    endpoint_point_bounds = []
    for chart, (first, last) in zip(chart_rows.tolist(), ranges, strict=True):
        chart_endpoint_keys = []
        chart_endpoint_bounds = []
        for endpoint_side, parameter in enumerate((first, last)):
            point_key = None
            point_bounds = None
            if isinstance(source.curves, SegmentMesh):
                vertex_row = int(source.curves.edges[chart, endpoint_side])
                point_id = int(source.vertex_entity_ids[vertex_row])
                point_key = source.source_entity_key_for_scope(0, point_id)
                point = tuple(
                    float(value) for value in source.curves.vertices[vertex_row]
                )
                point_bounds = (point, point)
            elif source.chart_endpoint_ids:
                authored_range = (
                    (0.0, 1.0)
                    if source.chart_parameter_ranges is None
                    else tuple(
                        float(value) for value in source.chart_parameter_ranges[chart]
                    )
                )
                source_endpoint_side = (
                    0
                    if parameter == authored_range[0]
                    else 1
                    if parameter == authored_range[1]
                    else None
                )
                if source_endpoint_side is not None:
                    endpoint_id = source.chart_endpoint_ids[chart][source_endpoint_side]
                    if endpoint_id is not None:
                        vertex_row = (
                            source.vertex_entity_keys.index(endpoint_id)
                            if source.gap_domain is None
                            else source.gap_domain.resolve_entity_id(0, endpoint_id)
                        )
                        point_id = int(source.vertex_entity_ids[vertex_row])
                        point_key = source.source_entity_key_for_scope(0, point_id)
                        if source.gap_domain is not None:
                            lower, upper = source.gap_domain.source_corner_point_bounds(
                                vertex_row
                            )
                            lower_float = []
                            upper_float = []
                            for lo, hi in zip(lower, upper, strict=True):
                                lo_float = float(lo)
                                hi_float = float(hi)
                                if Fraction.from_float(lo_float) > lo:
                                    lo_float = float(np.nextafter(lo_float, -np.inf))
                                if Fraction.from_float(hi_float) < hi:
                                    hi_float = float(np.nextafter(hi_float, np.inf))
                                lower_float.append(lo_float)
                                upper_float.append(hi_float)
                            point_bounds = (tuple(lower_float), tuple(upper_float))
            chart_endpoint_keys.append(point_key)
            chart_endpoint_bounds.append(point_bounds)
        endpoint_source_keys.append((chart_endpoint_keys[0], chart_endpoint_keys[1]))
        endpoint_point_bounds.append((chart_endpoint_bounds[0], chart_endpoint_bounds[1]))
    return ParametricCurveBoundarySource(
        carriers,
        ranges,
        source_id=source.source_id,
        source_revision=source.source_revision,
        curve_entity_ids=tuple(entity_id for entity_id, _ in source_keys),
        occurrence_paths=tuple(occurrences for _, occurrences in source_keys),
        chart_ids=chart_ids,
        endpoint_source_keys=tuple(endpoint_source_keys),
        endpoint_point_bounds=tuple(endpoint_point_bounds),
    )


class CurveCharts(StrictModule, NonTrainableState):
    """Selected source entities and their explicit parameter-chart rows."""

    mapping: AbstractBoundaryMap | None
    segments: Array | None
    curve_ids: np.ndarray
    chart_rows: np.ndarray
    chart_reversed: np.ndarray
    ambient_dimension: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        curves: BoundaryAtlas | SegmentMesh,
        curve_ids: np.ndarray,
        chart_rows: np.ndarray,
        chart_reversed: np.ndarray,
        /,
    ) -> None:
        if isinstance(curves, BoundaryAtlas):
            self.mapping = curves.mapping
            self.segments = None
            self.ambient_dimension = curves.ambient_dimension
        elif isinstance(curves, SegmentMesh):
            self.mapping = None
            self.segments = jnp.asarray(curves.segments, dtype=jnp.float64)
            self.ambient_dimension = int(curves.vertices.shape[1])
        else:
            raise TypeError("CurveCharts.curves must be BoundaryAtlas or SegmentMesh.")
        ids = np.asarray(curve_ids, dtype=np.int64)
        rows = np.asarray(chart_rows, dtype=np.int64)
        reversed_rows = np.asarray(chart_reversed, dtype=np.bool_)
        if ids.ndim != 1 or rows.shape != ids.shape or reversed_rows.shape != ids.shape:
            raise ValueError("Curve source IDs, chart rows, and orientations must align.")
        self.curve_ids = ids
        self.chart_rows = rows
        self.chart_reversed = reversed_rows

    def point(self, row: int, parameter: Array, /) -> Array:
        chart_row = self.chart_rows[row]
        local = 1.0 - parameter if self.chart_reversed[row] else parameter
        if self.mapping is not None:
            indices = jnp.full(parameter.shape, chart_row, dtype=jnp.int32)
            return self.mapping.map(indices, local[..., None])
        if self.segments is None:
            raise RuntimeError("CurveCharts requires a source map or line segments.")
        segment = self.segments[chart_row]
        return segment[0] + local[..., None] * (segment[1] - segment[0])

    def jet(self, row: int, parameter: Array, /) -> tuple[Array, Array, Array]:
        """Evaluate position and first two derivatives in the chart parameter."""
        ones = jnp.ones_like(parameter)
        curve = lambda value: self.point(row, value)
        point, velocity = jax.jvp(curve, (parameter,), (ones,))
        _, acceleration = jax.jvp(
            lambda value: jax.jvp(curve, (value,), (jnp.ones_like(value),))[1],
            (parameter,),
            (ones,),
        )
        return point, velocity, acceleration


@final
class CurveSizing(StrictModule, NonTrainableState):
    """Per-chart controls resolved by the shared physical size-field owner."""

    scope_entity_ids: np.ndarray
    controls: tuple[tuple[UniformSizeControl | CurvatureSizeControl, ...], ...]
    upper_size: Array
    size_combination: SizeCombinationPolicy = eqx.field(static=True)

    @checked
    def __init__(
        self,
        specification: CurveMeshingSpec,
        scope_entity_ids: np.ndarray,
        /,
    ) -> None:
        identifiers = np.asarray(scope_entity_ids, dtype=np.int64)
        controls_by_scope: list[
            tuple[UniformSizeControl | CurvatureSizeControl, ...]
        ] = []
        upper = np.full((identifiers.size,), 0.0, dtype=np.float64)
        supported_controls = tuple(
            control
            for control in specification.size_controls
            if isinstance(control, (UniformSizeControl, CurvatureSizeControl))
        )
        if len(supported_controls) != len(specification.size_controls):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "curve size fields support uniform and curvature controls only",
                provider_code="native_curve_sizing",
                stage="source_inspection",
            )
        for row, scope_id in enumerate(identifiers.tolist()):
            selected = tuple(
                control
                for control in supported_controls
                if int(scope_id) in np.asarray(control.scope.entity_ids, dtype=np.int64)
            )
            if not selected:
                raise MeshingFailure(
                    MeshingFailureCategory.CONTROL_CONFLICT,
                    f"curve scope entity {scope_id} has no supported size control",
                    provider_code="native_curve_sizing",
                    stage="source_inspection",
                    entity_ids=(int(scope_id),),
                )
            controls_by_scope.append(selected)
            for control in selected:
                if isinstance(control, UniformSizeControl):
                    value = (
                        control.target_size
                        if control.maximum_size is None
                        else min(control.target_size, control.maximum_size)
                    )
                    upper[row] = max(upper[row], value)
                else:
                    upper[row] = max(
                        upper[row],
                        math.inf
                        if control.maximum_size is None
                        else control.maximum_size,
                    )
        self.scope_entity_ids = identifiers
        self.controls = tuple(controls_by_scope)
        self.upper_size = jnp.asarray(upper, dtype=jnp.float64)
        self.size_combination = specification.size_combination

    def density(
        self,
        charts: CurveCharts,
        row: int,
        parameter: Array,
        /,
    ) -> tuple[Array, Array]:
        parameters = jnp.asarray(parameter, dtype=jnp.float64)
        if any(
            isinstance(control, CurvatureSizeControl) for control in self.controls[row]
        ):
            _, velocity, acceleration = charts.jet(row, parameters)
            speed = np.asarray(jax.device_get(jnp.linalg.norm(velocity, axis=-1)))
            acceleration_host = np.asarray(jax.device_get(acceleration))
            velocity_host = np.asarray(jax.device_get(velocity))
            squared_speed = np.sum(velocity_host * velocity_host, axis=-1)
            squared_acceleration = np.sum(acceleration_host * acceleration_host, axis=-1)
            dot = np.sum(velocity_host * acceleration_host, axis=-1)
            cross = np.sqrt(
                np.maximum(squared_speed * squared_acceleration - dot * dot, 0.0)
            )
            curvature = np.divide(
                cross,
                squared_speed * speed,
                out=np.zeros_like(speed),
                where=speed > 0.0,
            )
        else:
            velocity = jax.jvp(
                lambda value: charts.point(row, value),
                (parameters,),
                (jnp.ones_like(parameters),),
            )[1]
            speed = np.asarray(jax.device_get(jnp.linalg.norm(velocity, axis=-1)))
            curvature = None
        points = np.asarray(
            jax.device_get(charts.point(row, parameters)), dtype=np.float64
        )
        identifiers = np.full(
            parameters.shape, int(self.scope_entity_ids[row]), dtype=np.int64
        )
        try:
            field, _ = resolve_size_controls(
                self.controls[row],
                points,
                identifiers,
                SizeFieldDomain.EUCLIDEAN_VOLUME,
                curvature=curvature,
                combination=self.size_combination,
            )
        except ValueError as error:
            raise MeshingFailure(
                MeshingFailureCategory.CONTROL_CONFLICT,
                f"curve size controls do not resolve on source scope entity {int(self.scope_entity_ids[row])}: {error}",
                provider_code="native_curve_sizing",
                stage="source_inspection",
                entity_ids=(int(self.scope_entity_ids[row]),),
            ) from error
        sizes = np.asarray(field.values, dtype=np.float64)
        density = np.divide(
            speed,
            sizes,
            out=np.zeros_like(speed),
            where=np.isfinite(sizes) & (sizes > 0.0),
        )
        return (
            jnp.asarray(density, dtype=jnp.float64),
            jnp.asarray(speed, dtype=jnp.float64),
        )


@final
class _CurveEnclosure(StrictModule, NonTrainableState):
    """Outward interval programs for a curve's acceleration and coordinates."""

    acceleration_jaxpr: jax_core.Jaxpr = eqx.field(static=True)
    acceleration_consts: tuple[Array, ...]
    acceleration_constant_intervals: tuple[tuple[Array, Array], ...]
    point_jaxpr: jax_core.Jaxpr = eqx.field(static=True)
    point_consts: tuple[Array, ...]
    point_constant_intervals: tuple[tuple[Array, Array], ...]
    ambient_dimension: int = eqx.field(static=True)
    rigorous: bool = eqx.field(static=True)

    @staticmethod
    def _function(
        jaxpr: jax_core.Jaxpr,
        constants: tuple[Array, ...],
        constant_intervals: tuple[tuple[Array, Array], ...],
        output_dimension: int,
        /,
    ) -> PreparedIntervalFunction:
        closed = jax_core.ClosedJaxpr(
            jaxpr, tuple(np.asarray(value) for value in constants)
        )
        intervals = tuple(
            (np.asarray(lower), np.asarray(upper)) for lower, upper in constant_intervals
        )
        return PreparedIntervalFunction(closed, 1, (output_dimension,), 1, intervals)

    def __init__(self, charts: CurveCharts, row: int, /) -> None:
        acceleration = prepare_interval_function(
            lambda point: charts.jet(row, point[0])[2], 1, batch_capacity=1
        )
        point = prepare_interval_function(
            lambda parameter: charts.point(row, parameter[0]),
            1,
            batch_capacity=1,
        )
        acceleration_consts = tuple(
            jax.device_put(np.asarray(value))
            for value in acceleration.closed_jaxpr.consts
        )
        point_consts = tuple(
            jax.device_put(np.asarray(value)) for value in point.closed_jaxpr.consts
        )
        acceleration_intervals = tuple(
            (jax.device_put(np.asarray(lower)), jax.device_put(np.asarray(upper)))
            for lower, upper in acceleration.constant_intervals
        )
        point_intervals = tuple(
            (jax.device_put(np.asarray(lower)), jax.device_put(np.asarray(upper)))
            for lower, upper in point.constant_intervals
        )
        try:
            acceleration.evaluate(np.asarray([[0.0]]), np.asarray([[1.0]]))
            point.evaluate(np.asarray([[0.0]]), np.asarray([[0.0]]))
        except IntervalExtensionError:
            supported = False
        else:
            supported = True
        self.acceleration_jaxpr = acceleration.closed_jaxpr.jaxpr
        self.acceleration_consts = acceleration_consts
        self.acceleration_constant_intervals = acceleration_intervals
        self.point_jaxpr = point.closed_jaxpr.jaxpr
        self.point_consts = point_consts
        self.point_constant_intervals = point_intervals
        self.ambient_dimension = charts.ambient_dimension
        self.rigorous = supported

    def acceleration_upper(self, lower: float, upper: float, /) -> float:
        function = self._function(
            self.acceleration_jaxpr,
            self.acceleration_consts,
            self.acceleration_constant_intervals,
            self.ambient_dimension,
        )
        lower_values, upper_values = function.evaluate(
            np.asarray([[lower]], dtype=np.float64),
            np.asarray([[upper]], dtype=np.float64),
        )
        magnitude = np.maximum(np.abs(lower_values[0]), np.abs(upper_values[0]))
        return float(binary64_norm_upper(magnitude))

    def point_error(self, parameter: float, published: np.ndarray, /) -> float:
        function = self._function(
            self.point_jaxpr,
            self.point_consts,
            self.point_constant_intervals,
            self.ambient_dimension,
        )
        lower_values, upper_values = function.evaluate(
            np.asarray([[parameter]], dtype=np.float64),
            np.asarray([[parameter]], dtype=np.float64),
        )
        component = np.maximum(
            binary64_difference_magnitude(lower_values[0], published)[1],
            binary64_difference_magnitude(upper_values[0], published)[1],
        )
        return float(binary64_norm_upper(component))


@final
class PreparedCurveNetwork(StrictModule, NonTrainableState):
    """Source IDs, ordered source charts, junction identities, and size controls."""

    source: sources_owner.NativeCurveSource
    specification: CurveMeshingSpec
    charts: CurveCharts
    sizing: CurveSizing
    curve_entity_ids: np.ndarray
    scope_entity_ids: np.ndarray
    chart_rows: np.ndarray
    source_ranges: np.ndarray
    chart_ids: tuple[str, ...] = eqx.field(static=True)
    endpoints: np.ndarray
    junction_points: np.ndarray
    junction_spread: np.ndarray
    junction_names: tuple[str | None, ...] = eqx.field(static=True)
    enclosures: tuple[_CurveEnclosure, ...]
    schedule: NativeCurveSchedule
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativeCurveSource,
        specification: CurveMeshingSpec,
        schedule: NativeCurveSchedule,
        /,
    ) -> None:
        (
            curve_entity_ids,
            scope_entity_ids,
            chart_rows,
            source_ranges,
            reversed_rows,
            chart_ids,
        ) = _selected_curve_rows(source, specification)
        charts = CurveCharts(source.curves, curve_entity_ids, chart_rows, reversed_rows)
        (
            endpoints,
            junction_names,
            junction_points,
            junction_spread,
        ) = _endpoint_junctions(
            source,
            specification,
            scope_entity_ids,
            chart_rows,
            source_ranges,
            reversed_rows,
            charts,
        )
        sizing = CurveSizing(specification, scope_entity_ids)
        enclosures = tuple(
            _CurveEnclosure(charts, row) for row in range(curve_entity_ids.size)
        )
        self.source = source
        self.specification = specification
        self.charts = charts
        self.sizing = sizing
        self.curve_entity_ids = curve_entity_ids
        self.chart_rows = chart_rows
        self.scope_entity_ids = scope_entity_ids
        self.source_ranges = source_ranges
        self.chart_ids = chart_ids
        self.endpoints = endpoints
        self.junction_points = junction_points
        self.junction_spread = junction_spread
        self.junction_names = junction_names
        self.enclosures = enclosures
        self.schedule = schedule
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-curve-network",
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "source_binding": source.binding_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
                "scope_entity_ids": scope_entity_ids,
                "chart_ids": chart_ids,
                "chart_parameter_ranges": array_tree_fingerprint(source_ranges),
                "chart_rows": chart_rows,
                "junctions": junction_names,
                "junction_points": array_tree_fingerprint(junction_points),
            }
        )

    def physical_tolerance(self, row: int, /) -> float:
        """Return the strictest declared physical accuracy for this source curve."""
        specification = self.specification
        tolerances = (
            []
            if specification.maximum_deviation is None
            else [specification.maximum_deviation]
        )
        for feature in specification.protected_features:
            if int(self.scope_entity_ids[row]) in np.asarray(
                feature.scope.entity_ids, dtype=np.int64
            ):
                tolerances.append(feature.maximum_deviation)
        if tolerances:
            return min(tolerances)
        policy = specification.size_compliance
        for control in specification.size_controls:
            if not isinstance(control, (UniformSizeControl, CurvatureSizeControl)):
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "curve physical tolerance requires a supported size control",
                    provider_code="native_curve_sizing",
                    stage="source_inspection",
                )
            if int(self.scope_entity_ids[row]) not in np.asarray(
                control.scope.entity_ids, dtype=np.int64
            ):
                continue
            if isinstance(control, UniformSizeControl):
                tolerances.append(
                    policy.absolute_tolerance
                    + policy.relative_tolerance * control.target_size
                )
            elif isinstance(control, CurvatureSizeControl):
                angle_tolerance = (
                    policy.absolute_tolerance
                    + policy.relative_tolerance * control.normal_angle
                )
                parameters = np.linspace(0.0, 1.0, self.schedule.quadrature_intervals + 1)
                density, speed = self.sizing.density(
                    self.charts, row, jnp.asarray(parameters, dtype=jnp.float64)
                )
                density_host = np.asarray(jax.device_get(density), dtype=np.float64)
                speed_host = np.asarray(jax.device_get(speed), dtype=np.float64)
                size = np.divide(
                    speed_host,
                    density_host,
                    out=np.full_like(speed_host, np.inf),
                    where=density_host > 0.0,
                )
                finite = size[np.isfinite(size)]
                if finite.size:
                    tolerances.append(
                        angle_tolerance * float(np.max(finite)) / control.normal_angle
                    )
        if tolerances:
            return min(tolerances)
        upper_size = float(jax.device_get(self.sizing.upper_size[row]))

        relative = (
            policy.relative_tolerance * upper_size
            if math.isfinite(upper_size) and upper_size > 0.0
            else 0.0
        )
        fallback = policy.absolute_tolerance + relative
        if not math.isfinite(fallback) or fallback <= 0.0:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "curve placement has no positive physical tolerance from the declared size policy",
                provider_code="native_curve_unresolved",
                stage="curve_meshing",
                entity_ids=(int(self.scope_entity_ids[row]),),
            )
        return fallback

    def tolerance(self, row: int, /) -> float:
        """Convert physical accuracy to the dimensionless density integral."""
        physical = self.physical_tolerance(row)
        upper_size = float(jax.device_get(self.sizing.upper_size[row]))
        return (
            physical / upper_size
            if math.isfinite(upper_size) and upper_size > 0.0
            else physical
        )

    def density_values(self, row: int, parameters: np.ndarray, /) -> np.ndarray:
        values, _ = self.sizing.density(
            self.charts, row, jnp.asarray(parameters, dtype=jnp.float64)
        )
        return np.asarray(jax.device_get(values), dtype=np.float64)

    def points(self, row: int, parameters: np.ndarray, /) -> np.ndarray:
        return np.asarray(
            jax.device_get(
                self.charts.point(row, jnp.asarray(parameters, dtype=jnp.float64))
            ),
            dtype=np.float64,
        )


def _selected_curve_rows(
    source: sources_owner.NativeCurveSource,
    specification: CurveMeshingSpec,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    selected = np.asarray(specification.scope.entity_ids, dtype=np.int64)
    if selected.size == 0:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "curve scope must select at least one source entity",
            provider_code="native_curve_preflight",
            stage="source_inspection",
        )
    semantic_ids = np.asarray(source.curve_entity_ids, dtype=np.int64)
    scope_ids = np.asarray(source.chart_scope_entity_ids, dtype=np.int64)
    ranges = (
        np.tile(np.asarray(((0.0, 1.0),), dtype=np.float64), (semantic_ids.size, 1))
        if source.chart_parameter_ranges is None
        else np.asarray(source.chart_parameter_ranges, dtype=np.float64)
    )
    curve_entity_ids: list[int] = []
    selected_scope_rows: list[int] = []
    chart_rows: list[int] = []
    parameter_ranges: list[np.ndarray] = []
    chart_ids: list[str] = []
    for scope_id in selected.tolist():
        rows = np.flatnonzero(scope_ids == scope_id)
        if not rows.size:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                f"curve scope names absent source entity {scope_id}",
                provider_code="native_curve_preflight",
                stage="source_inspection",
                entity_ids=(int(scope_id),),
            )
        ordered = sorted(
            rows.tolist(),
            key=lambda row: (
                min(ranges[row]),
                max(ranges[row]),
                source.chart_ids[row],
            ),
        )
        previous_upper: float | None = None
        for row in ordered:
            lower, upper = sorted((float(ranges[row, 0]), float(ranges[row, 1])))
            if previous_upper is not None and lower < previous_upper:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    f"curve scope entity {scope_id} has overlapping parameter charts",
                    provider_code="native_curve_preflight",
                    stage="source_inspection",
                    entity_ids=(int(scope_id),),
                )
            previous_upper = upper
            curve_entity_ids.append(int(semantic_ids[row]))
            selected_scope_rows.append(int(scope_id))
            chart_rows.append(int(row))
            parameter_ranges.append(np.asarray(ranges[row], dtype=np.float64))
            chart_ids.append(source.chart_ids[row])
    return (
        np.asarray(curve_entity_ids, dtype=np.int64),
        np.asarray(selected_scope_rows, dtype=np.int64),
        np.asarray(chart_rows, dtype=np.int64),
        np.stack(parameter_ranges),
        np.asarray([first > last for first, last in parameter_ranges], dtype=np.bool_),
        tuple(chart_ids),
    )


def _endpoint_junctions(
    source: sources_owner.NativeCurveSource,
    specification: CurveMeshingSpec,
    scope_entity_ids: np.ndarray,
    chart_rows: np.ndarray,
    source_ranges: np.ndarray,
    chart_reversed: np.ndarray,
    charts: CurveCharts,
    /,
) -> tuple[
    np.ndarray,
    tuple[str | None, ...],
    np.ndarray,
    np.ndarray,
]:
    """Group endpoints by authored vertex, exact source parameter, or junction."""
    count = scope_entity_ids.size
    parent = np.arange(2 * count, dtype=np.int64)
    endpoint_sources: list[tuple[str, str | int] | None] = [None] * (2 * count)
    parameter_groups: dict[tuple[int, Fraction], list[int]] = {}
    source_groups: dict[tuple[str, str | int], list[int]] = {}

    def find(node: int, /) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            following = int(parent[node])
            parent[node] = root
            node = following
        return root

    def union(first: int, second: int, /) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for row, (scope_id, chart) in enumerate(
        zip(scope_entity_ids, chart_rows, strict=True)
    ):
        low, high = sorted(float(value) for value in source_ranges[row])
        chart_row = int(chart)
        source_endpoints: tuple[
            tuple[str, str | int] | None, tuple[str, str | int] | None
        ] = (None, None)
        if source.gap_domain is not None and isinstance(source.curves, BoundaryAtlas):
            curve_key, curve_occurrence = source.source_entity_key_for_scope(
                1, int(source.chart_scope_entity_ids[chart_row])
            )
            domain_slot = source.gap_domain.resolve_entity_id(
                1, curve_key, occurrence_path=curve_occurrence
            )
            domain_curve = source.gap_domain.curves[domain_slot]
            source_endpoints = (
                (
                    "source-vertex",
                    source.gap_domain.entity_id(0, domain_curve.start),
                ),
                (
                    "source-vertex",
                    source.gap_domain.entity_id(0, domain_curve.end),
                ),
            )
        elif source.chart_endpoint_ids:
            endpoint_ids = source.chart_endpoint_ids[chart_row]
            source_endpoints = (
                None if endpoint_ids[0] is None else ("source-vertex", endpoint_ids[0]),
                None if endpoint_ids[1] is None else ("source-vertex", endpoint_ids[1]),
            )
        elif isinstance(source.curves, SegmentMesh):
            edge = np.asarray(source.curves.edges, dtype=np.int64)[chart_row]
            source_endpoints = (
                ("source-vertex", source.vertex_entity_keys[int(edge[0])]),
                ("source-vertex", source.vertex_entity_keys[int(edge[1])]),
            )
        if chart_reversed[row]:
            source_endpoints = (source_endpoints[1], source_endpoints[0])
        for side, source_key in enumerate(source_endpoints):
            node = 2 * row + side
            endpoint_sources[node] = source_key
            if source_key is not None:
                source_groups.setdefault(source_key, []).append(node)
        parameter_groups.setdefault((int(scope_id), Fraction(low)), []).append(2 * row)
        parameter_groups.setdefault((int(scope_id), Fraction(high)), []).append(
            2 * row + 1
        )

    for nodes in source_groups.values():
        for node in nodes[1:]:
            union(nodes[0], node)
    for nodes in parameter_groups.values():
        source_entities = {
            endpoint_sources[node] for node in nodes if endpoint_sources[node] is not None
        }
        if len(source_entities) > 1:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "charts of one source curve disagree on a shared parameter endpoint identity",
                provider_code="native_curve_junction",
                stage="source_inspection",
            )
        for node in nodes[1:]:
            union(nodes[0], node)

    explicit_names: dict[int, str] = {}
    scope_rows: dict[int, list[int]] = {}
    for row, scope_id in enumerate(scope_entity_ids.tolist()):
        scope_rows.setdefault(int(scope_id), []).append(row)
    for junction in specification.junctions:
        members: list[int] = []
        for scope_id, end in junction.endpoints:
            rows = scope_rows.get(int(scope_id), [])
            if not rows:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SPECIFICATION,
                    f"curve junction {junction.name!r} names a curve scope entity outside the source scope",
                    provider_code="native_curve_preflight",
                    stage="source_inspection",
                    entity_ids=(int(scope_id),),
                )
            row = rows[0] if end == "start" else rows[-1]
            members.append(2 * row + (0 if end == "start" else 1))
        for node in members[1:]:
            union(members[0], node)
        explicit_names[members[0]] = junction.name

    components: dict[int, list[int]] = {}
    for node in range(2 * count):
        components.setdefault(find(node), []).append(node)
    named_roots: dict[int, set[str]] = {}
    for node, name in explicit_names.items():
        named_roots.setdefault(find(node), set()).add(name)
    if any(len(names) > 1 for names in named_roots.values()):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "distinct curve junction names cannot identify one shared source vertex",
            provider_code="native_curve_preflight",
            stage="source_inspection",
        )
    roots = sorted(
        (
            root
            for root, nodes in components.items()
            if len(nodes) > 1 or root in named_roots
        ),
        key=lambda root: min(components[root]),
    )
    group_rows = {root: row for row, root in enumerate(roots)}
    endpoints = np.full((count, 2), -1, dtype=np.int64)
    for row in range(count):
        for side in range(2):
            root = find(2 * row + side)
            if root in group_rows:
                endpoints[row, side] = group_rows[root]
    endpoint_points = np.stack(
        [
            np.asarray(
                jax.device_get(charts.point(row, jnp.asarray((0.0, 1.0)))),
                dtype=np.float64,
            )
            for row in range(count)
        ]
    )
    tolerance = _source_fidelity_tolerance(specification)
    junction_points = np.empty((len(roots), charts.ambient_dimension), dtype=np.float64)
    junction_spread = np.zeros((len(roots),), dtype=np.float64)
    junction_names: list[str | None] = []
    for group, root in enumerate(roots):
        nodes = components[root]
        members = np.asarray(
            [endpoint_points[node // 2, node % 2] for node in nodes],
            dtype=np.float64,
        )
        junction_points[group] = members[0]
        spread = float(np.max(np.linalg.norm(members - members[0], axis=1), initial=0.0))
        junction_spread[group] = spread
        source_keys = {
            endpoint_sources[node] for node in nodes if endpoint_sources[node] is not None
        }
        explicit = next(iter(named_roots.get(root, ())), None)
        if tolerance is not None and spread > tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "a source-identified curve junction exceeds its physical tolerance",
                provider_code="native_curve_junction",
                stage="source_inspection",
                locations=tuple(
                    tuple(float(value) for value in point) for point in members
                ),
            )
        if (
            tolerance is None
            and spread > 0.0
            and (len(source_keys) > 1 or (explicit is not None and len(source_keys) != 1))
        ):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "joining distinct source vertices requires an explicit physical tolerance",
                provider_code="native_curve_junction",
                stage="source_inspection",
            )
        junction_names.append(explicit)
    return (
        endpoints,
        tuple(junction_names),
        junction_points,
        junction_spread,
    )


def _selected_source_vertex_ids(
    source: sources_owner.NativeCurveSource,
    chart_rows: np.ndarray,
    /,
) -> np.ndarray:
    if isinstance(source.curves, SegmentMesh):
        vertex_rows = np.unique(
            np.asarray(source.curves.edges, dtype=np.int64)[chart_rows]
        )
        return np.asarray(source.vertex_entity_ids, dtype=np.int64)[vertex_rows]
    keys: set[str | int] = set()
    for chart in chart_rows.tolist():
        if source.gap_domain is not None:
            curve_key, curve_occurrence = source.source_entity_key_for_scope(
                1, int(source.chart_scope_entity_ids[chart])
            )
            domain_slot = source.gap_domain.resolve_entity_id(
                1, curve_key, occurrence_path=curve_occurrence
            )
            curve = source.gap_domain.curves[domain_slot]
            keys.update(
                (
                    source.gap_domain.entity_id(0, curve.start),
                    source.gap_domain.entity_id(0, curve.end),
                )
            )
        elif source.chart_endpoint_ids:
            keys.update(
                key for key in source.chart_endpoint_ids[chart] if key is not None
            )
    bank = {
        key: int(entity_id)
        for key, entity_id in zip(
            source.vertex_entity_keys,
            np.asarray(source.vertex_entity_ids, dtype=np.int64).tolist(),
            strict=True,
        )
    }
    return np.asarray(sorted(bank[key] for key in keys if key in bank), dtype=np.int64)


def curve_support_issues(
    source: sources_owner.NativeCurveSource,
    specification: CurveMeshingSpec,
    /,
) -> list[str]:
    """Return only physical requirements outside the affine interval route."""
    issues: list[str] = []
    target = specification.target
    families = target.cell_families
    if (
        target.topological_dimension != 1
        or target.ambient_dimension != source.ambient_dimension
        or target.geometry_order != 1
        or set((*families.required, *families.preferred)) != {"interval"}
        or families.allow_mixed
        or families.allowed_transitions
    ):
        issues.append(
            "a conforming affine interval target in the source ambient dimension"
        )
    scope = specification.scope
    curve_scope_entity_set_id = source.scope_entity_set_id(1)
    selected = np.asarray(scope.entity_ids, dtype=np.int64)
    source_scope_ids = np.asarray(source.scope_entity_ids(1), dtype=np.int64)
    chart_scope_ids = np.asarray(source.chart_scope_entity_ids, dtype=np.int64)
    if (
        (scope.source_id, scope.source_revision)
        != (source.source_id, source.source_revision)
        or scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != 1
        or scope.entity_set_id != curve_scope_entity_set_id
        or selected.size == 0
        or not np.all(np.isin(selected, source_scope_ids))
        or not np.all(np.isin(selected, chart_scope_ids))
    ):
        issues.append("source-bound curve scope IDs in the current source revision")
    covered = np.zeros(selected.shape, dtype=np.bool_)
    for control in specification.size_controls:
        if not isinstance(control, (UniformSizeControl, CurvatureSizeControl)):
            issues.append("uniform or curvature controls on source curves")
            continue
        control_scope = control.scope
        control_ids = np.asarray(control_scope.entity_ids, dtype=np.int64)
        if (
            (control_scope.source_id, control_scope.source_revision)
            != (source.source_id, source.source_revision)
            or control_scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or control_scope.entity_dimension != 1
            or control_scope.entity_set_id != curve_scope_entity_set_id
            or not np.all(np.isin(control_ids, source_scope_ids))
        ):
            issues.append("size controls scoped to source curve entities")
            continue
        covered |= np.isin(selected, control_ids)
    if selected.size and not np.all(covered):
        issues.append(
            "an explicit physical resolution control on every selected source curve"
        )
    for feature in specification.protected_features:
        feature_scope = feature.scope
        feature_ids = np.asarray(feature_scope.entity_ids, dtype=np.int64)
        if (feature_scope.source_id, feature_scope.source_revision) != (
            source.source_id,
            source.source_revision,
        ):
            issues.append("protected features bound to the current curve source revision")
        elif feature_scope.entity_kind is not MeshingEntityKind.GEOMETRY:
            issues.append("protected features on source geometry entities")
        elif feature.feature_kind is FeatureKind.CURVE:
            if (
                feature_scope.entity_dimension != 1
                or feature_scope.entity_set_id != curve_scope_entity_set_id
                or not np.all(np.isin(feature_ids, selected))
            ):
                issues.append("curve features on selected source scope entities")
        elif feature.feature_kind is FeatureKind.CORNER:
            vertex_scope_entity_set_id = source.scope_entity_set_id(0)
            source_vertex_scope_ids = np.asarray(
                source.scope_entity_ids(0), dtype=np.int64
            )
            if (
                feature_scope.entity_dimension != 0
                or feature_scope.entity_set_id != vertex_scope_entity_set_id
                or not np.all(np.isin(feature_ids, source_vertex_scope_ids))
            ):
                issues.append("corner features on the source-owned vertex scope")
            else:
                chart_rows = np.flatnonzero(np.isin(chart_scope_ids, selected))
                source_vertices = _selected_source_vertex_ids(source, chart_rows)
                if not source_vertices.size:
                    issues.append(
                        "corner feature identities require source endpoint incidence"
                    )
                elif not np.all(np.isin(feature_ids, source_vertices)):
                    issues.append("corner features on selected source vertices")
        else:
            issues.append(f"{feature.feature_kind.value} protected features")
    return list(dict.fromkeys(issues))


def _integrand(
    prepared: PreparedCurveNetwork,
    row: int,
    parameters: np.ndarray,
    /,
    *,
    density: bool,
) -> np.ndarray:
    curve_density, speed = prepared.sizing.density(
        prepared.charts, row, jnp.asarray(parameters, dtype=jnp.float64)
    )
    values = curve_density if density else speed
    result = np.asarray(jax.device_get(values), dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "curve density must remain finite and non-negative on the source chart",
            provider_code="native_curve_unresolved",
            stage="curve_meshing",
            entity_ids=(int(prepared.scope_entity_ids[row]),),
        )
    return result


def _panel_integral(
    prepared: PreparedCurveNetwork,
    row: int,
    lower: float,
    upper: float,
    order: int,
    /,
    *,
    density: bool,
) -> tuple[float, float]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = 0.5 * (upper - lower)
    parameters = lower + half * (nodes + 1.0)
    coarse = half * float(
        np.dot(weights, _integrand(prepared, row, parameters, density=density))
    )
    fine_nodes, fine_weights = np.polynomial.legendre.leggauss(2 * order)
    fine_parameters = lower + half * (fine_nodes + 1.0)
    fine = half * float(
        np.dot(fine_weights, _integrand(prepared, row, fine_parameters, density=density))
    )
    return fine, abs(fine - coarse)


def _adaptive_integral(
    prepared: PreparedCurveNetwork,
    row: int,
    lower: float,
    upper: float,
    tolerance: float,
    /,
    *,
    density: bool,
    intervals: int,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Refine Gauss panels until the physical request's error estimate is met.

    Returns the integral, its error estimate, the panel breaks, and the
    cumulative integral and cumulative error estimate at every break.
    """
    if lower == upper:
        zeros = np.zeros((2,), dtype=np.float64)
        return 0.0, 0.0, np.asarray((lower, upper)), zeros, zeros
    pieces = np.linspace(lower, upper, intervals + 1)
    panels = [
        (
            float(a),
            float(b),
            *_panel_integral(
                prepared,
                row,
                float(a),
                float(b),
                prepared.schedule.quadrature_order,
                density=density,
            ),
        )
        for a, b in zip(pieces[:-1], pieces[1:], strict=True)
    ]
    while True:
        total = math.fsum(panel[2] for panel in panels)
        error = math.fsum(panel[3] for panel in panels)
        if error <= tolerance:
            breaks = np.asarray(
                (panels[0][0], *(panel[1] for panel in panels)), dtype=np.float64
            )
            cumulative = np.concatenate(
                ([0.0], np.cumsum([panel[2] for panel in panels]))
            )
            cumulative_errors = np.concatenate(
                ([0.0], np.cumsum([panel[3] for panel in panels]))
            )
            return total, error, breaks, cumulative, cumulative_errors
        index = max(range(len(panels)), key=lambda item: panels[item][3])
        start, stop, _, _ = panels[index]
        middle = start + 0.5 * (stop - start)
        if middle <= start or middle >= stop:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "curve quadrature reached parameter representability before meeting the physical tolerance",
                provider_code="native_curve_unresolved",
                stage="curve_meshing",
                entity_ids=(int(prepared.scope_entity_ids[row]),),
            )
        left = (
            start,
            middle,
            *_panel_integral(
                prepared,
                row,
                start,
                middle,
                prepared.schedule.quadrature_order,
                density=density,
            ),
        )
        right = (
            middle,
            stop,
            *_panel_integral(
                prepared,
                row,
                middle,
                stop,
                prepared.schedule.quadrature_order,
                density=density,
            ),
        )
        panels[index : index + 1] = [left, right]


class _DensityTable(NamedTuple):
    """Adaptive cumulative size-density table of one source chart on ``[0, 1]``."""

    breaks: np.ndarray
    cumulative: np.ndarray
    cumulative_errors: np.ndarray

    @property
    def mass(self) -> float:
        return float(self.cumulative[-1])

    @property
    def error(self) -> float:
        return float(self.cumulative_errors[-1])


def _table_mass(
    prepared: PreparedCurveNetwork,
    row: int,
    table: _DensityTable,
    parameter: float,
    tolerance: float,
    /,
) -> tuple[float, float, float]:
    """Density mass on ``[0, parameter]`` and its two error parts.

    The mass is the table prefix through the last whole panel plus one
    in-panel integral. The error is returned as the prefix's cumulative panel
    estimate and the in-panel estimate, so differences of two masses can
    cancel the shared prefix exactly.
    """
    panel = int(np.searchsorted(table.breaks, parameter, side="right")) - 1
    if panel >= table.breaks.size - 1:
        return table.mass, table.error, 0.0
    start = float(table.breaks[panel])
    prefix = float(table.cumulative[panel])
    prefix_error = float(table.cumulative_errors[panel])
    if parameter == start:
        return prefix, prefix_error, 0.0
    local, local_error, _, _, _ = _adaptive_integral(
        prepared, row, start, parameter, tolerance, density=True, intervals=1
    )
    return prefix + local, prefix_error, local_error


def _place(
    prepared: PreparedCurveNetwork,
    row: int,
    table: _DensityTable,
    lower: float,
    upper: float,
    fraction: float,
    /,
) -> float:
    """Invert cumulative size density by bracketed bisection without a step ceiling.

    Every bisection step integrates only inside one panel of the chart's
    density table, so a placement costs one short integral per step rather
    than a full re-integration of ``[lower, middle]``.
    """
    tolerance = prepared.tolerance(row)
    lower_mass, lower_prefix_error, lower_local_error = _table_mass(
        prepared, row, table, lower, tolerance
    )
    upper_mass, upper_prefix_error, upper_local_error = _table_mass(
        prepared, row, table, upper, tolerance
    )
    total = upper_mass - lower_mass
    total_error = (
        upper_prefix_error - lower_prefix_error + upper_local_error + lower_local_error
    )
    if not math.isfinite(total) or total <= 0.0:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "curve size density has no positive finite mass on a requested interval",
            provider_code="native_curve_unresolved",
            stage="curve_meshing",
            entity_ids=(int(prepared.scope_entity_ids[row]),),
        )
    target = fraction * total
    left, right = lower, upper
    while True:
        middle = left + 0.5 * (right - left)
        if middle <= left or middle >= right:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "curve density inversion reached parameter representability before meeting the physical tolerance",
                provider_code="native_curve_unresolved",
                stage="curve_meshing",
                entity_ids=(int(prepared.scope_entity_ids[row]),),
            )
        mass, prefix_error, local_error = _table_mass(
            prepared, row, table, middle, tolerance
        )
        partial = mass - lower_mass
        error = prefix_error - lower_prefix_error + local_error + lower_local_error
        residual = partial - target
        if abs(residual) + error + fraction * total_error <= tolerance:
            return middle
        if residual < 0.0:
            left = middle
        else:
            right = middle


def _density_tables(prepared: PreparedCurveNetwork, /) -> tuple[_DensityTable, ...]:
    tables: list[_DensityTable] = []
    for row in range(prepared.curve_entity_ids.size):
        total, _, breaks, cumulative, cumulative_errors = _adaptive_integral(
            prepared,
            row,
            0.0,
            1.0,
            prepared.tolerance(row),
            density=True,
            intervals=prepared.schedule.quadrature_intervals,
        )
        if total < 0.0 or not math.isfinite(total):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "curve size density integral must be finite and non-negative",
                provider_code="native_curve_unresolved",
                stage="curve_meshing",
                entity_ids=(int(prepared.scope_entity_ids[row]),),
            )
        tables.append(_DensityTable(breaks, cumulative, cumulative_errors))
    return tuple(tables)


def _source_fidelity_tolerance(specification: CurveMeshingSpec, /) -> float | None:
    return specification.maximum_deviation


def _deviation_limits(prepared: PreparedCurveNetwork, /) -> np.ndarray:
    requested = _source_fidelity_tolerance(prepared.specification)
    limits = np.full(
        (prepared.curve_entity_ids.size,), math.inf if requested is None else requested
    )
    for feature in prepared.specification.protected_features:
        if not feature.hard or feature.feature_kind is not FeatureKind.CURVE:
            continue
        selected = np.isin(
            prepared.scope_entity_ids,
            np.asarray(feature.scope.entity_ids, dtype=np.int64),
        )
        limits[selected] = np.minimum(limits[selected], feature.maximum_deviation)
    return limits


def _linear_chord_bound(
    prepared: PreparedCurveNetwork,
    row: int,
    lower: float,
    upper: float,
    first: np.ndarray,
    last: np.ndarray,
    /,
) -> float:
    """Exact endpoint error bounds for a linear source chart."""
    if not isinstance(prepared.source.curves, SegmentMesh):
        raise TypeError("Linear chord bounds require a SegmentMesh source.")
    vertices = np.asarray(prepared.source.curves.vertices, dtype=np.float64)
    endpoints = np.asarray(prepared.source.curves.edges, dtype=np.int64)[
        prepared.chart_rows[row]
    ]
    start = tuple(Fraction(float(value)) for value in vertices[endpoints[0]])
    end = tuple(Fraction(float(value)) for value in vertices[endpoints[1]])
    first_parameter = 1.0 - lower if prepared.charts.chart_reversed[row] else lower
    last_parameter = 1.0 - upper if prepared.charts.chart_reversed[row] else upper

    def endpoint_error(parameter: float, published: np.ndarray, /) -> float:
        exact = tuple(
            a + Fraction(float(parameter)) * (b - a)
            for a, b in zip(start, end, strict=True)
        )
        squared = sum(
            (Fraction(float(value)) - expected) ** 2
            for value, expected in zip(published, exact, strict=True)
        )
        return (
            0.0
            if squared == 0
            else float(np.nextafter(math.sqrt(float(squared)), math.inf))
        )

    return max(
        endpoint_error(first_parameter, first),
        endpoint_error(last_parameter, last),
    )


def _chord_bound(
    prepared: PreparedCurveNetwork,
    row: int,
    lower: float,
    upper: float,
    first: np.ndarray,
    last: np.ndarray,
    /,
) -> float:
    """Continuous two-sided curve-to-chord bound, or infinity if unsupported."""
    if isinstance(prepared.source.curves, SegmentMesh):
        return _linear_chord_bound(prepared, row, lower, upper, first, last)
    enclosure = prepared.enclosures[row]
    if not enclosure.rigorous:
        return math.inf
    acceleration = enclosure.acceleration_upper(lower, upper)
    endpoint_error = max(
        enclosure.point_error(lower, first), enclosure.point_error(upper, last)
    )
    width = float(binary64_difference_magnitude(upper, lower)[1])
    curvature = math.nextafter(
        math.nextafter(0.125 * math.nextafter(width * width, math.inf), math.inf)
        * acceleration,
        math.inf,
    )
    return math.nextafter(endpoint_error + curvature, math.inf)


def _sampled_deviation(
    prepared: PreparedCurveNetwork,
    row: int,
    lower: float,
    upper: float,
    first: np.ndarray,
    last: np.ndarray,
    /,
) -> float:
    samples = prepared.schedule.fidelity_samples
    fractions = np.arange(1, samples + 1, dtype=np.float64) / (samples + 1)
    points = prepared.points(row, lower + fractions * (upper - lower))
    chord = last - first
    squared = float(np.dot(chord, chord))
    along = (
        np.zeros((points.shape[0],), dtype=np.float64)
        if squared == 0.0
        else np.clip(((points - first) @ chord) / squared, 0.0, 1.0)
    )
    distance = np.linalg.norm(points - (first + along[:, None] * chord), axis=1)
    return float(np.max(distance, initial=0.0))


def _discretize(
    prepared: PreparedCurveNetwork,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tables = _density_tables(prepared)
    integration_errors = np.asarray([table.error for table in tables], dtype=np.float64)
    limits = _deviation_limits(prepared)
    parameters_by_chart: list[list[float]] = [
        [0.0, 1.0] for _ in prepared.curve_entity_ids
    ]
    scope_rows: dict[int, list[int]] = {}
    for row, scope_id in enumerate(prepared.scope_entity_ids.tolist()):
        scope_rows.setdefault(int(scope_id), []).append(row)
    scope_ends: dict[int, tuple[int, int]] = {}
    for scope_id, rows in scope_rows.items():
        scope_ends[scope_id] = (
            int(prepared.endpoints[rows[0], 0]),
            int(prepared.endpoints[rows[-1], 1]),
        )
    parallel_counts: dict[tuple[int, int], int] = {}
    for start, end in scope_ends.values():
        if start >= 0 and end >= 0 and start != end:
            pair = (min(start, end), max(start, end))
            parallel_counts[pair] = parallel_counts.get(pair, 0) + 1
    for scope_id, rows in scope_rows.items():
        masses = [tables[row].mass for row in rows]
        for row, mass in zip(rows, masses, strict=True):
            if mass <= 0.0:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "a source curve chart has zero size-density measure",
                    provider_code="native_curve_unresolved",
                    stage="curve_meshing",
                    entity_ids=(int(prepared.scope_entity_ids[row]),),
                )
        total = math.fsum(masses)
        if not math.isfinite(total) or total < 0.0:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "curve size density integral must be finite and non-negative",
                provider_code="native_curve_unresolved",
                stage="curve_meshing",
                entity_ids=(scope_id,),
            )
        start, end = scope_ends[scope_id]
        minimum = len(rows)
        if start >= 0 and start == end:
            minimum = max(minimum, 3)
        elif start >= 0 and end >= 0:
            minimum = max(
                minimum,
                2
                if parallel_counts.get((min(start, end), max(start, end)), 0) > 1
                else 1,
            )
        count = max(math.ceil(total), minimum)
        prefixes = np.concatenate(([0.0], np.cumsum(masses)))
        for index in range(1, count):
            target = total * index / count
            chart = int(np.searchsorted(prefixes, target, side="right") - 1)
            chart = min(chart, len(rows) - 1)
            local_target = target - float(prefixes[chart])
            if local_target == 0.0 or local_target == masses[chart]:
                continue
            row = rows[chart]
            parameters_by_chart[row].append(
                _place(prepared, row, tables[row], 0.0, 1.0, local_target / masses[chart])
            )
        for row in rows:
            parameters = parameters_by_chart[row]
            while True:
                parameters.sort()
                split_parameters: list[float] = []
                for lower, upper in zip(parameters[:-1], parameters[1:], strict=True):
                    first, last = prepared.points(row, np.asarray((lower, upper)))
                    bound = _chord_bound(prepared, row, lower, upper, first, last)
                    if bound > limits[row]:
                        if not math.isfinite(bound):
                            raise MeshingFailure(
                                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                                "continuous curve fidelity is unavailable for the requested physical deviation",
                                provider_code="native_curve_unresolved",
                                stage="curve_meshing",
                                entity_ids=(scope_id,),
                            )
                        split_parameters.append(
                            _place(prepared, row, tables[row], lower, upper, 0.5)
                        )
                if not split_parameters:
                    break
                existing = set(parameters)
                if any(value in existing for value in split_parameters):
                    raise MeshingFailure(
                        MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        "curve chord refinement reached parameter representability before meeting the physical deviation",
                        provider_code="native_curve_unresolved",
                        stage="curve_meshing",
                        entity_ids=(scope_id,),
                    )
                parameters.extend(split_parameters)
    rows_out: list[int] = []
    parameters_out: list[float] = []
    for row, parameters in enumerate(parameters_by_chart):
        parameters.sort()
        rows_out.extend([row] * len(parameters))
        parameters_out.extend(parameters)
    rows = np.asarray(rows_out, dtype=np.int64)
    parameters = np.asarray(parameters_out, dtype=np.float64)
    order = np.lexsort((parameters, rows))
    rows, parameters = rows[order], parameters[order]
    return rows, parameters, integration_errors, limits


def _assemble(
    prepared: PreparedCurveNetwork,
    rows: np.ndarray,
    parameters: np.ndarray,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[tuple[int, ...], ...],
    np.ndarray,
]:
    """Build intervals and retain source-vertex identity without coordinate welding."""
    source_vertex_ids_by_key = {
        key: int(identifier)
        for key, identifier in zip(
            prepared.source.vertex_entity_keys,
            np.asarray(prepared.source.vertex_entity_ids, dtype=np.int64).tolist(),
            strict=True,
        )
    }
    points = np.empty((rows.size, prepared.charts.ambient_dimension), dtype=np.float64)
    source_vertices_at_points = np.full(rows.shape, -1, dtype=np.int64)
    for row in np.unique(rows):
        selected = np.flatnonzero(rows == row)
        points[selected] = prepared.points(int(row), parameters[selected])
        source_vertices_at_points[selected[0]] = _source_vertex_index(
            prepared, int(row), 0, source_vertex_ids_by_key
        )
        source_vertices_at_points[selected[-1]] = _source_vertex_index(
            prepared, int(row), 1, source_vertex_ids_by_key
        )
    starts_curve = np.concatenate(([True], rows[1:] != rows[:-1]))
    ends_curve = np.concatenate((rows[1:] != rows[:-1], [True]))
    endpoint_groups = np.full((rows.size,), -1, dtype=np.int64)
    endpoint_groups[starts_curve] = prepared.endpoints[rows[starts_curve], 0]
    endpoint_groups[ends_curve] = prepared.endpoints[rows[ends_curve], 1]
    vertex_rows = np.empty(rows.size, dtype=np.int64)
    coordinates: list[np.ndarray] = []
    vertex_groups: list[int] = []
    vertex_sources: list[set[int]] = []
    group_to_vertex: dict[int, int] = {}
    for index, group in enumerate(endpoint_groups.tolist()):
        source_vertex = int(source_vertices_at_points[index])
        if group >= 0 and group in group_to_vertex:
            vertex = group_to_vertex[group]
            vertex_rows[index] = vertex
            if source_vertex >= 0:
                vertex_sources[vertex].add(source_vertex)
            continue
        vertex = len(coordinates)
        vertex_rows[index] = vertex
        if group >= 0:
            group_to_vertex[group] = vertex
            coordinates.append(prepared.junction_points[group])
        else:
            coordinates.append(points[index])
        vertex_groups.append(group)
        vertex_sources.append(set() if source_vertex < 0 else {source_vertex})
    adjacent = rows[1:] == rows[:-1]
    cells = np.stack(
        (vertex_rows[:-1][adjacent], vertex_rows[1:][adjacent]), axis=1
    ).astype(np.int32)
    cell_rows = rows[:-1][adjacent]
    parameters_at_cells = np.stack(
        (parameters[:-1][adjacent], parameters[1:][adjacent]), axis=1
    )
    return (
        np.asarray(coordinates, dtype=np.float64),
        cells,
        cell_rows,
        parameters_at_cells,
        np.asarray(vertex_groups, dtype=np.int64),
        tuple(tuple(sorted(values)) for values in vertex_sources),
        points,
    )


def _source_vertex_numeric_id(
    source_vertex_ids_by_key: dict[str | int, int],
    key: str | int,
    /,
) -> int:
    try:
        return source_vertex_ids_by_key[key]
    except KeyError as error:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "a source curve endpoint key is absent from its authoritative vertex bank",
            provider_code="native_curve_association",
            stage="geometry_association",
        ) from error


def _source_vertex_index(
    prepared: PreparedCurveNetwork,
    row: int,
    side: int,
    source_vertex_ids_by_key: dict[str | int, int],
    /,
) -> int:
    source = prepared.source
    chart = int(prepared.chart_rows[row])
    local_side = 1 - side if prepared.charts.chart_reversed[row] else side
    if isinstance(source.curves, SegmentMesh):
        vertex = int(np.asarray(source.curves.edges, dtype=np.int64)[chart, local_side])
        return int(source.vertex_entity_ids[vertex])
    if source.gap_domain is not None:
        curve_key, curve_occurrence = source.source_entity_key_for_scope(
            1, int(source.chart_scope_entity_ids[chart])
        )
        domain_slot = source.gap_domain.resolve_entity_id(
            1, curve_key, occurrence_path=curve_occurrence
        )
        curve = source.gap_domain.curves[domain_slot]
        corner = curve.start if local_side == 0 else curve.end
        key = source.gap_domain.entity_id(0, corner)
        return _source_vertex_numeric_id(source_vertex_ids_by_key, key)
    if source.chart_endpoint_ids:
        key = source.chart_endpoint_ids[chart][local_side]
        return (
            -1
            if key is None
            else _source_vertex_numeric_id(source_vertex_ids_by_key, key)
        )
    return -1


def _mesh_scope(
    mesh: cell_mesh_owner.CellMesh, dimension: int, ids: np.ndarray, /
) -> MeshingScope:
    entity_set = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entity_set.entity_set_id,
        ids,
    )


def _organization(
    mesh: cell_mesh_owner.CellMesh,
    prepared: PreparedCurveNetwork,
    cell_rows: np.ndarray,
    cell_parameters: np.ndarray,
    deviation_bounds: np.ndarray,
    vertex_junctions: np.ndarray,
    vertex_source_vertices: tuple[tuple[int, ...], ...],
    /,
) -> NativeMeshOrganization:
    cell_set = mesh.entity_set(1)
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    curve_scope_ids = tuple(dict.fromkeys(prepared.scope_entity_ids.tolist()))
    patches = tuple(
        MeshPatch(
            f"curve:{scope_id}",
            _mesh_scope(
                mesh,
                1,
                cell_ids[np.isin(prepared.scope_entity_ids[cell_rows], (scope_id,))],
            ),
        )
        for scope_id in curve_scope_ids
    )
    curve_source_keys = {
        int(scope_id): prepared.source.source_entity_key_for_scope(1, int(scope_id))
        for scope_id in np.unique(prepared.scope_entity_ids).tolist()
    }
    labels: list[MeshLabel] = [
        MeshLabel(
            name,
            _mesh_scope(
                mesh,
                0,
                np.asarray(
                    (vertex_ids[np.flatnonzero(vertex_junctions == row)[0]],),
                    dtype=np.int64,
                ),
            ),
        )
        for row, name in enumerate(prepared.junction_names)
        if name is not None
    ]
    for feature in prepared.specification.protected_features:
        if feature.feature_kind is FeatureKind.CURVE:
            selected = np.isin(
                prepared.scope_entity_ids[cell_rows],
                np.asarray(feature.scope.entity_ids, dtype=np.int64),
            )
            rows = np.flatnonzero(selected)
            labels.append(
                MeshLabel(
                    f"feature:{feature.feature_id}",
                    _mesh_scope(mesh, 1, cell_ids[rows]),
                )
            )
            continue
        if feature.feature_kind is not FeatureKind.CORNER:
            continue
        feature_ids = set(
            int(value) for value in np.asarray(feature.scope.entity_ids, dtype=np.int64)
        )
        selected = np.asarray(
            [
                any(source_vertex in feature_ids for source_vertex in sources)
                for sources in vertex_source_vertices
            ],
            dtype=np.bool_,
        )
        rows = np.flatnonzero(selected)
        if not rows.size:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                f"protected curve corner {feature.feature_id!r} is absent from the constructed source network",
                provider_code="native_curve_junction",
                stage="geometry_association",
            )
        labels.append(
            MeshLabel(
                f"feature:{feature.feature_id}",
                _mesh_scope(mesh, 0, vertex_ids[rows]),
            )
        )

    def source_parameter(row: int, value: float, /) -> Fraction:
        lower, upper = sorted(
            Fraction(float(item)) for item in prepared.source_ranges[row]
        )
        return lower + Fraction(float(value)) * (upper - lower)

    represented_intervals = tuple(
        (
            source_parameter(row, lower),
            source_parameter(row, upper),
        )
        for row, (lower, upper) in zip(
            cell_rows.tolist(), cell_parameters.tolist(), strict=True
        )
    )
    source_parameter_values = np.asarray(
        [
            (
                float(first),
                float(last),
            )
            for first, last in represented_intervals
        ],
        dtype=np.float64,
    )
    exact_intervals = tuple(
        interval if np.isfinite(deviation_bounds[row]) else None
        for row, interval in enumerate(represented_intervals)
    )
    association = association_owner.GeometryAssociation(
        association_owner.GeometryAssociationKind.CURVE,
        prepared.source.source_id,
        prepared.source.source_revision,
        cell_set.entity_set_id,
        cell_ids,
        tuple(
            curve_source_keys[int(prepared.scope_entity_ids[row])][0]
            for row in cell_rows.tolist()
        ),
        np.where(np.isfinite(deviation_bounds), deviation_bounds, np.inf),
        resolved=np.isfinite(deviation_bounds),
        exact=bool(np.all(deviation_bounds == 0.0)),
        source_roles=(association_owner.GeometrySourceEntityRole.EDGE,) * cell_ids.size,
        source_occurrence_paths=tuple(
            curve_source_keys[int(prepared.scope_entity_ids[row])][1]
            for row in cell_rows.tolist()
        ),
        parameters=np.stack(
            (
                np.mean(source_parameter_values, axis=1),
                np.full((cell_ids.size,), np.nan, dtype=np.float64),
            ),
            axis=1,
        ),
        orientations=np.ones((cell_ids.size,), dtype=np.int8),
        source_parameter_intervals=exact_intervals,
    )
    vertex_set = mesh.entity_set(0)
    cell_vertices = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    vertex_parameters: list[dict[int, set[Fraction]]] = [
        {} for _ in range(prepared.curve_entity_ids.size)
    ]
    for cell, (row, endpoints) in enumerate(
        zip(cell_rows.tolist(), cell_parameters.tolist(), strict=True)
    ):
        for vertex, value in zip(cell_vertices[cell], endpoints, strict=True):
            vertex_row = int(vertex)
            parameter = source_parameter(row, float(value))
            vertex_parameters[row].setdefault(vertex_row, set()).add(parameter)

    def curve_vertex_association(
        row: int,
        rows: np.ndarray,
        point_parameters: tuple[Fraction, ...],
        /,
    ) -> association_owner.GeometryAssociation:
        count = rows.size
        return association_owner.GeometryAssociation(
            association_owner.GeometryAssociationKind.CURVE,
            prepared.source.source_id,
            prepared.source.source_revision,
            vertex_set.entity_set_id,
            vertex_ids[rows],
            (curve_source_keys[int(prepared.scope_entity_ids[row])][0],) * count,
            np.zeros((count,), dtype=np.float64),
            exact=True,
            source_roles=(association_owner.GeometrySourceEntityRole.EDGE,) * count,
            source_occurrence_paths=(
                curve_source_keys[int(prepared.scope_entity_ids[row])][1],
            )
            * count,
            parameters=np.stack(
                (
                    np.asarray(
                        [float(value) for value in point_parameters],
                        dtype=np.float64,
                    ),
                    np.full((count,), np.nan, dtype=np.float64),
                ),
                axis=1,
            ),
            source_parameter_points=point_parameters,
        )

    vertex_associations: list[association_owner.GeometryAssociation] = []
    for row, vertex_map in enumerate(vertex_parameters):
        single_rows: list[int] = []
        single_parameters: list[Fraction] = []
        for vertex, values in sorted(vertex_map.items()):
            ordered = sorted(values)
            if len(ordered) == 1:
                single_rows.append(vertex)
                single_parameters.append(ordered[0])
            else:
                for parameter in ordered:
                    vertex_associations.append(
                        curve_vertex_association(
                            row,
                            np.asarray((vertex,), dtype=np.int64),
                            (parameter,),
                        )
                    )
        if single_rows:
            vertex_associations.append(
                curve_vertex_association(
                    row,
                    np.asarray(single_rows, dtype=np.int64),
                    tuple(single_parameters),
                )
            )
    source_vertex_rows: dict[int, list[int]] = {}
    for vertex_row, source_ids in enumerate(vertex_source_vertices):
        for source_id in source_ids:
            source_vertex_rows.setdefault(int(source_id), []).append(vertex_row)
    source_vertex_keys = {
        int(source_id): prepared.source.source_entity_key_for_scope(0, int(source_id))
        for source_id in source_vertex_rows
    }
    source_vertex_associations: list[association_owner.GeometryAssociation] = []
    for source_id, mesh_rows in sorted(source_vertex_rows.items()):
        source_entity_key = source_vertex_keys[source_id]
        if len(mesh_rows) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "one source vertex maps to multiple constructed mesh vertices",
                provider_code="native_curve_association",
                stage="geometry_association",
            )
        source_vertex_associations.append(
            association_owner.GeometryAssociation(
                association_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                prepared.source.source_id,
                prepared.source.source_revision,
                vertex_set.entity_set_id,
                vertex_ids[np.asarray(mesh_rows, dtype=np.int64)],
                (source_entity_key[0],),
                np.zeros((1,), dtype=np.float64),
                exact=True,
                source_roles=(association_owner.GeometrySourceEntityRole.VERTEX,),
                source_occurrence_paths=(source_entity_key[1],),
            )
        )

    return NativeMeshOrganization(
        patches=patches,
        labels=tuple(labels),
        associations=(
            association,
            *vertex_associations,
            *source_vertex_associations,
        ),
    )


def _curve_compliance(
    prepared: PreparedCurveNetwork,
    vertices: np.ndarray,
    cells: np.ndarray,
    cell_rows: np.ndarray,
    parameters: np.ndarray,
    deviation_bounds: np.ndarray,
    witnessed_deviations: np.ndarray,
    integration_errors: np.ndarray,
    vertex_source_vertices: tuple[tuple[int, ...], ...],
    vertex_junctions: np.ndarray,
    /,
) -> MeshingComplianceReport:
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    arc_lengths = np.empty((cells.shape[0],), dtype=np.float64)
    for cell, (row, (lower, upper)) in enumerate(
        zip(cell_rows.tolist(), parameters.tolist(), strict=True)
    ):
        arc_lengths[cell] = _adaptive_integral(
            prepared,
            row,
            lower,
            upper,
            prepared.physical_tolerance(row),
            density=False,
            intervals=prepared.schedule.quadrature_intervals,
        )[0]
    for scope_id in dict.fromkeys(prepared.scope_entity_ids.tolist()):
        chart_rows = np.flatnonzero(prepared.scope_entity_ids == scope_id)
        achieved.append(
            (
                f"curve:{int(scope_id)}:density_integral_error",
                float(np.max(integration_errors[chart_rows], initial=0.0)),
            )
        )
    policy = prepared.specification.size_compliance
    for control in prepared.specification.size_controls:
        if not isinstance(control, (UniformSizeControl, CurvatureSizeControl)):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "curve compliance requires a supported size control",
                provider_code="native_curve_sizing",
                stage="specification_compliance",
            )
        key = f"size:{control.control_id}"
        selected = np.isin(
            prepared.scope_entity_ids[cell_rows],
            np.asarray(control.scope.entity_ids, dtype=np.int64),
        )
        lengths = arc_lengths[selected]
        if not lengths.size:
            issues.append(f"missing_controlled_intervals:{control.control_id}")
            continue
        if isinstance(control, UniformSizeControl):
            requested.append((f"{key}:target_size", control.target_size))
            statistics = {
                name: float(np.quantile(lengths, 0.5 if name == "p50" else 0.95))
                for name in policy.target_statistics
            }
            achieved.extend(
                (f"{key}:{name}_arc_length", value) for name, value in statistics.items()
            )
            tolerance = (
                policy.absolute_tolerance
                + policy.relative_tolerance * control.target_size
            )
            if control.strength is SizeControlStrength.HARD:
                issues.extend(
                    f"target_size_{name}:{control.control_id}"
                    for name, value in statistics.items()
                    if abs(value - control.target_size) > tolerance
                )
                if (
                    control.minimum_size is not None
                    and float(np.min(lengths)) < control.minimum_size - tolerance
                ):
                    issues.append(f"minimum_size:{control.control_id}")
                if (
                    control.maximum_size is not None
                    and float(np.max(lengths)) > control.maximum_size + tolerance
                ):
                    issues.append(f"maximum_size:{control.control_id}")
            if control.maximum_growth_rate is not None:
                requested.append(
                    (f"{key}:maximum_growth_rate", control.maximum_growth_rate)
                )
                growth = 1.0
                selected_rows = np.flatnonzero(selected)
                for first, second in zip(
                    selected_rows[:-1], selected_rows[1:], strict=True
                ):
                    if (
                        prepared.scope_entity_ids[cell_rows[first]]
                        == prepared.scope_entity_ids[cell_rows[second]]
                        and cells[first, 1] == cells[second, 0]
                    ):
                        growth = max(
                            growth,
                            float(
                                max(
                                    arc_lengths[first] / arc_lengths[second],
                                    arc_lengths[second] / arc_lengths[first],
                                )
                            ),
                        )
                achieved.append((f"{key}:maximum_interval_growth", growth))
                if (
                    control.strength is SizeControlStrength.HARD
                    and growth
                    > control.maximum_growth_rate
                    + policy.absolute_tolerance
                    + policy.relative_tolerance * control.maximum_growth_rate
                ):
                    issues.append(f"maximum_growth_rate:{control.control_id}")
        elif isinstance(control, CurvatureSizeControl):
            requested.append((f"{key}:normal_angle", control.normal_angle))
            local_rows = np.flatnonzero(selected)
            turns: list[float] = []
            for first, second in zip(local_rows[:-1], local_rows[1:], strict=True):
                if (
                    prepared.scope_entity_ids[cell_rows[first]]
                    != prepared.scope_entity_ids[cell_rows[second]]
                    or cells[first, 1] != cells[second, 0]
                ):
                    continue
                a = vertices[cells[first, 1]] - vertices[cells[first, 0]]
                b = vertices[cells[second, 1]] - vertices[cells[second, 0]]
                cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                turns.append(math.acos(float(np.clip(cosine, -1.0, 1.0))))
            turning = max(turns, default=0.0)
            achieved.append((f"{key}:maximum_chord_turning", turning))
            tolerance = (
                policy.absolute_tolerance
                + policy.relative_tolerance * control.normal_angle
            )
            if (
                control.strength is SizeControlStrength.HARD
                and turning > control.normal_angle + tolerance
            ):
                issues.append(f"normal_angle:{control.control_id}")
    for feature in prepared.specification.protected_features:
        key = f"protected:{feature.feature_id}:maximum_deviation"
        if feature.feature_kind is FeatureKind.CURVE:
            selected = np.isin(
                prepared.scope_entity_ids[cell_rows],
                np.asarray(feature.scope.entity_ids, dtype=np.int64),
            )
            achieved_value = float(np.max(deviation_bounds[selected], initial=0.0))
            witnessed = float(np.max(witnessed_deviations[selected], initial=0.0))
            achieved.append(
                (f"protected:{feature.feature_id}:witnessed_deviation", witnessed)
            )
        elif feature.feature_kind is FeatureKind.CORNER:
            selected_ids = {
                int(value) for value in np.asarray(feature.scope.entity_ids).tolist()
            }
            selected_vertices = [
                row
                for row, source_ids in enumerate(vertex_source_vertices)
                if selected_ids.intersection(source_ids)
            ]
            if not selected_vertices:
                issues.append(f"missing_protected_corners:{feature.feature_id}")
                continue
            achieved_value = float(
                max(
                    (
                        prepared.junction_spread[int(vertex_junctions[row])]
                        if vertex_junctions[row] >= 0
                        else 0.0
                    )
                    for row in selected_vertices
                )
            )
        else:
            continue
        requested.append((key, feature.maximum_deviation))
        achieved.append((key, achieved_value))
        if feature.hard and achieved_value > feature.maximum_deviation:
            issues.append(key)
    if prepared.specification.maximum_deviation is not None:
        maximum = float(np.max(deviation_bounds, initial=0.0))
        requested.append(("maximum_deviation", prepared.specification.maximum_deviation))
        achieved.append(("maximum_deviation", maximum))
        if maximum > prepared.specification.maximum_deviation:
            issues.append("maximum_deviation")
    return MeshingComplianceReport(
        prepared.specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _curve_chart_witness(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    query: ParametricCurveBoundarySource,
    prepared: PreparedCurveNetwork,
    cell_rows: np.ndarray,
    cell_parameters: np.ndarray,
    target_output_rows: np.ndarray | None = None,
    /,
) -> SourceBoundaryChartWitness:
    """Bind output intervals to a complete source-chart parameter partition."""
    rows = np.asarray(cell_rows, dtype=np.int64)
    local_parameters = np.asarray(cell_parameters, dtype=np.float64)
    if rows.ndim != 1 or local_parameters.shape != (rows.size, 2):
        raise ValueError("Curve source ancestry must align with output interval cells.")
    selected_rows = (
        np.arange(rows.size, dtype=np.int64)
        if target_output_rows is None
        else np.asarray(target_output_rows, dtype=np.int64)
    )
    if (
        selected_rows.ndim != 1
        or selected_rows.size == 0
        or np.any(selected_rows < 0)
        or np.any(selected_rows >= rows.size)
        or np.unique(selected_rows).size != selected_rows.size
    ):
        raise ValueError(
            "Curve witness output selection must contain unique nonempty cell rows."
        )
    rows = rows[selected_rows]
    local_parameters = local_parameters[selected_rows]
    chart_ids: list[str] = []
    entity_ids: list[str] = []
    occurrences: list[tuple[str, ...]] = []
    source_parameters = np.empty((rows.size, 2, 1), dtype=np.float64)
    for output_row, (chart_row, local_interval) in enumerate(
        zip(rows.tolist(), local_parameters, strict=True)
    ):
        if chart_row < 0 or chart_row >= len(prepared.chart_ids):
            raise ValueError("Curve source ancestry names an absent prepared chart.")
        numeric_id = int(prepared.scope_entity_ids[chart_row])
        entity_id, occurrence = prepared.source.source_entity_key_for_scope(1, numeric_id)
        first, last = (
            float(value) for value in np.asarray(prepared.source_ranges)[chart_row]
        )
        chart_ids.append(prepared.chart_ids[chart_row])
        entity_ids.append(entity_id)
        occurrences.append(occurrence)
        source_parameters[output_row, :, 0] = first + (last - first) * local_interval
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)[selected_rows]
    if cell_ids.shape != (rows.size,):
        raise ValueError(
            "Curve witness requires one interval cell per source-parameter row."
        )
    target_reference = np.broadcast_to(
        np.asarray(((0.0,), (1.0,)), dtype=np.float64),
        (rows.size, 2, 1),
    )
    orientations = np.where(
        source_parameters[:, 1, 0] > source_parameters[:, 0, 0], 1, -1
    ).astype(np.int8)
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.query_id,
        chart_dimension=1,
        reference_dimension=mesh.topological_dimension,
        cell_global_ids=cell_ids,
        entity_global_ids=cell_ids,
        local_entity_indices=np.full(rows.size, -1, dtype=np.int32),
        target_reference_vertices=target_reference,
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(entity_ids),
        occurrence_paths=tuple(occurrences),
        source_parameters=source_parameters,
        orientations=orientations,
    )


def _curve_point_witness(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    query: ParametricCurveBoundarySource,
    source: sources_owner.NativeCurveSource,
    requested_point_ids: np.ndarray,
    vertex_source_ids: tuple[tuple[int, ...], ...],
    /,
) -> SourceBoundaryPointWitness:
    identifiers = np.asarray(requested_point_ids, dtype=np.int64)
    if (
        identifiers.ndim != 1
        or identifiers.size == 0
        or np.unique(identifiers).size != identifiers.size
    ):
        raise ValueError("Curve point witness needs unique authored source point IDs.")
    if len(vertex_source_ids) != mesh.coordinates.shape[0]:
        raise ValueError("Curve point ancestry must align with target mesh vertices.")
    target_rows = []
    entity_ids = []
    occurrences = []
    for identifier in identifiers.tolist():
        matches = [
            row
            for row, source_ids in enumerate(vertex_source_ids)
            if int(identifier) in source_ids
        ]
        if len(matches) != 1:
            raise ValueError(
                "Every protected source point must map to one target vertex."
            )
        target_rows.append(matches[0])
        entity_id, occurrence = source.source_entity_key_for_scope(0, int(identifier))
        entity_ids.append(entity_id)
        occurrences.append(occurrence)
    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[
        np.asarray(target_rows, dtype=np.int64)
    ]
    if np.unique(target_ids).size != target_ids.size:
        raise ValueError(
            "Distinct protected source points cannot share one target vertex identity."
        )
    return SourceBoundaryPointWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.query_id,
        vertex_global_ids=target_ids,
        source_entity_ids=tuple(entity_ids),
        occurrence_paths=tuple(occurrences),
    )


@final
class _PreparedCurveRoute(StrictModule, NonTrainableState):
    source: sources_owner.NativeCurveSource
    specification: CurveMeshingSpec
    options: NativeMeshingOptions
    coordinate_contract: SpatialCoordinateContract
    prepared: PreparedCurveNetwork

    def __init__(
        self,
        source: sources_owner.NativeCurveSource,
        specification: CurveMeshingSpec,
        options: NativeMeshingOptions,
        coordinate_contract: SpatialCoordinateContract,
        prepared: PreparedCurveNetwork,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.options = options
        self.coordinate_contract = coordinate_contract
        self.prepared = prepared

    def execute(self, /) -> NativeMeshConstruction:
        prepared = self.prepared
        rows, parameters, integration_errors, limits = _discretize(prepared)
        (
            vertices,
            cells,
            cell_rows,
            cell_parameters,
            vertex_junctions,
            vertex_source_vertices,
            _,
        ) = _assemble(prepared, rows, parameters)
        bounds = np.empty((cells.shape[0],), dtype=np.float64)
        witnessed = np.empty((cells.shape[0],), dtype=np.float64)
        for cell, (row, (lower, upper)) in enumerate(
            zip(cell_rows.tolist(), cell_parameters.tolist(), strict=True)
        ):
            first, last = vertices[cells[cell]]
            bound = _chord_bound(prepared, row, lower, upper, first, last)
            if math.isfinite(limits[row]) and not math.isfinite(bound):
                raise MeshingFailure(
                    MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    "the requested curve deviation has no continuous source-interval enclosure",
                    provider_code="native_curve_unresolved",
                    stage="curve_meshing",
                    entity_ids=(int(prepared.scope_entity_ids[row]),),
                )
            bounds[cell] = bound
            witnessed[cell] = _sampled_deviation(prepared, row, lower, upper, first, last)
        mesh = cell_mesh_owner.CellMesh(
            vertices,
            (cell_mesh_owner.CellBlock("intervals", "interval", cells),),
            numeric_version=self.source.source_revision,
        )
        organization = _organization(
            mesh,
            prepared,
            cell_rows,
            cell_parameters,
            bounds,
            vertex_junctions,
            vertex_source_vertices,
        )
        compliance = _curve_compliance(
            prepared,
            vertices,
            cells,
            cell_rows,
            cell_parameters,
            bounds,
            witnessed,
            integration_errors,
            vertex_source_vertices,
            vertex_junctions,
        )
        geometry = CellGeometrySpec.affine(mesh)
        fidelity_tolerance = _source_fidelity_tolerance(self.specification)
        fidelity_source = (
            None if fidelity_tolerance is None else _curve_source_query(prepared)
        )
        junction_rows = np.flatnonzero(vertex_junctions >= 0)
        junction_vertices = np.asarray(
            [np.asarray(mesh.vertex_global_ids)[row] for row in junction_rows],
            dtype=np.int64,
        )
        scoped_fidelity = []
        for feature in self.specification.protected_features:
            if not feature.hard:
                continue
            feature_ids = np.asarray(feature.scope.entity_ids, dtype=np.int64)
            if feature.feature_kind is FeatureKind.CURVE:
                feature_query = _curve_source_query(prepared, feature_ids)
                selected_cells = np.flatnonzero(
                    np.isin(prepared.scope_entity_ids[cell_rows], feature_ids)
                )
                feature_witness = _curve_chart_witness(
                    mesh,
                    geometry,
                    feature_query,
                    prepared,
                    cell_rows,
                    cell_parameters,
                    selected_cells,
                )
                scoped_fidelity.append(
                    (feature_query, feature_witness, feature.maximum_deviation)
                )
            elif feature.feature_kind is FeatureKind.CORNER:
                feature_query = _curve_source_query(prepared)
                feature_witness = _curve_point_witness(
                    mesh,
                    geometry,
                    feature_query,
                    self.source,
                    feature_ids,
                    vertex_source_vertices,
                )
                scoped_fidelity.append(
                    (feature_query, feature_witness, feature.maximum_deviation)
                )
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("curve"),
            self.source.source_id,
            self.source.source_revision,
            fidelity_source=fidelity_source,
            fidelity_tolerance=fidelity_tolerance,
            scoped_fidelity=tuple(scoped_fidelity),
            junction_vertices=junction_vertices if junction_rows.size else None,
            protected_features=self.specification.protected_features,
            fidelity_witness=(
                None
                if fidelity_source is None
                else _curve_chart_witness(
                    mesh,
                    geometry,
                    fidelity_source,
                    prepared,
                    cell_rows,
                    cell_parameters,
                )
            ),
            feature_source=self.source,
        )
        stages = (
            MeshingStageReport(
                MeshingStageKind.SOURCE_INSPECTION,
                MeshingStageStatus.PASSED,
                input_ids=(self.source.source_revision,),
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
        return NativeMeshConstruction(
            mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            organization=organization,
            stages=stages,
        )


@checked
def prepare_native_curve(
    source: sources_owner.NativeCurveSource,
    specification: CurveMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedCurveRoute:
    """Prepare size-density placement and hand construction to central acceptance."""
    unsupported = curve_support_issues(source, specification)
    if unsupported:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "; ".join(unsupported),
            provider_code="native_curve_preflight",
            stage="source_inspection",
        )
    schedule = (
        NativeCurveSchedule()
        if options.curve_schedule is None
        else options.curve_schedule
    )
    prepared = PreparedCurveNetwork(source, specification, schedule)
    return _PreparedCurveRoute(
        source, specification, options, coordinate_contract, prepared
    )


__all__ = ["PreparedCurveNetwork", "curve_support_issues", "prepare_native_curve"]
