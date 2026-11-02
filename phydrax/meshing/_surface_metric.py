#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-chart anisotropic metric adaptation for embedded triangle surfaces.

The local reconnection algorithm remains the canonical planar metric executor.
This module prepares one UV carrier from the retained source charts, pulls the
ambient SPD field back through each source patch differential, and realizes the
accepted UV edit with exact source-chart evaluations.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import combinations
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._geometry_predicates import PredicateMode
from .._physical import SpatialCoordinateContract
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_enclosure import add, axes, constant, scale
from ..discretization._nested_reference_types import (
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
    validate_reference_cell_map_witnesses,
)
from ..discretization._periodic_cell import PeriodicCell
from ..discretization._periodic_exact import exact_periodic_element
from ..discretization._periodic_topology import PeriodicMeshTopology
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._surface_chart_records import (
    SurfaceChartWitness,
    SurfaceRootMeshCellBinding,
)
from ..geometry._meshing_domain import _PlacedSurfacePatch, MeshingDomain
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryQuery,
    SourceProjectionStatus,
)
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.brep._patches import RevolutionSurface, SpherePatch
from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ._association import GeometryAssociation
from ._local_metric import (
    _Counts,
    _edge_table,
    _exact_periodic_metric,
    _LOWER,
    _metric_lengths,
    _published_state,
    _quality,
    _resume_local_metric_adaptation,
    _State,
    _target_order,
    _UPPER,
    execute_local_metric_adaptation,
    LocalMetricEvidence,
    LocalMetricOutcome,
)
from ._metric import (
    _constant_preserving_metric_mean,
    _tensor_properties,
)
from ._periodic import periodic_entity_orbit_map, stage_periodic_topology_edit
from ._topology_edit import (
    assemble_topology_edit,
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    key_rows,
    TopologyEditBlock,
)


class SurfaceMetricStatus(StrEnum):
    """Scientific termination of one source-chart metric request."""

    COMPLETE = "complete"
    STALLED = "stalled"
    NUMERICAL_STOP = "numerical_stop"


class SurfaceMetricGeometryRelation(StrEnum):
    """How target surface cells relate to the source mesh geometry.

    ``NESTED_REFERENCE``: every target cell lies in one source cell's exact chart and
    is reconstructed from the source maps through retained reference-cell maps.
    ``EXACT_PROJECTION``: target vertices are exact source projections on one global
    analytic chart; cells need not nest in source cells, and fidelity is certified
    against the source itself.
    """

    NESTED_REFERENCE = "nested_reference"
    EXACT_PROJECTION = "exact_projection"


@dataclass(frozen=True, slots=True)
class _ImplicitParameterSurface:
    profile: AnalyticImplicitProfile

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return (
            (None, math.tau) if self.profile.family == "sphere" else (math.tau, math.tau)
        )

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        center = jnp.asarray(self.profile.center, dtype=jnp.float64)
        if self.profile.family == "sphere":
            polar = values[..., 0]
            longitude = values[..., 1]
            sine = jnp.sin(polar)
            direction = jnp.stack(
                (
                    sine * jnp.cos(longitude),
                    sine * jnp.sin(longitude),
                    jnp.cos(polar),
                ),
                axis=-1,
            )
            return center + self.profile.radius * direction
        longitude = values[..., 0]
        latitude = values[..., 1]
        ring = self.profile.major_radius + self.profile.radius * jnp.cos(latitude)
        height = self.profile.radius * jnp.sin(latitude)
        return center + jnp.stack(
            (
                ring * jnp.cos(longitude),
                ring * jnp.sin(longitude),
                height,
            ),
            axis=-1,
        )


@dataclass(frozen=True, slots=True)
class _ImplicitMetricPatch:
    surface: _ImplicitParameterSurface


@dataclass(frozen=True, slots=True)
class _ImplicitMetricDomain:
    profile: AnalyticImplicitProfile
    source_query_id: str
    source_entity_id: str
    occurrence_path: tuple[str, ...]
    surface: _ImplicitParameterSurface

    @property
    def source_id(self) -> str:
        return self.profile.source_id

    @property
    def source_revision(self) -> str:
        return self.profile.source_revision

    @property
    def domain_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "implicit-surface-metric-chart",
                "profile": self.profile.profile_id,
                "query": self.source_query_id,
            }
        )

    @property
    def patches(self) -> tuple[_ImplicitMetricPatch, ...]:
        return (_ImplicitMetricPatch(self.surface),)

    @property
    def source_occurrences(
        self,
    ) -> tuple[tuple[tuple[str, ...], ...], ...]:
        return (((),), ((),), (self.occurrence_path,))

    def require_current(
        self, source_id: str | None = None, source_revision: str | None = None
    ) -> None:
        self.profile.validate_source_integrity()
        if (
            source_id is not None
            and source_id != self.profile.source_id
            or source_revision is not None
            and source_revision != self.profile.source_revision
        ):
            raise ValueError("Implicit metric chart source identity changed.")

    def entity_id(self, dimension: int, patch: int, /) -> str:
        if dimension != 2 or patch != 0:
            raise ValueError(
                "Implicit metric charts expose one two-dimensional source entity."
            )
        return self.source_entity_id

    def evaluate(self, patches: ArrayLike, charts: ArrayLike, /) -> np.ndarray:
        rows = np.asarray(patches, dtype=np.int64).reshape((-1,))
        parameters = np.asarray(charts, dtype=np.float64).reshape((-1, 2))
        if rows.shape != (parameters.shape[0],) or np.any(rows != 0):
            raise ValueError(
                "Implicit metric point evaluation requires source chart zero."
            )
        return np.asarray(
            self.surface.evaluate(jnp.asarray(parameters)), dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class _ImplicitSurfaceChartBank:
    """Exact analytic parameter charts of an implicit surface's extracted cells.

    The certified witness keeps root-cell/root-edge ancestry; the metric carrier is the
    profile's analytic chart, read exactly from each vertex's projection parameters.
    """

    mesh: CellMesh
    geometry: CellGeometrySpec
    domain: _ImplicitMetricDomain
    witness: SourceBoundaryChartWitness
    fidelity_source: SourceBoundaryQuery
    coordinate_contract: SpatialCoordinateContract
    cell_global_ids: tuple[int, ...]
    source_witness_rows: np.ndarray
    cell_charts: tuple[_ExactTriangle, ...]
    collapsed_vertices: np.ndarray
    root_patches: np.ndarray
    source_root_cell_ids: np.ndarray
    source_piece_ids: tuple[tuple[int, ...], ...]
    source_id: str
    source_revision: str
    source_chart_id: str

    def root_parameter_chart(self, row: int, /) -> _ExactTriangle:
        if not 0 <= row < len(self.cell_charts):
            raise IndexError("Implicit source chart row is out of range.")
        return self.cell_charts[row]

    def require_current(self) -> None:
        self.witness.require_current(self.mesh, self.geometry, self.fidelity_source)
        self.domain.require_current(self.source_id, self.source_revision)

    def require_root(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        if (
            mesh.mesh_id != self.mesh.mesh_id
            or geometry.geometry_layout_id != self.geometry.geometry_layout_id
        ):
            raise ValueError("Implicit source chart root changed after preparation.")
        self.require_current()


type _ExactPoint = tuple[Fraction, Fraction]
type _ExactTriangle = tuple[_ExactPoint, _ExactPoint, _ExactPoint]


class SurfaceMetricSource(NamedTuple):
    """Immutable UV carrier and source bindings prepared for one adaptation."""

    source_mesh: CellMesh
    source_charts: SurfaceSourceCharts | _ImplicitSurfaceChartBank
    domain: MeshingDomain | _ImplicitMetricDomain
    work_mesh: CellMesh
    work_to_source_rows: np.ndarray
    work_vertex_patches: np.ndarray
    work_vertex_charts: tuple[_ExactPoint, ...]
    work_vertex_period_shifts: np.ndarray
    cell_source_rows: np.ndarray
    cell_patches: np.ndarray
    cell_root_ids: np.ndarray
    cell_source_pieces: tuple[tuple[int, ...], ...]
    cell_charts: tuple[_ExactTriangle, ...]
    pullback_metric: np.ndarray
    physical_metric: np.ndarray
    source_cell_classes: np.ndarray
    source_edge_classes: np.ndarray
    source_protected_edges: np.ndarray
    source_fixed_vertices: np.ndarray
    periodic_rank: int
    periodic_axes: tuple[int, ...]
    periods: tuple[Fraction, ...]
    collapsed_source_vertices: np.ndarray


class SurfaceMetricGeometryWitness(NamedTuple):
    """Exact source-chart realization of the target surface vertices and cells."""

    source_id: str
    source_revision: str
    domain_id: str
    source_chart_id: str
    target_chart_witness: SurfaceChartWitness | SourceBoundaryChartWitness
    maximum_vertex_residual: float
    periodic_topology_id: str | None
    periodic_vertex_orbits: int
    periodic_edge_orbits: int
    relation: SurfaceMetricGeometryRelation
    implicit_cell_keys: np.ndarray | None = None
    implicit_root_edge_keys: np.ndarray | None = None
    reference_maps: tuple[ReferenceCellMapWitnesses, ...] = ()


class SurfaceMetricOutcome(NamedTuple):
    """Adapted physical topology, source chart witness, target metric, and evidence."""

    edit: CellTopologyEdit
    metric: np.ndarray
    geometry: SurfaceMetricGeometryWitness
    evidence: LocalMetricEvidence
    status: SurfaceMetricStatus
    passes: int
    splits: int
    collapses: int
    flips: int
    relocations: int
    out_of_range_edges: int
    minimum_metric_length: float
    maximum_metric_length: float
    unit_fraction: float


def _source_cells(mesh: CellMesh, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
        raise ValueError(
            "Surface metric adaptation requires a two-dimensional mesh in 3-space."
        )
    if not mesh.blocks or any(block.cell_kind != "triangle" for block in mesh.blocks):
        raise ValueError("Surface metric adaptation requires triangular source cells.")
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks), axis=0
    )
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    block_rows = np.concatenate(
        tuple(
            np.full((block.cell_count,), index, dtype=np.int32)
            for index, block in enumerate(mesh.blocks)
        )
    )
    if np.unique(cell_ids).size != cell_ids.size:
        raise ValueError("Surface metric source cell IDs must be unique.")
    return cells, cell_ids, block_rows


def _source_metric_rows(mesh: CellMesh, values: ArrayLike, /) -> np.ndarray:
    metric = np.asarray(values, dtype=np.float64)
    if metric.shape != (mesh.coordinates.shape[0], 3, 3) or not np.all(
        np.isfinite(metric)
    ):
        raise ValueError(
            "Surface metric values must be finite (vertices, 3, 3) ambient tensors."
        )
    properties = _tensor_properties(metric)
    if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
        raise ValueError("Surface metric tensors must be symmetric positive definite.")
    return metric


def _exact_triangle(chart: tuple[tuple[Fraction, ...], ...], /) -> _ExactTriangle:
    if len(chart) != 3 or any(len(point) != 2 for point in chart):
        raise ValueError("Surface source charts must be exact two-dimensional triangles.")
    first, second, third = chart
    return ((first[0], first[1]), (second[0], second[1]), (third[0], third[1]))


type _PointTokens = tuple[tuple[int, int], tuple[int, int]]


def _point_tokens(point: _ExactPoint, /) -> _PointTokens:
    return (
        (point[0].numerator, point[0].denominator),
        (point[1].numerator, point[1].denominator),
    )


def _triangle_tokens(
    chart: _ExactTriangle, /
) -> tuple[_PointTokens, _PointTokens, _PointTokens]:
    return (_point_tokens(chart[0]), _point_tokens(chart[1]), _point_tokens(chart[2]))


def _chart_area(chart: _ExactTriangle, /) -> Fraction:
    first, second, third = chart
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def _integer_class_rows(keys: tuple[tuple[object, ...], ...], /) -> np.ndarray:
    identifiers = {key: row for row, key in enumerate(sorted(set(keys), key=repr))}
    return np.asarray([identifiers[key] for key in keys], dtype=np.int64)


def _periods(
    domain: MeshingDomain | _ImplicitMetricDomain,
    patches: np.ndarray,
    /,
) -> tuple[tuple[Fraction, ...], tuple[int, ...]]:
    # A chart stays periodic where it is periodic: singular loci such as sphere poles
    # are collapsed chart boundaries carried by distinct pole copies, not a reason to
    # discard the seam lattice.
    active: tuple[tuple[int, Fraction], ...] | None = None
    for patch in np.unique(patches).tolist():
        current = tuple(
            (axis, Fraction.from_float(float(value)))
            for axis, value in enumerate(domain.patches[int(patch)].surface.periods)
            if value is not None
        )
        if not current:
            continue
        if active is None:
            active = current
        elif active != current:
            raise ValueError(
                "A source-chart carrier cannot combine incompatible periodic lattices."
            )
    if active is None:
        return (), ()
    return (
        tuple(period for _, period in active),
        tuple(axis for axis, _ in active),
    )


def _lattice_class(
    chart: _ExactPoint,
    periods: tuple[Fraction, ...],
    periodic_axes: tuple[int, ...],
    /,
) -> tuple[_ExactPoint, tuple[int, ...]]:
    """Exact fundamental-cell image of one chart point and its integer winding."""
    reduced = [chart[0], chart[1]]
    winding: list[int] = []
    for period, axis in zip(periods, periodic_axes, strict=True):
        turns = math.floor(chart[axis] / period)
        reduced[axis] = chart[axis] - turns * period
        winding.append(turns)
    return (reduced[0], reduced[1]), tuple(winding)


def _exact_chart_copy(
    representative: _ExactPoint,
    copy: _ExactPoint,
    periods: tuple[Fraction, ...],
    periodic_axes: tuple[int, ...],
    /,
    *,
    collapsed: bool,
) -> bool:
    """Whether one chart copy is its representative's exact lattice image.

    On a declared collapsed boundary the periodic coordinates are free, so twins agree
    exactly on the remaining chart axes.
    """
    if any(
        copy[axis] != representative[axis]
        for axis in range(2)
        if axis not in periodic_axes
    ):
        return False
    return (
        collapsed
        or _lattice_class(copy, periods, periodic_axes)[0]
        == _lattice_class(representative, periods, periodic_axes)[0]
    )


def _periodic_rows(
    vertex_source_rows: np.ndarray,
    vertex_patches: np.ndarray,
    vertex_charts: tuple[_ExactPoint, ...],
    periods: tuple[Fraction, ...],
    periodic_axes: tuple[int, ...],
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = vertex_source_rows.size
    representatives = np.arange(count, dtype=np.int64)
    shifts = np.zeros((count, len(periods)), dtype=np.int64)
    orbit_codes = np.arange(count, dtype=np.int64)
    if not periods:
        return representatives, shifts, orbit_codes
    # Copies of one source vertex are one orbit exactly when their charts share the
    # exact fundamental-cell image; the integer winding difference is the lattice shift.
    classes: dict[tuple[int, int, _ExactPoint], list[tuple[int, tuple[int, ...]]]] = (
        defaultdict(list)
    )
    for row, (source_row, patch, chart) in enumerate(
        zip(
            vertex_source_rows.tolist(),
            vertex_patches.tolist(),
            vertex_charts,
            strict=True,
        )
    ):
        reduced, winding = _lattice_class(chart, periods, periodic_axes)
        classes[(int(source_row), int(patch), reduced)].append((row, winding))
    for members in classes.values():
        representative, reference = members[0]
        for row, winding in members[1:]:
            representatives[row] = representative
            shifts[row] = np.asarray(
                [value - base for value, base in zip(winding, reference, strict=True)],
                dtype=np.int64,
            )
            orbit_codes[row] = representative
    return representatives, shifts, orbit_codes


# The sphere chart (polar, azimuth) collapses the azimuth axis at the poles: the
# exact pullback there is rank one, ``R**2 * u.T @ M @ u`` on the polar axis and
# zero on the azimuth axis. The azimuth stretch ``R * sin(polar)`` along the
# tangent-plane direction ``w`` is therefore floored at this binary64 resolution:
# the azimuth/polar eigenvalue ratio stays at least ``sqrt(eps)``, the smallest
# ratio a backward-stable 2x2 eigendecomposition (log-Euclidean interpolation,
# SPD evidence) still resolves to half precision.
_AZIMUTH_STRETCH_FLOOR = float(np.finfo(np.float64).eps) ** 0.25


def _implicit_chart_frames(
    surface: _ImplicitParameterSurface, parameters: np.ndarray, /
) -> np.ndarray:
    """Chart Jacobians ``(n, 3, 2)`` of an analytic implicit surface.

    At a sphere pole, where the exact azimuth column ``R sin(polar) w`` falls
    below the binary64 floor, the column is ``R * floor * w`` on the pole's
    tangent plane, aligned with the point's own azimuth.
    """
    first, second = parameters[:, 0], parameters[:, 1]
    radius = float(surface.profile.radius)
    zero = np.zeros_like(first)
    if surface.profile.family == "sphere":
        polar_sine, polar_cosine = np.sin(first), np.cos(first)
        azimuth_sine, azimuth_cosine = np.sin(second), np.cos(second)
        polar_axis = np.stack(
            (polar_cosine * azimuth_cosine, polar_cosine * azimuth_sine, -polar_sine),
            axis=1,
        )
        azimuth_axis = np.stack(
            (-polar_sine * azimuth_sine, polar_sine * azimuth_cosine, zero), axis=1
        )
        frames = radius * np.stack((polar_axis, azimuth_axis), axis=2)
        floored = np.abs(polar_sine) < _AZIMUTH_STRETCH_FLOOR
        frames[floored, :, 1] = (radius * _AZIMUTH_STRETCH_FLOOR) * np.stack(
            (-azimuth_sine, azimuth_cosine, zero), axis=1
        )[floored]
        return frames
    longitude_sine, longitude_cosine = np.sin(first), np.cos(first)
    latitude_sine, latitude_cosine = np.sin(second), np.cos(second)
    ring = float(surface.profile.major_radius) + radius * latitude_cosine
    longitude_axis = np.stack(
        (-ring * longitude_sine, ring * longitude_cosine, zero), axis=1
    )
    ring_rate = -radius * latitude_sine
    latitude_axis = np.stack(
        (
            ring_rate * longitude_cosine,
            ring_rate * longitude_sine,
            radius * latitude_cosine,
        ),
        axis=1,
    )
    return np.stack((longitude_axis, latitude_axis), axis=2)


def _pullback_metrics(
    domain: MeshingDomain | _ImplicitMetricDomain,
    patches: np.ndarray,
    chart_values: np.ndarray,
    physical: np.ndarray,
    /,
) -> np.ndarray:
    """``J.T @ M @ J`` at binary64 chart points, with witnessed singular frames."""
    result = np.empty((chart_values.shape[0], 2, 2), dtype=np.float64)
    for patch in np.unique(patches).tolist():
        rows = np.flatnonzero(patches == patch)
        surface = domain.patches[int(patch)].surface
        if isinstance(surface, _ImplicitParameterSurface):
            jacobians = _implicit_chart_frames(surface, chart_values[rows])
            value = np.swapaxes(jacobians, 1, 2) @ physical[rows] @ jacobians
            result[rows] = 0.5 * (value + np.swapaxes(value, 1, 2))
            continue
        differential = jax.jacfwd(surface.evaluate)
        jacobians = np.asarray(
            jax.vmap(differential)(jnp.asarray(chart_values[rows], dtype=jnp.float64)),
            dtype=np.float64,
        )
        base_surface = surface
        chart_frame = np.eye(3, dtype=np.float64)
        while isinstance(base_surface, _PlacedSurfacePatch):
            chart_frame = chart_frame @ np.asarray(base_surface.linear, dtype=np.float64)
            base_surface = base_surface.source
        for position, row in enumerate(rows.tolist()):
            jacobian = jacobians[position]
            if isinstance(base_surface, SpherePatch):
                polar = chart_values[row, 1]
                if abs(abs(polar) - np.pi / 2.0) <= 32.0 * np.finfo(np.float64).eps:
                    frame = np.stack(
                        (
                            np.asarray(base_surface.first_axis),
                            np.asarray(base_surface.second_axis),
                        ),
                        axis=1,
                    )
                    jacobian = chart_frame @ (
                        2.0 * float(np.asarray(base_surface.radius)) * frame
                    )
            elif isinstance(base_surface, RevolutionSurface):
                scale = max(1.0, float(np.linalg.norm(jacobian)))
                if (
                    np.linalg.norm(jacobian[:, 0])
                    <= 32.0 * np.finfo(np.float64).eps * scale
                ):
                    axis = np.asarray(base_surface.axis_direction, dtype=np.float64)
                    profile_tangent = np.asarray(
                        jax.jacfwd(base_surface.curve.evaluate)(
                            jnp.asarray(chart_values[row, 1], dtype=jnp.float64)
                        ),
                        dtype=np.float64,
                    )
                    radial_tangent = profile_tangent - axis * float(
                        np.dot(axis, profile_tangent)
                    )
                    radial_speed = float(np.linalg.norm(radial_tangent))
                    if radial_speed <= (
                        32.0
                        * np.finfo(np.float64).eps
                        * max(1.0, float(np.linalg.norm(profile_tangent)))
                    ):
                        raise ValueError(
                            "A singular revolution chart has no witnessed two-dimensional tangent frame."
                        )
                    reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
                    if abs(float(np.dot(reference, axis))) > 0.8:
                        reference = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
                    first_axis = np.cross(axis, reference)
                    first_axis /= np.linalg.norm(first_axis)
                    second_axis = np.cross(axis, first_axis)
                    jacobian = chart_frame @ (
                        2.0 * radial_speed * np.stack((first_axis, second_axis), axis=1)
                    )
            value = jacobian.T @ physical[row] @ jacobian
            result[row] = 0.5 * (value + value.T)
    return result


def _chart_differentials(
    domain: MeshingDomain | _ImplicitMetricDomain,
    patches: np.ndarray,
    charts: tuple[_ExactPoint, ...],
    physical_metric: np.ndarray,
    source_rows: np.ndarray,
    periods: tuple[Fraction, ...],
    orbit_codes: np.ndarray,
    /,
) -> np.ndarray:
    chart_values = np.asarray(
        [[float(value) for value in point] for point in charts], dtype=np.float64
    )
    result = _pullback_metrics(
        domain, patches, chart_values, physical_metric[np.asarray(source_rows)]
    )
    for code in np.unique(orbit_codes).tolist():
        members = np.flatnonzero(orbit_codes == code)
        if members.size < 2:
            continue
        representative = int(members[0])
        reference = result[representative]
        for row in members[1:].tolist():
            candidate = result[int(row)]
            scale = max(1.0, float(np.linalg.norm(reference, ord=np.inf)))
            if np.linalg.norm(candidate - reference, ord=np.inf) > 2.0e-10 * scale:
                raise ValueError(
                    "Pulled-back SPD metrics disagree across a periodic source orbit."
                )
            result[int(row)] = reference
    properties = _tensor_properties(result)
    if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
        raise ValueError(
            "Physical metric pullback must remain symmetric positive definite in every chart."
        )
    return result


def _edge_incidence(mesh: CellMesh, /) -> np.ndarray:
    relation = mesh.topology.incidences[1].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    edge_rows = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    return np.bincount(edge_rows, minlength=mesh.entity_set(1).count)


def prepare_surface_metric_source(
    mesh: CellMesh,
    source_charts: SurfaceSourceCharts | _ImplicitSurfaceChartBank,
    metric: ArrayLike,
    /,
    *,
    cell_classes: ArrayLike | None = None,
    edge_classes: ArrayLike | None = None,
    protected_edges: ArrayLike | None = None,
    fixed_vertices: ArrayLike | None = None,
) -> SurfaceMetricSource:
    """Build a periodic UV carrier and pull back a physical SPD metric field."""
    if not isinstance(mesh, CellMesh) or not isinstance(
        source_charts, (SurfaceSourceCharts, _ImplicitSurfaceChartBank)
    ):
        raise TypeError(
            "Surface metric preparation requires a CellMesh and source chart bank."
        )
    source_charts.require_root(mesh, source_charts.geometry)
    domain = source_charts.domain
    domain.require_current(source_charts.source_id, source_charts.source_revision)
    physical_metric = _source_metric_rows(mesh, metric)
    source_cells, source_cell_ids, source_block_rows = _source_cells(mesh)
    chart_rows = {
        int(identifier): row
        for row, identifier in enumerate(source_charts.cell_global_ids)
    }
    if set(chart_rows) != set(source_cell_ids.tolist()):
        raise ValueError(
            "Surface source charts must cover every current source cell exactly once."
        )
    cell_source_rows = np.asarray(
        [chart_rows[int(identifier)] for identifier in source_cell_ids], dtype=np.int64
    )
    cell_patches = np.asarray(source_charts.root_patches, dtype=np.int64)[
        cell_source_rows
    ]
    cell_root_ids = np.asarray(source_charts.source_root_cell_ids, dtype=np.int64)[
        cell_source_rows
    ]
    cell_pieces = tuple(
        source_charts.source_piece_ids[int(row)] for row in cell_source_rows
    )
    root_parameter_chart = source_charts.root_parameter_chart
    exact_charts = tuple(
        _exact_triangle(root_parameter_chart(int(row))) for row in cell_source_rows
    )
    if domain.source_id != source_charts.source_id:
        raise ValueError("Source charts no longer bind their declared source domain.")
    periods, periodic_axes = _periods(domain, cell_patches)
    supplied_cell_classes = (
        np.zeros((source_cells.shape[0],), dtype=np.int64)
        if cell_classes is None
        else np.asarray(cell_classes, dtype=np.int64)
    )
    edge_count = mesh.entity_set(1).count
    supplied_edge_classes = (
        np.zeros((edge_count,), dtype=np.int64)
        if edge_classes is None
        else np.asarray(edge_classes, dtype=np.int64)
    )
    supplied_protected = (
        np.zeros((edge_count,), dtype=np.bool_)
        if protected_edges is None
        else np.asarray(protected_edges, dtype=np.bool_)
    )
    supplied_fixed = (
        np.zeros((mesh.coordinates.shape[0],), dtype=np.bool_)
        if fixed_vertices is None
        else np.asarray(fixed_vertices, dtype=np.bool_)
    )
    if supplied_cell_classes.shape != (source_cells.shape[0],):
        raise ValueError("cell_classes must align with source triangle rows.")
    if supplied_edge_classes.shape != (edge_count,) or supplied_protected.shape != (
        edge_count,
    ):
        raise ValueError("Edge classes and protection must align with source edges.")
    if supplied_fixed.shape != (mesh.coordinates.shape[0],):
        raise ValueError("fixed_vertices must align with source vertices.")
    if (
        supplied_cell_classes.dtype.kind not in "iu"
        or supplied_edge_classes.dtype.kind not in "iu"
    ):
        raise TypeError("Source metric classification arrays must be integer-valued.")
    if supplied_protected.dtype != np.bool_ or supplied_fixed.dtype != np.bool_:
        raise TypeError("Source metric protection arrays must be boolean.")

    vertex_records: list[tuple[int, int, _ExactPoint]] = []
    vertex_lookup: dict[tuple[int, int, _ExactPoint], int] = {}
    work_cells: list[tuple[int, int, int]] = []
    orientation_signs = np.empty((source_cells.shape[0],), dtype=np.int8)
    for cell_row, (cell, chart, patch) in enumerate(
        zip(source_cells, exact_charts, cell_patches.tolist(), strict=True)
    ):
        signed_area = _chart_area(chart)
        if not signed_area:
            raise ValueError("Source surface chart contains a degenerate triangle.")
        orientation = 1 if signed_area > 0 else -1
        orientation_signs[cell_row] = orientation
        ordered_local = (0, 1, 2) if orientation > 0 else (0, 2, 1)
        corners: list[int] = []
        for local in ordered_local:
            source_vertex_row = int(cell[local])
            point = chart[local]
            key = (source_vertex_row, int(patch), point)
            work_row = vertex_lookup.get(key)
            if work_row is None:
                work_row = len(vertex_records)
                vertex_lookup[key] = work_row
                vertex_records.append((source_vertex_row, int(patch), point))
            corners.append(work_row)
        work_cells.append((corners[0], corners[1], corners[2]))

    source_vertex_rows = np.asarray(
        [record[0] for record in vertex_records], dtype=np.int64
    )
    work_vertex_patches = np.asarray(
        [record[1] for record in vertex_records], dtype=np.int64
    )
    work_vertex_charts = tuple(record[2] for record in vertex_records)
    representatives, shifts, orbit_codes = _periodic_rows(
        source_vertex_rows,
        work_vertex_patches,
        work_vertex_charts,
        periods,
        periodic_axes,
    )
    uv = np.asarray(
        [[float(value) for value in point] for point in work_vertex_charts],
        dtype=np.float64,
    )
    source_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    source_to_work_ids = np.empty((len(vertex_records),), dtype=np.int64)
    next_id = int(np.max(source_vertex_ids, initial=-1)) + 1
    first_for_source: dict[int, int] = {}
    for row, source_row in enumerate(source_vertex_rows.tolist()):
        first = first_for_source.get(int(source_row))
        if first is None:
            source_to_work_ids[row] = source_vertex_ids[int(source_row)]
            first_for_source[int(source_row)] = row
        else:
            source_to_work_ids[row] = next_id
            next_id += 1
    work_block = CellBlock(
        "surface_metric_uv",
        "triangle",
        np.asarray(work_cells, dtype=np.int64),
        global_ids=source_cell_ids,
    )
    base_work_mesh = CellMesh(uv, (work_block,), vertex_global_ids=source_to_work_ids)
    if periods:
        period_vectors = np.zeros((len(periods), 2), dtype=np.float64)
        for period_row, (period, axis) in enumerate(
            zip(periods, periodic_axes, strict=True)
        ):
            period_vectors[period_row, axis] = float(period)
        periodic_cell = PeriodicCell(
            period_vectors, origin=np.zeros((2,), dtype=np.float64)
        )
        periodic = PeriodicMeshTopology(
            base_work_mesh, periodic_cell, representatives, shifts
        )
        work_mesh = CellMesh(
            uv,
            (work_block,),
            vertex_global_ids=source_to_work_ids,
            periodic_topology=periodic,
        )
    else:
        work_mesh = base_work_mesh

    implicit_source = isinstance(source_charts, _ImplicitSurfaceChartBank)
    if implicit_source or work_mesh.periodic_topology is None:
        cell_class_ids = source_cell_ids
    else:
        cell_orbit_map = periodic_entity_orbit_map(work_mesh, 2)
        cell_class_ids_by_id = {
            int(identifier): int(orbit)
            for identifier, orbit in zip(
                np.asarray(cell_orbit_map.lifted_entity_ids, dtype=np.int64),
                np.asarray(cell_orbit_map.lifted_to_quotient, dtype=np.int64),
                strict=True,
            )
        }
        cell_class_ids = np.asarray(
            [cell_class_ids_by_id[int(identifier)] for identifier in source_cell_ids],
            dtype=np.int64,
        )
    # BRep root charts are piecewise, so each source cell is its own chart class. The
    # implicit analytic chart is one global chart: extraction cells are not features,
    # and only declared classes separate its cells.
    cell_class_keys = tuple(
        (
            (
                int(supplied_cell_classes[row]),
                int(cell_patches[row]),
                int(orientation_signs[row]),
            )
            if implicit_source
            else (
                int(supplied_cell_classes[row]),
                int(cell_patches[row]),
                int(cell_class_ids[row]),
                int(cell_root_ids[row]),
                cell_pieces[row],
                int(orientation_signs[row]),
            )
        )
        for row in range(source_cells.shape[0])
    )
    work_cell_classes = _integer_class_rows(cell_class_keys)
    source_edge_keys = entity_keys(mesh, 1)
    work_edge_keys = entity_keys(work_mesh, 1)
    work_id_to_source_id = {
        int(work_id): int(source_vertex_ids[int(source_row)])
        for work_id, source_row in zip(
            source_to_work_ids, source_vertex_rows, strict=True
        )
    }
    work_edge_origin_keys = np.asarray(
        [
            tuple(sorted(work_id_to_source_id[int(value)] for value in key))
            for key in np.asarray(work_edge_keys, dtype=np.int64)
        ],
        dtype=np.int64,
    )
    work_edge_source_rows = key_rows(source_edge_keys, work_edge_origin_keys)
    counts = _edge_incidence(work_mesh)
    source_counts = _edge_incidence(mesh)
    edge_count_work = work_mesh.entity_set(1).count
    work_protected = np.zeros((edge_count_work,), dtype=np.bool_)
    work_edge_classes = np.zeros((edge_count_work,), dtype=np.int64)
    for row, source_edge_row in enumerate(work_edge_source_rows.tolist()):
        if source_edge_row >= 0:
            work_edge_classes[row] = supplied_edge_classes[int(source_edge_row)]
            work_protected[row] = (
                supplied_protected[int(source_edge_row)]
                or source_counts[int(source_edge_row)] == 1
            )
        elif counts[row] == 1:
            work_protected[row] = True
    source_vertex_id_to_row = {
        int(identifier): row for row, identifier in enumerate(source_vertex_ids.tolist())
    }
    boundary_source_vertex_rows = {
        source_vertex_id_to_row[int(identifier)]
        for identifier in np.unique(
            np.asarray(source_edge_keys, dtype=np.int64)[source_counts == 1]
        ).tolist()
    }
    # BRep target cells retain one exact source chart, so every source vertex is an
    # anchor. Implicit extraction vertices are not geometric features: only declared
    # fixed vertices, trim boundaries, and collapsed poles anchor; the rest may be
    # collapsed or relocated because exact source projection realizes any chart point.
    if implicit_source:
        source_collapsed = np.asarray(source_charts.collapsed_vertices, dtype=np.bool_)
        work_fixed = (
            supplied_fixed[source_vertex_rows] | source_collapsed[source_vertex_rows]
        )
    else:
        work_fixed = np.ones((source_vertex_rows.size,), dtype=np.bool_)
        work_fixed |= supplied_fixed[source_vertex_rows]
    work_fixed |= np.isin(
        source_vertex_rows,
        np.asarray(sorted(boundary_source_vertex_rows), dtype=np.int64),
    )
    work_id_to_row = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(work_mesh.vertex_global_ids).tolist())
    }
    work_edge_vertex_rows = np.asarray(
        [
            tuple(work_id_to_row[int(identifier)] for identifier in key)
            for key in np.asarray(work_edge_keys, dtype=np.int64)
        ],
        dtype=np.int64,
    )
    if work_mesh.periodic_topology is not None:
        orbit_map = periodic_entity_orbit_map(work_mesh, 1)
        edge_orbits = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
        orbit_sizes = np.bincount(
            edge_orbits, minlength=int(np.max(edge_orbits, initial=-1)) + 1
        )
    else:
        edge_orbits = np.arange(edge_count_work, dtype=np.int64)
        orbit_sizes = np.ones((edge_count_work,), dtype=np.int64)
    source_collapsed_rows = (
        np.asarray(source_charts.collapsed_vertices, dtype=np.bool_)[source_vertex_rows]
        if implicit_source
        else np.zeros((source_vertex_rows.size,), dtype=np.bool_)
    )
    for row, source_edge_row in enumerate(work_edge_source_rows.tolist()):
        if (
            counts[row] == 1
            and source_edge_row >= 0
            and source_counts[int(source_edge_row)] == 2
            and orbit_sizes[edge_orbits[row]] == 1
        ):
            # A chart cut. At a collapsed point copy it guards only the copy:
            # the regular end stays free (the local engine follows the cut).
            work_protected[row] = True
            ends = work_edge_vertex_rows[row]
            if np.any(source_collapsed_rows[ends]):
                work_fixed[ends[source_collapsed_rows[ends]]] = True
            else:
                work_fixed[ends] = True
    incidence = work_mesh.topology.incidences[1].relation
    valid_incidence = np.asarray(incidence.valid, dtype=np.bool_)
    incident_edges = np.asarray(incidence.source_indices, dtype=np.int64)[valid_incidence]
    incident_cells = np.asarray(incidence.target_indices, dtype=np.int64)[valid_incidence]
    interface_rows: list[int] = []
    interface_keys: list[tuple[object, ...]] = []
    for edge_row in np.unique(incident_edges).tolist():
        cells = incident_cells[incident_edges == edge_row]
        if cells.size != 2:
            continue
        first, second = int(cells[0]), int(cells[1])
        source_cell_interface = not implicit_source and int(
            source_cell_ids[first]
        ) != int(source_cell_ids[second])
        if work_cell_classes[first] != work_cell_classes[second] or source_cell_interface:
            interface_rows.append(int(edge_row))
            interface_keys.append(
                (
                    "periodic-edge",
                    int(edge_orbits[int(edge_row)]),
                )
                if work_mesh.periodic_topology is not None
                else (
                    "source-edge",
                    int(work_edge_source_rows[int(edge_row)])
                    if work_edge_source_rows[int(edge_row)] >= 0
                    else int(edge_row),
                )
            )
    class_offset = int(np.max(work_edge_classes, initial=0)) + 1
    interface_codes = _integer_class_rows(tuple(interface_keys))
    for edge_row, code in zip(interface_rows, interface_codes.tolist(), strict=True):
        work_edge_classes[edge_row] = class_offset + code
    unclassified_boundary = (counts == 1) & (work_edge_classes == 0)
    if np.any(unclassified_boundary):
        boundary_class = int(np.max(work_edge_classes, initial=0)) + 1
        work_edge_classes[unclassified_boundary] = boundary_class
    pulled = _chart_differentials(
        domain,
        work_vertex_patches,
        work_vertex_charts,
        physical_metric,
        source_vertex_rows,
        periods,
        orbit_codes,
    )
    return SurfaceMetricSource(
        mesh,
        source_charts,
        domain,
        work_mesh,
        source_vertex_rows,
        work_vertex_patches,
        work_vertex_charts,
        shifts,
        cell_source_rows,
        cell_patches,
        cell_root_ids,
        cell_pieces,
        exact_charts,
        pulled,
        physical_metric,
        work_cell_classes,
        work_edge_classes,
        work_protected,
        work_fixed,
        len(periods),
        periodic_axes,
        periods,
        (
            np.asarray(source_charts.collapsed_vertices, dtype=np.bool_)
            if isinstance(source_charts, _ImplicitSurfaceChartBank)
            else np.zeros((mesh.coordinates.shape[0],), dtype=np.bool_)
        ),
    )


def _implicit_cell_charts(
    profile: AnalyticImplicitProfile,
    coordinates: np.ndarray,
    parameters: np.ndarray,
    cells: np.ndarray,
    /,
) -> tuple[tuple[_ExactTriangle, ...], np.ndarray]:
    """Lift every source cell into one exact cell of the analytic chart lattice.

    A vertex's chart root is its binary64 projection parameter read exactly. Each cell
    takes the nearest lattice lift of its corners relative to its first regular corner
    and is then translated so its least periodic coordinate lies in ``[0, period)``.
    Sphere poles are a declared collapsed chart boundary: longitude is undefined there,
    so every incident cell carries its own pole copy at the exact mean longitude of its
    regular corners, keeping the polar coordinate of the pole root.
    """
    period = Fraction.from_float(math.tau)
    sphere = profile.family == "sphere"
    periodic_axes = (1,) if sphere else (0, 1)
    relative = coordinates - np.asarray(profile.center, dtype=np.float64)
    collapsed = (
        (relative[:, 0] == 0.0) & (relative[:, 1] == 0.0)
        if sphere
        else np.zeros((coordinates.shape[0],), dtype=np.bool_)
    )
    roots = tuple(
        (Fraction.from_float(first), Fraction.from_float(second))
        for first, second in parameters.tolist()
    )
    collapsed_rows = collapsed.tolist()
    charts: list[_ExactTriangle] = []
    for cell in cells.tolist():
        regular = [int(vertex) for vertex in cell if not collapsed_rows[int(vertex)]]
        if len(regular) < 2:
            raise ValueError(
                "A source cell meets the collapsed chart boundary at more than one corner."
            )
        lifted = {vertex: [roots[vertex][0], roots[vertex][1]] for vertex in regular}
        for axis in periodic_axes:
            reference = lifted[regular[0]][axis]
            for vertex in regular[1:]:
                turns = math.floor(
                    (lifted[vertex][axis] - reference) / period + Fraction(1, 2)
                )
                lifted[vertex][axis] -= turns * period
            values = [lifted[vertex][axis] for vertex in regular]
            if max(values) - min(values) >= period / 2:
                raise ValueError(
                    "A source cell spans half a chart period, so its lattice lift is ambiguous."
                )
            base = math.floor(min(values) / period)
            for vertex in regular:
                lifted[vertex][axis] -= base * period
        mean = sum((lifted[vertex][1] for vertex in regular), Fraction(0)) / len(regular)
        chart = tuple(
            (roots[int(vertex)][0], mean)
            if collapsed_rows[int(vertex)]
            else (lifted[int(vertex)][0], lifted[int(vertex)][1])
            for vertex in cell
        )
        charts.append((chart[0], chart[1], chart[2]))
    return tuple(charts), collapsed


def prepare_implicit_surface_metric_source(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    coordinate_contract: SpatialCoordinateContract,
    profile: AnalyticImplicitProfile,
    witness: SourceBoundaryChartWitness,
    fidelity_source: SourceBoundaryQuery,
    metric: ArrayLike,
    /,
    *,
    cell_classes: ArrayLike | None = None,
    edge_classes: ArrayLike | None = None,
    protected_edges: ArrayLike | None = None,
    fixed_vertices: ArrayLike | None = None,
) -> SurfaceMetricSource:
    """Prepare exact implicit sphere/torus parameter charts for metric edits."""
    if not isinstance(profile, AnalyticImplicitProfile):
        raise TypeError("profile must be AnalyticImplicitProfile.")
    if not isinstance(witness, SourceBoundaryChartWitness):
        raise TypeError("witness must be SourceBoundaryChartWitness.")
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
    profile.require_bound(profile.geometry, coordinate_contract)
    witness.require_current(mesh, geometry, fidelity_source)
    if (
        witness.source_id != profile.source_id
        or witness.source_revision != profile.source_revision
        or not witness.implicit_cell_keys.size
        or not witness.implicit_root_edge_keys.size
    ):
        raise ValueError(
            "Implicit metric source requires exact cell and root-edge ancestry."
        )
    source_cells, source_cell_ids, _ = _source_cells(mesh)
    witness_rows_by_cell = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(witness.cell_global_ids, dtype=np.int64).tolist()
        )
    }
    if set(witness_rows_by_cell) != set(source_cell_ids.tolist()):
        raise ValueError("Implicit source chart witness must cover every source cell.")
    witness_rows = np.asarray(
        [witness_rows_by_cell[int(identifier)] for identifier in source_cell_ids],
        dtype=np.int64,
    )
    implicit_cells = np.asarray(witness.implicit_cell_keys, dtype=np.int64)[witness_rows]
    root_tokens = tuple(
        tuple(int(value) for value in row) for row in implicit_cells.tolist()
    )
    ordered_roots = sorted(set(root_tokens))
    root_codes = {key: row for row, key in enumerate(ordered_roots)}
    root_ids = np.asarray([root_codes[key] for key in root_tokens], dtype=np.int64)
    source_pieces = tuple((0, int(root_id)) for root_id in root_ids.tolist())
    if (
        len(set(witness.source_entity_ids)) != 1
        or len(set(witness.occurrence_paths)) != 1
    ):
        raise ValueError(
            "One analytic implicit surface metric route requires one source occurrence."
        )
    domain = _ImplicitMetricDomain(
        profile,
        witness.source_query_id,
        witness.source_entity_ids[0],
        witness.occurrence_paths[0],
        _ImplicitParameterSurface(profile),
    )
    entity_id = witness.source_entity_ids[0]
    occurrences = tuple(
        witness.occurrence_paths[0] for _ in range(mesh.coordinates.shape[0])
    )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    projected = profile.project_boundary(
        coordinates,
        entity_ids=tuple(entity_id for _ in range(mesh.coordinates.shape[0])),
        occurrence_paths=occurrences,
    )
    if not np.all(
        np.isin(
            np.asarray(projected.status),
            (int(SourceProjectionStatus.UNIQUE), int(SourceProjectionStatus.SEAM)),
        )
    ):
        raise ValueError(
            "Implicit metric source vertices lack unique source-bound projections."
        )
    cell_charts, collapsed = _implicit_cell_charts(
        profile,
        coordinates,
        np.asarray(projected.parameters, dtype=np.float64),
        source_cells,
    )
    bank = _ImplicitSurfaceChartBank(
        mesh,
        geometry,
        domain,
        witness,
        fidelity_source,
        coordinate_contract,
        tuple(int(value) for value in source_cell_ids.tolist()),
        witness_rows,
        cell_charts,
        collapsed,
        np.zeros((source_cell_ids.size,), dtype=np.int64),
        root_ids,
        source_pieces,
        profile.source_id,
        profile.source_revision,
        witness.witness_id,
    )
    bank.require_root(mesh, geometry)
    return prepare_surface_metric_source(
        mesh,
        bank,
        metric,
        cell_classes=cell_classes,
        edge_classes=edge_classes,
        protected_edges=protected_edges,
        fixed_vertices=fixed_vertices,
    )


def _source_rows_for_cell(
    relation: EntityRelations,
    target_id: int,
    source_by_id: dict[int, int],
    /,
) -> tuple[int, ...]:
    source_keys = np.asarray(relation.source_keys, dtype=np.int64)
    target_keys = np.asarray(relation.target_keys, dtype=np.int64)
    matches = np.flatnonzero(target_keys[:, 0] == target_id)
    rows = tuple(sorted({source_by_id[int(source_keys[row, 0])] for row in matches}))
    if rows:
        return rows
    if target_id in source_by_id:
        return (source_by_id[target_id],)
    raise ValueError("A target metric cell lacks source-cell ancestry.")


type _WorkStencil = tuple[tuple[int, Fraction], ...]


def _work_stencils(
    local_edit: CellTopologyEdit,
    source: SurfaceMetricSource,
    /,
) -> list[_WorkStencil]:
    """Exact UV-carrier interpolation row of every local target vertex."""
    exact_weights = local_edit.stencil_exact_weights
    if exact_weights is None:
        raise ValueError(
            "Local metric execution omitted exact source interpolation weights."
        )
    sources = np.asarray(local_edit.stencil_sources, dtype=np.int64)
    valid = np.asarray(local_edit.stencil_valid, dtype=np.bool_)
    work_vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(source.work_mesh.vertex_global_ids, dtype=np.int64).tolist()
        )
    }
    stencils: list[_WorkStencil] = []
    for row in range(sources.shape[0]):
        combined: dict[int, Fraction] = defaultdict(Fraction)
        for source_id, weight, is_valid in zip(
            sources[row], exact_weights[row], valid[row], strict=True
        ):
            if is_valid:
                work_row = work_vertex_rows.get(int(source_id))
                if work_row is None:
                    raise ValueError(
                        "A local metric stencil names an unknown source chart vertex."
                    )
                combined[work_row] += weight
        stencil = tuple(
            (work_row, weight) for work_row, weight in sorted(combined.items()) if weight
        )
        if not stencil or sum((weight for _, weight in stencil), Fraction(0)) != 1:
            raise ValueError(
                "A source-chart target stencil must preserve constants exactly."
            )
        stencils.append(stencil)
    return stencils


def _edge_counts(cells: np.ndarray, /) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for first, second, third in cells.tolist():
        for left, right in ((first, second), (second, third), (third, first)):
            counts[(min(left, right), max(left, right))] += 1
    return counts


def _edge_projection_stencil(
    source: SurfaceMetricSource,
    edge: tuple[int, ...],
    shadow: np.ndarray,
    /,
) -> _WorkStencil:
    """Exact convex weights of a shadow's orthogonal projection onto one chart edge."""
    first = source.work_vertex_charts[edge[0]]
    second = source.work_vertex_charts[edge[1]]
    point = (
        Fraction.from_float(float(shadow[0])),
        Fraction.from_float(float(shadow[1])),
    )
    direction = (second[0] - first[0], second[1] - first[1])
    parameter = (
        (point[0] - first[0]) * direction[0] + (point[1] - first[1]) * direction[1]
    ) / (direction[0] * direction[0] + direction[1] * direction[1])
    parameter = min(max(parameter, Fraction(0)), Fraction(1))
    return tuple(
        (row, weight)
        for row, weight in zip(edge, (1 - parameter, parameter), strict=True)
        if weight
    )


def _parent_face_stencils(
    source: SurfaceMetricSource,
    stencils: list[_WorkStencil],
    coordinates: np.ndarray,
    target_cells: np.ndarray,
    cell_parents: tuple[tuple[int, ...], ...],
    /,
) -> list[_WorkStencil]:
    """Restrict target stencils to the exact source face their vertex lies on.

    Local metric execution moves binary64 shadows of the exact UV chart, so a
    split point of a source interface or chart-boundary edge can round off that
    edge. Its exact point must lie in the closure of every incident target
    cell's parent source cell, and on a source boundary edge when it bounds the
    target carrier. An edge face takes the exact orthogonal projection of the
    shadow onto that edge: renormalizing dropped barycentric mass would move the
    point along the edge by that mass times the cell aspect ratio. Larger faces
    are restricted and renormalized. These rows explicitly define the published
    source charts; the original local carrier is checked before restriction. The
    implicit analytic chart is global, so only its carrier-boundary vertices are
    restricted: an interior vertex is any chart point, realized by exact source
    projection.
    """
    global_chart = isinstance(source.source_charts, _ImplicitSurfaceChartBank)
    work_cells = np.concatenate(
        tuple(
            np.asarray(block.vertices, dtype=np.int64)
            for block in source.work_mesh.blocks
        ),
        axis=0,
    )
    work_boundary = {
        edge for edge, count in _edge_counts(work_cells).items() if count == 1
    }
    # Edges at a copy of a collapsed chart point (a pole) are chart cuts, not
    # carrier boundary: the regular vertex there is a free interior point.
    collapsed_work = source.collapsed_source_vertices[source.work_to_source_rows]
    copies = {
        vertex
        for vertex, stencil in enumerate(stencils)
        if len(stencil) == 1 and bool(collapsed_work[stencil[0][0]])
    }
    target_boundary_vertices = {
        vertex
        for edge, count in _edge_counts(target_cells).items()
        if count == 1 and not copies.intersection(edge)
        for vertex in edge
    }
    parents_by_vertex: dict[int, set[int]] = defaultdict(set)
    for cell, parents in zip(target_cells.tolist(), cell_parents, strict=True):
        for vertex in cell:
            parents_by_vertex[int(vertex)].update(parents)
    restricted = list(stencils)
    for vertex, parents in sorted(parents_by_vertex.items()):
        boundary = vertex in target_boundary_vertices
        stencil = stencils[vertex]
        support = {work_row for work_row, _ in stencil}
        # A global-chart vertex keeps its exact row unless it is a carrier-boundary
        # point off its exact boundary edge.
        if global_chart and (
            not boundary
            or len(support) == 1
            or (len(support) == 2 and tuple(sorted(support)) in work_boundary)
        ):
            continue
        common = set.intersection(
            *(set(work_cells[parent].tolist()) for parent in sorted(parents))
        )
        if support <= common and (
            len(support) == 1
            or not boundary
            or (len(support) == 2 and tuple(sorted(support)) in work_boundary)
        ):
            continue
        faces = (
            [
                frozenset(edge)
                for edge in combinations(sorted(common), 2)
                if edge in work_boundary
            ]
            if boundary
            else [frozenset(common)]
        )
        masses = [
            sum((weight for work_row, weight in stencil if work_row in face), Fraction(0))
            for face in faces
        ]
        if not faces or max(masses) <= 0:
            raise ValueError(
                "A surface metric target vertex has no exact source face shared by its parent cells."
            )
        face_row = max(range(len(faces)), key=lambda row: masses[row])
        face, retained = faces[face_row], masses[face_row]
        if len(face) == 2:
            restricted[vertex] = _edge_projection_stencil(
                source, tuple(sorted(face)), coordinates[vertex]
            )
        else:
            restricted[vertex] = tuple(
                (work_row, weight / retained)
                for work_row, weight in stencil
                if work_row in face and weight
            )
    return restricted


def _physical_stencil(
    source: SurfaceMetricSource, stencil: _WorkStencil, /
) -> tuple[tuple[int, Fraction], ...]:
    combined: dict[int, Fraction] = defaultdict(Fraction)
    for work_row, weight in stencil:
        combined[int(source.work_to_source_rows[work_row])] += weight
    result = tuple(
        (source_row, weight) for source_row, weight in sorted(combined.items()) if weight
    )
    if not result or sum((weight for _, weight in result), Fraction(0)) != 1:
        raise ValueError("A source-chart target stencil must preserve constants exactly.")
    return result


def _synchronized_copy_stencils(
    source: SurfaceMetricSource,
    stencils: list[_WorkStencil],
    group_rows: list[tuple[int, ...]],
    /,
) -> list[_WorkStencil]:
    """Express every periodic copy with its representative's exact weights.

    Copies are images of one quotient point; their binary64 shadows round
    independently, so the representative's exact row is transported onto each
    copy's own chart corners with the same physical source rows.
    """
    result = list(stencils)
    for copies in group_rows:
        reference = _physical_stencil(source, stencils[copies[0]])
        for copy in copies[1:]:
            if _physical_stencil(source, stencils[copy]) == reference:
                continue
            rows_by_source: dict[int, int] = {}
            for work_row, _ in stencils[copy]:
                rows_by_source.setdefault(
                    int(source.work_to_source_rows[work_row]), work_row
                )
            if any(source_row not in rows_by_source for source_row, _ in reference):
                raise ValueError(
                    "Synchronized source-chart copies have different physical interpolation ancestry."
                )
            result[copy] = tuple(
                sorted(
                    (rows_by_source[source_row], weight)
                    for source_row, weight in reference
                )
            )
    return result


def _exact_target_chart(
    source: SurfaceMetricSource,
    stencils: list[_WorkStencil],
    /,
) -> tuple[_ExactPoint, ...]:
    """Realize source-chart stencils exactly, including explicit face projections."""
    charts: list[_ExactPoint] = []
    for stencil in stencils:
        first = Fraction(0)
        second = Fraction(0)
        for work_row, coefficient in stencil:
            point = source.work_vertex_charts[work_row]
            first += coefficient * point[0]
            second += coefficient * point[1]
        exact_point = (first, second)
        charts.append(exact_point)
    return tuple(charts)


def _relation_rows(
    dimension: int,
    source_keys: np.ndarray,
    target_keys: np.ndarray,
    kinds: np.ndarray,
    width: int,
    /,
) -> EntityRelations:
    rows: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    for source_key, target_key, kind in zip(
        source_keys, target_keys, kinds.tolist(), strict=True
    ):
        key = (
            tuple(int(value) for value in source_key),
            tuple(int(value) for value in target_key),
        )
        previous = rows.get(key)
        if previous is not None and previous != int(kind):
            raise ValueError("Chart-cut copies disagree on source topology lineage.")
        rows[key] = int(kind)
    ordered = sorted(rows.items())
    if not ordered:
        empty = np.empty((0, width), dtype=np.int64)
        return EntityRelations(
            dimension, empty, empty.copy(), np.empty((0,), dtype=np.int8)
        )
    return EntityRelations(
        dimension,
        np.asarray([key[0] for key, _ in ordered], dtype=np.int64),
        np.asarray([key[1] for key, _ in ordered], dtype=np.int64),
        np.asarray([kind for _, kind in ordered], dtype=np.int8),
    )


def _implicit_target_witness(
    prepared: SurfaceMetricSource,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    target_cell_ids: np.ndarray,
    chart_by_cell_id: dict[
        int, tuple[int, int, tuple[int, ...], _ExactTriangle, tuple[int, ...]]
    ],
    /,
) -> tuple[SourceBoundaryChartWitness, np.ndarray, np.ndarray]:
    """Bind each target cell to the projective root chart of its parent cell.

    The analytic UV carrier only steers the edit. The source-bound witness keeps the
    adaptive root chart, so every target row retains its parent's standard root
    parameters, orientation, and root-cell/root-edge keys; the certified root map
    lifts the target cell itself.
    """
    source_charts = prepared.source_charts
    if not isinstance(source_charts, _ImplicitSurfaceChartBank):
        raise TypeError(
            "Implicit target witnesses require an implicit source chart bank."
        )
    source_witness = source_charts.witness
    source_rows: list[int] = []
    cell_keys: list[np.ndarray] = []
    root_edges: list[np.ndarray] = []
    for identifier_raw in target_cell_ids.tolist():
        identifier = int(identifier_raw)
        metadata = chart_by_cell_id[identifier]
        parent_rows = metadata[4]
        witness_rows = source_charts.source_witness_rows[
            np.asarray(parent_rows, dtype=np.int64)
        ]
        if witness_rows.size == 0:
            raise ValueError("An implicit target chart has no exact source-row ancestry.")
        source_key_rows = np.asarray(source_witness.implicit_cell_keys, dtype=np.int64)[
            witness_rows
        ]
        source_edge_rows = np.asarray(
            source_witness.implicit_root_edge_keys, dtype=np.int64
        )[witness_rows]
        if any(
            not np.array_equal(source_key_rows[0], source_key_rows[row])
            or not np.array_equal(source_edge_rows[0], source_edge_rows[row])
            for row in range(1, witness_rows.size)
        ):
            raise ValueError(
                "An implicit metric target cell straddles source root charts without a witnessed transition."
            )
        source_rows.append(int(witness_rows[0]))
        cell_keys.append(source_key_rows[0])
        root_edges.append(source_edge_rows[0])
    reference_points = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology("triangle").vertices
    )
    reference_tokens = tuple(
        tuple((value.numerator, value.denominator) for value in point)
        for point in reference_points
    )
    target_rows = np.asarray(
        [
            [[float(value) for value in point] for point in reference_points]
            for _ in target_cell_ids
        ],
        dtype=np.float64,
    )
    witness = SourceBoundaryChartWitness(
        mesh_id=target_mesh.mesh_id,
        geometry_id=cell_geometry_id(target_geometry),
        topology_id=target_mesh.topology_id,
        source_id=source_witness.source_id,
        source_revision=source_witness.source_revision,
        source_query_id=source_witness.source_query_id,
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=target_cell_ids,
        entity_global_ids=target_cell_ids,
        local_entity_indices=np.full(target_cell_ids.shape, -1, dtype=np.int32),
        target_reference_vertices=target_rows,
        target_reference_rational_tokens=tuple(reference_tokens for _ in target_cell_ids),
        source_chart_ids=tuple(
            source_witness.source_chart_ids[int(row)] for row in source_rows
        ),
        source_entity_ids=tuple(
            source_witness.source_entity_ids[int(row)] for row in source_rows
        ),
        occurrence_paths=tuple(
            source_witness.occurrence_paths[int(row)] for row in source_rows
        ),
        source_parameters=np.asarray(source_witness.source_parameters, dtype=np.float64)[
            np.asarray(source_rows, dtype=np.int64)
        ],
        source_parameter_rational_tokens=tuple(
            source_witness.source_parameter_rational_tokens[row] for row in source_rows
        ),
        orientations=np.asarray(source_witness.orientations, dtype=np.int8)[
            np.asarray(source_rows, dtype=np.int64)
        ],
        implicit_cell_keys=np.asarray(cell_keys, dtype=np.int64),
        implicit_root_edge_keys=np.asarray(root_edges, dtype=np.int64),
    )
    return (
        witness,
        np.asarray(cell_keys, dtype=np.int64),
        np.asarray(root_edges, dtype=np.int64),
    )


def _chart_barycentric_weights(
    point: _ExactPoint, chart: _ExactTriangle, /
) -> tuple[Fraction, Fraction, Fraction]:
    first, second, third = chart
    determinant = _chart_area(chart)
    if not determinant:
        raise ValueError("Source reference chart is exactly degenerate.")
    delta = (point[0] - first[0], point[1] - first[1])
    first_edge = (second[0] - first[0], second[1] - first[1])
    second_edge = (third[0] - first[0], third[1] - first[1])
    first_weight = (delta[0] * second_edge[1] - delta[1] * second_edge[0]) / determinant
    second_weight = (first_edge[0] * delta[1] - first_edge[1] * delta[0]) / determinant
    return 1 - first_weight - second_weight, first_weight, second_weight


def _reference_maps_from_charts(
    prepared: SurfaceMetricSource,
    target_cell_ids: np.ndarray,
    chart_by_cell_id: dict[
        int, tuple[int, int, tuple[int, ...], _ExactTriangle, tuple[int, ...]]
    ],
    /,
) -> tuple[ReferenceCellMapWitnesses, ...]:
    source_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64)
            for block in prepared.source_mesh.blocks
        )
    )
    reference_topology = reference_cell_topology("triangle")
    reference_points = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(reference_topology.vertices, dtype=np.float64)
    )
    variables = axes(2)
    selected_source_ids: list[int] = []
    exact_reference_rows: list[tuple[tuple[Fraction, Fraction], ...]] = []
    argument_rows = []
    target_ids = np.asarray(target_cell_ids, dtype=np.int64)
    for target_id in target_ids.tolist():
        target_chart = chart_by_cell_id[int(target_id)][3]
        parent_rows = chart_by_cell_id[int(target_id)][4]
        selected: tuple[int, _ExactTriangle] | None = None
        for source_row in parent_rows:
            source_chart = prepared.cell_charts[int(source_row)]
            barycentric = tuple(
                _chart_barycentric_weights(point, source_chart) for point in target_chart
            )
            if any(any(weight < 0 for weight in weights) for weights in barycentric):
                continue
            source_reference = tuple(
                (
                    sum(
                        (
                            weights[source_corner] * reference_points[source_corner][0]
                            for source_corner in range(3)
                        ),
                        Fraction(0),
                    ),
                    sum(
                        (
                            weights[source_corner] * reference_points[source_corner][1]
                            for source_corner in range(3)
                        ),
                        Fraction(0),
                    ),
                )
                for weights in barycentric
            )
            selected = (
                int(source_ids[int(source_row)]),
                (source_reference[0], source_reference[1], source_reference[2]),
            )
            break
        if selected is None:
            raise ValueError(
                f"Surface metric target cell {target_id} spans source chart cells without an exact transition."
            )
        source_id, reference_vertices = selected
        selected_source_ids.append(source_id)
        exact_reference_rows.append(reference_vertices)
        arguments = []
        for axis in range(2):
            expression = constant(reference_vertices[0][axis], 2)
            expression = add(
                expression,
                scale(
                    variables[0],
                    reference_vertices[1][axis] - reference_vertices[0][axis],
                ),
            )
            expression = add(
                expression,
                scale(
                    variables[1],
                    reference_vertices[2][axis] - reference_vertices[0][axis],
                ),
            )
            arguments.append(expression)
        argument_rows.append(tuple(arguments))
    maps = ReferenceCellMapWitnesses(
        target_ids,
        np.asarray(selected_source_ids, dtype=np.int64),
        "triangle",
        "triangle",
        np.asarray(
            [
                [[float(value) for value in point] for point in cell]
                for cell in exact_reference_rows
            ],
            dtype=np.float64,
        ),
        tuple(exact_reference_rows),
        freeze_reference_chart_arguments(tuple(argument_rows)),
    )
    validate_reference_cell_map_witnesses(maps)
    return (maps,)


def _source_chart_points(
    domain: MeshingDomain | _ImplicitMetricDomain,
    patches: np.ndarray,
    parameters: tuple[_ExactPoint, ...],
    /,
) -> np.ndarray:
    """Evaluate every exact target chart point in one source batch."""
    # Patch evaluators are rank-strict per chart row; the bound evaluator is
    # hoisted because binding a Module method flattens the whole domain.
    evaluate = domain.evaluate
    points = np.asarray(
        [
            np.asarray(
                evaluate(
                    np.asarray((int(patch),), dtype=np.int64),
                    np.asarray([[float(point[0]), float(point[1])]], dtype=np.float64),
                ),
                dtype=np.float64,
            )[0]
            for patch, point in zip(
                np.asarray(patches, dtype=np.int64).tolist(), parameters, strict=True
            )
        ],
        dtype=np.float64,
    ).reshape((-1, 3))
    if not isinstance(domain, _ImplicitMetricDomain):
        return points
    count = points.shape[0]
    projection = domain.profile.project_boundary(
        points,
        entity_ids=(domain.source_entity_id,) * count,
        occurrence_paths=(domain.occurrence_path,) * count,
    )
    status = np.asarray(projection.status, dtype=np.int32)
    if status.shape != (count,) or not np.all(
        np.isin(
            status,
            (int(SourceProjectionStatus.UNIQUE), int(SourceProjectionStatus.SEAM)),
        )
    ):
        raise ValueError(
            "Implicit metric target point lacks a resolved source projection."
        )
    projected = np.asarray(projection.points, dtype=np.float64)
    residual = np.max(np.abs(projected - points), axis=1, initial=0.0)
    scale = np.maximum(1.0, np.max(np.abs(projected), axis=1, initial=0.0))
    if np.any(residual > 64.0 * np.finfo(np.float64).eps * scale):
        raise ValueError(
            "Implicit chart evaluation disagrees with its certified projection."
        )
    return projected


def _root_chart_parent(
    prepared: SurfaceMetricSource,
    chart: _ExactTriangle,
    ancestry: tuple[int, ...],
    /,
) -> int:
    """Ancestor source cell whose root chart seeds one implicit target cell.

    The ancestor whose analytic chart holds the target centroid most deeply (largest
    least barycentric weight over the period lattice images) is the one whose root
    cell covers the target cell; the root chart proof certifies the choice.
    """
    images = [np.mean([[float(value) for value in point] for point in chart], axis=0)]
    for axis, period in zip(prepared.periodic_axes, prepared.periods, strict=True):
        offset = np.zeros((2,), dtype=np.float64)
        offset[axis] = float(period)
        images = [image + shift * offset for image in images for shift in (-1, 0, 1)]
    best_row, best_depth = ancestry[0], -np.inf
    for row in sorted(ancestry):
        corners = np.asarray(
            [[float(value) for value in point] for point in prepared.cell_charts[row]],
            dtype=np.float64,
        )
        frame = np.stack((corners[1] - corners[0], corners[2] - corners[0]), axis=1)
        for image in images:
            second, third = np.linalg.solve(frame, image - corners[0])
            depth = min(1.0 - second - third, second, third)
            if depth > best_depth:
                best_row, best_depth = row, depth
    return int(best_row)


def _remeasure_publication(
    prepared: SurfaceMetricSource,
    local_outcome: LocalMetricOutcome,
    work_target: CellMesh,
    target_uv: tuple[_ExactPoint, ...],
    vertex_patches: np.ndarray,
    target_metric: np.ndarray,
    work_to_physical_row: np.ndarray,
    local_cells: np.ndarray,
    target_ids: np.ndarray,
    work_stencils: list[_WorkStencil],
    local_stencils: list[_WorkStencil],
    /,
) -> tuple[LocalMetricEvidence, _State]:
    """Bind metric acceptance to the actually published chart and physical tensors."""
    local_edit = local_outcome.edit
    # Publication may project chart points or synchronize periodic stencils.
    # Measure the published physical tensor pulled back at those final charts,
    # rather than inheriting the local optimizer's pre-publication measurement.
    chart_points = np.asarray([[float(value) for value in point] for point in target_uv])
    chart_metric = _pullback_metrics(
        prepared.domain, vertex_patches, chart_points, target_metric[work_to_physical_row]
    )
    if work_target.periodic_topology is not None:
        vertex_map = periodic_entity_orbit_map(work_target, 0)
        vertex_orbits = np.asarray(vertex_map.lifted_to_quotient, dtype=np.int64)
        vertex_representatives = np.asarray(
            vertex_map.representative_lifted_rows, dtype=np.int64
        )
        windings = np.asarray(vertex_map.winding_shifts, dtype=np.int64)
        for row, orbit in enumerate(vertex_orbits):
            representative = int(vertex_representatives[orbit])
            if row != representative:
                transform = exact_periodic_element(
                    work_target.periodic_topology.cell,
                    windings[row] - windings[representative],
                )
                chart_metric[row] = _exact_periodic_metric(
                    transform, chart_metric[representative]
                )
    edges, _, _, _, cell_edges = _edge_table(local_cells)
    protected = {
        tuple(int(value) for value in key) for key in local_outcome.protected_edge_keys
    }
    measured = np.asarray(
        [
            tuple(sorted((int(target_ids[a]), int(target_ids[b])))) not in protected
            for a, b in edges
        ],
        dtype=np.bool_,
    )
    lengths = _metric_lengths(chart_metric, chart_points, edges)
    if work_target.periodic_topology is not None:
        edge_map = periodic_entity_orbit_map(work_target, 1)
        target_rows = key_rows(
            entity_keys(work_target, 1), np.sort(target_ids[edges], axis=1)
        )
        if np.any(target_rows < 0):
            raise ValueError("Published periodic chart lost a lifted edge.")
        state_rows = np.empty((edges.shape[0],), dtype=np.int64)
        state_rows[target_rows] = np.arange(edges.shape[0])
        representatives = state_rows[
            np.asarray(edge_map.representative_lifted_rows, dtype=np.int64)
        ]
        orbits = np.asarray(edge_map.lifted_to_quotient, dtype=np.int64)[target_rows]
        lengths = lengths[representatives[orbits]]
    selected = lengths[measured] if np.any(measured) else lengths
    inside = (selected >= _LOWER) & (selected <= _UPPER)
    previous = local_outcome.evidence
    converged = bool(np.all(inside))
    unchanged = (
        work_stencils == local_stencils
        and np.array_equal(chart_points, local_edit.coordinates)
        and np.array_equal(chart_metric, local_outcome.metric)
    )
    evidence = LocalMetricEvidence(
        passes=previous.passes,
        splits=previous.splits,
        collapses=previous.collapses,
        flips=previous.flips,
        relocations=previous.relocations,
        rejected_uncertain=previous.rejected_uncertain,
        rejected_invalid=previous.rejected_invalid,
        converged=converged,
        stalled=not converged
        and (
            (previous.stalled and unchanged)
            or (not previous.topology_operations and not previous.relocation)
        ),
        measured_edges=selected.size,
        out_of_range_edges=int(np.count_nonzero(~inside)),
        minimum_metric_length=float(np.min(selected)),
        maximum_metric_length=float(np.max(selected)),
        unit_fraction=float(np.count_nonzero(inside) / selected.size),
        minimum_metric_quality=float(np.min(_quality(lengths[cell_edges]))),
        topology_operations=previous.topology_operations,
        relocation=previous.relocation,
        predicate_mode=previous.predicate_mode,
        stalled_edges=(
            previous.stalled_edges
            if previous.stalled and not converged and unchanged
            else ()
        ),
    )

    source_ids = np.asarray(prepared.work_mesh.vertex_global_ids, dtype=np.int64)
    published_stencils = tuple(
        tuple((int(source_ids[row]), weight) for row, weight in stencil)
        for stencil in work_stencils
    )
    published_state = _published_state(
        local_outcome, chart_points, chart_metric, published_stencils
    )
    return evidence, published_state


def _physical_edit(
    prepared: SurfaceMetricSource,
    local_outcome: LocalMetricOutcome,
    /,
) -> tuple[
    CellTopologyEdit,
    np.ndarray,
    SurfaceMetricGeometryWitness,
    LocalMetricEvidence,
    _State,
]:
    """Assemble ordered physical identities, lineage and source witnesses together.

    Reindexing affects every published field, so this assembly retains one ordered
    owner; geometric primitives and final metric acceptance are delegated.
    """
    local_edit = local_outcome.edit
    local_blocks = tuple(
        block for block in local_edit.blocks if isinstance(block, TopologyEditBlock)
    )
    if len(local_blocks) != len(local_edit.blocks):
        raise ValueError(
            "Surface metric UV edits must retain fixed-family triangle blocks."
        )
    staged = stage_periodic_topology_edit(prepared.work_mesh, local_edit)
    work_target = staged
    if work_target is None:
        work_target = CellMesh(
            local_edit.coordinates,
            tuple(
                CellBlock(
                    block.name, block.cell_kind, block.cells, global_ids=block.cell_ids
                )
                for block in local_blocks
            ),
            vertex_global_ids=local_edit.vertex_global_ids,
        )
    target_ids = np.asarray(work_target.vertex_global_ids, dtype=np.int64)
    if not np.array_equal(
        target_ids, np.asarray(local_edit.vertex_global_ids, dtype=np.int64)
    ):
        raise ValueError("Periodic chart staging changed canonical UV vertex identities.")
    _, source_cell_ids, source_block_rows = _source_cells(prepared.source_mesh)
    source_rows_by_cell = {
        int(identifier): row for row, identifier in enumerate(source_cell_ids.tolist())
    }
    source_patch_by_cell = {
        int(identifier): int(prepared.cell_patches[row])
        for row, identifier in enumerate(source_cell_ids.tolist())
    }
    source_root_by_cell = {
        int(identifier): int(prepared.cell_root_ids[row])
        for row, identifier in enumerate(source_cell_ids.tolist())
    }
    source_piece_by_cell = {
        int(identifier): prepared.cell_source_pieces[row]
        for row, identifier in enumerate(source_cell_ids.tolist())
    }
    source_block_by_cell = {
        int(identifier): prepared.source_mesh.blocks[int(source_block_rows[row])].name
        for row, identifier in enumerate(source_cell_ids.tolist())
    }
    source_orientation_by_cell: dict[int, int] = {}
    for row, identifier in enumerate(source_cell_ids.tolist()):
        chart = prepared.cell_charts[row]
        source_orientation_by_cell[int(identifier)] = 1 if _chart_area(chart) > 0 else -1

    cell_relation = local_edit.relations[2]
    local_cells = np.concatenate(
        tuple(np.asarray(block.cells, dtype=np.int64) for block in local_blocks),
        axis=0,
    )
    local_cell_ids = np.concatenate(
        tuple(np.asarray(block.cell_ids, dtype=np.int64) for block in local_blocks)
    )
    orbit_codes = np.arange(target_ids.size, dtype=np.int64)
    if work_target.periodic_topology is not None:
        orbit_codes = np.asarray(
            periodic_entity_orbit_map(work_target, 0).lifted_to_quotient,
            dtype=np.int64,
        )
    source_vertex_ids = np.asarray(prepared.source_mesh.vertex_global_ids, dtype=np.int64)
    initial_work_ids = np.asarray(prepared.work_mesh.vertex_global_ids, dtype=np.int64)
    old_id_to_source_row = {
        int(work_id): int(source_row)
        for work_id, source_row in zip(
            initial_work_ids, prepared.work_to_source_rows, strict=True
        )
    }
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row, identifier in enumerate(target_ids.tolist()):
        source_row = old_id_to_source_row.get(int(identifier))
        if source_row is not None:
            groups[("source", int(source_vertex_ids[source_row]))].append(row)
        elif work_target.periodic_topology is not None:
            groups[("periodic", int(orbit_codes[row]))].append(row)
        else:
            groups[("new", int(identifier))].append(row)
    next_vertex_id = int(np.max(source_vertex_ids, initial=-1)) + 1
    group_records = sorted(
        groups.items(),
        key=lambda item: (0 if item[0][0] == "source" else 1, item[0][1]),
    )
    target_physical_id_by_work_row = np.empty((target_ids.size,), dtype=np.int64)
    group_rows: list[tuple[int, ...]] = []
    collapsed_groups: list[bool] = []
    for key, rows in group_records:
        physical_id = key[1] if key[0] == "source" else next_vertex_id
        if key[0] != "source":
            next_vertex_id += 1
        for row in rows:
            target_physical_id_by_work_row[row] = physical_id
        group_rows.append(tuple(rows))
        collapsed_groups.append(
            key[0] == "source"
            and bool(
                prepared.collapsed_source_vertices[
                    old_id_to_source_row[int(target_ids[rows[0]])]
                ]
            )
        )
    cell_parents = tuple(
        _source_rows_for_cell(cell_relation, int(target_id), source_rows_by_cell)
        for target_id in local_cell_ids.tolist()
    )
    local_stencils = _work_stencils(local_edit, prepared)
    local_uv = _exact_target_chart(prepared, local_stencils)
    local_carriers = np.asarray(local_edit.coordinates, dtype=np.float64)
    if local_carriers.shape != (len(local_uv), 2) or not np.array_equal(
        local_carriers,
        np.asarray([[float(value) for value in point] for point in local_uv]),
    ):
        raise ValueError(
            "A surface metric target carrier does not canonically round its exact source stencil."
        )
    work_stencils = _parent_face_stencils(
        prepared,
        local_stencils,
        np.asarray(local_edit.coordinates, dtype=np.float64),
        local_cells,
        cell_parents,
    )
    if work_target.periodic_topology is not None:
        work_stencils = _synchronized_copy_stencils(prepared, work_stencils, group_rows)
    target_uv = _exact_target_chart(prepared, work_stencils)
    global_chart = isinstance(prepared.source_charts, _ImplicitSurfaceChartBank)
    cell_meta: dict[
        int, tuple[int, int, tuple[int, ...], str, int, _ExactTriangle, tuple[int, ...]]
    ] = {}
    for cell, target_id_raw, ancestry in zip(
        local_cells.tolist(), local_cell_ids.tolist(), cell_parents, strict=True
    ):
        target_id = int(target_id_raw)
        chart_points = tuple(target_uv[int(vertex)] for vertex in cell)
        if len(chart_points) != 3:
            raise ValueError("Surface metric target cells must remain triangles.")
        chart: _ExactTriangle = (chart_points[0], chart_points[1], chart_points[2])
        area = _chart_area(chart)
        if area == 0 or (1 if area > 0 else -1) != 1:
            raise ValueError("A UV operation produced a nonpositive chart triangle.")
        ancestry_ids = tuple(int(source_cell_ids[row]) for row in ancestry)
        patches = {source_patch_by_cell[value] for value in ancestry_ids}
        blocks = {source_block_by_cell[value] for value in ancestry_ids}
        orientations = {source_orientation_by_cell[value] for value in ancestry_ids}
        # A collapsed or relocated cell of the global implicit chart may overlap
        # several extraction cells; one of them seeds its root chart.
        parents = (
            (_root_chart_parent(prepared, chart, ancestry),) if global_chart else ancestry
        )
        parent_ids = tuple(int(source_cell_ids[row]) for row in parents)
        roots = {source_root_by_cell[value] for value in parent_ids}
        pieces = {source_piece_by_cell[value] for value in parent_ids}
        if any(
            len(values) != 1 for values in (patches, roots, pieces, blocks, orientations)
        ):
            raise ValueError("A surface metric operation crossed source chart ancestry.")
        patch, root, piece, block_name, orientation = (
            next(iter(patches)),
            next(iter(roots)),
            next(iter(pieces)),
            next(iter(blocks)),
            next(iter(orientations)),
        )
        if orientation < 0:
            chart = (chart[0], chart[2], chart[1])
        cell_meta[target_id] = (
            patch,
            root,
            piece,
            block_name,
            orientation,
            chart,
            parents,
        )

    physical_ids = np.asarray(
        sorted(set(target_physical_id_by_work_row.tolist())), dtype=np.int64
    )
    physical_row_by_id = {
        int(identifier): row for row, identifier in enumerate(physical_ids.tolist())
    }
    work_to_physical_row = np.asarray(
        [
            physical_row_by_id[int(identifier)]
            for identifier in target_physical_id_by_work_row
        ],
        dtype=np.int64,
    )
    vertex_patches = np.full((target_ids.size,), -1, dtype=np.int64)
    for cell, target_id_raw in zip(
        local_cells.tolist(), local_cell_ids.tolist(), strict=True
    ):
        patch = cell_meta[int(target_id_raw)][0]
        for vertex in cell:
            previous = int(vertex_patches[vertex])
            if previous >= 0 and previous != patch:
                raise ValueError(
                    "One UV carrier vertex is shared by different source patches."
                )
            vertex_patches[vertex] = patch
    if np.any(vertex_patches < 0):
        raise ValueError("A target source-chart vertex has no source patch occurrence.")
    work_points = _source_chart_points(prepared.domain, vertex_patches, target_uv)
    physical_coordinates = np.empty((physical_ids.size, 3), dtype=np.float64)
    physical_stencils: list[tuple[tuple[int, Fraction], ...]] = []
    maximum_residual = 0.0
    for physical_row, (copies, collapsed) in enumerate(
        zip(group_rows, collapsed_groups, strict=True)
    ):
        representative = copies[0]
        point = work_points[representative]
        physical_coordinates[physical_row] = point
        stencil = _physical_stencil(prepared, work_stencils[representative])
        for copy_row in copies[1:]:
            # Copies are one vertex when their exact charts are lattice images or
            # collapsed-boundary twins; the binary64 evaluations are only roundings of
            # that identity, so their residual is evidence, not the decision.
            residual = float(np.linalg.norm(point - work_points[copy_row], ord=np.inf))
            maximum_residual = max(maximum_residual, residual)
            scale = max(1.0, float(np.linalg.norm(point, ord=np.inf)))
            if not (
                vertex_patches[copy_row] == vertex_patches[representative]
                and _exact_chart_copy(
                    target_uv[representative],
                    target_uv[copy_row],
                    prepared.periods,
                    prepared.periodic_axes,
                    collapsed=collapsed,
                )
            ) and (residual > 2.0e-10 * scale):
                raise ValueError(
                    "Periodic and chart-copy source evaluations disagree at one target vertex."
                )
            if (
                work_target.periodic_topology is not None
                and _physical_stencil(prepared, work_stencils[copy_row]) != stencil
            ):
                raise ValueError(
                    "Synchronized source-chart copies have different physical interpolation ancestry."
                )
        physical_stencils.append(stencil)
    metric_width = max(len(row) for row in physical_stencils)
    metric_rows = np.empty((len(physical_stencils), metric_width), dtype=np.int64)
    metric_weights = np.zeros((len(physical_stencils), metric_width), dtype=np.float64)
    for row, stencil in enumerate(physical_stencils):
        # Padding repeats the first sample with zero log-Euclidean weight.
        metric_rows[row] = stencil[0][0]
        for column, (source_row, weight) in enumerate(stencil):
            metric_rows[row, column] = source_row
            metric_weights[row, column] = float(weight)
    target_metric = np.asarray(
        _constant_preserving_metric_mean(
            jnp.asarray(prepared.physical_metric[metric_rows]),
            jnp.asarray(metric_weights),
        ),
        dtype=np.float64,
    )
    evidence, published_state = _remeasure_publication(
        prepared,
        local_outcome,
        work_target,
        target_uv,
        vertex_patches,
        target_metric,
        work_to_physical_row,
        local_cells,
        target_ids,
        work_stencils,
        local_stencils,
    )

    vertex_order = np.argsort(physical_ids, kind="stable")
    physical_ids = physical_ids[vertex_order]
    physical_coordinates = physical_coordinates[vertex_order]
    target_metric = target_metric[vertex_order]
    physical_stencils = [physical_stencils[int(row)] for row in vertex_order]
    reindex = np.empty((vertex_order.size,), dtype=np.int64)
    reindex[vertex_order] = np.arange(vertex_order.size, dtype=np.int64)
    work_to_physical_row = reindex[work_to_physical_row]
    stencil_width = max(len(row) for row in physical_stencils)
    stencil_sources = np.zeros((len(physical_stencils), stencil_width), dtype=np.int64)
    stencil_weights = np.zeros((len(physical_stencils), stencil_width), dtype=np.float64)
    stencil_valid = np.zeros((len(physical_stencils), stencil_width), dtype=np.bool_)
    exact_stencils: list[tuple[Fraction, ...]] = []
    for row, stencil in enumerate(physical_stencils):
        exact_row = [Fraction(0)] * stencil_width
        for column, (source_row, weight) in enumerate(stencil):
            stencil_sources[row, column] = source_row
            stencil_weights[row, column] = float(weight)
            stencil_valid[row, column] = True
            exact_row[column] = weight
        exact_stencils.append(tuple(exact_row))

    block_cells: dict[str, list[tuple[int, np.ndarray]]] = {
        block.name: [] for block in prepared.source_mesh.blocks
    }
    chart_by_cell_id: dict[
        int, tuple[int, int, tuple[int, ...], _ExactTriangle, tuple[int, ...]]
    ] = {}
    for cell, target_id_raw in zip(local_cells, local_cell_ids, strict=True):
        target_id = int(target_id_raw)
        patch, root, piece, block_name, orientation, chart, parent_rows = cell_meta[
            target_id
        ]
        physical_cell = work_to_physical_row[cell]
        if orientation < 0:
            physical_cell = physical_cell[[0, 2, 1]]
        if np.unique(physical_cell).size != 3:
            raise ValueError("A source-chart target cell has repeated physical vertices.")
        block_cells[block_name].append((target_id, physical_cell))
        chart_by_cell_id[target_id] = (patch, root, piece, chart, parent_rows)
    target_blocks: list[TopologyEditBlock] = []
    for source_block in prepared.source_mesh.blocks:
        entries = sorted(block_cells[source_block.name], key=lambda item: item[0])
        if entries:
            target_blocks.append(
                TopologyEditBlock(
                    source_block.name,
                    "triangle",
                    source_block.cell_kind,
                    np.asarray([cell for _, cell in entries], dtype=np.int64),
                    np.asarray([identifier for identifier, _ in entries], dtype=np.int64),
                )
            )

    source_work_to_physical_id = {
        int(work_id): int(source_vertex_ids[int(source_row)])
        for work_id, source_row in zip(
            initial_work_ids, prepared.work_to_source_rows, strict=True
        )
    }
    target_work_id_to_row = {
        int(identifier): row for row, identifier in enumerate(target_ids.tolist())
    }
    local_vertex_relations = local_edit.relations[0]
    vertex_relation_rows: list[tuple[int, int, int]] = []
    for source_key, target_key, kind in zip(
        np.asarray(local_vertex_relations.source_keys, dtype=np.int64).tolist(),
        np.asarray(local_vertex_relations.target_keys, dtype=np.int64).tolist(),
        np.asarray(local_vertex_relations.kinds, dtype=np.int32).tolist(),
        strict=True,
    ):
        source_id = source_work_to_physical_id.get(int(source_key[0]))
        target_row = target_work_id_to_row.get(int(target_key[0]))
        if source_id is not None and target_row is not None:
            vertex_relation_rows.append(
                (
                    source_id,
                    int(target_physical_id_by_work_row[target_row]),
                    int(kind),
                )
            )
    vertex_relations = _relation_rows(
        0,
        np.asarray([[row[0]] for row in vertex_relation_rows], dtype=np.int64).reshape(
            (-1, 1)
        ),
        np.asarray([[row[1]] for row in vertex_relation_rows], dtype=np.int64).reshape(
            (-1, 1)
        ),
        np.asarray([row[2] for row in vertex_relation_rows], dtype=np.int32),
        1,
    )

    edge_source_rows: list[tuple[int, int]] = []
    edge_target_rows: list[tuple[int, int]] = []
    edge_kind_rows: list[int] = []
    for source_key, target_key, kind in zip(
        np.asarray(local_edit.relations[1].source_keys, dtype=np.int64).tolist(),
        np.asarray(local_edit.relations[1].target_keys, dtype=np.int64).tolist(),
        np.asarray(local_edit.relations[1].kinds, dtype=np.int32).tolist(),
        strict=True,
    ):
        source_key_mapped = tuple(
            sorted(
                source_work_to_physical_id[int(value)]
                for value in source_key
                if value >= 0
            )
        )
        target_key_mapped = tuple(
            sorted(
                int(target_physical_id_by_work_row[target_work_id_to_row[int(value)]])
                for value in target_key
                if value >= 0 and int(value) in target_work_id_to_row
            )
        )
        if (
            len(source_key_mapped) == 2
            and len(target_key_mapped) == 2
            and source_key_mapped[0] != source_key_mapped[1]
            and target_key_mapped[0] != target_key_mapped[1]
        ):
            edge_source_rows.append(source_key_mapped)
            edge_target_rows.append(target_key_mapped)
            edge_kind_rows.append(int(kind))
    edge_relations = _relation_rows(
        1,
        np.asarray(edge_source_rows, dtype=np.int64).reshape((-1, 2)),
        np.asarray(edge_target_rows, dtype=np.int64).reshape((-1, 2)),
        np.asarray(edge_kind_rows, dtype=np.int32),
        2,
    )
    cell_relations = _relation_rows(
        2,
        np.asarray(local_edit.relations[2].source_keys, dtype=np.int64),
        np.asarray(local_edit.relations[2].target_keys, dtype=np.int64),
        np.asarray(local_edit.relations[2].kinds, dtype=np.int32),
        1,
    )
    edit = CellTopologyEdit(
        physical_coordinates,
        physical_ids,
        tuple(target_blocks),
        stencil_sources,
        stencil_weights,
        stencil_valid,
        (vertex_relations, edge_relations, cell_relations),
        stencil_exact_weights=tuple(exact_stencils),
    )
    target_mesh, _, _ = assemble_topology_edit(
        prepared.source_mesh, edit, numeric_version="surface-metric"
    )
    target_geometry = CellGeometrySpec.affine(target_mesh)
    target_cell_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks
        )
    )
    charts = tuple(chart_by_cell_id[int(identifier)][3] for identifier in target_cell_ids)
    patches = np.asarray(
        [chart_by_cell_id[int(identifier)][0] for identifier in target_cell_ids],
        dtype=np.int64,
    )
    roots = np.asarray(
        [chart_by_cell_id[int(identifier)][1] for identifier in target_cell_ids],
        dtype=np.int64,
    )
    pieces = tuple(chart_by_cell_id[int(identifier)][2] for identifier in target_cell_ids)
    implicit_cell_keys = None
    implicit_root_edge_keys = None
    reference_maps: tuple[ReferenceCellMapWitnesses, ...] = ()
    if isinstance(prepared.source_charts, _ImplicitSurfaceChartBank):
        relation = SurfaceMetricGeometryRelation.EXACT_PROJECTION
        witness, implicit_cell_keys, implicit_root_edge_keys = _implicit_target_witness(
            prepared,
            target_mesh,
            target_geometry,
            target_cell_ids,
            chart_by_cell_id,
        )
    else:
        relation = SurfaceMetricGeometryRelation.NESTED_REFERENCE
        reference_maps = _reference_maps_from_charts(
            prepared, target_cell_ids, chart_by_cell_id
        )
        entity_id = prepared.domain.entity_id
        surface_occurrences = prepared.domain.source_occurrences[2]
        patch_entity_ids = {
            int(patch): entity_id(2, int(patch)) for patch in np.unique(patches).tolist()
        }
        patch_occurrences = {
            int(patch): surface_occurrences[int(patch)]
            for patch in np.unique(patches).tolist()
        }
        geometry_id = cell_geometry_id(target_geometry)
        witness = SurfaceChartWitness(
            geometry_id=geometry_id,
            topology_id=target_mesh.topology_id,
            domain_id=prepared.domain.domain_id,
            cell_global_ids=target_cell_ids,
            patches=patches,
            charts=np.asarray(
                [
                    [[float(value) for value in point] for point in chart]
                    for chart in charts
                ],
                dtype=np.float64,
            ),
            source_root_cell_ids=roots,
            source_piece_ids=pieces,
            geometry_entity_ids=tuple(
                patch_entity_ids[int(patch)] for patch in patches.tolist()
            ),
            occurrence_paths=tuple(
                patch_occurrences[int(patch)] for patch in patches.tolist()
            ),
            root_mesh_cell_bindings=tuple(
                SurfaceRootMeshCellBinding(
                    geometry_id, target_mesh.topology_id, int(identifier)
                )
                for identifier in target_cell_ids.tolist()
            ),
            chart_rational_tokens=tuple(_triangle_tokens(chart) for chart in charts),
        )
    periodic_topology_id = None
    periodic_vertex_orbits = 0
    periodic_edge_orbits = 0
    if work_target.periodic_topology is not None:
        periodic_topology_id = work_target.periodic_topology.periodic_topology_id
        periodic_vertex_orbits = int(
            np.asarray(periodic_entity_orbit_map(work_target, 0).quotient_entity_ids).size
        )
        periodic_edge_orbits = int(
            np.asarray(periodic_entity_orbit_map(work_target, 1).quotient_entity_ids).size
        )
    geometry = SurfaceMetricGeometryWitness(
        prepared.domain.source_id,
        prepared.domain.source_revision,
        prepared.domain.domain_id,
        prepared.source_charts.source_chart_id,
        witness,
        maximum_residual,
        periodic_topology_id,
        periodic_vertex_orbits,
        periodic_edge_orbits,
        relation,
        implicit_cell_keys,
        implicit_root_edge_keys,
        reference_maps,
    )
    return edit, target_metric, geometry, evidence, published_state


class _PullbackField:
    """Physical log-Euclidean source field pulled back through its chart."""

    def __init__(self, source: SurfaceMetricSource, /) -> None:
        self.source = source

    def __call__(
        self, points: np.ndarray, rows: np.ndarray, weights: np.ndarray, /
    ) -> np.ndarray:
        source = self.source
        count = points.shape[0]
        capacity = 1 << max(count - 1, 0).bit_length()
        samples = np.broadcast_to(np.eye(3), (capacity, rows.shape[1], 3, 3)).copy()
        samples[:count] = source.physical_metric[source.work_to_source_rows[rows]]
        coefficients = np.zeros((capacity, rows.shape[1]), dtype=np.float64)
        coefficients[:, 0] = 1.0
        coefficients[:count] = weights
        physical = np.asarray(
            _constant_preserving_metric_mean(
                jnp.asarray(samples), jnp.asarray(coefficients)
            ),
            dtype=np.float64,
        )[:count]
        return _pullback_metrics(
            source.domain,
            source.work_vertex_patches[rows[:, 0]],
            points,
            physical,
        )


def _pullback_field(source: SurfaceMetricSource, /) -> _PullbackField:
    return _PullbackField(source)


def execute_surface_metric_adaptation(
    source: SurfaceMetricSource,
    /,
    *,
    predicate_mode: PredicateMode,
    topology_operations: bool,
    relocation: bool,
) -> SurfaceMetricOutcome:
    """Run split/collapse/flip/relocation on source UV with tensor pullback."""
    if not isinstance(source, SurfaceMetricSource):
        raise TypeError("source must be SurfaceMetricSource.")
    if not isinstance(topology_operations, bool) or not isinstance(relocation, bool):
        raise TypeError("topology_operations and relocation must be bool values.")
    # Every UV copy of a collapsed source vertex (a sphere pole) is one physical
    # vertex: the local engine frees the regular vertices around it and decides
    # collapses and flips on the physical surface.
    collapsed = source.collapsed_source_vertices[source.work_to_source_rows]
    local = execute_local_metric_adaptation(
        source.work_mesh,
        source.pullback_metric,
        cell_classes=source.source_cell_classes,
        edge_classes=source.source_edge_classes,
        protected_edges=source.source_protected_edges,
        fixed_vertices=source.source_fixed_vertices,
        predicate_mode=predicate_mode,
        topology_operations=topology_operations,
        relocation=relocation,
        collapsed_groups=(
            np.where(collapsed, source.work_to_source_rows, -1)
            if np.any(collapsed)
            else None
        ),
        # A collapsed chart point makes the pullback singular there, so generated
        # points take the exact pullback rather than chart-linear interpolation.
        metric_field=_pullback_field(source) if np.any(collapsed) else None,
        exact_source_points=source.work_vertex_charts,
    )
    # Brent checkpoints retain one numerical publication state. A cycle stops at
    # executed chart/tensor resolution; it proves neither exact-source equality
    # nor infeasibility. The current exact witnesses remain on the returned state.
    checkpoint: tuple[bytes, bytes, bytes, bytes] | None = None
    checkpoint_span = 1
    checkpoint_age = 0
    while True:
        edit, target_metric, geometry, evidence, published = _physical_edit(source, local)
        if evidence.converged or evidence.stalled or evidence.rejected_uncertain:
            break
        if not topology_operations and not relocation:
            break
        order, _, final_ids = _target_order(published)
        signature = (
            published.points[order].tobytes(),
            published.metric[order].tobytes(),
            final_ids[published.cells].tobytes(),
            local.protected_edge_keys.tobytes(),
        )
        if signature == checkpoint:
            break
        if checkpoint is None or checkpoint_age == checkpoint_span:
            if checkpoint is not None:
                checkpoint_span *= 2
            checkpoint = signature
            checkpoint_age = 0
        checkpoint_age += 1
        old = local.evidence
        counts = _Counts(
            old.splits,
            old.collapses,
            old.flips,
            old.relocations,
            old.rejected_uncertain,
            old.rejected_invalid,
        )
        local = _resume_local_metric_adaptation(
            source.work_mesh,
            local.source,
            published,
            predicate_mode,
            topology_operations,
            relocation,
            counts,
            old.passes,
        )
    if evidence.converged:
        status = SurfaceMetricStatus.COMPLETE
    elif evidence.rejected_uncertain:
        status = SurfaceMetricStatus.NUMERICAL_STOP
    elif evidence.stalled:
        status = SurfaceMetricStatus.STALLED
    else:
        status = SurfaceMetricStatus.NUMERICAL_STOP
    return SurfaceMetricOutcome(
        edit,
        target_metric,
        geometry,
        evidence,
        status,
        evidence.passes,
        evidence.splits,
        evidence.collapses,
        evidence.flips,
        evidence.relocations,
        evidence.out_of_range_edges,
        evidence.minimum_metric_length,
        evidence.maximum_metric_length,
        evidence.unit_fraction,
    )


def target_surface_source(
    source_charts: SurfaceSourceCharts,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    associations: tuple[GeometryAssociation, ...],
    outcome: SurfaceMetricOutcome,
    /,
) -> SurfaceSourceCharts:
    """Refresh the exact source-chart bank after route-owned lineage assembly."""
    if not isinstance(source_charts, SurfaceSourceCharts):
        raise TypeError("source_charts must be SurfaceSourceCharts.")
    if not isinstance(target_mesh, CellMesh) or not isinstance(
        target_geometry, CellGeometrySpec
    ):
        raise TypeError("Target source charts require CellMesh and CellGeometrySpec.")
    if not isinstance(outcome, SurfaceMetricOutcome):
        raise TypeError("outcome must be SurfaceMetricOutcome.")
    source_charts.require_current()
    geometry = outcome.geometry
    if (
        geometry.source_id != source_charts.source_id
        or geometry.source_revision != source_charts.source_revision
        or geometry.domain_id != source_charts.domain_id
        or geometry.source_chart_id != source_charts.source_chart_id
    ):
        raise ValueError(
            "Surface metric chart transition changed its fixed source identity."
        )
    witness = geometry.target_chart_witness
    if not isinstance(witness, SurfaceChartWitness):
        raise TypeError(
            "Implicit source charts require their source-bound witness route."
        )
    if (
        witness.topology_id != target_mesh.topology_id
        or witness.geometry_id != cell_geometry_id(target_geometry)
    ):
        raise ValueError(
            "Surface metric target chart witness is stale for its geometry endpoint."
        )
    target_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks
        )
    )
    witness_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    witness_rows = {
        int(identifier): row for row, identifier in enumerate(witness_ids.tolist())
    }
    if target_ids.size != witness_ids.size or set(target_ids.tolist()) != set(
        witness_rows
    ):
        raise ValueError("Surface metric chart witness does not cover every target cell.")
    rows = np.asarray(
        [witness_rows[int(identifier)] for identifier in target_ids], dtype=np.int64
    )
    exact_chart = witness.exact_chart
    exact_charts = tuple(exact_chart(int(row)) for row in rows.tolist())
    return SurfaceSourceCharts(
        source_charts.domain,
        target_mesh,
        target_geometry,
        source_charts.coordinate_contract,
        np.asarray(witness.charts, dtype=np.float64)[rows],
        np.asarray(witness.patches, dtype=np.int64)[rows],
        tuple(associations),
        source_charts.boundary_source,
        source_root_cell_ids=np.asarray(witness.source_root_cell_ids, dtype=np.int64)[
            rows
        ],
        source_piece_ids=tuple(
            witness.source_piece_ids[int(row)] for row in rows.tolist()
        ),
        root_parameter_exact=exact_charts,
    )


__all__ = [
    "SurfaceMetricGeometryRelation",
    "SurfaceMetricGeometryWitness",
    "SurfaceMetricOutcome",
    "SurfaceMetricSource",
    "SurfaceMetricStatus",
    "execute_surface_metric_adaptation",
    "prepare_implicit_surface_metric_source",
    "prepare_surface_metric_source",
    "target_surface_source",
]
