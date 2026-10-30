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

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from .._fingerprint import canonical_fingerprint
from .._physical import SpatialCoordinateContract
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._periodic_cell import PeriodicCell
from ..discretization._periodic_topology import PeriodicMeshTopology
from ..discretization._surface_chart_records import (
    SurfaceChartWitness,
    SurfaceRootMeshCellBinding,
)
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._coordinate_enclosure import add, axes, constant, scale
from ..discretization._nested_reference_types import (
    ReferenceCellMapWitnesses,
    freeze_reference_chart_arguments,
    validate_reference_cell_map_witnesses,
)
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryQuery,
    SourceProjectionStatus,
)
from ..geometry._meshing_domain import MeshingDomain, _PlacedSurfacePatch
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.brep._patches import RevolutionSurface, SpherePatch
from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ._association import GeometryAssociation
from ._local_metric import LocalMetricEvidence, LocalMetricOutcome, execute_local_metric_adaptation
from ._metric import _tensor_properties, interpolate_mesh_metric
from ._periodic import periodic_entity_orbit_map, stage_periodic_topology_edit
from ._topology_edit import (
    CellTopologyEdit,
    EntityRelations,
    TopologyEditBlock,
    assemble_topology_edit,
    entity_keys,
    key_rows,
)




class SurfaceMetricStatus(StrEnum):
    """Scientific termination of one source-chart metric request."""

    COMPLETE = "complete"
    STALLED = "stalled"
    NUMERICAL_STOP = "numerical_stop"


@dataclass(frozen=True, slots=True)
class _ImplicitParameterSurface:
    profile: AnalyticImplicitProfile

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return (None, float(np.tau)) if self.profile.family == "sphere" else (
            float(np.tau),
            float(np.tau),
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
            raise ValueError("Implicit metric charts expose one two-dimensional source entity.")
        return self.source_entity_id

    def evaluate(self, patches: ArrayLike, charts: ArrayLike, /) -> np.ndarray:
        rows = np.asarray(patches, dtype=np.int64).reshape((-1,))
        parameters = np.asarray(charts, dtype=np.float64).reshape((-1, 2))
        if rows.shape != (parameters.shape[0],) or np.any(rows != 0):
            raise ValueError("Implicit metric point evaluation requires source chart zero.")
        return np.asarray(self.surface.evaluate(jnp.asarray(parameters)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _ImplicitSurfaceChartBank:
    mesh: CellMesh
    geometry: CellGeometrySpec
    domain: _ImplicitMetricDomain
    witness: SourceBoundaryChartWitness
    fidelity_source: SourceBoundaryQuery
    coordinate_contract: SpatialCoordinateContract
    cell_global_ids: tuple[int, ...]
    source_witness_rows: np.ndarray
    root_parameters: np.ndarray
    root_patches: np.ndarray
    source_root_cell_ids: np.ndarray
    source_piece_ids: tuple[tuple[int, ...], ...]
    source_id: str
    source_revision: str
    source_chart_id: str

    def root_parameter_chart(self, row: int, /) -> tuple[tuple[Fraction, ...], ...]:
        if not 0 <= row < self.source_witness_rows.size:
            raise IndexError("Implicit source chart row is out of range.")
        return self.witness.source_parameter_chart(int(self.source_witness_rows[row]))

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
    pullback_metric: np.ndarray
    physical_metric: np.ndarray
    source_cell_classes: np.ndarray
    source_edge_classes: np.ndarray
    source_protected_edges: np.ndarray
    source_fixed_vertices: np.ndarray
    periodic_rank: int
    periodic_axes: tuple[int, ...]


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
        raise ValueError("Surface metric adaptation requires a two-dimensional mesh in 3-space.")
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


def _source_metric_rows(
    mesh: CellMesh, values: ArrayLike, /
) -> np.ndarray:
    metric = np.asarray(values, dtype=np.float64)
    if metric.shape != (mesh.coordinates.shape[0], 3, 3) or not np.all(np.isfinite(metric)):
        raise ValueError("Surface metric values must be finite (vertices, 3, 3) ambient tensors.")
    properties = _tensor_properties(metric)
    if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
        raise ValueError("Surface metric tensors must be symmetric positive definite.")
    return metric




def _chart_area(chart: _ExactTriangle, /) -> Fraction:
    first, second, third = chart
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _integer_class_rows(keys: tuple[tuple[object, ...], ...], /) -> np.ndarray:
    identifiers = {key: row for row, key in enumerate(sorted(set(keys), key=repr))}
    return np.asarray([identifiers[key] for key in keys], dtype=np.int64)


def _periods(
    domain: MeshingDomain | _ImplicitMetricDomain,
    patches: np.ndarray,
    charts: tuple[tuple[_ExactPoint, ...], ...],
    /,
) -> tuple[tuple[Fraction, ...], tuple[int, ...]]:
    active: tuple[tuple[int, Fraction], ...] | None = None
    for patch in np.unique(patches).tolist():
        surface = domain.patches[int(patch)].surface
        values = surface.periods
        current = tuple(
            (axis, Fraction.from_float(float(value)))
            for axis, value in enumerate(values)
            if value is not None
        )
        if current:
            chart_parameters = np.asarray(
                [
                    [float(value) for value in point]
                    for row, chart in enumerate(charts)
                    if int(patches[row]) == int(patch)
                    for point in chart
                ],
                dtype=np.float64,
            )
            differential = jax.jacfwd(surface.evaluate)
            jacobians = np.asarray(
                jax.vmap(differential)(
                    jnp.asarray(chart_parameters, dtype=jnp.float64)
                ),
                dtype=np.float64,
            )
            scales = np.maximum(
                1.0, np.linalg.norm(jacobians, axis=(1, 2))
            )
            if any(
                np.any(
                    np.linalg.norm(jacobians[:, :, axis], axis=1)
                    <= 32.0 * np.finfo(np.float64).eps * scales
                )
                for axis, _ in current
            ):
                current = ()
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
    if not periods:
        return representatives, shifts, np.arange(count, dtype=np.int64)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row, (source_row, patch) in enumerate(
        zip(vertex_source_rows.tolist(), vertex_patches.tolist(), strict=True)
    ):
        groups[(int(source_row), int(patch))].append(row)
    orbit_codes = np.arange(count, dtype=np.int64)
    for rows in groups.values():
        if len(rows) < 2:
            continue
        anchor = min(rows)
        anchor_chart = vertex_charts[anchor]
        compatible: list[tuple[int, tuple[int, ...]]] = []
        for row in rows:
            valid = True
            winding: list[int] = []
            chart = vertex_charts[row]
            for period, axis in zip(periods, periodic_axes, strict=True):
                difference = chart[axis] - anchor_chart[axis]
                turns = difference / period
                if turns.denominator != 1:
                    valid = False
                    break
                winding.append(turns.numerator)
            if valid and any(
                chart[axis] != anchor_chart[axis]
                for axis in range(2)
                if axis not in periodic_axes
            ):
                valid = False
            if valid:
                compatible.append((row, tuple(winding)))
        if len(compatible) < 2:
            continue
        representative = min(row for row, _ in compatible)
        reference = next(winding for row, winding in compatible if row == representative)
        for row, winding in compatible:
            relative = tuple(value - base for value, base in zip(winding, reference, strict=True))
            if row != representative or any(relative):
                representatives[row] = representative
                shifts[row] = np.asarray(relative, dtype=np.int64)
                orbit_codes[row] = representative
    return representatives, shifts, orbit_codes


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
    result = np.empty((len(charts), 2, 2), dtype=np.float64)
    chart_values = np.asarray([[float(value) for value in point] for point in charts], dtype=np.float64)
    for patch in np.unique(patches).tolist():
        rows = np.flatnonzero(patches == patch)
        surface = domain.patches[int(patch)].surface
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
                        (np.asarray(base_surface.first_axis), np.asarray(base_surface.second_axis)),
                        axis=1,
                    )
                    jacobian = chart_frame @ (
                        2.0 * float(np.asarray(base_surface.radius)) * frame
                    )
            elif isinstance(base_surface, RevolutionSurface):
                scale = max(1.0, float(np.linalg.norm(jacobian)))
                if np.linalg.norm(jacobian[:, 0]) <= 32.0 * np.finfo(np.float64).eps * scale:
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
                        2.0
                        * radial_speed
                        * np.stack((first_axis, second_axis), axis=1)
                    )
            elif (
                isinstance(surface, _ImplicitParameterSurface)
                and surface.profile.family == "sphere"
            ):
                polar = chart_values[row, 0]
                if min(abs(polar), abs(np.pi - polar)) <= 32.0 * np.finfo(np.float64).eps:
                    frame = np.asarray(
                        ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
                        dtype=np.float64,
                    )
                    jacobian = 2.0 * surface.profile.radius * frame
            value = jacobian.T @ physical_metric[int(source_rows[row])] @ jacobian
            result[row] = 0.5 * (value + value.T)
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
                raise ValueError("Pulled-back SPD metrics disagree across a periodic source orbit.")
            result[int(row)] = reference
    properties = _tensor_properties(result)
    if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
        raise ValueError("Physical metric pullback must remain symmetric positive definite in every chart.")
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
        raise TypeError("Surface metric preparation requires a CellMesh and source chart bank.")
    source_charts.require_root(mesh, source_charts.geometry)
    domain = source_charts.domain
    domain.require_current(source_charts.source_id, source_charts.source_revision)
    physical_metric = _source_metric_rows(mesh, metric)
    source_cells, source_cell_ids, source_block_rows = _source_cells(mesh)
    chart_rows = {
        int(identifier): row for row, identifier in enumerate(source_charts.cell_global_ids)
    }
    if set(chart_rows) != set(source_cell_ids.tolist()):
        raise ValueError("Surface source charts must cover every current source cell exactly once.")
    cell_source_rows = np.asarray(
        [chart_rows[int(identifier)] for identifier in source_cell_ids], dtype=np.int64
    )
    cell_patches = np.asarray(source_charts.root_patches, dtype=np.int64)[cell_source_rows]
    cell_root_ids = np.asarray(source_charts.source_root_cell_ids, dtype=np.int64)[
        cell_source_rows
    ]
    cell_pieces = tuple(
        source_charts.source_piece_ids[int(row)] for row in cell_source_rows
    )
    exact_charts = tuple(
        source_charts.root_parameter_chart(int(row)) for row in cell_source_rows
    )
    if domain.source_id != source_charts.source_id:
        raise ValueError("Source charts no longer bind their declared source domain.")
    periods, periodic_axes = _periods(domain, cell_patches, exact_charts)
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
    if supplied_edge_classes.shape != (edge_count,) or supplied_protected.shape != (edge_count,):
        raise ValueError("Edge classes and protection must align with source edges.")
    if supplied_fixed.shape != (mesh.coordinates.shape[0],):
        raise ValueError("fixed_vertices must align with source vertices.")
    if supplied_cell_classes.dtype.kind not in "iu" or supplied_edge_classes.dtype.kind not in "iu":
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

    source_vertex_rows = np.asarray([record[0] for record in vertex_records], dtype=np.int64)
    work_vertex_patches = np.asarray([record[1] for record in vertex_records], dtype=np.int64)
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
    if work_mesh.periodic_topology is None:
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
    cell_class_keys = tuple(
        (
            (
                int(supplied_cell_classes[row]),
                int(cell_patches[row]),
                int(cell_class_ids[row]),
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
    # Anchor source vertices so relocated target cells retain one exact source chart.
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
    for row, source_edge_row in enumerate(work_edge_source_rows.tolist()):
        if (
            counts[row] == 1
            and source_edge_row >= 0
            and source_counts[int(source_edge_row)] == 2
            and orbit_sizes[edge_orbits[row]] == 1
        ):
            work_protected[row] = True
            work_fixed[work_edge_vertex_rows[row]] = True
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
        source_cell_interface = int(source_cell_ids[first]) != int(source_cell_ids[second])
        root_interface = (
            implicit_source
            and (
                cell_root_ids[first] != cell_root_ids[second]
                or cell_pieces[first] != cell_pieces[second]
            )
        )
        if (
            work_cell_classes[first] != work_cell_classes[second]
            or source_cell_interface
            or root_interface
        ):
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
        pulled,
        physical_metric,
        work_cell_classes,
        work_edge_classes,
        work_protected,
        work_fixed,
        len(periods),
        periodic_axes,
    )


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
        raise ValueError("Implicit metric source requires exact cell and root-edge ancestry.")
    _, source_cell_ids, _ = _source_cells(mesh)
    witness_rows_by_cell = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(witness.cell_global_ids, dtype=np.int64).tolist())
    }
    if set(witness_rows_by_cell) != set(source_cell_ids.tolist()):
        raise ValueError("Implicit source chart witness must cover every source cell.")
    witness_rows = np.asarray(
        [witness_rows_by_cell[int(identifier)] for identifier in source_cell_ids],
        dtype=np.int64,
    )
    implicit_cells = np.asarray(witness.implicit_cell_keys, dtype=np.int64)[witness_rows]
    root_tokens = tuple(tuple(int(value) for value in row) for row in implicit_cells.tolist())
    ordered_roots = sorted(set(root_tokens))
    root_codes = {key: row for row, key in enumerate(ordered_roots)}
    root_ids = np.asarray([root_codes[key] for key in root_tokens], dtype=np.int64)
    source_pieces = tuple((0, int(root_id)) for root_id in root_ids.tolist())
    if len(set(witness.source_entity_ids)) != 1 or len(set(witness.occurrence_paths)) != 1:
        raise ValueError("One analytic implicit surface metric route requires one source occurrence.")
    domain = _ImplicitMetricDomain(
        profile,
        witness.source_query_id,
        witness.source_entity_ids[0],
        witness.occurrence_paths[0],
        _ImplicitParameterSurface(profile),
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
        np.asarray(witness.source_parameters, dtype=np.float64)[witness_rows],
        np.zeros((source_cell_ids.size,), dtype=np.int64),
        root_ids,
        source_pieces,
        profile.source_id,
        profile.source_revision,
        witness.witness_id,
    )
    bank.require_root(mesh, geometry)
    entity_id = witness.source_entity_ids[0]
    occurrences = tuple(witness.occurrence_paths[0] for _ in range(mesh.coordinates.shape[0]))
    projected = profile.project_boundary(
        np.asarray(mesh.coordinates, dtype=np.float64),
        entity_ids=tuple(entity_id for _ in range(mesh.coordinates.shape[0])),
        occurrence_paths=occurrences,
    )
    if not np.all(
        np.isin(
            np.asarray(projected.status),
            (int(SourceProjectionStatus.UNIQUE), int(SourceProjectionStatus.SEAM)),
        )
    ):
        raise ValueError("Implicit metric source vertices lack unique source-bound projections.")
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


def _exact_target_chart(
    outcome: LocalMetricOutcome,
    source: SurfaceMetricSource,
    /,
) -> tuple[tuple[_ExactPoint, ...], ...]:
    edit = outcome.edit
    exact_weights = edit.stencil_exact_weights
    if exact_weights is None:
        raise ValueError("Local metric execution did not retain exact source stencils.")
    sources = np.asarray(edit.stencil_sources, dtype=np.int64)
    valid = np.asarray(edit.stencil_valid, dtype=np.bool_)
    coordinates = np.asarray(edit.coordinates, dtype=np.float64)
    if coordinates.shape != (sources.shape[0], 2):
        raise ValueError("Local metric UV coordinates must align with exact source stencils.")
    charts: list[_ExactPoint] = []
    work_vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(source.work_mesh.vertex_global_ids, dtype=np.int64).tolist()
        )
    }
    for row in range(sources.shape[0]):
        first = Fraction(0)
        second = Fraction(0)
        for source_row, coefficient, is_valid in zip(
            sources[row], exact_weights[row], valid[row], strict=True
        ):
            if is_valid:
                work_row = work_vertex_rows.get(int(source_row))
                if work_row is None:
                    raise ValueError(
                        "A local metric stencil names an unknown source chart vertex."
                    )
                point = source.work_vertex_charts[work_row]
                first += coefficient * point[0]
                second += coefficient * point[1]
        exact_point = (first, second)
        exact_shadow = np.asarray(tuple(float(value) for value in exact_point))
        coordinate_scale = max(1.0, float(np.linalg.norm(coordinates[row], ord=np.inf)))
        if (
            np.linalg.norm(exact_shadow - coordinates[row], ord=np.inf)
            <= 4.0 * np.finfo(np.float64).eps * coordinate_scale
        ):
            charts.append(exact_point)
        else:
            charts.append(
                (
                    Fraction.from_float(float(coordinates[row, 0])),
                    Fraction.from_float(float(coordinates[row, 1])),
                )
            )
    return tuple(charts)


def _target_source_vertex_stencil(
    local_edit: CellTopologyEdit,
    source: SurfaceMetricSource,
    work_vertex_rows: dict[int, int],
    row: int,
    /,
) -> tuple[tuple[int, Fraction], ...]:
    exact_weights = local_edit.stencil_exact_weights
    if exact_weights is None:
        raise ValueError("Local metric execution omitted exact source interpolation weights.")
    sources = np.asarray(local_edit.stencil_sources, dtype=np.int64)[row]
    weights = exact_weights[row]
    valid = np.asarray(local_edit.stencil_valid, dtype=np.bool_)[row]
    combined: dict[int, Fraction] = defaultdict(Fraction)
    for source_id, weight, is_valid in zip(sources, weights, valid, strict=True):
        if is_valid:
            work_row = work_vertex_rows.get(int(source_id))
            if work_row is None:
                raise ValueError("A local metric stencil names an unknown source chart vertex.")
            combined[int(source.work_to_source_rows[work_row])] += weight
    result = tuple((source_row, weight) for source_row, weight in sorted(combined.items()) if weight)
    if not result or sum((weight for _, weight in result), Fraction(0)) != 1:
        raise ValueError("A source-chart target stencil must preserve constants exactly.")
    return result


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
        return EntityRelations(dimension, empty, empty.copy(), np.empty((0,), dtype=np.int8))
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
    source_charts = prepared.source_charts
    if not isinstance(source_charts, _ImplicitSurfaceChartBank):
        raise TypeError("Implicit target witnesses require an implicit source chart bank.")
    source_witness = source_charts.witness
    source_rows: list[int] = []
    cell_keys: list[np.ndarray] = []
    root_edges: list[np.ndarray] = []
    exact_charts: list[_ExactTriangle] = []
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
        source_edge_rows = np.asarray(source_witness.implicit_root_edge_keys, dtype=np.int64)[
            witness_rows
        ]
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
        exact_charts.append(metadata[3])
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
    source_rows_array = np.asarray(source_rows, dtype=np.int64)
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
        target_reference_rational_tokens=tuple(
            reference_tokens for _ in target_cell_ids
        ),
        source_chart_ids=tuple(
            source_witness.source_chart_ids[int(row)] for row in source_rows
        ),
        source_entity_ids=tuple(
            source_witness.source_entity_ids[int(row)] for row in source_rows
        ),
        occurrence_paths=tuple(
            source_witness.occurrence_paths[int(row)] for row in source_rows
        ),
        source_parameters=np.asarray(
            [
                [[float(value) for value in point] for point in chart]
                for chart in exact_charts
            ],
            dtype=np.float64,
        ),
        source_parameter_rational_tokens=tuple(
            tuple(
                tuple((value.numerator, value.denominator) for value in point)
                for point in chart
            )
            for chart in exact_charts
        ),
        orientations=np.asarray(
            [1 if _chart_area(chart) > 0 else -1 for chart in exact_charts],
            dtype=np.int8,
        ),
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
    first_weight = (
        delta[0] * second_edge[1] - delta[1] * second_edge[0]
    ) / determinant
    second_weight = (
        first_edge[0] * delta[1] - first_edge[1] * delta[0]
    ) / determinant
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
    source_rows_by_id = {
        int(identifier): row for row, identifier in enumerate(source_ids.tolist())
    }
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
        selected: tuple[int, tuple[tuple[Fraction, Fraction], ...]] | None = None
        for source_row in parent_rows:
            chart_row = int(prepared.cell_source_rows[int(source_row)])
            source_chart_raw = prepared.source_charts.root_parameter_chart(chart_row)
            source_chart: _ExactTriangle = (
                source_chart_raw[0],
                source_chart_raw[1],
                source_chart_raw[2],
            )
            barycentric = tuple(
                _chart_barycentric_weights(point, source_chart)
                for point in target_chart
            )
            if any(
                any(weight < 0 for weight in weights)
                for weights in barycentric
            ):
                continue
            source_reference = tuple(
                tuple(
                    sum(
                        (
                            barycentric[corner][source_corner]
                            * reference_points[source_corner][axis]
                            for source_corner in range(3)
                        ),
                        Fraction(0),
                    )
                    for axis in range(2)
                )
                for corner in range(3)
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


def _source_chart_point(
    domain: MeshingDomain | _ImplicitMetricDomain,
    patch: int,
    parameters: _ExactPoint,
    /,
) -> np.ndarray:
    charts = np.asarray(
        [[float(parameters[0]), float(parameters[1])]], dtype=np.float64
    )
    point = domain.evaluate(np.asarray((patch,), dtype=np.int64), charts)[0]
    if isinstance(domain, _ImplicitMetricDomain):
        projection = domain.profile.project_boundary(
            point[None, :],
            entity_ids=(domain.source_entity_id,),
            occurrence_paths=(domain.occurrence_path,),
        )
        status = np.asarray(projection.status, dtype=np.int32)
        if status.shape != (1,) or int(status[0]) not in (
            int(SourceProjectionStatus.UNIQUE),
            int(SourceProjectionStatus.SEAM),
        ):
            raise ValueError("Implicit metric target point lacks a resolved source projection.")
        projected = np.asarray(projection.points, dtype=np.float64)
        residual = float(np.linalg.norm(projected[0] - point, ord=np.inf))
        scale = max(1.0, float(np.linalg.norm(projected[0], ord=np.inf)))
        if residual > 64.0 * np.finfo(np.float64).eps * scale:
            raise ValueError("Implicit chart evaluation disagrees with its certified projection.")
        return projected[0]
    return point


def _physical_edit(
    prepared: SurfaceMetricSource,
    local_outcome: LocalMetricOutcome,
    /,
) -> tuple[CellTopologyEdit, np.ndarray, SurfaceMetricGeometryWitness]:
    local_edit = local_outcome.edit
    staged = stage_periodic_topology_edit(prepared.work_mesh, local_edit)
    work_target = staged
    if work_target is None:
        work_target = CellMesh(
            local_edit.coordinates,
            tuple(
                CellBlock(block.name, block.cell_kind, block.cells, global_ids=block.cell_ids)
                for block in local_edit.blocks
            ),
            vertex_global_ids=local_edit.vertex_global_ids,
        )
    target_ids = np.asarray(work_target.vertex_global_ids, dtype=np.int64)
    if not np.array_equal(target_ids, np.asarray(local_edit.vertex_global_ids, dtype=np.int64)):
        raise ValueError("Periodic chart staging changed canonical UV vertex identities.")
    target_uv = _exact_target_chart(local_outcome, prepared)
    work_vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(prepared.work_mesh.vertex_global_ids, dtype=np.int64).tolist()
        )
    }
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
        chart_row = int(prepared.cell_source_rows[row])
        chart = prepared.source_charts.root_parameter_chart(chart_row)
        source_orientation_by_cell[int(identifier)] = 1 if _chart_area(chart) > 0 else -1

    cell_relation = local_edit.relations[2]
    local_cells = np.concatenate(
        tuple(np.asarray(block.cells, dtype=np.int64) for block in local_edit.blocks), axis=0
    )
    local_cell_ids = np.concatenate(
        tuple(np.asarray(block.cell_ids, dtype=np.int64) for block in local_edit.blocks)
    )
    cell_meta: dict[
        int, tuple[int, int, tuple[int, ...], str, int, _ExactTriangle, tuple[int, ...]]
    ] = {}
    for cell, target_id_raw in zip(local_cells, local_cell_ids, strict=True):
        target_id = int(target_id_raw)
        parents = _source_rows_for_cell(cell_relation, target_id, source_rows_by_cell)
        parent_ids = tuple(int(source_cell_ids[row]) for row in parents)
        patches = {source_patch_by_cell[value] for value in parent_ids}
        roots = {source_root_by_cell[value] for value in parent_ids}
        pieces = {source_piece_by_cell[value] for value in parent_ids}
        blocks = {source_block_by_cell[value] for value in parent_ids}
        orientations = {source_orientation_by_cell[value] for value in parent_ids}
        if any(len(values) != 1 for values in (patches, roots, pieces, blocks, orientations)):
            raise ValueError("A surface metric operation crossed source chart ancestry.")
        patch, root, piece, block_name, orientation = (
            next(iter(patches)),
            next(iter(roots)),
            next(iter(pieces)),
            next(iter(blocks)),
            next(iter(orientations)),
        )
        chart_points = tuple(target_uv[int(vertex)] for vertex in cell)
        if len(chart_points) != 3:
            raise ValueError("Surface metric target cells must remain triangles.")
        chart: _ExactTriangle = (chart_points[0], chart_points[1], chart_points[2])
        area = _chart_area(chart)
        if area == 0 or (1 if area > 0 else -1) != 1:
            raise ValueError("A UV operation produced a nonpositive chart triangle.")
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
    for key, rows in group_records:
        physical_id = key[1] if key[0] == "source" else next_vertex_id
        if key[0] != "source":
            next_vertex_id += 1
        for row in rows:
            target_physical_id_by_work_row[row] = physical_id
        group_rows.append(tuple(rows))

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
    for cell, target_id_raw in zip(local_cells, local_cell_ids, strict=True):
        patch = cell_meta[int(target_id_raw)][0]
        for vertex in cell.tolist():
            previous = int(vertex_patches[vertex])
            if previous >= 0 and previous != patch:
                raise ValueError("One UV carrier vertex is shared by different source patches.")
            vertex_patches[vertex] = patch
    physical_coordinates = np.empty((physical_ids.size, 3), dtype=np.float64)
    target_metric = np.empty((physical_ids.size, 3, 3), dtype=np.float64)
    physical_stencils: list[tuple[tuple[int, Fraction], ...]] = []
    maximum_residual = 0.0
    for physical_row, copies in enumerate(group_rows):
        representative = copies[0]
        patch = int(vertex_patches[representative])
        if patch < 0:
            raise ValueError("A target source-chart vertex has no source patch occurrence.")
        point = _source_chart_point(
            prepared.domain, patch, target_uv[representative]
        )
        physical_coordinates[physical_row] = point
        stencil = _target_source_vertex_stencil(
            local_edit, prepared, work_vertex_rows, representative
        )
        for copy_row in copies[1:]:
            copy_patch = int(vertex_patches[copy_row])
            copy_point = _source_chart_point(
                prepared.domain, copy_patch, target_uv[copy_row]
            )
            residual = float(np.linalg.norm(point - copy_point, ord=np.inf))
            maximum_residual = max(maximum_residual, residual)
            scale = max(1.0, float(np.linalg.norm(point, ord=np.inf)))
            if residual > 2.0e-10 * scale:
                raise ValueError(
                    "Periodic and chart-copy source evaluations disagree at one target vertex."
                )
            if (
                work_target.periodic_topology is not None
                and _target_source_vertex_stencil(
                    local_edit, prepared, work_vertex_rows, copy_row
                )
                != stencil
            ):
                raise ValueError(
                    "Synchronized source-chart copies have different physical interpolation ancestry."
                )
        physical_stencils.append(stencil)
        source_rows = np.asarray([source_row for source_row, _ in stencil], dtype=np.int64)
        weights = np.asarray([float(weight) for _, weight in stencil], dtype=np.float64)
        target_metric[physical_row] = np.asarray(
            interpolate_mesh_metric(
                prepared.physical_metric[source_rows][None, :, :],
                weights[None, :],
            )[0],
            dtype=np.float64,
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
        np.asarray(local_vertex_relations.source_keys, dtype=np.int64),
        np.asarray(local_vertex_relations.target_keys, dtype=np.int64),
        np.asarray(local_vertex_relations.kinds, dtype=np.int32),
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
        np.asarray([[row[0]] for row in vertex_relation_rows], dtype=np.int64).reshape((-1, 1)),
        np.asarray([[row[1]] for row in vertex_relation_rows], dtype=np.int64).reshape((-1, 1)),
        np.asarray([row[2] for row in vertex_relation_rows], dtype=np.int32),
        1,
    )

    edge_source_rows: list[tuple[int, int]] = []
    edge_target_rows: list[tuple[int, int]] = []
    edge_kind_rows: list[int] = []
    for source_key, target_key, kind in zip(
        np.asarray(local_edit.relations[1].source_keys, dtype=np.int64),
        np.asarray(local_edit.relations[1].target_keys, dtype=np.int64),
        np.asarray(local_edit.relations[1].kinds, dtype=np.int32),
        strict=True,
    ):
        source_key_mapped = tuple(
            sorted(source_work_to_physical_id[int(value)] for value in source_key if value >= 0)
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
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks)
    )
    reference_maps = _reference_maps_from_charts(
        prepared, target_cell_ids, chart_by_cell_id
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
    if isinstance(prepared.source_charts, _ImplicitSurfaceChartBank):
        witness, implicit_cell_keys, implicit_root_edge_keys = _implicit_target_witness(
            prepared,
            target_mesh,
            target_geometry,
            target_cell_ids,
            chart_by_cell_id,
        )
    else:
        witness = SurfaceChartWitness(
            geometry_id=cell_geometry_id(target_geometry),
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
                prepared.domain.entity_id(2, int(patch))
                for patch in patches.tolist()
            ),
            occurrence_paths=tuple(
                prepared.domain.source_occurrences[2][int(patch)]
                for patch in patches.tolist()
            ),
            root_mesh_cell_bindings=tuple(
                SurfaceRootMeshCellBinding(
                    cell_geometry_id(target_geometry),
                    target_mesh.topology_id,
                    int(identifier),
                )
                for identifier in target_cell_ids.tolist()
            ),
            chart_rational_tokens=tuple(
                tuple(
                    tuple((value.numerator, value.denominator) for value in point)
                    for point in chart
                )
                for chart in charts
            ),
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
        implicit_cell_keys,
        implicit_root_edge_keys,
        reference_maps,
    )
    return edit, target_metric, geometry


def execute_surface_metric_adaptation(
    source: SurfaceMetricSource,
    /,
    *,
    predicate_mode: object,
    topology_operations: bool,
    relocation: bool,
) -> SurfaceMetricOutcome:
    """Run split/collapse/flip/relocation on source UV with tensor pullback."""
    if not isinstance(source, SurfaceMetricSource):
        raise TypeError("source must be SurfaceMetricSource.")
    if not isinstance(topology_operations, bool) or not isinstance(relocation, bool):
        raise TypeError("topology_operations and relocation must be bool values.")
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
    )
    edit, target_metric, geometry = _physical_edit(source, local)
    evidence = local.evidence
    if evidence.converged:
        status = SurfaceMetricStatus.COMPLETE
    elif evidence.rejected_uncertain:
        status = SurfaceMetricStatus.NUMERICAL_STOP
    elif evidence.stalled:
        status = (
            SurfaceMetricStatus.STALLED
            if topology_operations
            else SurfaceMetricStatus.COMPLETE
        )
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
        raise ValueError("Surface metric chart transition changed its fixed source identity.")
    witness = geometry.target_chart_witness
    if not isinstance(witness, SurfaceChartWitness):
        raise TypeError("Implicit source charts require their source-bound witness route.")
    if (
        witness.topology_id != target_mesh.topology_id
        or witness.geometry_id != cell_geometry_id(target_geometry)
    ):
        raise ValueError("Surface metric target chart witness is stale for its geometry endpoint.")
    target_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks)
    )
    witness_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    witness_rows = {int(identifier): row for row, identifier in enumerate(witness_ids.tolist())}
    if target_ids.size != witness_ids.size or set(target_ids.tolist()) != set(witness_rows):
        raise ValueError("Surface metric chart witness does not cover every target cell.")
    rows = np.asarray([witness_rows[int(identifier)] for identifier in target_ids], dtype=np.int64)
    exact_charts = tuple(witness.exact_chart(int(row)) for row in rows.tolist())
    return SurfaceSourceCharts(
        source_charts.domain,
        target_mesh,
        target_geometry,
        source_charts.coordinate_contract,
        np.asarray(witness.charts, dtype=np.float64)[rows],
        np.asarray(witness.patches, dtype=np.int64)[rows],
        tuple(associations),
        source_charts.boundary_source,
        source_root_cell_ids=np.asarray(witness.source_root_cell_ids, dtype=np.int64)[rows],
        source_piece_ids=tuple(witness.source_piece_ids[int(row)] for row in rows.tolist()),
        root_parameter_exact=exact_charts,
    )


__all__ = [
    "SurfaceMetricGeometryWitness",
    "SurfaceMetricOutcome",
    "SurfaceMetricSource",
    "SurfaceMetricStatus",
    "execute_surface_metric_adaptation",
    "prepare_implicit_surface_metric_source",
    "prepare_surface_metric_source",
    "target_surface_source",
]
