#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike
from scipy.spatial import Delaunay, QhullError

from ...measurement.lidar import LidarPointProduct
from .._capabilities import (
    ClosestPointProvider,
    ContactCurvatureProvider,
    SeamDiagnosticsProvider,
    SupportMapProvider,
)
from .._contracts import (
    ClosestPointResult,
    ContactCurvatureResult,
    GeometryKernel,
    GeometrySource,
)
from .._cubature import CubatureComponent
from .._validity import representation_validity
from ..design._schema import _ParameterCollector
from ..simplicial._io import (
    _canonical_triangle_arrays,
    planar_region_from_triangles,
)
from ..simplicial._regions import MeshRegion, PlanarMeshRegion
from ..simplicial._topology import TriangleTopology


@dataclass(frozen=True, slots=True)
class ReconstructionReport:
    """Provenance, filtering, topology, and approximation facts for reconstruction."""

    source_kind: str
    algorithm: str
    input_digest: str
    input_points: int
    retained_points: int
    output_vertices: int
    output_cells: int
    connected_components: int
    watertight: bool
    winding_consistent: bool
    recenter_offset: tuple[float, ...]
    parameters: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...] = ()
    source_product_id: str | None = None


@runtime_checkable
class ReconstructionReportProvider(Protocol):
    report: ReconstructionReport


class ReconstructionFailure(ValueError):
    """Reconstruction failure retaining the diagnostics produced before rejection."""

    report: ReconstructionReport

    def __init__(self, message: str, report: ReconstructionReport):
        super().__init__(message)
        self.report = report


class ReconstructedGeometrySource(GeometrySource):
    """Geometry source carrying an immutable reconstruction report."""

    source: GeometrySource
    report: ReconstructionReport = eqx.field(static=True)

    def __init__(self, source: GeometrySource, report: ReconstructionReport):
        if not isinstance(source, GeometrySource):
            raise TypeError("source must implement GeometrySource.")
        self.source = source
        self.report = report

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        return _ReconstructedGeometryKernel(self.source._compile(context), self.report)


class _ReconstructedGeometryKernel(GeometryKernel):
    child: GeometryKernel
    report: ReconstructionReport = eqx.field(static=True)

    def __init__(self, child: GeometryKernel, report: ReconstructionReport):
        self.child = child
        self.report = report

    @property
    def ambient_dimension(self):
        return self.child.ambient_dimension

    @property
    def intrinsic_dimension(self):
        return self.child.intrinsic_dimension

    @property
    def kind(self):
        return self.child.kind

    @property
    def capabilities(self):
        return self.child.capabilities

    @property
    def field_certificate(self):
        return self.child.field_certificate

    def geometry_validity(self, state, /):
        return representation_validity(self.child, state)

    def boundary_field(self, state, points, /):
        return self.child.boundary_field(state, points)

    def contains(self, state, points, /):
        return self.child.contains(state, points)

    def boundary_normal(self, state, points, /):
        return self.child.boundary_normal(state, points)

    def closest_point(self, state, points, /):
        if not isinstance(self.child, ClosestPointProvider):
            raise TypeError("Reconstructed child lacks a closest-point provider.")
        result = self.child.closest_point(state, points)
        if not isinstance(result, ClosestPointResult):
            raise TypeError("Child closest-point query returned an invalid result.")
        return result

    def contact_curvature(self, state, points, /):
        if not isinstance(self.child, ContactCurvatureProvider):
            raise TypeError("Reconstructed child lacks a contact-curvature provider.")
        result = self.child.contact_curvature(state, points)
        if not isinstance(result, ContactCurvatureResult):
            raise TypeError("Child contact-curvature query returned an invalid result.")
        return result

    def support_map(self, state, directions, /):
        if not isinstance(self.child, SupportMapProvider):
            raise TypeError("Reconstructed child lacks a support-map provider.")
        return self.child.support_map(state, directions)

    def bounds(self, state, /):
        return self.child.bounds(state)

    def measure(self, state, /):
        return self.child.measure(state)

    def boundary_measure(self, state, /):
        return self.child.boundary_measure(state)

    def interior_mass(self, state, /):
        return self.child.interior_mass(state)

    def boundary_mass(self, state, /):
        return self.child.boundary_mass(state)

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        return self.child.sample_interior(
            state,
            num_points,
            key=key,
            plan=plan,
        )

    def sample_boundary(self, state, num_points, /, *, key):
        return self.child.sample_boundary(state, num_points, key=key)

    def boundary_atlas(self, state, /):
        return self.child.boundary_atlas(state)

    def cubature_atlas(self, state, component: CubatureComponent, /):
        return self.child.cubature_atlas(state, component)

    def seam_residual(self, state, /):
        if not isinstance(self.child, SeamDiagnosticsProvider):
            raise TypeError("Reconstructed child lacks a seam-diagnostics provider.")
        return self.child.seam_residual(state)


def _point_digest(points: np.ndarray) -> str:
    canonical = np.ascontiguousarray(points, dtype=np.float64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _validated_points(points: ArrayLike, dimension: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < dimension:
        raise ValueError(f"points must have shape (num_points, >= {dimension}).")
    values = values[:, :dimension]
    if values.shape[0] < dimension + 1:
        raise ValueError("The point cloud has too few points for reconstruction.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Reconstruction points must all be finite.")
    return values


def _recenter(points: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    offset = (
        0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
        if enabled
        else np.zeros((points.shape[1],), dtype=np.float64)
    )
    return points - offset, offset


def _require_pyvista():
    try:
        import pyvista
    except ImportError as error:
        raise ImportError(
            "Implicit point-cloud reconstruction requires the optional "
            "'geometry-pyvista' dependency group."
        ) from error
    return pyvista


def _planar_triangulation(
    points: np.ndarray,
    /,
    *,
    tolerance: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = max(float(np.max(np.ptp(points, axis=0))), 1.0)
    if tolerance > 0.0:
        quantized = np.rint(points / (tolerance * scale)).astype(np.int64)
        _, retained = np.unique(quantized, axis=0, return_index=True)
    else:
        _, retained = np.unique(points, axis=0, return_index=True)
    retained = np.sort(retained)
    vertices = points[retained]
    if vertices.shape[0] < 3:
        raise ValueError("Planar reconstruction retained fewer than three points.")
    try:
        faces = np.asarray(Delaunay(vertices).simplices, dtype=np.int32)
    except QhullError as error:
        raise ValueError("Planar Delaunay triangulation failed.") from error
    triangles = vertices[faces]
    doubled_area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (
        triangles[:, 2, 0] - triangles[:, 0, 0]
    )
    negative = doubled_area < 0.0
    faces[negative] = faces[negative][:, [0, 2, 1]]
    doubled_area = np.abs(doubled_area)
    if alpha > 0.0:
        first = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
        second = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
        third = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
        radius = first * second * third / np.maximum(2.0 * doubled_area, 1.0e-300)
        faces = faces[radius <= alpha]
    if faces.shape[0] == 0:
        raise ValueError("Planar reconstruction produced no retained triangles.")
    return vertices, faces, retained


def _polydata_triangles(polydata: Any) -> tuple[np.ndarray, np.ndarray]:
    surface = polydata.triangulate()
    vertices = np.asarray(surface.points, dtype=np.float64)
    packed = np.asarray(surface.faces, dtype=np.int64)
    if packed.size == 0 or packed.size % 4 != 0:
        raise ValueError("Reconstruction produced no triangular cells.")
    records = packed.reshape((-1, 4))
    if np.any(records[:, 0] != 3):
        raise ValueError("Triangulated PolyData contains a non-triangle cell.")
    return vertices, records[:, 1:].astype(np.int32)


def _clean_surface_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, TriangleTopology]:
    vertices_, faces_ = _canonical_triangle_arrays(vertices, faces)
    topology = TriangleTopology(faces_, num_vertices=vertices_.shape[0])
    return vertices_, faces_, topology


def _parameter_records(**parameters) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((name, repr(value)) for name, value in parameters.items()))


def reconstruct_planar_region(
    points: ArrayLike,
    *,
    recenter: bool = True,
    alpha: float = 0.0,
    tolerance: float = 1e-5,
    offset: float = 1.0,
    bound: bool = False,
    progress_bar: bool = False,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Reconstruct one planar region and report every host-side approximation."""

    points_ = _validated_points(points, 2)
    if alpha < 0.0 or tolerance < 0.0 or offset <= 0.0:
        raise ValueError("alpha/tolerance must be non-negative and offset positive.")
    vertices_2d, faces, _ = _planar_triangulation(
        points_,
        tolerance=float(tolerance),
        alpha=float(alpha),
    )
    planar = planar_region_from_triangles(
        vertices_2d,
        faces,
        recenter=False,
        feature_id=feature_id,
    )
    planar_vertices = np.asarray(planar.vertices, dtype=np.float64)
    vertices, center = _recenter(planar_vertices, recenter)
    offsets = np.asarray(planar.loop_offsets, dtype=np.int32)
    loops = tuple(
        np.arange(offsets[index], offsets[index + 1], dtype=np.int32)
        for index in range(offsets.shape[0] - 1)
    )
    source = PlanarMeshRegion(vertices, loops, feature_id=feature_id)
    algorithm = "scipy_delaunay_2d_native_boundary"
    parameters = _parameter_records(
        alpha=float(alpha),
        tolerance=float(tolerance),
        offset=float(offset),
        bound=bool(bound),
    )
    report = ReconstructionReport(
        source_kind="planar_point_cloud",
        algorithm=algorithm,
        input_digest=_point_digest(points_),
        input_points=points_.shape[0],
        retained_points=points_.shape[0],
        output_vertices=vertices.shape[0],
        output_cells=faces.shape[0],
        connected_components=1,
        watertight=True,
        winding_consistent=True,
        recenter_offset=tuple(float(value) for value in center),
        parameters=parameters,
    )
    return ReconstructedGeometrySource(source, report)


def _surface_source(
    points: np.ndarray,
    surface: Any,
    *,
    source_kind: str,
    algorithm: str,
    recenter: bool,
    parameters: tuple[tuple[str, str], ...],
    input_points: int,
    warnings: Sequence[str] = (),
    feature_id: str | None = None,
    source_product_id: str | None = None,
) -> ReconstructedGeometrySource:
    vertices, faces = (
        surface if isinstance(surface, tuple) else _polydata_triangles(surface)
    )
    try:
        vertices_clean, faces_clean, topology = _clean_surface_mesh(vertices, faces)
        winding_consistent = True
    except ValueError as error:
        report = ReconstructionReport(
            source_kind=source_kind,
            algorithm=algorithm,
            input_digest=_point_digest(points),
            input_points=input_points,
            retained_points=points.shape[0],
            output_vertices=vertices.shape[0],
            output_cells=faces.shape[0],
            connected_components=0,
            watertight=False,
            winding_consistent=False,
            recenter_offset=(0.0, 0.0, 0.0),
            parameters=parameters,
            warnings=(*warnings, str(error)),
            source_product_id=source_product_id,
        )
        raise ReconstructionFailure(
            "Surface reconstruction produced invalid triangle topology.", report
        ) from error
    vertices_clean, center = _recenter(vertices_clean, recenter)
    report = ReconstructionReport(
        source_kind=source_kind,
        algorithm=algorithm,
        input_digest=_point_digest(points),
        input_points=input_points,
        retained_points=points.shape[0],
        output_vertices=vertices_clean.shape[0],
        output_cells=faces_clean.shape[0],
        connected_components=topology.num_face_components,
        watertight=topology.watertight,
        winding_consistent=winding_consistent,
        recenter_offset=tuple(float(value) for value in center),
        parameters=parameters,
        warnings=tuple(warnings),
        source_product_id=source_product_id,
    )
    if not report.watertight or not report.winding_consistent:
        raise ReconstructionFailure(
            "Surface reconstruction did not produce a watertight consistently wound solid.",
            report,
        )
    source = MeshRegion(vertices_clean, faces_clean, feature_id=feature_id)
    return ReconstructedGeometrySource(source, report)


def reconstruct_surface_region(
    points: ArrayLike,
    *,
    recenter: bool = True,
    neighborhood_size: int | None = None,
    sample_spacing: float | None = None,
    progress_bar: bool = False,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Reconstruct a watertight surface point cloud through a reported implicit fit."""

    points_ = _validated_points(points, 3)
    if neighborhood_size is not None and neighborhood_size <= 0:
        raise ValueError("neighborhood_size must be positive when provided.")
    if sample_spacing is not None and sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be positive when provided.")
    pyvista = _require_pyvista()
    surface = pyvista.PolyData(points_).reconstruct_surface(
        nbr_sz=neighborhood_size,
        sample_spacing=sample_spacing,
        progress_bar=bool(progress_bar),
    )
    return _surface_source(
        points_,
        surface,
        source_kind="surface_point_cloud",
        algorithm="pyvista_implicit_surface",
        recenter=recenter,
        parameters=_parameter_records(
            neighborhood_size=neighborhood_size,
            sample_spacing=sample_spacing,
        ),
        input_points=points_.shape[0],
        feature_id=feature_id,
    )


def _terrain_points(
    points_or_grid: ArrayLike,
    *,
    x: ArrayLike | None,
    y: ArrayLike | None,
) -> np.ndarray:
    values = np.asarray(points_or_grid, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] != 3:
        rows, columns = values.shape
        x_values = (
            np.arange(columns, dtype=np.float64)
            if x is None
            else np.asarray(x, dtype=np.float64)
        )
        y_values = (
            np.arange(rows, dtype=np.float64)
            if y is None
            else np.asarray(y, dtype=np.float64)
        )
        if x_values.shape != (columns,) or y_values.shape != (rows,):
            raise ValueError("x and y coordinate vectors must match the height grid.")
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        points = np.column_stack((x_grid.ravel(), y_grid.ravel(), values.ravel()))
    else:
        points = _validated_points(values, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError("Terrain samples must all be finite.")
    return points


def reconstruct_dem_region(
    points_or_grid: ArrayLike,
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    recenter: bool = True,
    alpha: float = 0.0,
    tolerance: float = 1e-5,
    bound: bool = False,
    extrude_depth: float = 1.0,
    progress_bar: bool = False,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Triangulate a terrain and cap a downward extrusion as a reported solid."""

    points = _terrain_points(points_or_grid, x=x, y=y)
    if alpha < 0.0 or tolerance < 0.0 or extrude_depth <= 0.0:
        raise ValueError(
            "alpha/tolerance must be non-negative and extrude_depth positive."
        )
    planar_vertices, top_faces, retained_indices = _planar_triangulation(
        points[:, :2],
        tolerance=float(tolerance),
        alpha=float(alpha),
    )
    top_vertices = points[retained_indices].copy()
    top_vertices[:, :2] = planar_vertices
    topology = TriangleTopology(top_faces, num_vertices=top_vertices.shape[0])
    boundary = np.asarray(topology.boundary_halfedges, dtype=np.int32)
    origins = np.asarray(topology.halfedge_origin, dtype=np.int32)[boundary]
    destinations = np.asarray(topology.halfedge_destination, dtype=np.int32)[boundary]
    count = top_vertices.shape[0]
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 2] -= float(extrude_depth)
    side_first = np.stack((origins + count, destinations + count, destinations), axis=1)
    side_second = np.stack((origins + count, destinations, origins), axis=1)
    solid_vertices = np.concatenate((top_vertices, bottom_vertices), axis=0)
    solid_faces = np.concatenate(
        (
            top_faces,
            top_faces[:, [0, 2, 1]] + count,
            side_first,
            side_second,
        ),
        axis=0,
    )
    return _surface_source(
        points,
        (solid_vertices, solid_faces),
        source_kind="digital_elevation_model",
        algorithm="native_delaunay_2d_capped_extrusion",
        recenter=recenter,
        parameters=_parameter_records(
            alpha=float(alpha),
            tolerance=float(tolerance),
            bound=bool(bound),
            extrude_depth=float(extrude_depth),
        ),
        input_points=points.shape[0],
        feature_id=feature_id,
    )


def reconstruct_point_region(
    points: ArrayLike,
    *,
    recenter: bool = True,
    roi: tuple[float, float, float, float, float, float] | None = None,
    voxel_size: float | None = None,
    neighborhood_size: int | None = None,
    sample_spacing: float | None = None,
    progress_bar: bool = False,
    feature_id: str | None = None,
    source_product_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Crop/downsample Cartesian points, then run a reported implicit surface fit."""

    original = _validated_points(points, 3)
    retained = original
    warnings: list[str] = []
    if roi is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = map(float, roi)
        if not (x_min < x_max and y_min < y_max and z_min < z_max):
            raise ValueError("roi minima must be strictly below maxima.")
        mask = (
            (retained[:, 0] >= x_min)
            & (retained[:, 0] <= x_max)
            & (retained[:, 1] >= y_min)
            & (retained[:, 1] <= y_max)
            & (retained[:, 2] >= z_min)
            & (retained[:, 2] <= z_max)
        )
        retained = retained[mask]
    if voxel_size is not None:
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive when provided.")
        voxel = np.floor(retained / float(voxel_size)).astype(np.int64)
        _, indices = np.unique(voxel, axis=0, return_index=True)
        retained = retained[np.sort(indices)]
    if retained.shape[0] < 4:
        raise ValueError("LiDAR filtering retained too few points for reconstruction.")
    if retained.shape[0] < original.shape[0] / 10:
        warnings.append("Filtering retained fewer than ten percent of input points.")
    if neighborhood_size is not None and neighborhood_size <= 0:
        raise ValueError("neighborhood_size must be positive when provided.")
    if sample_spacing is not None and sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be positive when provided.")
    pyvista = _require_pyvista()
    surface = pyvista.PolyData(retained).reconstruct_surface(
        nbr_sz=neighborhood_size,
        sample_spacing=sample_spacing,
        progress_bar=bool(progress_bar),
    )
    return _surface_source(
        retained,
        surface,
        source_kind="point_cloud",
        algorithm="voxel_filter_then_pyvista_implicit_surface",
        recenter=recenter,
        parameters=_parameter_records(
            roi=roi,
            voxel_size=voxel_size,
            neighborhood_size=neighborhood_size,
            sample_spacing=sample_spacing,
        ),
        input_points=original.shape[0],
        warnings=warnings,
        feature_id=feature_id,
        source_product_id=source_product_id,
    )


def reconstruct_lidar_region(
    product: LidarPointProduct,
    *,
    recenter: bool = True,
    roi: tuple[float, float, float, float, float, float] | None = None,
    voxel_size: float | None = None,
    neighborhood_size: int | None = None,
    sample_spacing: float | None = None,
    progress_bar: bool = False,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Reconstruct valid derived LiDAR points while retaining acquisition lineage."""
    if not isinstance(product, LidarPointProduct):
        raise TypeError("product must be LidarPointProduct.")
    active = np.asarray(product.support.active_mask, dtype=np.bool_)
    return reconstruct_point_region(
        np.asarray(product.support.points)[active],
        recenter=recenter,
        roi=roi,
        voxel_size=voxel_size,
        neighborhood_size=neighborhood_size,
        sample_spacing=sample_spacing,
        progress_bar=progress_bar,
        feature_id=feature_id,
        source_product_id=product.point_product_id,
    )


__all__ = [
    "ReconstructedGeometrySource",
    "ReconstructionReportProvider",
    "ReconstructionFailure",
    "ReconstructionReport",
    "reconstruct_dem_region",
    "reconstruct_lidar_region",
    "reconstruct_point_region",
    "reconstruct_planar_region",
    "reconstruct_surface_region",
]
