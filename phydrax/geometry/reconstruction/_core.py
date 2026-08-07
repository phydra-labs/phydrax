#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import equinox as eqx
import numpy as np
import pyvista as pv
import trimesh
from jaxtyping import ArrayLike
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .._contracts import GeometryKernel, GeometrySource
from ..design._schema import _ParameterCollector
from ..simplicial import MeshRegion, PlanarMeshRegion


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

    def boundary_field(self, state, points, /):
        return self.child.boundary_field(state, points)

    def contains(self, state, points, /):
        return self.child.contains(state, points)

    def boundary_normal(self, state, points, /):
        return self.child.boundary_normal(state, points)

    def bounds(self, state, /):
        return self.child.bounds(state)

    def measure(self, state, /):
        return self.child.measure(state)

    def boundary_measure(self, state, /):
        return self.child.boundary_measure(state)

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


def _point_digest(points: np.ndarray) -> str:
    canonical = np.ascontiguousarray(points, dtype=np.float64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _validated_points(points: ArrayLike, dimension: int) -> np.ndarray:
    values = np.asarray(points, dtype=float)
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
        else np.zeros((points.shape[1],), dtype=float)
    )
    return points - offset, offset


def _polydata_triangles(polydata: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    surface = polydata.triangulate()
    vertices = np.asarray(surface.points, dtype=float)
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
) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
    mesh.update_faces(mesh.unique_faces() & mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals(multibody=True)
    if not mesh.is_watertight:
        mesh.fill_holes()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals(multibody=True)
    return mesh


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
    embedded = np.column_stack((points_, np.zeros((points_.shape[0],), dtype=float)))
    surface = pv.PolyData(embedded).delaunay_2d(
        tol=float(tolerance),
        alpha=float(alpha),
        offset=float(offset),
        bound=bool(bound),
        progress_bar=bool(progress_bar),
    )
    vertices_3d, faces = _polydata_triangles(surface)
    triangle_polygons = []
    for face in faces:
        polygon = ShapelyPolygon(vertices_3d[face, :2])
        if polygon.area > 0.0:
            triangle_polygons.append(polygon)
    region = unary_union(triangle_polygons)
    algorithm = "pyvista_delaunay_2d_union"
    parameters = _parameter_records(
        alpha=float(alpha),
        tolerance=float(tolerance),
        offset=float(offset),
        bound=bool(bound),
    )
    if region.geom_type != "Polygon":
        components = len(region.geoms) if region.geom_type == "MultiPolygon" else 0
        report = ReconstructionReport(
            source_kind="planar_point_cloud",
            algorithm=algorithm,
            input_digest=_point_digest(points_),
            input_points=points_.shape[0],
            retained_points=points_.shape[0],
            output_vertices=vertices_3d.shape[0],
            output_cells=faces.shape[0],
            connected_components=components,
            watertight=False,
            winding_consistent=False,
            recenter_offset=(0.0, 0.0),
            parameters=parameters,
            warnings=("Triangulation did not produce one connected planar polygon.",),
        )
        raise ReconstructionFailure(
            "Planar reconstruction must produce one connected polygon.", report
        )
    region = orient(region, sign=1.0)
    loops_host = [np.asarray(region.exterior.coords[:-1], dtype=float)]
    loops_host.extend(
        np.asarray(interior.coords[:-1], dtype=float) for interior in region.interiors
    )
    vertices = np.concatenate(loops_host, axis=0)
    vertices, center = _recenter(vertices, recenter)
    loops: list[np.ndarray] = []
    cursor = 0
    for loop_points in loops_host:
        loop = np.arange(cursor, cursor + loop_points.shape[0], dtype=np.int32)
        loops.append(loop)
        cursor += loop_points.shape[0]
    source = PlanarMeshRegion(vertices, loops, feature_id=feature_id)
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
    surface: pv.PolyData,
    *,
    source_kind: str,
    algorithm: str,
    recenter: bool,
    parameters: tuple[tuple[str, str], ...],
    input_points: int,
    warnings: Sequence[str] = (),
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    vertices, faces = _polydata_triangles(surface)
    mesh = _clean_surface_mesh(vertices, faces)
    components = len(mesh.split(only_watertight=False))
    vertices_clean = np.asarray(mesh.vertices, dtype=float)
    faces_clean = np.asarray(mesh.faces, dtype=np.int32)
    vertices_clean, center = _recenter(vertices_clean, recenter)
    report = ReconstructionReport(
        source_kind=source_kind,
        algorithm=algorithm,
        input_digest=_point_digest(points),
        input_points=input_points,
        retained_points=points.shape[0],
        output_vertices=vertices_clean.shape[0],
        output_cells=faces_clean.shape[0],
        connected_components=components,
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        recenter_offset=tuple(float(value) for value in center),
        parameters=parameters,
        warnings=tuple(warnings),
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
    surface = pv.PolyData(points_).reconstruct_surface(
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
    values = np.asarray(points_or_grid, dtype=float)
    if values.ndim == 2 and values.shape[1] != 3:
        rows, columns = values.shape
        x_values = (
            np.arange(columns, dtype=float) if x is None else np.asarray(x, dtype=float)
        )
        y_values = (
            np.arange(rows, dtype=float) if y is None else np.asarray(y, dtype=float)
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
    surface = pv.PolyData(points).delaunay_2d(
        tol=float(tolerance),
        alpha=float(alpha),
        bound=bool(bound),
        progress_bar=bool(progress_bar),
    )
    solid = surface.triangulate().extrude(
        (0.0, 0.0, -float(extrude_depth)),
        capping=True,
        progress_bar=bool(progress_bar),
    )
    return _surface_source(
        points,
        solid,
        source_kind="digital_elevation_model",
        algorithm="pyvista_delaunay_2d_capped_extrusion",
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


def reconstruct_lidar_region(
    points: ArrayLike,
    *,
    recenter: bool = True,
    roi: tuple[float, float, float, float, float, float] | None = None,
    voxel_size: float | None = None,
    neighborhood_size: int | None = None,
    sample_spacing: float | None = None,
    progress_bar: bool = False,
    feature_id: str | None = None,
) -> ReconstructedGeometrySource:
    """Crop/downsample LiDAR points, then run the reported implicit surface fit."""

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
    surface = pv.PolyData(retained).reconstruct_surface(
        nbr_sz=neighborhood_size,
        sample_spacing=sample_spacing,
        progress_bar=bool(progress_bar),
    )
    return _surface_source(
        retained,
        surface,
        source_kind="lidar_point_cloud",
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
    )


__all__ = [
    "ReconstructedGeometrySource",
    "ReconstructionReportProvider",
    "ReconstructionFailure",
    "ReconstructionReport",
    "reconstruct_dem_region",
    "reconstruct_lidar_region",
    "reconstruct_planar_region",
    "reconstruct_surface_region",
]
