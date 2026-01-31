#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import inspect
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyvista as pv
from jaxtyping import ArrayLike

from ._mesh import Geometry3DFromCAD


def Geometry3DFromPointCloud(
    points: ArrayLike,
    *,
    recenter: bool = True,
    nbr_sz: int | None = None,
    radius: float | None = None,
    sample_spacing: float | None = None,
    progress_bar: bool = False,
) -> Geometry3DFromCAD:
    r"""Reconstruct a 3D mesh geometry from a surface point cloud.

    Interprets `points` as samples from (or near) the surface $\partial\Omega$ of an
    unknown solid $\Omega\subset\mathbb{R}^3$. A triangulated surface is reconstructed
    and used to build a `Geometry3DFromCAD`.

    **Arguments:**

    - `points`: Array-like of shape `(N, 3)` (or `(N, >=3)`, where only the first three
      coordinates are used).
    - `recenter`: Whether to recenter the reconstructed mesh coordinates.
    - `nbr_sz`: Optional PyVista `reconstruct_surface` control (passed through when supported).
    - `radius`: Optional PyVista `reconstruct_surface` control (passed through when supported).
    - `sample_spacing`: Optional PyVista `reconstruct_surface` control (passed through when supported).
    - `progress_bar`: Optional PyVista `reconstruct_surface` control (passed through when supported).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, 3)")
    if pts.shape[1] > 3:
        pts = pts[:, :3]

    cloud = pv.PolyData(pts)
    surface = cloud.reconstruct_surface(
        nbr_sz=int(nbr_sz) if nbr_sz is not None else None,
        sample_spacing=float(sample_spacing) if sample_spacing is not None else None,
        progress_bar=bool(progress_bar),
    )

    tmp_path = Path(f"/tmp/pv_pointcloud_{uuid4().hex}.stl").resolve()
    surface.save(tmp_path)

    geom = Geometry3DFromCAD(tmp_path, recenter=recenter)
    tmp_path.unlink(missing_ok=True)
    return geom


def Geometry3DFromDEM(
    points_or_grid: ArrayLike,
    *,
    recenter: bool = True,
    alpha: float | None = None,
    tol: float | None = None,
    bound: float | str | None = None,
    progress_bar: bool = False,
    extrude_depth: float = 1.0,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
) -> Geometry3DFromCAD:
    r"""Create a (roughly) watertight 3D geometry from a 2.5D DEM.

    The input is interpreted as a height field $z=z(x,y)$ sampled on a grid or as
    scattered $(x,y,z)$ points. A triangulated surface is constructed and (when
    supported) extruded downward to produce a closed solid suitable for sampling.

    **Arguments:**

    - `points_or_grid`: Either a grid of shape `(ny, nx)` with heights, or an array of
      points of shape `(N, 3)` (or `(N, >=3)`).
    - `recenter`: Whether to recenter the reconstructed mesh coordinates.
    - `alpha`: Optional PyVista `delaunay_2d` control (passed through when supported).
    - `tol`: Optional PyVista `delaunay_2d` control (passed through when supported).
    - `bound`: Optional PyVista `delaunay_2d` control (passed through when supported).
    - `progress_bar`: Optional PyVista `delaunay_2d` control (passed through when supported).
    - `extrude_depth`: Extrusion depth used to "close" the surface.
    - `x`: Optional coordinate vector for the height grid.
    - `y`: Optional coordinate vector for the height grid.
    """
    arr = np.asarray(points_or_grid, dtype=float)
    if arr.ndim == 2:
        ny, nx = arr.shape
        xs = np.asarray(x) if x is not None else np.arange(nx, dtype=float)
        ys = np.asarray(y) if y is not None else np.arange(ny, dtype=float)
        assert xs.ndim == 1 and ys.ndim == 1
        assert xs.size == nx and ys.size == ny
        X, Y = np.meshgrid(xs, ys)
        pts = np.column_stack([X.ravel(), Y.ravel(), arr.ravel()])
    else:
        pts = np.asarray(points_or_grid, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("points_or_grid must be (ny,nx) grid or (N,3) points")
        if pts.shape[1] > 3:
            pts = pts[:, :3]

    cloud = pv.PolyData(pts)
    tol_value = 1e-5 if tol is None else float(tol)
    alpha_value = 0.0 if alpha is None else float(alpha)
    bound_value = False if bound is None else bool(bound)

    surface = cloud.delaunay_2d(
        tol=tol_value,
        alpha=alpha_value,
        bound=bound_value,
        progress_bar=bool(progress_bar),
    )
    if hasattr(surface, "triangulate"):
        surface = surface.triangulate()

    if hasattr(surface, "extrude"):
        try:
            sig_ex = inspect.signature(surface.extrude)
            if "capping" in sig_ex.parameters:
                solid = surface.extrude((0.0, 0.0, -float(extrude_depth)), capping=True)
            else:
                solid = surface.extrude((0.0, 0.0, -float(extrude_depth)))
        except Exception:
            solid = surface
    else:
        solid = surface

    tmp_path = Path(f"/tmp/pv_dem_{uuid4().hex}.stl").resolve()
    solid.save(tmp_path)

    geom = Geometry3DFromCAD(tmp_path, recenter=recenter)
    tmp_path.unlink(missing_ok=True)
    return geom


def Geometry3DFromLidarScene(
    points: ArrayLike,
    *,
    recenter: bool = True,
    roi: tuple[float, float, float, float, float, float] | None = None,
    voxel_size: float | None = None,
    nbr_sz: int | None = None,
    radius: float | None = None,
    sample_spacing: float | None = None,
    progress_bar: bool = False,
    close_depth: float = 0.5,
) -> Geometry3DFromCAD:
    r"""Construct a mesh geometry from LiDAR scene points.

    This helper optionally:

    - crops the point cloud to a region of interest (ROI),
    - voxel-downsamples the points,
    - reconstructs a surface $\partial\Omega$,
    - extrudes it to obtain a closed solid.

    **Arguments:**

    - `points`: Array-like of shape `(N, 3)` (or `(N, >=3)`).
    - `recenter`: Whether to recenter the reconstructed mesh coordinates.
    - `roi`: Optional axis-aligned ROI `(xmin, xmax, ymin, ymax, zmin, zmax)`.
    - `voxel_size`: Optional voxel downsampling size.
    - `nbr_sz`: Optional PyVista `reconstruct_surface` control (passed through when supported).
    - `radius`: Optional PyVista `reconstruct_surface` control (passed through when supported).
    - `sample_spacing`: Optional PyVista `reconstruct_surface` control (passed through when supported).
    - `progress_bar`: Optional PyVista `reconstruct_surface` control (passed through when supported).
    - `close_depth`: Extrusion depth used to close the reconstructed surface.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, 3)")
    if pts.shape[1] > 3:
        pts = pts[:, :3]

    if roi is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = map(float, roi)
        mask = (
            (pts[:, 0] >= xmin)
            & (pts[:, 0] <= xmax)
            & (pts[:, 1] >= ymin)
            & (pts[:, 1] <= ymax)
            & (pts[:, 2] >= zmin)
            & (pts[:, 2] <= zmax)
        )
        pts = pts[mask]

    if voxel_size and voxel_size > 0:
        v = float(voxel_size)
        q = np.floor(pts / v).astype(np.int64)
        _, idx = np.unique(q, axis=0, return_index=True)
        pts = pts[np.sort(idx)]

    cloud = pv.PolyData(pts)
    surf = cloud.reconstruct_surface(
        nbr_sz=int(nbr_sz) if nbr_sz is not None else None,
        sample_spacing=float(sample_spacing) if sample_spacing is not None else None,
        progress_bar=bool(progress_bar),
    )
    if hasattr(surf, "triangulate"):
        surf = surf.triangulate()

    if hasattr(surf, "extrude"):
        try:
            sig_ex = inspect.signature(surf.extrude)
            if "capping" in sig_ex.parameters:
                solid = surf.extrude((0.0, 0.0, -float(close_depth)), capping=True)
            else:
                solid = surf.extrude((0.0, 0.0, -float(close_depth)))
        except Exception:
            solid = surf
    else:
        solid = surf

    tmp_path = Path(f"/tmp/pv_lidar_{uuid4().hex}.stl").resolve()
    solid.save(tmp_path)

    geom = Geometry3DFromCAD(tmp_path, recenter=recenter)
    tmp_path.unlink(missing_ok=True)
    return geom


__all__ = [
    "Geometry3DFromDEM",
    "Geometry3DFromLidarScene",
    "Geometry3DFromPointCloud",
]
