#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from jaxtyping import ArrayLike

from ...geometry import (
    reconstruct_dem_region,
    reconstruct_lidar_region,
    reconstruct_surface_region,
)
from .._geometry import GeometryDomain


def Geometry3DFromPointCloud(
    points: ArrayLike,
    *,
    recenter: bool = True,
    nbr_sz: int | None = None,
    sample_spacing: float | None = None,
    progress_bar: bool = False,
) -> GeometryDomain:
    """Reconstruct a reported watertight 3D region from surface samples."""

    source = reconstruct_surface_region(
        points,
        recenter=recenter,
        neighborhood_size=nbr_sz,
        sample_spacing=sample_spacing,
        progress_bar=progress_bar,
    )
    return GeometryDomain(source.compile())


def Geometry3DFromDEM(
    points_or_grid: ArrayLike,
    *,
    recenter: bool = True,
    alpha: float = 0.0,
    tol: float = 1e-5,
    bound: bool = False,
    progress_bar: bool = False,
    extrude_depth: float = 1.0,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
) -> GeometryDomain:
    """Reconstruct a reported capped solid from terrain or elevation samples."""

    source = reconstruct_dem_region(
        points_or_grid,
        x=x,
        y=y,
        recenter=recenter,
        alpha=alpha,
        tolerance=tol,
        bound=bound,
        extrude_depth=extrude_depth,
        progress_bar=progress_bar,
    )
    return GeometryDomain(source.compile())


def Geometry3DFromLidarScene(
    points: ArrayLike,
    *,
    recenter: bool = True,
    roi: tuple[float, float, float, float, float, float] | None = None,
    voxel_size: float | None = None,
    nbr_sz: int | None = None,
    sample_spacing: float | None = None,
    progress_bar: bool = False,
) -> GeometryDomain:
    """Crop/downsample LiDAR samples and reconstruct a reported 3D region."""

    source = reconstruct_lidar_region(
        points,
        recenter=recenter,
        roi=roi,
        voxel_size=voxel_size,
        neighborhood_size=nbr_sz,
        sample_spacing=sample_spacing,
        progress_bar=progress_bar,
    )
    return GeometryDomain(source.compile())


__all__ = [
    "Geometry3DFromDEM",
    "Geometry3DFromLidarScene",
    "Geometry3DFromPointCloud",
]
