#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

from jaxtyping import ArrayLike

from ...geometry import reconstruct_planar_region
from ...geometry.simplicial import planar_region_from_source
from .._geometry import GeometryDomain


def Geometry2DFromCAD(
    mesh: Any,
    *,
    recenter: bool = True,
) -> GeometryDomain:
    """Adapt a triangulated planar mesh to the canonical geometry domain."""

    source = planar_region_from_source(mesh, recenter=recenter)
    return GeometryDomain(source.compile())


def Geometry2DFromPointCloud(
    points: ArrayLike,
    *,
    recenter: bool = True,
    alpha: float = 0.0,
    tol: float = 1e-5,
    offset: float = 1.0,
    bound: bool = False,
    progress_bar: bool = False,
) -> GeometryDomain:
    """Reconstruct a reported planar region from boundary samples."""

    source = reconstruct_planar_region(
        points,
        recenter=recenter,
        alpha=alpha,
        tolerance=tol,
        offset=offset,
        bound=bound,
        progress_bar=progress_bar,
    )
    return GeometryDomain(source.compile())


__all__ = ["Geometry2DFromCAD", "Geometry2DFromPointCloud"]
