#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path
from typing import Any

import build123d as bd
import numpy as np
from OCP.TopoDS import TopoDS_Shape

from ...geometry import BRep, BRepSource, model_from_occt_shape
from ...geometry.simplicial import mesh_region_from_source
from .._geometry import GeometryDomain


_BREP_SUFFIXES = frozenset({".brep", ".brp", ".iges", ".igs", ".step", ".stp"})


def _recentered_brep_source(source: BRepSource, recenter: bool):
    if not recenter:
        return source
    vertices = np.asarray(source.model.mesh_vertices)
    center = 0.5 * (np.min(vertices, axis=0) + np.max(vertices, axis=0))
    return source.translated(-center)


def Geometry3DFromCAD(
    mesh: Any,
    *,
    recenter: bool = True,
    cleanup_path: Path | None = None,
) -> GeometryDomain:
    """Adapt direct B-Rep CAD or a watertight triangle mesh to a 3D domain."""

    if cleanup_path is not None:
        raise ValueError(
            "cleanup_path was removed; callers own the lifetime of their input files."
        )
    if isinstance(mesh, (str, Path)) and Path(mesh).suffix.lower() in _BREP_SUFFIXES:
        source = _recentered_brep_source(BRep(mesh), recenter)
    elif isinstance(mesh, TopoDS_Shape):
        source = _recentered_brep_source(
            BRepSource(model_from_occt_shape(mesh)), recenter
        )
    elif isinstance(mesh, bd.Shape):
        source = _recentered_brep_source(
            BRepSource(model_from_occt_shape(mesh.wrapped)), recenter
        )
    else:
        source = mesh_region_from_source(mesh, recenter=recenter)
    return GeometryDomain(source.compile())


__all__ = ["Geometry3DFromCAD"]
