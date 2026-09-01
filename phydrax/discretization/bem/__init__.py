"""Boundary-element trace complexes and conforming surface-current spaces."""

from ._rwg import RWGSurfaceCurrentSpace3D, TangentialTracePairing3D
from ._surface_complex import OrientedTriangleSurfaceComplex3D, SurfaceTopologyReport3D


__all__ = [
    "OrientedTriangleSurfaceComplex3D",
    "RWGSurfaceCurrentSpace3D",
    "SurfaceTopologyReport3D",
    "TangentialTracePairing3D",
]
