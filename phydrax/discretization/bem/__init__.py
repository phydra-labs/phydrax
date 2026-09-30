"""Boundary-element trace complexes and conforming surface-current spaces."""

from ._bc_dual import (
    BuffaChristiansenDualEvidence3D,
    BuffaChristiansenDualSpace3D,
    prepare_buffa_christiansen_dual_3d,
)
from ._reconstruction import (
    prepare_boundary_form_query,
    PreparedBoundaryFormQuery,
    SurfaceFormReconstructionKernel,
)
from ._rwg import RWGSurfaceCurrentSpace3D, TangentialTracePairing3D
from ._surface_complex import OrientedTriangleSurfaceComplex3D, SurfaceTopologyReport3D


__all__ = [
    "BuffaChristiansenDualEvidence3D",
    "BuffaChristiansenDualSpace3D",
    "prepare_buffa_christiansen_dual_3d",
    "PreparedBoundaryFormQuery",
    "SurfaceFormReconstructionKernel",
    "prepare_boundary_form_query",
    "OrientedTriangleSurfaceComplex3D",
    "RWGSurfaceCurrentSpace3D",
    "SurfaceTopologyReport3D",
    "TangentialTracePairing3D",
]
