# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Source union for native route planning, without alternate source identities."""

from __future__ import annotations

from typing import TypeAlias

from ...geometry._compartments import CompartmentMeshingSource
from ...geometry._meshing_domain import MeshingDomain
from ...geometry.brep._model import BRepModel
from ...geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ...geometry.multiregion_surface._label_extraction import (
    LabelFieldSurfaceExtractionResult,
)
from ...geometry.reconstruction._core import ReconstructedGeometrySource
from ...geometry.surface._model import SurfaceModel
from .._surface_envelope import SurfaceEnvelopeSource
from ._native_periodic import NativePeriodicSource
from ._native_sources import (
    NativeCurveSource,
    NativeImplicitSource,
    NativeLayerCoreSource,
    NativeMappedHexSource,
    NativePlanarSource,
    NativePlcSource,
    NativePolyhedralSource,
    NativeStructuredSource,
    NativeSweepSource,
)


NativeMeshingSource: TypeAlias = (
    AnalyticImplicitProfile
    | BRepModel
    | MeshingDomain
    | CompartmentMeshingSource
    | NativeCurveSource
    | LabelFieldSurfaceExtractionResult
    | ReconstructedGeometrySource
    | NativeImplicitSource
    | NativeLayerCoreSource
    | NativeMappedHexSource
    | NativePlanarSource
    | NativePeriodicSource
    | NativePlcSource
    | NativePolyhedralSource
    | NativeStructuredSource
    | NativeSweepSource
    | SurfaceModel
    | SurfaceEnvelopeSource
)

__all__ = ["NativeMeshingSource"]
