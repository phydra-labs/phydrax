# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native source union, separated from implementations to avoid import cycles."""
from __future__ import annotations

from ...geometry._compartments import CompartmentMeshingSource
from ...geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding
from ._native_cad_volume import NativeCadVolumeSource
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
    NativeSurfaceEnvelopeSource,
    NativeSurfaceSource,
    NativeSweepSource,
)


type NativeMeshingSource = (
    NativeCadVolumeSource
    | NativeCurveSource
    | CompartmentMeshingSource
    | LabelFieldVolumeBinding
    | NativeImplicitSource
    | NativeLayerCoreSource
    | NativeMappedHexSource
    | NativePlanarSource
    | NativePeriodicSource
    | NativePlcSource
    | NativePolyhedralSource
    | NativeSurfaceSource
    | NativeStructuredSource
    | NativeSweepSource
    | NativeSurfaceEnvelopeSource
)

__all__ = ["NativeMeshingSource"]
