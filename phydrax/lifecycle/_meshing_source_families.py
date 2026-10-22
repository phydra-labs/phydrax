#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Explicit native-family owners admitted by the scientific source archive.

This is a constructor/admission boundary, not an identity registry or serializer.
Execution worksets are rebuilt by their providers from these authored values.
"""

from __future__ import annotations

from typing import get_args, TypeGuard

import phydrax.meshing._periodic_constraint as periodic_constraint_owner

from ..discretization import _cell_complex, _cell_geometry, _hexahedral
from ..discretization._exact_algebraic_geometry import (
    ExactAlgebraicCellGeometrySource,
    RadicalElement,
)
from ..discretization._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ..discretization._exact_rational_geometry import ExactRationalCellGeometrySource
from ..discretization._periodic_geometry import PeriodicCellGeometrySource
from ..discretization._reference_cell_types import PolyhedralBlock
from ..geometry import _mapped_reference_domain, _source_fidelity as source_fidelity_owner
from ..geometry._compartments import CompartmentMeshingSource
from ..geometry._periodic_power_preparation import PeriodicPowerPreparation
from ..geometry.brep import _intersection
from ..geometry.surface import _contracts as _surface_contracts, _model as _surface_model
from ..meshing import (
    _boundary_layer,
    _contracts,
    _controls,
    _hex_generation,
    _periodic,
    _polyhedral_generation,
    _structured,
    _surface_envelope,
    _sweep,
    _volume_generation,
)
from ..meshing._plc_source import PiecewiseLinearComplex
from ..meshing.providers import _native_cad_volume, _native_periodic, _native_sources


type NativeGenerationSource = (
    _native_sources.NativePlanarSource
    | _native_sources.NativeImplicitSource
    | _native_sources.NativePlcSource
    | _native_cad_volume.NativeCadVolumeSource
    | _native_sources.NativeStructuredSource
    | _native_sources.NativeMappedHexSource
    | _native_sources.NativePolyhedralSource
    | _native_sources.NativeSurfaceSource
    | _native_sources.NativeSweepSource
    | _native_periodic.NativePeriodicSource
    | _native_sources.NativeLayerCoreSource
    | _native_sources.NativeSurfaceEnvelopeSource
    | CompartmentMeshingSource
)


def native_generation_source_types() -> tuple[type, ...]:
    """Return the declared native source owners."""
    return get_args(NativeGenerationSource.__value__)


def is_native_generation_source(value: object, /) -> TypeGuard[NativeGenerationSource]:
    "Admit only closed native authored-source owners."
    return type(value) in native_generation_source_types()


def native_family_artifact_types() -> tuple[tuple[str, tuple[type, ...]], ...]:
    """Finite, reviewed owner list; artifact registration remains in _artifacts."""
    return (
        (
            "phydrax.meshing",
            (
                _native_sources.NativePlcSource,
                _native_cad_volume.NativeCadVolumeSource,
                _native_sources.NativeStructuredSource,
                _native_sources.NativeMappedHexSource,
                _native_sources.NativePolyhedralSource,
                _native_sources.NativeSurfaceSource,
                PiecewiseLinearComplex,
                _volume_generation.NativeVolumeSchedule,
                _polyhedral_generation.NativePolyhedralSchedule,
                _hex_generation.NativeHexGridSchedule,
                _structured.TransfiniteBlock,
                _contracts.VolumeMeshingSpec,
                _contracts.VolumeFillStrategy,
                _controls.RegionSeed,
                _controls.HoleSeed,
                _controls.TransfiniteCurveControl,
                _controls.TransfiniteSurfaceControl,
                _controls.BlockInterfaceControl,
                _native_periodic.NativePeriodicSource,
                _periodic.PeriodicPointOrbits,
                _native_periodic.PeriodicAssociationTransfer,
                _native_sources.NativeSweepSource,
                _sweep.SweepControl,
                _sweep.SweepMapKind,
                _controls.LayerSchedule,
                _native_sources.NativeLayerCoreSource,
                _boundary_layer.BoundaryLayerMesh,
                _boundary_layer.BoundaryLayerEvidence,
                _boundary_layer.BoundaryLayerPolicy,
                _controls.BoundaryLayerControl,
                _controls.BoundaryLayerCollisionPolicy,
                _controls.BoundaryLayerCornerPolicy,
                _controls.BoundaryLayerRoute,
                periodic_constraint_owner.PeriodicConstraint,
                _native_sources.NativeSurfaceEnvelopeSource,
                _surface_envelope.RawTriangleSoup,
                _surface_envelope.SurfaceEnvelopePolicy,
                _surface_envelope.SurfaceEnvelope,
                _surface_envelope.SurfaceEnvelopeEvidence,
                _surface_envelope.EnvelopeTopology,
                _surface_envelope.EnvelopeFeaturePermission,
                _surface_envelope.EnvelopeTopologyPermission,
            ),
        ),
        (
            "phydrax.discretization",
            (
                PolyhedralBlock,
                _cell_complex.PolyhedralConnectivity,
                _hexahedral.HexahedralConnectivity,
                _cell_geometry.RestrictedCellGeometryElement,
                _cell_geometry.PolynomialComposedCellGeometryElement,
                _cell_geometry.RationalComposedCellGeometryElement,
                _cell_geometry.BarycentricCellGeometryElement,
                _cell_geometry.SplineCellGeometryElement,
                _cell_geometry.LayerColumnCellGeometryElement,
                _cell_geometry.CellVertexGeometryElement,
                _cell_geometry.CellGeometryRestrictionSource,
                ExactPlcCellGeometrySource,
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPlcCellGeometryConvexSource,
                ExactPowerCellGeometryLinearActionSource,
                ExactRationalCellGeometrySource,
                RadicalElement,
                ExactAlgebraicCellGeometrySource,
                PeriodicCellGeometrySource,
            ),
        ),
        (
            "phydrax.geometry",
            (
                _mapped_reference_domain.MappedReferenceDomain,
                source_fidelity_owner.MappedDomainBoundarySource,
                PeriodicPowerPreparation,
            ),
        ),
        (
            "phydrax.geometry.surface",
            (
                _surface_contracts.SurfaceMetadata,
                _surface_contracts.SurfaceSelection,
                _surface_contracts.SurfaceInterface,
                _surface_contracts.SurfaceAuditPolicy,
                _surface_contracts.SurfaceOrientationRepair,
                _surface_contracts.SurfaceChartMappingEvidence,
                _surface_contracts.SurfaceAuditReport,
                _surface_contracts.SurfaceValidityCertificate,
                _surface_model.SurfaceModel,
                _surface_model.SurfaceRealization,
            ),
        ),
        ("phydrax.geometry.brep", (_intersection.BranchRootEndpoint,)),
    )
