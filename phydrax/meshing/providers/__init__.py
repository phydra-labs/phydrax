#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lazy native and optional comparison meshing provider facade."""

from typing import TYPE_CHECKING

from ..._deferred_exports import deferred_exports


_DEFERRED_EXPORTS: dict[str, str] = {
    "FTetWildMeshingPlan": "._ftetwild",
    "FTetWildOptions": "._ftetwild",
    "FTetWildProvider": "._ftetwild",
    "GmshHighOrderOptimization": "._gmsh_options",
    "GmshMeshingPlan": "._gmsh",
    "GmshOptions": "._gmsh_options",
    "GmshProvider": "._gmsh",
    "GmshRemeshingPlan": "._gmsh",
    "GmshSession": "._gmsh",
    "GmshSurfaceAlgorithm": "._gmsh_options",
    "GmshVolumeAlgorithm": "._gmsh_options",
    "ManifoldProvider": "._manifold",
    "MmgAdaptationPlan": "._mmg",
    "MmgAdaptationResult": "._mmg",
    "MmgFieldTransfer": "._mmg",
    "MmgLagrangianMode": "._mmg",
    "MmgLagrangianMotion": "._mmg",
    "MmgLevelSet": "._mmg",
    "MmgOptions": "._mmg",
    "MmgProvider": "._mmg",
    "MmgReference": "._mmg",
    "MmgReferenceRetention": "._mmg",
    "MmgSessionEvidence": "._mmg",
    "NativeCurveSchedule": "._native_schedules",
    "NativeCurveSource": "._native_sources",
    "NativeHexGridRoute": ".._hex_generation",
    "NativeHexGridSchedule": ".._hex_generation",
    "NativeImplicitSource": "._native_sources",
    "NativeLayerCoreSource": "._native_sources",
    "NativeMappedHexSource": "._native_sources",
    "NativeMeshingOptions": "._native_options",
    "NativeMeshingPlan": "._native",
    "NativeMeshingProvider": "._native",
    "NativeMeshingRoute": "._native_options",
    "NativeMeshingSession": "._native_session",
    "NativeMeshingSource": "._native_sources",
    "NativePeriodicSource": "._native_periodic",
    "NativePlanarSource": "._native_sources",
    "NativePlcSource": "._native_sources",
    "NativePolyhedralSchedule": ".._polyhedral_generation",
    "NativePolyhedralSource": "._native_sources",
    "NativeStructuredSchedule": "._native_schedules",
    "NativeStructuredSource": "._native_sources",
    "NativeSurfaceEnvelopeSource": "._native_sources",
    "NativeSurfaceSchedule": "._native_schedules",
    "NativeSurfaceSource": "._native_sources",
    "NativeSweepSource": "._native_sources",
    "OmegaHAdaptationEvidence": "._omega_h",
    "OmegaHAdaptationResult": "._omega_h",
    "OmegaHClassification": "._omega_h",
    "OmegaHField": "._omega_h",
    "OmegaHFieldEvidence": "._omega_h",
    "OmegaHFieldTransfer": "._omega_h",
    "OmegaHOptions": "._omega_h",
    "OmegaHPartition": "._omega_h",
    "OmegaHProvider": "._omega_h",
    "OmegaHTransferredField": "._omega_h",
    "OpenVDBLevelSetRebuild": "._openvdb",
    "OpenVDBMeshingSpec": "._openvdb",
    "OpenVDBProvider": "._openvdb",
    "OrientedPointCloud": "._poisson",
    "PoissonBoundaryCondition": "._poisson",
    "PoissonProvider": "._poisson",
    "PoissonReconstructionSpec": "._poisson",
    "PreparedPlanarDomain": "._native_planar_domain",
    "TiogaAssemblyResult": "._tioga",
    "TiogaDonorEvidence": "._tioga",
    "TiogaOptions": "._tioga",
    "TiogaPartBlanking": "._tioga",
    "TiogaProvider": "._tioga",
    "TiogaRegistration": "._tioga",
    "VoroCrustOptions": "._vorocrust",
    "VoroCrustProvider": "._vorocrust",
    "native_provider_source_path": "._sources",
}

__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


# BEGIN GENERATED FACADE EXPORTS
if TYPE_CHECKING:
    from .._hex_generation import (
        NativeHexGridRoute as NativeHexGridRoute,
        NativeHexGridSchedule as NativeHexGridSchedule,
    )
    from .._polyhedral_generation import (
        NativePolyhedralSchedule as NativePolyhedralSchedule,
    )
    from ._ftetwild import (
        FTetWildMeshingPlan as FTetWildMeshingPlan,
        FTetWildOptions as FTetWildOptions,
        FTetWildProvider as FTetWildProvider,
    )
    from ._gmsh import (
        GmshMeshingPlan as GmshMeshingPlan,
        GmshProvider as GmshProvider,
        GmshRemeshingPlan as GmshRemeshingPlan,
        GmshSession as GmshSession,
    )
    from ._gmsh_options import (
        GmshHighOrderOptimization as GmshHighOrderOptimization,
        GmshOptions as GmshOptions,
        GmshSurfaceAlgorithm as GmshSurfaceAlgorithm,
        GmshVolumeAlgorithm as GmshVolumeAlgorithm,
    )
    from ._manifold import ManifoldProvider as ManifoldProvider
    from ._mmg import (
        MmgAdaptationPlan as MmgAdaptationPlan,
        MmgAdaptationResult as MmgAdaptationResult,
        MmgFieldTransfer as MmgFieldTransfer,
        MmgLagrangianMode as MmgLagrangianMode,
        MmgLagrangianMotion as MmgLagrangianMotion,
        MmgLevelSet as MmgLevelSet,
        MmgOptions as MmgOptions,
        MmgProvider as MmgProvider,
        MmgReference as MmgReference,
        MmgReferenceRetention as MmgReferenceRetention,
        MmgSessionEvidence as MmgSessionEvidence,
    )
    from ._native import (
        NativeMeshingPlan as NativeMeshingPlan,
        NativeMeshingProvider as NativeMeshingProvider,
    )
    from ._native_options import (
        NativeMeshingOptions as NativeMeshingOptions,
        NativeMeshingRoute as NativeMeshingRoute,
    )
    from ._native_periodic import NativePeriodicSource as NativePeriodicSource
    from ._native_planar_domain import PreparedPlanarDomain as PreparedPlanarDomain
    from ._native_schedules import (
        NativeCurveSchedule as NativeCurveSchedule,
        NativeStructuredSchedule as NativeStructuredSchedule,
        NativeSurfaceSchedule as NativeSurfaceSchedule,
    )
    from ._native_session import NativeMeshingSession as NativeMeshingSession
    from ._native_sources import (
        NativeCurveSource as NativeCurveSource,
        NativeImplicitSource as NativeImplicitSource,
        NativeLayerCoreSource as NativeLayerCoreSource,
        NativeMappedHexSource as NativeMappedHexSource,
        NativeMeshingSource as NativeMeshingSource,
        NativePlanarSource as NativePlanarSource,
        NativePlcSource as NativePlcSource,
        NativePolyhedralSource as NativePolyhedralSource,
        NativeStructuredSource as NativeStructuredSource,
        NativeSurfaceEnvelopeSource as NativeSurfaceEnvelopeSource,
        NativeSurfaceSource as NativeSurfaceSource,
        NativeSweepSource as NativeSweepSource,
    )
    from ._omega_h import (
        OmegaHAdaptationEvidence as OmegaHAdaptationEvidence,
        OmegaHAdaptationResult as OmegaHAdaptationResult,
        OmegaHClassification as OmegaHClassification,
        OmegaHField as OmegaHField,
        OmegaHFieldEvidence as OmegaHFieldEvidence,
        OmegaHFieldTransfer as OmegaHFieldTransfer,
        OmegaHOptions as OmegaHOptions,
        OmegaHPartition as OmegaHPartition,
        OmegaHProvider as OmegaHProvider,
        OmegaHTransferredField as OmegaHTransferredField,
    )
    from ._openvdb import (
        OpenVDBLevelSetRebuild as OpenVDBLevelSetRebuild,
        OpenVDBMeshingSpec as OpenVDBMeshingSpec,
        OpenVDBProvider as OpenVDBProvider,
    )
    from ._poisson import (
        OrientedPointCloud as OrientedPointCloud,
        PoissonBoundaryCondition as PoissonBoundaryCondition,
        PoissonProvider as PoissonProvider,
        PoissonReconstructionSpec as PoissonReconstructionSpec,
    )
    from ._sources import native_provider_source_path as native_provider_source_path
    from ._tioga import (
        TiogaAssemblyResult as TiogaAssemblyResult,
        TiogaDonorEvidence as TiogaDonorEvidence,
        TiogaOptions as TiogaOptions,
        TiogaPartBlanking as TiogaPartBlanking,
        TiogaProvider as TiogaProvider,
        TiogaRegistration as TiogaRegistration,
    )
    from ._vorocrust import (
        VoroCrustOptions as VoroCrustOptions,
        VoroCrustProvider as VoroCrustProvider,
    )

__all__ = [
    "FTetWildMeshingPlan",
    "FTetWildOptions",
    "FTetWildProvider",
    "GmshHighOrderOptimization",
    "GmshMeshingPlan",
    "GmshOptions",
    "GmshProvider",
    "GmshRemeshingPlan",
    "GmshSession",
    "GmshSurfaceAlgorithm",
    "GmshVolumeAlgorithm",
    "ManifoldProvider",
    "MmgAdaptationPlan",
    "MmgAdaptationResult",
    "MmgFieldTransfer",
    "MmgLagrangianMode",
    "MmgLagrangianMotion",
    "MmgLevelSet",
    "MmgOptions",
    "MmgProvider",
    "MmgReference",
    "MmgReferenceRetention",
    "MmgSessionEvidence",
    "NativeCurveSchedule",
    "NativeCurveSource",
    "NativeHexGridRoute",
    "NativeHexGridSchedule",
    "NativeImplicitSource",
    "NativeLayerCoreSource",
    "NativeMappedHexSource",
    "NativeMeshingOptions",
    "NativeMeshingPlan",
    "NativeMeshingProvider",
    "NativeMeshingRoute",
    "NativeMeshingSession",
    "NativeMeshingSource",
    "NativePeriodicSource",
    "NativePlanarSource",
    "NativePlcSource",
    "NativePolyhedralSchedule",
    "NativePolyhedralSource",
    "NativeStructuredSchedule",
    "NativeStructuredSource",
    "NativeSurfaceEnvelopeSource",
    "NativeSurfaceSchedule",
    "NativeSurfaceSource",
    "NativeSweepSource",
    "OmegaHAdaptationEvidence",
    "OmegaHAdaptationResult",
    "OmegaHClassification",
    "OmegaHField",
    "OmegaHFieldEvidence",
    "OmegaHFieldTransfer",
    "OmegaHOptions",
    "OmegaHPartition",
    "OmegaHProvider",
    "OmegaHTransferredField",
    "OpenVDBLevelSetRebuild",
    "OpenVDBMeshingSpec",
    "OpenVDBProvider",
    "OrientedPointCloud",
    "PoissonBoundaryCondition",
    "PoissonProvider",
    "PoissonReconstructionSpec",
    "PreparedPlanarDomain",
    "TiogaAssemblyResult",
    "TiogaDonorEvidence",
    "TiogaOptions",
    "TiogaPartBlanking",
    "TiogaProvider",
    "TiogaRegistration",
    "VoroCrustOptions",
    "VoroCrustProvider",
    "native_provider_source_path",
]
# END GENERATED FACADE EXPORTS
