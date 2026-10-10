"""Lazy native and optional comparison meshing provider facade."""

from importlib import import_module
from typing import Any, TYPE_CHECKING


_SYMBOL_MODULES: dict[str, tuple[str, str]] = {
    "FTetWildMeshingPlan": ("._ftetwild", "FTetWildMeshingPlan"),
    "FTetWildOptions": ("._ftetwild", "FTetWildOptions"),
    "FTetWildProvider": ("._ftetwild", "FTetWildProvider"),
    "GmshHighOrderOptimization": ("._gmsh_options", "GmshHighOrderOptimization"),
    "GmshMeshingPlan": ("._gmsh", "GmshMeshingPlan"),
    "GmshOptions": ("._gmsh_options", "GmshOptions"),
    "GmshProvider": ("._gmsh", "GmshProvider"),
    "GmshRemeshingPlan": ("._gmsh", "GmshRemeshingPlan"),
    "GmshSession": ("._gmsh", "GmshSession"),
    "GmshSurfaceAlgorithm": ("._gmsh_options", "GmshSurfaceAlgorithm"),
    "GmshVolumeAlgorithm": ("._gmsh_options", "GmshVolumeAlgorithm"),
    "ManifoldProvider": ("._manifold", "ManifoldProvider"),
    "MmgAdaptationPlan": ("._mmg", "MmgAdaptationPlan"),
    "MmgAdaptationResult": ("._mmg", "MmgAdaptationResult"),
    "MmgFieldTransfer": ("._mmg", "MmgFieldTransfer"),
    "MmgLagrangianMode": ("._mmg", "MmgLagrangianMode"),
    "MmgLagrangianMotion": ("._mmg", "MmgLagrangianMotion"),
    "MmgLevelSet": ("._mmg", "MmgLevelSet"),
    "MmgOptions": ("._mmg", "MmgOptions"),
    "MmgProvider": ("._mmg", "MmgProvider"),
    "MmgReference": ("._mmg", "MmgReference"),
    "MmgReferenceRetention": ("._mmg", "MmgReferenceRetention"),
    "MmgSessionEvidence": ("._mmg", "MmgSessionEvidence"),
    "NativeCurveSchedule": ("._native_options", "NativeCurveSchedule"),
    "NativeCurveSource": ("._native_sources", "NativeCurveSource"),
    "NativeHexGridRoute": (".._hex_generation", "NativeHexGridRoute"),
    "NativeHexGridSchedule": (".._hex_generation", "NativeHexGridSchedule"),
    "NativeImplicitSource": ("._native_sources", "NativeImplicitSource"),
    "NativeLayerCoreSource": ("._native_sources", "NativeLayerCoreSource"),
    "NativeMappedHexSource": ("._native_sources", "NativeMappedHexSource"),
    "NativeMeshingOptions": ("._native_options", "NativeMeshingOptions"),
    "NativeMeshingPlan": ("._native", "NativeMeshingPlan"),
    "NativeMeshingProvider": ("._native", "NativeMeshingProvider"),
    "NativeMeshingRoute": ("._native_options", "NativeMeshingRoute"),
    "NativeMeshingSource": ("._native_sources", "NativeMeshingSource"),
    "NativePeriodicSource": ("._native_periodic", "NativePeriodicSource"),
    "NativePlanarSource": ("._native_sources", "NativePlanarSource"),
    "NativePlcSource": ("._native_sources", "NativePlcSource"),
    "NativePolyhedralSchedule": (".._polyhedral_generation", "NativePolyhedralSchedule"),
    "NativePolyhedralSource": ("._native_sources", "NativePolyhedralSource"),
    "NativeStructuredSchedule": ("._native_options", "NativeStructuredSchedule"),
    "NativeStructuredSource": ("._native_sources", "NativeStructuredSource"),
    "NativeSurfaceEnvelopeSource": ("._native_sources", "NativeSurfaceEnvelopeSource"),
    "NativeSurfaceSchedule": ("._native_options", "NativeSurfaceSchedule"),
    "NativeSurfaceSource": ("._native_sources", "NativeSurfaceSource"),
    "NativeSweepSource": ("._native_sources", "NativeSweepSource"),
    "OmegaHAdaptationEvidence": ("._omega_h", "OmegaHAdaptationEvidence"),
    "OmegaHAdaptationResult": ("._omega_h", "OmegaHAdaptationResult"),
    "OmegaHClassification": ("._omega_h", "OmegaHClassification"),
    "OmegaHField": ("._omega_h", "OmegaHField"),
    "OmegaHFieldEvidence": ("._omega_h", "OmegaHFieldEvidence"),
    "OmegaHFieldTransfer": ("._omega_h", "OmegaHFieldTransfer"),
    "OmegaHOptions": ("._omega_h", "OmegaHOptions"),
    "OmegaHPartition": ("._omega_h", "OmegaHPartition"),
    "OmegaHProvider": ("._omega_h", "OmegaHProvider"),
    "OmegaHTransferredField": ("._omega_h", "OmegaHTransferredField"),
    "OpenVDBLevelSetRebuild": ("._openvdb", "OpenVDBLevelSetRebuild"),
    "OpenVDBMeshingSpec": ("._openvdb", "OpenVDBMeshingSpec"),
    "OpenVDBProvider": ("._openvdb", "OpenVDBProvider"),
    "OrientedPointCloud": ("._poisson", "OrientedPointCloud"),
    "PoissonBoundaryCondition": ("._poisson", "PoissonBoundaryCondition"),
    "PoissonProvider": ("._poisson", "PoissonProvider"),
    "PoissonReconstructionSpec": ("._poisson", "PoissonReconstructionSpec"),
    "TiogaAssemblyResult": ("._tioga", "TiogaAssemblyResult"),
    "TiogaDonorEvidence": ("._tioga", "TiogaDonorEvidence"),
    "TiogaOptions": ("._tioga", "TiogaOptions"),
    "TiogaPartBlanking": ("._tioga", "TiogaPartBlanking"),
    "TiogaProvider": ("._tioga", "TiogaProvider"),
    "TiogaRegistration": ("._tioga", "TiogaRegistration"),
    "VoroCrustOptions": ("._vorocrust", "VoroCrustOptions"),
    "VoroCrustProvider": ("._vorocrust", "VoroCrustProvider"),
    "native_provider_source_path": ("._sources", "native_provider_source_path"),
}


if TYPE_CHECKING:
    from .._hex_generation import (
        NativeHexGridRoute,
        NativeHexGridSchedule,
    )
    from .._polyhedral_generation import (
        NativePolyhedralSchedule,
    )
    from ._ftetwild import (
        FTetWildMeshingPlan,
        FTetWildOptions,
        FTetWildProvider,
    )
    from ._gmsh import (
        GmshMeshingPlan,
        GmshProvider,
        GmshRemeshingPlan,
        GmshSession,
    )
    from ._gmsh_options import (
        GmshHighOrderOptimization,
        GmshOptions,
        GmshSurfaceAlgorithm,
        GmshVolumeAlgorithm,
    )
    from ._manifold import (
        ManifoldProvider,
    )
    from ._mmg import (
        MmgAdaptationPlan,
        MmgAdaptationResult,
        MmgFieldTransfer,
        MmgLagrangianMode,
        MmgLagrangianMotion,
        MmgLevelSet,
        MmgOptions,
        MmgProvider,
        MmgReference,
        MmgReferenceRetention,
        MmgSessionEvidence,
    )
    from ._native import (
        NativeMeshingPlan,
        NativeMeshingProvider,
    )
    from ._native_options import (
        NativeCurveSchedule,
        NativeMeshingOptions,
        NativeMeshingRoute,
        NativeStructuredSchedule,
        NativeSurfaceSchedule,
    )
    from ._native_periodic import (
        NativePeriodicSource,
    )
    from ._native_sources import (
        NativeCurveSource,
        NativeImplicitSource,
        NativeLayerCoreSource,
        NativeMappedHexSource,
        NativeMeshingSource,
        NativePlanarSource,
        NativePlcSource,
        NativePolyhedralSource,
        NativeStructuredSource,
        NativeSurfaceEnvelopeSource,
        NativeSurfaceSource,
        NativeSweepSource,
    )
    from ._omega_h import (
        OmegaHAdaptationEvidence,
        OmegaHAdaptationResult,
        OmegaHClassification,
        OmegaHField,
        OmegaHFieldEvidence,
        OmegaHFieldTransfer,
        OmegaHOptions,
        OmegaHPartition,
        OmegaHProvider,
        OmegaHTransferredField,
    )
    from ._openvdb import (
        OpenVDBLevelSetRebuild,
        OpenVDBMeshingSpec,
        OpenVDBProvider,
    )
    from ._poisson import (
        OrientedPointCloud,
        PoissonBoundaryCondition,
        PoissonProvider,
        PoissonReconstructionSpec,
    )
    from ._sources import (
        native_provider_source_path,
    )
    from ._tioga import (
        TiogaAssemblyResult,
        TiogaDonorEvidence,
        TiogaOptions,
        TiogaPartBlanking,
        TiogaProvider,
        TiogaRegistration,
    )
    from ._vorocrust import (
        VoroCrustOptions,
        VoroCrustProvider,
    )


def __getattr__(name: str) -> Any:
    owner = _SYMBOL_MODULES.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol = owner
    value = getattr(import_module(module_name, __package__), symbol)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


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
    "NativeImplicitSource",
    "NativeLayerCoreSource",
    "NativeHexGridRoute",
    "NativeHexGridSchedule",
    "NativeMappedHexSource",
    "NativeMeshingOptions",
    "NativeMeshingPlan",
    "NativeMeshingProvider",
    "NativeMeshingRoute",
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
