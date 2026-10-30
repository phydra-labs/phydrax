"""Native and optional meshing providers, loaded only when requested."""

from typing import TYPE_CHECKING

from ..._deferred_exports import deferred_exports


_DEFERRED_EXPORTS = {
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
    "NativeMeshingOptions": "._native_options",
    "NativeMeshingPlan": "._native",
    "NativeMeshingProvider": "._native",
    "NativeMeshingRoute": "._native_options",
    "NativeMeshingSource": "._native_source_type",
    "NativePeriodicSource": "._native_periodic",
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
    "SurfaceBooleanOperation": "._manifold",
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

__all__ = list(_DEFERRED_EXPORTS)
__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
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
    from ._manifold import (
        ManifoldProvider as ManifoldProvider,
        SurfaceBooleanOperation as SurfaceBooleanOperation,
    )
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
    from ._native_source_type import NativeMeshingSource as NativeMeshingSource
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
    from ._sources import (
        native_provider_source_path as native_provider_source_path,
    )
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
