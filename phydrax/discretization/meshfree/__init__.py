# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Prepared meshfree approximation, conservative metrics, and surface transport."""

from importlib import import_module
from typing import Any, TYPE_CHECKING


_FACADE_EXPORT_MODULES: dict[str, str] = {
    "MeshfreeApproximation": "._types",
    "MeshfreeRowStatus": "._types",
    "StencilAcceptance": "._types",
    "StencilWeightKernel": "._types",
    "MeshfreeNeighborhoodPlan": "._neighbors",
    "PreparedMeshfreeNeighborhood": "._neighbors",
    "MeshfreeEdgeRelationPlan": "._neighbors",
    "PreparedMeshfreeEdgeRelation": "._neighbors",
    "LocalStencilPolicy": "._stencils",
    "MeshfreeFunctional": "._stencils",
    "LocalStencilEvidence": "._stencils",
    "LocalStencilReport": "._stencils",
    "PreparedLocalStencils": "._stencils",
    "prepare_local_stencils": "._stencils",
    "prepare_chart_stencils": "._stencils",
    "MeshfreeOperator": "._operators",
    "HyperviscosityPlan": "._stabilization",
    "PreparedHyperviscosity": "._stabilization",
    "HyperviscosityEvidence": "._stabilization",
    "MeshfreeCoarseningPolicy": "._multilevel",
    "MeshfreeHierarchyPlan": "._multilevel",
    "PreparedMeshfreeHierarchy": "._multilevel",
    "MeshfreeHierarchyEvidence": "._multilevel",
    "meshfree_multigrid_builder": "._multilevel",
    "ImplicitSurfaceGeometry": "._surface_geometry",
    "SampledSurfaceGeometry": "._surface_geometry",
    "SurfaceGeometryStatus": "._surface_geometry",
    "SurfaceGeometryEvidence": "._surface_geometry",
    "SurfaceGeometryEvaluation": "._surface_geometry",
    "SurfaceNodeDim": "._surface_geometry",
    "SurfaceQuadratureKind": "._surface_quadrature",
    "SurfaceQuadratureEvidence": "._surface_quadrature",
    "SurfaceQuadraturePolicy": "._surface_quadrature",
    "SurfacePointCloudPlan": "._surface",
    "PreparedSurfacePointCloud": "._surface",
    "SurfaceRefreshResult": "._surface",
    "SurfaceFieldReconstructionKernel": "._surface",
    "MeshfreeExteriorCalculusPlan": "._exterior",
    "PreparedMeshfreeExteriorCalculus": "._exterior",
    "MeshfreeDiffusionOperator": "._exterior",
    "MeshfreeStiffnessEvidence": "._exterior",
    "MeshfreeBoundaryQuadrature": "._exterior",
    "EdgeCoefficientAverage": "._exterior",
    "MeshfreeMetricPolicy": "._exterior_metric",
    "MeshfreeMetricResult": "._exterior_metric",
    "MeshfreeMetricStatus": "._exterior_metric",
    "MeshfreeMomentRowStatus": "._exterior_metric",
    "PreparedMeshfreeMetric": "._exterior_metric",
    "MetricSign": "._exterior_metric",
    "MetricAcceptance": "._exterior_metric",
    "MeshfreeAdvection": "._exterior_transport",
    "MeshfreeAdvectionResult": "._exterior_transport",
    "MeshfreeAdvectionStatus": "._exterior_transport",
    "edge_upwind_content": "._exterior_transport",
    "MovingSurfaceStatus": "._moving",
    "MovingGeometryRefresh": "._moving",
    "MovingSurfaceState": "._moving",
    "MovingSurfaceEvidence": "._moving",
    "MovingSurfaceStepResult": "._moving",
    "MovingSurfaceCheckpoint": "._moving",
    "MovingSurfaceEpochResult": "._moving",
    "MovingSurfacePlan": "._moving",
    "MeshfreeCapacityMap": "._capacity",
    "MeshfreeCapacityPolicy": "._capacity",
    "SurfaceShiftPolicy": "._shifting",
    "SurfaceShiftResult": "._shifting",
    "SurfaceRelativeAdvectionResult": "._shifting",
    "surface_relative_advection": "._shifting",
    "SurfaceQualityEvidence": "._resampling",
    "SurfaceResamplingPolicy": "._resampling",
    "SurfaceResamplingResult": "._resampling",
    "SurfaceTransferMode": "._surface_transfer",
    "SurfaceTransferEvidence": "._surface_transfer",
    "PreparedSurfaceTransfer": "._surface_transfer",
    "SurfaceTransferPlan": "._surface_transfer",
    "AbstractEdgeConstitutiveLaw": "._constitutive",
    "EdgeFeatureField": "._constitutive",
    "EdgeFeatureKind": "._constitutive",
    "EdgeFrameFeatures": "._constitutive",
    "EdgeModelLipschitzCertificate": "._constitutive",
    "LipschitzEdgeFlux": "._constitutive",
    "MonotoneEdgeConductance": "._constitutive",
    "EdgeCoverageAssessment": "._coverage",
    "EdgeCoveragePolicy": "._coverage",
    "EdgeCoverageStatus": "._coverage",
    "EdgeFeatureCoverage": "._coverage",
    "MeshfreeConservationAdjoint": "._conservation_solve",
    "MeshfreeConservationLedger": "._conservation_solve",
    "MeshfreeConservationProblem": "._conservation_solve",
    "MeshfreeConservationResult": "._conservation_solve",
    "MeshfreeConstitutiveEvidence": "._conservation_solve",
    "MeshfreeContractionStatus": "._conservation_solve",
    "PreparedMeshfreeConservationSolve": "._conservation_solve",
    "prepare_meshfree_conservation_solve": "._conservation_solve",
    "meshfree_candidate_profiles": "._profiles",
}

__all__ = list(_FACADE_EXPORT_MODULES)

if TYPE_CHECKING:
    from ._capacity import (
        MeshfreeCapacityMap as MeshfreeCapacityMap,
        MeshfreeCapacityPolicy as MeshfreeCapacityPolicy,
    )
    from ._conservation_solve import (
        MeshfreeConservationAdjoint as MeshfreeConservationAdjoint,
        MeshfreeConservationLedger as MeshfreeConservationLedger,
        MeshfreeConservationProblem as MeshfreeConservationProblem,
        MeshfreeConservationResult as MeshfreeConservationResult,
        MeshfreeConstitutiveEvidence as MeshfreeConstitutiveEvidence,
        MeshfreeContractionStatus as MeshfreeContractionStatus,
        prepare_meshfree_conservation_solve as prepare_meshfree_conservation_solve,
        PreparedMeshfreeConservationSolve as PreparedMeshfreeConservationSolve,
    )
    from ._constitutive import (
        AbstractEdgeConstitutiveLaw as AbstractEdgeConstitutiveLaw,
        EdgeFeatureField as EdgeFeatureField,
        EdgeFeatureKind as EdgeFeatureKind,
        EdgeFrameFeatures as EdgeFrameFeatures,
        EdgeModelLipschitzCertificate as EdgeModelLipschitzCertificate,
        LipschitzEdgeFlux as LipschitzEdgeFlux,
        MonotoneEdgeConductance as MonotoneEdgeConductance,
    )
    from ._coverage import (
        EdgeCoverageAssessment as EdgeCoverageAssessment,
        EdgeCoveragePolicy as EdgeCoveragePolicy,
        EdgeCoverageStatus as EdgeCoverageStatus,
        EdgeFeatureCoverage as EdgeFeatureCoverage,
    )
    from ._exterior import (
        EdgeCoefficientAverage as EdgeCoefficientAverage,
        MeshfreeBoundaryQuadrature as MeshfreeBoundaryQuadrature,
        MeshfreeDiffusionOperator as MeshfreeDiffusionOperator,
        MeshfreeExteriorCalculusPlan as MeshfreeExteriorCalculusPlan,
        MeshfreeStiffnessEvidence as MeshfreeStiffnessEvidence,
        PreparedMeshfreeExteriorCalculus as PreparedMeshfreeExteriorCalculus,
    )
    from ._exterior_metric import (
        MeshfreeMetricPolicy as MeshfreeMetricPolicy,
        MeshfreeMetricResult as MeshfreeMetricResult,
        MeshfreeMetricStatus as MeshfreeMetricStatus,
        MeshfreeMomentRowStatus as MeshfreeMomentRowStatus,
        MetricAcceptance as MetricAcceptance,
        MetricSign as MetricSign,
        PreparedMeshfreeMetric as PreparedMeshfreeMetric,
    )
    from ._exterior_transport import (
        edge_upwind_content as edge_upwind_content,
        MeshfreeAdvection as MeshfreeAdvection,
        MeshfreeAdvectionResult as MeshfreeAdvectionResult,
        MeshfreeAdvectionStatus as MeshfreeAdvectionStatus,
    )
    from ._moving import (
        MovingGeometryRefresh as MovingGeometryRefresh,
        MovingSurfaceCheckpoint as MovingSurfaceCheckpoint,
        MovingSurfaceEpochResult as MovingSurfaceEpochResult,
        MovingSurfaceEvidence as MovingSurfaceEvidence,
        MovingSurfacePlan as MovingSurfacePlan,
        MovingSurfaceState as MovingSurfaceState,
        MovingSurfaceStatus as MovingSurfaceStatus,
        MovingSurfaceStepResult as MovingSurfaceStepResult,
    )
    from ._multilevel import (
        meshfree_multigrid_builder as meshfree_multigrid_builder,
        MeshfreeCoarseningPolicy as MeshfreeCoarseningPolicy,
        MeshfreeHierarchyEvidence as MeshfreeHierarchyEvidence,
        MeshfreeHierarchyPlan as MeshfreeHierarchyPlan,
        PreparedMeshfreeHierarchy as PreparedMeshfreeHierarchy,
    )
    from ._neighbors import (
        MeshfreeEdgeRelationPlan as MeshfreeEdgeRelationPlan,
        MeshfreeNeighborhoodPlan as MeshfreeNeighborhoodPlan,
        PreparedMeshfreeEdgeRelation as PreparedMeshfreeEdgeRelation,
        PreparedMeshfreeNeighborhood as PreparedMeshfreeNeighborhood,
    )
    from ._operators import MeshfreeOperator as MeshfreeOperator
    from ._profiles import meshfree_candidate_profiles as meshfree_candidate_profiles
    from ._resampling import (
        SurfaceQualityEvidence as SurfaceQualityEvidence,
        SurfaceResamplingPolicy as SurfaceResamplingPolicy,
        SurfaceResamplingResult as SurfaceResamplingResult,
    )
    from ._shifting import (
        surface_relative_advection as surface_relative_advection,
        SurfaceRelativeAdvectionResult as SurfaceRelativeAdvectionResult,
        SurfaceShiftPolicy as SurfaceShiftPolicy,
        SurfaceShiftResult as SurfaceShiftResult,
    )
    from ._stabilization import (
        HyperviscosityEvidence as HyperviscosityEvidence,
        HyperviscosityPlan as HyperviscosityPlan,
        PreparedHyperviscosity as PreparedHyperviscosity,
    )
    from ._stencils import (
        LocalStencilEvidence as LocalStencilEvidence,
        LocalStencilPolicy as LocalStencilPolicy,
        LocalStencilReport as LocalStencilReport,
        MeshfreeFunctional as MeshfreeFunctional,
        prepare_chart_stencils as prepare_chart_stencils,
        prepare_local_stencils as prepare_local_stencils,
        PreparedLocalStencils as PreparedLocalStencils,
    )
    from ._surface import (
        PreparedSurfacePointCloud as PreparedSurfacePointCloud,
        SurfaceFieldReconstructionKernel as SurfaceFieldReconstructionKernel,
        SurfacePointCloudPlan as SurfacePointCloudPlan,
        SurfaceRefreshResult as SurfaceRefreshResult,
    )
    from ._surface_geometry import (
        ImplicitSurfaceGeometry as ImplicitSurfaceGeometry,
        SampledSurfaceGeometry as SampledSurfaceGeometry,
        SurfaceGeometryEvaluation as SurfaceGeometryEvaluation,
        SurfaceGeometryEvidence as SurfaceGeometryEvidence,
        SurfaceGeometryStatus as SurfaceGeometryStatus,
        SurfaceNodeDim as SurfaceNodeDim,
    )
    from ._surface_quadrature import (
        SurfaceQuadratureEvidence as SurfaceQuadratureEvidence,
        SurfaceQuadratureKind as SurfaceQuadratureKind,
        SurfaceQuadraturePolicy as SurfaceQuadraturePolicy,
    )
    from ._surface_transfer import (
        PreparedSurfaceTransfer as PreparedSurfaceTransfer,
        SurfaceTransferEvidence as SurfaceTransferEvidence,
        SurfaceTransferMode as SurfaceTransferMode,
        SurfaceTransferPlan as SurfaceTransferPlan,
    )
    from ._types import (
        MeshfreeApproximation as MeshfreeApproximation,
        MeshfreeRowStatus as MeshfreeRowStatus,
        StencilAcceptance as StencilAcceptance,
        StencilWeightKernel as StencilWeightKernel,
    )


def __getattr__(name: str) -> Any:
    module = _FACADE_EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
