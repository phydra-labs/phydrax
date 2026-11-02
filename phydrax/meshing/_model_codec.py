#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact registered-recipe persistence for native scientific models."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .._array_archive import (
    array_payload_digest,
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    read_array_archive,
    write_array_archive,
)
from .._model._artifacts import artifact_value, artifact_value_id
from .._model._structure import (
    _recipe_array_nodes,
    _row_record_recipe_types,
    AdaptiveImplicitBoundarySourceArchiveRecipe,
    AnalyticImplicitProfileArchiveRecipe,
    CADGraphArchiveRecipe,
    CADRevisionArchiveRecipe,
    CADTransactionArchiveRecipe,
    DistributedFiniteElementTransferArchiveRecipe,
    expand_shared_model_archive_recipe,
    FiniteElementAcceptedStateArchiveRecipe,
    FiniteVolumeOversetPartSpecArchiveRecipe,
    FiniteVolumeOversetPolicyArchiveRecipe,
    FiniteVolumeOwnerLocalCellStateBankArchiveRecipe,
    FiniteVolumeOwnerLocalRepartitionContextArchiveRecipe,
    FiniteVolumeRuntimeStateArchiveRecipe,
    MappedReferenceBoundarySourceArchiveRecipe,
    MeshingDomainArchiveRecipe,
    MeshingDomainBoundarySourceArchiveRecipe,
    model_archive_root_projection,
    model_from_array_recipe,
    model_structure_matches_recipe,
    model_structure_recipe,
    NativeImplicitBoundaryQueryArchiveRecipe,
    NativeMappedHexSourceArchiveRecipe,
    NativePlcSourceArchiveRecipe,
    pack_model_array_tree,
    ParametricCurveBoundarySourceArchiveRecipe,
    PartitionOperandArchiveRecipe,
    PartitionPlanArchiveRecipe,
    PERIODIC_FINITE_ELEMENT_TRANSFER_ROOT_ENTRY_IDS,
    PeriodicFiniteElementTransferArchiveRecipe,
    PiecewiseLinearBoundarySourceArchiveRecipe,
    PreparedSphereChartDeformationArchiveRecipe,
    PreparedSurfaceSourceSupportArchiveRecipe,
    ReconstructedTriangleBoundarySourceArchiveRecipe,
    restore_row_record_recipes,
    share_model_archive_recipe,
    SparseCoordinateOperatorArchiveRecipe,
    SphereMaterialCellAtlasArchiveRecipe,
    SurfaceChartBoundarySourceArchiveRecipe,
    SurfaceEnvelopeSourceArchiveRecipe,
    SurfaceNativeRestrictionBoundarySourceArchiveRecipe,
    SurfaceSourceChartsArchiveRecipe,
    SurfaceSubdivisionBoundarySourceArchiveRecipe,
    validate_model_structure_recipe,
)
from .._strict import Strict
from .._typing_plan import validate_tree


_MODEL_KIND = "phydrax-native-model-artifact"
_MODEL_ARRAYS = "model/leaves"

# Admission limits for untrusted model-recipe archives. Every recipe node costs at
# most two manifest JSON levels (the node object and its ``fields``/``items``
# container), so this admits 31 nested native objects. The deepest exact CAD chain
# in the schema (model, trim domain, curve loop, segment, root endpoint, carrier,
# surface patch, profile curve, exact rational coordinate, with their tuple
# containers) uses 12 recipe nodes and 25 manifest levels; placements and offsets
# add one node each. Model recipes store one archive member per dynamic array leaf,
# so composite models admit 4,096 leaves (plus the manifest); the central-directory,
# member, and aggregate byte limits still bound every allocation. Every other
# admission bound keeps its conservative default.
MODEL_ARCHIVE_LIMITS = ArrayArchiveLimits(max_manifest_nesting=64, max_members=4097)


def _register_model_archive_types() -> None:
    """Register built-in scientific models supported by the generic recipe codec."""
    from .._archive_columns import (
        ExactTensorArchiveRecipe,
        IntegerRowsArchiveRecipe,
        StaticColumnArchiveRecipe,
        TextColumnArchiveRecipe,
        TextRowsArchiveRecipe,
    )
    from .._geometry_predicates import PredicateMode
    from .._identity import SemanticProvenance
    from .._model._artifacts import register_artifact_value
    from .._physical import SpatialCoordinateContract
    from ..discretization._axis import AxisDiscretization
    from ..discretization._axis_domain import AxisDomain
    from ..discretization._cell_complex import (
        IntervalConnectivity,
        PolygonalConnectivity,
        PolyhedralConnectivity,
        TetrahedralConnectivity,
    )
    from ..discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        CellGeometryRestrictionSource,
        CellGeometrySpec,
        CellVertexGeometryElement,
        LayerColumnCellGeometryElement,
        PerCellRestrictedCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )
    from ..discretization._cell_geometry_transition_types import (
        CellGeometryTransition,
        CellGeometryTransitionEvidence,
    )
    from ..discretization._cell_geometry_validity import (
        CellValidityCertificate,
        CellValidityPolicy,
    )
    from ..discretization._cell_mesh import (
        CellBlock,
        CellMesh,
        EmptyLocalCellMeshAuthority,
        PolyhedralBlock,
        SimplicialConnectivity,
    )
    from ..discretization._distributed_field import (
        _TransferExchange,
        DistributedSparseTransfer,
    )
    from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
    from ..discretization._exact_power_geometry import (
        ExactPeriodicPowerSourceContext,
        ExactPowerCellGeometrySource,
    )
    from ..discretization._fv_precision import FiniteVolumePrecisionPolicy
    from ..discretization._hexahedral import HexahedralConnectivity
    from ..discretization._mapped_power_geometry import MappedPowerCellGeometrySource
    from ..discretization._measure import DiscreteMeasure
    from ..discretization._nested_reference_types import (
        NestedReferenceWitnesses,
        ReferenceCellMapWitnesses,
    )
    from ..discretization._partition import CellPartition
    from ..discretization._periodic_cell import PeriodicCell
    from ..discretization._periodic_exact import PeriodicIsometryGroup
    from ..discretization._periodic_topology import PeriodicMeshTopology
    from ..discretization._sphere_chart_records import (
        PreparedSphereChartDeformation,
        PreparedSphereChartOccurrence,
        PreparedSphereChartPiece,
    )
    from ..discretization._support import DiscreteSupport
    from ..discretization._surface_chart_records import (
        PreparedSurfaceChartDeformation,
        PreparedSurfaceChartOccurrence,
        PreparedSurfaceChartPiece,
        SurfaceChartWitness,
        SurfaceChartWitnessArchiveRecipe,
        SurfaceRootMeshCellBinding,
    )
    from ..discretization._tensor_entities import StructuredAxis, TensorEntityLayout
    from ..discretization._tensor_support import GridLocation, PreparedTensorGrid
    from ..discretization._topology import (
        CellComplexTopology,
        EntitySet,
        EntitySubset,
        OrientedIncidence,
        TensorTopology,
    )
    from ..discretization._view_support import (
        OwnerLocalCellBlock,
        OwnerLocalCellGeometry,
    )
    from ..discretization.fem._de_rham import (
        DistributedFiniteElementTransfer,
        FiniteElementTransferResult,
        PeriodicFiniteElementTransfer,
        SparseCommutingEvidence,
        SparseTransferEntryEvidence,
    )
    from ..discretization.fem._distributed import FiniteElementDofOwnership
    from ..discretization.fem._reference import FiniteElementSpec
    from ..discretization.fem._topology_transfer import FiniteElementTopologyTransfer
    from ..discretization.finite_volume._owner_local_remap import (
        OwnerLocalConservativeRemapEvidence,
        PreparedOwnerLocalConservativeRemap,
    )
    from ..equations._finite_element_material import (
        MaterialSiteId,
        MaterialState,
        MaterialTransaction,
    )
    from ..exterior._form_type import FormType, FormValueSpec
    from ..geometry._atlas import BoundaryAtlas, BoundaryFrame
    from ..geometry._cad_revision import (
        AssociationCoverageEvidence,
        AssociationGraph,
        CADOccurrence,
        CADRevision,
        CADSelectionSet,
        CADSelector,
        OccurrenceCorrespondence,
        OccurrenceCorrespondenceTransaction,
    )
    from ..geometry._common_root_volume_overlap import (
        CommonRootVolumeWitness,
        CurvedOverlapIntegrationEvidence,
        CurvedOverlapIntegrationPolicy,
        MaterialReferenceVolumeOverlap,
        MaterialReferenceVolumeWitness,
    )
    from ..geometry._contracts import CompiledGeometry, GeometryTolerance
    from ..geometry._curve_boundary import ParametricCurveBoundarySource
    from ..geometry._domain_boundary import MeshingDomainBoundarySource
    from ..geometry._domain_certification import DomainCoverageCertificate
    from ..geometry._domain_coverage import PiecewiseLinearDomain
    from ..geometry._embedding_state import (
        EmbeddingCounts,
        MeshCertificateFinding,
        MeshEmbeddingWitness,
        MeshEmbeddingWitnessArchiveRecipe,
        PeriodicAffineTorusDegreeEvidence,
        PeriodicMappedTorusDegreeEvidence,
        PeriodicRestrictionPartitionEvidence,
    )
    from ..geometry._global_embedding import GlobalEmbeddingCertificate
    from ..geometry._mapped_reference_domain import MappedReferenceDomain
    from ..geometry._mesh_certificates import MeshCertificateBinding
    from ..geometry._meshing_domain import (
        _PlacedSurfacePatch,
        DomainCurveMap,
        MeshingDomain,
        MeshingDomainCurve,
        MeshingDomainRegion,
        MeshingSurfacePatch,
        PatchCurveUse,
        PatchPoleUse,
    )
    from ..geometry._periodic_mapped_restriction import PeriodicMappedRestrictionEvidence
    from ..geometry._periodic_restriction import PeriodicRestrictionPremise
    from ..geometry._planar_coverage import FacetOverlap, PlcSourcePointWitness
    from ..geometry._planar_embedding import PlanarEmbedding
    from ..geometry._source_boundary_records import (
        SourceBoundaryChartCover,
        SourceBoundaryChartWitness,
        SourceBoundaryDistance,
        SourceBoundaryDistanceWitness,
        SourceBoundaryPointWitness,
        SourceBoundarySamples,
        SurfaceEnvelopeDistanceWitness,
    )
    from ..geometry._source_fidelity_certificate import SourceFidelityCertificate
    from ..geometry._sphere_material_atlas import (
        _SphereMaterialFidelityBinding,
        SphereMaterialCellAtlas,
        SphereMaterialFidelityCertificate,
    )
    from ..geometry._supermesh import (
        CommonRefinementCoverage,
        CommonRefinementEvidence,
        CommonRefinementPolicy,
        CommonRefinementStatus,
        PreparedCommonRefinement,
    )
    from ..geometry._surface_source_support import (
        _SplineSurfaceCoordinateMap,
        PreparedSurfaceSourceSupport,
        SurfaceNativeRestrictionBoundarySource,
        SurfaceSourceCharts,
        SurfaceSourceRootAtlas,
    )
    from ..geometry._volume_material_deformation import (
        MaterialReferenceDeclaration,
        MaterialReferencePolicy,
        VolumeMaterialDeformation,
    )
    from ..geometry._volume_material_map import (
        EulerianVolumeOverlapChain,
        EulerianVolumeOverlapMap,
        MaterialMapActionEvidence,
        MaterialSourceMapCapability,
        VolumeMaterialChain,
        VolumeMaterialMap,
    )
    from ..geometry.analytic._extended import _TorusKernel
    from ..geometry.analytic._primitives import _BallKernel
    from ..geometry.brep._constructors import BRepTessellationPolicy
    from ..geometry.brep._partition import (
        BRepPartitionPatch,
        BRepPartitionRegion,
        BRepPartitionReport,
        BRepPartitionResult,
        BRepPartitionResultArchiveRecipe,
    )
    from ..geometry.brep._projection_contracts import BRepProjectionResult
    from ..geometry.design._continuation import SketchContinuationReferenceArchiveRecipe
    from ..geometry.design._reduced import (
        _SketchDistanceRigidityWitness,
        _SketchDistanceTriangleWitness,
    )
    from ..geometry.design._schema import (
        DesignState,
        ParameterBinding,
        ParameterId,
        ParameterSchema,
        ParameterSpec,
    )
    from ..geometry.design._sketch_certificate import (
        LineCircleConstruction,
        TwoCircleConstruction,
    )
    from ..geometry.implicit._adaptive_discovery import (
        AdaptiveImplicitSurfaceArchiveRecipe,
    )
    from ..geometry.implicit._chart_restriction import (
        ImplicitChartPremise,
        ImplicitChartRestrictionEvidence,
        ImplicitChartRows,
    )
    from ..geometry.implicit._policy import AdaptiveImplicitSurfacePolicy
    from ..geometry.implicit._restricted_delaunay_coverage import (
        RestrictedDelaunayBoundaryWitness,
        RestrictedDelaunayCoverageTransportWitness,
    )
    from ..geometry.reconstruction._boundary import (
        _TriangleBVHNode,
        ReconstructedTriangleBoundarySource,
    )
    from ..geometry.reconstruction._core import (
        _ReconstructedTrianglePatch,
        ReconstructedGeometrySourceArchiveRecipe,
        ReconstructionReport,
    )
    from ..geometry.reconstruction._envelope import SurfaceEnvelopeReproduction
    from ..geometry.reconstruction._normals import NormalEstimationEvidence
    from ..geometry.reconstruction._poisson import (
        PoissonSolveEvidence,
        SampledSurfaceDeviation,
    )
    from ..geometry.reconstruction._robustness import (
        ComponentFitEvidence,
        OutlierRemovalEvidence,
        SamplingCoverageEvidence,
        ThinFeatureEvidence,
    )
    from ..geometry.simplicial._regions import MeshRegion, PlanarMeshRegion
    from ..geometry.surface._contracts import (
        SurfaceInterface,
        SurfaceMetadata,
        SurfaceMetadataArchiveRecipe,
        SurfaceOrientationRepair,
        SurfaceSelection,
    )
    from ..geometry.surface._model import SurfaceModel
    from ..graph._distributed_partition import DistributedGraphPartitionResult
    from ..graph._partition import GraphPartitionPlan
    from ..lifecycle._composition_rebind import (
        Composition,
        CompositionDependency,
        CompositionEntry,
    )
    from ..linalg._complexes import ComplexMap, HilbertComplex
    from ..linalg._materialization import MaterializationPolicy
    from ..linalg._operator_pairing import OperatorPairing, OperatorPairingArchiveRecipe
    from ..linalg._pairings import DiagonalPairing, EuclideanPairing
    from ..linalg._policies import (
        DifferentiationPolicy,
        FailurePolicy,
        LinearDerivativeSolvePolicy,
        LinearSolvePolicy,
        PCG,
        RankPolicy,
        SolveResourcePolicy,
        TolerancePolicy,
    )
    from ..linalg._preconditioning import (
        JacobiPreconditionerBuilder,
        PreconditioningPolicy,
    )
    from ..linalg._properties import OperatorCapabilities, OperatorProperties
    from ..linalg._spaces import ArraySpace, ArraySpaceArchiveRecipe
    from ..meshing._association import (
        GeometryAssociation,
        GeometryAssociationArchiveRecipe,
        GeometryAssociationKind,
        GeometryAssociationProvenance,
        GeometrySourceEntityRole,
    )
    from ..meshing._certification import MeshCertificationReport
    from ..meshing._certification_inputs import MeshCertificationInputs
    from ..meshing._certification_schedule import MeshCertificationSchedule
    from ..meshing._controls import (
        BackgroundMetricControl,
        BackgroundMetricMode,
        BoundaryLayerCollisionPolicy,
        BoundaryLayerControl,
        BoundaryLayerCornerPolicy,
        BoundaryLayerRoute,
        FeatureKind,
        HoleSeed,
        LayerSchedule,
        PatchControl,
        PeriodicConstraint,
        ProtectedFeature,
        RegionControl,
        RegionSeed,
    )
    from ..meshing._metric import MeshMetricField
    from ..meshing._organization import (
        MeshAttribute,
        MeshAttributeRole,
        MeshLabel,
        MeshPatch,
        MeshZone,
        MeshZoneRole,
        RegionBoundaryEvidence,
        RegionRole,
    )
    from ..meshing._scope import MeshingEntityKind, MeshingScope
    from ..meshing._sizing import (
        CurvatureSizeControl,
        ProximitySizeControl,
        SizeCombinationPolicy,
        SizeCompliancePolicy,
        SizeControlStrength,
        UniformSizeControl,
    )
    from ..meshing._surface_boundary_sources import (
        SurfaceChartBoundarySource,
        SurfaceSubdivisionBoundarySource,
    )
    from ..solver._finite_element_schedule import (
        FiniteElementAcceptedElementRecipe,
        FiniteElementAcceptedFieldLayoutRecipe,
        FiniteElementAcceptedState,
        FiniteElementFieldTransferSemantics,
        FiniteElementRestartManifest,
        FiniteElementRestartStateAssociation,
        FiniteElementRestartStateLayout,
        FiniteElementRestartStateRole,
    )
    from ..solver._finite_volume_content import FiniteVolumeConservativeContentState
    from ..solver._finite_volume_runtime import FiniteVolumeOwnerLocalCellStateBankSpec
    from ..sparse._linear import SparseLinearMap
    from ..sparse._relation import EdgeRelation
    from ..units._dimension import DimensionSignature
    from ..units._unit import UnitDefinition
    from ._adaptation import MeshAdaptationStatus
    from ._archive_layer_columns import (
        ExactExpressionRowsArchiveRecipe,
        ImplicitChartRowsArchiveRecipe,
        LayerRecordArchiveRecipe,
    )
    from ._assembly import MeshCarrierKind, MeshPart
    from ._audit import CellMeshAuditDisposition, CellMeshAuditPolicy, CellMeshAuditReport
    from ._boundary_layer import BoundaryLayerEvidence
    from ._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        MeshingCapability,
        MeshingDerivativeMode,
        MeshingExecutionMode,
        MeshingOperation,
        MeshingProviderInfo,
        MeshingSourceKind,
        MeshQualityTarget,
        SurfaceMeshingSpec,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )
    from ._curving import _ExactCoordinateBankMap
    from ._device_adaptation import (
        OwnerLocalPeriodicAdaptiveResult,
        OwnerLocalPeriodicCellMetadata,
        OwnerLocalPeriodicSizeEvidence,
        OwnerLocalPeriodicSizeStatistic,
    )
    from ._distribution import (
        MeshDistribution,
        MeshPartitionEvidence,
        MeshPartitionKind,
        MeshPartitionPolicy,
        OwnerLocalGraphMigrationEvidence,
        OwnerLocalGraphPartitionEvidence,
        ShardedOwnerDirectoryData,
        ShardedOwnerDirectoryWitness,
    )
    from ._distribution_forest import PeriodicSimplexForest, SimplexForestRecords
    from ._distribution_migration import OwnerLocalMigrationState
    from ._dual_extraction import _RestrictedCoordinateMap
    from ._hex_cut_generation import _CutCoordinateMap
    from ._hex_cut_preparation import BalancedCutGridPlan
    from ._hex_frame_records import HexFrameIntegerLayout
    from ._hex_schedule import NativeHexGridSchedule
    from ._layer_curving_records import (
        LayerColumnProjectionIdentity,
        LayerCurvingImplicitRootTrace,
        LayerCurvingMappedChartTrace,
        LayerCurvingProjectionTrace,
        LayerCurvingSourceEvidence,
    )
    from ._lineage import (
        CellMeshTransition,
        EntityLineage,
        EntityLineageKind,
        MeshLineage,
        MeshTransitionKind,
        VertexInterpolationStencil,
    )
    from ._periodic import PeriodicPointOrbits
    from ._plc_source import PiecewiseLinearComplex
    from ._quality import (
        CellQualityEvaluation,
        CellQualityReport,
        CellQualityReportArchiveRecipe,
    )
    from ._result import (
        CellMeshingResult,
        MeshingComplianceReport,
        MeshingRuntimeInfo,
    )
    from ._trace import (
        MeshingDiagnostic,
        MeshingDiagnosticSeverity,
        MeshingStageKind,
        MeshingStageReport,
        MeshingStageStatus,
        MeshingTrace,
    )
    from .providers._native_periodic import NativePeriodicSource

    types = (
        ("phydrax.geometry", CompiledGeometry),
        ("phydrax.geometry", GeometryTolerance),
        ("phydrax.geometry.design", DesignState),
        ("phydrax.geometry.design", ParameterBinding),
        ("phydrax.geometry.design", ParameterId),
        ("phydrax.geometry.design", ParameterSchema),
        ("phydrax.geometry.design", ParameterSpec),
        ("phydrax.geometry.analytic.internal", _BallKernel),
        ("phydrax.geometry.analytic.internal", _TorusKernel),
        ("phydrax.geometry.implicit", AdaptiveImplicitSurfaceArchiveRecipe),
        ("phydrax.geometry.implicit", AdaptiveImplicitSurfacePolicy),
        ("phydrax.geometry.implicit", ImplicitChartPremise),
        ("phydrax.geometry.implicit", ImplicitChartRestrictionEvidence),
        ("phydrax.geometry.implicit", ImplicitChartRows),
        ("phydrax.geometry.reconstruction", ReconstructionReport),
        ("phydrax.geometry.reconstruction", ReconstructedGeometrySourceArchiveRecipe),
        ("phydrax.geometry.reconstruction.internal", _ReconstructedTrianglePatch),
        ("phydrax.geometry.reconstruction.internal", SurfaceEnvelopeReproduction),
        ("phydrax.model.archive", AdaptiveImplicitBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", MeshingDomainArchiveRecipe),
        ("phydrax.model.archive", MeshingDomainBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", NativeImplicitBoundaryQueryArchiveRecipe),
        ("phydrax.model.archive", AnalyticImplicitProfileArchiveRecipe),
        ("phydrax.model.archive", ParametricCurveBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", ReconstructedTriangleBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", SurfaceSourceChartsArchiveRecipe),
        ("phydrax.model.archive", PreparedSurfaceSourceSupportArchiveRecipe),
        ("phydrax.model.archive", SurfaceNativeRestrictionBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", SurfaceChartBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", SurfaceSubdivisionBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", SphereMaterialCellAtlasArchiveRecipe),
        ("phydrax.model.archive", PreparedSphereChartDeformationArchiveRecipe),
        ("phydrax.model.archive", SurfaceEnvelopeSourceArchiveRecipe),
        ("phydrax.model.archive", PiecewiseLinearBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", NativePlcSourceArchiveRecipe),
        ("phydrax.model.archive", NativeMappedHexSourceArchiveRecipe),
        ("phydrax.model.archive", MappedReferenceBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", SparseCoordinateOperatorArchiveRecipe),
        ("phydrax.model.archive", PeriodicFiniteElementTransferArchiveRecipe),
        ("phydrax.model.archive", PartitionOperandArchiveRecipe),
        ("phydrax.model.archive", PartitionPlanArchiveRecipe),
        ("phydrax.model.archive", CADGraphArchiveRecipe),
        ("phydrax.model.archive", CADRevisionArchiveRecipe),
        ("phydrax.model.archive", CADTransactionArchiveRecipe),
        ("phydrax.geometry.brep", BRepTessellationPolicy),
        ("phydrax.geometry.simplicial", PlanarMeshRegion),
        ("phydrax.model.archive", DistributedFiniteElementTransferArchiveRecipe),
        ("phydrax.geometry.reconstruction", NormalEstimationEvidence),
        ("phydrax.geometry.reconstruction", PoissonSolveEvidence),
        ("phydrax.geometry.reconstruction", SampledSurfaceDeviation),
        ("phydrax.geometry.reconstruction", OutlierRemovalEvidence),
        ("phydrax.geometry.reconstruction", SamplingCoverageEvidence),
        ("phydrax.geometry.reconstruction", ThinFeatureEvidence),
        ("phydrax.geometry.reconstruction", ComponentFitEvidence),
        ("phydrax.geometry.reconstruction", ReconstructedTriangleBoundarySource),
        ("phydrax.geometry", BoundaryAtlas),
        ("phydrax.geometry", AssociationCoverageEvidence),
        ("phydrax.geometry", AssociationGraph),
        ("phydrax.geometry", CADOccurrence),
        ("phydrax.geometry", CADRevision),
        ("phydrax.geometry", CADSelectionSet),
        ("phydrax.geometry", CADSelector),
        ("phydrax.geometry", OccurrenceCorrespondence),
        ("phydrax.geometry", OccurrenceCorrespondenceTransaction),
        ("phydrax.geometry.brep", BRepPartitionPatch),
        ("phydrax.geometry.brep", BRepPartitionRegion),
        ("phydrax.geometry.brep", BRepPartitionReport),
        ("phydrax.geometry.brep", BRepPartitionResult),
        ("phydrax.geometry.brep", BRepPartitionResultArchiveRecipe),
        ("phydrax.geometry", BoundaryFrame),
        ("phydrax.geometry", DomainCurveMap),
        ("phydrax.geometry", MeshingDomain),
        ("phydrax.geometry", MeshingDomainBoundarySource),
        ("phydrax.geometry", MeshingDomainCurve),
        ("phydrax.geometry", MeshingDomainRegion),
        ("phydrax.geometry", MeshingSurfacePatch),
        ("phydrax.geometry.internal", _PlacedSurfacePatch),
        ("phydrax.geometry", MappedReferenceDomain),
        ("phydrax.geometry", ParametricCurveBoundarySource),
        ("phydrax.geometry", PatchCurveUse),
        ("phydrax.geometry", PatchPoleUse),
        ("phydrax.geometry", PiecewiseLinearDomain),
        ("phydrax.geometry", SourceBoundaryChartCover),
        ("phydrax.geometry", SourceBoundaryChartWitness),
        ("phydrax.geometry", SourceBoundaryDistance),
        ("phydrax.geometry", SourceBoundaryPointWitness),
        ("phydrax.geometry", SourceBoundarySamples),
        ("phydrax.geometry", SourceFidelityCertificate),
        ("phydrax.geometry", SourceBoundaryDistanceWitness),
        ("phydrax.geometry", SurfaceEnvelopeDistanceWitness),
        ("phydrax.geometry", SurfaceSourceCharts),
        ("phydrax.geometry", DomainCoverageCertificate),
        ("phydrax.geometry", FacetOverlap),
        ("phydrax.geometry", PlcSourcePointWitness),
        ("phydrax.geometry.brep", BRepProjectionResult),
        ("phydrax.geometry.implicit", RestrictedDelaunayBoundaryWitness),
        ("phydrax.geometry.implicit", RestrictedDelaunayCoverageTransportWitness),
        ("phydrax.geometry.simplicial", MeshRegion),
        ("phydrax", SemanticProvenance),
        ("phydrax", SpatialCoordinateContract),
        ("phydrax.units", DimensionSignature),
        ("phydrax.units", UnitDefinition),
        ("phydrax.lifecycle", Composition),
        ("phydrax.lifecycle", CompositionDependency),
        ("phydrax.lifecycle", CompositionEntry),
        ("phydrax.discretization", CellBlock),
        ("phydrax.discretization", EmptyLocalCellMeshAuthority),
        ("phydrax.discretization", CellComplexTopology),
        ("phydrax.discretization", CellGeometryRestrictionSource),
        ("phydrax.discretization", ExactPlcCellGeometrySource),
        ("phydrax.discretization", ExactPowerCellGeometrySource),
        ("phydrax.discretization", ExactPeriodicPowerSourceContext),
        ("phydrax.discretization", MappedPowerCellGeometrySource),
        ("phydrax.discretization", CellGeometrySpec),
        ("phydrax.discretization", OwnerLocalCellBlock),
        ("phydrax.discretization", OwnerLocalCellGeometry),
        ("phydrax.discretization", CellMesh),
        ("phydrax.discretization", CellPartition),
        ("phydrax.discretization", CellValidityCertificate),
        ("phydrax.discretization", CellVertexGeometryElement),
        ("phydrax.discretization", BarycentricCellGeometryElement),
        ("phydrax.discretization", LayerColumnCellGeometryElement),
        ("phydrax.discretization", PerCellRestrictedCellGeometryElement),
        ("phydrax.discretization", PolynomialComposedCellGeometryElement),
        ("phydrax.discretization", RestrictedCellGeometryElement),
        ("phydrax.discretization", SplineCellGeometryElement),
        ("phydrax.meshing", BalancedCutGridPlan),
        ("phydrax.meshing", PiecewiseLinearComplex),
        ("phydrax.meshing.internal", _CutCoordinateMap),
        ("phydrax.meshing", NativeHexGridSchedule),
        ("phydrax.meshing", HexFrameIntegerLayout),
        ("phydrax.meshing.internal", _RestrictedCoordinateMap),
        ("phydrax.meshing.internal", _ExactCoordinateBankMap),
        ("phydrax.discretization", RationalComposedCellGeometryElement),
        ("phydrax.discretization", CellValidityPolicy),
        ("phydrax.discretization", CellGeometryTransition),
        ("phydrax.discretization", CellGeometryTransitionEvidence),
        ("phydrax.discretization", NestedReferenceWitnesses),
        ("phydrax.discretization", ReferenceCellMapWitnesses),
        ("phydrax.discretization", SurfaceChartWitness),
        ("phydrax.discretization", SurfaceChartWitnessArchiveRecipe),
        ("phydrax.discretization", PreparedSurfaceChartDeformation),
        ("phydrax.discretization", PreparedSurfaceChartPiece),
        ("phydrax.discretization", PreparedSurfaceChartOccurrence),
        ("phydrax.discretization", PreparedSphereChartDeformation),
        ("phydrax.discretization", PreparedSphereChartPiece),
        ("phydrax.discretization", PreparedSphereChartOccurrence),
        ("phydrax.geometry", SphereMaterialCellAtlas),
        ("phydrax.geometry", SphereMaterialFidelityCertificate),
        ("phydrax.geometry.internal", _SphereMaterialFidelityBinding),
        ("phydrax.geometry", PreparedSurfaceSourceSupport),
        ("phydrax.geometry", SurfaceSourceRootAtlas),
        ("phydrax.geometry.internal", SurfaceNativeRestrictionBoundarySource),
        ("phydrax.geometry.internal", _SplineSurfaceCoordinateMap),
        ("phydrax.discretization", SurfaceRootMeshCellBinding),
        ("phydrax.discretization", DiscreteSupport),
        ("phydrax.discretization", EntitySet),
        ("phydrax.discretization", EntitySubset),
        ("phydrax.discretization", GridLocation),
        ("phydrax.discretization", IntervalConnectivity),
        ("phydrax.discretization", OrientedIncidence),
        ("phydrax.discretization", PeriodicCell),
        ("phydrax.discretization", PeriodicIsometryGroup),
        ("phydrax.discretization", PeriodicMeshTopology),
        ("phydrax.discretization", HexahedralConnectivity),
        ("phydrax.meshing", PeriodicPointOrbits),
        ("phydrax.discretization", PolygonalConnectivity),
        ("phydrax.discretization", PolyhedralBlock),
        ("phydrax.discretization", HexahedralConnectivity),
        ("phydrax.discretization", PolyhedralConnectivity),
        ("phydrax.discretization", PreparedTensorGrid),
        ("phydrax.discretization", SimplicialConnectivity),
        ("phydrax.discretization", StructuredAxis),
        ("phydrax.discretization", TensorEntityLayout),
        ("phydrax.discretization", TensorTopology),
        ("phydrax.discretization", TetrahedralConnectivity),
        ("phydrax.discretization", AxisDiscretization),
        ("phydrax.discretization", AxisDomain),
        ("phydrax.discretization", DiscreteMeasure),
        ("phydrax.discretization.fem", FiniteElementSpec),
        ("phydrax.discretization.fem", FiniteElementTransferResult),
        ("phydrax.discretization.fem", DistributedFiniteElementTransfer),
        ("phydrax.discretization.fem", FiniteElementTopologyTransfer),
        ("phydrax.discretization.fem", PeriodicFiniteElementTransfer),
        ("phydrax.discretization", DistributedSparseTransfer),
        ("phydrax.discretization", _TransferExchange),
        ("phydrax.discretization.fem", SparseCommutingEvidence),
        ("phydrax.discretization.fem", SparseTransferEntryEvidence),
        ("phydrax.linalg", ArraySpace),
        ("phydrax.linalg", ComplexMap),
        ("phydrax.linalg", HilbertComplex),
        ("phydrax.linalg", EuclideanPairing),
        ("phydrax.linalg", DiagonalPairing),
        ("phydrax.linalg", OperatorCapabilities),
        ("phydrax.linalg", OperatorProperties),
        ("phydrax.linalg", OperatorPairing),
        ("phydrax.linalg._operator_pairing", OperatorPairingArchiveRecipe),
        ("phydrax.linalg._spaces", ArraySpaceArchiveRecipe),
        ("phydrax.linalg._policies", LinearSolvePolicy),
        ("phydrax.linalg._policies", PCG),
        ("phydrax.linalg._policies", TolerancePolicy),
        ("phydrax.linalg._policies", RankPolicy),
        ("phydrax.linalg._policies", DifferentiationPolicy),
        ("phydrax.linalg._policies", FailurePolicy),
        ("phydrax.linalg._policies", LinearDerivativeSolvePolicy),
        ("phydrax.linalg._policies", SolveResourcePolicy),
        ("phydrax.linalg._materialization", MaterializationPolicy),
        ("phydrax.linalg._preconditioning", PreconditioningPolicy),
        ("phydrax.linalg._preconditioning", JacobiPreconditionerBuilder),
        ("phydrax.equations", MaterialSiteId),
        ("phydrax.equations", MaterialState),
        ("phydrax.equations", MaterialTransaction),
        ("phydrax.exterior", FormType),
        ("phydrax.exterior", FormValueSpec),
        ("phydrax.meshing", CellMeshAuditReport),
        ("phydrax.geometry.surface", SurfaceInterface),
        ("phydrax.geometry.surface", SurfaceMetadata),
        ("phydrax.geometry.surface", SurfaceMetadataArchiveRecipe),
        ("phydrax.geometry.surface", SurfaceModel),
        ("phydrax.geometry.surface", SurfaceOrientationRepair),
        ("phydrax.geometry.surface", SurfaceSelection),
        ("phydrax.meshing", CellMeshAuditDisposition),
        ("phydrax.meshing", CellMeshAuditPolicy),
        ("phydrax", PredicateMode),
        ("phydrax.geometry", CommonRefinementCoverage),
        ("phydrax.geometry", CommonRefinementEvidence),
        ("phydrax.geometry", CommonRefinementPolicy),
        ("phydrax.geometry", CommonRefinementStatus),
        ("phydrax.geometry", CommonRootVolumeWitness),
        ("phydrax.geometry", CurvedOverlapIntegrationEvidence),
        ("phydrax.geometry", CurvedOverlapIntegrationPolicy),
        ("phydrax.geometry", EmbeddingCounts),
        ("phydrax.geometry._embedding_state", PeriodicAffineTorusDegreeEvidence),
        ("phydrax.geometry._embedding_state", PeriodicMappedTorusDegreeEvidence),
        ("phydrax.geometry._embedding_state", PeriodicRestrictionPartitionEvidence),
        ("phydrax.geometry._periodic_restriction", PeriodicRestrictionPremise),
        (
            "phydrax.geometry._periodic_mapped_restriction",
            PeriodicMappedRestrictionEvidence,
        ),
        ("phydrax.geometry", GlobalEmbeddingCertificate),
        ("phydrax.geometry", MaterialMapActionEvidence),
        ("phydrax.geometry", MaterialReferenceDeclaration),
        ("phydrax.geometry", MaterialReferencePolicy),
        ("phydrax.geometry", MaterialReferenceVolumeWitness),
        ("phydrax.geometry", MaterialReferenceVolumeOverlap),
        ("phydrax.geometry", MaterialSourceMapCapability),
        ("phydrax.geometry", MeshCertificateBinding),
        ("phydrax.meshing", LayerColumnProjectionIdentity),
        ("phydrax.meshing", LayerCurvingProjectionTrace),
        ("phydrax.meshing", LayerCurvingImplicitRootTrace),
        ("phydrax.meshing", LayerCurvingMappedChartTrace),
        ("phydrax.meshing", LayerCurvingSourceEvidence),
        ("phydrax.meshing", BoundaryLayerCollisionPolicy),
        ("phydrax.meshing", BoundaryLayerEvidence),
        ("phydrax.meshing", SimplexForestRecords),
        ("phydrax.meshing", PeriodicSimplexForest),
        ("phydrax.meshing", OwnerLocalMigrationState),
        ("phydrax.meshing", CellMeshTransition),
        ("phydrax.meshing", EntityLineage),
        ("phydrax.meshing", EntityLineageKind),
        ("phydrax.meshing", MeshLineage),
        ("phydrax.meshing", MeshTransitionKind),
        ("phydrax.meshing", VertexInterpolationStencil),
        ("phydrax.meshing", MeshAdaptationStatus),
        ("phydrax.meshing.internal", OwnerLocalPeriodicAdaptiveResult),
        ("phydrax.meshing.internal", OwnerLocalPeriodicCellMetadata),
        ("phydrax.meshing.internal", OwnerLocalPeriodicSizeEvidence),
        ("phydrax.meshing.internal", OwnerLocalPeriodicSizeStatistic),
        ("phydrax.geometry", MeshCertificateFinding),
        ("phydrax.geometry", MeshEmbeddingWitness),
        ("phydrax.geometry", MeshEmbeddingWitnessArchiveRecipe),
        ("phydrax.model.archive", StaticColumnArchiveRecipe),
        ("phydrax.model.archive", TextColumnArchiveRecipe),
        ("phydrax.model.archive", TextRowsArchiveRecipe),
        ("phydrax.model.archive", IntegerRowsArchiveRecipe),
        ("phydrax.model.archive", ExactTensorArchiveRecipe),
        ("phydrax.model.archive", ExactExpressionRowsArchiveRecipe),
        ("phydrax.model.archive", ImplicitChartRowsArchiveRecipe),
        ("phydrax.model.archive", LayerRecordArchiveRecipe),
        ("phydrax.geometry", PreparedCommonRefinement),
        ("phydrax.geometry", EulerianVolumeOverlapChain),
        ("phydrax.geometry", EulerianVolumeOverlapMap),
        ("phydrax.geometry", VolumeMaterialChain),
        ("phydrax.geometry", VolumeMaterialMap),
        ("phydrax.geometry", VolumeMaterialDeformation),
        ("phydrax.meshing", CellMeshingResult),
        ("phydrax.meshing", MeshCertificationInputs),
        ("phydrax.meshing", MeshCertificationSchedule),
        ("phydrax.meshing", MeshCertificationReport),
        ("phydrax.meshing", ProtectedFeature),
        ("phydrax.meshing", MeshingEntityKind),
        ("phydrax.meshing", MeshingScope),
        ("phydrax.geometry", PlanarEmbedding),
        ("phydrax.meshing", BackgroundMetricControl),
        ("phydrax.meshing", BackgroundMetricMode),
        ("phydrax.meshing", BoundaryLayerControl),
        ("phydrax.meshing", BoundaryLayerCornerPolicy),
        ("phydrax.meshing", BoundaryLayerRoute),
        ("phydrax.meshing", CellFamilyPolicy),
        ("phydrax.meshing", CellMeshingTarget),
        ("phydrax.meshing", CurvatureSizeControl),
        ("phydrax.meshing", FeatureKind),
        ("phydrax.meshing", HoleSeed),
        ("phydrax.meshing", LayerSchedule),
        ("phydrax.meshing", MeshMetricField),
        ("phydrax.meshing", MeshQualityTarget),
        ("phydrax.meshing", PatchControl),
        ("phydrax.meshing", PeriodicConstraint),
        ("phydrax.meshing", ProximitySizeControl),
        ("phydrax.meshing", RegionControl),
        ("phydrax.meshing", RegionRole),
        ("phydrax.meshing", RegionSeed),
        ("phydrax.meshing", SizeCombinationPolicy),
        ("phydrax.meshing", SizeCompliancePolicy),
        ("phydrax.meshing", SizeControlStrength),
        ("phydrax.meshing", SurfaceMeshingSpec),
        ("phydrax.meshing", UniformSizeControl),
        ("phydrax.meshing", VolumeFillStrategy),
        ("phydrax.meshing", VolumeMeshingSpec),
        ("phydrax.meshing", GeometryAssociation),
        ("phydrax.meshing", GeometryAssociationArchiveRecipe),
        ("phydrax.meshing", GeometryAssociationKind),
        ("phydrax.meshing", GeometryAssociationProvenance),
        ("phydrax.meshing", GeometrySourceEntityRole),
        ("phydrax.meshing", MeshAttribute),
        ("phydrax.meshing", MeshAttributeRole),
        ("phydrax.meshing", MeshLabel),
        ("phydrax.meshing", MeshPatch),
        ("phydrax.meshing", MeshZone),
        ("phydrax.meshing", MeshZoneRole),
        ("phydrax.meshing", RegionBoundaryEvidence),
        ("phydrax.meshing", SurfaceChartBoundarySource),
        ("phydrax.meshing", SurfaceSubdivisionBoundarySource),
        ("phydrax.geometry.reconstruction.internal", _TriangleBVHNode),
        ("phydrax.meshing", CellQualityEvaluation),
        ("phydrax.meshing", CellQualityReport),
        ("phydrax.meshing", CellQualityReportArchiveRecipe),
        ("phydrax.meshing", MeshCarrierKind),
        ("phydrax.meshing", MeshDistribution),
        ("phydrax.meshing", ShardedOwnerDirectoryWitness),
        ("phydrax.meshing", ShardedOwnerDirectoryData),
        ("phydrax.meshing", MeshPartitionEvidence),
        ("phydrax.meshing", MeshPartitionKind),
        ("phydrax.meshing", MeshPartitionPolicy),
        ("phydrax.meshing", OwnerLocalGraphMigrationEvidence),
        ("phydrax.meshing", OwnerLocalGraphPartitionEvidence),
        ("phydrax.meshing", MeshPart),
        ("phydrax.meshing", MeshingComplianceReport),
        ("phydrax.meshing", MeshingCapability),
        ("phydrax.meshing", MeshingDerivativeMode),
        ("phydrax.meshing", MeshingDiagnostic),
        ("phydrax.meshing", MeshingDiagnosticSeverity),
        ("phydrax.meshing", MeshingExecutionMode),
        ("phydrax.meshing", MeshingOperation),
        ("phydrax.meshing", MeshingProviderInfo),
        ("phydrax.meshing", MeshingSourceKind),
        ("phydrax.meshing", MeshingRuntimeInfo),
        ("phydrax.meshing", MeshingStageKind),
        ("phydrax.meshing", NativePeriodicSource),
        ("phydrax.graph", GraphPartitionPlan),
        ("phydrax.graph", DistributedGraphPartitionResult),
        ("phydrax.meshing", MeshingStageReport),
        ("phydrax.meshing", MeshingStageStatus),
        ("phydrax.meshing", MeshingTrace),
        ("phydrax.solver", FiniteElementAcceptedState),
        ("phydrax.discretization.fem", FiniteElementDofOwnership),
        ("phydrax.solver", FiniteElementRestartManifest),
        ("phydrax.solver", FiniteElementRestartStateAssociation),
        ("phydrax.solver", FiniteElementRestartStateLayout),
        ("phydrax.solver", FiniteElementRestartStateRole),
        ("phydrax.solver", FiniteElementAcceptedStateArchiveRecipe),
        ("phydrax.solver", FiniteElementFieldTransferSemantics),
        ("phydrax.solver", FiniteElementAcceptedFieldLayoutRecipe),
        ("phydrax.solver", FiniteElementAcceptedElementRecipe),
        ("phydrax.solver", FiniteVolumeConservativeContentState),
        ("phydrax.solver", FiniteVolumeOwnerLocalCellStateBankSpec),
        ("phydrax.solver", FiniteVolumeRuntimeStateArchiveRecipe),
        ("phydrax.solver", FiniteVolumeOwnerLocalCellStateBankArchiveRecipe),
        ("phydrax.solver", FiniteVolumeOversetPartSpecArchiveRecipe),
        ("phydrax.solver", FiniteVolumeOversetPolicyArchiveRecipe),
        ("phydrax.solver", FiniteVolumeOwnerLocalRepartitionContextArchiveRecipe),
        ("phydrax.discretization", FiniteVolumePrecisionPolicy),
        (
            "phydrax.discretization.finite_volume",
            OwnerLocalConservativeRemapEvidence,
        ),
        (
            "phydrax.discretization.finite_volume",
            PreparedOwnerLocalConservativeRemap,
        ),
        ("phydrax.sparse", EdgeRelation),
        ("phydrax.sparse", SparseLinearMap),
        ("phydrax.geometry.design", SketchContinuationReferenceArchiveRecipe),
        ("phydrax.geometry.design", TwoCircleConstruction),
        ("phydrax.geometry.design", LineCircleConstruction),
        ("phydrax.geometry.design", _SketchDistanceRigidityWitness),
        ("phydrax.geometry.design", _SketchDistanceTriangleWitness),
    )
    for namespace, value in types:
        try:
            artifact_value_id(value)
        except TypeError:
            register_artifact_value(f"{namespace}:{value.__name__}", value)

    # B-Rep and T-spline classes are owned and registered by their CAD codec.
    from ..interchange._cad_archive import _register_brep_types

    _register_brep_types()


def restore_model_archive_value(
    value: Any,
    /,
    *,
    execution_group: Any | None = None,
) -> Any:
    """Rebuild projected FE/linalg models through their canonical constructors."""
    import inspect
    from collections.abc import Mapping

    import jax
    import numpy as np

    from .._model._structure import _FiniteElementSpecArchiveProjection
    from ..discretization._cell_mesh import CellMesh
    from ..discretization._mapped_power_geometry import MappedPowerCellGeometrySource
    from ..discretization.fem._de_rham import (
        _distribute_transfer_result,
        DistributedFiniteElementTransfer,
        FiniteElementTransferResult,
        PeriodicFiniteElementTransfer,
        SparseCommutingEvidence,
        SparseTransferEntryEvidence,
    )
    from ..geometry._cad_revision import (
        _restore_association_graph,
        _restore_cad_selection,
        AssociationCoverageEvidence,
        AssociationGraph,
        CADOccurrence,
        CADRevision,
        CADSelectionSet,
        CADSelector,
        OccurrenceCorrespondence,
        OccurrenceCorrespondenceTransaction,
    )
    from ..geometry._common_root_volume_overlap import (
        CurvedOverlapIntegrationPolicy,
    )
    from ..geometry._planar_embedding import PlanarEmbedding
    from ..geometry.brep._constructors import BRepTessellationPolicy
    from ..geometry.brep._model import BRepEntityId, BRepModel
    from ..geometry.brep._partition import (
        BRepPartitionOperand,
        BRepPartitionPatch,
        BRepPartitionPlan,
        BRepPartitionPolicy,
        BRepPartitionRegion,
        BRepPartitionReport,
        BRepPartitionResult,
        BRepPartitionResultArchiveRecipe,
        BRepPartitionRole,
        cad_revision_from_brep_model,
        partition_brep,
    )
    from ..geometry.brep._planar import (
        partition_planar,
        PlanarPartitionOperand,
        PlanarPartitionPlan,
    )
    from ..geometry.simplicial._regions import PlanarMeshRegion
    from ..interchange._cad_archive import _reconstruct_brep_graph
    from ..lifecycle._composition_rebind import (
        Composition,
        CompositionDependency,
        CompositionEntry,
    )
    from ..linalg._complexes import ComplexMap, HilbertComplex
    from ..linalg._operator_pairing import OperatorPairingArchiveRecipe
    from ..linalg._pairings import DiagonalPairing, EuclideanPairing
    from ..linalg._policies import LinearSolvePolicy
    from ..linalg._preconditioning import PreconditioningPolicy
    from ..linalg._properties import OperatorProperties
    from ..linalg._spaces import ArraySpaceArchiveRecipe
    from ..sparse._linear import SparseCoordinateOperator
    from ..sparse._relation import EdgeRelation
    from ._distribution_forest import PeriodicSimplexForest, SimplexForestRecords
    from ._distribution_migration import OwnerLocalMigrationState

    _register_model_archive_types()
    in_composition = False
    pending_graphs: list[Any] = []
    pending_selections: list[Any] = []

    def same_content(left: Any, right: Any) -> bool:
        left = model_archive_root_projection(left)
        right = model_archive_root_projection(right)
        left_recipe = model_structure_recipe(left)
        if left_recipe != model_structure_recipe(right):
            return False
        left_arrays = pack_model_array_tree(
            left, left_recipe, prefix="compare", limits=MODEL_ARCHIVE_LIMITS
        )
        right_arrays = pack_model_array_tree(
            right, left_recipe, prefix="compare", limits=MODEL_ARCHIVE_LIMITS
        )
        return left_arrays.keys() == right_arrays.keys() and all(
            np.array_equal(left_arrays[key], right_arrays[key]) for key in left_arrays
        )

    def restore_partition_plan(stored: PartitionPlanArchiveRecipe) -> Any:
        if type(stored) is not PartitionPlanArchiveRecipe:
            raise TypeError("Partition archive requires its operation plan recipe.")
        contract = _reconstruct_brep_graph(stored.coordinate_contract)
        planar = stored.topological_dimension == 2
        if not planar and stored.topological_dimension != 3:
            raise ValueError("Unsupported partition plan dimension.")
        solid_operands: list[BRepPartitionOperand] = []
        planar_operands: list[PlanarPartitionOperand] = []
        for operand in stored.operands:
            if type(operand) is not PartitionOperandArchiveRecipe:
                raise TypeError("Invalid partition operand recipe.")
            source = operand.source
            if type(source) is BRepModel:
                source = _reconstruct_brep_graph(source)
            elif planar and type(source) is PlanarMeshRegion:
                edges = np.asarray(source.edges)
                offsets = np.asarray(source.loop_offsets)
                if (
                    offsets.ndim != 1
                    or len(offsets) < 2
                    or offsets[0] != 0
                    or offsets[-1] != len(edges)
                    or np.any(np.diff(offsets) < 3)
                    or edges.ndim != 2
                    or edges.shape[1] != 2
                ):
                    raise ValueError("Invalid stored planar mesh loops.")
                loops = tuple(
                    [int(index) for index in edges[int(a) : int(b), 0]]
                    for a, b in zip(offsets[:-1], offsets[1:], strict=True)
                )
                source = PlanarMeshRegion(
                    source.vertices, loops, feature_id=source.feature_id
                )
                if not same_content(source, operand.source):
                    raise ValueError("Stored planar mesh differs from its constructor.")
            else:
                raise TypeError("Unsupported partition operand source.")
            if operand.selection is not None and not isinstance(
                operand.selection, CADSelectionSet
            ):
                raise TypeError("Invalid stored partition selection.")
            selection = (
                None
                if operand.selection is None
                else _restore_cad_selection(operand.selection)
            )
            if not isinstance(operand.role, str):
                raise TypeError("Invalid stored partition role.")
            role = BRepPartitionRole(operand.role)
            if planar:
                if not isinstance(source, (BRepModel, PlanarMeshRegion)):
                    raise TypeError("Invalid planar partition source.")
                planar_operands.append(
                    PlanarPartitionOperand(
                        operand.operand_id,
                        source,
                        role,
                        selection,
                        operand.target_region_ids,
                    )
                )
            else:
                if not isinstance(source, BRepModel):
                    raise TypeError("Invalid solid partition source.")
                solid_operands.append(
                    BRepPartitionOperand(
                        operand.operand_id,
                        source,
                        role,
                        selection,
                        operand.target_region_ids,
                    )
                )
        policy = BRepPartitionPolicy(stored.precedence)
        if planar:
            e = stored.embedding
            if type(e) is not PlanarEmbedding:
                raise TypeError("Planar plan requires its embedding.")
            embedding = PlanarEmbedding(e.origin, e.x_axis, e.y_axis, e.normal)
            if embedding != e:
                raise ValueError("Stored planar embedding differs from its constructor.")
            plan = PlanarPartitionPlan(
                contract, embedding, tuple(planar_operands), policy
            )
        else:
            if stored.embedding is not None:
                raise ValueError("Solid partition cannot declare a planar embedding.")
            plan = BRepPartitionPlan(contract, tuple(solid_operands), policy)
        if plan.plan_id != stored.plan_id:
            raise ValueError("Stored partition plan identity does not reproduce.")
        return plan

    def rebuild_policy(policy: Any) -> Any:
        if policy is None or isinstance(policy, (bool, int, float, str, bytes, np.dtype)):
            return policy
        if isinstance(policy, tuple):
            items = tuple(rebuild_policy(item) for item in policy)
            return type(policy)(*items) if "_fields" in vars(type(policy)) else items
        if isinstance(policy, list):
            return [rebuild_policy(item) for item in policy]
        if isinstance(policy, Mapping):
            return {
                rebuild_policy(key): rebuild_policy(item) for key, item in policy.items()
            }
        if type(policy) is PreconditioningPolicy:
            if policy.preconditioner is not None:
                source = restore(policy.preconditioner)
            else:
                source = rebuild_policy(policy.builder)
            setup_operator = restore(policy.setup_operator)
            rebuilt = PreconditioningPolicy(
                source,
                setup_operator=setup_operator,
                side=policy.side,
                refresh=policy.refresh_policy,
            )
        elif dataclasses.is_dataclass(policy) and not isinstance(policy, type):
            cls = type(policy)
            stored_fields = {
                field.name: rebuild_policy(object.__getattribute__(policy, field.name))
                for field in dataclasses.fields(policy)
            }
            signature = inspect.signature(cls.__init__)
            positional: list[Any] = []
            keywords: dict[str, Any] = {}
            for parameter in tuple(signature.parameters.values())[1:]:
                if parameter.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if parameter.name not in stored_fields:
                    if parameter.default is inspect.Parameter.empty:
                        raise TypeError(
                            f"Policy constructor {cls.__qualname__} has an "
                            f"unbound required parameter {parameter.name!r}."
                        )
                    continue
                item = stored_fields[parameter.name]
                if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                    positional.append(item)
                else:
                    keywords[parameter.name] = item
            rebuilt = cls(*positional, **keywords)
            if model_structure_recipe(rebuilt) != model_structure_recipe(policy):
                raise ValueError(
                    f"Stored {cls.__qualname__} fields differ from its constructor."
                )
        else:
            return policy
        return rebuilt

    def restore(item: Any, /) -> Any:
        nonlocal in_composition
        if type(item) in (
            CADOccurrence,
            CADRevision,
            CADSelector,
            OccurrenceCorrespondence,
            OccurrenceCorrespondenceTransaction,
            BRepPartitionPatch,
            BRepPartitionRegion,
            BRepPartitionReport,
            BRepPartitionResult,
            PartitionOperandArchiveRecipe,
            PartitionPlanArchiveRecipe,
            AssociationCoverageEvidence,
        ):
            raise ValueError(
                "CAD partition components require an owning partition replay."
            )
        if type(item) is AssociationGraph:
            if not in_composition:
                raise ValueError(
                    "CAD association graph requires an owning partition replay."
                )
            graph = _restore_association_graph(item)
            pending_graphs.append(graph)
            return graph
        if type(item) is CADSelectionSet:
            if not in_composition:
                raise ValueError("CAD selection requires an owning partition replay.")
            selection = _restore_cad_selection(item)
            pending_selections.append(selection)
            return selection
        if type(item) is BRepPartitionResultArchiveRecipe:
            from .._model._structure import model_archive_projection

            plan = restore_partition_plan(item.plan)
            if not same_content(model_archive_projection(plan), item.plan):
                raise ValueError("Stored partition plan differs from its constructors.")
            policy = item.tessellation_policy
            if policy is not None:
                if type(policy) is not BRepTessellationPolicy:
                    raise TypeError("Invalid partition tessellation policy.")
                policy = BRepTessellationPolicy(
                    linear_deflection=policy.linear_deflection,
                    angular_deflection=policy.angular_deflection,
                    trim_samples_per_edge=policy.trim_samples_per_edge,
                )
                if not same_content(policy, item.tessellation_policy):
                    raise ValueError(
                        "Stored tessellation policy differs from its constructor."
                    )
            replay = (
                partition_planar(plan, tessellation=policy)
                if type(plan) is PlanarPartitionPlan
                else partition_brep(plan, tessellation=policy)
            )
            model = _reconstruct_brep_graph(item.model)
            graph = _restore_association_graph(item.association_graph)
            if (
                cad_revision_from_brep_model(model) != graph.target_revision
                or item.revision != graph.target_revision
            ):
                raise ValueError(
                    "Partition history target differs from its restored B-Rep."
                )

            def entity(value: BRepEntityId) -> BRepEntityId:
                return BRepEntityId(
                    value.source_id,
                    value.source_revision,
                    value.kind,
                    value.index,
                    value.occurrence_path,
                )

            regions = tuple(
                BRepPartitionRegion(
                    value.name,
                    tuple(entity(x) for x in value.entity_ids),
                    value.topological_dimension,
                )
                for value in item.regions
            )
            patches = tuple(
                BRepPartitionPatch(
                    value.name,
                    tuple(entity(x) for x in value.entity_ids),
                    value.adjacent_region_ids,
                    value.topological_dimension,
                )
                for value in item.patches
            )
            report = BRepPartitionReport(
                **{
                    field.name: getattr(item.report, field.name)
                    for field in dataclasses.fields(BRepPartitionReport)
                    if field.init
                }
            )
            operand_graphs = tuple(
                (name, _restore_association_graph(value))
                for name, value in item.operand_association_graphs
            )
            if (
                not same_content(model, replay.model)
                or not same_content(graph, replay.association_graph)
                or not same_content(operand_graphs, replay.operand_association_graphs)
                or not same_content(regions, replay.regions)
                or not same_content(patches, replay.patches)
                or not same_content(report, replay.report)
                or item.revision != replay.revision
            ):
                raise ValueError(
                    "Stored partition declarations differ from operation replay."
                )
            rebuilt = BRepPartitionResult(
                model,
                graph.target_revision,
                graph,
                regions,
                patches,
                report,
                plan,
                item.topological_dimension,
                operand_association_graphs=operand_graphs,
                tessellation_policy=policy,
            )
            if model_structure_recipe(rebuilt) != model_structure_recipe(item):
                raise ValueError("Stored partition result changed during reconstruction.")
            return rebuilt
        if type(item) is SimplexForestRecords:
            forest = SimplexForestRecords(
                item.cell_ids,
                item.cell_owner_ranks,
                dict(item.cell_fields),
                item.vertex_ids,
                item.vertex_owner_ranks,
                dict(item.vertex_fields),
                item.protected_edge_ids,
                item.dimension,
                item.ambient_dimension,
                item.next_vertex_id,
                item.next_cell_id,
            )
            if forest.forest_id != item.forest_id:
                raise ValueError("Stored simplex forest identity does not reproduce.")
            return forest
        if type(item) is PeriodicSimplexForest:
            forest = PeriodicSimplexForest(
                restore(item.records),
                item.vertex_source_ids,
                item.vertex_absolute_windings,
                item.exact_vertex_coordinates,
                item.block_names,
                item.source_id,
                item.source_revision,
                item.periodic_cell_id,
            )
            if forest.forest_id != item.forest_id:
                raise ValueError("Stored periodic forest identity does not reproduce.")
            return forest
        if type(item) is OwnerLocalMigrationState:
            state = OwnerLocalMigrationState(
                restore(item.forest),
                cell_payloads=dict(item.cell_payloads),
                vertex_payloads=dict(item.vertex_payloads),
            )
            if state.state_id != item.state_id:
                raise ValueError(
                    "Stored owner-local migration identity does not reproduce."
                )
            return state
        if type(item) is SparseCoordinateOperatorArchiveRecipe:
            if not isinstance(item.accumulation_dtype, np.dtype):
                raise TypeError("Stored sparse accumulation dtype must be a NumPy dtype.")
            relation = restore(item.relation)
            source = restore(item.source)
            target = restore(item.target)
            properties = restore(item.properties)
            operator = SparseCoordinateOperator(
                relation,
                np.asarray(item.coefficients),
                source=source,
                target=target,
                properties=properties,
                operator_id=item.operator_id,
                accumulation_dtype=item.accumulation_dtype,
                block_shape=item.block_shape,
            )
            if (
                operator.operator_id != item.operator_id
                or operator.accumulation_dtype != item.accumulation_dtype
                or operator.block_shape != item.block_shape
                or not np.array_equal(
                    np.asarray(operator.coefficients), np.asarray(item.coefficients)
                )
            ):
                raise ValueError("Stored sparse operator differs from its owner fields.")
            return operator
        if type(item) is _FiniteElementSpecArchiveProjection:
            return item.rebuild()
        if type(item) is ArraySpaceArchiveRecipe:
            pairing = restore(item.pairing)
            return item.rebuild(pairing)
        if type(item) is OperatorPairingArchiveRecipe:
            operator = restore(item.operator)
            policy = rebuild_policy(restore(item.solve_policy))
            if type(policy) is not LinearSolvePolicy:
                raise TypeError("Stored Riesz recipe lost its LinearSolvePolicy.")
            return item.rebuild(operator, policy)
        if type(item) is EdgeRelation:
            relation = EdgeRelation(
                item.source_indices,
                item.target_indices,
                source_size=item.source_size,
                target_size=item.target_size,
                valid=item.valid,
            )
            if (
                relation.source_size != item.source_size
                or relation.target_size != item.target_size
                or not np.array_equal(relation.source_indices, item.source_indices)
                or not np.array_equal(relation.target_indices, item.target_indices)
                or not np.array_equal(relation.valid, item.valid)
            ):
                raise ValueError("Stored sparse relation differs from its exact rows.")
            return relation
        if type(item) is OperatorProperties:
            properties = OperatorProperties(
                diagonal=item.diagonal,
                triangular=item.triangular,
                self_adjoint=item.self_adjoint,
                positive_definite=item.positive_definite,
                positive_semidefinite=item.positive_semidefinite,
                block_diagonal=item.block_diagonal,
                rank=item.rank,
                evidence=dict(item.evidence),
            )
            if model_structure_recipe(properties) != model_structure_recipe(item):
                raise ValueError("Stored operator properties changed on restoration.")
            return properties
        if type(item) is EuclideanPairing:
            pairing = EuclideanPairing()
            if pairing.pairing_id != item.pairing_id:
                raise ValueError("Stored Euclidean pairing identity changed.")
            return pairing
        if type(item) is DiagonalPairing:
            pairing = DiagonalPairing(item.weights, pairing_id=item.pairing_id)
            if pairing.pairing_id != item.pairing_id or not np.array_equal(
                pairing.weights, item.weights
            ):
                raise ValueError("Stored diagonal pairing differs from its weights.")
            return pairing
        if type(item) is LinearSolvePolicy:
            return rebuild_policy(item)
        if type(item) is CellMesh:
            return CellMesh.from_stored_fields(item)
        if type(item) is MappedPowerCellGeometrySource:
            return MappedPowerCellGeometrySource.from_stored_fields(item)
        if type(item) is HilbertComplex:
            spaces = tuple(restore(space) for space in item.spaces)
            differentials = tuple(restore(value) for value in item.differentials)
            rebuilt = HilbertComplex(
                spaces,
                differentials,
                complex_id=item.complex_id,
            )
            if rebuilt.complex_id != item.complex_id:
                raise ValueError("Stored Hilbert complex identity changed.")
            return rebuilt
        if type(item) is ComplexMap:
            rebuilt = ComplexMap(
                restore(item.source),
                restore(item.target),
                tuple(restore(value) for value in item.maps),
                map_id=item.map_id,
                degree_offset=item.degree_offset,
            )
            if (
                rebuilt.map_id != item.map_id
                or rebuilt.first_degree != item.first_degree
                or rebuilt.degree_offset != item.degree_offset
            ):
                raise ValueError("Stored FE complex-map identity changed.")
            return rebuilt
        if type(item) is SparseTransferEntryEvidence:
            return SparseTransferEntryEvidence(
                item.degree,
                item.source_size,
                item.target_size,
                item.target_indices,
                item.source_indices,
                item.absolute_error_bounds,
                item.exact_integral_counts,
                item.enclosed_integral_counts,
            )
        if type(item) is SparseCommutingEvidence:
            return SparseCommutingEvidence(
                item.kind,
                item.degree,
                item.source_size,
                item.target_size,
                item.target_indices,
                item.source_indices,
                item.observed_defects,
                item.integration_error_bounds,
                item.roundoff_error_bounds,
            )
        if type(item) is CurvedOverlapIntegrationPolicy:
            return CurvedOverlapIntegrationPolicy(
                relative_tolerance=item.relative_tolerance,
                absolute_tolerance=item.absolute_tolerance,
            )
        if type(item) is FiniteElementTransferResult:
            rebuilt = FiniteElementTransferResult(
                restore(item.complex_map),
                tuple(restore(value) for value in item.entry_evidence),
                tuple(restore(value) for value in item.commuting_evidence),
                None
                if item.integration_policy is None
                else restore(item.integration_policy),
                item.exact_integral_count,
                item.enclosed_integral_count,
                constraint_evidence=tuple(
                    restore(value) for value in item.constraint_evidence
                ),
                expected_evidence_id=item.evidence_id,
            )
            if (
                rebuilt.transfer_id != item.transfer_id
                or rebuilt.evidence_id != item.evidence_id
            ):
                raise ValueError("Stored FE transfer identity changed.")
            return rebuilt
        if type(item) is PeriodicFiniteElementTransfer:
            rebuilt = PeriodicFiniteElementTransfer(
                restore(item.complex_map),
                restore(item.lifted_transfer_result),
                restore(item.quotient_transfer_result),
                tuple(restore(value) for value in item.source_lifts),
                tuple(restore(value) for value in item.source_collapses),
                tuple(restore(value) for value in item.target_lifts),
                tuple(restore(value) for value in item.target_collapses),
                item.source_orbit_map_ids,
                item.target_orbit_map_ids,
                item.source_active_indices,
                item.target_active_indices,
                item.boundary,
                item.source_closure_defects,
                item.target_closure_defects,
                item.source_nilpotency_defects,
                item.target_nilpotency_defects,
                item.deck_descent_defects,
                item.deck_descent_error_bounds,
                item.commuting_defects,
                item.commuting_error_bounds,
                item.transfer_id,
            )
            return rebuilt
        if type(item) is DistributedFiniteElementTransfer:
            transfer_result = restore(item.transfer_result)
            if not isinstance(item.degree_maps, tuple) or any(
                type(value).__module__ != "phydrax.discretization._distributed_field"
                or type(value).__name__ != "DistributedSparseTransfer"
                for value in item.degree_maps
            ):
                raise TypeError("Distributed FE maps lost their typed owner rows.")
            part_counts = {value.part_count for value in item.degree_maps}
            if len(part_counts) != 1:
                raise ValueError("Distributed FE degrees do not share a partition count.")
            part_count = next(iter(part_counts))
            rebuilt = _distribute_transfer_result(
                transfer_result,
                tuple(value.source_owner for value in item.degree_maps),
                tuple(value.target_owner for value in item.degree_maps),
                part_count,
            )
            if rebuilt.transfer_id != item.transfer_id:
                raise ValueError("Distributed FE transfer identity changed.")
            return rebuilt
        if type(item) is CompositionDependency:
            return CompositionDependency(item.entry_id, item.facet, item.bound_id)
        if type(item) is CompositionEntry:
            entry = CompositionEntry(
                restore(item.value),
                entry_id=item.entry_id,
                role=item.role,
                owner_id=item.owner_id,
                structure_id=item.structure_id,
                revision_id=item.revision_id,
                semantics_id=item.semantics_id,
                dependencies=tuple(restore(value) for value in item.dependencies),
            )
            if entry.entry_id != item.entry_id:
                raise ValueError("Stored Composition entry identity changed.")
            return entry
        if type(item) is Composition:
            remap_ids = {
                "owner-local/specification",
                "owner-local/view",
                "owner-local/remap",
            }
            if set(item.entry_ids) == remap_ids:
                from .._execution_runtime import ExecutionGroup
                from ..discretization._view_support import OwnerLocalCellGeometry
                from ..discretization.finite_volume._owner_local_remap import (
                    PreparedOwnerLocalConservativeRemap,
                )
                from ..meshing._contracts import VolumeMeshingSpec

                if not isinstance(execution_group, ExecutionGroup):
                    raise TypeError(
                        "Owner-local remap restore requires its explicit ExecutionGroup."
                    )
                entries = {entry.entry_id: entry for entry in item.entries}
                stored_specification = entries["owner-local/specification"].value
                if type(stored_specification) is not VolumeMeshingSpec:
                    raise TypeError("Owner-local remap lost its current volume spec.")
                specification = VolumeMeshingSpec.from_stored_fields(stored_specification)
                stored_view = entries["owner-local/view"].value
                if type(stored_view) is not OwnerLocalCellGeometry:
                    raise TypeError("Owner-local remap lost its current geometry view.")
                view = OwnerLocalCellGeometry.from_stored_fields(
                    stored_view,
                    execution_group,
                    current_specification=specification,
                    axis_name="model_archive_owner_local_remap_restore",
                )
                stored_remap = entries["owner-local/remap"].value
                if type(stored_remap) is not PreparedOwnerLocalConservativeRemap:
                    raise TypeError("Owner-local remap lost its prepared transfer.")
                remap = PreparedOwnerLocalConservativeRemap.from_stored_fields(
                    stored_remap,
                    view,
                    view,
                    execution_group,
                )
                rebuilt_values = {
                    "owner-local/specification": specification,
                    "owner-local/view": view,
                    "owner-local/remap": remap,
                }
                rebuilt_entries = tuple(
                    CompositionEntry(
                        rebuilt_values[entry.entry_id],
                        entry_id=entry.entry_id,
                        role=entry.role,
                        owner_id=entry.owner_id,
                        structure_id=entry.structure_id,
                        revision_id=entry.revision_id,
                        semantics_id=entry.semantics_id,
                        dependencies=tuple(
                            restore(dependency) for dependency in entry.dependencies
                        ),
                    )
                    for entry in item.entries
                )
                composition = Composition(
                    rebuilt_entries,
                    boundary_id=item.boundary_id,
                )
                if (
                    composition.composition_id != item.composition_id
                    or specification.specification_id
                    != entries["owner-local/specification"].revision_id
                    or view.distribution_id != entries["owner-local/view"].revision_id
                    or remap.remap_id != entries["owner-local/remap"].revision_id
                ):
                    raise ValueError(
                        "Owner-local remap archive changed its current binding."
                    )
                return composition
            adaptive_ids = {
                "owner-local-adaptive/specification",
                "owner-local-adaptive/graph-plan",
                "owner-local-adaptive/view",
            }
            if set(item.entry_ids) == adaptive_ids:
                from .._execution_runtime import ExecutionGroup
                from ..discretization._view_support import OwnerLocalCellGeometry
                from ..graph._partition import GraphPartitionPlan
                from ..meshing._contracts import (
                    SurfaceMeshingSpec,
                    VolumeMeshingSpec,
                )

                if not isinstance(execution_group, ExecutionGroup):
                    raise TypeError(
                        "Owner-local adaptive restore requires its explicit source "
                        "ExecutionGroup."
                    )
                entries = {entry.entry_id: entry for entry in item.entries}
                stored_specification = entries["owner-local-adaptive/specification"].value
                if type(stored_specification) is SurfaceMeshingSpec:
                    specification = SurfaceMeshingSpec.from_stored_fields(
                        stored_specification
                    )
                elif type(stored_specification) is VolumeMeshingSpec:
                    specification = VolumeMeshingSpec.from_stored_fields(
                        stored_specification
                    )
                else:
                    raise TypeError("Owner-local adaptive archive lost its current spec.")
                stored_plan = entries["owner-local-adaptive/graph-plan"].value
                if type(stored_plan) is not GraphPartitionPlan:
                    raise TypeError(
                        "Owner-local adaptive archive lost its graph partition plan."
                    )
                graph_partition_plan = GraphPartitionPlan.from_stored_fields(stored_plan)
                stored_view = entries["owner-local-adaptive/view"].value
                if type(stored_view) is not OwnerLocalCellGeometry:
                    raise TypeError(
                        "Owner-local adaptive archive lost its compact source view."
                    )
                # The compact replay evidence (adaptive result vertex transfer)
                # stores its sparse maps' spaces and operators as archive
                # projections; decode them before the view replays them.
                nested_recipe_types = (
                    ArraySpaceArchiveRecipe,
                    _FiniteElementSpecArchiveProjection,
                    OperatorPairingArchiveRecipe,
                    SparseCoordinateOperatorArchiveRecipe,
                    PeriodicSimplexForest,
                )
                stored_view = jax.tree_util.tree_map(
                    lambda node: (
                        restore(node) if type(node) in nested_recipe_types else node
                    ),
                    stored_view,
                    is_leaf=lambda node: type(node) in nested_recipe_types,
                )
                view = OwnerLocalCellGeometry.from_stored_fields(
                    stored_view,
                    execution_group,
                    current_specification=specification,
                    graph_partition_plan=graph_partition_plan,
                    axis_name="model_archive_owner_local_restore",
                )
                rebuilt_values = {
                    "owner-local-adaptive/specification": specification,
                    "owner-local-adaptive/graph-plan": graph_partition_plan,
                    "owner-local-adaptive/view": view,
                }
                rebuilt_entries = tuple(
                    CompositionEntry(
                        rebuilt_values[entry.entry_id],
                        entry_id=entry.entry_id,
                        role=entry.role,
                        owner_id=entry.owner_id,
                        structure_id=entry.structure_id,
                        revision_id=entry.revision_id,
                        semantics_id=entry.semantics_id,
                        dependencies=tuple(
                            restore(dependency) for dependency in entry.dependencies
                        ),
                    )
                    for entry in item.entries
                )
                composition = Composition(
                    rebuilt_entries,
                    boundary_id=item.boundary_id,
                )
                if (
                    composition.composition_id != item.composition_id
                    or specification.specification_id
                    != entries["owner-local-adaptive/specification"].revision_id
                    or graph_partition_plan.plan_id
                    != entries["owner-local-adaptive/graph-plan"].revision_id
                    or view.distribution_id
                    != entries["owner-local-adaptive/view"].revision_id
                ):
                    raise ValueError(
                        "Owner-local adaptive archive changed its current binding."
                    )
                return composition
            if (
                frozenset(item.entry_ids)
                == PERIODIC_FINITE_ELEMENT_TRANSFER_ROOT_ENTRY_IDS
            ):
                return _restore_periodic_finite_element_transfer_root(item, restore)
            previous = in_composition
            graph_start = len(pending_graphs)
            selection_start = len(pending_selections)
            in_composition = True
            try:
                entries = tuple(restore(entry) for entry in item.entries)
            finally:
                in_composition = previous
            graphs = tuple(
                graph
                for entry in entries
                if type(entry.value) is BRepPartitionResult
                for graph in (
                    entry.value.association_graph,
                    *(value for _, value in entry.value.operand_association_graphs),
                )
            )
            for graph in pending_graphs[graph_start:]:
                if not any(same_content(graph, owner) for owner in graphs):
                    raise ValueError("CAD graph has no replayed owning partition.")
            for selection in pending_selections[selection_start:]:
                inventories = tuple(
                    revision
                    for graph in graphs
                    for revision in (graph.source_revision, graph.target_revision)
                    if revision.revision_id == selection.revision_id
                )
                if not inventories or any(
                    inventory != inventories[0] for inventory in inventories
                ):
                    raise ValueError(
                        "CAD selection has no unique current graph inventory."
                    )
                for selector in selection.selectors:
                    if inventories[0].select(selector.occurrence_id) != selector:
                        raise ValueError(
                            "CAD selection does not match its current inventory."
                        )
            del pending_graphs[graph_start:]
            del pending_selections[selection_start:]
            composition = Composition(entries, boundary_id=item.boundary_id)
            if model_structure_recipe(composition) != model_structure_recipe(item):
                raise ValueError("Stored Composition content changed.")
            return composition
        if isinstance(item, tuple):
            values = tuple(restore(value) for value in item)
            return type(item)(*values) if "_fields" in vars(type(item)) else values
        if isinstance(item, list):
            return [restore(value) for value in item]
        if isinstance(item, Mapping):
            return {restore(key): restore(value) for key, value in item.items()}
        return item

    return restore(value)


def _restore_periodic_finite_element_transfer_root(
    item: Any, restore: Callable[[Any], Any], /
) -> Any:
    """Rebuild a periodic FE transfer root from its mesh, settings and owner rows.

    The archive keeps only these authoritative inputs plus the transfer
    identities; every quotient operator, its evidence and the distributed
    sparse plans are recomputed here and must reproduce the stored identities.
    """
    from .._fingerprint import canonical_fingerprint
    from ..discretization._cell_mesh import CellMesh
    from ..discretization.fem._de_rham import FiniteElementDeRhamComplex
    from ..lifecycle._composition_rebind import Composition, CompositionEntry

    stored = {entry.entry_id: entry for entry in item.entries}
    mesh = restore(stored["mesh/periodic-topology"].value)
    settings = restore(stored["transfer/settings"].value)
    periodic_recipe = stored["transfer/periodic"].value
    distributed_recipe = stored["transfer/distributed"].value
    if (
        type(mesh) is not CellMesh
        or not isinstance(settings, tuple)
        or len(settings) != 6
    ):
        raise TypeError(
            "Periodic FE archive lacks its exact source mesh/settings context."
        )
    if (
        type(periodic_recipe) is not PeriodicFiniteElementTransferArchiveRecipe
        or type(distributed_recipe) is not DistributedFiniteElementTransferArchiveRecipe
    ):
        raise TypeError("Periodic FE archive transfers must be stored by identity.")
    source_family, source_order, target_family, target_order, boundary, part_count = (
        settings
    )
    if (
        not isinstance(source_family, str)
        or type(source_order) is not int
        or not isinstance(target_family, str)
        or type(target_order) is not int
        or boundary not in ("absolute", "relative")
        or type(part_count) is not int
        or part_count <= 0
    ):
        raise TypeError("Periodic FE archive settings are malformed.")
    source_complex = FiniteElementDeRhamComplex(
        mesh,
        family=source_family,
        order=source_order,
    )
    target_complex = FiniteElementDeRhamComplex(
        mesh,
        family=target_family,
        order=target_order,
    )
    periodic = source_complex.transfer_periodic_quotient(
        target_complex,
        boundary=boundary,
    )
    if (
        periodic.transfer_id,
        periodic.complex_map.map_id,
        periodic.lifted_transfer_result.evidence_id,
        periodic.quotient_transfer_result.evidence_id,
    ) != tuple(periodic_recipe):
        raise ValueError("Periodic FE evidence differs from its current mesh/settings.")
    distributed = periodic.distributed(
        distributed_recipe.source_owners,
        distributed_recipe.target_owners,
        part_count,
    )
    if distributed.transfer_id != distributed_recipe.transfer_id:
        raise ValueError("Distributed FE evidence differs from its current owner rows.")
    rebuilt = {
        "mesh/periodic-topology": mesh,
        "transfer/settings": settings,
        "transfer/periodic": periodic,
        "transfer/distributed": distributed,
    }
    composition = Composition(
        tuple(
            CompositionEntry(
                rebuilt[entry.entry_id],
                entry_id=entry.entry_id,
                role=entry.role,
                owner_id=entry.owner_id,
                structure_id=entry.structure_id,
                revision_id=entry.revision_id,
                semantics_id=entry.semantics_id,
                dependencies=tuple(restore(value) for value in entry.dependencies),
            )
            for entry in item.entries
        ),
        boundary_id=item.boundary_id,
    )
    if composition.composition_id != item.composition_id:
        raise ValueError("Periodic FE archive changed its composition identity.")
    settings_id = canonical_fingerprint(
        {"kind": "periodic-finite-element-transfer-settings", "settings": settings}
    )
    entries = {entry.entry_id: entry for entry in composition.entries}
    mesh_entry = entries["mesh/periodic-topology"]
    settings_entry = entries["transfer/settings"]
    periodic_entry = entries["transfer/periodic"]
    distributed_entry = entries["transfer/distributed"]
    if (
        mesh_entry.role != "topology"
        or mesh_entry.owner_id != "phydrax.discretization"
        or mesh_entry.structure_id != mesh.topology_id
        or mesh_entry.revision_id != mesh.mesh_id
        or mesh_entry.semantics_id != "periodic-finite-element-cover-mesh"
        or mesh_entry.dependencies
        or settings_entry.role != "model-parameter"
        or settings_entry.owner_id != "phydrax.discretization.fem"
        or settings_entry.structure_id != settings_id
        or settings_entry.revision_id != settings_id
        or settings_entry.semantics_id != "periodic-finite-element-transfer-settings"
        or frozenset(settings_entry.dependencies)
        != {mesh_entry.binding("structure"), mesh_entry.binding("revision")}
        or periodic_entry.role != "transfer"
        or periodic_entry.owner_id != "phydrax.discretization.fem"
        or periodic_entry.structure_id != periodic.complex_map.map_id
        or periodic_entry.revision_id != periodic.transfer_id
        or periodic_entry.semantics_id != "periodic-finite-element-quotient-transfer"
        or frozenset(periodic_entry.dependencies)
        != {
            mesh_entry.binding("structure"),
            mesh_entry.binding("revision"),
            settings_entry.binding("semantics"),
        }
        or distributed_entry.role != "transfer"
        or distributed_entry.owner_id != "phydrax.discretization.fem"
        or distributed_entry.structure_id != distributed.complex_map.map_id
        or distributed_entry.revision_id != distributed.transfer_id
        or distributed_entry.semantics_id
        != "distributed-finite-element-quotient-transfer"
        or frozenset(distributed_entry.dependencies)
        != {
            periodic_entry.binding("revision"),
            settings_entry.binding("semantics"),
        }
        or composition.boundary_id != "periodic-distributed-fe-transfer:archive"
    ):
        raise ValueError("Periodic FE source-context bindings changed.")
    return composition


def _unaliased_recipe(node: Any, /) -> Any:
    """Copy a recipe so every occurrence is its own node (stack, no recursion)."""
    root: dict[str, Any] = {"node": None}
    pending: list[tuple[Any, dict[Any, Any] | list[Any], Any]] = [(node, root, "node")]
    while pending:
        value, parent, slot = pending.pop()
        if isinstance(value, dict):
            child: Any = dict.fromkeys(value)
            pending.extend((item, child, key) for key, item in value.items())
        elif isinstance(value, list):
            child = [None] * len(value)
            pending.extend((item, child, index) for index, item in enumerate(value))
        else:
            child = value
        parent[slot] = child
    return root["node"]


def write_model_archive(
    path: str | Path,
    /,
    *,
    manifest: dict[str, Any],
    arrays: dict[str, Any],
    limits: ArrayArchiveLimits | None,
) -> Path:
    """Write an expanded native model manifest in canonical shared form.

    ``manifest`` holds the full ``model_recipe`` and ``arrays`` its exact
    per-occurrence inventory, as returned by :func:`read_model_archive`. Every
    occurrence is keyed by its own content, so editing one occurrence or one
    definition is stored as exactly that content; the recipe is not validated
    here, so readers decide whether it is admissible.
    """
    if set(manifest) != {"kind", "model_recipe"} or manifest["kind"] != _MODEL_KIND:
        raise ValueError(
            "Native model archive manifests hold only kind and model_recipe."
        )
    return _write_model_archive_recipe(
        path, _unaliased_recipe(manifest["model_recipe"]), arrays, limits
    )


def _write_model_archive_recipe(
    path: str | Path,
    recipe: dict[str, Any],
    arrays: dict[str, Any],
    limits: ArrayArchiveLimits | None,
    /,
) -> Path:
    shared_recipe, members = share_model_archive_recipe(
        recipe, arrays, prefix=_MODEL_ARRAYS
    )
    return write_array_archive(
        path,
        manifest={"kind": _MODEL_KIND, "model_recipe": shared_recipe},
        limits=limits,
        arrays=members,
    )


def read_model_archive(
    path: str | Path,
    /,
    *,
    limits: ArrayArchiveLimits | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a native model archive as its expanded manifest and array inventory.

    This is the one expansion point of shared archive nodes: the returned
    ``model_recipe`` is the validated full recipe (equal definitions are one
    node), and ``arrays`` is its exact per-occurrence inventory. No reader of a
    native model archive sees a shared reference.
    """
    _register_model_archive_types()
    manifest, members = read_array_archive(path, limits=limits)
    if (
        set(manifest) != {"arrays", "kind", "model_recipe"}
        or manifest["kind"] != _MODEL_KIND
    ):
        raise ArrayArchiveCorruptionError(
            "Native model archive fields are not canonical."
        )
    shared_recipe = manifest["model_recipe"]
    if not isinstance(shared_recipe, dict) or shared_recipe.get("kind") != "dataclass":
        raise ArrayArchiveCorruptionError("Native model archive has no dataclass recipe.")
    try:
        recipe, arrays = expand_shared_model_archive_recipe(
            shared_recipe, members, prefix=_MODEL_ARRAYS, limits=limits
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ArrayArchiveCorruptionError(
            "Native model archive is incompatible with its registered recipe."
        ) from error
    return {"kind": _MODEL_KIND, "model_recipe": recipe}, arrays


def _validate_archive_columns(
    recipe: dict[str, Any],
    payloads: dict[int, Any],
    limits: ArrayArchiveLimits | None,
    /,
) -> None:
    """Admit every column record without rebuilding its enclosing geometry."""
    from .._archive_columns import (
        ExactTensorArchiveRecipe,
        IntegerRowsArchiveRecipe,
        StaticColumnArchiveRecipe,
        TextColumnArchiveRecipe,
        TextRowsArchiveRecipe,
    )
    from ._archive_layer_columns import (
        ExactExpressionRowsArchiveRecipe,
        ImplicitChartRowsArchiveRecipe,
    )

    row_types = _row_record_recipe_types()
    column_types = frozenset(
        (
            ExactTensorArchiveRecipe,
            IntegerRowsArchiveRecipe,
            StaticColumnArchiveRecipe,
            TextColumnArchiveRecipe,
            TextRowsArchiveRecipe,
            ExactExpressionRowsArchiveRecipe,
            ImplicitChartRowsArchiveRecipe,
        )
    )
    seen: set[int] = set()
    pending: list[Any] = [recipe]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, dict):
            owner = (
                artifact_value(node["type"]) if node.get("kind") == "namedtuple" else None
            )
            if owner in row_types or owner in column_types:
                arrays = {
                    f"{_MODEL_ARRAYS}/{index:06d}": payloads[id(array)]
                    for index, array in enumerate(_recipe_array_nodes(node))
                }
                value = model_from_array_recipe(
                    node,
                    arrays,
                    prefix=_MODEL_ARRAYS,
                    limits=limits,
                    validate=False,
                    restore_rows=False,
                )
                if type(value) is not owner or model_structure_recipe(value) != node:
                    raise ValueError(
                        "Restored column fields differ from the exact recipe."
                    )
                if owner in column_types:
                    value.decode()
                else:
                    restore_row_record_recipes(value)
                # Restore validates each column and local content identity.
                # Retain no unrequested record or decoded exact coordinate bank.
                del value
            else:
                pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)


def read_model_archive_fields(
    path: str | Path,
    /,
    *,
    expected_type: type,
    fields: tuple[str, ...],
    limits: ArrayArchiveLimits | None,
) -> dict[str, Any]:
    """Decode selected registered field paths for prerequisite validation.

    The complete archive is admitted first. The returned fields are stored
    inputs, not certified authority; the owner must replay their mathematics.
    """
    _register_model_archive_types()
    _require_model_type(expected_type)
    if (
        not fields
        or any(
            type(name) is not str or any(not part for part in name.split("."))
            for name in fields
        )
        or len(set(fields)) != len(fields)
    ):
        raise ValueError("Selected model fields must be distinct nonempty names.")
    manifest, arrays = read_model_archive(path, limits=limits)
    recipe = manifest["model_recipe"]
    try:
        if artifact_value(recipe["type"]) is not expected_type:
            raise ValueError("Native model archive has a different registered root type.")
        subtrees = {}
        for name in fields:
            subtree = recipe
            for part in name.split("."):
                if subtree.get("kind") not in ("dataclass", "strict_value"):
                    raise ValueError("Selected model path crosses a non-record value.")
                subtree = subtree["fields"][part]
            subtrees[name] = subtree
        nodes = _recipe_array_nodes(recipe)
        if len(nodes) != len(arrays):
            raise ValueError("Model recipe array inventory is inconsistent.")
        payloads = {
            id(node): arrays[f"{_MODEL_ARRAYS}/{index:06d}"]
            for index, node in enumerate(nodes)
        }
        _validate_archive_columns(recipe, payloads, limits)
        selected = {}
        for name in fields:
            subtree = subtrees[name]
            subnodes = _recipe_array_nodes(subtree)
            subarrays = {
                f"{_MODEL_ARRAYS}/{index:06d}": payloads[id(node)]
                for index, node in enumerate(subnodes)
            }
            raw = model_from_array_recipe(
                subtree,
                subarrays,
                prefix=_MODEL_ARRAYS,
                limits=limits,
                validate=False,
                restore_rows=False,
            )
            if model_structure_recipe(raw) != subtree:
                raise ValueError("Restored selected fields differ from the exact recipe.")
            selected[name] = restore_row_record_recipes(raw)
        return selected
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ArrayArchiveCorruptionError(
            "Native model archive is incompatible with its registered recipe."
        ) from error


def write_model_artifact(
    path: str | Path,
    model: Any,
    /,
    *,
    limits: ArrayArchiveLimits | None,
) -> Path:
    """Write one exact registered Strict dataclass and its dynamic array bank."""
    _register_model_archive_types()
    _require_model_type(type(model))
    archive_model = model_archive_root_projection(model)
    validate_tree(archive_model)
    recipe = model_structure_recipe(archive_model)
    if recipe.get("kind") != "dataclass" or recipe.get("type") != artifact_value_id(
        type(model)
    ):
        raise TypeError(
            "Native model recipe does not identify its exact registered root type."
        )
    validate_model_structure_recipe(recipe, limits=limits)
    arrays = pack_model_array_tree(
        archive_model, recipe, prefix=_MODEL_ARRAYS, limits=limits
    )
    # model_structure_recipe creates a distinct node for every occurrence;
    # unlike an expanded shared archive, it needs no unaliasing copy.
    return _write_model_archive_recipe(path, recipe, arrays, limits)


def read_model_artifact(
    path: str | Path,
    /,
    *,
    expected_type: type,
    restore: Callable[[Any], Any],
    limits: ArrayArchiveLimits | None,
) -> Any:
    """Restore a root recipe, then delegate scientific reconstruction to its owner."""
    _register_model_archive_types()
    _require_model_type(expected_type)
    if not callable(restore):
        raise TypeError("restore must be the owning model's reconstruction function.")
    manifest, arrays = read_model_archive(path, limits=limits)
    recipe = manifest["model_recipe"]
    try:
        declared_type = artifact_value(recipe["type"])
        _require_model_type(declared_type)
        if declared_type is not expected_type:
            raise ValueError("Native model archive has a different registered root type.")
        raw = model_from_array_recipe(
            recipe,
            arrays,
            prefix=_MODEL_ARRAYS,
            limits=limits,
            validate=False,
            restore_rows=False,
        )
        if type(raw) is not expected_type or model_structure_recipe(raw) != recipe:
            raise ValueError("Restored model fields differ from the exact recipe.")
        raw = restore_row_record_recipes(raw)
        rebuilt = restore(raw)
        del raw
        if type(rebuilt) is not expected_type:
            raise TypeError(
                "The owning reconstruction function returned a different model type."
            )
        validate_tree(rebuilt)
        if not model_structure_matches_recipe(rebuilt, recipe):
            raise ValueError(
                "The owning constructor graph did not reproduce the model recipe."
            )
        rebuilt_arrays = pack_model_array_tree(
            rebuilt, recipe, prefix=_MODEL_ARRAYS, limits=limits
        )
        if set(rebuilt_arrays) != set(arrays) or any(
            array_payload_digest(rebuilt_arrays[name])
            != array_payload_digest(arrays[name])
            for name in arrays
        ):
            raise ValueError("Owning constructors changed a serialized model array.")
    except ArrayArchiveCorruptionError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ArrayArchiveCorruptionError(
            "Native model archive is incompatible with its registered recipe."
        ) from error
    return rebuilt


def _require_model_type(model_type: Any, /) -> None:
    if not isinstance(model_type, type) or not dataclasses.is_dataclass(model_type):
        raise TypeError("Native model archive roots must be registered dataclass types.")
    if not issubclass(model_type, Strict):
        raise TypeError("Native model archive roots must implement the Strict contract.")
    artifact_value_id(model_type)
    fields = dataclasses.fields(model_type)
    if not fields or len({field.name for field in fields}) != len(fields):
        raise TypeError("Native model archive root has no exact registered field set.")


__all__ = [
    "read_model_archive",
    "read_model_archive_fields",
    "read_model_artifact",
    "write_model_archive",
    "write_model_artifact",
]
