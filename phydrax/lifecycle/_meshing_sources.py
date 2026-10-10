#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Canonical, callback-free native source closures at the archive boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from inspect import Parameter, signature
from math import prod
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast, TYPE_CHECKING

import jax
import numpy as np

from .._array_archive import (
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from .._bvh import BVHBuildKind, BVHBuildPolicy, PackedBVH
from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._frozendict import frozendict
from .._geometry_predicates import PredicateMode
from .._identity import SemanticProvenance
from .._model._artifacts import (
    ArtifactValueCodec,
    register_artifact_value,
    register_artifact_value_codec,
    registered_artifact_value_id,
)
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from ..applications.neurofluid._model import (
    NeurofluidTransportCheckpoint,
    NeurofluidTransportParameters,
    NeurofluidTransportUnits,
)
from ..discretization import (
    _adaptive_simplex,
    _axis,
    _axis_domain,
    _cell_complex,
    _cell_geometry,
    _cell_geometry_transfer,
    _cell_geometry_validity,
    _cell_mesh,
    _core as _discretization_core,
    _partition,
    _periodic_cell,
    _periodic_topology,
    _spaces as _discretization_spaces,
    _support,
    _tensor_entities,
    _tensor_support,
    _topology,
)
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._distributed_field import DistributedHaloPlan
from ..discretization._integration_domain import IntegrationDomain
from ..discretization._measure import DiscreteMeasure
from ..discretization._metric_network import MetricNetworkPlan
from ..discretization._point_cloud import (
    _require_point_cloud_plan_integrity,
    PointCloudPlan,
)
from ..discretization._simplicial_locator import SimplicialLocationPolicy
from ..discretization._sphere_chart_deformation import (
    PreparedSphereChartDeformation,
    PreparedSphereChartOccurrence,
    PreparedSphereChartPiece,
    SphereGeometryReconstruction,
)
from ..discretization._surface_chart_deformation import (
    PreparedSurfaceChartDeformation,
    PreparedSurfaceChartOccurrence,
    PreparedSurfaceChartPiece,
    SurfaceChartWitness,
)
from ..discretization.fem import _distributed as _fem_distributed
from ..discretization.fem._generic import (
    FiniteElementBlockGeometry,
    FiniteElementDiscretization,
    FiniteElementDofMap,
    FiniteElementDofSourceProjection,
    FiniteElementPlan,
    FiniteElementRuntimeData,
)
from ..discretization.fem._reference import FiniteElementSpec
from ..discretization.fem._restoration import finite_element_restoration_validation_scope
from ..discretization.fem._surface_chart_compatible import (
    PreparedSurfaceChartCompatibleTransfer,
)
from ..discretization.fem._surface_chart_transfer import (
    PreparedSurfaceChartFiniteVolumeContents,
)
from ..discretization.fem._topology_transfer import (
    FiniteElementFieldTransfer,
    FiniteElementTopologyTransfer,
    FiniteElementTransferEvidence,
    TransferGeometryBinding,
)
from ..discretization.finite_volume._remap_evidence import (
    MappedNestedRemapEvidence,
    MappedSurfaceChartRemapEvidence,
    PreparedUnstructuredConservativeRemap,
    RemapPreparationFailure,
)
from ..discretization.finite_volume._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
    UnstructuredRemapReport,
)
from ..discretization.iga._basis import (
    IsogeometricFieldSpec,
    IsogeometricQuadraturePolicy,
    SplineAxisPlan,
    TensorSplineBasisSpec,
)
from ..discretization.iga._geometry import (
    IsogeometricH1QualificationPolicy,
    NURBSGeometryState,
)
from ..discretization.iga._identity import BaseSpanId
from ..discretization.iga._plan import IsogeometricPlan
from ..discretization.iga._topology import SplineSpanTopology
from ..discretization.meshfree._stencils import LocalStencilPolicy
from ..discretization.spatial._morton import MortonAddressPlan
from ..equations._mixed_dimensional import MixedDimensionalTransportState
from ..exterior._form_type import FormType, FormValueSpec
from ..geometry import (
    _atlas as atlas,
    _certificate as field,
    _certified_implicit,
    _compartments as _geometry_compartments,
    _contracts as geometry,
    _mesh_certificates as certificate,
    _meshing_domain as domain,
)
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..geometry._planar_embedding import PlanarEmbedding
from ..geometry._sphere_material_atlas import (
    SphereMaterialCellAtlas,
    SphereMaterialInverse,
    SphereProjectiveReferenceMap,
    SphereProjectiveTriangleBounds,
)
from ..geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementEvidence,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    PreparedCommonRefinement,
)
from ..geometry._surface_source_support import (
    PreparedSurfaceSourceSupport,
    SurfaceNativeRestrictionBoundarySource,
    SurfaceSourceCharts,
    SurfaceSourceReceipts,
    SurfaceSourceRootAtlas,
)
from ..geometry.analytic import (
    _expressions,
    _extended,
    _operations,
    _primitives,
    _superquadric,
    _sweeps,
)
from ..geometry.brep import (
    _constructors,
    _intersection,
    _intersection_curve,
    _model,
    _patches,
    _projection_contracts,
    _query,
    _root_bindings,
    _source,
)
from ..geometry.design import _schema
from ..geometry.design._qualification import DerivativeTier, DesignQualificationEvidence
from ..geometry.implicit import (
    _adaptive_discovery,
    _analytic_profile,
    _enclosure,
    _policy as _implicit_policy,
    _projection as _implicit_projection,
    _realization as _implicit_realization,
)
from ..geometry.multiregion_surface import (
    _contracts as _multi_contracts,
    _label_extraction,
    _seeding as _multi_seeding,
    _topology as _multi_topology,
)
from ..geometry.simplicial import (
    _mesh as _simplicial_mesh,
    _regions,
    _topology as _simplicial_topology,
)
from ..geometry.simplicial._topology import SegmentTopology
from ..imaging import _asset as _image_asset
from ..linalg._distributed import DistributedPairing
from ..linalg._pairings import EuclideanPairing
from ..linalg._properties import OperatorCapabilities, OperatorProperties
from ..linalg._spaces import ArraySpace, BlockSpace
from ..measurement import (
    _asset as _measurement_asset,
    _field as _measurement_field,
    _quantity as _measurement_quantity,
    _time as _measurement_time,
)
from ..meshing import (
    _adaptation,
    _assembly,
    _audit,
    _bisection,
    _contracts,
    _controls,
    _coupling,
    _curving,
    _decision,
    _distribution,
    _implicit_volume,
    _level_set,
    _lineage,
    _mixed_adaptation,
    _organization,
    _overset,
    _periodic,
    _polyhedral_adaptation,
    _proposals,
    _quality,
    _result,
    _scope,
    _sizing,
    _topology_edit,
    _trace,
)
from ..meshing._association import (
    AssociationPropagationPolicy,
    BRepAssociationTransfer,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
    MappedReferenceAssociationTransfer,
    PlcAssociationTransfer,
    SurfaceAssociationTransfer,
)
from ..meshing._association_composition import ComposedAssociationTransfer
from ..meshing._audit import CellMeshAuditReport
from ..meshing._certification import (
    MeshCertificationOutcome,
    MeshCertificationReport,
    MeshCertificationSchedule,
)
from ..meshing._certification_inputs import MeshCertificationInputs
from ..meshing._implicit_association_transfer import ImplicitAssociationTransfer
from ..meshing._measurements import NativeExecutionRecord
from ..meshing._metric import MeshMetricField
from ..meshing._surface_association_transfer import (
    PreparedSurfaceCurveWitness,
    SurfaceChartBoundarySource,
    SurfaceSubdivisionBoundarySource,
)
from ..meshing._tetra_metric import (
    MetricRemeshingCriterion,
    MetricRemeshingEvidence,
    MetricRemeshingStatus,
)
from ..meshing.providers._native import NativeMeshingPlan
from ..meshing.providers._native_compartment import PreparedCompartmentVolume
from ..meshing.providers._native_options import (
    NativeCurveSchedule,
    NativeMeshingOptions,
    NativeStructuredSchedule,
    NativeSurfaceSchedule,
)
from ..meshing.providers._native_sources import (
    NativeImplicitSource,
    NativeLayerCoreSource,
    NativePlanarSource,
)
from ..optim._iterative import _types as _optim_types
from ..qualification._reference import ReferenceArtifactManifest
from ..solver._finite_element_schedule import FiniteElementAcceptedState
from ..sparse._linear import SparseLinearMap
from ..sparse._relation import EdgeRelation, RowRelation
from ..typing import validate
from ..units import DimensionSignature, UnitDefinition
from ._meshing_field_records import (
    meshing_field_artifact_types,
    MeshingAcceptedEpoch,
    rebuild_meshing_field_record,
    validate_meshing_accepted_epochs,
    validate_meshing_field_declarations,
)
from ._meshing_source_families import (
    is_native_generation_source,
    native_family_artifact_types,
    NativeGenerationSource,
    rebuild_native_family_value,
    validate_registered_native_part,
)


if TYPE_CHECKING:
    from .._model._structure import ModelRecipeArray
    from ..meshing._initial_certification import InitialCollectiveMeshEvidence


_NATIVE_DESIGN_MODEL_TYPE: type | None = None


def register_native_design_source_artifacts() -> None:
    """Admit Source1's exact original code dependency only on explicit request."""
    from examples.curved_native_transfer import LinearMarker

    global _NATIVE_DESIGN_MODEL_TYPE
    register_artifact_value("examples.curved_native_transfer:LinearMarker", LinearMarker)
    _NATIVE_DESIGN_MODEL_TYPE = LinearMarker
    register_meshing_source_artifacts()


def _owning_artifact_types() -> tuple[tuple[str, tuple[type, ...]], ...]:
    # This is an admission whitelist, not a second identity/decoder registry.
    # Only the canonical artifact owner records and resolves these identities.
    from ..geometry.brep._placed import PlacedCurve, PlacedSurface
    from ..geometry.multiregion_surface import (
        _geometry as _multi_geometry,
        _seeding as _multi_seeding,
        _state as _multi_state,
        _topology as _multi_topology,
    )
    from ..imaging import _core as _image_core, _segmentation as _image_segmentation
    from ..meshing._device_generation import DeviceGenerationLayout
    from ..meshing._domain import (
        CompiledSurfaceDomain,
        SourcePatchNeighborhood,
        SourceSurfaceMetric,
        SourceSurfaceSizing,
    )
    from ..meshing._initial_certification import InitialCollectiveMeshEvidence
    from ..meshing._publication_lowering import PublicationProjection

    return (
        (
            (
                "phydrax.imaging",
                (
                    _image_core.MedicalImageAsset,
                    _image_core.LabelOntology,
                    _image_core.LabelVolume,
                    _image_core.ImageIndexAffine,
                    _image_core.LabelDefinition,
                    _image_core.MedicalImageSupport,
                    _image_core.ImageAxisConvention,
                    _image_core.VoxelReference,
                    _image_core.DeidentificationEvidence,
                    _image_asset.ImageFieldSpec,
                    _image_segmentation.CompartmentSurface,
                    _image_segmentation.CompartmentSurfaceResult,
                ),
            ),
            (
                "phydrax.measurement",
                (
                    _measurement_asset.AcquisitionIdentity,
                    _measurement_asset.DerivationRecord,
                    _measurement_asset.MeasurementAsset,
                    _measurement_asset.DataOrigin,
                    _measurement_asset.DataStage,
                    _measurement_field.SamplingSemantics,
                    _measurement_field.IndependentStandardUncertainty,
                    _measurement_field.QualityFlag,
                    _measurement_field.QuantityField,
                    _measurement_field.SpatialSamplingKind,
                    _measurement_quantity.QuantitySpec,
                    _measurement_quantity.ValueLayout,
                    _measurement_quantity.ValueKind,
                    _measurement_time.SampleTimeAxis,
                    _measurement_time.TemporalSampling,
                    _measurement_time.TimeBasis,
                    _measurement_time.TemporalSamplingKind,
                ),
            ),
            ("phydrax.qualification", (ReferenceArtifactManifest,)),
            (
                "phydrax.applications.neurofluid",
                (
                    NeurofluidTransportUnits,
                    NeurofluidTransportParameters,
                    NeurofluidTransportCheckpoint,
                ),
            ),
            ("phydrax.equations", (MixedDimensionalTransportState,)),
            (
                "phydrax.geometry",
                (
                    _geometry_compartments.CompartmentMeshingSource,
                    _geometry_compartments.CompartmentComplex,
                    _geometry_compartments.CompartmentDefinition,
                    _geometry_compartments.CompartmentInterfaceDefinition,
                    _geometry_compartments.CompartmentAdjacencyReport,
                ),
            ),
            (
                "phydrax.geometry.multiregion_surface",
                (
                    _multi_seeding.MultiRegionSurfaceSeed,
                    _multi_topology.MultiRegionSurfaceTopology,
                    _multi_state.MultiRegionSurfaceState,
                    _multi_geometry.PreparedMultiRegionSurface,
                    _label_extraction.LabelFieldSurfaceLineage,
                    _label_extraction.LabelFieldSurfaceExtractionEvidence,
                    _label_extraction.LabelFieldSurfaceExtractionStatus,
                    _label_extraction.LabelFieldSurfaceExtractionResult,
                    _label_extraction.LabelFieldVolumeBinding,
                    _multi_contracts.MultiRegionSurfaceStatus,
                    _multi_contracts.MultiRegionSurfaceCounts,
                    _multi_contracts.MultiRegionSurfaceCapacityPlan,
                    _multi_contracts.MultiRegionSurfaceCapacityEvidence,
                    _multi_contracts.MultiRegionSurfaceValidationPolicy,
                    _multi_contracts.MultiRegionSurfaceEvidence,
                ),
            ),
            (
                "phydrax.geometry",
                (
                    geometry.CompiledGeometry,
                    geometry.GeometryTolerance,
                    PlanarEmbedding,
                    field.FieldCertificate,
                    field.ExactSDFEnclosureCertificate,
                    field.ZeroSetAccuracy,
                    field.SignReliability,
                    field.DistanceSemantics,
                    field.FieldRegularity,
                    atlas.BoundaryAtlas,
                    atlas.TrimDomain,
                    atlas.PolygonTrimLoop,
                    atlas.CurveTrimLoop,
                    atlas.TrimTopologyEvidence,
                    atlas._SelectedBoundaryMap,
                    atlas._TranslatedBoundaryMap,
                    atlas._CircleBoundaryMap,
                    atlas._SphereBoundaryMap,
                    atlas._BoxBoundaryMap,
                    _schema.ParameterId,
                    _schema.ParameterSpec,
                    _schema.ParameterSchema,
                    _schema.ParameterBinding,
                    _schema.DesignState,
                    _analytic_profile.AnalyticImplicitProfile,
                    _analytic_profile._BoundaryMapBounds,
                    _enclosure._IntervalFieldBounds,
                    _adaptive_discovery.AdaptiveImplicitSurface,
                    _adaptive_discovery.AdaptiveImplicitSurfaceEvidence,
                    _adaptive_discovery.ImplicitVolumeQuery,
                    _adaptive_discovery.ImplicitVolumeClassification,
                    _adaptive_discovery.AdaptiveImplicitBoundarySource,
                    _adaptive_discovery.ImplicitNormalFiberMacrochart,
                    _implicit_policy.AdaptiveImplicitSurfacePolicy,
                    _implicit_policy.ImplicitSurfacePolicy,
                    _implicit_policy.ImplicitProjectionPolicy,
                    _implicit_realization.ImplicitSurfacePlan,
                    _implicit_projection.ImplicitPointProjectionPlan,
                    _certified_implicit.CertifiedImplicitCover,
                    _certified_implicit.CertifiedImplicitTopology,
                    domain.MeshingDomain,
                    domain.MeshingSurfacePatch,
                    domain.MeshingDomainCurve,
                    domain.MeshingDomainRegion,
                    domain.PatchCurveUse,
                    domain.PatchPoleUse,
                    domain.MeshingDomainBoundarySource,
                    domain._DomainCurveMap,
                    domain._PhysicalCurveMap,
                    certificate.MeshCertificateLimits,
                    certificate.MeshCertificateFinding,
                    certificate.MeshCertificateBinding,
                    certificate.MappedBoundaryDegreeEvidence,
                    certificate.GlobalEmbeddingCertificate,
                    certificate.PiecewiseLinearDomain,
                    certificate.DomainCoverageCertificate,
                    certificate.SourceBoundaryDistance,
                    certificate.SourceBoundarySamples,
                    certificate.SourceBoundaryChartCover,
                    certificate.ParametricCurveBoundarySource,
                    certificate.ImplicitBoundarySource,
                    certificate.ImplicitProjectionBoundarySource,
                    certificate.ImplicitProjectionCoverageEvidence,
                    certificate.SourceFidelityCertificate,
                    PreparedSurfaceSourceSupport,
                    SurfaceSourceReceipts,
                    SurfaceSourceRootAtlas,
                    SurfaceNativeRestrictionBoundarySource,
                    SurfaceSourceCharts,
                    SphereMaterialCellAtlas,
                    SphereMaterialInverse,
                    SphereProjectiveReferenceMap,
                    SphereProjectiveTriangleBounds,
                    CommonRefinementCoverage,
                    CommonRefinementPolicy,
                    CommonRefinementEvidence,
                    CommonRefinementStatus,
                    PreparedCommonRefinement,
                ),
            ),
            (
                "phydrax.geometry.brep",
                (
                    _model.BRepEntityId,
                    _model.BRepImportReport,
                    _model.BRepTopology,
                    _model.BRepOccurrence,
                    _model.BRepAssemblyContainer,
                    _model.BRepQualifiedIncidence,
                    _model.BRepGeometry,
                    _model.BRepModel,
                    _model.BRepBoundaryMap,
                    _model.BRepPlacedBoundaryMap,
                    _patches.RationalBezierPiece,
                    _patches.LineCurve,
                    _patches.CircleCurve,
                    _patches.EllipseCurve,
                    _patches.BSplineCurve,
                    _patches.PlanePatch,
                    _patches.CylinderPatch,
                    _patches.ConePatch,
                    _patches.SpherePatch,
                    _patches.TorusPatch,
                    _patches.BSplineSurfacePatch,
                    _patches.ExtrusionSurface,
                    _patches.RevolutionSurface,
                    _patches.RuledSurface,
                    _patches.OffsetSurface,
                    _constructors._NormalizedTrimCurve,
                    PlacedSurface,
                    PlacedCurve,
                    _intersection_curve.SurfaceRegion,
                    _intersection_curve.CurveRange,
                    _intersection_curve.BernsteinSurfacePiece,
                    _intersection_curve.BernsteinCurvePiece,
                    _intersection_curve.SurfacePiece,
                    _intersection_curve.CurvePiece,
                    _intersection_curve.SurfacePairSystem,
                    _intersection_curve.IntersectionCurve,
                    _intersection_curve.IntersectionPCurve,
                    _intersection_curve.CurveTrimSegment,
                    _intersection_curve.AffinePCurve,
                    _intersection_curve.PeriodicPCurve,
                    _intersection.TrimIntersectionRoot,
                    _intersection.CurveSurfaceIntersectionRoot,
                    _intersection.TripleSurfaceIntersectionRoot,
                    _intersection.IntersectionCurvePointRoot,
                    _intersection.TrimRootEndpoint,
                    _intersection.NativePeriodEndpoint,
                    _root_bindings.BRepRootSupport,
                    _root_bindings.BRepVertexRoot,
                    _root_bindings.BRepPlacedVertex,
                    _query.BRepQueryPolicy,
                    _query._Tolerances,
                    _query._EdgeData,
                    _query._FaceData,
                    _query._Quadrature,
                    _query.BRepMeasureResult,
                    _query.PreparedBRepQuery,
                    _query.NativeBRepProjection,
                    _source.BRepSource,
                    _source._NativeBRepKernel,
                    _projection_contracts.BRepProjectionPolicy,
                    _projection_contracts.BRepEntityDimension,
                    _projection_contracts.BRepProjectionStatus,
                ),
            ),
            (
                "phydrax.geometry.analytic",
                (
                    _primitives.Ball,
                    _primitives.Circle,
                    _primitives.Sphere,
                    _primitives.Orthotope,
                    _primitives.Box,
                    _primitives._BallKernel,
                    _primitives._OrthotopeKernel,
                    _extended.Ellipse,
                    _extended.Rectangle,
                    _extended.Polygon,
                    _extended.Ellipsoid,
                    _extended.AxisAlignedEllipsoid,
                    _extended.Cylinder,
                    _extended.Cone,
                    _extended._EllipseKernel,
                    _extended._PolygonKernel,
                    _extended._EllipsoidKernel,
                    _extended._AxisAlignedEllipsoidKernel,
                    _extended._CylinderKernel,
                    _extended._ConeKernel,
                    _extended._TorusKernel,
                    _extended.Torus,
                    _extended.Wedge,
                    _extended._WedgeKernel,
                    _extended._TriangleBoundaryMap,
                    _extended._EllipseBoundaryMap,
                    _extended._PolygonBoundaryMap,
                    _extended._RectangleBoundaryMap,
                    _extended._EllipsoidBoundaryMap,
                    _extended._CylinderBoundaryMap,
                    _extended._ConeBoundaryMap,
                    _extended._TorusBoundaryMap,
                    _expressions.Translation,
                    _expressions.Union,
                    _expressions._TranslationKernel,
                    _expressions._UnionKernel,
                    _operations.RigidFrame,
                    _operations.RigidTransform,
                    _operations.Scaling,
                    _operations.SharpCSG,
                    _operations.BlendCSG,
                    _operations._RigidTransformKernel,
                    _operations._ScalingKernel,
                    _operations._SharpCSGKernel,
                    _operations._BlendCSGKernel,
                    _operations._AffineBoundaryMap,
                    _superquadric.Superquadric,
                    _superquadric._SuperquadricKernel,
                    _sweeps.Extrusion,
                    _sweeps.Revolution,
                    _sweeps._ExtrusionKernel,
                    _sweeps._RevolutionKernel,
                ),
            ),
            (
                "phydrax.meshing",
                (
                    MeshCertificationInputs,
                    MeshCertificationSchedule,
                    MeshCertificationOutcome,
                    MeshCertificationReport,
                    GeometryAssociation,
                    GeometryAssociationKind,
                    GeometryAssociationProvenance,
                    GeometrySourceEntityRole,
                    _implicit_volume.PreparedAdaptiveImplicitVolume,
                    PreparedCompartmentVolume,
                    _level_set.LevelSetMeshAdaptation,
                    _level_set.LevelSetEvidence,
                    _assembly.MeshAssembly,
                    _assembly.MeshPart,
                    _assembly.MeshCarrierKind,
                    # Point-carrier parts are restored verbatim and checked by
                    # their owner's integrity validator (re-running the
                    # constructor would renormalize stored normals).
                    PointCloudPlan,
                    LocalStencilPolicy,
                    MortonAddressPlan,
                    _coupling.ContactCoupling,
                    _coupling.CouplingSearchEvidence,
                    _coupling.MeshCouplingKind,
                    _overset.OversetRegistration,
                    _overset.OversetPartSpec,
                    _overset.OversetPolicy,
                    _result.CellMeshingResult,
                    _result.MeshingComplianceReport,
                    _result.MeshingRuntimeInfo,
                    _audit.CellMeshAuditReport,
                    _audit.CellMeshAuditPolicy,
                    _audit.CellMeshAuditDisposition,
                    _audit.CellMeshAuditScope,
                    _quality.CellQualityReport,
                    _quality.CellQualityEvaluation,
                    _trace.MeshingTrace,
                    _trace.MeshingEvidenceBinding,
                    _trace.MeshingStageReport,
                    _trace.MeshingStageKind,
                    _trace.MeshingStageStatus,
                    _trace.MeshingDiagnostic,
                    _trace.MeshingDiagnosticSeverity,
                    _scope.MeshingScope,
                    _scope.MeshingEntityKind,
                    _organization.MeshPatch,
                    _organization.MeshZone,
                    _organization.MeshLabel,
                    _organization.MeshAttribute,
                    _organization.MeshZoneRole,
                    _organization.RegionRole,
                    _organization.MeshAttributeRole,
                    _organization.RegionBoundaryEvidence,
                    _organization.RegionMeshingEvidence,
                    _contracts.MeshingProviderInfo,
                    _contracts.MeshingCapability,
                    _contracts.MeshingDerivativeMode,
                    _contracts.MeshingExecutionMode,
                    _contracts.MeshingOperation,
                    _contracts.MeshingSourceKind,
                    _contracts.SurfaceMeshingSpec,
                    _contracts.MeshingLimits,
                    _contracts.CellFamilyPolicy,
                    _contracts.CellMeshingTarget,
                    _contracts.MeshQualityTarget,
                    _controls.ProtectedFeature,
                    _controls.FeatureKind,
                    _controls.PatchControl,
                    _controls.RegionControl,
                    _sizing.UniformSizeControl,
                    _sizing.CurvatureSizeControl,
                    _sizing.ProximitySizeControl,
                    NativeImplicitSource,
                    NativeMeshingPlan,
                    _contracts.ProviderSupportReport,
                    _sizing.SizeCompliancePolicy,
                    _sizing.SizeCombinationPolicy,
                    _sizing.SizeControlStrength,
                    NativePlanarSource,
                    NativeMeshingOptions,
                    NativeCurveSchedule,
                    NativeSurfaceSchedule,
                    NativeStructuredSchedule,
                    _result.CollectiveMeshEvidence,
                    _result.CollectiveMeshStorageBinding,
                    _adaptation.MarkedMeshAdaptation,
                    _adaptation.MeshAdaptationPolicy,
                    _adaptation.MetricMeshAdaptation,
                    _adaptation.RelocationMeshAdaptation,
                    _polyhedral_adaptation.PolyhedralMeshAdaptation,
                    _polyhedral_adaptation.PolyhedralAdaptationOperation,
                    _polyhedral_adaptation.PolyhedralAdaptationEvidence,
                    _periodic.PeriodicRefinement,
                    _periodic.PeriodicQuotientEvidence,
                    MeshMetricField,
                    MetricRemeshingCriterion,
                    MetricRemeshingEvidence,
                    MetricRemeshingStatus,
                    NativeExecutionRecord,
                    _decision.NativeDesignSourceState,
                    _decision.AdaptationAction,
                    _decision.PhysicalErrorEvidence,
                    _decision.DecisionBudget,
                    _decision.MeasuredAdaptationCost,
                    _decision.RouteFeasibility,
                    _decision.SolverAwareCandidate,
                    _decision.SolverAwareDecision,
                    _decision.FixedEpochDerivativeEvidence,
                    _proposals.LearnedMeshProposer,
                    _proposals.MeshProposalFeatures,
                    _proposals.MeshMarkingProposal,
                    _proposals.MeshProposalSafetyPolicy,
                    _proposals.MeshProposalProjection,
                    _proposals.MeshProposalTransaction,
                    _adaptation.MeshAdaptationResult,
                    _adaptation.PreparedMeshAdaptation,
                    _adaptation.MeshAdaptationRoute,
                    _adaptation.MeshAdaptationStatus,
                    _adaptation._AdaptationConstraints,
                    _adaptation.GeometryRealizationMeshAdaptation,
                    _curving.HighOrderCurvingResult,
                    _curving.CurvedGeometryEvidence,
                    _curving.HighOrderCurvingStatus,
                    _curving.HighOrderCurvingPolicy,
                    _bisection.BisectionCompatibility,
                    _bisection.BisectionEvidence,
                    _bisection.BisectionHierarchy,
                    _bisection.BisectionUniformRefinement,
                    _mixed_adaptation.MixedLayerColumns,
                    _mixed_adaptation.MixedAdaptationHierarchy,
                    _mixed_adaptation._Sibling,
                    _mixed_adaptation.MixedAdaptationEvidence,
                    _topology_edit.PeriodicEntityIdentityBank,
                    _topology_edit.PeriodicNonnestedGeometryAuthority,
                    _topology_edit.PeriodicVertexOrbitWitness,
                    _topology_edit.CellTopologyEdit,
                    _topology_edit.TopologyEditBlock,
                    _topology_edit.PolyhedralTopologyEditBlock,
                    _topology_edit.EntityRelations,
                    _topology_edit.PrescribedEntityIds,
                    _topology_edit.SharedFaceWitnesses,
                    _distribution.MeshDistribution,
                    _distribution.MeshDistributionTransition,
                    _distribution.MeshPartitionEvidence,
                    _distribution.MeshPartitionKind,
                    _distribution.MeshPartitionPolicy,
                    _lineage.CellMeshTransition,
                    _lineage.EntityLineage,
                    _lineage.MeshLineage,
                    _lineage.MeshTransitionKind,
                    _lineage.VertexInterpolationStencil,
                    AssociationPropagationPolicy,
                    BRepAssociationTransfer,
                    PlcAssociationTransfer,
                    SurfaceAssociationTransfer,
                    MappedReferenceAssociationTransfer,
                    ComposedAssociationTransfer,
                    ImplicitAssociationTransfer,
                    SurfaceSubdivisionBoundarySource,
                    SurfaceChartBoundarySource,
                    PreparedSurfaceCurveWitness,
                    _scope.MeshScopeProjection,
                    _organization.MeshAttributeProjection,
                    PublicationProjection,
                    InitialCollectiveMeshEvidence,
                    CompiledSurfaceDomain,
                    SourceSurfaceSizing,
                    SourceSurfaceMetric,
                    SourcePatchNeighborhood,
                    DeviceGenerationLayout,
                ),
            ),
            (
                "phydrax.discretization",
                (
                    _cell_mesh.CellMesh,
                    _cell_mesh.CellBlock,
                    _cell_geometry.CellGeometrySpec,
                    _cell_geometry._CoordinateTabulator,
                    _cell_geometry_validity.CellValidityCertificate,
                    _cell_geometry._SweptCoordinateTabulator,
                    _cell_complex.PolygonalConnectivity,
                    _cell_complex.IntervalConnectivity,
                    _cell_complex.TetrahedralConnectivity,
                    _topology.CellComplexTopology,
                    _topology.EntitySet,
                    _topology.EntitySubset,
                    _topology.OrientedIncidence,
                    _support.DiscreteSupport,
                    FiniteElementSpec,
                    SimplicialLocationPolicy,
                    FiniteElementDofMap,
                    FiniteElementPlan,
                    MetricNetworkPlan,
                    _discretization_core.DiscretizationKey,
                    _discretization_core.DiscretizationCapability,
                    _discretization_core.DiscretizationRole,
                    _discretization_spaces.DiscreteFieldSpace,
                    _discretization_spaces.EntityDofLayout,
                    DistributedHaloPlan,
                    FiniteElementDofSourceProjection,
                    _fem_distributed.FiniteElementGlobalDofOwnership,
                    _fem_distributed.OwnerLocalFiniteElementTransfer,
                    _fem_distributed.FiniteElementDofOwnershipPlan,
                    _fem_distributed.OwnerLocalFiniteElementDiscretization,
                    _fem_distributed.FiniteElementClosurePreparation,
                    _fem_distributed.FiniteElementExecutionLimits,
                    FiniteElementDiscretization,
                    FiniteElementRuntimeData,
                    FiniteElementBlockGeometry,
                    IntegrationDomain,
                    DiscreteMeasure,
                    _discretization_spaces.BlockDofLayout,
                    _discretization_core.PreparationReport,
                    _tensor_support.PreparedTensorGrid,
                    _tensor_support.GridLocation,
                    _axis.AxisDiscretization,
                    _axis_domain.AxisDomain,
                    _topology.TensorTopology,
                    _tensor_entities.StructuredAxis,
                    _tensor_entities.TensorEntityLayout,
                    _cell_mesh.CellMeshStorage,
                    _cell_geometry.CellGeometryStorageProjection,
                    _cell_geometry_validity.CellValidityPolicy,
                    _cell_geometry_transfer.CellGeometryTransitionPolicy,
                    _cell_geometry_transfer.CellGeometryTransition,
                    _cell_geometry_transfer.CellGeometryTransitionEvidence,
                    _cell_geometry_transfer.SourceGeometryRealization,
                    SurfaceChartWitness,
                    PreparedSurfaceChartDeformation,
                    PreparedSurfaceChartOccurrence,
                    PreparedSurfaceChartPiece,
                    _cell_geometry_transfer.SurfaceGeometryReconstruction,
                    PreparedSphereChartDeformation,
                    PreparedSphereChartOccurrence,
                    PreparedSphereChartPiece,
                    SphereGeometryReconstruction,
                    PreparedSurfaceChartFiniteVolumeContents,
                    PreparedSurfaceChartCompatibleTransfer,
                    _adaptive_simplex.AdaptiveSimplexLayout,
                    _adaptive_simplex.AdaptiveSimplexPolicy,
                    _adaptive_simplex.AdaptiveSimplexState,
                    _adaptive_simplex.MaskedSimplexMesh,
                    _partition.CellPartition,
                    FiniteElementTopologyTransfer,
                    FiniteElementFieldTransfer,
                    FiniteElementTransferEvidence,
                    TransferGeometryBinding,
                    MappedNestedRemapEvidence,
                    MappedSurfaceChartRemapEvidence,
                    PreparedUnstructuredConservativeRemap,
                    RemapPreparationFailure,
                    UnstructuredConservativeRemapPlan,
                    UnstructuredRemapReport,
                    _periodic_cell.PeriodicCell,
                    _periodic_topology.PeriodicIsometryGroup,
                    _periodic_topology.PeriodicMeshTopology,
                ),
            ),
            (
                "phydrax.discretization.iga",
                (
                    IsogeometricPlan,
                    SplineAxisPlan,
                    TensorSplineBasisSpec,
                    SplineSpanTopology,
                    BaseSpanId,
                    NURBSGeometryState,
                    IsogeometricFieldSpec,
                    IsogeometricQuadraturePolicy,
                    IsogeometricH1QualificationPolicy,
                ),
            ),
            (
                "phydrax.geometry.simplicial",
                (
                    _regions.PlanarMeshRegion,
                    _regions.SegmentMesh,
                    SegmentTopology,
                    _simplicial_mesh.TriangleMesh,
                    _simplicial_topology.TriangleTopology,
                ),
            ),
            ("phydrax.sparse", (EdgeRelation,)),
            (
                "phydrax.linear",
                (
                    EuclideanPairing,
                    DistributedPairing,
                    OperatorCapabilities,
                    OperatorProperties,
                    ArraySpace,
                    BlockSpace,
                    SparseLinearMap,
                    RowRelation,
                ),
            ),
            ("phydrax.solver", (FiniteElementAcceptedState,)),
            ("phydrax.geometry.design", (DerivativeTier, DesignQualificationEvidence)),
            ("phydrax.predicates", (PredicateMode,)),
            (
                "phydrax.optim",
                (
                    _optim_types.MinimizationResult,
                    _optim_types.OptimizationDiagnostics,
                    _optim_types.OptimizationProvenance,
                    _optim_types.OptimizationTermination,
                    _optim_types.OptimizationStatus,
                    _optim_types.ConstrainedOptimalityCertificate,
                ),
            ),
            (
                "phydrax.core",
                (
                    BVHBuildKind,
                    BVHBuildPolicy,
                    PackedBVH,
                    SpatialCoordinateContract,
                    SemanticProvenance,
                ),
            ),
            ("phydrax.units", (UnitDefinition, DimensionSignature)),
        )
        + native_family_artifact_types()
        + meshing_field_artifact_types()
        + (
            ()
            if _NATIVE_DESIGN_MODEL_TYPE is None
            else (("examples.curved_native_transfer", (_NATIVE_DESIGN_MODEL_TYPE,)),)
        )
    )


def register_meshing_source_artifacts() -> None:
    """Admit exact built-in scientific owners to the one canonical registry."""
    register_artifact_value_codec(
        ArtifactValueCodec(
            "phydrax.exterior:FormType",
            FormType,
            FormType.to_dict,
            FormType.from_dict,
        )
    )
    register_artifact_value_codec(
        ArtifactValueCodec(
            "phydrax.exterior:FormValueSpec",
            FormValueSpec,
            FormValueSpec.to_dict,
            FormValueSpec.from_dict,
        )
    )
    for namespace, types in _owning_artifact_types():
        for cls in types:
            if registered_artifact_value_id(cls) is None:
                register_artifact_value(f"{namespace}:{cls.__name__}", cls)


def _rebuild_source_root(node: Any, /) -> StrictModule | None:
    if isinstance(node, _intersection.BranchRootEndpoint):
        return _intersection.BranchRootEndpoint(
            node.root,
            node.curve,
            node.chart,
            source_pcurve=node.source_pcurve,
            source_side=node.source_side,
            source_first=node.source_first,
            source_last=node.source_last,
            periodic_shifts=node.periodic_shifts,
            affine_transforms=node.affine_transforms,
        )
    if isinstance(node, _intersection.TrimIntersectionRoot):
        return _intersection.TrimIntersectionRoot(
            node.first,
            node.second,
            parameter_lower=node.parameter_lower,
            parameter_upper=node.parameter_upper,
        )
    if isinstance(node, _intersection.CurveSurfaceIntersectionRoot):
        return _intersection.CurveSurfaceIntersectionRoot(
            node.curve,
            node.surface,
            parameter_lower=node.parameter_lower,
            parameter_upper=node.parameter_upper,
        )
    if isinstance(node, _intersection.TripleSurfaceIntersectionRoot):
        return _intersection.TripleSurfaceIntersectionRoot(
            node.first,
            node.second,
            node.third,
            parameter_lower=node.parameter_lower,
            parameter_upper=node.parameter_upper,
        )
    if isinstance(node, _intersection.IntersectionCurvePointRoot):
        return _intersection.IntersectionCurvePointRoot(node.curve, node.parameter)
    if isinstance(node, _intersection.TrimRootEndpoint):
        return _intersection.TrimRootEndpoint(
            node.root,
            node.operand,
            affine_parameter_offset=node.affine_parameter_offset,
            affine_parameter_scale=node.affine_parameter_scale,
            affine_transforms=node.affine_transforms,
        )
    if isinstance(node, _intersection.NativePeriodEndpoint):
        return _intersection.NativePeriodEndpoint(
            node.curve,
            node.patch,
            node.axis,
            rational=node.rational,
            turns=node.turns,
        )
    if isinstance(node, _root_bindings.BRepRootSupport):
        return _root_bindings.BRepRootSupport(node.patch, node.root)
    if isinstance(node, _root_bindings.BRepVertexRoot):
        return _root_bindings.BRepVertexRoot(
            node.primary,
            aliases=node.aliases,
            spatial_root=node.spatial_root,
            joint_root=node.joint_root,
        )
    if isinstance(node, _root_bindings.BRepPlacedVertex):
        return _root_bindings.BRepPlacedVertex(
            node.source_point,
            node.rotation,
            node.translation,
            node.source_entity_id,
            source_root=node.source_root,
        )
    return None


def _rebuild_brep_authority(node: Any, /) -> StrictModule | None:
    if isinstance(node, _model.BRepGeometry):
        names = (
            "vertex_points",
            "curves",
            "edge_curves",
            "edge_ranges",
            "edge_vertices",
            "pcurves",
            "coedge_edges",
            "coedge_senses",
            "face_loops",
            "shell_faces",
            "shell_orientations",
            "solid_shells",
            "occurrences",
            "assembly_containers",
            "vertex_roots",
            "edge_endpoint_roots",
            "coedge_endpoint_roots",
        )
        return _model.BRepGeometry(
            **{name: object.__getattribute__(node, name) for name in names}
        )
    if isinstance(node, _model.BRepModel):
        names = (
            "patches",
            "parameter_bounds",
            "orientation",
            "trim_domains",
            "topology",
            "mesh_vertices",
            "mesh_faces",
            "triangle_face_ids",
            "triangle_parameters",
            "physical_tags",
            "report",
            "geometry",
            "tessellation_deviation_bounds",
            "tessellation_normal_bounds",
            "mesh_vertex_source_dimensions",
            "mesh_vertex_source_indices",
            "mesh_vertex_parameters",
            "mesh_chart_restriction_vertices",
            "mesh_chart_restriction_edges",
            "mesh_chart_restriction_endpoint_parameters",
            "mesh_chart_restriction_parameters",
            "coedge_deviation_bounds",
            "triangle_occurrence_ids",
            "vertex_occurrence_ids",
        )
        return _model.BRepModel(
            coordinate_contract=node.report.coordinate_contract,
            **{name: object.__getattribute__(node, name) for name in names},
        )
    if isinstance(node, _model.BRepTopology):
        return _model.BRepTopology(
            face_edges=node.face_edges,
            edge_faces=node.edge_faces,
            face_wires=node.face_wires,
            solid_faces=node.solid_faces,
            solid_face_orientations=node.solid_face_orientations,
            num_vertices=node.num_vertices,
        )
    if isinstance(node, domain.MeshingDomain):
        authority = node.brep_authority
        if type(authority) is _model.BRepModel:
            return domain.MeshingDomain.from_brep(authority)
        if type(authority) is _model.BRepGeometry:
            return domain.MeshingDomain.from_brep_geometry(
                authority,
                tuple(patch.surface for patch in node.patches),
                np.asarray(
                    [patch.orientation for patch in node.patches], dtype=np.float64
                ),
                source_id=node.source_id,
                source_revision=node.source_revision,
                authority_id=node.authority_id,
            )
        if authority is not None:
            raise TypeError("Meshing domain requires its exact owning B-Rep authority.")
        return domain.MeshingDomain(
            node.patches,
            node.curves,
            node.corner_points.shape[0],
            source_id=node.source_id,
            source_revision=node.source_revision,
            regions=node.regions,
            tolerance=node.tolerance,
            accuracy=node.accuracy,
            source_indices=node.source_indices,
            source_occurrences=node.source_occurrences,
            region_source_indices=node.region_source_indices,
            region_source_occurrences=node.region_source_occurrences,
            source_kinds=node.source_kinds,
            authority_id=node.authority_id,
        )
    return None


def _rebuild_native_carrier(node: Any, /) -> StrictModule | None:
    from ..geometry.brep._placed import PlacedCurve, PlacedSurface

    if isinstance(node, _intersection_curve.AffinePCurve):
        return _intersection_curve.AffinePCurve(node.curve, node.matrix, node.offset)
    if isinstance(node, _intersection_curve.PeriodicPCurve):
        return _intersection_curve.PeriodicPCurve(
            node.source_curve, node.patch, node.period_shifts
        )
    if isinstance(node, _patches.OffsetSurface):
        return _patches.OffsetSurface(node.base, node.distance)
    if type(node) in (
        _patches.LineCurve,
        _patches.CircleCurve,
        _patches.EllipseCurve,
        _patches.BSplineCurve,
        _patches.PlanePatch,
        _patches.CylinderPatch,
        _patches.ConePatch,
        _patches.SpherePatch,
        _patches.TorusPatch,
        _patches.BSplineSurfacePatch,
        _patches.ExtrusionSurface,
        _patches.RevolutionSurface,
        _patches.RuledSurface,
        _intersection_curve.BernsteinSurfacePiece,
        _intersection_curve.BernsteinCurvePiece,
        _intersection_curve.IntersectionPCurve,
        _intersection_curve.CurveTrimSegment,
        PlacedSurface,
        PlacedCurve,
    ):
        return type(node)(
            **{
                member.name: object.__getattribute__(node, member.name)
                for member in fields(node)
            }
        )
    if isinstance(node, _intersection_curve.SurfaceRegion):
        return _intersection_curve.SurfaceRegion(node.patch, node.parameter_box)
    if isinstance(node, _intersection_curve.CurveRange):
        return _intersection_curve.CurveRange(node.curve, node.first, node.last)
    if isinstance(node, _intersection_curve.IntersectionCurve):
        names = (
            "chart_axes",
            "chart_start",
            "chart_end",
            "chart_pieces",
            "box_lower",
            "box_upper",
            "preconditioners",
            "contraction",
            "certified",
            "transition_shifts",
            "closed",
            "start_kind",
            "end_kind",
            "node_axes",
            "node_values",
            "node_pieces",
            "node_references",
            "node_period_shifts",
        )
        return _intersection_curve.IntersectionCurve(
            node.first,
            node.second,
            **{name: object.__getattribute__(node, name) for name in names},
        )
    if isinstance(node, domain.PatchCurveUse):
        return domain.PatchCurveUse(
            node.curve,
            node.pcurve,
            node.first,
            node.last,
            first_root=node.first_root,
            last_root=node.last_root,
            start_vertex_root=node.start_vertex_root,
            end_vertex_root=node.end_vertex_root,
            trim_curve=node.trim_curve,
        )
    if isinstance(node, domain.MeshingSurfacePatch):
        return domain.MeshingSurfacePatch(
            node.surface, node.loops, reversed=node.reversed
        )
    if isinstance(node, domain.MeshingDomainCurve):
        return domain.MeshingDomainCurve(node.start, node.end)
    if isinstance(node, domain.MeshingDomainRegion):
        return domain.MeshingDomainRegion(node.name, node.boundary)
    return None


def _rebuild_native_query(
    node: Any,
    model: _model.BRepModel | None,
    path: str,
    /,
    *,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None = None,
) -> StrictModule | None:
    if isinstance(node, _query.PreparedBRepQuery):
        if type(node.model) is not _model.BRepModel:
            raise ValueError(f"{path} requires its complete original owning BRepModel.")
        if model is not None and not _native_authority_equal(
            node.model, model, limits=limits, source_validation=source_validation
        ):
            raise ValueError(
                f"{path} differs from its exact defining BRepModel authority."
            )
        return _query.prepare_brep_query(
            node.model,
            policy=node.policy,
            projection_policy=node.projection_policy,
        )
    if isinstance(node, _query.NativeBRepProjection):
        if type(node.query.model) is not _model.BRepModel:
            raise ValueError(f"{path} requires its complete original owning BRepModel.")
        if model is not None and not _native_authority_equal(
            node.query.model, model, limits=limits, source_validation=source_validation
        ):
            raise ValueError(
                f"{path} differs from its exact defining BRepModel authority."
            )
        return _query.NativeBRepProjection(
            node.query.model, node.policy, node.embedding, node.query.policy
        )
    if isinstance(node, _source.BRepSource):
        return _source.BRepSource(node.model)
    if isinstance(node, _source._NativeBRepKernel):
        return _source._NativeBRepKernel(node.model, node.query)
    return None


def _rebuild_declarative_contract(
    node: Any,
    /,
    *,
    meshes: tuple[CellMesh, ...],
    results: tuple[_result.CellMeshingResult, ...] = (),
    theorems: Mapping[tuple[str, str], _result.AbstractCollectiveMeshTheorem]
    | None = None,
) -> StrictModule | None:
    if type(node) is SplineAxisPlan:
        return SplineAxisPlan(node.name, node.knots, degree=node.degree)
    if type(node) is TensorSplineBasisSpec:
        return TensorSplineBasisSpec(node.axes)
    if type(node) is SplineSpanTopology:
        return SplineSpanTopology(
            node.axis_names, node.span_indices, patch_id=node.patch_id
        )
    if isinstance(node, _scope.MeshingScope):
        if node._local_entity_universe is not None:
            matches = tuple(
                mesh
                for mesh in meshes
                if mesh.mesh_id == node.source_id
                and mesh.numeric_version == node.source_revision
            )
            projection = node._scope_projection
            if projection is not None:
                fresh = _rebuild_meshing_carrier(
                    projection,
                    meshes=meshes,
                    results=results,
                    theorems=theorems,
                )
                if not isinstance(fresh, _scope.MeshScopeProjection):
                    raise ValueError(
                        "Mesh-backed scope lacks its validated source projection replay."
                    )
                projection = fresh
            qualified = []
            for mesh in matches:
                if node.entity_dimension > mesh.topological_dimension:
                    continue
                entities = mesh.entity_set(node.entity_dimension)
                if (
                    entities.entity_set_id != node.entity_set_id
                    or not _native_authority_equal(
                        entities.entity_ids,
                        node._local_entity_universe,
                        limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
                    )
                ):
                    continue
                try:
                    rebuilt = _scope.MeshingScope(
                        node.source_id,
                        node.source_revision,
                        node.entity_kind,
                        node.entity_dimension,
                        node.entity_set_id,
                        node.global_entity_ids
                        if projection is None
                        else projection.members,
                        local_mesh=mesh,
                        _projection=projection,
                    )
                except ValueError:
                    continue
                if _native_authority_equal(
                    node, rebuilt, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
                ):
                    qualified.append((mesh, rebuilt))
            if not qualified:
                raise ValueError(
                    "Mesh-backed scope lacks its complete authenticated local constructor context."
                )
            if any(
                not _native_authority_equal(
                    qualified[0][0],
                    mesh,
                    limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
                )
                for mesh, _ in qualified[1:]
            ):
                storage_ids = tuple(
                    None if mesh.storage is None else mesh.storage.storage_id
                    for mesh, _ in qualified
                )
                raise ValueError(
                    f"Mesh-backed scope has conflicting complete primitives within its authenticated local context: {storage_ids}."
                )
            return qualified[0][1]
        return _scope.MeshingScope(
            node.source_id,
            node.source_revision,
            node.entity_kind,
            node.entity_dimension,
            node.entity_set_id,
            node.global_entity_ids,
        )
    if type(node) not in (
        _overset.OversetPartSpec,
        _overset.OversetPolicy,
        _scope.MeshingScope,
        NativeMeshingOptions,
        NativeCurveSchedule,
        NativeSurfaceSchedule,
        NativeStructuredSchedule,
        _contracts.SurfaceMeshingSpec,
        _contracts.MeshingLimits,
        _contracts.CellFamilyPolicy,
        _contracts.CellMeshingTarget,
        _contracts.MeshQualityTarget,
        _controls.ProtectedFeature,
        _controls.PatchControl,
        _controls.RegionControl,
        _sizing.UniformSizeControl,
        _sizing.CurvatureSizeControl,
        _sizing.ProximitySizeControl,
        _sizing.SizeCompliancePolicy,
        _audit.CellMeshAuditPolicy,
        FiniteElementSpec,
        SimplicialLocationPolicy,
        _adaptation.MarkedMeshAdaptation,
        _adaptation.MeshAdaptationPolicy,
        _adaptation.GeometryRealizationMeshAdaptation,
        _level_set.LevelSetMeshAdaptation,
        _polyhedral_adaptation.PolyhedralMeshAdaptation,
        FiniteElementDofSourceProjection,
        FiniteElementPlan,
        _discretization_core.DiscretizationKey,
        IsogeometricPlan,
        BaseSpanId,
        NURBSGeometryState,
        IsogeometricFieldSpec,
        IsogeometricQuadraturePolicy,
        IsogeometricH1QualificationPolicy,
        _discretization_spaces.DiscreteFieldSpace,
        _discretization_spaces.EntityDofLayout,
        IntegrationDomain,
        DiscreteMeasure,
        _discretization_spaces.BlockDofLayout,
        BlockSpace,
        DistributedPairing,
        _discretization_core.PreparationReport,
        FiniteElementAcceptedState,
        _decision.PhysicalErrorEvidence,
        _decision.DecisionBudget,
        _decision.RouteFeasibility,
        _decision.SolverAwareCandidate,
        _decision.SolverAwareDecision,
        _proposals.LearnedMeshProposer,
        _proposals.MeshProposalTransaction,
        _periodic.PeriodicRefinement,
        NativeImplicitSource,
        NativeMeshingPlan,
        _contracts.ProviderSupportReport,
        _axis.AxisDiscretization,
        _axis_domain.AxisDomain,
        _topology.TensorTopology,
        _tensor_support.GridLocation,
        _implicit_policy.ImplicitSurfacePolicy,
        _implicit_policy.ImplicitProjectionPolicy,
        certificate.MeshCertificateFinding,
        MixedDimensionalTransportState,
        _adaptation.MetricMeshAdaptation,
        _adaptation.RelocationMeshAdaptation,
        MeshMetricField,
        SurfaceChartWitness,
        CommonRefinementPolicy,
        _multi_contracts.MultiRegionSurfaceCounts,
        _multi_contracts.MultiRegionSurfaceCapacityPlan,
        _multi_contracts.MultiRegionSurfaceValidationPolicy,
        _label_extraction.LabelFieldSurfaceLineage,
        _label_extraction.LabelFieldSurfaceExtractionEvidence,
        _label_extraction.LabelFieldSurfaceExtractionResult,
        _curving.HighOrderCurvingPolicy,
        _curving.CurvedGeometryEvidence,
        _curving.HighOrderCurvingResult,
        _optim_types.MinimizationResult,
        _optim_types.OptimizationDiagnostics,
        _optim_types.OptimizationProvenance,
        _optim_types.OptimizationTermination,
        _optim_types.ConstrainedOptimalityCertificate,
        _adaptive_simplex.AdaptiveSimplexPolicy,
        _adaptive_simplex.AdaptiveSimplexLayout,
        _distribution.MeshPartitionPolicy,
        AssociationPropagationPolicy,
        BRepAssociationTransfer,
        PlcAssociationTransfer,
        SurfaceAssociationTransfer,
        MappedReferenceAssociationTransfer,
        ComposedAssociationTransfer,
        ImplicitAssociationTransfer,
        SurfaceSubdivisionBoundarySource,
        SurfaceChartBoundarySource,
        SurfaceSourceRootAtlas,
        SurfaceNativeRestrictionBoundarySource,
        SurfaceSourceCharts,
        domain.MeshingDomainBoundarySource,
    ):
        return None
    declared = {member.name for member in fields(node)}
    positional: list[Any] = []
    keywords: dict[str, Any] = {}
    for name, parameter in signature(type(node)).parameters.items():
        if (
            parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
            or name not in declared
        ):
            raise TypeError(
                f"{type(node).__name__} has no retained owning constructor input {name!r}."
            )
        value = object.__getattribute__(node, name)
        if parameter.kind is Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keywords[name] = value
    return type(node)(*positional, **keywords)


def _rebuild_meshing_carrier(
    node: Any,
    /,
    *,
    meshes: tuple[CellMesh, ...],
    results: tuple[_result.CellMeshingResult, ...] = (),
    implicit_geometries: Mapping[int, geometry.CompiledGeometry] | None = None,
    theorems: Mapping[tuple[str, str], _result.AbstractCollectiveMeshTheorem]
    | None = None,
) -> Any | None:
    if type(node) is _scope.MeshScopeProjection:
        if theorems is None:
            indexed: dict[tuple[str, str], _result.AbstractCollectiveMeshTheorem] = {}
            for result in results:
                evidence = result.collective_evidence
                if evidence is not None:
                    indexed.setdefault((evidence.evidence_id, evidence.mesh_id), evidence)
        else:
            indexed = dict(theorems)
        theorem = indexed.get((node.evidence_id, node.source_id))
        if theorem is None:
            raise ValueError(
                "A scientific scope projection requires its actual accepted source theorem."
            )
        from ..meshing._initial_certification import InitialCollectiveMeshEvidence

        if not isinstance(
            theorem, (_result.CollectiveMeshEvidence, InitialCollectiveMeshEvidence)
        ):
            raise TypeError(
                "Scientific scope projection requires its actual accepted theorem."
            )
        publication = node.publication

        def rebind(
            projection: _scope.MeshScopeProjection,
        ) -> _scope.MeshScopeProjection:
            # A mutually exclusive scope group shares one accepted inventory
            # object. Restoration does not preserve object sharing, so each peer
            # is rebound to it only after exact content equality.
            if projection.publication is not publication and not (
                _native_authority_equal(
                    projection.publication,
                    publication,
                    limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
                )
            ):
                raise ValueError(
                    "Exclusive scientific scopes reference different accepted inventories."
                )
            return _scope.MeshScopeProjection(
                theorem,
                publication,
                projection.source_revision,
                projection.dimension,
                projection.membership_name,
                exclusive_scopes=tuple(
                    rebind(other) for other in projection.exclusive_scopes
                ),
            )

        return rebind(node)
    if type(node) is _periodic.PeriodicQuotientEvidence:
        originals = tuple(
            mesh
            for mesh in meshes
            if mesh.periodic_topology is not None
            and mesh.periodic_topology.periodic_topology_id == node.periodic_topology_id
        )
        if not originals:
            raise ValueError(
                "Periodic quotient evidence requires its actual immutable mesh source."
            )
        rebuilt = tuple(_periodic.PeriodicQuotientEvidence(mesh) for mesh in originals)
        if any(
            jax.tree_util.tree_structure(value)
            != jax.tree_util.tree_structure(rebuilt[0])
            for value in rebuilt[1:]
        ):
            raise ValueError(
                "Periodic quotient evidence has ambiguous actual numerical source binding."
            )
        return rebuilt[0]
    if type(node) is PreparedSurfaceSourceSupport:
        return PreparedSurfaceSourceSupport(
            node.domain,
            node.original,
            node.root_parameters,
            node.root_boundary_source,
            maximum_support_queries=node.maximum_support_queries,
        )
    if type(node) in (
        MetricNetworkPlan,
        NeurofluidTransportUnits,
        NeurofluidTransportParameters,
        NeurofluidTransportCheckpoint,
    ):
        return replace(node)
    if type(node) in (
        _implicit_realization.ImplicitSurfacePlan,
        _implicit_projection.ImplicitPointProjectionPlan,
    ):
        original = (
            None if implicit_geometries is None else implicit_geometries.get(id(node))
        )
        if original is None:
            raise ValueError(
                "Implicit preparation requires its actual original compiled source in the same design closure."
            )
        if type(node) is _implicit_projection.ImplicitPointProjectionPlan:
            return _implicit_projection.ImplicitPointProjectionPlan(
                original,
                node.anchors,
                node.trust_radii,
                policy=node.policy,
                source_id=node.source_id,
                plan_id=node.plan_id,
            )
        inputs = {
            name: getattr(node, name)
            for name in signature(type(node)).parameters
            if name != "geometry"
        }
        return _implicit_realization.ImplicitSurfacePlan(geometry=original, **inputs)
    if type(node) is _tensor_support.PreparedTensorGrid:
        return _tensor_support.PreparedTensorGrid(
            node.axes,
            axis_names=node.axis_names,
            plan_id=node.plan_id,
            embedding_id=node.support.embedding_id,
            prepared_id=node.prepared_id,
        )
    if _NATIVE_DESIGN_MODEL_TYPE is not None and type(node) is _NATIVE_DESIGN_MODEL_TYPE:
        return _NATIVE_DESIGN_MODEL_TYPE(node.weight)
    if type(node) in (
        _proposals.MeshProposalFeatures,
        _proposals.MeshMarkingProposal,
        _proposals.MeshProposalSafetyPolicy,
        _proposals.MeshProposalProjection,
    ):
        source_id = (
            node.policy.source_result_id
            if type(node) is _proposals.MeshProposalProjection
            else node.source_result_id
        )
        sources = tuple(result for result in results if result.result_id == source_id)
        if len(sources) != 1:
            raise ValueError(
                "A proposal source record requires its actual unique original result in the same closure."
            )
        source = sources[0]
        if type(node) is _proposals.MeshProposalFeatures:
            return _proposals.MeshProposalFeatures(
                source,
                node.scope,
                node.values,
                feature_ids=node.feature_ids,
                feature_owner_id=node.feature_owner_id,
            )
        if type(node) is _proposals.MeshMarkingProposal:
            return _proposals.MeshMarkingProposal(
                source, node.scope, node.values, proposer_id=node.proposer_id
            )
        if type(node) is _proposals.MeshProposalSafetyPolicy:
            inputs = {
                name: getattr(node, name)
                for name in signature(type(node)).parameters
                if name != "source"
            }
            return _proposals.MeshProposalSafetyPolicy(source, **inputs)
        return _proposals.project_mesh_proposal(source, node.proposal, node.policy)
    if type(node) is _decision.NativeDesignSourceState:
        return replace(node)
    if type(node) is DesignQualificationEvidence:
        return replace(node)
    if type(node) is _decision.MeasuredAdaptationCost:
        names = (
            "preparation_seconds",
            "compilation_seconds",
            "solve_seconds",
            "transfer_seconds",
            "reanalysis_seconds",
            "decision_seconds",
        )
        if len(node.phase_seconds) != len(names):
            raise ValueError(
                "Measured adaptation cost requires its complete original phase table."
            )
        return _decision.MeasuredAdaptationCost(
            node.sample_id,
            node.candidate_id,
            **dict(zip(names, node.phase_seconds, strict=True)),
            peak_memory_bytes=node.peak_memory_bytes,
        )
    if type(node) is _decision.FixedEpochDerivativeEvidence:
        if len(node.qualifications) != 3:
            raise ValueError(
                "Fixed-epoch derivatives require all three original owner qualifications."
            )
        return _decision.FixedEpochDerivativeEvidence(
            node.epoch_id,
            node.parameter_id,
            geometry=node.qualifications[0],
            pde=node.qualifications[1],
            transfer=node.qualifications[2],
            routes=node.routes,
        )
    if type(node) is _multi_seeding.MultiRegionSurfaceSeed:
        return _multi_seeding.MultiRegionSurfaceSeed(
            node.positions,
            node.faces,
            node.face_labels,
            node.region_ids,
            node.region_kinds,
            source=node.source,
            vertex_global_ids=node.vertex_global_ids,
            face_global_ids=node.face_global_ids,
            vertex_sets=dict(node.vertex_sets),
        )
    if type(node) is _multi_topology.MultiRegionSurfaceTopology:
        return _multi_topology.MultiRegionSurfaceTopology(
            node.plan,
            node.faces[: node.face_count],
            node.face_labels[: node.face_count],
            node.region_ids,
            node.region_kinds,
            vertex_count=node.vertex_count,
            vertex_global_ids=node.vertex_global_ids[: node.vertex_count],
            face_global_ids=node.face_global_ids[: node.face_count],
            epoch=node.epoch,
            domain=node.domain,
        )
    from ..imaging import _core as _image_core

    if type(node) in (
        _image_core.MedicalImageAsset,
        _image_core.ImageIndexAffine,
        _image_core.LabelDefinition,
        _image_core.LabelOntology,
        _image_core.LabelVolume,
        _image_core.MedicalImageSupport,
        _image_core.DeidentificationEvidence,
        _image_asset.ImageFieldSpec,
        _measurement_asset.AcquisitionIdentity,
        _measurement_asset.DerivationRecord,
        _measurement_asset.MeasurementAsset,
        _measurement_field.SamplingSemantics,
        _measurement_field.IndependentStandardUncertainty,
        _measurement_field.QualityFlag,
        _measurement_field.QuantityField,
        _measurement_quantity.QuantitySpec,
        _measurement_quantity.ValueLayout,
        _measurement_time.SampleTimeAxis,
        _measurement_time.TemporalSampling,
    ):
        return replace(node)
    if type(node) is ReferenceArtifactManifest:
        inputs = {
            name: getattr(node, name)
            for name in signature(ReferenceArtifactManifest).parameters
        }
        inputs["nondimensionalization"] = dict(node.nondimensionalization)
        inputs["uncertainty"] = (
            None if node.uncertainty is None else dict(node.uncertainty)
        )
        return ReferenceArtifactManifest(inputs.pop("artifact_name"), **inputs)
    if type(node) is DistributedHaloPlan and node.owner_local:
        return DistributedHaloPlan(
            None,
            None,
            node.part_count,
            local_global_ids=node.local_global_ids,
            local_valid=node.local_valid,
            local_owned=node.local_owned,
            global_entity_count=node.entity_count,
            partition_index=node.partition_index,
            phase_send_indices=node.phase_send_indices,
            phase_receive_indices=node.phase_receive_indices,
            phase_send_valid=node.phase_send_valid,
            phase_receive_valid=node.phase_receive_valid,
            permutations=node.permutations,
            evidence_id=node.evidence_id,
        )
    if isinstance(node, NativeExecutionRecord):
        node.require_valid()
        return None
    if isinstance(node, FiniteElementTransferEvidence):
        rebuilt = FiniteElementTransferEvidence(
            dict(node.defects),
            node.tolerance,
            estimates=dict(node.estimates),
            bounds=dict(node.bounds),
            execution_evidence=node.execution_evidence,
        )
        if rebuilt.evidence_id != node.evidence_id or rebuilt.passed != node.passed:
            raise ValueError(
                "Restored transfer evidence changes its actual certified quantities or execution record."
            )
        return rebuilt
    if isinstance(node, PreparedUnstructuredConservativeRemap):
        rebuilt = PreparedUnstructuredConservativeRemap(
            node.refinement,
            node.plan,
            status=node.status,
            reason=node.reason,
            nested_evidence=node.nested_evidence,
            surface_chart_evidence=node.surface_chart_evidence,
            preparation_failure=node.preparation_failure,
            execution_evidence=node.execution_evidence,
        )
        if rebuilt.remap_id != node.remap_id or rebuilt.succeeded != node.succeeded:
            raise ValueError(
                "Restored remap changes its actual preparation, scope record, or acceptance identity."
            )
        return rebuilt
    if isinstance(
        node, (PreparedSurfaceChartDeformation, PreparedSphereChartDeformation)
    ):
        node.require_bound(
            node.source_mesh, node.source_geometry, node.target_mesh, node.target_geometry
        )
        return None
    if isinstance(node, SphereMaterialCellAtlas):
        node.require_bound(
            node.domain, node.source_mesh, node.source_geometry, node.coordinate_contract
        )
        return None
    if isinstance(node, SphereGeometryReconstruction):
        mesh = node.target_atlas.source_mesh
        node.target_atlas.require_bound(
            node.target_atlas.domain,
            mesh,
            node.geometry,
            node.target_atlas.coordinate_contract,
        )
        node.target_validity.require_bound(node.geometry, mesh=mesh)
        node.target_embedding.binding.require(mesh, node.geometry)
        if (
            node.target_geometry_id
            != _cell_geometry_validity.cell_geometry_id(node.geometry)
            or node.target_topology_id != mesh.topology_id
        ):
            raise ValueError(
                "Restored sphere reconstruction changes its actual coordinate map or topology."
            )
        return None
    if isinstance(node, _adaptive_discovery.ImplicitNormalFiberMacrochart):
        return _adaptive_discovery.ImplicitNormalFiberMacrochart(
            node.bounds,
            node.domain,
            node.maximum_level,
            node.source_id,
            node.normal_axis,
            node.integer_lower,
            node.integer_size,
            node.face_key,
        )
    if isinstance(node, _adaptive_discovery.ImplicitVolumeQuery):
        return _adaptive_discovery.ImplicitVolumeQuery(
            node.bounds,
            node.leaf_lower,
            node.leaf_upper,
            node.leaf_value_lower,
            node.leaf_value_upper,
            node.leaf_code_starts,
            node.domain,
            maximum_level=node.maximum_level,
        )
    if isinstance(node, _adaptive_discovery.AdaptiveImplicitSurfaceEvidence):
        return _adaptive_discovery.AdaptiveImplicitSurfaceEvidence(
            node.unresolved_boxes,
            node.unresolved_issues,
            **{
                member.name: object.__getattribute__(node, member.name)
                for member in fields(node)
                if member.name
                not in (
                    "unresolved_boxes",
                    "unresolved_issues",
                    "rounding_model",
                    "evidence_id",
                )
            },
        )
    if isinstance(node, _adaptive_discovery.AdaptiveImplicitSurface):
        # Replay the actual bounded discovery, including its original batching
        # and root/face decisions. Re-pinning enclosures to a differently
        # partitioned evaluation would change their numerical source identity.
        return _adaptive_discovery.discover_adaptive_implicit_surface(
            node.geometry,
            domain=node.volume.domain,
            policy=node.policy,
            source_id=node.source_id,
            maximum_root_solves=node.root_solve_capacity,
        )
    if isinstance(node, _adaptive_discovery.AdaptiveImplicitBoundarySource):
        return _adaptive_discovery.AdaptiveImplicitBoundarySource(
            node.geometry,
            node.surface,
            node.source_revision,
            maximum_distance_pairs=node.maximum_distance_pairs,
            maximum_scratch_bytes=node.maximum_scratch_bytes,
        )
    if isinstance(node, _implicit_volume.PreparedAdaptiveImplicitVolume):
        return _implicit_volume.PreparedAdaptiveImplicitVolume(
            node.geometry,
            node.surface,
            node.specification,
            node.schedule,
            node.source_revision,
            node.fidelity_tolerance,
            node.geometry_queries,
            node.source_work_units,
            node.preparation_seconds,
            node.coordinate_contract,
        )
    if isinstance(node, _simplicial_mesh.TriangleMesh):
        return _simplicial_mesh.TriangleMesh(
            node.vertices, node.faces, source_id=node.source_id
        )
    if isinstance(node, _simplicial_topology.TriangleTopology):
        return _simplicial_topology.TriangleTopology(
            node.faces, num_vertices=node.num_vertices
        )
    if isinstance(node, _certified_implicit.CertifiedImplicitTopology):
        return _certified_implicit.CertifiedImplicitTopology(
            node.cover, node.topology, premise=node.premise
        )
    if isinstance(node, NativePlanarSource):
        return NativePlanarSource(
            node.region, node.source_revision, embedded=node.embedded
        )
    if isinstance(node, certificate.ImplicitBoundarySource):
        if not bool(np.asarray(node.geometry.validity().accepted)):
            raise ValueError(
                "Implicit source requires its actual valid immutable numerical geometry state."
            )
        return certificate.ImplicitBoundarySource(
            node.geometry, source_id=node.source_id, spacing=node.spacing
        )
    if isinstance(node, certificate.ParametricCurveBoundarySource):
        return certificate.ParametricCurveBoundarySource(
            node.curves,
            node.parameter_ranges,
            source_id=node.source_id,
            source_revision=node.source_revision,
            covering_radius=node.covering_radius,
            maximum_samples=node.maximum_samples,
        )
    if isinstance(node, _regions.PlanarMeshRegion):
        offsets, edges = np.asarray(node.loop_offsets), np.asarray(node.edges)
        loops = tuple(
            tuple(edges[first:last, 0].tolist())
            for first, last in zip(offsets[:-1], offsets[1:], strict=True)
        )
        return _regions.PlanarMeshRegion(node.vertices, loops, feature_id=node.feature_id)
    if isinstance(node, _regions.SegmentMesh):
        return _regions.SegmentMesh(node.vertices, node.edges, source_id=node.source_id)
    if isinstance(node, _cell_mesh.CellBlock):
        return _cell_mesh.CellBlock(
            node.name,
            node.cell_kind,
            node.vertices,
            vertex_valid=node.vertex_valid,
            global_ids=node.global_ids,
        )
    if isinstance(node, CellMesh):
        if node.storage is not None:
            return CellMesh(
                node.coordinates,
                node.blocks,
                storage=node.storage,
                periodic_topology=node.periodic_topology,
                numeric_version=node.numeric_version,
            )
        return CellMesh(
            node.coordinates,
            node.blocks,
            vertex_global_ids=node.vertex_global_ids,
            entity_global_ids={
                dimension: node.entity_set(dimension).entity_ids
                for dimension in range(node.topological_dimension + 1)
            },
            periodic_topology=node.periodic_topology,
            numeric_version=node.numeric_version,
        )
    if isinstance(node, CellGeometrySpec):
        return CellGeometrySpec(
            dict(zip(node.block_names, node.elements, strict=True)),
            dict(zip(node.block_names, node.geometry_dofs, strict=True)),
            node.coordinates,
            storage=node.storage,
            restriction_source=node.restriction_source,
            exact_source=node.exact_source,
            periodic_source=node.periodic_source,
        )
    if isinstance(node, _cell_mesh.CellMeshStorage):
        return _cell_mesh.CellMeshStorage(
            node.global_entity_counts,
            node.entity_global_ids,
            node.entity_owner,
            partition_index=node.partition_index,
            partition_count=node.partition_count,
            logical_topology_id=node.logical_topology_id,
            logical_geometry_id=node.logical_geometry_id,
            evidence_id=node.evidence_id,
            logical_arrays=node.logical_arrays,
            local_coordinates=node.local_coordinates,
            local_blocks=node.local_blocks,
            maximum_materialization_bytes=node.maximum_materialization_bytes,
            local_physical_boundary_facets=node.local_physical_boundary_facets,
            local_neighborhood_complete=node.local_neighborhood_complete,
            neighborhood_depth=node.neighborhood_depth,
            global_coordinate_count=node.global_coordinate_count,
            coordinate_global_ids=node.coordinate_global_ids,
            coordinate_owner=node.coordinate_owner,
            local_geometry=node.local_geometry,
            geometry_source_blocks=dict(node.geometry_source_blocks),
            logical_coordinate_geometry_id=node.logical_coordinate_geometry_id,
            geometry_projection=node.geometry_projection,
        )
    if isinstance(node, _bisection.BisectionUniformRefinement):
        return _bisection.BisectionUniformRefinement(
            node.dimension,
            *node.host_arrays().values(),
            source=node.source,
        )
    if isinstance(node, _bisection.BisectionHierarchy):
        return _bisection.BisectionHierarchy(
            node.dimension,
            node.cell_global_ids,
            node.ordered_vertices,
            node.tags,
            node.generations,
            scientific_cell_ids=node.scientific_cell_ids,
            scientific_block_ids=node.scientific_block_ids,
            record_parent_ids=node.record_parent_ids,
            record_parent_blocks=node.record_parent_blocks,
            record_parent_rows=node.record_parent_rows,
            record_parent_vertices=node.record_parent_vertices,
            record_parent_tags=node.record_parent_tags,
            record_child_ids=node.record_child_ids,
            record_vertex_ids=node.record_vertex_ids,
            retired_entity_keys=node.retired_entity_keys,
            retired_entity_ids=node.retired_entity_ids,
            next_vertex_id=node.next_vertex_id,
            next_cell_id=node.next_cell_id,
            uniform_refinement=node.uniform_refinement,
        )
    if isinstance(node, _mixed_adaptation.MixedLayerColumns):
        return _mixed_adaptation.MixedLayerColumns(
            node.cell_ids,
            node.column_ids,
            node.interval_indices,
            hard_first_thickness=node.hard_first_thickness,
            axial_refinement=node.axial_refinement,
            allow_schedule_change=node.allow_schedule_change,
        )
    if isinstance(node, _mixed_adaptation.MixedAdaptationHierarchy):
        return _mixed_adaptation.MixedAdaptationHierarchy(
            node.records,
            next_cell_id=node.next_cell_id,
            next_vertex_id=node.next_vertex_id,
            retired_entities=node.retired_entities,
            quotient_entities=node.quotient_entities,
            layer_columns=node.layer_columns,
            identity_cells=node.identity_cells,
        )
    if (
        type(node) is _adaptation.MeshAdaptationResult
        and node.target.collective_evidence is None
    ):
        prepared = _adaptation.prepare_mesh_adaptation(
            node.source, node.request, policy=node.policy
        )
        if prepared.prepared_id != node.prepared_id:
            raise ValueError(
                "Accepted adaptation no longer binds its actual original source preparation."
            )
        outcome = _adaptation._RouteOutcome(
            node.status,
            node.target,
            node.transition,
            node.lineage,
            node.stencil,
            node.transfer,
            node.metric,
            node.evidence,
            node.hierarchy,
            node.common_refinement,
        )
        return _adaptation.MeshAdaptationResult(
            prepared,
            outcome,
            node.compliance,
            node.distribution,
            node.elapsed_seconds,
        )
    if isinstance(node, _adaptation.PreparedMeshAdaptation):
        if (
            node.policy.route
            is _adaptation.MeshAdaptationRoute.NATIVE_GEOMETRY_REALIZATION
        ):
            return _adaptation.prepare_mesh_adaptation(
                node.source, node.request, policy=node.policy
            )
        _adaptation._check_route_request(node.source, node.request, node.policy)
        constraints = _adaptation._resolve_constraints(
            node.source, node.request, node.policy
        )
        return _adaptation.PreparedMeshAdaptation(
            node.source,
            node.request,
            node.policy,
            constraints,
        )
    if isinstance(node, _result.CollectiveMeshEvidence):
        if node.raw_source_blocks is None or node.publication_source_blocks is None:
            raise ValueError(
                "Collective source evidence must retain both actual raw and independent publication source block axes."
            )
        return _result.CollectiveMeshEvidence(node, node.compiled_states)
    if isinstance(node, _result.CellMeshingResult):
        scopes, attributes = _organization.retained_mesh_organization_projections(node)
        required = (
            "mesh",
            "geometry",
            "coordinate_contract",
            "audit",
            "quality",
            "compliance",
            "trace",
            "provider",
            "runtime",
            "derivative_mode",
            "provenance",
        )
        optional = (
            "boundary",
            "patches",
            "zones",
            "labels",
            "attributes",
            "associations",
            "surface_source",
            "adapter_reports",
            "certification",
            "region_evidence",
            "region_boundary_evidence",
            "collective_evidence",
            "storage_binding",
            "collective_certificates",
            "execution_evidence",
        )
        return _result.CellMeshingResult(
            *(object.__getattribute__(node, name) for name in required),
            **{name: object.__getattribute__(node, name) for name in optional},
            scope_projections=scopes,
            attribute_projections=attributes,
        )
    if isinstance(node, _assembly.MeshPart):
        return _assembly.MeshPart(
            node.name, node.carrier, coordinate_contract=node.coordinate_contract
        )
    if isinstance(node, _assembly.MeshAssembly):
        return _assembly.MeshAssembly(node.parts, couplings=node.couplings)
    return _rebuild_declarative_contract(
        node, meshes=meshes, results=results, theorems=theorems
    )


def _native_authority_equal(
    first: Any,
    second: Any,
    /,
    *,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None = None,
    freshly_rebuilt: bool = False,
) -> bool:
    """Compare every authored field through bounded logical content identity.

    Object sharing and NumPy mutability are archive storage metadata, not
    geometric/source identity; the durable archive retains its complete recipe.
    """
    if (
        freshly_rebuilt
        and type(first) is _result.CellMeshingResult
        and type(second) is _result.CellMeshingResult
    ):
        # ``_rebuild_meshing_carrier`` reuses every admitted child authority and
        # invokes the complete constructor again. Its recomputed content-bound
        # result identity is therefore the bounded equality proof; lowering the
        # entire nested carrier as one recipe would duplicate the archive graph
        # and can exceed the unchanged per-recipe manifest limit.
        return first.result_id == second.result_id
    if (
        freshly_rebuilt
        and type(first) is _assembly.MeshPart
        and type(second) is _assembly.MeshPart
    ):
        if (
            first.name != second.name
            or first.carrier_kind is not second.carrier_kind
            or first.coordinate_contract.spatial_id
            != second.coordinate_contract.spatial_id
            or first.intrinsic_dimension != second.intrinsic_dimension
            or first.ambient_dimension != second.ambient_dimension
            or first.part_id != second.part_id
        ):
            return False
        return _native_authority_equal(
            first.carrier,
            second.carrier,
            limits=limits,
            source_validation=source_validation,
            freshly_rebuilt=True,
        )
    if source_validation is not None:
        return source_validation.equivalent(first, second, limits=limits)
    from .._model._structure import model_recipe_array_pairs

    pairs = model_recipe_array_pairs(first, second, limits=limits)
    return pairs is not None and all(
        left is right
        or logical_array_value_collection_digest({"value": left})
        == logical_array_value_collection_digest({"value": right})
        for left, right in pairs
    )


def _validate_native_authority(
    node: Any,
    path: str,
    model: _model.BRepModel | None,
    /,
    *,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None = None,
    meshes: tuple[CellMesh, ...],
    results: tuple[_result.CellMeshingResult, ...] = (),
    implicit_geometries: Mapping[int, geometry.CompiledGeometry] | None = None,
    theorems: Mapping[tuple[str, str], _result.AbstractCollectiveMeshTheorem]
    | None = None,
) -> None:
    if type(node) in (_level_set.LevelSetEvidence, PreparedCompartmentVolume):
        node.validate_source_integrity()
    if type(node) in (
        _geometry_compartments.CompartmentMeshingSource,
        _label_extraction.LabelFieldVolumeBinding,
        _label_extraction.LabelFieldSurfaceExtractionResult,
    ):
        node.validate_source_integrity()
    if type(node) is FiniteElementDofMap:
        node.validate_restored()
    if type(node) is MetricRemeshingEvidence:
        node.validate_restored()
    if type(node) is PreparedSurfaceCurveWitness:
        node.validate_restored()
    if type(node) is _periodic_cell.PeriodicCell:
        node.validate_restored()
    if type(node) is PointCloudPlan:
        _require_point_cloud_plan_integrity(node)
    if type(node) in (
        FiniteElementDiscretization,
        _fem_distributed.FiniteElementExecutionLimits,
        _fem_distributed.FiniteElementGlobalDofOwnership,
        _fem_distributed.FiniteElementDofOwnershipPlan,
        _fem_distributed.OwnerLocalFiniteElementDiscretization,
        _fem_distributed.OwnerLocalFiniteElementTransfer,
        _fem_distributed.FiniteElementClosurePreparation,
    ):
        node.validate_restored()
    if type(node) is _topology_edit.PeriodicNonnestedGeometryAuthority:
        source = node.source.periodic_topology
        if source is None or node.target.mesh_id != node.common_refinement.target_mesh_id:
            raise ValueError(
                "Nonnested periodic authority requires its actual original source and target mesh."
            )
        _topology_edit.require_periodic_nonnested_geometry(source, node.target, node)
    if type(node) is _topology_edit.PeriodicVertexOrbitWitness:
        authority = node.nonnested_geometry
        if authority is None:
            raise ValueError(
                "A persisted periodic orbit witness requires its actual nonnested geometry authority."
            )
        target = _periodic.bind_periodic_topology_edit(
            authority.source, authority.target, node
        )
        if not _native_authority_equal(
            target, authority.target, limits=limits, source_validation=source_validation
        ):
            raise ValueError(
                "Restored periodic orbit witness changes its actual target SCI descriptor."
            )
    if type(node) is _topology_edit.CellTopologyEdit:
        witness = node.periodic_orbits
        authority = None if witness is None else witness.nonnested_geometry
        if authority is None:
            raise ValueError(
                "A persisted edit requires its actual original nonnested source and target."
            )
        target, _, _ = _topology_edit.assemble_topology_edit(
            authority.source,
            node,
            numeric_version=authority.target.numeric_version,
        )
        if not _native_authority_equal(
            target, authority.target, limits=limits, source_validation=source_validation
        ):
            raise ValueError(
                "Restored topology edit changes its actual independently bound target."
            )
    if isinstance(node, _cell_geometry_transfer.SourceGeometryRealization):
        node.require_current()
    if type(node) is _decision.NativeDesignSourceState:
        node.validate_source_integrity()
    fresh = _rebuild_source_root(node)
    if fresh is None:
        fresh = _rebuild_brep_authority(node)
    if fresh is None:
        fresh = _rebuild_native_carrier(node)
    if fresh is None:
        fresh = _rebuild_native_query(
            node, model, path, limits=limits, source_validation=source_validation
        )
    if fresh is None:
        fresh = rebuild_native_family_value(node)
    if fresh is None:
        fresh = rebuild_meshing_field_record(node)
    if fresh is None:
        fresh = _rebuild_meshing_carrier(
            node,
            meshes=meshes,
            results=results,
            implicit_geometries=implicit_geometries,
            theorems=theorems,
        )
    if fresh is not None and not _native_authority_equal(
        node,
        fresh,
        limits=limits,
        source_validation=source_validation,
        freshly_rebuilt=True,
    ):
        raise ValueError(
            f"{path} differs from its freshly certified native source authority."
        )
    if type(node) in (
        _model.BRepEntityId,
        _model.BRepOccurrence,
        _model.BRepAssemblyContainer,
        _model.BRepQualifiedIncidence,
        _model.BRepImportReport,
    ):
        fresh_record = replace(node)
        if fresh_record != node:
            raise ValueError(f"{path} differs from its owning native source record.")
    if isinstance(node, PlanarEmbedding):
        if PlanarEmbedding(node.origin, node.x_axis, node.y_axis, node.normal) != node:
            raise ValueError(f"{path} differs from its owning native coordinate binding.")


def _validate_source_value(
    value: Any,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    source_validation: MeshingSourceValidation | None = None,
) -> None:
    """Refuse callbacks, live handles, cycles and nonowning plugin objects.

    Traversal includes every declared field, including static fields and frozen
    plain dataclasses; PyTree runtime leaves are not a scientific source closure.
    Arrays are inspected structurally without materializing global JAX arrays.
    """
    from .._model._structure import (
        model_structure_recipe,
        validate_model_structure_recipe,
    )

    register_meshing_source_artifacts()
    admitted = frozenset(cls for _, types in _owning_artifact_types() for cls in types)
    mesh_nodes: dict[int, CellMesh] = {}
    result_nodes: dict[int, _result.CellMeshingResult] = {}
    theorem_nodes: dict[tuple[str, str], _result.AbstractCollectiveMeshTheorem] = {}
    collected: set[int] = set()
    material_children: set[int] = set()
    image_children: set[int] = set()
    finite_element_children: set[int] = set()
    topology_edit_children: set[int] = set()
    grid_children: set[int] = set()
    design_plans: set[int] = set()
    implicit_geometries: dict[int, geometry.CompiledGeometry] = {}
    accepted_epoch_nodes: set[int] = set()
    if isinstance(value, Mapping):
        accepted = value.get("accepted_data")
        if (
            isinstance(accepted, Mapping)
            and type(accepted.get("accepted_epochs")) is tuple
        ):
            accepted_epoch_nodes.update(
                id(epoch) for epoch in accepted["accepted_epochs"]
            )

    def collect_meshes(node: Any, /) -> None:
        if id(node) in collected:
            return
        collected.add(id(node))
        if type(node) is _decision.NativeDesignSourceState:
            if type(node.plan.prepared) is not _implicit_realization.ImplicitSurfacePlan:
                raise TypeError(
                    "Original design state requires its actual fixed implicit source preparation."
                )
            design_plans.add(id(node.plan))
            implicit_geometries[id(node.plan.prepared)] = node.source.geometry
            implicit_geometries[id(node.plan.prepared.projection)] = node.source.geometry
        if type(node) is _tensor_support.PreparedTensorGrid:
            grid_children.update(
                id(child) for child in (*node.structured_axes, *node.entity_layouts)
            )
        if type(node) is _result.CellMeshingResult:
            result_nodes[id(node)] = node
        if isinstance(node, _result.AbstractCollectiveMeshTheorem):
            theorem_nodes.setdefault((node.evidence_id, node.mesh_id), node)
        if type(node) is _topology_edit.CellTopologyEdit:
            topology_edit_children.update(
                id(child)
                for child in (*node.blocks, *node.relations, *node.prescribed_entity_ids)
            )
            if node.shared_faces is not None:
                topology_edit_children.add(id(node.shared_faces))
        if type(node) is FiniteElementDiscretization:
            finite_element_children.add(id(node.default_runtime))
            finite_element_children.update(
                id(geometry) for block in node.block_geometries for geometry in block
            )
        if type(node) is _label_extraction.LabelFieldSurfaceExtractionResult:
            image_children.update(
                id(child) for child in (node.state, node.surface) if child is not None
            )
        if type(node) is PreparedSurfaceChartDeformation:
            for occurrence in node.occurrences:
                material_children.add(id(occurrence))
                material_children.update(id(piece) for piece in occurrence.pieces)
                material_children.update(id(piece) for piece in occurrence.root_pieces)
        if isinstance(node, CellMesh):
            mesh_nodes[id(node)] = node
        elif type(node) in (dict, frozendict, MappingProxyType):
            for child in node.values():
                collect_meshes(child)
        elif type(node) in (tuple, frozenset):
            for child in node:
                collect_meshes(child)
        elif type(node) in (
            _adaptation._AdaptationConstraints,
            _topology_edit.PeriodicNonnestedGeometryAuthority,
            _topology_edit.PeriodicVertexOrbitWitness,
            _topology_edit.CellTopologyEdit,
        ):
            for child in node:
                collect_meshes(child)
        elif type(node) in admitted and is_dataclass(node):
            for member in fields(node):
                collect_meshes(object.__getattribute__(node, member.name))

    collect_meshes(value)
    meshes = tuple(mesh_nodes.values())
    results = tuple(result_nodes.values())
    theorems = theorem_nodes
    active: set[int] = set()
    # A source closure is a DAG: an object reached again under the same owning
    # model and partition context was already admitted by identical checks.
    admitted_nodes: set[tuple[int, int, int | None]] = set()

    def visit(
        node: Any,
        path: str,
        model: _model.BRepModel | None = None,
        batch_parts: int | None = None,
    ) -> None:
        if type(node) is MeshingAcceptedEpoch and id(node) not in accepted_epoch_nodes:
            raise ValueError(
                f"{path} requires its actual original generation and complete ordered accepted epoch history."
            )
        if type(node) is NativeMeshingPlan and id(node) not in design_plans:
            raise ValueError(
                f"{path} requires its complete original native design source closure."
            )
        if (
            type(node)
            in (_tensor_entities.StructuredAxis, _tensor_entities.TensorEntityLayout)
            and id(node) not in grid_children
        ):
            raise ValueError(
                f"{path} requires its actual original materialized tensor grid."
            )
        if (
            type(node) in (PreparedSurfaceChartPiece, PreparedSurfaceChartOccurrence)
            and id(node) not in material_children
        ):
            raise ValueError(
                f"{path} lacks its original owning material deformation and source banks."
            )
        if (
            type(node)
            in (
                _label_extraction.MultiRegionSurfaceState,
                _label_extraction.PreparedMultiRegionSurface,
            )
            and id(node) not in image_children
        ):
            raise ValueError(
                f"{path} lacks its original extraction topology, state and validation controls."
            )
        if (
            type(node) in (FiniteElementRuntimeData, FiniteElementBlockGeometry)
            and id(node) not in finite_element_children
        ):
            raise ValueError(
                f"{path} lacks its original owning finite-element plan and source geometry."
            )
        if (
            type(node)
            in (
                _topology_edit.TopologyEditBlock,
                _topology_edit.PolyhedralTopologyEditBlock,
                _topology_edit.EntityRelations,
                _topology_edit.PrescribedEntityIds,
                _topology_edit.SharedFaceWitnesses,
            )
            and id(node) not in topology_edit_children
        ):
            raise ValueError(
                f"{path} lacks its original independently bound topology edit."
            )
        if isinstance(node, (jax.Array, np.ndarray)):
            if node.dtype.hasobject:
                raise TypeError(f"{path} cannot archive object arrays or live handles.")
            return
        if isinstance(node, np.dtype):
            if node.hasobject:
                raise TypeError(f"{path} cannot archive an object dtype.")
            return
        if isinstance(node, Enum):
            if type(node) not in admitted:
                raise TypeError(f"{path} has a nonowning enum type.")
            return
        if node is None or type(node) in (str, bool, int, float, bytes, Fraction):
            return
        if isinstance(node, np.generic):
            visit(node.item(), path)
            return
        if callable(node) and type(node) not in admitted:
            raise TypeError(
                f"{path} contains an unresolved callback/provider or live handle."
            )
        if id(node) in active:
            raise ValueError(f"{path} contains a cyclic source closure.")
        context = (id(node), id(model), batch_parts)
        if context in admitted_nodes:
            return
        active.add(id(node))
        if isinstance(node, _query.PreparedBRepQuery) and model is not None:
            if not _native_authority_equal(
                node.model, model, limits=limits, source_validation=source_validation
            ):
                raise ValueError(
                    f"{path} differs from its exact defining BRepModel authority."
                )
        source_model = (
            node.model
            if isinstance(node, (_source._NativeBRepKernel, _query.PreparedBRepQuery))
            else model
        )
        try:
            if type(node) is tuple:
                for index, child in enumerate(node):
                    visit(child, f"{path}[{index}]", source_model)
            elif type(node) in (
                _adaptation._AdaptationConstraints,
                _mixed_adaptation._Sibling,
                _mixed_adaptation.MixedAdaptationEvidence,
                _polyhedral_adaptation.PolyhedralAdaptationEvidence,
                _topology_edit.PeriodicNonnestedGeometryAuthority,
                _topology_edit.PeriodicVertexOrbitWitness,
                _topology_edit.CellTopologyEdit,
                _topology_edit.TopologyEditBlock,
                _topology_edit.PolyhedralTopologyEditBlock,
                _topology_edit.EntityRelations,
                _topology_edit.PrescribedEntityIds,
                _topology_edit.SharedFaceWitnesses,
            ):
                for name, child in zip(node._fields, node, strict=True):
                    visit(child, f"{path}.{name}", source_model)
                if type(node) in (
                    _topology_edit.PeriodicNonnestedGeometryAuthority,
                    _topology_edit.PeriodicVertexOrbitWitness,
                    _topology_edit.CellTopologyEdit,
                ):
                    _validate_native_authority(
                        node,
                        path,
                        source_model,
                        limits=limits,
                        source_validation=source_validation,
                        meshes=meshes,
                        results=results,
                        implicit_geometries=implicit_geometries,
                        theorems=theorems,
                    )
            elif type(node) is _topology_edit.PeriodicEntityIdentityBank:
                # Immutable keyed identity payload: its owner validates it exactly.
                _periodic._require_periodic_identity_bank(node)
            elif type(node) is frozenset:
                for child in node:
                    visit(child, f"{path}.item", source_model)
            elif type(node) in (dict, frozendict, MappingProxyType):
                for key, child in node.items():
                    if type(key) is not str:
                        raise TypeError(f"{path} requires exact string closure keys.")
                    visit(child, f"{path}.{key}", source_model)
            elif type(node) in (FormType, FormValueSpec) and type(node) in admitted:
                # These exact native-owned immutable values are not dataclasses.
                # Admit their complete explicit payload through the same bounded
                # constructor codec used by the model recipe, including its
                # array/callback rejection and canonical re-encoding checks.
                validate_model_structure_recipe(
                    model_structure_recipe(node), limits=limits
                )
                fresh = rebuild_meshing_field_record(node)
                if type(fresh) is not type(node) or fresh != node:
                    raise ValueError(
                        f"{path} differs from its native immutable value authority."
                    )
            elif type(node) in admitted and is_dataclass(node):
                batched = batch_parts is not None and type(node) in (
                    _adaptive_simplex.AdaptiveSimplexState,
                    _adaptive_simplex.MaskedSimplexMesh,
                )
                if isinstance(node, StrictModule) and type(node)._strict_contract_:
                    validate(node)
                for member in fields(node):
                    child = object.__getattribute__(node, member.name)
                    child_parts = (
                        node.partition_count
                        if type(node) is _result.CollectiveMeshEvidence
                        and member.name in ("initial_states", "compiled_states")
                        else batch_parts
                        if batched and member.name == "mesh"
                        else None
                    )
                    if (
                        batched
                        and isinstance(child, jax.Array)
                        and child.shape[0] != batch_parts
                    ):
                        raise ValueError(
                            f"{path}.{member.name} differs from its typed collective partition axis."
                        )
                    child_model = (
                        None
                        if type(node) is _query.PreparedBRepQuery
                        and member.name == "world_query"
                        else source_model
                    )
                    visit(child, f"{path}.{member.name}", child_model, child_parts)
                _validate_native_authority(
                    node,
                    path,
                    source_model,
                    limits=limits,
                    source_validation=source_validation,
                    meshes=meshes,
                    results=results,
                    implicit_geometries=implicit_geometries,
                    theorems=theorems,
                )
            else:
                raise TypeError(
                    f"{path} has nonowning source type {type(node).__name__}; callbacks/providers/live handles cannot be archived."
                )
            admitted_nodes.add(context)
        finally:
            active.remove(id(node))

    visit(value, "source_closure")


def _primary_generation_source(
    records: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    source_validation: MeshingSourceValidation | None = None,
) -> NativeGenerationSource | None:
    singular = records.get("generation_source")
    if singular is not None:
        if not is_native_generation_source(singular):
            raise TypeError(
                "generation_source requires an owning native authored source."
            )
        return singular
    sources = records.get("generation_sources")
    registration = records.get("registration")
    if sources is None or registration is None:
        return None
    if (
        not isinstance(sources, Mapping)
        or type(registration) is not _overset.OversetRegistration
    ):
        raise TypeError("Generation source maps require the owning overset registration.")
    request = records["certification_inputs"]
    named = records.get("primary_generation_part")
    if named is not None:
        if type(named) is not str or named not in sources:
            raise ValueError(
                "Primary generation part must explicitly name one retained registration occurrence."
            )
        selected = [part for part in registration.assembly.parts if part.name == named]
    else:
        selected = [
            part
            for part in registration.assembly.parts
            if type(part.carrier) is _result.CellMeshingResult
            and part.carrier.certification is not None
            and part.carrier.certification.request.request_id == request.request_id
        ]
    if len(selected) != 1:
        raise ValueError(
            "Primary certification request must identify one actual registered carrier; repeated requests require primary_generation_part."
        )
    part = selected[0]
    if (
        type(part.carrier) is not _result.CellMeshingResult
        or part.carrier.certification is None
    ):
        raise ValueError(
            "Primary generation part must retain its complete certified scientific carrier."
        )
    if not _native_authority_equal(
        part.carrier.certification.request,
        request,
        limits=limits,
        source_validation=source_validation,
    ):
        raise ValueError(
            "Primary generation occurrence must bind the complete retained scientific request."
        )
    if not _native_authority_equal(
        part.carrier.certification,
        records["report"],
        limits=limits,
        source_validation=source_validation,
    ):
        raise ValueError(
            "Primary generation occurrence must bind the complete retained scientific report."
        )
    source = sources[part.name]
    if not is_native_generation_source(source):
        raise TypeError(
            "Generation registration requires its original owning native source."
        )
    return source


def _validate_generation_registration(
    records: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None = None,
) -> None:
    registration = records.get("registration")
    if registration is None:
        if {
            "generation_sources",
            "generation_specifications",
            "primary_generation_part",
        } & set(records):
            raise ValueError(
                "Named generation sources and hard requests require their full registration."
            )
        part, source = records.get("generation_part"), records.get("generation_source")
        specification, options = (
            records.get("generation_specification"),
            records.get("generation_options"),
        )
        if any(value is not None for value in (part, source, specification, options)):
            if (
                type(part) is not _assembly.MeshPart
                or source is None
                or specification is None
                or options is None
            ):
                raise ValueError(
                    "Single native generation requires its whole MeshPart, original source, hard request and options."
                )
            validate_registered_native_part(
                part, source, specification, options, limits=limits
            )
            if not _native_authority_equal(
                part.carrier.certification,
                records["report"],
                limits=limits,
                source_validation=source_validation,
            ):
                raise ValueError(
                    "Single generation part must retain the complete primary scientific report."
                )
            if not _native_authority_equal(
                part.carrier.certification.request,
                records["certification_inputs"],
                limits=limits,
                source_validation=source_validation,
            ):
                raise ValueError(
                    "Single generation part must bind the complete primary scientific request."
                )
            if not _native_authority_equal(
                part.carrier.associations,
                records["associations"],
                limits=limits,
                source_validation=source_validation,
            ):
                raise ValueError(
                    "Single generation part must retain every original qualified association."
                )
        return
    if type(registration) is not _overset.OversetRegistration:
        raise TypeError(
            "registration must be the exact owning OversetRegistration descriptor."
        )
    if {"generation_part", "generation_source", "generation_specification"} & set(
        records
    ):
        raise ValueError(
            "Named registration and single-part generation roles cannot be mixed."
        )
    sources = records.get("generation_sources")
    specifications = records.get("generation_specifications")
    options = records.get("generation_options")
    names = {part.name for part in registration.assembly.parts}
    if (
        not isinstance(sources, Mapping)
        or not isinstance(specifications, Mapping)
        or set(sources) != names
        or set(specifications) != names
    ):
        raise ValueError(
            "Full registration requires original generation sources and hard requests for every part."
        )
    if (
        not isinstance(options, Mapping)
        or set(options) != names
        or any(type(value) is not NativeMeshingOptions for value in options.values())
    ):
        raise ValueError(
            "Generation options must retain an exact native algorithm declaration for every part."
        )
    for part in registration.assembly.parts:
        source, specification = sources[part.name], specifications[part.name]
        validate_registered_native_part(
            part,
            source,
            specification,
            options[part.name],
            limits=limits,
        )
    registration.prepare().require_complete()


def _validate_neurofluid_transport_role(records: Mapping[str, Any], /) -> None:
    """Bind the complete mixed-carrier owner to its original scientific root."""
    from ..meshing._assembly import MeshPart

    accepted = records.get("accepted_data")
    if not isinstance(accepted, Mapping) or set(accepted) != {"neurofluid_transport"}:
        raise ValueError(
            "Neurofluid transport requires its sole complete owning checkpoint role."
        )
    checkpoint = accepted["neurofluid_transport"]
    if type(checkpoint) is not NeurofluidTransportCheckpoint:
        raise TypeError(
            "Neurofluid transport requires the exact registered physical checkpoint."
        )
    rebuilt = replace(checkpoint)
    if not _native_authority_equal(
        checkpoint, rebuilt, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
    ):
        raise ValueError(
            "Neurofluid checkpoint differs from its complete owning reconstruction."
        )
    generation = records.get("generation_part")
    if (
        type(generation) is not MeshPart
        or type(generation.carrier) is not _result.CellMeshingResult
    ):
        raise TypeError(
            "Neurofluid transport requires its actual original generation part."
        )
    bindings = (
        (records.get("generation_source"), checkpoint.source),
        (generation.carrier, checkpoint.result),
        (records.get("accepted_target"), checkpoint.result),
        (records.get("certification_inputs"), checkpoint.result.certification.request),
        (records.get("report"), checkpoint.result.certification),
        (records.get("associations"), checkpoint.result.associations),
    )
    if any(
        not _native_authority_equal(actual, expected, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
        for actual, expected in bindings
    ):
        raise ValueError(
            "Neurofluid checkpoint changes its original source, carrier, or complete theorem bindings."
        )
    if "field_declarations" in records:
        raise ValueError(
            "Mixed-carrier transport is owned by its physical checkpoint, not fictitious mesh-part fields."
        )


def _validate_meshing_accepted_roles(records: Mapping[str, Any], /) -> None:
    """Admit only authored physical arrays and owning accepted adaptation facts."""
    accepted = records.get("accepted_data")
    if accepted is None:
        return
    if not isinstance(accepted, Mapping) or set(accepted) - {
        "fields",
        "adaptation_results",
        "adaptation_event",
        "accepted_epochs",
        "neurofluid_transport",
    }:
        raise ValueError(
            "Accepted scientific data requires exact physical field or accepted epoch roles."
        )
    if "neurofluid_transport" in accepted:
        _validate_neurofluid_transport_role(records)
        return
    if "accepted_epochs" in accepted:
        if set(accepted) != {"accepted_epochs"}:
            raise ValueError(
                "Accepted epochs own the sole complete ordered field and transition history."
            )
        validate_meshing_accepted_epochs(records)
        return
    if "fields" in accepted:
        values = accepted["fields"]
        if not isinstance(values, Mapping) or any(
            type(name) is not str
            or not name.strip()
            or not isinstance(value, (jax.Array, np.ndarray))
            for name, value in values.items()
        ):
            raise ValueError(
                "Accepted fields must retain exact named numerical arrays, not nested containers."
            )
        if "field_declarations" not in records:
            raise ValueError(
                "Accepted fields require their original complete physical field declarations."
            )
    if "adaptation_results" in accepted:
        from ..meshing._adaptation import MeshAdaptationResult

        results = accepted["adaptation_results"]
        if (
            type(results) is not tuple
            or not results
            or any(type(value) is not MeshAdaptationResult for value in results)
        ):
            raise TypeError(
                "Accepted adaptation_results must retain a nonempty tuple of exact MeshAdaptationResult owners."
            )


def _validate_serial_accepted_target(
    records: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None = None,
) -> None:
    """Bind one dense adapted target to its complete accepted transition event."""
    target = records.get("accepted_target")
    if target is None:
        return
    from ..meshing._lineage import MeshLineage

    if records.get("generation_part") is not None:
        raise ValueError(
            "An accepted adapted target must not be mislabeled as original generation."
        )
    if (
        type(target) is not _result.CellMeshingResult
        or target.mesh.storage is not None
        or target.certification is None
        or not target.certification.passed
        or not target.audit.passed
        or not target.compliance.passed
    ):
        raise ValueError(
            "accepted_target must be one dense, audited, compliant, certified CellMeshingResult."
        )
    bindings = (
        (records.get("certification_inputs"), target.certification.request),
        (records.get("report"), target.certification),
        (records.get("associations"), target.associations),
    )
    if any(
        not _native_authority_equal(
            actual,
            expected,
            limits=limits,
            source_validation=source_validation,
        )
        for actual, expected in bindings
    ):
        raise ValueError(
            "Accepted target changes its certification inputs, report, or source associations."
        )
    accepted = records.get("accepted_data")
    event = (
        None if not isinstance(accepted, Mapping) else accepted.get("adaptation_event")
    )
    required_event = {
        "source_result_id",
        "source_mesh_id",
        "source_topology_id",
        "target_result_id",
        "target_mesh_id",
        "target_topology_id",
        "transition_id",
        "lineage",
        "accepted_event_id",
    }
    if not isinstance(event, Mapping) or set(event) != required_event:
        raise ValueError(
            "Accepted target requires one complete canonical adaptation event."
        )
    lineage = event["lineage"]
    values = {
        name: event[name] for name in required_event - {"lineage", "accepted_event_id"}
    }
    if not isinstance(lineage, MeshLineage) or any(
        type(value) is not str or not value for value in values.values()
    ):
        raise TypeError(
            "Accepted adaptation event identities and lineage must be explicit."
        )
    identity = canonical_fingerprint(
        {
            "kind": "accepted-serial-adaptation-event",
            **values,
            "lineage": lineage.lineage_id,
        }
    )
    if (
        event["accepted_event_id"] != identity
        or event["target_result_id"] != target.result_id
        or event["target_mesh_id"] != target.mesh.mesh_id
        or event["target_topology_id"] != target.mesh.topology_id
        or event["source_result_id"] == target.result_id
        or event["source_mesh_id"] == target.mesh.mesh_id
        or lineage.source_topology_id != event["source_topology_id"]
        or lineage.target_topology_id != target.mesh.topology_id
    ):
        raise ValueError(
            "Accepted target lacks its exact source-to-target lineage and accepted-event identity."
        )
    if (
        target.certification.mesh_id != target.mesh.mesh_id
        or target.certification.topology_id != target.mesh.topology_id
        or target.certification.geometry_id != target.audit.geometry_id
    ):
        raise ValueError(
            "Accepted target certification changes its mesh, geometry, or audit authority."
        )


def _validate_collective_target(
    records: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None = None,
) -> None:
    """Join the retained original source and every exact accepted local result."""
    target = records.get("collective_target")
    accepted = records.get("accepted_data", {})
    results = accepted.get("adaptation_results", ())
    if target is None:
        if any(result.target.mesh.storage is not None for result in results):
            raise ValueError(
                "Accepted distributed results require their explicit collective_target role."
            )
        return
    if type(target) is not _result.CellMeshingResult or target.mesh.storage is None:
        raise TypeError(
            "collective_target must be an exact accepted owner-local CellMeshingResult."
        )
    original = _result.require_original_meshing_source(target)
    from ..meshing._initial_certification import InitialCollectiveMeshEvidence

    if isinstance(original, InitialCollectiveMeshEvidence):
        authored = {
            "generation_source": original.authored_source,
            "generation_specification": original.specification,
            "generation_schedule": original.schedule,
            "generation_layout": original.layout,
        }
        for name, expected in authored.items():
            if name not in records or not _native_authority_equal(
                records[name],
                expected,
                limits=limits,
                source_validation=source_validation,
            ):
                raise ValueError(
                    f"Initial archive must retain its exact authored {name} inputs."
                )
    else:
        certification = original.certification
        if certification is None:
            raise ValueError(
                "Collective scientific source lacks its actual full original theorem."
            )
        if not _native_authority_equal(
            certification,
            records["report"],
            limits=limits,
            source_validation=source_validation,
        ):
            raise ValueError(
                "Collective archive must preserve the complete original source theorem."
            )
        if not _native_authority_equal(
            original.associations,
            records["associations"],
            limits=limits,
            source_validation=source_validation,
        ):
            raise ValueError(
                "Collective archive must preserve all original qualified source associations."
            )
    from ..meshing._device_adaptation import validate_partitioned_mesh_evidence

    current = target
    while current.collective_evidence is not None:
        evidence = current.collective_evidence
        if source_validation is not None:
            source_validation.require_epoch(evidence)
        elif isinstance(evidence, InitialCollectiveMeshEvidence):
            evidence.require_current()
        else:
            storage = current.mesh.storage
            if storage is None:
                raise ValueError(
                    "Collective archive predecessor lacks its physical numerical storage."
                )
            validate_partitioned_mesh_evidence(storage, evidence)
        if isinstance(evidence, InitialCollectiveMeshEvidence):
            break
        current = evidence.source
    epoch = target.collective_evidence
    if epoch is None:
        raise ValueError("Collective target lacks its consumed numerical theorem.")
    for result in results:
        evidence = result.target.collective_evidence
        if (
            not isinstance(evidence, _result.CollectiveMeshEvidence)
            or evidence.evidence_id != epoch.evidence_id
            or result.target.mesh.mesh_id != target.mesh.mesh_id
            or result.prepared_id != evidence.preparation.prepared_id
            or result.request.request_id != evidence.preparation.request.request_id
            or result.policy.policy_id != evidence.preparation.policy.policy_id
            or result.source.result_id != evidence.source.result_id
        ):
            raise ValueError(
                "Accepted results must consume the same exact request, source and collective epoch."
            )
        outcome = _adaptation._RouteOutcome(
            result.status,
            result.target,
            result.transition,
            result.lineage,
            result.stencil,
            result.transfer,
            result.metric,
            result.evidence,
            result.hierarchy,
            result.common_refinement,
        )
        fresh = _adaptation.MeshAdaptationResult(
            evidence.preparation,
            outcome,
            result.compliance,
            result.distribution,
            result.elapsed_seconds,
        )
        if not _native_authority_equal(
            result, fresh, limits=limits, source_validation=source_validation
        ):
            raise ValueError(
                "Accepted adaptation result differs from its exact owning reconstruction."
            )


@finite_element_restoration_validation_scope()
def validate_meshing_source_closure(
    value: Any,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    source_logical_arrays: Mapping[str, jax.Array] | None = None,
    source_logical_content_digest: str | None = None,
    maximum_chunk_bytes: int | None = None,
    source_validation: MeshingSourceValidation | None = None,
) -> None:
    """Validate one complete checkpoint closure or one owning source record."""
    if source_validation is not None and source_logical_arrays is not None:
        if source_logical_content_digest is None:
            raise ValueError(
                "Shared source validation requires its actual admitted content identity."
            )
        source_validation.require_binding(
            source_logical_arrays, source_logical_content_digest
        )
    if source_validation is None and source_logical_arrays is not None:
        if source_logical_content_digest is None or not isinstance(value, Mapping):
            raise ValueError(
                "Admitted shared source banks require their exact content and owning closure."
            )
        source_validation = validate_collective_mesh_source_epochs(
            (value,),
            source_logical_arrays=source_logical_arrays,
            source_logical_content_digest=source_logical_content_digest,
            maximum_chunk_bytes=maximum_chunk_bytes,
        )
    if (
        isinstance(value, Mapping)
        and value.get("accepted_target") is not None
        and value.get("generation_part") is not None
    ):
        raise ValueError(
            "An accepted adapted target must not be mislabeled as original generation."
        )
    _validate_source_value(value, limits=limits, source_validation=source_validation)
    if not isinstance(value, Mapping):
        return
    from ..meshing._initial_certification import InitialCollectiveMeshEvidence

    target = value.get("collective_target")
    if isinstance(target, _result.CellMeshingResult) and isinstance(
        _result.require_original_meshing_source(target),
        InitialCollectiveMeshEvidence,
    ):
        required_initial = {
            "collective_target",
            "generation_source",
            "generation_specification",
            "generation_schedule",
            "generation_layout",
        }
        optional_initial = {"accepted_data", "field_declarations", "generation_options"}
        if (
            not required_initial <= set(value)
            or set(value) - required_initial - optional_initial
        ):
            raise ValueError(
                "Initial source closure requires its exact authored generation inputs, not a fabricated report."
            )
        _validate_meshing_accepted_roles(value)
        _validate_collective_target(
            value, limits=limits, source_validation=source_validation
        )
        validate_meshing_field_declarations(value)
        return
    required = {"certification_inputs", "report", "associations"}
    optional = {
        "source",
        "generation_source",
        "generation_sources",
        "generation_specifications",
        "registration",
        "accepted_data",
        "generation_options",
        "generation_part",
        "generation_specification",
        "field_declarations",
        "primary_generation_part",
        "collective_target",
        "accepted_target",
    }
    if not required <= set(value) or set(value) - required - optional:
        raise ValueError(
            "Source closure requires its exact certification roots and declared source/registration/accepted-data roles."
        )
    inputs = value["certification_inputs"]
    report = value["report"]
    associations = value["associations"]
    if type(associations) is not tuple or any(
        type(item) is not GeometryAssociation for item in associations
    ):
        raise TypeError(
            "Source closure associations must be a tuple of exact GeometryAssociation records."
        )
    if (
        type(inputs) is not MeshCertificationInputs
        or type(report) is not MeshCertificationReport
    ):
        raise TypeError("Source closure requires the exact owning request and report.")
    source = value.get("source", inputs.source)
    validate_restored_meshing_source_bindings(
        source,
        inputs,
        report,
        limits=limits,
        source_validation=source_validation,
    )
    generation_source = _primary_generation_source(
        value, limits=limits, source_validation=source_validation
    )
    for association in associations:
        _validate_association_binding(
            inputs, association, generation_source=generation_source
        )
    _validate_serial_accepted_target(
        value,
        limits=limits,
        source_validation=source_validation,
    )
    _validate_generation_registration(
        value, limits=limits, source_validation=source_validation
    )
    _validate_meshing_accepted_roles(value)
    _validate_collective_target(value, limits=limits, source_validation=source_validation)
    validate_meshing_field_declarations(value)


def validate_restored_meshing_source_bindings(
    source: Any,
    certification_inputs: MeshCertificationInputs,
    report: MeshCertificationReport,
    /,
    *,
    association: GeometryAssociation | None = None,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    source_validation: MeshingSourceValidation | None = None,
) -> None:
    """Validate restored scientific ownership; acceptance still requires renewal."""
    _validate_source_value(
        {
            "source": source,
            "certification_inputs": certification_inputs,
            "report": report,
            "associations": () if association is None else (association,),
        },
        limits=limits,
        source_validation=source_validation,
    )
    if (
        type(certification_inputs) is not MeshCertificationInputs
        or type(report) is not MeshCertificationReport
    ):
        raise TypeError(
            "Restored certification requires its exact owning request and report."
        )
    certification_inputs.validate_source_integrity()
    if not _native_authority_equal(
        source,
        certification_inputs.source,
        limits=limits,
        source_validation=source_validation,
    ):
        raise ValueError(
            "Restored source must be scientifically identical to the retained source query."
        )
    if not _native_authority_equal(
        report.request,
        certification_inputs,
        limits=limits,
        source_validation=source_validation,
    ):
        raise ValueError(
            "Restored report request must retain the complete scientific request, not only its ID."
        )
    if report.request.request_id != certification_inputs.request_id:
        raise ValueError("Restored report must bind the retained certification request.")
    if (
        report.mesh_id,
        report.topology_id,
        report.geometry_id,
        report.schedule.schedule_id,
    ) != (
        certification_inputs.mesh_id,
        certification_inputs.topology_id,
        certification_inputs.geometry_id,
        certification_inputs.schedule.schedule_id,
    ):
        raise ValueError(
            "Restored report must bind the retained mesh, geometry and schedule."
        )
    if len(report.scoped_fidelity) != len(certification_inputs.scoped_fidelity):
        raise ValueError("Restored report lost a scoped original-source fidelity proof.")
    for scoped_certificate, (query, identifiers, tolerance) in zip(
        report.scoped_fidelity, certification_inputs.scoped_fidelity, strict=True
    ):
        if (
            scoped_certificate.source_scope_id != query.source_scope_id
            or scoped_certificate.target_facet_ids
            != tuple(np.asarray(identifiers, dtype=np.int64).tolist())
            or scoped_certificate.tolerance != tolerance
        ):
            raise ValueError(
                "Restored scoped proof binds another original stratum, trace, or tolerance."
            )
    if association is not None:
        _validate_association_binding(certification_inputs, association)


def _validate_association_binding(
    inputs: MeshCertificationInputs,
    association: GeometryAssociation,
    /,
    *,
    generation_source: NativeGenerationSource | None = None,
) -> None:
    from ..meshing.providers._native_sources import NativeLayerCoreSource

    if type(generation_source) is NativeLayerCoreSource:
        _validate_layer_core_association(inputs, association, generation_source)
        return
    source_binding = (
        (inputs.source_id, inputs.source_revision)
        if generation_source is None
        else (generation_source.source_id, generation_source.source_revision)
    )
    if (association.source_id, association.source_revision) != source_binding:
        raise ValueError("Restored association must bind the retained source revision.")
    if association.target_entity_set_id not in dict(inputs.entity_set_ids).values():
        raise ValueError(
            "Restored association must bind an actual certified mesh entity set."
        )


def _validate_mapped_layer_core_association(
    inputs: MeshCertificationInputs,
    association: GeometryAssociation,
    mapped_domain: MappedReferenceDomain,
    /,
    *,
    mesh: CellMesh | None,
) -> None:
    """Bind restored mapped tables to the actual original-root image namespace."""
    if inputs.domain is None or inputs.domain.domain_id != mapped_domain.domain_id:
        raise ValueError(
            "Mapped layer-core association must retain its original mapped domain."
        )
    if (
        association.association_kind is not GeometryAssociationKind.MAPPED_REFERENCE
        or (association.source_id, association.source_revision)
        != (mapped_domain.source_id, mapped_domain.source_revision)
        or not association.exact
        or not association.complete
    ):
        raise ValueError(
            "Mapped layer-core association must bind its complete exact source revision."
        )
    if association.target_entity_set_id not in dict(inputs.entity_set_ids).values():
        raise ValueError(
            "Mapped layer-core association must bind an actual certified entity set."
        )
    dimensions = np.asarray(association.parent_dimensions, dtype=np.int64)
    identifiers = np.asarray(association.parent_ids, dtype=np.int64)
    if np.any(dimensions < 0) or np.any(dimensions > 3):
        raise ValueError(
            "Mapped layer-core association requires explicit bounded root strata."
        )
    for degree in range(4):
        authority = np.asarray(mapped_domain.reference_mesh.entity_set(degree).entity_ids)
        if np.any(~np.isin(identifiers[dimensions == degree], authority)):
            raise ValueError(
                "Mapped layer-core association exceeds its original root namespace."
            )
    names = tuple(
        mapped_domain.image_entity_id(int(degree), int(identifier))
        for degree, identifier in zip(dimensions, identifiers, strict=True)
    )
    if association.source_entity_ids != names:
        raise ValueError(
            "Mapped layer-core association names differ from actual root image identities."
        )
    if mesh is not None:
        degree = next(
            degree
            for degree, identifier in inputs.entity_set_ids
            if identifier == association.target_entity_set_id
        )
        association.validate_target(mesh.entity_set(degree))


def _validate_layer_core_association(
    inputs: MeshCertificationInputs,
    association: GeometryAssociation,
    source: Any,
    /,
    *,
    mesh: CellMesh | None = None,
) -> None:
    """Bind each composed association to one exact authored constituent stratum."""
    if source.mapped_domain is not None:
        _validate_mapped_layer_core_association(
            inputs,
            association,
            source.mapped_domain,
            mesh=mesh,
        )
        return
    binding = (association.source_id, association.source_revision)
    roles = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.FACET,
        GeometrySourceEntityRole.REGION,
    )
    dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
    indices = np.asarray(association.source_indices, dtype=np.int64)
    if association.target_entity_set_id not in dict(inputs.entity_set_ids).values():
        raise ValueError(
            "Layer-core association must bind an actual certified entity set."
        )
    if np.any(dimensions < 0) or np.any(dimensions > 3) or np.any(indices < 0):
        raise ValueError(
            "Layer-core association requires explicit bounded source strata."
        )
    if association.source_entity_roles != tuple(
        roles[int(value)] for value in dimensions
    ):
        raise ValueError(
            "Layer-core association source roles must match their authored dimensions."
        )
    expected_names = tuple(
        f"{association.source_revision}:{roles[int(dimension)].value}:{int(index)}"
        for dimension, index in zip(dimensions, indices, strict=True)
    )
    if association.source_entity_ids != expected_names:
        raise ValueError(
            "Layer-core association entity names must bind exact source indices."
        )
    if binding == (source.source_id, source.source_revision):
        if association.target_entity_set_id != dict(inputs.entity_set_ids)[3] or np.any(
            dimensions != 3
        ):
            raise ValueError(
                "Composite layer-core source association must own material cells."
            )
        if np.any(indices >= len(source.region_ids)):
            raise ValueError(
                "Composite layer-core material association exceeds its complete namespace."
            )
    elif binding == (source.complex.complex_id, source.complex.complex_id):
        from .._meshcore import plc_source_constraints

        complex_ = source.complex
        constraints = plc_source_constraints(
            complex_.vertices,
            complex_.polygon_offsets,
            complex_.polygon_vertices,
            complex_.polygon_facets,
            complex_.facet_regions,
            segments=complex_.segments,
            work_limit=inputs.limits.maximum_work_units,
            max_scratch_bytes=inputs.limits.maximum_scratch_bytes,
        )
        counts = (
            complex_.vertices.shape[0],
            constraints.plc_edges.shape[0],
            complex_.facet_regions.shape[0],
            complex_.region_count,
        )
        if any(
            int(index) >= counts[int(dimension)]
            for dimension, index in zip(dimensions, indices, strict=True)
        ):
            raise ValueError("Core association exceeds its exact authored PLC strata.")
    elif binding == (source.layers.result_id, source.layers.result_id):
        from ..meshing._layer_core_association import _layer_source_tables
        from ..meshing._layer_core_resources import _row_entity_vertex_keys

        layer_mesh = source.layers.mesh
        tables = _layer_source_tables(
            source.layers, source.layer_regions, source.region_ids
        )
        authorities = (
            np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64),
            np.asarray(layer_mesh.entity_set(1).entity_ids, dtype=np.int64),
            tables.facet_indices,
            tables.region_indices,
        )
        if any(
            np.any(~np.isin(indices[dimensions == kind], authority))
            for kind, authority in enumerate(authorities)
        ):
            raise ValueError(
                "Layer association exceeds its exact immutable source strata."
            )
        target_dimension = next(
            dimension
            for dimension, identifier in inputs.entity_set_ids
            if identifier == association.target_entity_set_id
        )
        if target_dimension == 2:
            cap = source.layers.cap
            if cap is not None:
                cap_ids = np.concatenate(
                    [np.asarray(block.global_ids, dtype=np.int64) for block in cap.blocks]
                )
                if np.any(~np.isin(cap_ids, indices[dimensions == 2])):
                    raise ValueError(
                        "Layer association must retain every exact cap global facet identity."
                    )
        if mesh is not None:
            layer_vertex_ids = np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64)
            original_keys = (
                tuple((int(identifier),) for identifier in layer_vertex_ids)
                if target_dimension == 0
                else tuple(
                    tuple(sorted(int(layer_vertex_ids[row]) for row in vertices))
                    for vertices in _row_entity_vertex_keys(layer_mesh, target_dimension)
                )
            )
            source_keys = dict(
                zip(authorities[target_dimension].tolist(), original_keys, strict=True)
            )
            target_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
            target_keys = (
                {int(identifier): (int(identifier),) for identifier in target_vertex_ids}
                if target_dimension == 0
                else {
                    int(identifier): tuple(
                        sorted(int(target_vertex_ids[row]) for row in vertices)
                    )
                    for identifier, vertices in zip(
                        np.asarray(mesh.entity_set(target_dimension).entity_ids),
                        _row_entity_vertex_keys(mesh, target_dimension),
                        strict=True,
                    )
                }
            )
            if np.any(dimensions != target_dimension) or any(
                target_keys.get(int(target)) != source_keys[int(index)]
                for target, index in zip(
                    np.asarray(association.target_global_ids), indices, strict=True
                )
            ):
                raise ValueError(
                    "Layer association target must bind the exact immutable layer/cap vertex ancestry."
                )
    else:
        raise ValueError(
            "Layer-core association must bind an exact retained constituent source revision."
        )


def recertify_restored_meshing_source(
    inputs: MeshCertificationInputs,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    audit: CellMeshAuditReport,
    /,
    *,
    archive_limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> MeshCertificationReport:
    """Reestablish all actual native source theorems on restored mesh coordinates."""
    from ..meshing._certification import certify_meshing_acceptance

    _validate_source_value(inputs, limits=archive_limits)
    inputs.validate_source_integrity()
    fresh = certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=inputs.schedule,
        domain=inputs.domain,
        cell_regions=inputs.cell_regions,
        source=inputs.source,
        fidelity_tolerance=inputs.fidelity_tolerance,
        fidelity_sample_order=inputs.fidelity_sample_order,
        limits=inputs.limits,
        junction_vertices=inputs.junction_vertices,
        scoped_fidelity=inputs.scoped_fidelity,
    )
    if fresh.request.request_id != inputs.request_id:
        raise ValueError(
            "Restored request does not bind the actual mesh, geometry and native source."
        )
    fresh.require_passed()
    return fresh


@dataclass(frozen=True, slots=True)
class _RetainedLayerCoreValidation:
    """Content-bound proof that the live producer validated one exact lineage."""

    plan_id: str
    source_binding_id: str
    source_revision: str
    # ``None`` for a direct source; the source binding then owns its complex.
    mapped_domain_id: str | None
    specification_id: str
    options_id: str
    result_id: str
    audit_report_id: str
    certification_report_id: str
    execution_id: str
    adaptation_ids: tuple[str, ...]
    adaptation_execution_ids: tuple[str, ...]
    evidence_id: str

    @classmethod
    def from_content(
        cls, content: Mapping[str, Any], /
    ) -> "_RetainedLayerCoreValidation":
        normalized = {
            "plan_id": content["plan_id"],
            "source_binding_id": content["source_binding_id"],
            "source_revision": content["source_revision"],
            "mapped_domain_id": content["mapped_domain_id"],
            "specification_id": content["specification_id"],
            "options_id": content["options_id"],
            "result_id": content["result_id"],
            "audit_report_id": content["audit_report_id"],
            "certification_report_id": content["certification_report_id"],
            "execution_id": content["execution_id"],
            "adaptation_ids": tuple(content["adaptation_ids"]),
            "adaptation_execution_ids": tuple(content["adaptation_execution_ids"]),
        }
        if (
            any(
                type(value) is not str or not value
                for name, value in normalized.items()
                if name
                not in ("adaptation_ids", "adaptation_execution_ids", "mapped_domain_id")
            )
            or not (
                normalized["mapped_domain_id"] is None
                or (
                    type(normalized["mapped_domain_id"]) is str
                    and normalized["mapped_domain_id"]
                )
            )
            or any(
                type(value) is not str or not value
                for name in ("adaptation_ids", "adaptation_execution_ids")
                for value in normalized[name]
            )
        ):
            raise ValueError("Retained layer/core validation identities are invalid.")
        return cls(
            **normalized,
            evidence_id=canonical_fingerprint(
                {"kind": "retained-layer-core-validation", **normalized}
            ),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> "_RetainedLayerCoreValidation":
        fields_ = {
            "kind",
            "plan_id",
            "source_binding_id",
            "source_revision",
            "mapped_domain_id",
            "specification_id",
            "options_id",
            "result_id",
            "audit_report_id",
            "certification_report_id",
            "execution_id",
            "adaptation_ids",
            "adaptation_execution_ids",
            "evidence_id",
        }
        if set(record) != fields_ or record["kind"] != "retained-layer-core-validation":
            raise ArrayArchiveCorruptionError(
                "Retained layer/core validation record is invalid."
            )
        try:
            evidence = cls.from_content(record)
        except (KeyError, TypeError, ValueError) as error:
            raise ArrayArchiveCorruptionError(
                "Retained layer/core validation identities are invalid."
            ) from error
        if record["evidence_id"] != evidence.evidence_id:
            raise ArrayArchiveCorruptionError(
                "Retained layer/core validation identity changed."
            )
        return evidence

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": "retained-layer-core-validation",
            "plan_id": self.plan_id,
            "source_binding_id": self.source_binding_id,
            "source_revision": self.source_revision,
            "mapped_domain_id": self.mapped_domain_id,
            "specification_id": self.specification_id,
            "options_id": self.options_id,
            "result_id": self.result_id,
            "audit_report_id": self.audit_report_id,
            "certification_report_id": self.certification_report_id,
            "execution_id": self.execution_id,
            "adaptation_ids": list(self.adaptation_ids),
            "adaptation_execution_ids": list(self.adaptation_execution_ids),
            "evidence_id": self.evidence_id,
        }


def _layer_core_validation_content(
    records: Any,
    plan_id: str,
    /,
    *,
    live_plan: NativeMeshingPlan | None = None,
) -> dict[str, Any]:
    import json

    required = {
        "certification_inputs",
        "report",
        "associations",
        "generation_source",
        "generation_specification",
        "generation_options",
        "generation_part",
        "accepted_data",
    }
    if not isinstance(records, Mapping) or set(records) != required:
        raise ValueError(
            "Retained layer/core validation requires its exact lifecycle roles."
        )
    source = records["generation_source"]
    specification = records["generation_specification"]
    options = records["generation_options"]
    part = records["generation_part"]
    if (
        type(source) is not NativeLayerCoreSource
        or type(options) is not NativeMeshingOptions
        or type(part) is not _assembly.MeshPart
        or type(part.carrier) is not _result.CellMeshingResult
    ):
        raise TypeError(
            "Retained layer/core validation requires its source, options, and carrier."
        )
    result = part.carrier
    report = result.certification
    execution = result.execution_evidence
    if report is None or execution is None:
        raise ValueError(
            "Retained layer/core validation requires passed certification and execution."
        )
    result.audit.require_passed()
    result.audit.require_decided()
    report.require_passed()
    execution.require_valid()
    if (
        records["certification_inputs"].request_id != report.request.request_id
        or records["report"].report_id != report.report_id
        or tuple(item.association_id for item in records["associations"])
        != tuple(item.association_id for item in result.associations)
        or execution.owner_id != plan_id
        or result.compliance.specification_id != specification.specification_id
    ):
        raise ValueError(
            "Retained layer/core validation changed its source theorem or execution owner."
        )
    provenance = json.loads(result.provenance.content_json)
    if (
        not isinstance(provenance, dict)
        or provenance.get("kind") != "mapping"
        or dict(provenance.get("items", ())).get("plan") != plan_id
    ):
        raise ValueError("Retained layer/core result binds another generation plan.")
    if live_plan is not None:
        live_source = live_plan.source
        if type(live_source) is not NativeLayerCoreSource:
            raise ValueError("Retained layer/core plan does not own a layer/core source.")
        live_layer_source = cast(NativeLayerCoreSource, live_source)
        if (
            live_plan.plan_id != plan_id
            or live_layer_source.binding_id != source.binding_id
            or live_plan.specification.specification_id != specification.specification_id
            or live_plan.options.options_id != options.options_id
            or live_plan.coordinate_contract.spatial_id
            != result.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Retained layer/core plan is not the actual live generation preparation."
            )
    accepted = records["accepted_data"]
    if not isinstance(accepted, Mapping) or set(accepted) != {"adaptation_results"}:
        raise ValueError(
            "Retained layer/core validation requires one adaptation lineage."
        )
    adaptations = accepted["adaptation_results"]
    if type(adaptations) is not tuple or len(adaptations) != 2:
        raise ValueError(
            "Retained layer/core validation requires refine and inverse-coarsen results."
        )
    prior = result
    execution_ids = []
    for adaptation in adaptations:
        if (
            type(adaptation) is not _adaptation.MeshAdaptationResult
            or adaptation.status is not _adaptation.MeshAdaptationStatus.COMPLETE
            or adaptation.source.result_id != prior.result_id
        ):
            raise ValueError(
                "Retained layer/core adaptation lineage is incomplete or discontinuous."
            )
        adaptation_evidence = adaptation.evidence
        if (
            type(adaptation_evidence) is not _mixed_adaptation.MixedAdaptationEvidence
            or adaptation_evidence.execution_evidence is None
        ):
            raise ValueError(
                "Retained layer/core adaptation lacks its mixed execution evidence."
            )
        adaptation_evidence.execution_evidence.require_valid()
        if adaptation_evidence.execution_evidence.owner_id != adaptation.prepared_id:
            raise ValueError(
                "Retained layer/core adaptation execution binds another preparation."
            )
        execution_ids.append(
            canonical_fingerprint(adaptation_evidence.execution_evidence.to_record())
        )
        prior = adaptation.target
    if adaptations[1].coarsening_witnesses is None:
        raise ValueError("Retained layer/core lineage lacks inverse-coarsening evidence.")
    return {
        "plan_id": plan_id,
        "source_binding_id": source.binding_id,
        "source_revision": source.source_revision,
        # Mapped composition declares reference and mapped roots jointly; a
        # direct source has neither and is bound by its source binding alone.
        "mapped_domain_id": (
            None if source.mapped_domain is None else source.mapped_domain.domain_id
        ),
        "specification_id": specification.specification_id,
        "options_id": options.options_id,
        "result_id": result.result_id,
        "audit_report_id": result.audit.report_id,
        "certification_report_id": report.report_id,
        "execution_id": canonical_fingerprint(execution.to_record()),
        "adaptation_ids": tuple(value.result_id for value in adaptations),
        "adaptation_execution_ids": tuple(execution_ids),
    }


def _retain_layer_core_validation(
    records: Any, plan: NativeMeshingPlan, /
) -> _RetainedLayerCoreValidation:
    if type(plan) is not NativeMeshingPlan:
        raise TypeError("retained_layer_core_plan must be a NativeMeshingPlan.")
    return _RetainedLayerCoreValidation.from_content(
        _layer_core_validation_content(records, plan.plan_id, live_plan=plan)
    )


def _require_retained_layer_core_validation(
    records: Any, evidence: _RetainedLayerCoreValidation, /
) -> None:
    expected = _RetainedLayerCoreValidation.from_content(
        _layer_core_validation_content(records, evidence.plan_id)
    )
    if expected != evidence:
        raise ValueError(
            "Restored layer/core lineage differs from its retained validation evidence."
        )


@dataclass(frozen=True, slots=True)
class MeshingSourceClosureReceipt:
    """Durable publication identity of one complete native scientific closure."""

    path: Path
    content_id: str
    validation_id: str | None


_SOURCE_PREFIX = "source-record"
_SOURCE_MANIFEST_FIELDS = frozenset(
    {"kind", "recipe", "content_id", "retained_validation", "arrays"}
)


def _source_bank_name(dtype: str, index: int, /) -> str:
    return f"source-bank:{dtype}:{index:06d}"


def _source_banks(
    inventory: tuple[ModelRecipeArray, ...],
    values: Mapping[str, np.ndarray],
    limits: ArrayArchiveLimits,
    /,
) -> dict[str, np.ndarray]:
    """Pack every distinct logical array, in inventory order, into dtype banks.

    The archive member count stays independent of the number of scientific
    arrays. A bank closes only when the next array does not fit its member
    bounds, so the reader recovers each extent from bank sizes alone.
    """
    parts: dict[str, list[list[np.ndarray]]] = {}
    filled: dict[str, int] = {}
    for entry in inventory:
        dtype = np.dtype(entry.dtype)
        capacity = min(
            limits.max_array_elements,
            limits.max_axis_length,
            (limits.max_member_bytes - limits.max_npy_header_bytes) // dtype.itemsize,
        )
        flat = np.ravel(values[entry.name], order="C")
        if flat.size > capacity:
            raise ValueError(
                f"Source array {entry.path} exceeds the archive member bounds."
            )
        banks = parts.setdefault(entry.dtype, [[]])
        if filled.get(entry.dtype, 0) + flat.size > capacity:
            banks.append([])
            filled[entry.dtype] = 0
        banks[-1].append(flat)
        filled[entry.dtype] = filled.get(entry.dtype, 0) + flat.size
    return {
        _source_bank_name(dtype, index): (
            np.concatenate(bank) if bank else np.empty((0,), dtype=np.dtype(dtype))
        )
        for dtype, banks in parts.items()
        for index, bank in enumerate(banks)
    }


def _source_bank_placement(
    inventory: tuple[ModelRecipeArray, ...],
    archived: Any,
    /,
) -> tuple[dict[str, tuple[str, int]], dict[str, tuple[tuple[int, ...], str]]]:
    """Bind each logical array to one exact bank extent before any payload read."""
    if not isinstance(archived, dict):
        raise ArrayArchiveCorruptionError("Meshing source bank inventory is invalid.")
    expected: dict[str, tuple[tuple[int, ...], str]] = {}

    def bank_size(dtype: str, index: int, /) -> int:
        name = _source_bank_name(dtype, index)
        record = archived.get(name)
        shape = record.get("shape") if isinstance(record, dict) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 1
            or type(shape[0]) is not int
            or record.get("dtype") != dtype
        ):
            raise ArrayArchiveCorruptionError("Meshing source bank inventory is invalid.")
        expected[name] = ((shape[0],), dtype)
        return shape[0]

    placement: dict[str, tuple[str, int]] = {}
    cursors: dict[str, tuple[int, int]] = {}
    for entry in inventory:
        size = prod(entry.shape)
        index, used = cursors.get(entry.dtype, (0, 0))
        capacity = bank_size(entry.dtype, index)
        if used + size > capacity:
            if used != capacity:
                raise ArrayArchiveCorruptionError(
                    "Meshing source bank extents are inconsistent."
                )
            index, used = index + 1, 0
            if size > bank_size(entry.dtype, index):
                raise ArrayArchiveCorruptionError(
                    "Meshing source bank extents are inconsistent."
                )
        placement[entry.name] = (_source_bank_name(entry.dtype, index), used)
        cursors[entry.dtype] = (index, used + size)
    if any(used != bank_size(dtype, index) for dtype, (index, used) in cursors.items()):
        raise ArrayArchiveCorruptionError("Meshing source bank extents are inconsistent.")
    if set(archived) != set(expected):
        raise ArrayArchiveCorruptionError(
            "Meshing source archive contains undeclared banks."
        )
    return placement, expected


@finite_element_restoration_validation_scope()
def write_meshing_source_closure(
    path: str | PathLike[str],
    records: Any,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    retained_layer_core_plan: NativeMeshingPlan | None = None,
) -> MeshingSourceClosureReceipt:
    """Publish the canonical node-table recipe and every authored numerical field."""
    from .._model._structure import (
        model_recipe_array_inventory,
        model_recipe_array_values,
        model_recipe_wire_json,
        model_structure_recipe,
    )

    register_meshing_source_artifacts()
    retained_validation = (
        None
        if retained_layer_core_plan is None
        else _retain_layer_core_validation(records, retained_layer_core_plan)
    )
    if retained_validation is None:
        validate_meshing_source_closure(records, limits=limits)
    recipe = model_structure_recipe(records)
    values = model_recipe_array_values(
        records, recipe, prefix=_SOURCE_PREFIX, limits=limits
    )
    if any(
        isinstance(value, jax.Array) and not value.is_fully_addressable
        for value in values.values()
    ):
        raise ValueError(
            "Host source closure publication requires fully addressable JAX arrays; use distributed meshing checkpoint."
        )
    arrays = {name: np.asarray(value) for name, value in values.items()}
    content_id = canonical_fingerprint(
        {
            "recipe": recipe,
            "arrays": logical_array_value_collection_digest(arrays),
        }
    )
    inventory = model_recipe_array_inventory(recipe, prefix=_SOURCE_PREFIX, limits=limits)
    published = write_array_archive(
        path,
        manifest={
            "kind": "meshing-source-closure",
            "recipe": model_recipe_wire_json(recipe, limits=limits),
            "content_id": content_id,
            "retained_validation": (
                None if retained_validation is None else retained_validation.to_record()
            ),
        },
        arrays=_source_banks(inventory, arrays, limits),
        limits=limits,
    )
    return MeshingSourceClosureReceipt(
        published,
        content_id,
        None if retained_validation is None else retained_validation.evidence_id,
    )


def _read_meshing_source_archive(
    path: str | PathLike[str],
    /,
    *,
    expected_content_id: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    _RetainedLayerCoreValidation | None,
]:
    """Authenticate the one canonical recipe and every original numerical bank."""
    from .._model._structure import (
        model_recipe_array_inventory,
        model_recipe_from_wire_json,
    )

    register_meshing_source_artifacts()
    admitted: dict[str, Any] = {}

    def admit(manifest: Mapping[str, Any], /) -> dict[str, tuple[tuple[int, ...], str]]:
        if (
            set(manifest) != _SOURCE_MANIFEST_FIELDS
            or manifest["kind"] != "meshing-source-closure"
        ):
            raise ArrayArchiveCorruptionError(
                "Meshing source closure manifest is invalid."
            )
        if manifest["content_id"] != expected_content_id:
            raise ArrayArchiveCorruptionError(
                "Meshing source closure scientific content identity changed."
            )
        retained_record = manifest["retained_validation"]
        if retained_record is not None and not isinstance(retained_record, Mapping):
            raise ArrayArchiveCorruptionError(
                "Meshing source retained validation record is invalid."
            )
        retained_validation = (
            None
            if retained_record is None
            else _RetainedLayerCoreValidation.from_record(retained_record)
        )
        try:
            recipe = model_recipe_from_wire_json(manifest["recipe"], limits=limits)
            inventory = model_recipe_array_inventory(
                recipe, prefix=_SOURCE_PREFIX, limits=limits
            )
        except (TypeError, ValueError) as error:
            raise ArrayArchiveCorruptionError(
                "Meshing source closure recipe is invalid."
            ) from error
        placement, expected = _source_bank_placement(inventory, manifest["arrays"])
        admitted.update(
            recipe=recipe,
            inventory=inventory,
            placement=placement,
            retained_validation=retained_validation,
        )
        return expected

    _, banks = read_array_archive(path, limits=limits, admit_manifest=admit)
    recipe, placement = admitted["recipe"], admitted["placement"]
    arrays: dict[str, np.ndarray] = {}
    for entry in admitted["inventory"]:
        bank, offset = placement[entry.name]
        arrays[entry.name] = banks[bank][offset : offset + prod(entry.shape)].reshape(
            entry.shape
        )
    content_id = canonical_fingerprint(
        {
            "recipe": recipe,
            "arrays": logical_array_value_collection_digest(arrays),
        }
    )
    if content_id != expected_content_id:
        raise ArrayArchiveCorruptionError(
            "Meshing source closure scientific content identity changed."
        )
    return recipe, arrays, admitted["retained_validation"]


type _MeshingSourceArchive = tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    _RetainedLayerCoreValidation | None,
]


def _restore_meshing_source_archive(
    archive: _MeshingSourceArchive,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> Any:
    from .._model._structure import model_from_logical_array_recipe

    recipe, arrays, retained_validation = archive
    restored = model_from_logical_array_recipe(
        recipe, arrays, prefix=_SOURCE_PREFIX, limits=limits
    )
    if retained_validation is None:
        validate_meshing_source_closure(restored, limits=limits)
    else:
        _require_retained_layer_core_validation(restored, retained_validation)
    return restored


@finite_element_restoration_validation_scope()
def read_meshing_source_closure(
    path: str | PathLike[str],
    /,
    *,
    expected_content_id: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Any:
    """Restore the authenticated registered graph and renew every scientific owner."""
    archive = _read_meshing_source_archive(
        path,
        expected_content_id=expected_content_id,
        limits=limits,
    )
    return _restore_meshing_source_archive(archive, limits=limits)


@dataclass(frozen=True, slots=True)
class MeshingSourceExecutionControls:
    """Authenticated original controls only; not an accepted scientific source."""

    execution_evidence: NativeExecutionRecord
    limits: _contracts.MeshingLimits
    source_id: str
    source_revision: str
    specification_id: str
    source_binding_id: str
    content_id: str


def read_meshing_source_execution_controls(
    path: str | PathLike[str],
    /,
    *,
    expected_content_id: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    _authenticated_archive: _MeshingSourceArchive | None = None,
) -> MeshingSourceExecutionControls:
    """Decode original execution controls before any scientific constructor.

    Full manifest/type/extent/content admission is shared with the normal reader.
    Only the exact registered native receipt graph, limits and bound immutable
    identity literals are reconstructed. The source itself remains unaccepted
    until the full normal reader renews every theorem under these remaining
    original controls.
    """
    import json

    from .._model._artifacts import artifact_value
    from .._model._structure import _model_array_catalog, _restore_nodes, _table_children
    from ..meshing._scope import MeshingScope
    from ._meshing_source_families import native_generation_source_types

    recipe, arrays, _ = (
        _read_meshing_source_archive(
            path,
            expected_content_id=expected_content_id,
            limits=limits,
        )
        if _authenticated_archive is None
        else _authenticated_archive
    )
    table, catalog = _model_array_catalog(recipe, prefix=_SOURCE_PREFIX, limits=limits)
    nodes = table.nodes
    root = nodes[-1]
    if root["kind"] not in ("mapping", "frozendict", "mappingproxy"):
        raise ArrayArchiveCorruptionError(
            "Execution controls require their whole source-role namespace."
        )

    def role(name: str) -> int:
        for key, value in root["items"]:
            key_node = nodes[key]
            if key_node["kind"] == "literal" and key_node["value"] == name:
                return value
        raise ArrayArchiveCorruptionError(
            f"Source closure lacks owning execution role {name!r}."
        )

    def fields_at(index: int, expected_type: type | tuple[type, ...]) -> dict[str, int]:
        node = nodes[index]
        if node["kind"] != "dataclass" or artifact_value(node["type"]) not in (
            expected_type if isinstance(expected_type, tuple) else (expected_type,)
        ):
            raise ArrayArchiveCorruptionError(
                "Execution controls bind an incorrect registered owner type."
            )
        return node["fields"]

    specification_fields = fields_at(
        role("generation_specification"),
        (
            _contracts.CurveMeshingSpec,
            _contracts.SurfaceMeshingSpec,
            _contracts.VolumeMeshingSpec,
        ),
    )
    part_fields = fields_at(role("generation_part"), _assembly.MeshPart)
    carrier_fields = fields_at(part_fields["carrier"], _result.CellMeshingResult)
    receipt_index = carrier_fields["execution_evidence"]
    fields_at(receipt_index, NativeExecutionRecord)
    limit_index = specification_fields["limits"]
    fields_at(limit_index, _contracts.MeshingLimits)
    scope_field = (
        "boundary_scope" if "boundary_scope" in specification_fields else "scope"
    )
    scope_fields = fields_at(specification_fields[scope_field], MeshingScope)
    source_node = nodes[role("generation_source")]
    if (
        source_node["kind"] != "dataclass"
        or artifact_value(source_node["type"]) not in native_generation_source_types()
    ):
        raise ArrayArchiveCorruptionError(
            "Execution controls require their registered authored generation source."
        )
    source_fields = source_node["fields"]
    if not {"source_id", "source_revision", "binding_id"} <= source_fields.keys():
        raise ArrayArchiveCorruptionError(
            "Authored generation source lacks its exact immutable identity."
        )
    input_fields = fields_at(role("certification_inputs"), MeshCertificationInputs)
    provenance_fields = fields_at(carrier_fields["provenance"], SemanticProvenance)
    selected = {
        "source_id": source_fields["source_id"],
        "source_revision": source_fields["source_revision"],
        "source_binding_id": source_fields["binding_id"],
        "specification_id": specification_fields["specification_id"],
        "scope_source_id": scope_fields["source_id"],
        "scope_source_revision": scope_fields["source_revision"],
        "certified_source_id": input_fields["source_id"],
        "certified_source_revision": input_fields["source_revision"],
        "provenance_json": provenance_fields["content_json"],
    }
    reached: set[int] = set()
    pending = [receipt_index, limit_index, *selected.values()]
    while pending:
        index = pending.pop()
        if index in reached:
            continue
        node = nodes[index]
        if node["kind"] == "dataclass":
            if artifact_value(node["type"]) not in (
                NativeExecutionRecord,
                _contracts.MeshingLimits,
            ):
                raise ArrayArchiveCorruptionError(
                    "Execution-control decoding cannot construct scientific source owners."
                )
        elif node["kind"] not in ("literal", "nonfinite_float", "array"):
            raise ArrayArchiveCorruptionError(
                "Execution controls contain an undeclared control-node kind."
            )
        reached.add(index)
        pending.extend(_table_children(node))
    entries = {index: entry for entry, index in catalog}

    def restore_array(index: int, _node: Mapping[str, Any]) -> jax.Array:
        entry = entries[index]
        if entry.backend != "jax":
            raise ArrayArchiveCorruptionError(
                "Native execution controls require their actual JAX numerical leaves."
            )
        return jax.numpy.asarray(arrays[entry.name])

    restored = _restore_nodes(nodes, restore_array, sorted(reached))
    record = restored[receipt_index]
    original_limits = restored[limit_index]
    if (
        type(record) is not NativeExecutionRecord
        or type(original_limits) is not _contracts.MeshingLimits
    ):
        raise ArrayArchiveCorruptionError(
            "Execution controls lost their exact registered numerical owners."
        )
    record.require_valid()
    counts = {
        field.name: object.__getattribute__(original_limits, field.name)
        for field in fields(original_limits)
        if field.name != "limits_id"
    }
    validated_limits = _contracts.MeshingLimits(**counts)
    if validated_limits.limits_id != original_limits.limits_id:
        raise ArrayArchiveCorruptionError("Original execution limit identity changed.")
    identity = {name: restored[index] for name, index in selected.items()}
    if any(
        type(identity[name]) is not str or not identity[name]
        for name in (
            "source_id",
            "source_revision",
            "source_binding_id",
            "specification_id",
            "scope_source_id",
            "scope_source_revision",
            "provenance_json",
        )
    ):
        raise ArrayArchiveCorruptionError(
            "Execution control identities must be original nonempty literals."
        )
    if (
        identity["source_id"] != identity["scope_source_id"]
        or identity["source_revision"] != identity["scope_source_revision"]
        or identity["source_id"] != identity["certified_source_id"]
        or identity["source_revision"] != identity["certified_source_revision"]
    ):
        raise ArrayArchiveCorruptionError(
            "Execution controls disagree on original source/specification identity."
        )
    provenance = json.loads(identity["provenance_json"])
    if (
        not isinstance(provenance, dict)
        or provenance.get("kind") != "mapping"
        or not isinstance(provenance.get("items"), list)
    ):
        raise ArrayArchiveCorruptionError(
            "Historical execution receipt has noncanonical semantic provenance."
        )
    try:
        provenance_items = dict(provenance["items"])
    except (TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            "Historical execution receipt has malformed semantic provenance."
        ) from error
    if (
        provenance_items.get("plan") != record.owner_id
        or provenance_items.get("specification") != identity["specification_id"]
    ):
        raise ArrayArchiveCorruptionError(
            "Historical execution receipt binds another original plan/specification."
        )
    return MeshingSourceExecutionControls(
        record,
        original_limits,
        identity["source_id"],
        identity["source_revision"],
        identity["specification_id"],
        identity["source_binding_id"],
        expected_content_id,
    )


def validate_meshing_source_checkpoint_references(
    closure: Mapping[str, Any],
    references: Mapping[str, str | None],
    /,
) -> None:
    """Join exact archived owners to the declared computational checkpoint."""
    target = closure.get("collective_target")
    if target is None:
        inputs = closure["certification_inputs"]
        report = closure["report"]
        if (
            type(inputs) is not MeshCertificationInputs
            or type(report) is not MeshCertificationReport
        ):
            raise TypeError("Source closure requires exact owning inputs and report.")
        if (
            inputs.topology_id != references["topology_id"]
            or inputs.geometry_id != references["geometry_id"]
            or (
                inputs.source_revision is not None
                and inputs.source_revision != references["source_revision_id"]
            )
        ):
            raise ValueError(
                "Source closure does not bind the accepted checkpoint records."
            )
        return
    if (
        type(target) is not _result.CellMeshingResult
        or target.collective_evidence is None
    ):
        raise TypeError(
            "A collective checkpoint requires its actual accepted computational target."
        )
    evidence = target.collective_evidence
    from ..discretization._cell_geometry_validity import cell_geometry_id
    from ..meshing._initial_certification import InitialCollectiveMeshEvidence

    expected: dict[str, str | None] = {
        "topology_id": target.mesh.topology_id,
        "geometry_id": cell_geometry_id(target.geometry),
        "result_id": target.result_id,
        "coordinate_contract_id": target.coordinate_contract.spatial_id,
        "evidence_id": evidence.evidence_id,
    }
    original = _result.require_original_meshing_source(target)
    if isinstance(original, InitialCollectiveMeshEvidence):
        revision = original.compiled.domain.source_revision
    else:
        certification = original.certification
        if certification is None:
            raise ValueError(
                "A dense scientific source requires its actual original theorem."
            )
        revision = certification.request.source_revision
    if revision is not None:
        expected["source_revision_id"] = revision
    if isinstance(evidence, _result.CollectiveMeshEvidence):
        expected.update(
            {
                "source_topology_id": evidence.source.mesh.topology_id,
                "source_geometry_id": cell_geometry_id(evidence.source.geometry),
                "request_id": evidence.preparation.request.request_id,
            }
        )
        accepted = closure.get("accepted_data", {})
        results = accepted.get("adaptation_results", ())
        if results:
            result = results[0]
            expected["lineage_id"] = (
                None if result.lineage is None else result.lineage.lineage_id
            )
            expected["transfer_id"] = (
                None if result.transfer is None else result.transfer.transfer_id
            )
    elif isinstance(evidence, InitialCollectiveMeshEvidence):
        expected.update(
            {
                "source_topology_id": None,
                "source_geometry_id": None,
                "lineage_id": None,
                "transfer_id": None,
                "request_id": evidence.specification.specification_id,
            }
        )
    if any(references[name] != identity for name, identity in expected.items()):
        raise ValueError(
            "Collective source records differ from the declared accepted computational checkpoint."
        )


def meshing_source_shared_array_aliases(
    closure: Mapping[str, Any],
    /,
) -> tuple[dict[str, jax.Array], dict[int, str]]:
    """Declare shared scientific banks by exact producer object identity."""
    from ..meshing._initial_certification import InitialCollectiveMeshEvidence
    from ..meshing._publication_lowering import PublicationProjection

    arrays: dict[str, jax.Array] = {}
    identities: dict[int, str] = {}
    visited: set[int] = set()

    def bind(name: str, value: Any) -> None:
        if not isinstance(value, jax.Array) or value.is_fully_addressable:
            return
        existing = identities.get(id(value))
        if existing is not None:
            return
        previous = arrays.get(name)
        if previous is not None and previous is not value:
            raise ValueError(
                f"Scientific bank {name!r} has independently allocated buffers; "
                "the producer must share its actual prepublication receipt."
            )
        identities[id(value)] = name
        arrays[name] = value

    def publication(value: PublicationProjection, prefix: str) -> None:
        for name, array in value.source_arrays:
            bind(f"{prefix}/bank/{name}", array)
        for name, array in value.projected_arrays:
            bind(f"{prefix}/publication/{name}", array)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, (jax.Array, np.ndarray)):
            return
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, _result.CollectiveMeshEvidence):
            visit(value.source, f"{path}.source")
            prefix = f"collective/{value.evidence_id}"
            for name, array in value.logical_arrays:
                bind(f"{prefix}/bank/{name}", array)
            bind(f"{prefix}/partition_checks", value.partition_checks)
            bind(f"{prefix}/global_checks", value.global_checks)
            bind(f"{prefix}/source_exterior", value.source_exterior)
            for key, array in jax.tree_util.tree_flatten_with_path(value.initial_states)[
                0
            ]:
                bind(f"{prefix}/initial/{jax.tree_util.keystr(key)}", array)
            for key, array in jax.tree_util.tree_flatten_with_path(value.compiled_states)[
                0
            ]:
                bind(f"{prefix}/compiled/{jax.tree_util.keystr(key)}", array)
            for degree, (keys, identifiers, owners) in enumerate(
                zip(
                    value.entity_keys,
                    value.entity_ids,
                    value.entity_owners,
                    strict=True,
                )
            ):
                bind(f"{prefix}/entity/{degree}/keys", keys)
                bind(f"{prefix}/entity/{degree}/ids", identifiers)
                bind(f"{prefix}/entity/{degree}/owners", owners)
        elif isinstance(value, InitialCollectiveMeshEvidence):
            prefix = f"collective/{value.evidence_id}"
            for name, array in value.logical_arrays:
                bind(f"{prefix}/bank/{name}", array)
            bind(f"{prefix}/partition_checks", value.partition_checks)
            bind(f"{prefix}/global_checks", value.global_checks)
            for degree, (keys, identifiers, owners) in enumerate(
                zip(
                    value.entity_keys,
                    value.entity_ids,
                    value.entity_owners,
                    strict=True,
                )
            ):
                bind(f"{prefix}/entity/{degree}/keys", keys)
                bind(f"{prefix}/entity/{degree}/ids", identifiers)
                bind(f"{prefix}/entity/{degree}/owners", owners)
        elif isinstance(value, _cell_mesh.CellMeshStorage):
            prefix = f"collective/{value.evidence_id}"
            for name, array in value.logical_arrays:
                bind(f"{prefix}/bank/{name}", array)
            if value.geometry_projection is not None:
                for name, array in value.geometry_projection.projected_arrays:
                    bind(f"{prefix}/geometry_projection/{name}", array)
        elif isinstance(value, _scope.MeshScopeProjection):
            prefix = f"collective/{value.evidence_id}"
            publication(value.publication, prefix)
            bind(f"{prefix}/scope/{value.membership_name}/universe", value.global_ids)
            bind(f"{prefix}/scope/{value.membership_name}/members", value.members)
        elif isinstance(value, _organization.MeshAttribute):
            receipt = value.scope._scope_projection
            if receipt is not None:
                bind(
                    f"collective/{receipt.evidence_id}/scope/{receipt.membership_name}/values",
                    value.global_values,
                )
        if is_dataclass(value):
            for member in fields(value):
                visit(
                    object.__getattribute__(value, member.name), f"{path}.{member.name}"
                )
        elif isinstance(value, tuple):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, Mapping):
            for name, child in value.items():
                visit(child, f"{path}.{name}")

    target = closure.get("collective_target")
    if isinstance(target, _result.CellMeshingResult):
        if target.collective_evidence is None:
            raise ValueError(
                "Shared source aliases require an actual collective numerical owner."
            )
        visit(target.collective_evidence, "collective_target.collective_evidence")
        accepted = closure.get("accepted_data", {})
        for name, array in accepted.get("fields", {}).items():
            bind(
                f"collective/{target.collective_evidence.evidence_id}/field/{name}", array
            )
    visit(closure, "source_closure")
    return dict(sorted(arrays.items())), identities


@dataclass(frozen=True, slots=True, init=False)
class MeshingSourceValidation:
    """Actual common-epoch admission reused by owner-local checkpoint phases."""

    source_logical_arrays: Mapping[str, jax.Array]
    source_logical_content_digest: str
    maximum_chunk_bytes: int | None
    epochs: tuple[
        tuple[str, _result.CollectiveMeshEvidence | InitialCollectiveMeshEvidence], ...
    ]

    def __init__(
        self,
        closures: tuple[Mapping[str, Any], ...],
        source_logical_arrays: Mapping[str, jax.Array],
        source_logical_content_digest: str,
        /,
        *,
        maximum_chunk_bytes: int | None = None,
    ) -> None:
        from ..meshing._device_adaptation import validate_partitioned_mesh_evidence
        from ..meshing._initial_certification import InitialCollectiveMeshEvidence

        epochs: dict[
            str, _result.CollectiveMeshEvidence | InitialCollectiveMeshEvidence
        ] = {}
        active: set[int] = set()

        def consume(target: _result.CellMeshingResult) -> None:
            if id(target) in active:
                raise ValueError(
                    "Scientific checkpoint source epochs contain a cyclic predecessor."
                )
            active.add(id(target))
            evidence = target.collective_evidence
            if isinstance(evidence, _result.CollectiveMeshEvidence):
                consume(evidence.source)
                if evidence.evidence_id not in epochs:
                    if target.mesh.storage is None:
                        raise ValueError(
                            "Collective source admission requires its actual numerical storage."
                        )
                    validate_partitioned_mesh_evidence(target.mesh.storage, evidence)
                    epochs[evidence.evidence_id] = evidence
            elif isinstance(evidence, InitialCollectiveMeshEvidence):
                if evidence.evidence_id not in epochs:
                    evidence.require_current()
                    epochs[evidence.evidence_id] = evidence
            active.remove(id(target))

        for closure in closures:
            target = closure.get("collective_target")
            if isinstance(target, _result.CellMeshingResult):
                consume(target)
        object.__setattr__(
            self, "source_logical_arrays", MappingProxyType(dict(source_logical_arrays))
        )
        object.__setattr__(
            self, "source_logical_content_digest", source_logical_content_digest
        )
        object.__setattr__(self, "maximum_chunk_bytes", maximum_chunk_bytes)
        object.__setattr__(self, "epochs", tuple(sorted(epochs.items())))

    def require_binding(
        self,
        arrays: Mapping[str, jax.Array],
        digest: str,
        /,
    ) -> None:
        if digest != self.source_logical_content_digest or set(arrays) != set(
            self.source_logical_arrays
        ):
            raise ValueError(
                "Source admission belongs to a different complete scientific bank."
            )
        if any(
            arrays[name] is not value
            for name, value in self.source_logical_arrays.items()
        ):
            raise ValueError(
                "Source admission lost its exact immutable common-bank object references."
            )

    def _equal_local_values(
        self, first: jax.Array | np.ndarray, second: jax.Array | np.ndarray, /
    ) -> bool:
        if first.shape != second.shape or first.dtype != second.dtype:
            return False
        size = first.size
        step = (
            size
            if self.maximum_chunk_bytes is None
            else max(1, self.maximum_chunk_bytes // first.dtype.itemsize)
        )
        left, right = first.reshape(-1), second.reshape(-1)
        for start in range(0, size, max(1, step)):
            stop = min(size, start + step)
            if not np.array_equal(
                np.asarray(jax.device_get(left[start:stop])),
                np.asarray(jax.device_get(right[start:stop])),
                equal_nan=True,
            ):
                return False
        return True

    def require_epoch(
        self, evidence: _result.CollectiveMeshEvidence | InitialCollectiveMeshEvidence, /
    ) -> None:
        admitted = dict(self.epochs).get(evidence.evidence_id)
        if admitted is None or (
            admitted.mesh_id != evidence.mesh_id
            or admitted.source_evidence_id != evidence.source_evidence_id
            or admitted.coordinate_geometry_id != evidence.coordinate_geometry_id
            or admitted.global_entity_counts != evidence.global_entity_counts
        ):
            raise ValueError(
                "An owner record does not belong to the mathematically admitted common epoch."
            )
        if not self.equivalent(admitted.logical_arrays, evidence.logical_arrays):
            raise ValueError(
                "Owner numerical source banks differ from the actually admitted common epoch."
            )

    def equivalent(
        self,
        first: Any,
        second: Any,
        /,
        *,
        limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ) -> bool:
        from .._model._structure import model_recipe_array_pairs

        pairs = model_recipe_array_pairs(first, second, limits=limits)
        if pairs is None:
            return False
        admitted = {id(array) for array in self.source_logical_arrays.values()}
        for value, other in pairs:
            if isinstance(value, jax.Array) and not value.is_fully_addressable:
                if id(value) not in admitted:
                    raise ValueError(
                        "A shared source field lacks its exact common-bank object binding."
                    )
                if not isinstance(other, jax.Array):
                    return False
                first_shards = value.addressable_shards
                second_shards = other.addressable_shards
                if len(first_shards) != len(second_shards):
                    return False
                if any(
                    a.index != b.index or not self._equal_local_values(a.data, b.data)
                    for a, b in zip(first_shards, second_shards, strict=True)
                ):
                    return False
            elif not self._equal_local_values(value, other):
                return False
        return True


def validate_collective_mesh_source_epochs(
    closures: tuple[Mapping[str, Any], ...],
    /,
    *,
    source_logical_arrays: Mapping[str, jax.Array],
    source_logical_content_digest: str,
    maximum_chunk_bytes: int | None = None,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> MeshingSourceValidation:
    """Replay common source mathematics before any asymmetric owner operation."""
    return MeshingSourceValidation(
        closures,
        source_logical_arrays,
        source_logical_content_digest,
        maximum_chunk_bytes=maximum_chunk_bytes,
    )
