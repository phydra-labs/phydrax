#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Callback-free native source closures at the archive boundary."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from functools import cache
from hashlib import sha256
from inspect import signature
from math import prod
from os import getpid, PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, assert_never, Literal

import equinox as eqx
import jax
import numpy as np

import phydrax.geometry._source_boundary_records as source_boundary_records_owner
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometryAssociationSourceBank,
    GeometrySourceEntityRole,
)
from phydrax.meshing._certification import (
    certify_meshing_acceptance,
    MeshCertificationOutcome,
    MeshCertificationReport,
)
from phydrax.meshing._certification_schedule import MeshCertificationSchedule

from .._array_archive import (
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from .._bvh import BVHBuildKind, BVHBuildPolicy, PackedBVH
from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_json,
    logical_array_value_collection_digest,
)
from .._frozendict import frozendict
from .._geometry_predicates import PredicateMode
from .._identity import SemanticProvenance
from .._meshcore import (
    NativeExecutionMemoryCounters,
    NativeExecutionScope,
    NativeExecutionWorkCounters,
    plc_source_constraints,
)
from .._model._array import AbstractArrayModel
from .._model._artifacts import (
    artifact_value,
    ArtifactValueCodec,
    register_artifact_value,
    register_artifact_value_codec,
    registered_artifact_value_id,
)
from .._model._structure import (
    _with_children,
    model_array_catalog,
    model_from_logical_array_recipe,
    model_recipe_array_inventory,
    model_recipe_array_pairs,
    model_recipe_array_values,
    model_recipe_from_wire_json,
    model_recipe_wire_json,
    model_structure_recipe,
    ModelRecipeArray,
    prepare_model_recipe_catalog,
    RecipeTable,
    restore_nodes,
    table_children,
    validate_model_structure_recipe,
)
from .._physical import SpatialCoordinateContract
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._validation import canonical_identifier
from ..applications.neurofluid._model import (
    NeurofluidTransportCheckpoint,
    NeurofluidTransportParameters,
    NeurofluidTransportUnits,
)
from ..config import snapshot as configuration_snapshot
from ..discretization import (
    _adaptive_simplex,
    _axis,
    _axis_domain,
    _cell_complex,
    _cell_geometry,
    _cell_geometry_transition_types as cell_geometry_transition_owner,
    _cell_geometry_validity,
    _cell_mesh,
    _core as _discretization_core,
    _hexahedral,
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
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._chart_reconstruction import SurfaceGeometryReconstruction
from ..discretization._coordinate_budget import CoordinateEnclosurePhaseEvidence
from ..discretization._coordinate_enclosure import PreparedCoordinateLatticeExecution
from ..discretization._distributed_field import DistributedHaloPlan
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
from ..discretization._integration_domain import IntegrationDomain
from ..discretization._measure import DiscreteMeasure
from ..discretization._metric_network import MetricNetworkPlan
from ..discretization._periodic_exact import PeriodicIsometryGroup
from ..discretization._periodic_geometry import PeriodicCellGeometrySource
from ..discretization._point_cloud import (
    PointCloudPlan,
    require_point_cloud_plan_integrity,
)
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._reference_cell_types import CellBlock
from ..discretization._simplicial_locator import SimplicialLocationPolicy
from ..discretization._source_geometry_realization import SourceGeometryRealization
from ..discretization._sphere_chart_records import (
    PreparedSphereChartDeformation,
    PreparedSphereChartOccurrence,
    PreparedSphereChartPiece,
    SphereGeometryReconstruction,
)
from ..discretization._surface_chart_records import (
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
from ..discretization.fem._restoration import RestorationValidationContext
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
    _certificate as field_certificates,
    _certified_implicit,
    _compartments as _geometry_compartments,
    _contracts as geometry_contracts,
    _domain_boundary as domain_boundary_owner,
    _domain_coverage as domain_coverage_owner,
    _global_embedding as global_embedding_owner,
    _mesh_certificates as certificate,
    _meshing_domain as domain,
    _source_fidelity as source_fidelity_owner,
)
from ..geometry._compartments import CompartmentMeshingSource
from ..geometry._domain_boundary import MeshingDomainBoundarySource
from ..geometry._embedding_state import (
    MappedRestrictionRootEmbeddingEvidence,
    MeshCertificateFinding,
    MeshCertificateLimits,
    PeriodicAffineQuotientEvidence,
)
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..geometry._periodic_power_preparation import PeriodicPowerPreparation
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
    _carrier_pieces,
    _constructors,
    _intersection,
    _intersection_curve,
    _measure,
    _model,
    _patches,
    _projection_contracts,
    _query,
    _query_projection,
    _query_types,
    _root_bindings,
    _source,
)
from ..geometry.brep._placed import PlacedCurve, PlacedSurface
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
    _geometry as _multi_geometry,
    _label_extraction,
    _seeding as _multi_seeding,
    _state as _multi_state,
    _topology as _multi_topology,
)
from ..geometry.simplicial import (
    _bvh as _simplicial_bvh,
    _mesh as _simplicial_mesh,
    _regions,
    _topology as _simplicial_topology,
)
from ..geometry.simplicial._topology import SegmentTopology
from ..geometry.surface import _contracts as _surface_contracts, _model as _surface_model
from ..imaging import (
    _asset as _image_asset,
    _core as _image_core,
    _segmentation as _image_segmentation,
)
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
    _boundary_layer,
    _contracts,
    _controls,
    _coupling,
    _curving,
    _decision,
    _distribution,
    _distribution_migration,
    _implicit_volume,
    _layer_core_association,
    _level_set,
    _lineage,
    _lineage_records as lineage_records_owner,
    _migration_anchor,
    _migration_record,
    _migration_state_record,
    _mixed_adaptation,
    _organization_records as organization_records_owner,
    _overset,
    _periodic,
    _polyhedral_adaptation,
    _proposals,
    _quality,
    _restart_distribution,
    _result,
    _scope,
    _sizing,
    _surface_envelope,
    _sweep,
    _topology_edit,
    _trace,
    _volume_generation,
)
from ..meshing._adaptation import MeshAdaptationResult
from ..meshing._association import (
    AssociationPropagationPolicy,
    BRepAssociationTransfer,
    MappedReferenceAssociationTransfer,
    PlcAssociationTransfer,
    SurfaceAssociationTransfer,
)
from ..meshing._association_composition import ComposedAssociationTransfer
from ..meshing._audit import CellMeshAuditReport
from ..meshing._certification_inputs import MeshCertificationInputs
from ..meshing._compartments import revalidate_region_evidence
from ..meshing._device_adaptation import validate_partitioned_mesh_evidence
from ..meshing._device_generation import DeviceGenerationLayout
from ..meshing._domain import (
    compile_surface_domain,
    CompiledSurfaceDomain,
    SourcePatchNeighborhood,
    SourceSurfaceMetric,
    SourceSurfaceSizing,
)
from ..meshing._implicit_association_transfer import ImplicitAssociationTransfer
from ..meshing._initial_certification import InitialCollectiveMeshEvidence
from ..meshing._layer_core_association import layer_source_tables
from ..meshing._layer_core_resources import LayerCoreSourceWork, row_entity_vertex_keys
from ..meshing._lineage_records import MeshLineage
from ..meshing._measurements import (
    NativeExecutionDeviceMemoryCounters,
    NativeExecutionDeviceWorkCounters,
    NativeExecutionRecord,
)
from ..meshing._mesh_part import MeshCarrierKind, MeshPart
from ..meshing._metric import MeshMetricField
from ..meshing._plc_source import PiecewiseLinearComplex
from ..meshing._publication_lowering import PublicationProjection
from ..meshing._result import CellMeshingResult
from ..meshing._scope import MeshingScope
from ..meshing._surface_association_transfer import (
    PreparedSurfaceCurveWitness,
)
from ..meshing._surface_boundary_sources import (
    SurfaceChartBoundarySource,
    SurfaceSubdivisionBoundarySource,
)
from ..meshing._tetra_metric import (
    MetricMemoryEvidence,
    MetricOptimizerEvidence,
    MetricRemeshingCriterion,
    MetricRemeshingEvidence,
    MetricRemeshingStatus,
)
from ..meshing.providers import _native_periodic, _native_sources
from ..meshing.providers._native import NativeMeshingPlan, NativeMeshingProvider
from ..meshing.providers._native_compartment import PreparedCompartmentVolume
from ..meshing.providers._native_options import NativeMeshingOptions
from ..meshing.providers._native_route_policy import publication_policy
from ..meshing.providers._native_schedules import (
    NativeCurveSchedule,
    NativeStructuredSchedule,
    NativeSurfaceSchedule,
)
from ..meshing.providers._native_sources import (
    NativeImplicitSource,
    NativeLayerCoreSource,
    NativePlanarSource,
)
from ..meshing.providers._native_surface import closed
from ..optim._iterative import _types as _optim_types
from ..qualification._reference import ReferenceArtifactManifest
from ..solver._finite_element_schedule import FiniteElementAcceptedState
from ..sparse._linear import SparseLinearMap
from ..sparse._relation import EdgeRelation, RowRelation
from ..typing import checked, parse, validate
from ..units._dimension import DimensionSignature
from ..units._unit import UnitDefinition
from ._meshing_field_records import (
    _PreparedMeshingFieldOwner,
    meshing_field_artifact_types,
    MeshingAcceptedEpoch,
    MeshingFieldDeclaration,
    rebuild_meshing_field_record,
    validate_meshing_accepted_epochs,
    validate_meshing_field_declarations,
)
from ._meshing_source_families import (
    is_native_generation_source,
    native_family_artifact_types,
    native_generation_source_types,
    NativeGenerationSource,
)
from ._meshing_source_records import (
    content_equal,
    reconstruct_distributed_stored_fields,
    reconstruct_stored_fields,
)


# Caller-owned array-model types admitted to source closures. Each is also
# registered with the artifact registry under its explicit identity.
_ADMITTED_MODEL_TYPES: set[type[AbstractArrayModel]] = set()


type _SourceMappingKind = Literal["mapping", "frozendict", "mappingproxy"]


class MeshingSourceClosure(StrictModule, Mapping[str, Any]):
    """Retain typed source roles and dynamic numerical leaves.

    The mapping view is the interchange representation. Optional roles retain
    their presence independently of whether their stored value is ``None``.
    """

    certification_inputs: MeshCertificationInputs | None = None
    report: MeshCertificationReport | None = None
    associations: tuple[GeometryAssociation, ...] | None = None
    source: source_boundary_records_owner.SourceBoundaryQuery | None = None
    generation_source: NativeGenerationSource | None = None
    generation_sources: Mapping[str, NativeGenerationSource] | None = None
    generation_specifications: (
        Mapping[str, _contracts.SurfaceMeshingSpec | _contracts.VolumeMeshingSpec] | None
    ) = None
    registration: _overset.OversetRegistration | None = None
    accepted_data: Mapping[str, Any] | None = None
    generation_options: (
        NativeMeshingOptions | Mapping[str, NativeMeshingOptions] | None
    ) = None
    generation_part: MeshPart | None = None
    generation_specification: (
        _contracts.SurfaceMeshingSpec | _contracts.VolumeMeshingSpec | None
    ) = None
    field_declarations: (
        Mapping[str, MeshingFieldDeclaration | tuple[MeshingFieldDeclaration, ...]] | None
    ) = None
    primary_generation_part: str | None = eqx.field(static=True, default=None)
    collective_target: _result.CellMeshingResult | None = None
    accepted_target: _result.CellMeshingResult | None = None
    generation_schedule: NativeSurfaceSchedule | None = None
    generation_layout: DeviceGenerationLayout | None = None
    present_roles: frozenset[str] | None = eqx.field(static=True, default=None)
    mapping_kind: _SourceMappingKind = eqx.field(static=True, default="mapping")

    def __check_init__(self) -> None:
        parse(self.mapping_kind, _SourceMappingKind, "source_closure.mapping_kind")
        if self.present_roles is None:
            return
        if type(self.present_roles) is not frozenset:
            raise TypeError(
                f"source_closure.present_roles must be a frozenset of role names; got {type(self.present_roles).__name__}"
            )
        unknown = self.present_roles - frozenset(_CLOSURE_ROLE_NAMES)
        missing = tuple(
            name
            for name in _CLOSURE_ROLE_NAMES
            if name not in self.present_roles
            and object.__getattribute__(self, name) is not None
        )
        if unknown or missing:
            raise ValueError(
                f"source_closure.present_roles must retain every populated declared role; got unknown={sorted(unknown)}, omitted={missing}"
            )

    @classmethod
    def from_records(
        cls: type[MeshingSourceClosure], records: Mapping[str, Any], /
    ) -> MeshingSourceClosure:
        """Convert a persisted role namespace to typed source fields."""
        match records:
            case frozendict() if type(records) is frozendict:
                mapping_kind = "frozendict"
            case MappingProxyType():
                mapping_kind = "mappingproxy"
            case dict() if type(records) is dict:
                mapping_kind = "mapping"
            case _:
                raise TypeError(
                    f"source_closure.records must have a supported mapping representation; got {type(records).__name__}"
                )
        names = frozenset(_CLOSURE_ROLE_NAMES)
        for name in records:
            canonical_identifier(name, "source_closure.role")
        if set(records) - names:
            raise ValueError(
                f"source_closure.roles must name declared source roles; got {sorted(set(records) - names)}"
            )
        return cls(**records, present_roles=frozenset(records), mapping_kind=mapping_kind)

    def __iter__(self) -> Iterator[str]:
        for name in _CLOSURE_ROLE_NAMES:
            if self.present_roles is not None:
                if name in self.present_roles:
                    yield name
            elif object.__getattribute__(self, name) is not None:
                yield name

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __eq__(self, other: object, /) -> bool:
        return bool(super().__eq__(other))

    def __getitem__(self, name: str) -> Any:
        if name not in _CLOSURE_ROLE_NAMES:
            raise KeyError(name)
        value = object.__getattribute__(self, name)
        if self.present_roles is not None:
            if name not in self.present_roles:
                raise KeyError(name)
        elif value is None:
            raise KeyError(name)
        return value

    def to_records(self) -> Mapping[str, Any]:
        values = {name: object.__getattribute__(self, name) for name in self}
        match self.mapping_kind:
            case "mapping":
                return values
            case "frozendict":
                return frozendict(values)
            case "mappingproxy":
                return MappingProxyType(values)
            case _ as invalid:
                assert_never(invalid)


_CLOSURE_ROLE_NAMES = tuple(
    member.name
    for member in fields(MeshingSourceClosure)
    if member.name not in ("present_roles", "mapping_kind")
)


def admit_meshing_source_model(identity: str, model_type: type, /) -> None:
    """Admit a caller-owned array-model type to meshing-source closures.

    ``identity`` names the type in archives. A restored instance is rebuilt by
    calling its constructor with its retained fields, so every constructor
    input must be a retained field of the same name.
    """
    if not isinstance(model_type, type) or not issubclass(model_type, AbstractArrayModel):
        raise TypeError(
            f"source_closure.model_type must be an AbstractArrayModel subclass; got {model_type!r}"
        )
    register_artifact_value(canonical_identifier(identity, "identity"), model_type)
    _ADMITTED_MODEL_TYPES.add(model_type)
    register_meshing_source_artifacts()


_MIGRATION_ARTIFACT_TYPES = (
    _migration_anchor.MIGRATION_ANCHOR_ARTIFACT_TYPES
    + _migration_state_record.MIGRATION_STATE_ARTIFACT_TYPES
    + _migration_record.MIGRATION_RECORD_ARTIFACT_TYPES
)


_SOURCE_ARTIFACT_GROUPS = (
    (
        (
            "phydrax._meshcore",
            (NativeExecutionWorkCounters, NativeExecutionMemoryCounters),
        ),
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
                geometry_contracts.CompiledGeometry,
                geometry_contracts.GeometryTolerance,
                PlanarEmbedding,
                field_certificates.FieldCertificate,
                field_certificates.ExactSDFEnclosureCertificate,
                field_certificates.ZeroSetAccuracy,
                field_certificates.SignReliability,
                field_certificates.DistanceSemantics,
                field_certificates.FieldRegularity,
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
                _enclosure.IntervalFieldBounds,
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
                MeshingDomainBoundarySource,
                domain.DomainCurveMap,
                domain.PhysicalCurveMap,
                MeshCertificateLimits,
                MeshCertificateFinding,
                certificate.MeshCertificateBinding,
                PeriodicAffineQuotientEvidence,
                MappedRestrictionRootEmbeddingEvidence,
                global_embedding_owner.MappedBoundaryDegreeEvidence,
                global_embedding_owner.GlobalEmbeddingCertificate,
                domain_coverage_owner.PiecewiseLinearDomain,
                domain_coverage_owner.DomainCoverageCertificate,
                source_boundary_records_owner.SourceBoundaryDistance,
                source_boundary_records_owner.SourceBoundarySamples,
                source_boundary_records_owner.SourceBoundaryChartCover,
                source_fidelity_owner.ParametricCurveBoundarySource,
                source_fidelity_owner.ImplicitBoundarySource,
                source_fidelity_owner.ImplicitProjectionBoundarySource,
                source_fidelity_owner.ImplicitProjectionCoverageEvidence,
                source_fidelity_owner.SourceFidelityCertificate,
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
                _simplicial_bvh.SegmentBVH,
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
                _constructors.NormalizedTrimCurve,
                PlacedSurface,
                PlacedCurve,
                _intersection_curve.SurfaceRegion,
                _intersection_curve.CurveRange,
                _patches.BernsteinSurfacePiece,
                _patches.BernsteinCurvePiece,
                _carrier_pieces.SurfacePiece,
                _carrier_pieces.CurvePiece,
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
                _query_types.Tolerances,
                _query_types.EdgeData,
                _query_types.FaceData,
                _measure.BRepQuadrature,
                _measure.BRepMeasureResult,
                _query.PreparedBRepQuery,
                _query_projection.NativeBRepProjection,
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
                _primitives.BallKernel,
                _primitives.OrthotopeKernel,
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
                _extended.TorusKernel,
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
                GeometryAssociationSourceBank,
                GeometrySourceEntityRole,
                _implicit_volume.PreparedAdaptiveImplicitVolume,
                PreparedCompartmentVolume,
                _level_set.LevelSetMeshAdaptation,
                _level_set.LevelSetEvidence,
                _assembly.MeshAssembly,
                MeshPart,
                MeshCarrierKind,
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
                organization_records_owner.MeshPatch,
                organization_records_owner.MeshZone,
                organization_records_owner.MeshLabel,
                organization_records_owner.MeshAttribute,
                organization_records_owner.MeshZoneRole,
                organization_records_owner.RegionRole,
                organization_records_owner.MeshAttributeRole,
                organization_records_owner.RegionBoundaryEvidence,
                organization_records_owner.RegionMeshingEvidence,
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
                MetricOptimizerEvidence,
                MetricMemoryEvidence,
                MetricRemeshingEvidence,
                MetricOptimizerEvidence,
                MetricMemoryEvidence,
                MetricRemeshingStatus,
                NativeExecutionRecord,
                NativeExecutionDeviceWorkCounters,
                NativeExecutionDeviceMemoryCounters,
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
                _native_periodic._PeriodicSourceAssociationPreparation,
                _native_periodic._PeriodicSourceAssociationReceipt,
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
                _mixed_adaptation.Sibling,
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
                _restart_distribution.SimplexRestartPlacement,
                _restart_distribution.SimplexRestartRepack,
                _distribution_migration.SimplexGraphRestartProposal,
                _distribution.MeshPartitionEvidence,
                _distribution.MeshPartitionKind,
                _distribution.MeshPartitionPolicy,
                _lineage.CellMeshTransition,
                lineage_records_owner.EntityLineage,
                lineage_records_owner.MeshLineage,
                lineage_records_owner.MeshTransitionKind,
                lineage_records_owner.VertexInterpolationStencil,
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
                organization_records_owner.MeshAttributeProjection,
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
                CellBlock,
                _cell_geometry.CellGeometrySpec,
                _cell_geometry.CoordinateTabulator,
                PreparedCoordinateLatticeExecution,
                CoordinateEnclosurePhaseEvidence,
                _cell_geometry_validity.CellValidityCertificate,
                _cell_geometry.SweptCoordinateTabulator,
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
                cell_geometry_transition_owner.CellGeometryTransitionPolicy,
                cell_geometry_transition_owner.CellGeometryTransition,
                cell_geometry_transition_owner.CellGeometryTransitionEvidence,
                SourceGeometryRealization,
                SurfaceChartWitness,
                PreparedSurfaceChartDeformation,
                PreparedSurfaceChartOccurrence,
                PreparedSurfaceChartPiece,
                SurfaceGeometryReconstruction,
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
                PeriodicIsometryGroup,
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
        (
            "phydrax.meshing",
            _MIGRATION_ARTIFACT_TYPES
            + (
                _distribution.SimplexNeighborhoodWorkset,
                _distribution.PartitionedRebalanceProposal,
            ),
        ),
    )
)

_FIELD_ARTIFACT_TYPES = frozenset(
    owner for _, owners in meshing_field_artifact_types() for owner in owners
)


@dataclass(frozen=True, slots=True)
class _SourceAdmission:
    namespace: str
    owner: type
    reconstruct: Callable[[Any], Any] | None
    validate_restored: Callable[[Any], None] | None


@cache
def _source_admissions() -> dict[type, _SourceAdmission]:
    # The admission table also owns each type's stored-state reconstruction.
    # Context-bound owners are rebound by the explicit source dispatch below.
    constructors: dict[type, Callable[[Any], Any] | None] = {
        _certified_implicit.CertifiedImplicitCover: _certified_implicit.CertifiedImplicitCover.from_stored_fields,
        _enclosure.IntervalFieldBounds: _enclosure.IntervalFieldBounds.from_stored_fields,
        _model.BRepTopology: _model.BRepTopology.from_stored_fields,
        _model.BRepGeometry: _model.BRepGeometry.from_stored_fields,
        _model.BRepModel: _model.BRepModel.from_stored_fields,
        _query.PreparedBRepQuery: _query.PreparedBRepQuery.from_stored_fields,
        _query_projection.NativeBRepProjection: _query_projection.NativeBRepProjection.from_stored_fields,
        _source.BRepSource: _source.BRepSource.from_stored_fields,
        _source._NativeBRepKernel: _source._NativeBRepKernel.from_stored_fields,
        _intersection_curve.SurfaceRegion: _intersection_curve.SurfaceRegion.from_stored_fields,
        _intersection_curve.CurveRange: _intersection_curve.CurveRange.from_stored_fields,
        _intersection_curve.PeriodicPCurve: _intersection_curve.PeriodicPCurve.from_stored_fields,
        _intersection_curve.IntersectionCurve: _intersection_curve.IntersectionCurve.from_stored_fields,
        domain.MeshingDomain: domain.MeshingDomain.from_stored_fields,
        domain.PhysicalCurveMap: domain.PhysicalCurveMap.from_stored_fields,
        CellBlock: CellBlock.from_stored_fields,
        CellMesh: CellMesh.from_stored_fields,
        _cell_mesh.CellMeshStorage: _cell_mesh.CellMeshStorage.from_stored_fields,
        CellGeometrySpec: CellGeometrySpec.from_stored_fields,
        _cell_geometry.CoordinateTabulator: _cell_geometry.CoordinateTabulator.from_stored_fields,
        _cell_geometry.SweptCoordinateTabulator: _cell_geometry.SweptCoordinateTabulator.from_stored_fields,
        _cell_geometry.LayerColumnCellGeometryElement: _cell_geometry.LayerColumnCellGeometryElement.from_stored_fields,
        _cell_geometry.PolynomialComposedCellGeometryElement: _cell_geometry.PolynomialComposedCellGeometryElement.from_stored_fields,
        _cell_geometry.RationalComposedCellGeometryElement: _cell_geometry.RationalComposedCellGeometryElement.from_stored_fields,
        FiniteElementSpec: FiniteElementSpec.from_stored_fields,
        FiniteElementPlan: FiniteElementPlan.from_stored_fields,
        DistributedHaloPlan: DistributedHaloPlan.from_stored_fields,
        _tensor_support.PreparedTensorGrid: _tensor_support.PreparedTensorGrid.from_stored_fields,
        _bisection.BisectionUniformRefinement: _bisection.BisectionUniformRefinement.from_stored_fields,
        _bisection.BisectionHierarchy: _bisection.BisectionHierarchy.from_stored_fields,
        _periodic.PeriodicRefinement: _reconstruct_periodic_refinement,
        _mixed_adaptation.MixedLayerColumns: _mixed_adaptation.MixedLayerColumns.from_stored_fields,
        _mixed_adaptation.MixedAdaptationHierarchy: _mixed_adaptation.MixedAdaptationHierarchy.from_stored_fields,
        _adaptation.PreparedMeshAdaptation: _adaptation.PreparedMeshAdaptation.from_stored_fields,
        _native_periodic._PeriodicSourceAssociationPreparation: _native_periodic._PeriodicSourceAssociationPreparation.from_stored_fields,
        _native_periodic._PeriodicSourceAssociationReceipt: _native_periodic._PeriodicSourceAssociationReceipt.from_stored_fields,
        _adaptation.MeshAdaptationResult: _adaptation.MeshAdaptationResult.from_stored_fields,
        _result.CollectiveMeshEvidence: _result.CollectiveMeshEvidence.from_stored_fields,
        _result.CellMeshingResult: _result.CellMeshingResult.from_stored_fields,
        _simplicial_bvh.SegmentBVH: _simplicial_bvh.SegmentBVH.from_stored_fields,
        _decision.MeasuredAdaptationCost: _decision.MeasuredAdaptationCost.from_stored_fields,
        _decision.FixedEpochDerivativeEvidence: _decision.FixedEpochDerivativeEvidence.from_stored_fields,
        _boundary_layer.BoundaryLayerMesh: _boundary_layer.BoundaryLayerMesh.from_stored_fields,
        _boundary_layer.BoundaryLayerEvidence: _boundary_layer.BoundaryLayerEvidence.from_stored_fields,
        PiecewiseLinearComplex: PiecewiseLinearComplex.from_stored_fields,
        _sweep.SweepControl: _sweep.SweepControl.from_stored_fields,
        _surface_model.SurfaceRealization: _surface_model.SurfaceRealization.from_stored_fields,
        _surface_envelope.SurfaceEnvelope: _surface_envelope.SurfaceEnvelope.from_stored_fields,
        _native_sources.NativePlanarSource: _native_sources.NativePlanarSource.from_stored_fields,
        _native_sources.NativePlcSource: _native_sources.NativePlcSource.from_stored_fields,
        _native_sources.NativePolyhedralSource: _native_sources.NativePolyhedralSource.from_stored_fields,
        _native_sources.NativeLayerCoreSource: _native_sources.NativeLayerCoreSource.from_stored_fields,
        _native_sources.NativeSurfaceEnvelopeSource: _native_sources.NativeSurfaceEnvelopeSource.from_stored_fields,
        _native_periodic.NativePeriodicSource: _native_periodic.NativePeriodicSource.from_stored_fields,
        PeriodicPowerPreparation: PeriodicPowerPreparation.from_stored_fields,
        ExactPlcCellGeometrySource: ExactPlcCellGeometrySource.from_stored_fields,
        ExactPlcCellGeometryConvexSource: ExactPlcCellGeometryConvexSource.from_stored_fields,
        ExactPowerCellGeometrySource: ExactPowerCellGeometrySource.from_stored_fields,
        ExactPowerCellGeometryLinearActionSource: ExactPowerCellGeometryLinearActionSource.from_stored_fields,
        ExactPowerCellGeometryRestrictionSource: ExactPowerCellGeometryRestrictionSource.from_stored_fields,
        ExactRationalCellGeometrySource: ExactRationalCellGeometrySource.from_stored_fields,
        RadicalElement: RadicalElement.from_stored_fields,
        ExactAlgebraicCellGeometrySource: ExactAlgebraicCellGeometrySource.from_stored_fields,
        PeriodicCellGeometrySource: PeriodicCellGeometrySource.from_stored_fields,
        PreparedSurfaceChartDeformation: PreparedSurfaceChartDeformation.from_stored_fields,
        PreparedSphereChartDeformation: PreparedSphereChartDeformation.from_stored_fields,
        SphereGeometryReconstruction: SphereGeometryReconstruction.from_stored_fields,
        SphereMaterialCellAtlas: SphereMaterialCellAtlas.from_stored_fields,
        source_fidelity_owner.ImplicitBoundarySource: source_fidelity_owner.ImplicitBoundarySource.from_stored_fields,
        source_fidelity_owner.ParametricCurveBoundarySource: source_fidelity_owner.ParametricCurveBoundarySource.from_stored_fields,
        _adaptive_discovery.AdaptiveImplicitSurface: _adaptive_discovery.AdaptiveImplicitSurface.from_stored_fields,
        _adaptive_discovery.AdaptiveImplicitSurfaceEvidence: _adaptive_discovery.AdaptiveImplicitSurfaceEvidence.from_stored_fields,
        _adaptive_discovery.AdaptiveImplicitBoundarySource: _adaptive_discovery.AdaptiveImplicitBoundarySource.from_stored_fields,
        _adaptive_discovery.ImplicitVolumeQuery: _adaptive_discovery.ImplicitVolumeQuery.from_stored_fields,
        FiniteElementTransferEvidence: FiniteElementTransferEvidence.from_stored_fields,
        PreparedUnstructuredConservativeRemap: PreparedUnstructuredConservativeRemap.from_stored_fields,
        MeshCertificationInputs: None,
        MeshCertificationReport: None,
        certificate.MeshCertificateBinding: None,
        domain_coverage_owner.DomainCoverageCertificate: None,
        global_embedding_owner.GlobalEmbeddingCertificate: global_embedding_owner.GlobalEmbeddingCertificate.from_stored_fields,
        MappedRestrictionRootEmbeddingEvidence: MappedRestrictionRootEmbeddingEvidence.from_stored_fields,
        source_fidelity_owner.SourceFidelityCertificate: source_fidelity_owner.SourceFidelityCertificate.from_stored_fields,
        InitialCollectiveMeshEvidence: None,
        PublicationProjection: None,
        organization_records_owner.MeshAttributeProjection: None,
        SourceGeometryRealization: None,
        # Tensor-grid children have no retained axis constructor binding.
        # The enclosing grid rebuilds and compares their full stored fields.
        _tensor_entities.StructuredAxis: None,
        _tensor_entities.TensorEntityLayout: None,
        NativeExecutionRecord: None,
        NativeExecutionDeviceWorkCounters: None,
        NativeExecutionDeviceMemoryCounters: None,
        NativeExecutionWorkCounters: None,
        NativeExecutionMemoryCounters: None,
        PointCloudPlan: None,
        PreparedCoordinateLatticeExecution: None,
        _level_set.LevelSetEvidence: None,
        PreparedCompartmentVolume: None,
        _geometry_compartments.CompartmentMeshingSource: None,
        _label_extraction.LabelFieldVolumeBinding: None,
        _label_extraction.LabelFieldSurfaceExtractionResult: None,
        FiniteElementDofMap: None,
        MetricRemeshingEvidence: None,
        PreparedSurfaceCurveWitness: None,
        _periodic_cell.PeriodicCell: None,
        FiniteElementDiscretization: None,
        # The enclosing discretization replays and compares both geometry caches.
        FiniteElementRuntimeData: None,
        FiniteElementBlockGeometry: None,
        _fem_distributed.FiniteElementExecutionLimits: None,
        _fem_distributed.FiniteElementGlobalDofOwnership: None,
        _fem_distributed.FiniteElementDofOwnershipPlan: None,
        _fem_distributed.OwnerLocalFiniteElementDiscretization: None,
        _fem_distributed.OwnerLocalFiniteElementTransfer: None,
        _fem_distributed.FiniteElementClosurePreparation: None,
        UnitDefinition: UnitDefinition.from_stored_fields,
        DimensionSignature: DimensionSignature.from_stored_fields,
    }
    for owner in _MIGRATION_ARTIFACT_TYPES:
        constructor = getattr(owner, "from_stored_fields")
        if not callable(constructor):
            raise TypeError(
                f"source_artifact.{owner.__name__}.constructor must provide a stored-record validator; got missing callable"
            )
        constructors[owner] = constructor
    validators: dict[type, Callable[[Any], None]] = {
        NativeExecutionWorkCounters: NativeExecutionWorkCounters.validate_restored,
        NativeExecutionMemoryCounters: NativeExecutionMemoryCounters.validate_restored,
        NativeExecutionDeviceWorkCounters: NativeExecutionDeviceWorkCounters.__post_init__,
        NativeExecutionDeviceMemoryCounters: NativeExecutionDeviceMemoryCounters.__post_init__,
        NativeExecutionRecord: NativeExecutionRecord.require_valid,
        _boundary_layer.BoundaryLayerEvidence: _boundary_layer.BoundaryLayerEvidence.validate_restored,
        _boundary_layer.BoundaryLayerMesh: _boundary_layer.BoundaryLayerMesh.validate_restored,
        PreparedCoordinateLatticeExecution: PreparedCoordinateLatticeExecution.validate_restored,
        _enclosure.IntervalFieldBounds: _enclosure.IntervalFieldBounds.validate_source_integrity,
        PackedBVH: PackedBVH.validate_restored,
        _simplicial_bvh.SegmentBVH: _simplicial_bvh.SegmentBVH.validate_restored,
    }
    for owner in _MIGRATION_ARTIFACT_TYPES:
        validator = getattr(owner, "validate_restored")
        if not callable(validator):
            raise TypeError(
                f"source_artifact.{owner.__name__}.validator must validate restored fields; got missing callable"
            )
        validators[owner] = validator
    return {
        owner: _SourceAdmission(
            namespace,
            owner,
            constructors.get(
                owner,
                rebuild_meshing_field_record
                if owner in _FIELD_ARTIFACT_TYPES
                else reconstruct_stored_fields,
            ),
            validators.get(owner),
        )
        for namespace, owners in _SOURCE_ARTIFACT_GROUPS
        for owner in owners
    }


def meshing_source_artifact_types() -> tuple[tuple[str, tuple[type, ...]], ...]:
    """Return the source types admitted to the artifact registry."""
    grouped: dict[str, list[type]] = {}
    for admission in _source_admissions().values():
        grouped.setdefault(admission.namespace, []).append(admission.owner)
    return tuple((namespace, tuple(owners)) for namespace, owners in grouped.items())


def _reconstruct_scope(
    node: MeshingScope,
    meshes: tuple[CellMesh, ...],
    results: tuple[_result.CellMeshingResult, ...],
    evidence: Mapping[tuple[str, str], _result.AbstractCollectiveMeshTheorem] | None,
    /,
) -> MeshingScope:
    if node.local_entity_universe is None:
        return MeshingScope.from_stored_fields(node)
    projection = node.projection
    if projection is not None:
        projection = _reconstruct_scope_projection(projection, results, evidence)
    qualified = tuple(
        (
            mesh,
            MeshingScope.from_stored_fields(node, local_mesh=mesh, projection=projection),
        )
        for mesh in meshes
        if node.matches_local_mesh(mesh, projection=projection)
    )
    if not qualified:
        raise ValueError(
            "MeshingScope.local_mesh must retain an authenticated constructor context; got no matching mesh"
        )
    if any(
        not content_equal(qualified[0][0], mesh, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
        for mesh, _ in qualified[1:]
    ):
        storage_ids = tuple(
            None if mesh.storage is None else mesh.storage.storage_id
            for mesh, _ in qualified
        )
        raise ValueError(
            f"MeshingScope.local_mesh must have one content binding; got conflicting storage IDs {storage_ids}"
        )
    return qualified[0][1]


def _reconstruct_scope_projection(
    node: _scope.MeshScopeProjection,
    results: tuple[_result.CellMeshingResult, ...],
    evidence: Mapping[tuple[str, str], _result.AbstractCollectiveMeshTheorem] | None,
    /,
) -> _scope.MeshScopeProjection:
    indexed = {} if evidence is None else dict(evidence)
    for result in results:
        retained = result.collective_evidence
        if retained is not None:
            indexed.setdefault((retained.evidence_id, retained.mesh_id), retained)
    retained = indexed.get((node.evidence_id, node.source_id))
    match retained:
        case _result.CollectiveMeshEvidence() | InitialCollectiveMeshEvidence():
            pass
        case None:
            raise ValueError(
                "MeshScopeProjection.evidence must retain accepted source evidence; got missing evidence"
            )
        case _:
            raise TypeError(
                f"MeshScopeProjection.evidence must have an admitted evidence owner; got {type(retained).__name__}"
            )
    publication = node.publication

    def rebind(projection: _scope.MeshScopeProjection) -> _scope.MeshScopeProjection:
        if projection.publication is not publication and not content_equal(
            projection.publication, publication, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
        ):
            raise ValueError(
                "MeshScopeProjection.publication must match its exclusive scope inventory; got different content"
            )
        return _scope.MeshScopeProjection(
            retained,
            publication,
            projection.source_revision,
            projection.dimension,
            projection.membership_name,
            exclusive_scopes=tuple(
                rebind(other) for other in projection.exclusive_scopes
            ),
        )

    return rebind(node)


def _reconstruct_periodic_refinement(
    node: _periodic.PeriodicRefinement, /
) -> _periodic.PeriodicRefinement:
    """Revalidate retained periodic ancestry with one bounded replay ledger."""
    work, storage, queries = node.reconstruction_preparation_bounds()
    with _cell_geometry.owned_coordinate_preparation(
        work, storage, maximum_geometry_queries=queries
    ) as budget:
        return reconstruct_stored_fields(node, bindings={"budget": budget})


def _reconstruct_periodic_evidence(
    node: _periodic.PeriodicQuotientEvidence,
    meshes: tuple[CellMesh, ...],
    results: tuple[_result.CellMeshingResult, ...],
    /,
    *,
    limits: ArrayArchiveLimits,
) -> _periodic.PeriodicQuotientEvidence:
    originals = tuple(
        mesh
        for mesh in meshes
        if mesh.periodic_topology is not None
        and mesh.periodic_topology.periodic_topology_id == node.periodic_topology_id
    )
    if not originals:
        raise ValueError(
            "PeriodicQuotientEvidence.mesh must retain its periodic mesh source; got no matching topology"
        )
    rebuilt = []
    for mesh in originals:
        geometries = tuple(
            result.geometry
            for result in results
            if result.mesh is mesh or content_equal(result.mesh, mesh, limits=limits)
        )
        if geometries and any(
            cell_geometry_id(geometry) != cell_geometry_id(geometries[0])
            for geometry in geometries[1:]
        ):
            raise ValueError(
                "PeriodicQuotientEvidence.geometry must have one retained declared source binding; got ambiguous geometry"
            )
        rebuilt.append(
            _periodic.PeriodicQuotientEvidence.from_stored_fields(
                node, mesh=mesh, geometry=geometries[0] if geometries else None
            )
        )
    if any(
        jax.tree_util.tree_structure(value) != jax.tree_util.tree_structure(rebuilt[0])
        for value in rebuilt[1:]
    ):
        raise ValueError(
            "PeriodicQuotientEvidence.mesh must have one numerical source binding; got ambiguous topology"
        )
    return rebuilt[0]


def _reconstruct_proposal(
    node: Any, results: tuple[_result.CellMeshingResult, ...], /
) -> Any:
    source_id = (
        node.policy.source_result_id
        if type(node) is _proposals.MeshProposalProjection
        else node.source_result_id
    )
    sources = tuple(result for result in results if result.result_id == source_id)
    if len(sources) != 1:
        raise ValueError(
            f"{type(node).__name__}.source must retain one result in its source closure; got {len(sources)} matching results"
        )
    source = sources[0]
    match node:
        case _proposals.MeshProposalProjection():
            return _proposals.project_mesh_proposal(source, node.proposal, node.policy)
        case (
            _proposals.MeshProposalFeatures()
            | _proposals.MeshMarkingProposal()
            | _proposals.MeshProposalSafetyPolicy()
        ):
            return reconstruct_stored_fields(node, bindings={"source": source})
        case _:
            raise TypeError(
                f"source_closure.proposal must have an admitted proposal type; got {type(node).__name__}"
            )


def _reconstruct_surface_support(
    node: PreparedSurfaceSourceSupport, results: tuple[_result.CellMeshingResult, ...], /
) -> PreparedSurfaceSourceSupport:
    if not isinstance(node.original, InitialCollectiveMeshEvidence):
        return PreparedSurfaceSourceSupport.from_stored_fields(node)
    matches: list[_result.CellMeshingResult] = []
    for result in results:
        original = _result.require_original_meshing_source(result)
        if (
            isinstance(original, InitialCollectiveMeshEvidence)
            and original.evidence_id == node.original.evidence_id
            and original.mesh_id == node.original.mesh_id
        ):
            matches.append(result)
    if len(matches) != 1:
        raise ValueError(
            f"PreparedSurfaceSourceSupport.root_result must retain one accepted source carrier; got {len(matches)} matching carriers"
        )
    original = _result.require_original_meshing_source(matches[0])
    if not content_equal(original, node.original, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS):
        raise ValueError(
            "PreparedSurfaceSourceSupport.original must match the supplied carrier source; got changed source content"
        )
    return PreparedSurfaceSourceSupport.from_stored_fields(node, root_result=matches[0])


def _reconstruct_region_evidence(
    node: organization_records_owner.RegionMeshingEvidence
    | organization_records_owner.RegionBoundaryEvidence,
    results: tuple[_result.CellMeshingResult, ...],
    limits: ArrayArchiveLimits,
    /,
) -> (
    organization_records_owner.RegionMeshingEvidence
    | organization_records_owner.RegionBoundaryEvidence
):
    """Reconstruct region evidence against retained result constructor inputs."""
    owners: list[_result.CellMeshingResult] = []
    for result in results:
        match node:
            case organization_records_owner.RegionMeshingEvidence():
                retained = result.region_evidence
                inventory_matches = (
                    tuple(
                        np.asarray(
                            result.mesh.entity_set(3).entity_ids, dtype=np.int64
                        ).tolist()
                    )
                    == node.cell_global_ids
                    if result.mesh.topological_dimension == 3
                    else False
                )
            case organization_records_owner.RegionBoundaryEvidence():
                retained = next(
                    (
                        value
                        for value in result.region_boundary_evidence
                        if value.evidence_id == node.evidence_id
                    ),
                    None,
                )
                inventory_matches = retained is not None and content_equal(
                    retained.boundary_label.scope,
                    node.boundary_label.scope,
                    limits=limits,
                )
            case _ as invalid:
                assert_never(invalid)
        if (
            retained is not None
            and retained.evidence_id == node.evidence_id
            and result.mesh.mesh_id == node.target_mesh_id
            and result.mesh.topology_id == node.target_topology_id
            and inventory_matches
        ):
            if not content_equal(retained, node, limits=limits):
                raise ValueError(
                    f"{type(node).__name__}.evidence_id must bind retained result evidence; got different evidence content"
                )
            owners.append(result)
    if not owners:
        raise ValueError(
            f"{type(node).__name__}.owner must retain matching evidence, mesh, topology, and local inventory; got no matching CellMeshingResult"
        )
    owner = owners[0]
    if any(
        not content_equal(
            (owner.mesh, owner.geometry),
            (other.mesh, other.geometry),
            limits=limits,
        )
        for other in owners[1:]
    ):
        raise ValueError(
            f"{type(node).__name__}.owner must have one mesh and geometry binding; got conflicting CellMeshingResult owners"
        )
    match node:
        case organization_records_owner.RegionMeshingEvidence():
            return reconstruct_stored_fields(
                node, bindings={"mesh": owner.mesh, "geometry": owner.geometry}
            )
        case organization_records_owner.RegionBoundaryEvidence():
            return reconstruct_stored_fields(node, bindings={"mesh": owner.mesh})
        case _ as invalid:
            assert_never(invalid)


def _reconstruct_oriented_incidence(
    node: _topology.OrientedIncidence,
    entity_sets: tuple[_topology.EntitySet, ...],
    /,
) -> _topology.OrientedIncidence:
    lowers = tuple(
        entities
        for entities in entity_sets
        if entities.entity_set_id == node.lower_entity_set_id
        and entities.intrinsic_dimension == node.degree - 1
    )
    uppers = tuple(
        entities
        for entities in entity_sets
        if entities.entity_set_id == node.upper_entity_set_id
        and entities.intrinsic_dimension == node.degree
    )
    if not lowers or not uppers:
        raise ValueError(
            "OrientedIncidence.entity_sets must retain its lower and upper bindings; got missing entity sets"
        )
    for candidates in (lowers, uppers):
        if any(
            not content_equal(candidates[0], other, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
            for other in candidates[1:]
        ):
            raise ValueError(
                "OrientedIncidence.entity_sets must have one content binding per stored ID; got conflicting entity sets"
            )
    return reconstruct_stored_fields(
        node, bindings={"lower": lowers[0], "upper": uppers[0]}
    )


def _reconstruct_periodic_lift(mesh: CellMesh, limits: ArrayArchiveLimits, /) -> CellMesh:
    """Admit local source arrays before rebuilding a storage-free lifted carrier."""
    if mesh.storage is None:
        return CellMesh.lifted_from_stored_fields(mesh)
    entity_ids = {
        entities.intrinsic_dimension: entities.entity_ids
        for entities in mesh.topology.entity_sets
    }
    match mesh.connectivity:
        case _cell_complex.PolyhedralConnectivity() as connectivity:
            polyhedral = connectivity
        case (
            _cell_complex.IntervalConnectivity()
            | _cell_complex.PolygonalConnectivity()
            | _cell_complex.TetrahedralConnectivity()
            | _hexahedral.HexahedralConnectivity()
            | _cell_mesh.SimplicialConnectivity()
        ):
            polyhedral = None
        case _ as invalid:
            assert_never(invalid)
    _, _, arrays = prepare_model_recipe_catalog(
        (mesh.coordinates, mesh.blocks, mesh.vertex_global_ids, entity_ids, polyhedral),
        prefix="periodic-lift",
        limits=limits,
    )
    if any(
        isinstance(array, jax.Array) and not array.is_fully_addressable
        for array in arrays.values()
    ):
        raise ValueError(
            "stored_fields.PeriodicMeshTopology.lifted must contain addressable local arrays; got a non-addressable carrier"
        )
    return CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        entity_global_ids=entity_ids,
        polyhedral_connectivity=polyhedral,
        numeric_version=mesh.numeric_version,
    )


def _reconstruct_periodic_topology(
    node: _periodic_topology.PeriodicMeshTopology,
    meshes: tuple[CellMesh, ...],
    limits: ArrayArchiveLimits,
    /,
) -> _periodic_topology.PeriodicMeshTopology:
    """Rebuild quotient descriptors from bounded local lifted carriers."""
    lifted = tuple(
        mesh
        if mesh.periodic_topology is None
        else _reconstruct_periodic_lift(mesh, limits)
        for mesh in meshes
        if (
            mesh.periodic_topology is None and mesh.topology_id == node.lifted_topology_id
        )
        or mesh.periodic_topology is node
    )
    if not lifted:
        raise ValueError(
            "stored_fields.PeriodicMeshTopology.lifted must retain its declared carrier; got missing lifted mesh"
        )
    if any(not content_equal(lifted[0], other, limits=limits) for other in lifted[1:]):
        raise ValueError(
            "stored_fields.PeriodicMeshTopology.lifted must have one content binding; got conflicting carriers"
        )
    return reconstruct_stored_fields(node, bindings={"lifted": lifted[0]})


def _reconstruct_image_descendant(
    node: _multi_state.MultiRegionSurfaceState
    | _multi_geometry.PreparedMultiRegionSurface,
    extractions: Mapping[int, _label_extraction.LabelFieldSurfaceExtractionResult] | None,
    /,
) -> _multi_state.MultiRegionSurfaceState | _multi_geometry.PreparedMultiRegionSurface:
    """Bind image descendants to the retained extraction topology and state."""
    extraction = None if extractions is None else extractions.get(id(node))
    if extraction is None or extraction.topology is None or extraction.state is None:
        raise ValueError(
            f"stored_fields.{type(node).__name__}.extraction must retain its topology and state; got missing image extraction"
        )
    match node:
        case _multi_state.MultiRegionSurfaceState():
            return reconstruct_stored_fields(
                node, bindings={"topology": extraction.topology}
            )
        case _multi_geometry.PreparedMultiRegionSurface():
            return reconstruct_stored_fields(
                node,
                bindings={
                    "topology": extraction.topology,
                    "state": extraction.state,
                    "policy": node.validation_policy,
                },
            )
        case _ as invalid:
            assert_never(invalid)


def _reconstruct_partition(
    node: _partition.CellPartition, meshes: tuple[CellMesh, ...], /
) -> _partition.CellPartition:
    """Bind owner-local partition restoration to its retained mesh storage."""
    if node.storage_id is None:
        return reconstruct_stored_fields(node)
    storage = next(
        (
            mesh.storage
            for mesh in meshes
            if mesh.storage is not None and mesh.storage.storage_id == node.storage_id
        ),
        None,
    )
    if storage is None:
        raise ValueError(
            "stored_fields.CellPartition.storage must retain its mesh ownership binding; got missing storage owner"
        )
    return reconstruct_stored_fields(node, bindings={"storage": storage})


@dataclass(slots=True)
class _SourceOwnerReconstructionScope:
    process_id: int
    admitted: set[tuple[type, int, int, str, str, str]]
    contexts: dict[int, MeshingSourceValidation | None]
    constructors: dict[int, Callable[..., Any]]
    array_digests: dict[tuple[int, int], tuple[jax.Array, str]]
    certifications: dict[tuple[type, int, int, str, str, str], MeshCertificationReport]
    field_owners: dict[tuple[type, int, int, str, str, str], _PreparedMeshingFieldOwner]
    archive_limits: ArrayArchiveLimits


_source_owner_reconstruction_scope: ContextVar[_SourceOwnerReconstructionScope | None] = (
    ContextVar("source_owner_reconstruction_scope", default=None)
)


@contextmanager
def meshing_source_owner_reconstruction_scope() -> Iterator[None]:
    """Reuse authenticated source reconstruction within one request.

    Nested scopes share earned validation, certification, and field preparation.
    Cached owners remain process-local and are released when the outer scope ends.
    """
    current = _source_owner_reconstruction_scope.get()
    if current is not None and current.process_id == getpid():
        yield
        return
    handle = _source_owner_reconstruction_scope.set(
        _SourceOwnerReconstructionScope(
            getpid(), set(), {}, {}, {}, {}, {}, DEFAULT_ARRAY_ARCHIVE_LIMITS
        )
    )
    try:
        yield
    finally:
        _source_owner_reconstruction_scope.reset(handle)


def _source_owner_static_sort_key(value: Any) -> tuple[str, str]:
    return (
        canonical_json(model_structure_recipe(value)),
        _source_owner_static_scalar_bits(value) or "",
    )


def _source_owner_static_scalar_bits(node: Any) -> str | None:
    """Snapshot scalar payload bits along declared owner fields.

    Opaque codecs retain ordinary reconstruction when normalized nonfinite
    recipe tokens cannot authenticate their scalar bytes.
    """
    pending: list[tuple[tuple[int, ...], Any]] = [((), node)]
    seen: set[int] = set()
    records: list[tuple[tuple[int, ...], str, str]] = []
    while pending:
        path, value = pending.pop()
        match value:
            case jax.Array() | np.ndarray():
                continue
            case float() | complex() | np.generic():
                scalar = np.asarray(value)
                records.append((path, scalar.dtype.str, scalar.tobytes().hex()))
                continue
            case FormType() | FormValueSpec():
                pending.append((path, value.to_dict()))
                continue
            case (
                None
                | str()
                | bytes()
                | int()
                | bool()
                | type()
                | np.dtype()
                | Enum()
                | Fraction()
            ):
                continue
            case _:
                pass
        if id(value) in seen:
            continue
        seen.add(id(value))
        if is_dataclass(value):
            children = tuple(
                object.__getattribute__(value, member.name) for member in fields(value)
            )
        elif isinstance(value, Mapping):
            children = tuple(child for pair in value.items() for child in pair)
        elif isinstance(value, (tuple, list)):
            children = tuple(value)
        elif isinstance(value, (set, frozenset)):
            children = tuple(sorted(value, key=_source_owner_static_sort_key))
        else:
            return None
        pending.extend(
            (path + (index,), child)
            for index, child in reversed(tuple(enumerate(children)))
        )
    return canonical_json(records)


def _source_owner_array_digest(
    arrays: Mapping[str, jax.Array | np.ndarray],
    scope: _SourceOwnerReconstructionScope,
    maximum_chunk_bytes: int,
) -> str:
    """Authenticate current named leaves, reusing only live immutable objects."""
    records: list[tuple[str, str]] = []
    for name, value in sorted(arrays.items()):
        if isinstance(value, jax.Array) and not value.is_deleted():
            key = (id(value), maximum_chunk_bytes)
            prior = scope.array_digests.get(key)
            if prior is not None and prior[0] is value:
                digest = prior[1]
            else:
                digest = logical_array_value_collection_digest(
                    {"value": value}, maximum_chunk_bytes=maximum_chunk_bytes
                )
                scope.array_digests[key] = (value, digest)
        else:
            digest = logical_array_value_collection_digest(
                {"value": value}, maximum_chunk_bytes=maximum_chunk_bytes
            )
        records.append((name, digest))
    return canonical_fingerprint(records)


def _source_owner_reconstruction_key(
    node: Any,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None,
    reconstruct: Callable[..., Any],
    scope: _SourceOwnerReconstructionScope,
) -> tuple[type, int, int, str, str, str] | None:
    recipe, _, arrays = prepare_model_recipe_catalog(
        node, prefix="source-owner", limits=limits
    )
    serialized = canonical_json(recipe)
    if '"nonfinite_float"' in serialized:
        scalar_bits = _source_owner_static_scalar_bits(node)
        if scalar_bits is None:
            return None
        serialized = canonical_json((serialized, scalar_bits))
    return (
        type(node),
        id(reconstruct),
        id(source_validation),
        canonical_fingerprint(
            {
                "limits": {
                    member.name: sorted(value) if isinstance(value, frozenset) else value
                    for member in fields(limits)
                    for value in (getattr(limits, member.name),)
                },
                "configuration": configuration_snapshot().to_record(),
            }
        ),
        sha256(serialized.encode("utf-8")).hexdigest(),
        _source_owner_array_digest(
            arrays,
            scope,
            (
                1 << 20
                if source_validation is None
                or source_validation.maximum_chunk_bytes is None
                else source_validation.maximum_chunk_bytes
            ),
        ),
    )


def _validate_context_free_source_owner(
    node: Any,
    reconstruct: Callable[[Any], Any],
    path: str,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None,
) -> Any | None:
    scope = _source_owner_reconstruction_scope.get()
    if scope is None or scope.process_id != getpid():
        return reconstruct(node)
    key = None
    if scope is not None and scope.process_id == getpid():
        try:
            key = _source_owner_reconstruction_key(
                node, limits, source_validation, reconstruct, scope
            )
        except (TypeError, ValueError):
            # Snapshot admission cannot replace the constructor's refusal.
            key = None
        if key is not None and key in scope.admitted:
            return
    fresh = reconstruct(node)
    if fresh is not None and not source_content_equal(
        node,
        fresh,
        limits=limits,
        source_validation=source_validation,
        freshly_rebuilt=True,
    ):
        raise ValueError(
            f"{path}.stored_fields must match its reconstructed source content; got different fields or arrays"
        )
    if key is not None and scope is not None and fresh is not None:
        scope.contexts[id(source_validation)] = source_validation
        scope.constructors[id(reconstruct)] = reconstruct
        scope.admitted.add(key)


def reconstruct_meshing_source(
    node: Any,
    /,
    *,
    meshes: tuple[CellMesh, ...] = (),
    entity_sets: tuple[_topology.EntitySet, ...] = (),
    results: tuple[_result.CellMeshingResult, ...] = (),
    implicit_geometries: Mapping[int, geometry_contracts.CompiledGeometry] | None = None,
    evidence: Mapping[tuple[str, str], _result.AbstractCollectiveMeshTheorem]
    | None = None,
    surface_realizations: Mapping[int, _surface_model.SurfaceRealization] | None = None,
    common_refinements: Mapping[int, PreparedCommonRefinement] | None = None,
    refinement_authorities: Mapping[
        int, _topology_edit.PeriodicNonnestedGeometryAuthority
    ]
    | None = None,
    implicit_profiles: Mapping[str, _analytic_profile.AnalyticImplicitProfile]
    | None = None,
    partition_distributions: Mapping[int, _distribution.MeshDistribution] | None = None,
    image_extractions: Mapping[int, _label_extraction.LabelFieldSurfaceExtractionResult]
    | None = None,
    contact_parts: Mapping[int, tuple[MeshPart, MeshPart]] | None = None,
    source_validation: MeshingSourceValidation | None = None,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    path: str = "source_closure",
) -> Any | None:
    """Revalidate stored fields and bind owners that require external context.

    Owner cases stay explicit to preserve reconstruction and refusal ordering.
    """
    match node:
        case _coupling.ContactCoupling():
            endpoints = None if contact_parts is None else contact_parts.get(id(node))
            if endpoints is None:
                raise ValueError(
                    "stored_fields.ContactCoupling.endpoints must retain both assembly part revisions; got missing endpoint binding"
                )
            return reconstruct_stored_fields(
                node, bindings={"source": endpoints[0], "target": endpoints[1]}
            )
        case (
            _multi_state.MultiRegionSurfaceState()
            | _multi_geometry.PreparedMultiRegionSurface()
        ):
            return _reconstruct_image_descendant(node, image_extractions)
        case _regions.SegmentMesh():
            return reconstruct_stored_fields(
                node, bindings={"edges": node.topology.edges}
            )
        case _partition.CellPartition():
            return _reconstruct_partition(node, meshes)
        case _restart_distribution.SimplexRestartRepack():
            _restart_distribution.simplex_restart_repack_content_id(node)
            return reconstruct_stored_fields(node)
        case _distribution.MeshPartitionEvidence():
            distribution = (
                None
                if partition_distributions is None
                else partition_distributions.get(id(node))
            )
            if distribution is None:
                raise ValueError(
                    "stored_fields.MeshPartitionEvidence.distribution must retain its ownership and balance inputs; got missing distribution"
                )
            return reconstruct_stored_fields(distribution).evidence
        case _cell_geometry.CellGeometryStorageProjection():
            logical_arrays = next(
                (
                    mesh.storage.logical_arrays
                    for mesh in meshes
                    if mesh.storage is not None
                    and mesh.storage.geometry_projection is node
                ),
                None,
            )
            if logical_arrays is None:
                raise ValueError(
                    "stored_fields.CellGeometryStorageProjection.logical_arrays must retain its storage query binding; got missing storage owner"
                )
            return reconstruct_stored_fields(
                node,
                bindings={
                    "logical_arrays": logical_arrays,
                    "source_elements": dict(node.source_elements),
                },
                array_equal=None
                if source_validation is None
                else source_validation.array_equal,
            )
        case source_fidelity_owner.ImplicitProjectionCoverageEvidence():
            profile = (
                None
                if implicit_profiles is None
                else implicit_profiles.get(node.profile_id)
            )
            if profile is None:
                raise ValueError(
                    "stored_fields.ImplicitProjectionCoverageEvidence.profile must retain its source query profile; got missing profile"
                )
            return reconstruct_stored_fields(node, bindings={"profile": profile})
        case CommonRefinementEvidence():
            refinement = (
                None if common_refinements is None else common_refinements.get(id(node))
            )
            if refinement is None:
                raise ValueError(
                    "stored_fields.CommonRefinementEvidence.measures must retain its refinement owner; got missing refinement"
                )
            return reconstruct_stored_fields(
                node,
                bindings={
                    "source_measures": refinement.source_measures,
                    "target_measures": refinement.target_measures,
                },
            )
        case PreparedCommonRefinement():
            authority = (
                None
                if refinement_authorities is None
                else refinement_authorities.get(id(node))
            )
            if authority is None:
                raise ValueError(
                    "stored_fields.PreparedCommonRefinement.meshes must retain its source and target geometry owner; got missing owner"
                )
            return reconstruct_stored_fields(
                node,
                bindings={
                    "source_mesh": authority.source,
                    "target_mesh": authority.target,
                    "source_geometry": authority.source_geometry,
                    "target_geometry": authority.target_geometry,
                },
            )
        case GeometryAssociation():
            return reconstruct_distributed_stored_fields(node, limits=limits)
        case _periodic_topology.PeriodicMeshTopology():
            return _reconstruct_periodic_topology(node, meshes, limits)
        case (
            _surface_contracts.SurfaceAuditReport()
            | _surface_contracts.SurfaceValidityCertificate()
        ):
            realization = (
                None
                if surface_realizations is None
                else surface_realizations.get(id(node))
            )
            if realization is None:
                raise ValueError(
                    f"stored_fields.{type(node).__name__}.realization must retain its surface admission context; got missing realization"
                )
            match node:
                case _surface_contracts.SurfaceAuditReport():
                    return reconstruct_stored_fields(
                        node, bindings={"policy": realization.policy}
                    )
                case _surface_contracts.SurfaceValidityCertificate():
                    return reconstruct_stored_fields(
                        node, bindings={"report": realization.audit}
                    )
        case (
            organization_records_owner.RegionMeshingEvidence()
            | organization_records_owner.RegionBoundaryEvidence()
        ):
            return _reconstruct_region_evidence(node, results, limits)
        case _topology.OrientedIncidence():
            return _reconstruct_oriented_incidence(node, entity_sets)
        case PreparedSurfaceSourceSupport():
            return _reconstruct_surface_support(node, results)
        case MeshingScope():
            return _reconstruct_scope(node, meshes, results, evidence)
        case _scope.MeshScopeProjection():
            return _reconstruct_scope_projection(node, results, evidence)
        case _periodic.PeriodicQuotientEvidence():
            return _reconstruct_periodic_evidence(node, meshes, results, limits=limits)
        case (
            _implicit_projection.ImplicitPointProjectionPlan()
            | _implicit_realization.ImplicitSurfacePlan()
        ):
            original = (
                None if implicit_geometries is None else implicit_geometries.get(id(node))
            )
            if original is None:
                raise ValueError(
                    f"{type(node).__name__}.geometry must retain its compiled source in the design closure; got missing geometry"
                )
            match node:
                case _implicit_projection.ImplicitPointProjectionPlan():
                    return _implicit_projection.ImplicitPointProjectionPlan.from_stored_fields(
                        node, geometry=original
                    )
                case _implicit_realization.ImplicitSurfacePlan():
                    return _implicit_realization.ImplicitSurfacePlan.from_stored_fields(
                        node, geometry=original
                    )
                case _ as invalid:
                    assert_never(invalid)
        case (
            _proposals.MeshProposalFeatures()
            | _proposals.MeshMarkingProposal()
            | _proposals.MeshProposalSafetyPolicy()
            | _proposals.MeshProposalProjection()
        ):
            return _reconstruct_proposal(node, results)
        case _:
            if type(node) in _ADMITTED_MODEL_TYPES:
                return _validate_context_free_source_owner(
                    node, reconstruct_stored_fields, path, limits, source_validation
                )
            admission = _source_admissions().get(type(node))
            if admission is None:
                raise TypeError(
                    f"source_closure.owner must have an admitted registered type; got {type(node).__name__}"
                )
            if admission.reconstruct is None:
                return None
            return _validate_context_free_source_owner(
                node, admission.reconstruct, path, limits, source_validation
            )


def register_meshing_source_artifacts() -> None:
    "Admit built-in scientific owners to the artifact registry."
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
    register_artifact_value_codec(
        ArtifactValueCodec(
            "phydrax.meshing:GeometryAssociationSourceBank",
            GeometryAssociationSourceBank,
            GeometryAssociationSourceBank.to_record,
            GeometryAssociationSourceBank.from_record,
        )
    )
    for admission in _source_admissions().values():
        identity = registered_artifact_value_id(admission.owner)
        if identity is None:
            identity = f"{admission.namespace}:{admission.owner.__name__}"
        register_artifact_value(
            identity, admission.owner, validate_restored=admission.validate_restored
        )


@checked
def source_content_equal(
    first: Any,
    second: Any,
    /,
    *,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None = None,
    freshly_rebuilt: bool = False,
) -> bool:
    "Compare every authored field through bounded logical content identity.\n\n    Object sharing and NumPy mutability are archive storage metadata, not\n    geometric/source identity; the durable archive retains its recipe.\n"
    if isinstance(first, MeshingSourceClosure):
        first = first.to_records()
    if isinstance(second, MeshingSourceClosure):
        second = second.to_records()
    if (
        freshly_rebuilt
        and type(first) is _result.CellMeshingResult
        and type(second) is _result.CellMeshingResult
    ):
        # The result owner revalidates its admitted children and recomputes the
        # content-bound result ID. Expanding that graph a second time can exceed
        # the unchanged per-recipe manifest limit.
        return first.result_id == second.result_id
    if freshly_rebuilt and type(first) is MeshPart and type(second) is MeshPart:
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
        return source_content_equal(
            first.carrier,
            second.carrier,
            limits=limits,
            source_validation=source_validation,
            freshly_rebuilt=True,
        )
    if source_validation is not None:
        return source_validation.equivalent(first, second, limits=limits)

    return content_equal(first, second, limits=limits)


def _validate_periodic_geometry_source(
    node: _topology_edit.PeriodicNonnestedGeometryAuthority, /
) -> None:
    periodic = node.source.periodic_topology
    if periodic is None or node.target.mesh_id != node.common_refinement.target_mesh_id:
        raise ValueError(
            "source_closure.periodic_geometry.source must bind its retained source and target meshes; got a missing quotient or different target"
        )
    bounds = tuple(
        _cell_geometry.cell_geometry_preparation_bounds(geometry)
        for geometry in (node.source_geometry, node.target_geometry)
    )
    queries = sum(
        _cell_geometry.cell_geometry_preparation_query_upper(geometry)
        for geometry in (node.source_geometry, node.target_geometry)
    )
    with _cell_geometry.owned_coordinate_preparation(
        sum(work for work, _ in bounds),
        sum(storage for _, storage in bounds),
        maximum_geometry_queries=queries,
    ) as budget:
        _topology_edit.require_periodic_nonnested_geometry(
            periodic, node.target, node, budget=budget
        )


def _validate_periodic_orbit_source(
    node: _topology_edit.PeriodicVertexOrbitWitness,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None,
    /,
) -> None:
    source = node.nonnested_geometry
    if source is None:
        raise ValueError(
            "source_closure.periodic_orbits.source must retain its nonnested geometry binding; got missing source"
        )
    work_upper, bytes_upper, query_upper = (
        _periodic.periodic_topology_bind_preparation_bounds(
            source.source, source.target, node
        )
    )
    with _cell_geometry.owned_coordinate_preparation(
        work_upper, bytes_upper, maximum_geometry_queries=query_upper
    ) as budget:
        target = _periodic.bind_periodic_topology_edit(
            source.source, source.target, node, budget=budget
        )
    if not source_content_equal(
        target, source.target, limits=limits, source_validation=source_validation
    ):
        raise ValueError(
            "source_closure.periodic_orbits.target must match its retained SCI descriptor; got different target content"
        )


def _validate_topology_edit_source(
    node: _topology_edit.CellTopologyEdit,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None,
    /,
) -> None:
    orbits = node.periodic_orbits
    source = None if orbits is None else orbits.nonnested_geometry
    if source is None:
        raise ValueError(
            "source_closure.topology_edit.source must retain its nonnested source and target; got missing source"
        )
    work_upper, bytes_upper, query_upper = (
        _periodic.topology_edit_assembly_preparation_bounds(source.source, node)
    )
    with _cell_geometry.owned_coordinate_preparation(
        work_upper, bytes_upper, maximum_geometry_queries=query_upper
    ) as budget:
        target, _, _ = _topology_edit.assemble_topology_edit(
            source.source,
            node,
            numeric_version=source.target.numeric_version,
            budget=budget,
        )
    if not source_content_equal(
        target, source.target, limits=limits, source_validation=source_validation
    ):
        raise ValueError(
            "source_closure.topology_edit.target must match its independently bound target; got different target content"
        )


def _validate_source_owner(
    node: Any,
    path: str,
    model: _model.BRepModel | None,
    /,
    *,
    limits: ArrayArchiveLimits,
    validation_context: RestorationValidationContext,
    source_validation: MeshingSourceValidation | None = None,
    meshes: tuple[CellMesh, ...],
    entity_sets: tuple[_topology.EntitySet, ...] = (),
    results: tuple[_result.CellMeshingResult, ...] = (),
    implicit_geometries: Mapping[int, geometry_contracts.CompiledGeometry] | None = None,
    evidence: Mapping[tuple[str, str], _result.AbstractCollectiveMeshTheorem]
    | None = None,
    surface_realizations: Mapping[int, _surface_model.SurfaceRealization] | None = None,
    common_refinements: Mapping[int, PreparedCommonRefinement] | None = None,
    refinement_authorities: Mapping[
        int, _topology_edit.PeriodicNonnestedGeometryAuthority
    ]
    | None = None,
    implicit_profiles: Mapping[str, _analytic_profile.AnalyticImplicitProfile]
    | None = None,
    partition_distributions: Mapping[int, _distribution.MeshDistribution] | None = None,
    image_extractions: Mapping[int, _label_extraction.LabelFieldSurfaceExtractionResult]
    | None = None,
    contact_parts: Mapping[int, tuple[MeshPart, MeshPart]] | None = None,
) -> None:
    match node:
        case (
            PreparedCoordinateLatticeExecution()
            | MetricRemeshingEvidence()
            | MetricOptimizerEvidence()
            | MetricMemoryEvidence()
            | PreparedSurfaceCurveWitness()
            | _periodic_cell.PeriodicCell()
        ):
            node.validate_restored()
        case (
            _level_set.LevelSetEvidence()
            | PreparedCompartmentVolume()
            | _geometry_compartments.CompartmentMeshingSource()
            | _label_extraction.LabelFieldVolumeBinding()
            | _label_extraction.LabelFieldSurfaceExtractionResult()
            | _decision.NativeDesignSourceState()
        ):
            node.validate_source_integrity()
        case (
            FiniteElementDofMap()
            | FiniteElementDiscretization()
            | _fem_distributed.FiniteElementExecutionLimits()
            | _fem_distributed.FiniteElementGlobalDofOwnership()
            | _fem_distributed.FiniteElementDofOwnershipPlan()
            | _fem_distributed.OwnerLocalFiniteElementDiscretization()
            | _fem_distributed.OwnerLocalFiniteElementTransfer()
            | _fem_distributed.FiniteElementClosurePreparation()
        ):
            node.validate_restored(validation_context=validation_context)
        case PointCloudPlan():
            require_point_cloud_plan_integrity(node)
        case _topology_edit.PeriodicNonnestedGeometryAuthority():
            _validate_periodic_geometry_source(node)
        case _topology_edit.PeriodicVertexOrbitWitness():
            _validate_periodic_orbit_source(node, limits, source_validation)
        case _topology_edit.CellTopologyEdit():
            _validate_topology_edit_source(node, limits, source_validation)
        case SourceGeometryRealization():
            node.require_current()
        case _:
            # Other registered owners are validated by their stored constructor.
            pass
    fresh = reconstruct_meshing_source(
        node,
        meshes=meshes,
        entity_sets=entity_sets,
        results=results,
        implicit_geometries=implicit_geometries,
        evidence=evidence,
        surface_realizations=surface_realizations,
        common_refinements=common_refinements,
        refinement_authorities=refinement_authorities,
        implicit_profiles=implicit_profiles,
        partition_distributions=partition_distributions,
        image_extractions=image_extractions,
        contact_parts=contact_parts,
        source_validation=source_validation,
        limits=limits,
        path=path,
    )
    if fresh is not None and not source_content_equal(
        node,
        fresh,
        limits=limits,
        source_validation=source_validation,
        freshly_rebuilt=True,
    ):
        raise ValueError(
            f"{path}.stored_fields must match its reconstructed source content; got different fields or arrays"
        )


@dataclass(frozen=True, slots=True)
class _SourceCollection:
    """Host traversal inventories, not a numerical model PyTree."""

    admitted: frozenset[type]
    meshes: tuple[CellMesh, ...]
    entity_sets: tuple[_topology.EntitySet, ...]
    results: tuple[_result.CellMeshingResult, ...]
    evidence: dict[tuple[str, str], _result.AbstractCollectiveMeshTheorem]
    material_children: set[int]
    image_children: set[int]
    finite_element_children: set[int]
    topology_edit_children: set[int]
    grid_children: set[int]
    design_plans: set[int]
    implicit_geometries: dict[int, geometry_contracts.CompiledGeometry]
    accepted_epoch_nodes: set[int]
    surface_realizations: dict[int, _surface_model.SurfaceRealization]
    common_refinements: dict[int, PreparedCommonRefinement]
    refinement_authorities: dict[int, _topology_edit.PeriodicNonnestedGeometryAuthority]
    implicit_profiles: dict[str, _analytic_profile.AnalyticImplicitProfile]
    partition_distributions: dict[int, _distribution.MeshDistribution]
    image_extractions: dict[int, _label_extraction.LabelFieldSurfaceExtractionResult]
    contact_parts: dict[int, tuple[MeshPart, MeshPart]]


def _collect_implicit_profile(
    source: source_fidelity_owner.ImplicitProjectionBoundarySource,
    profiles: dict[str, _analytic_profile.AnalyticImplicitProfile],
    /,
) -> None:
    match source.profile:
        case _analytic_profile.AnalyticImplicitProfile() as profile:
            previous = profiles.get(profile.profile_id)
            if previous is not None and not content_equal(
                previous, profile, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
            ):
                raise ValueError(
                    "stored_fields.ImplicitProjectionCoverageEvidence.profile must have one retained source binding; got conflicting profiles"
                )
            profiles[profile.profile_id] = profile
        case invalid:
            raise TypeError(
                f"stored_fields.ImplicitProjectionBoundarySource.profile must retain AnalyticImplicitProfile; got {type(invalid).__name__}"
            )


def _collect_contact_bindings(
    assembly: _assembly.MeshAssembly,
    bindings: dict[int, tuple[MeshPart, MeshPart]],
    /,
) -> None:
    """Retain contact endpoint revisions from their assembled parts."""
    by_name = {part.name: part for part in assembly.parts}
    if len(by_name) != len(assembly.parts):
        raise ValueError(
            "stored_fields.ContactCoupling.endpoints must have unique assembly part names; got repeated names"
        )
    for overlay in assembly.couplings:
        if type(overlay) is not _coupling.ContactCoupling:
            continue
        source = by_name.get(overlay.source_scope.source_id)
        target = by_name.get(overlay.target_scope.source_id)
        if source is None or target is None:
            raise ValueError(
                "stored_fields.ContactCoupling.endpoints must retain both assembly parts; got missing source or target part"
            )
        endpoints = (source, target)
        previous = bindings.get(id(overlay))
        if previous is not None and not content_equal(
            previous, endpoints, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
        ):
            raise ValueError(
                "stored_fields.ContactCoupling.endpoints must have one retained revision binding; got conflicting assembly parts"
            )
        bindings[id(overlay)] = endpoints


def _collect_image_bindings(
    extraction: _label_extraction.LabelFieldSurfaceExtractionResult,
    bindings: dict[int, _label_extraction.LabelFieldSurfaceExtractionResult],
    /,
) -> None:
    """Retain image topology and state inputs for each prepared descendant."""
    for child in (extraction.state, extraction.surface):
        if child is None:
            continue
        previous = bindings.get(id(child))
        if previous is not None and not content_equal(
            (previous.topology, previous.state),
            (extraction.topology, extraction.state),
            limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        ):
            raise ValueError(
                f"stored_fields.{type(child).__name__}.extraction must have one retained topology and state binding; got conflicting image extractions"
            )
        bindings[id(child)] = extraction


def _collect_retained_source_bindings(
    node: Any,
    implicit_profiles: dict[str, _analytic_profile.AnalyticImplicitProfile],
    common_refinements: dict[int, PreparedCommonRefinement],
    refinement_authorities: dict[int, _topology_edit.PeriodicNonnestedGeometryAuthority],
    surface_realizations: dict[int, _surface_model.SurfaceRealization],
    design_plans: set[int],
    implicit_geometries: dict[int, geometry_contracts.CompiledGeometry],
    partition_distributions: dict[int, _distribution.MeshDistribution],
    image_extractions: dict[int, _label_extraction.LabelFieldSurfaceExtractionResult],
    contact_parts: dict[int, tuple[MeshPart, MeshPart]],
    /,
) -> None:
    """Retain unique scientific bindings before descending into their children."""
    match node:
        case _assembly.MeshAssembly():
            _collect_contact_bindings(node, contact_parts)
        case _label_extraction.LabelFieldSurfaceExtractionResult():
            _collect_image_bindings(node, image_extractions)
        case _distribution.MeshDistribution():
            previous = partition_distributions.get(id(node.evidence))
            if (
                previous is not None
                and previous is not node
                and not content_equal(previous, node, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
            ):
                raise ValueError(
                    "stored_fields.MeshPartitionEvidence.distribution must have one retained ownership binding; got conflicting distributions"
                )
            partition_distributions[id(node.evidence)] = node
        case source_fidelity_owner.ImplicitProjectionBoundarySource():
            _collect_implicit_profile(node, implicit_profiles)
        case PreparedCommonRefinement():
            previous_refinement = common_refinements.get(id(node.evidence))
            if previous_refinement is not None and not content_equal(
                (
                    previous_refinement.source_measures,
                    previous_refinement.target_measures,
                ),
                (node.source_measures, node.target_measures),
                limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
            ):
                raise ValueError(
                    "stored_fields.CommonRefinementEvidence.measures must have one retained refinement binding; got conflicting measures"
                )
            common_refinements[id(node.evidence)] = node
        case _topology_edit.PeriodicNonnestedGeometryAuthority():
            previous_authority = refinement_authorities.get(id(node.common_refinement))
            if previous_authority is not None and not content_equal(
                (
                    previous_authority.source,
                    previous_authority.target,
                    previous_authority.source_geometry,
                    previous_authority.target_geometry,
                ),
                (
                    node.source,
                    node.target,
                    node.source_geometry,
                    node.target_geometry,
                ),
                limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
            ):
                raise ValueError(
                    "stored_fields.PreparedCommonRefinement.meshes must have one retained geometry binding; got conflicting authorities"
                )
            refinement_authorities[id(node.common_refinement)] = node
        case _surface_model.SurfaceRealization():
            for child in (node.audit, node.certificate):
                previous = surface_realizations.get(id(child))
                if previous is not None and not content_equal(
                    previous.policy, node.policy, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
                ):
                    raise ValueError(
                        f"stored_fields.{type(child).__name__}.policy must have one retained binding; got conflicting policies"
                    )
                surface_realizations[id(child)] = node
        case _decision.NativeDesignSourceState():
            if type(node.plan.prepared) is not _implicit_realization.ImplicitSurfacePlan:
                raise TypeError(
                    "source_closure.design.plan must retain an implicit source preparation; got unsupported preparation"
                )
            design_plans.add(id(node.plan))
            implicit_geometries[id(node.plan.prepared)] = node.source.geometry
            implicit_geometries[id(node.plan.prepared.projection)] = node.source.geometry
        case _:
            pass


def _collect_owned_source_children(
    node: Any,
    grid_children: set[int],
    result_nodes: dict[int, _result.CellMeshingResult],
    evidence_nodes: dict[tuple[str, str], _result.AbstractCollectiveMeshTheorem],
    topology_edit_children: set[int],
    finite_element_children: set[int],
    image_children: set[int],
    material_children: set[int],
    /,
) -> None:
    """Record child ownership before traversal admits detached source values."""
    match node:
        case _tensor_support.PreparedTensorGrid():
            grid_children.update(
                id(child) for child in (*node.structured_axes, *node.entity_layouts)
            )
        case _result.CellMeshingResult():
            result_nodes[id(node)] = node
        case _result.AbstractCollectiveMeshTheorem():
            evidence_nodes.setdefault((node.evidence_id, node.mesh_id), node)
        case _topology_edit.CellTopologyEdit():
            topology_edit_children.update(
                id(child)
                for child in (
                    *node.blocks,
                    *node.relations,
                    *node.prescribed_entity_ids,
                )
            )
            if node.shared_faces is not None:
                topology_edit_children.add(id(node.shared_faces))
        case FiniteElementDiscretization():
            finite_element_children.add(id(node.default_runtime))
            finite_element_children.update(
                id(geometry) for block in node.block_geometries for geometry in block
            )
        case _label_extraction.LabelFieldSurfaceExtractionResult():
            image_children.update(
                id(child) for child in (node.state, node.surface) if child is not None
            )
        case PreparedSurfaceChartDeformation():
            for occurrence in node.occurrences:
                material_children.add(id(occurrence))
                material_children.update(id(piece) for piece in occurrence.pieces)
                material_children.update(id(piece) for piece in occurrence.root_pieces)
        case _:
            pass


def _collect_source_values(value: Any, /) -> _SourceCollection:
    admitted = (
        frozenset(cls for _, types in meshing_source_artifact_types() for cls in types)
        | _ADMITTED_MODEL_TYPES
    )
    mesh_nodes: dict[int, CellMesh] = {}
    entity_nodes: dict[int, _topology.EntitySet] = {}
    result_nodes: dict[int, _result.CellMeshingResult] = {}
    evidence_nodes: dict[tuple[str, str], _result.AbstractCollectiveMeshTheorem] = {}
    collected: set[int] = set()
    material_children: set[int] = set()
    image_children: set[int] = set()
    finite_element_children: set[int] = set()
    topology_edit_children: set[int] = set()
    grid_children: set[int] = set()
    design_plans: set[int] = set()
    implicit_geometries: dict[int, geometry_contracts.CompiledGeometry] = {}
    accepted_epoch_nodes: set[int] = set()
    surface_realizations: dict[int, _surface_model.SurfaceRealization] = {}
    common_refinements: dict[int, PreparedCommonRefinement] = {}
    refinement_authorities: dict[
        int, _topology_edit.PeriodicNonnestedGeometryAuthority
    ] = {}
    implicit_profiles: dict[str, _analytic_profile.AnalyticImplicitProfile] = {}
    partition_distributions: dict[int, _distribution.MeshDistribution] = {}
    image_extractions: dict[int, _label_extraction.LabelFieldSurfaceExtractionResult] = {}
    contact_parts: dict[int, tuple[MeshPart, MeshPart]] = {}
    if isinstance(value, Mapping):
        accepted = value.get("accepted_data")
        if (
            isinstance(accepted, Mapping)
            and type(accepted.get("accepted_epochs")) is tuple
        ):
            accepted_epoch_nodes.update(
                id(epoch) for epoch in accepted["accepted_epochs"]
            )

    def collect(node: Any, /) -> None:
        if id(node) in collected:
            return
        collected.add(id(node))
        _collect_retained_source_bindings(
            node,
            implicit_profiles,
            common_refinements,
            refinement_authorities,
            surface_realizations,
            design_plans,
            implicit_geometries,
            partition_distributions,
            image_extractions,
            contact_parts,
        )
        _collect_owned_source_children(
            node,
            grid_children,
            result_nodes,
            evidence_nodes,
            topology_edit_children,
            finite_element_children,
            image_children,
            material_children,
        )
        match node:
            case CellMesh():
                mesh_nodes[id(node)] = node
                for degree in range(node.topological_dimension + 1):
                    entities = node.entity_set(degree)
                    entity_nodes[id(entities)] = entities
                for member in fields(node):
                    collect(object.__getattribute__(node, member.name))
            case _topology.EntitySet():
                entity_nodes[id(node)] = node
                for member in fields(node):
                    collect(object.__getattribute__(node, member.name))
            case dict() | frozendict() | MappingProxyType():
                for child in node.values():
                    collect(child)
            case tuple() | frozenset():
                for child in node:
                    collect(child)
            case _:
                if type(node) in admitted and is_dataclass(node):
                    for member in fields(node):
                        collect(object.__getattribute__(node, member.name))

    collect(value)
    return _SourceCollection(
        admitted,
        tuple(mesh_nodes.values()),
        tuple(entity_nodes.values()),
        tuple(result_nodes.values()),
        evidence_nodes,
        material_children,
        image_children,
        finite_element_children,
        topology_edit_children,
        grid_children,
        design_plans,
        implicit_geometries,
        accepted_epoch_nodes,
        surface_realizations,
        common_refinements,
        refinement_authorities,
        implicit_profiles,
        partition_distributions,
        image_extractions,
        contact_parts,
    )


@dataclass(frozen=True, slots=True)
class _SourceVisitContext:
    """Retain one host admission walk without flattening its source graph."""

    collection: _SourceCollection
    limits: ArrayArchiveLimits
    restoration: RestorationValidationContext
    validation: MeshingSourceValidation | None
    active: set[int]
    visited: set[tuple[int, int, int | None]]


def _require_source_parent(
    node: Any, path: str, collection: _SourceCollection, /
) -> None:
    match node:
        case MeshingAcceptedEpoch():
            permitted = collection.accepted_epoch_nodes
            condition = "retain its ordered accepted epoch history"
        case NativeMeshingPlan():
            permitted = collection.design_plans
            condition = "retain its native design source closure"
        case _tensor_entities.StructuredAxis() | _tensor_entities.TensorEntityLayout():
            permitted = collection.grid_children
            condition = "retain its materialized tensor grid"
        case PreparedSurfaceChartPiece() | PreparedSurfaceChartOccurrence():
            permitted = collection.material_children
            condition = "retain its material deformation and source banks"
        case (
            _label_extraction.MultiRegionSurfaceState()
            | _label_extraction.PreparedMultiRegionSurface()
        ):
            permitted = collection.image_children
            condition = "retain its extraction topology, state and controls"
        case FiniteElementRuntimeData() | FiniteElementBlockGeometry():
            permitted = collection.finite_element_children
            condition = "retain its finite-element plan and source geometry"
        case (
            _topology_edit.TopologyEditBlock()
            | _topology_edit.PolyhedralTopologyEditBlock()
            | _topology_edit.EntityRelations()
            | _topology_edit.PrescribedEntityIds()
            | _topology_edit.SharedFaceWitnesses()
        ):
            permitted = collection.topology_edit_children
            condition = "retain its independently bound topology edit"
        case _:
            return
    if id(node) not in permitted:
        raise ValueError(f"{path} must {condition}; got detached {type(node).__name__}")


def _validate_collected_source(
    node: Any, path: str, model: _model.BRepModel | None, context: _SourceVisitContext, /
) -> None:
    collection = context.collection
    _validate_source_owner(
        node,
        path,
        model,
        limits=context.limits,
        validation_context=context.restoration,
        source_validation=context.validation,
        meshes=collection.meshes,
        entity_sets=collection.entity_sets,
        results=collection.results,
        implicit_geometries=collection.implicit_geometries,
        evidence=collection.evidence,
        surface_realizations=collection.surface_realizations,
        common_refinements=collection.common_refinements,
        refinement_authorities=collection.refinement_authorities,
        implicit_profiles=collection.implicit_profiles,
        partition_distributions=collection.partition_distributions,
        image_extractions=collection.image_extractions,
        contact_parts=collection.contact_parts,
    )


def _visit_source_fields(
    node: Any,
    path: str,
    model: _model.BRepModel | None,
    batch_parts: int | None,
    context: _SourceVisitContext,
    /,
) -> None:
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
        if batched and isinstance(child, jax.Array) and child.shape[0] != batch_parts:
            raise ValueError(
                f"{path}.{member.name} must bind the collective partition axis; got {child.shape[0]}, expected {batch_parts}"
            )
        child_model = (
            None
            if type(node) is _query.PreparedBRepQuery and member.name == "world_query"
            else model
        )
        _visit_source_value(
            child, f"{path}.{member.name}", child_model, child_parts, context
        )
    _validate_collected_source(node, path, model, context)


def _visit_source_value(
    node: Any,
    path: str,
    model: _model.BRepModel | None,
    batch_parts: int | None,
    context: _SourceVisitContext,
    /,
) -> None:
    _require_source_parent(node, path, context.collection)
    match node:
        case jax.Array() | np.ndarray():
            if node.dtype.hasobject:
                raise TypeError(
                    f"{path}.dtype must exclude object arrays and live handles; got {node.dtype}"
                )
            return
        case np.dtype():
            if node.hasobject:
                raise TypeError(f"{path}.dtype must exclude objects; got {node}")
            return
        case Enum():
            if type(node) not in context.collection.admitted:
                raise TypeError(
                    f"{path} must have an admitted enum type; got {type(node).__name__}"
                )
            return
        case np.generic():
            _visit_source_value(node.item(), path, model, batch_parts, context)
            return
        case _:
            if node is None or type(node) in (str, bool, int, float, bytes, Fraction):
                return
    if callable(node) and type(node) not in context.collection.admitted:
        raise TypeError(
            f"{path} must exclude callbacks, providers and live handles; got {type(node).__name__}"
        )
    if id(node) in context.active:
        raise ValueError(f"{path} must be acyclic; got a cyclic source closure")
    key = (id(node), id(model), batch_parts)
    if key in context.visited:
        return
    context.active.add(id(node))
    source_model = model
    match node:
        case _query.PreparedBRepQuery():
            if model is not None and not source_content_equal(
                node.model,
                model,
                limits=context.limits,
                source_validation=context.validation,
            ):
                raise ValueError(
                    f"{path}.model must match its defining BRepModel; got a different model"
                )
            source_model = node.model
        case _source._NativeBRepKernel():
            source_model = node.model
        case _:
            pass
    try:
        _visit_source_children(node, path, source_model, batch_parts, context)
        context.visited.add(key)
    finally:
        context.active.remove(id(node))


def _visit_source_children(
    node: Any,
    path: str,
    model: _model.BRepModel | None,
    batch_parts: int | None,
    context: _SourceVisitContext,
    /,
) -> None:
    kind = type(node)
    if kind not in context.collection.admitted and kind not in (
        tuple,
        frozenset,
        dict,
        frozendict,
        MappingProxyType,
    ):
        raise TypeError(f"{path} must have an admitted source type; got {kind.__name__}")
    match node:
        case _topology_edit.PeriodicEntityIdentityBank():
            _periodic.require_periodic_identity_bank(node)
        case tuple():
            names = (
                node._fields
                if kind is not tuple
                else tuple(str(index) for index in range(len(node)))
            )
            for name, child in zip(names, node, strict=True):
                child_path = f"{path}[{name}]" if kind is tuple else f"{path}.{name}"
                _visit_source_value(child, child_path, model, None, context)
            if kind in (
                _topology_edit.PeriodicNonnestedGeometryAuthority,
                _topology_edit.PeriodicVertexOrbitWitness,
                _topology_edit.CellTopologyEdit,
            ):
                _validate_collected_source(node, path, model, context)
        case frozenset():
            for child in node:
                _visit_source_value(child, f"{path}.item", model, None, context)
        case dict() | frozendict() | MappingProxyType():
            for name, child in node.items():
                if type(name) is not str:
                    raise TypeError(
                        f"{path}.keys must be strings; got {type(name).__name__}"
                    )
                _visit_source_value(child, f"{path}.{name}", model, None, context)
        case FormType() | FormValueSpec():
            validate_model_structure_recipe(
                model_structure_recipe(node), limits=context.limits
            )
            fresh = rebuild_meshing_field_record(node)
            if type(fresh) is not kind or fresh != node:
                raise ValueError(
                    f"{path} must match its reconstructed immutable value; got changed fields"
                )
        case _:
            if not is_dataclass(node):
                raise TypeError(
                    f"{path} must have registered stored fields; got {kind.__name__}"
                )
            _visit_source_fields(node, path, model, batch_parts, context)


def _validate_source_value(
    value: Any,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    source_validation: MeshingSourceValidation | None = None,
) -> None:
    """Collect source bindings, then validate every field in the source graph."""
    register_meshing_source_artifacts()
    scope = _source_owner_reconstruction_scope.get()
    if scope is not None and scope.process_id == getpid():
        scope.archive_limits = limits
    collection = _collect_source_values(value)
    context = _SourceVisitContext(
        collection,
        limits,
        RestorationValidationContext(),
        source_validation,
        set(),
        set(),
    )
    _visit_source_value(value, "source_closure", None, None, context)


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
                "source_closure.generation_source must retain its declared source type and fields; got generation_source requires an native authored source"
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
        raise TypeError(
            "source_closure.generation_source must retain its declared source type and fields; got Generation source maps require the overset registration"
        )
    request = records["certification_inputs"]
    named = records.get("primary_generation_part")
    if named is not None:
        if type(named) is not str or named not in sources:
            raise ValueError(
                "source_closure.generation_source must match its retained source binding; got Primary generation part must explicitly name one retained registration occurrence"
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
            "source_closure.generation_source must retain its declared source type and fields; got Primary certification request must identify one registered carrier; repeated requests require primary_generation_part"
        )
    part = selected[0]
    if (
        type(part.carrier) is not _result.CellMeshingResult
        or part.carrier.certification is None
    ):
        raise ValueError(
            "source_closure.generation_source must match its retained source binding; got Primary generation part must retain its certified scientific carrier"
        )
    if not source_content_equal(
        part.carrier.certification.request,
        request,
        limits=limits,
        source_validation=source_validation,
    ):
        raise ValueError(
            "source_closure.generation_source must match its retained source binding; got Primary generation occurrence must bind the retained scientific request"
        )
    if not source_content_equal(
        part.carrier.certification,
        records["report"],
        limits=limits,
        source_validation=source_validation,
    ):
        raise ValueError(
            "source_closure.generation_source must match its retained source binding; got Primary generation occurrence must bind the retained scientific report"
        )
    source = sources[part.name]
    if not is_native_generation_source(source):
        raise TypeError(
            "source_closure.generation_source must retain its declared source type and fields; got Generation registration requires its native source"
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
                "source_closure.generation_specification must retain its declared source type and fields; got Named generation sources and hard requests require their full registration"
            )
        part, source = records.get("generation_part"), records.get("generation_source")
        accepted = records.get("accepted_data")
        if isinstance(accepted, Mapping) and "accepted_epochs" in accepted:
            epochs = validate_meshing_accepted_epochs(records)
            part = MeshPart(epochs[0].declarations[0].part_name, epochs[0].carrier)
            if not is_native_generation_source(source):
                raise TypeError(
                    f"source_closure.generation_source must retain a native generation owner; got {type(source).__name__}"
                )
            specification = records.get("generation_specification")
            if not isinstance(
                specification,
                (_contracts.SurfaceMeshingSpec, _contracts.VolumeMeshingSpec),
            ):
                raise TypeError(
                    f"source_closure.generation_specification must retain a surface or volume request; got {type(specification).__name__}"
                )
            validate_registered_native_part(
                part,
                source,
                specification,
                records.get("generation_options"),
                limits=limits,
            )
            return
        specification, options = (
            records.get("generation_specification"),
            records.get("generation_options"),
        )
        if any(value is not None for value in (part, source, specification, options)):
            if (
                type(part) is not MeshPart
                or source is None
                or specification is None
                or options is None
            ):
                raise ValueError(
                    "source_closure.generation_specification must retain its declared source type and fields; got Single native generation requires its whole MeshPart, source, hard request and options"
                )
            validate_registered_native_part(
                part, source, specification, options, limits=limits
            )
            if not source_content_equal(
                part.carrier.certification,
                records["report"],
                limits=limits,
                source_validation=source_validation,
            ):
                raise ValueError(
                    "source_closure.generation_specification must match its retained source binding; got Single generation part must retain the primary scientific report"
                )
            if not source_content_equal(
                part.carrier.certification.request,
                records["certification_inputs"],
                limits=limits,
                source_validation=source_validation,
            ):
                raise ValueError(
                    "source_closure.generation_specification must match its retained source binding; got Single generation part must bind the primary scientific request"
                )
            if not source_content_equal(
                part.carrier.associations,
                records["associations"],
                limits=limits,
                source_validation=source_validation,
            ):
                raise ValueError(
                    "source_closure.generation_specification must match its retained source binding; got Single generation part must retain every qualified association"
                )
        return
    if type(registration) is not _overset.OversetRegistration:
        raise TypeError(
            "source_closure.generation_specification must match its retained source binding; got registration must be the OversetRegistration descriptor"
        )
    if {"generation_part", "generation_source", "generation_specification"} & set(
        records
    ):
        raise ValueError(
            "source_closure.generation_specification must match its retained source binding; got Named registration and single-part generation roles cannot be mixed"
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
            "source_closure.generation_specification must retain its declared source type and fields; got Full registration requires generation sources and hard requests for every part"
        )
    if (
        not isinstance(options, Mapping)
        or set(options) != names
        or any(type(value) is not NativeMeshingOptions for value in options.values())
    ):
        raise ValueError(
            "source_closure.generation_specification must match its retained source binding; got Generation options must retain an native algorithm declaration for every part"
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


def validate_neurofluid_transport_role(records: Mapping[str, Any], /) -> None:
    "Bind the mixed-carrier owner to its scientific root."

    accepted = records.get("accepted_data")
    if not isinstance(accepted, Mapping) or set(accepted) != {"neurofluid_transport"}:
        raise ValueError(
            "source_closure.accepted_data.neurofluid_transport must retain its declared source type and fields; got Neurofluid transport requires its sole checkpoint role"
        )
    checkpoint = accepted["neurofluid_transport"]
    if type(checkpoint) is not NeurofluidTransportCheckpoint:
        raise TypeError(
            "source_closure.accepted_data.neurofluid_transport must retain its declared source type and fields; got Neurofluid transport requires the registered physical checkpoint"
        )
    rebuilt = replace(checkpoint)
    if not source_content_equal(checkpoint, rebuilt, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS):
        raise ValueError(
            "source_closure.accepted_data.neurofluid_transport must match its retained source binding; got Neurofluid checkpoint differs from its reconstruction"
        )
    generation = records.get("generation_part")
    if (
        type(generation) is not MeshPart
        or type(generation.carrier) is not _result.CellMeshingResult
    ):
        raise TypeError(
            "source_closure.accepted_data.neurofluid_transport must retain its declared source type and fields; got Neurofluid transport requires its generation part"
        )
    bindings = (
        (records.get("generation_source"), checkpoint.source),
        (generation.carrier, checkpoint.result),
        (records.get("certification_inputs"), checkpoint.result.certification.request),
        (records.get("report"), checkpoint.result.certification),
        (records.get("associations"), checkpoint.result.associations),
    )
    if any(
        not source_content_equal(actual, expected, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
        for actual, expected in bindings
    ):
        raise ValueError(
            "source_closure.accepted_data.neurofluid_transport must match its retained source binding; got Neurofluid checkpoint changes its source, carrier, or evidence bindings"
        )
    if "field_declarations" in records:
        raise ValueError(
            "source_closure.accepted_data.neurofluid_transport must match its retained source binding; got Mixed-carrier transport is owned by its physical checkpoint, not fictitious mesh-part fields"
        )


def _validate_meshing_accepted_roles(records: Mapping[str, Any], /) -> None:
    "Admit only authored physical arrays and accepted adaptation facts."
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
            "source_closure.accepted_data must match its declared source identity and lineage; got Accepted scientific data requires physical field or accepted epoch roles"
        )
    if "neurofluid_transport" in accepted:
        validate_neurofluid_transport_role(records)
        return
    if "accepted_epochs" in accepted:
        if set(accepted) != {"accepted_epochs"}:
            raise ValueError(
                "source_closure.accepted_data must match its declared source identity and lineage; got Accepted epochs own the sole ordered field and transition history"
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
                "source_closure.accepted_data must retain its declared numerical inventory; got Accepted fields must retain named numerical arrays, not nested containers"
            )
        if "field_declarations" not in records:
            raise ValueError(
                "source_closure.accepted_data must retain its declared source type and fields; got Accepted fields require their physical field declarations"
            )
    if "adaptation_results" in accepted:
        results = accepted["adaptation_results"]
        if (
            type(results) is not tuple
            or not results
            or any(type(value) is not MeshAdaptationResult for value in results)
        ):
            raise TypeError(
                "source_closure.accepted_data must retain its declared source type and fields; got Accepted adaptation_results must retain a nonempty tuple of MeshAdaptationResult owners"
            )


def _validate_serial_accepted_target(
    records: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None = None,
) -> None:
    "Bind one dense adapted target to its accepted transition event."
    target = records.get("accepted_target")
    if target is None:
        return

    if records.get("generation_part") is not None:
        raise ValueError(
            "source_closure.accepted_target must match its retained source binding; got An accepted adapted target must not be mislabeled as generation"
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
            "source_closure.accepted_target must match its retained source binding; got accepted_target must be one dense, audited, compliant, certified CellMeshingResult"
        )
    bindings = (
        (records.get("certification_inputs"), target.certification.request),
        (records.get("report"), target.certification),
        (records.get("associations"), target.associations),
    )
    if any(
        not source_content_equal(
            actual,
            expected,
            limits=limits,
            source_validation=source_validation,
        )
        for actual, expected in bindings
    ):
        raise ValueError(
            "source_closure.accepted_target must match its retained source binding; got Accepted target changes its certification inputs, report, or source associations"
        )
    accepted = records.get("accepted_data")
    if isinstance(accepted, Mapping) and "accepted_epochs" in accepted:
        validate_meshing_accepted_epochs(records)
        return
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
            "source_closure.accepted_target must retain its declared source type and fields; got Accepted target requires one declared adaptation event"
        )
    lineage = event["lineage"]
    values = {
        name: event[name] for name in required_event - {"lineage", "accepted_event_id"}
    }
    if not isinstance(lineage, MeshLineage) or any(
        type(value) is not str or not value for value in values.values()
    ):
        raise TypeError(
            "source_closure.accepted_target must match its declared source identity and lineage; got Accepted adaptation event identities and lineage must be explicit"
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
            "source_closure.accepted_target must match its declared source identity and lineage; got Accepted target lacks its source-to-target lineage and accepted-event identity"
        )
    if (
        target.certification.mesh_id != target.mesh.mesh_id
        or target.certification.topology_id != target.mesh.topology_id
        or target.certification.geometry_id != target.audit.geometry_id
    ):
        raise ValueError(
            "source_closure.accepted_target must match its retained source binding; got Accepted target certification changes its mesh, geometry, or audit binding"
        )


def _validate_collective_target(
    records: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits,
    source_validation: MeshingSourceValidation | None = None,
) -> None:
    "Join the retained source and every accepted local result."
    target = records.get("collective_target")
    accepted = records.get("accepted_data", {})
    results = accepted.get("adaptation_results", ())
    if target is None:
        if any(result.target.mesh.storage is not None for result in results):
            raise ValueError(
                "source_closure.collective_target must retain its declared source type and fields; got Accepted distributed results require their explicit collective_target role"
            )
        return
    if type(target) is not _result.CellMeshingResult or target.mesh.storage is None:
        raise TypeError(
            "source_closure.collective_target must retain its declared source type and fields; got collective_target must be an accepted owner-local CellMeshingResult"
        )
    original = _result.require_original_meshing_source(target)

    if isinstance(original, InitialCollectiveMeshEvidence):
        authored = {
            "generation_source": original.authored_source,
            "generation_specification": original.specification,
            "generation_schedule": original.schedule,
            "generation_layout": original.layout,
        }
        for name, expected in authored.items():
            if name not in records or not source_content_equal(
                records[name],
                expected,
                limits=limits,
                source_validation=source_validation,
            ):
                raise ValueError(
                    f"source_closure.collective_target must match its retained source binding; got Initial archive must retain its authored {name} inputs."
                )
    else:
        certification = original.certification
        if certification is None:
            raise ValueError(
                "source_closure.collective_target must match its retained source binding; got Collective scientific source lacks its full evidence"
            )
        if not source_content_equal(
            certification,
            records["report"],
            limits=limits,
            source_validation=source_validation,
        ):
            raise ValueError(
                "source_closure.collective_target must match its retained source binding; got Collective archive must preserve the source evidence"
            )
        if not source_content_equal(
            original.associations,
            records["associations"],
            limits=limits,
            source_validation=source_validation,
        ):
            raise ValueError(
                "source_closure.collective_target must match its retained source binding; got Collective archive must preserve all qualified source associations"
            )

    current = target
    while current.collective_evidence is not None:
        evidence = current.collective_evidence
        if source_validation is not None:
            source_validation.require_epoch(evidence)
        else:
            match evidence:
                case InitialCollectiveMeshEvidence():
                    evidence.require_current()
                case _:
                    storage = current.mesh.storage
                    if storage is None:
                        raise ValueError(
                            "source_closure.collective_target must match its retained source binding; got Collective archive predecessor lacks its physical numerical storage"
                        )
                    validate_partitioned_mesh_evidence(storage, evidence)
        if isinstance(evidence, InitialCollectiveMeshEvidence):
            break
        current = evidence.source
    epoch = target.collective_evidence
    if epoch is None:
        raise ValueError(
            "source_closure.collective_target must match its retained source binding; got Collective target lacks its consumed numerical evidence"
        )
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
                "source_closure.collective_target must match its declared source identity and lineage; got Accepted results must consume the same request, source and collective epoch"
            )
        outcome = _adaptation.RouteOutcome(
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
        if not source_content_equal(
            result, fresh, limits=limits, source_validation=source_validation
        ):
            raise ValueError(
                "source_closure.collective_target must match its retained source binding; got Accepted adaptation result differs from its reconstruction"
            )


@checked
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
    "Validate one checkpoint closure or one source record."
    if source_validation is not None and source_logical_arrays is not None:
        if source_logical_content_digest is None:
            raise ValueError(
                "source_closure.roles must match its declared source identity and lineage; got Shared source validation requires its admitted content identity"
            )
        source_validation.require_binding(
            source_logical_arrays, source_logical_content_digest
        )
    if source_validation is None and source_logical_arrays is not None:
        if source_logical_content_digest is None or not isinstance(value, Mapping):
            raise ValueError(
                "source_closure.roles must retain its declared numerical inventory; got Admitted shared source banks require their content and closure"
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
            "source_closure.roles must match its retained source binding; got An accepted adapted target must not be mislabeled as generation"
        )
    if isinstance(value, MeshingSourceClosure):
        value = value.to_records()
    _validate_source_value(value, limits=limits, source_validation=source_validation)
    if not isinstance(value, Mapping):
        return

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
                "source_closure.roles must retain its declared source type and fields; got Initial source closure requires its authored generation inputs, not a fabricated report"
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
            "source_closure.roles must retain its declared source type and fields; got Source closure requires its certification roots and declared source/registration/accepted-data roles"
        )
    inputs = value["certification_inputs"]
    report = value["report"]
    associations = value["associations"]
    if type(associations) is not tuple or any(
        type(item) is not GeometryAssociation for item in associations
    ):
        raise TypeError(
            "source_closure.roles must match its retained source binding; got Source closure associations must be a tuple of GeometryAssociation records"
        )
    if (
        type(inputs) is not MeshCertificationInputs
        or type(report) is not MeshCertificationReport
    ):
        raise TypeError(
            "source_closure.roles must retain its declared source type and fields; got Source closure requires the request and report"
        )
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
        validate_association_binding(
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


@checked
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
            "source_closure.certification_inputs must retain its declared source type and fields; got Restored certification requires its request and report"
        )
    certification_inputs.validate_source_integrity()
    if not source_content_equal(
        source,
        certification_inputs.source,
        limits=limits,
        source_validation=source_validation,
    ):
        raise ValueError(
            "source_closure.certification_inputs must match its retained source binding; got Restored source must be scientifically identical to the retained source query"
        )
    if not source_content_equal(
        report.request,
        certification_inputs,
        limits=limits,
        source_validation=source_validation,
    ):
        raise ValueError(
            "source_closure.certification_inputs must match its retained source binding; got Restored report request must retain the scientific request, not only its ID"
        )
    if report.request.request_id != certification_inputs.request_id:
        raise ValueError(
            "source_closure.certification_inputs must match its retained source binding; got Restored report must bind the retained certification request"
        )
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
            "source_closure.certification_inputs must match its retained source binding; got Restored report must bind the retained mesh, geometry and schedule"
        )
    if len(report.scoped_fidelity) != len(certification_inputs.scoped_fidelity):
        raise ValueError(
            "source_closure.certification_inputs must match its retained source binding; got Restored report lost scoped source fidelity evidence"
        )
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
                "source_closure.certification_inputs must match its retained source binding; got Restored scoped evidence binds another stratum, trace, or tolerance"
            )
    if association is not None:
        validate_association_binding(certification_inputs, association)


@checked
def validate_association_binding(
    inputs: MeshCertificationInputs,
    association: GeometryAssociation,
    /,
    *,
    generation_source: NativeGenerationSource | None = None,
) -> None:

    if type(generation_source) is NativeLayerCoreSource:
        validate_layer_core_association(inputs, association, generation_source)
        return
    source_binding = (
        (inputs.source_id, inputs.source_revision)
        if generation_source is None
        else (generation_source.source_id, generation_source.source_revision)
    )
    if (association.source_id, association.source_revision) != source_binding:
        raise ValueError(
            "source_closure.validate_association_binding.stored_fields must match its declared source identity and lineage; got Restored association must bind the retained source revision"
        )
    if association.target_entity_set_id not in dict(inputs.entity_set_ids).values():
        raise ValueError(
            "source_closure.validate_association_binding.stored_fields must match its retained source binding; got Restored association must bind an certified mesh entity set"
        )


def _validate_mapped_layer_core_association(
    inputs: MeshCertificationInputs,
    association: GeometryAssociation,
    mapped_domain: MappedReferenceDomain,
    /,
    *,
    mesh: CellMesh | None,
) -> None:
    "Bind restored mapped tables to the declared root image namespace."
    if inputs.domain is None or inputs.domain.domain_id != mapped_domain.domain_id:
        raise ValueError(
            "source_closure.validate_mapped_layer_core_association.stored_fields must match its retained source binding; got Mapped layer-core association must retain its mapped domain"
        )
    if (
        association.association_kind is not GeometryAssociationKind.MAPPED_REFERENCE
        or (association.source_id, association.source_revision)
        != (mapped_domain.source_id, mapped_domain.source_revision)
        or not association.exact
        or not association.complete
    ):
        raise ValueError(
            "source_closure.validate_mapped_layer_core_association.stored_fields must match its declared source identity and lineage; got Mapped layer-core association must bind its source revision"
        )
    if association.target_entity_set_id not in dict(inputs.entity_set_ids).values():
        raise ValueError(
            "source_closure.validate_mapped_layer_core_association.stored_fields must match its retained source binding; got Mapped layer-core association must bind an certified entity set"
        )
    dimensions = np.asarray(association.parent_dimensions, dtype=np.int64)
    identifiers = np.asarray(association.parent_ids, dtype=np.int64)
    if np.any(dimensions < 0) or np.any(dimensions > 3):
        raise ValueError(
            "source_closure.validate_mapped_layer_core_association.stored_fields must retain its declared source type and fields; got Mapped layer-core association requires explicit bounded root strata"
        )
    for degree in range(4):
        authority = np.asarray(mapped_domain.reference_mesh.entity_set(degree).entity_ids)
        if np.any(~np.isin(identifiers[dimensions == degree], authority)):
            raise ValueError(
                "source_closure.validate_mapped_layer_core_association.stored_fields must match its retained source binding; got Mapped layer-core association exceeds its root namespace"
            )
    names = tuple(
        mapped_domain.image_entity_id(int(degree), int(identifier))
        for degree, identifier in zip(dimensions, identifiers, strict=True)
    )
    if association.source_entities.entity_ids() != names:
        raise ValueError(
            "source_closure.validate_mapped_layer_core_association.stored_fields must match its retained source binding; got Mapped layer-core association names differ from root image identities"
        )
    if mesh is not None:
        degree = next(
            degree
            for degree, identifier in inputs.entity_set_ids
            if identifier == association.target_entity_set_id
        )
        association.validate_target(mesh.entity_set(degree))


class _LayerAssociationStratum(StrictModule):
    dimensions: np.ndarray
    indices: np.ndarray


def _prepare_layer_association(
    inputs: MeshCertificationInputs, association: GeometryAssociation, /
) -> _LayerAssociationStratum:
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
            "source_closure.prepare_layer_association.stored_fields must match its retained source binding; got Layer-core association must bind an certified entity set"
        )
    if np.any(dimensions < 0) or np.any(dimensions > 3) or np.any(indices < 0):
        raise ValueError(
            "source_closure.prepare_layer_association.stored_fields must retain its declared source type and fields; got Layer-core association requires explicit bounded source strata"
        )
    if association.source_entities.roles() != tuple(
        roles[int(value)] for value in dimensions
    ):
        raise ValueError(
            "source_closure.prepare_layer_association.stored_fields must match its retained source binding; got Layer-core association source roles must match their authored dimensions"
        )
    expected_names = tuple(
        f"{association.source_revision}:{roles[int(dimension)].value}:{int(index)}"
        for dimension, index in zip(dimensions, indices, strict=True)
    )
    if association.source_entities.entity_ids() != expected_names:
        raise ValueError(
            "source_closure.prepare_layer_association.stored_fields must match its retained source binding; got Layer-core association entity names must bind source indices"
        )
    return _LayerAssociationStratum(dimensions, indices)


def _validate_composite_layer_stratum(
    inputs: MeshCertificationInputs,
    association: GeometryAssociation,
    source: NativeLayerCoreSource,
    stratum: _LayerAssociationStratum,
    /,
) -> None:
    dimensions, indices = stratum.dimensions, stratum.indices
    if association.target_entity_set_id != dict(inputs.entity_set_ids)[3] or np.any(
        dimensions != 3
    ):
        raise ValueError(
            "source_closure.validate_composite_layer_stratum.stored_fields must match its retained source binding; got Composite layer-core source association must own material cells"
        )
    if np.any(indices >= len(source.region_ids)):
        raise ValueError(
            "source_closure.validate_composite_layer_stratum.stored_fields must match its retained source binding; got Composite layer-core material association exceeds its namespace"
        )


def _validate_core_plc_stratum(
    inputs: MeshCertificationInputs,
    association: GeometryAssociation,
    source: NativeLayerCoreSource,
    stratum: _LayerAssociationStratum,
    /,
) -> None:
    dimensions, indices = stratum.dimensions, stratum.indices
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
        raise ValueError(
            "source_closure.validate_core_plc_stratum.stored_fields must match its retained source binding; got Core association exceeds its authored PLC strata"
        )


def _require_layer_cap_identity(
    source: NativeLayerCoreSource, dimensions: np.ndarray, indices: np.ndarray, /
) -> None:
    cap = source.layers.cap
    if cap is not None:
        cap_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in cap.blocks]
        )
        if np.any(~np.isin(cap_ids, indices[dimensions == 2])):
            raise ValueError(
                "source_closure.require_layer_cap_identity.stored_fields must match its declared source identity and lineage; got Layer association must retain every cap global facet identity"
            )


def _require_layer_target_ancestry(
    association: GeometryAssociation,
    layer_mesh: CellMesh,
    mesh: CellMesh,
    target_dimension: int,
    authorities: tuple[np.ndarray, ...],
    dimensions: np.ndarray,
    indices: np.ndarray,
    /,
) -> None:
    layer_vertex_ids = np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64)
    original_keys = (
        tuple((int(identifier),) for identifier in layer_vertex_ids)
        if target_dimension == 0
        else tuple(
            tuple(sorted(int(layer_vertex_ids[row]) for row in vertices))
            for vertices in row_entity_vertex_keys(layer_mesh, target_dimension)
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
                row_entity_vertex_keys(mesh, target_dimension),
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
            "source_closure.require_layer_target_ancestry.stored_fields must match its retained source binding; got Layer association target must bind the immutable layer/cap vertex ancestry"
        )


def _validate_layer_mesh_stratum(
    inputs: MeshCertificationInputs,
    association: GeometryAssociation,
    source: NativeLayerCoreSource,
    stratum: _LayerAssociationStratum,
    /,
    *,
    mesh: CellMesh | None,
) -> None:
    dimensions, indices = stratum.dimensions, stratum.indices
    layer_mesh = source.layers.mesh
    face_rows = row_entity_vertex_keys(layer_mesh, 2)
    work_upper = (
        sum(
            block.cell_count * len(reference_cell_topology(block.cell_kind).entities[2])
            for block in layer_mesh.blocks
        )
        + layer_mesh.coordinates.shape[0]
        + layer_mesh.entity_set(1).count
        + len(face_rows)
        + sum(max(0, len(row) - 2) for row in face_rows)
    )
    ticket = HostBatchTicket(work_upper)
    with ticket:
        tables = layer_source_tables(
            source.layers,
            source.layer_regions,
            source.region_ids,
            work=LayerCoreSourceWork(work_upper, ticket=ticket),
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
            "source_closure.validate_layer_mesh_stratum.stored_fields must match its retained source binding; got Layer association exceeds its immutable source strata"
        )
    target_dimension = next(
        dimension
        for dimension, identifier in inputs.entity_set_ids
        if identifier == association.target_entity_set_id
    )
    if target_dimension == 2:
        _require_layer_cap_identity(source, dimensions, indices)
    if mesh is not None:
        _require_layer_target_ancestry(
            association,
            layer_mesh,
            mesh,
            target_dimension,
            authorities,
            dimensions,
            indices,
        )


@checked
def validate_layer_core_association(
    inputs: MeshCertificationInputs,
    association: GeometryAssociation,
    source: NativeLayerCoreSource,
    /,
    *,
    mesh: CellMesh | None = None,
) -> None:
    """Bind a composed association to its declared material, PLC or layer stratum."""
    if source.mapped_domain is not None:
        _validate_mapped_layer_core_association(
            inputs, association, source.mapped_domain, mesh=mesh
        )
        return
    stratum = _prepare_layer_association(inputs, association)
    binding = (association.source_id, association.source_revision)
    match binding:
        case _ if binding == (source.source_id, source.source_revision):
            _validate_composite_layer_stratum(inputs, association, source, stratum)
        case _ if binding == (source.complex.complex_id, source.complex.complex_id):
            _validate_core_plc_stratum(inputs, association, source, stratum)
        case _ if binding == (source.layers.result_id, source.layers.result_id):
            _validate_layer_mesh_stratum(inputs, association, source, stratum, mesh=mesh)
        case _:
            raise ValueError(
                f"GeometryAssociation.source_revision must bind a retained layer/core constituent; got {binding}"
            )


@checked
def recertify_restored_meshing_source(
    inputs: MeshCertificationInputs,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    audit: CellMeshAuditReport,
    /,
    *,
    archive_limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> MeshCertificationReport:
    "Reestablish all native source evidence on restored mesh coordinates."

    _validate_source_value(inputs, limits=archive_limits)
    inputs.validate_source_integrity()
    scope = _source_owner_reconstruction_scope.get()
    key = (
        _source_owner_reconstruction_key(
            (inputs, mesh, geometry, audit),
            archive_limits,
            None,
            certify_meshing_acceptance,
            scope,
        )
        if scope is not None and scope.process_id == getpid()
        else None
    )
    if key is not None and scope is not None and key in scope.certifications:
        return scope.certifications[key]
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
            "source_closure.recertify_restored_meshing_source.stored_fields must match its retained source binding; got Restored request does not bind the mesh, geometry and native source"
        )
    fresh.require_passed()
    if key is not None and scope is not None:
        scope.constructors[id(certify_meshing_acceptance)] = certify_meshing_acceptance
        scope.certifications[key] = fresh
    return fresh


@dataclass(frozen=True, slots=True)
class _RetainedLayerCoreValidation:
    "Content-bound evidence that the live producer validated one lineage."

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
            member.name: content[member.name]
            for member in fields(cls)
            if member.name != "evidence_id"
        }
        for member in fields(cls):
            name = member.name
            if name == "evidence_id":
                continue
            value = normalized[name]
            field_name = f"retained_layer_core_validation.{name}"
            if name in ("adaptation_ids", "adaptation_execution_ids"):
                normalized[name] = tuple(
                    canonical_identifier(item, field_name) for item in value
                )
            elif name != "mapped_domain_id" or value is not None:
                normalized[name] = canonical_identifier(value, field_name)
        return cls(
            **normalized,
            evidence_id=canonical_fingerprint(
                {"kind": "retained-layer-core-validation", **normalized}
            ),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> "_RetainedLayerCoreValidation":
        fields_ = {"kind"} | {member.name for member in fields(cls)}
        if set(record) != fields_ or record["kind"] != "retained-layer-core-validation":
            raise ArrayArchiveCorruptionError(
                "source_closure.from_record.stored_fields must match its retained source binding; got Retained layer/core validation record is invalid"
            )
        try:
            evidence = cls.from_content(record)
        except (KeyError, TypeError, ValueError) as error:
            raise ArrayArchiveCorruptionError(
                "source_closure.from_record.stored_fields must match its retained source binding; got Retained layer/core validation identities are invalid"
            ) from error
        if record["evidence_id"] != evidence.evidence_id:
            raise ArrayArchiveCorruptionError(
                "source_closure.from_record.stored_fields must match its declared source identity and lineage; got Retained layer/core validation identity changed"
            )
        return evidence

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": "retained-layer-core-validation",
            **{
                member.name: list(value)
                if member.name in ("adaptation_ids", "adaptation_execution_ids")
                else value
                for member in fields(self)
                for value in (object.__getattribute__(self, member.name),)
            },
        }


def _layer_core_adaptation_lineage(
    accepted: Any, result: _result.CellMeshingResult, /
) -> tuple[tuple[_adaptation.MeshAdaptationResult, ...], list[str]]:
    """Validate the ordered refine/inverse-coarsen lineage and its execution."""
    if not isinstance(accepted, Mapping) or set(accepted) != {"adaptation_results"}:
        raise ValueError(
            "source_closure.retained_validation must match its declared source identity and lineage; got Retained layer/core validation requires one adaptation lineage"
        )
    adaptations = accepted["adaptation_results"]
    if type(adaptations) is not tuple or len(adaptations) != 2:
        raise ValueError(
            "source_closure.retained_validation must retain its declared source type and fields; got Retained layer/core validation requires refine and inverse-coarsen results"
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
                "source_closure.retained_validation must match its declared source identity and lineage; got Retained layer/core adaptation lineage is incomplete or discontinuous"
            )
        adaptation_evidence = adaptation.evidence
        if (
            type(adaptation_evidence) is not _mixed_adaptation.MixedAdaptationEvidence
            or adaptation_evidence.execution_evidence is None
        ):
            raise ValueError(
                "source_closure.retained_validation must match its retained source binding; got Retained layer/core adaptation lacks its mixed execution evidence"
            )
        adaptation_evidence.execution_evidence.require_valid()
        if adaptation_evidence.execution_evidence.owner_id != adaptation.prepared_id:
            raise ValueError(
                "source_closure.retained_validation must match its retained source binding; got Retained layer/core adaptation execution binds another preparation"
            )
        execution_ids.append(
            canonical_fingerprint(adaptation_evidence.execution_evidence.to_record())
        )
        prior = adaptation.target
    if adaptations[1].coarsening_witnesses is None:
        raise ValueError(
            "source_closure.retained_validation must match its declared source identity and lineage; got Retained layer/core lineage lacks inverse-coarsening evidence"
        )
    return adaptations, execution_ids


def _layer_core_validation_content(
    records: Any,
    plan_id: str,
    /,
    *,
    live_plan: NativeMeshingPlan | None = None,
) -> dict[str, Any]:

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
            "source_closure.retained_validation must retain its declared source type and fields; got Retained layer/core validation requires its lifecycle roles"
        )
    source = records["generation_source"]
    specification = records["generation_specification"]
    options = records["generation_options"]
    part = records["generation_part"]
    if (
        type(source) is not NativeLayerCoreSource
        or type(options) is not NativeMeshingOptions
        or type(part) is not MeshPart
        or type(part.carrier) is not _result.CellMeshingResult
    ):
        raise TypeError(
            "source_closure.retained_validation must retain its declared source type and fields; got Retained layer/core validation requires its source, options, and carrier"
        )
    result = part.carrier
    report = result.certification
    execution = result.execution_evidence
    if report is None or execution is None:
        raise ValueError(
            "source_closure.retained_validation must retain its declared source type and fields; got Retained layer/core validation requires passed certification and execution"
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
            "source_closure.retained_validation must retain its declared source type and fields; got Retained layer/core validation changed its source evidence or execution owner"
        )
    provenance = json.loads(result.provenance.content_json)
    if (
        not isinstance(provenance, dict)
        or provenance.get("kind") != "mapping"
        or dict(provenance.get("items", ())).get("plan") != plan_id
    ):
        raise ValueError(
            "source_closure.retained_validation must match its retained source binding; got Retained layer/core result binds another generation plan"
        )
    if live_plan is not None:
        live_source = live_plan.source
        if type(live_source) is not NativeLayerCoreSource or not isinstance(
            live_source, NativeLayerCoreSource
        ):
            raise ValueError(
                "source_closure.retained_validation must match its retained source binding; got Retained layer/core plan does not own a layer/core source"
            )
        if (
            live_plan.plan_id != plan_id
            or live_source.binding_id != source.binding_id
            or live_plan.specification.specification_id != specification.specification_id
            or live_plan.options.options_id != options.options_id
            or live_plan.coordinate_contract.spatial_id
            != result.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "source_closure.retained_validation must match its retained source binding; got Retained layer/core plan is not the live generation preparation"
            )
    adaptations, execution_ids = _layer_core_adaptation_lineage(
        records["accepted_data"], result
    )
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
        raise TypeError(
            "source_closure.retain_layer_core_validation.stored_fields must match its retained source binding; got retained_layer_core_plan must be a NativeMeshingPlan"
        )
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
            "source_closure.require_retained_layer_core_validation.stored_fields must match its declared source identity and lineage; got Restored layer/core lineage differs from its retained validation evidence"
        )


@dataclass(frozen=True, slots=True)
class MeshingSourceClosureReceipt:
    "Durable publication identity of one native scientific closure."

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
                f"source_closure.source_banks.stored_fields must retain its declared numerical inventory; got Source array {entry.path} exceeds the archive member bounds."
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
    "Bind each logical array to one bank extent before any payload read."
    if not isinstance(archived, dict):
        raise ArrayArchiveCorruptionError(
            "source_closure.arrays must retain its declared numerical inventory; got Meshing source bank inventory is invalid"
        )
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
            raise ArrayArchiveCorruptionError(
                "source_closure.bank_size.stored_fields must retain its declared numerical inventory; got Meshing source bank inventory is invalid"
            )
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
                    "source_closure.arrays must retain its declared numerical inventory; got Meshing source bank extents are inconsistent"
                )
            index, used = index + 1, 0
            if size > bank_size(entry.dtype, index):
                raise ArrayArchiveCorruptionError(
                    "source_closure.arrays must retain its declared numerical inventory; got Meshing source bank extents are inconsistent"
                )
        placement[entry.name] = (_source_bank_name(entry.dtype, index), used)
        cursors[entry.dtype] = (index, used + size)
    if any(used != bank_size(dtype, index) for dtype, (index, used) in cursors.items()):
        raise ArrayArchiveCorruptionError(
            "source_closure.arrays must retain its declared numerical inventory; got Meshing source bank extents are inconsistent"
        )
    if set(archived) != set(expected):
        raise ArrayArchiveCorruptionError(
            "source_closure.arrays must retain its declared numerical inventory; got Meshing source archive contains undeclared banks"
        )
    return placement, expected


@meshing_source_owner_reconstruction_scope()
@checked
def write_meshing_source_closure(
    path: str | PathLike[str],
    records: Any,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    retained_layer_core_plan: NativeMeshingPlan | None = None,
) -> MeshingSourceClosureReceipt:
    """Publish the node-table recipe and every authored numerical field."""

    if isinstance(records, MeshingSourceClosure):
        records = records.to_records()
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
            "source_closure.write_meshing_source_closure.stored_fields must retain its declared numerical inventory; got Host source closure publication requires fully addressable JAX arrays; use distributed meshing checkpoint"
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
    "Authenticate the declared recipe and every numerical bank."

    register_meshing_source_artifacts()
    admitted: dict[str, Any] = {}

    def admit(manifest: Mapping[str, Any], /) -> dict[str, tuple[tuple[int, ...], str]]:
        if (
            set(manifest) != _SOURCE_MANIFEST_FIELDS
            or manifest["kind"] != "meshing-source-closure"
        ):
            raise ArrayArchiveCorruptionError(
                "source_closure.admit.stored_fields must match its retained source binding; got Meshing source closure manifest is invalid"
            )
        if manifest["content_id"] != expected_content_id:
            raise ArrayArchiveCorruptionError(
                "source_closure.admit.stored_fields must match its declared source identity and lineage; got Meshing source closure scientific content identity changed"
            )
        retained_record = manifest["retained_validation"]
        if retained_record is not None and not isinstance(retained_record, Mapping):
            raise ArrayArchiveCorruptionError(
                "source_closure.admit.stored_fields must match its retained source binding; got Meshing source retained validation record is invalid"
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
                "source_closure.admit.stored_fields must match its retained source binding; got Meshing source closure recipe is invalid"
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
            "source_closure.manifest must match its declared source identity and lineage; got Meshing source closure scientific content identity changed"
        )
    return recipe, arrays, admitted["retained_validation"]


type _MeshingSourceArchive = tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    _RetainedLayerCoreValidation | None,
]


@meshing_source_owner_reconstruction_scope()
def _restore_meshing_source_archive(
    archive: _MeshingSourceArchive,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> Any:

    recipe, arrays, retained_validation = archive
    restored = model_from_logical_array_recipe(
        recipe, arrays, prefix=_SOURCE_PREFIX, limits=limits
    )
    if retained_validation is None:
        validate_meshing_source_closure(restored, limits=limits)
    else:
        _require_retained_layer_core_validation(restored, retained_validation)
    return (
        MeshingSourceClosure.from_records(restored)
        if isinstance(restored, Mapping)
        else restored
    )


@checked
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
    "Authenticated controls only; not an accepted scientific source."

    execution_evidence: NativeExecutionRecord
    limits: _contracts.MeshingLimits
    source_id: str
    source_revision: str
    specification_id: str
    source_binding_id: str
    content_id: str


class _ExecutionControlSelection(StrictModule):
    table: RecipeTable
    catalog: tuple[tuple[ModelRecipeArray, int], ...]
    receipt_index: int
    limit_index: int
    selected: dict[str, int]
    implicit_source_index: int | None


def _select_execution_controls(
    recipe: Mapping[str, Any], limits: ArrayArchiveLimits, /
) -> _ExecutionControlSelection:
    table, catalog = model_array_catalog(recipe, prefix=_SOURCE_PREFIX, limits=limits)
    nodes = table.nodes
    root = nodes[-1]
    try:
        parse(root["kind"], _SourceMappingKind, "source_closure.controls.mapping_kind")
    except (TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            f"source_closure.controls.mapping_kind must retain a role namespace; got {root['kind']!r}"
        ) from error

    def role(name: str) -> int:
        for key, value in root["items"]:
            key_node = nodes[key]
            if key_node["kind"] == "literal" and key_node["value"] == name:
                return value
        raise ArrayArchiveCorruptionError(
            f"source_closure.role.stored_fields must match its retained source binding; got Source closure lacks execution role {name!r}."
        )

    def fields_at(index: int, expected_type: type | tuple[type, ...]) -> dict[str, int]:
        node = nodes[index]
        owner = artifact_value(node["type"]) if node["kind"] == "dataclass" else None
        if owner not in (
            expected_type if isinstance(expected_type, tuple) else (expected_type,)
        ):
            raise ArrayArchiveCorruptionError(
                "source_closure.fields_at.stored_fields must retain its declared source type and fields; got Execution controls bind an incorrect registered owner type"
            )
        # Dataclass recipes list child nodes in field-declaration order.
        names = tuple(member.name for member in fields(owner))
        if len(node["items"]) != len(names):
            raise ArrayArchiveCorruptionError(
                f"source_closure.{owner.__name__}.fields must have {len(names)} entries; "
                f"got {len(node['items'])}."
            )
        return dict(zip(names, node["items"], strict=True))

    specification_fields = fields_at(
        role("generation_specification"),
        (
            _contracts.CurveMeshingSpec,
            _contracts.SurfaceMeshingSpec,
            _contracts.VolumeMeshingSpec,
        ),
    )
    part_fields = fields_at(role("generation_part"), MeshPart)
    carrier_fields = fields_at(part_fields["carrier"], _result.CellMeshingResult)
    receipt_index = carrier_fields["execution_evidence"]
    fields_at(receipt_index, NativeExecutionRecord)
    limit_index = specification_fields["limits"]
    fields_at(limit_index, _contracts.MeshingLimits)
    scope_field = (
        "boundary_scope" if "boundary_scope" in specification_fields else "scope"
    )
    scope_fields = fields_at(specification_fields[scope_field], MeshingScope)
    source_index = role("generation_source")
    source_fields = fields_at(source_index, native_generation_source_types())
    implicit_source_index = (
        source_index
        if artifact_value(nodes[source_index]["type"]) is NativeImplicitSource
        else None
    )
    required = (
        {"source_id", "source_revision", "geometry", "grid"}
        if implicit_source_index is not None
        else {"source_id", "source_revision", "binding_id"}
    )
    if not required <= source_fields.keys():
        raise ArrayArchiveCorruptionError(
            "source_closure.controls must retain its source identity fields; got missing source fields"
        )
    compliance_fields = fields_at(
        carrier_fields["compliance"], _result.MeshingComplianceReport
    )
    input_fields = fields_at(role("certification_inputs"), MeshCertificationInputs)
    trace_fields = fields_at(carrier_fields["trace"], _trace.MeshingTrace)
    trace_binding = fields_at(trace_fields["binding"], _trace.MeshingEvidenceBinding)
    provenance_fields = fields_at(carrier_fields["provenance"], SemanticProvenance)
    selected = {
        "source_id": source_fields["source_id"],
        "source_revision": source_fields["source_revision"],
        "specification_id": specification_fields["specification_id"],
        "published_specification_id": compliance_fields["specification_id"],
        "scope_source_id": scope_fields["source_id"],
        "scope_source_revision": scope_fields["source_revision"],
        "trace_source_id": trace_binding["source_id"],
        "trace_source_revision": trace_binding["source_revision"],
        "provenance_json": provenance_fields["content_json"],
    }
    # A domain-only request has no fidelity-query identity. Its native trace
    # still binds the generation source; present query identities must agree.
    if not all(
        nodes[input_fields[name]] == {"kind": "literal", "value": None}
        for name in ("source", "source_id", "source_revision")
    ):
        selected["certified_source_id"] = input_fields["source_id"]
        selected["certified_source_revision"] = input_fields["source_revision"]
    if implicit_source_index is None:
        selected["source_binding_id"] = source_fields["binding_id"]
    return _ExecutionControlSelection(
        table, catalog, receipt_index, limit_index, selected, implicit_source_index
    )


class _RestoredExecutionControls(StrictModule):
    record: NativeExecutionRecord
    limits: _contracts.MeshingLimits
    identity: dict[str, str]


def _implicit_source_control_fingerprint(
    selection: _ExecutionControlSelection,
    arrays: Mapping[str, np.ndarray],
    /,
) -> str:
    """Fingerprint retained source data without reconstructing scientific owners."""
    source_index = selection.implicit_source_index
    if source_index is None:
        raise RuntimeError(
            "source_closure.implicit_controls must select an implicit source; got no source node"
        )
    nodes = selection.table.nodes
    entries = {index: entry for entry, index in selection.catalog}
    reached: set[int] = set()
    pending = [source_index]
    while pending:
        index = pending.pop()
        if index not in reached:
            reached.add(index)
            pending.extend(table_children(nodes[index]))
    digests: dict[int, str] = {}
    for index in sorted(reached):
        node = nodes[index]
        children = table_children(node)
        if children:
            payload = _with_children(node, [digests[child] for child in children])
        elif node["kind"] == "array":
            entry = entries[index]
            payload = {
                "kind": "array",
                "shape": entry.shape,
                "dtype": entry.dtype,
                "backend": entry.backend,
                "content": array_tree_fingerprint(arrays[entry.name]),
            }
        else:
            payload = node
        digests[index] = canonical_fingerprint(payload)
    return digests[source_index]


def _restore_execution_controls(
    selection: _ExecutionControlSelection, arrays: Mapping[str, np.ndarray], /
) -> _RestoredExecutionControls:
    nodes, catalog = selection.table.nodes, selection.catalog
    receipt_index, limit_index, selected = (
        selection.receipt_index,
        selection.limit_index,
        selection.selected,
    )
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
                NativeExecutionDeviceWorkCounters,
                NativeExecutionDeviceMemoryCounters,
                _contracts.MeshingLimits,
            ):
                raise ArrayArchiveCorruptionError(
                    "source_closure.controls must retain its declared source type and fields; got Execution-control decoding cannot construct scientific source owners"
                )
        elif node["kind"] not in ("literal", "nonfinite_float", "array"):
            raise ArrayArchiveCorruptionError(
                "source_closure.controls must match its retained source binding; got Execution controls contain an undeclared control-node kind"
            )
        reached.add(index)
        pending.extend(table_children(node))
    entries = {index: entry for entry, index in catalog}

    def restore_array(index: int, _node: Mapping[str, Any]) -> jax.Array:
        entry = entries[index]
        if entry.backend != "jax":
            raise ArrayArchiveCorruptionError(
                "source_closure.restore_array.stored_fields must retain its declared source type and fields; got Native execution controls require their JAX numerical leaves"
            )
        return jax.numpy.asarray(arrays[entry.name])

    restored = restore_nodes(nodes, restore_array, sorted(reached))
    record = restored[receipt_index]
    original_limits = restored[limit_index]
    if (
        type(record) is not NativeExecutionRecord
        or type(original_limits) is not _contracts.MeshingLimits
    ):
        raise ArrayArchiveCorruptionError(
            "source_closure.controls must retain its declared source type and fields; got Execution controls lost their registered numerical owners"
        )
    record.require_valid()
    counts = {
        member.name: object.__getattribute__(original_limits, member.name)
        for member in fields(original_limits)
        if member.name != "limits_id"
    }
    validated_limits = _contracts.MeshingLimits(**counts)
    if validated_limits.limits_id != original_limits.limits_id:
        raise ArrayArchiveCorruptionError(
            "source_closure.controls must match its declared source identity and lineage; got execution limit identity changed"
        )
    identity: dict[str, str] = {}
    for name, index in selected.items():
        value = restored[index]
        if type(value) is not str or not value:
            raise ArrayArchiveCorruptionError(
                f"source_closure.controls.{name} must be a nonempty string; got {type(value).__name__}"
            )
        identity[name] = value
    implicit = selection.implicit_source_index is not None
    if implicit:
        identity["source_binding_id"] = _implicit_source_control_fingerprint(
            selection, arrays
        )
    return _RestoredExecutionControls(record, original_limits, identity)


def _require_execution_control_binding(controls: _RestoredExecutionControls, /) -> None:
    record, identity = controls.record, controls.identity
    if (
        identity["source_id"] != identity["scope_source_id"]
        or identity["source_revision"] != identity["scope_source_revision"]
        or identity["source_id"] != identity["trace_source_id"]
        or identity["source_revision"] != identity["trace_source_revision"]
        or (
            "certified_source_id" in identity
            and (
                identity["source_id"] != identity["certified_source_id"]
                or identity["source_revision"] != identity["certified_source_revision"]
            )
        )
    ):
        raise ArrayArchiveCorruptionError(
            "source_closure.controls must match its declared source identity and lineage; got Execution controls disagree on source/specification identity"
        )
    provenance = json.loads(identity["provenance_json"])
    if (
        not isinstance(provenance, dict)
        or provenance.get("kind") != "mapping"
        or not isinstance(provenance.get("items"), list)
    ):
        raise ArrayArchiveCorruptionError(
            "source_closure.controls must match its retained source binding; got Historical execution records has noncanonical semantic provenance"
        )
    try:
        provenance_items = dict(provenance["items"])
    except (TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            "source_closure.controls must match its retained source binding; got Historical execution records has malformed semantic provenance"
        ) from error
    if (
        provenance_items.get("plan") != record.owner_id
        or identity["published_specification_id"] != identity["specification_id"]
        or provenance_items.get("specification") != identity["specification_id"]
    ):
        raise ArrayArchiveCorruptionError(
            "source_closure.controls must match its retained source binding; got Historical execution records binds another plan/specification"
        )


@checked
def read_meshing_source_execution_controls(
    path: str | PathLike[str],
    /,
    *,
    expected_content_id: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    _authenticated_archive: _MeshingSourceArchive | None = None,
) -> MeshingSourceExecutionControls:
    """Decode authenticated execution controls before constructing source owners."""
    recipe, arrays, _ = (
        _read_meshing_source_archive(
            path,
            expected_content_id=expected_content_id,
            limits=limits,
        )
        if _authenticated_archive is None
        else _authenticated_archive
    )
    selection = _select_execution_controls(recipe, limits)
    controls = _restore_execution_controls(selection, arrays)
    _require_execution_control_binding(controls)
    identity = controls.identity
    return MeshingSourceExecutionControls(
        controls.record,
        controls.limits,
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
    "Join archived owners to the declared computational checkpoint."
    target = closure.get("collective_target")
    if target is None:
        inputs = closure["certification_inputs"]
        report = closure["report"]
        if (
            type(inputs) is not MeshCertificationInputs
            or type(report) is not MeshCertificationReport
        ):
            raise TypeError(
                "source_closure.validate_meshing_source_checkpoint_references.stored_fields must retain its declared source type and fields; got Source closure requires inputs and report"
            )
        if (
            inputs.topology_id != references["topology_id"]
            or inputs.geometry_id != references["geometry_id"]
            or (
                inputs.source_revision is not None
                and inputs.source_revision != references["source_revision_id"]
            )
        ):
            raise ValueError(
                "source_closure.validate_meshing_source_checkpoint_references.stored_fields must match its retained source binding; got Source closure does not bind the accepted checkpoint records"
            )
        return
    if (
        type(target) is not _result.CellMeshingResult
        or target.collective_evidence is None
    ):
        raise TypeError(
            "source_closure.validate_meshing_source_checkpoint_references.stored_fields must retain its declared source type and fields; got A collective checkpoint requires its accepted computational target"
        )
    evidence = target.collective_evidence

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
                "source_closure.validate_meshing_source_checkpoint_references.stored_fields must retain its declared source type and fields; got A dense scientific source requires its evidence"
            )
        revision = certification.request.source_revision
    if revision is not None:
        expected["source_revision_id"] = revision
    match evidence:
        case _result.CollectiveMeshEvidence():
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
        case InitialCollectiveMeshEvidence():
            expected.update(
                {
                    "source_topology_id": None,
                    "source_geometry_id": None,
                    "lineage_id": None,
                    "transfer_id": None,
                    "request_id": evidence.specification.specification_id,
                }
            )
        case _:
            pass
    if any(references[name] != identity for name, identity in expected.items()):
        raise ValueError(
            "source_closure.validate_meshing_source_checkpoint_references.stored_fields must match its retained source binding; got Collective source records differ from the declared accepted computational checkpoint"
        )


class _SourceArrayAliasCollection(StrictModule):
    arrays: dict[str, jax.Array]
    identities: dict[int, str]
    visited: set[int]


def _bind_source_array_alias(
    collection: _SourceArrayAliasCollection, name: str, value: Any, /
) -> None:
    if not isinstance(value, jax.Array) or value.is_fully_addressable:
        return
    if id(value) in collection.identities:
        return
    previous = collection.arrays.get(name)
    if previous is not None and previous is not value:
        raise ValueError(
            f"source_closure.arrays.{name} must share its producer buffer; got independently allocated buffers"
        )
    collection.identities[id(value)] = name
    collection.arrays[name] = value


def _collect_publication_aliases(
    value: PublicationProjection, prefix: str, collection: _SourceArrayAliasCollection, /
) -> None:
    for name, array in value.source_arrays:
        _bind_source_array_alias(collection, f"{prefix}/bank/{name}", array)
    for name, array in value.projected_arrays:
        _bind_source_array_alias(collection, f"{prefix}/publication/{name}", array)


def _collect_collective_array_aliases(
    value: _result.CollectiveMeshEvidence | InitialCollectiveMeshEvidence,
    path: str,
    collection: _SourceArrayAliasCollection,
    /,
) -> None:
    prefix = f"collective/{value.evidence_id}"
    if isinstance(value, _result.CollectiveMeshEvidence):
        _visit_source_array_aliases(value.source, f"{path}.source", collection)
    for name, array in value.logical_arrays:
        _bind_source_array_alias(collection, f"{prefix}/bank/{name}", array)
    _bind_source_array_alias(
        collection, f"{prefix}/partition_checks", value.partition_checks
    )
    _bind_source_array_alias(collection, f"{prefix}/global_checks", value.global_checks)
    if isinstance(value, _result.CollectiveMeshEvidence):
        _bind_source_array_alias(
            collection, f"{prefix}/source_exterior", value.source_exterior
        )
        for role, states in (
            ("initial", value.initial_states),
            ("compiled", value.compiled_states),
        ):
            for key, array in jax.tree_util.tree_flatten_with_path(states)[0]:
                _bind_source_array_alias(
                    collection, f"{prefix}/{role}/{jax.tree_util.keystr(key)}", array
                )
    for degree, (keys, identifiers, owners) in enumerate(
        zip(value.entity_keys, value.entity_ids, value.entity_owners, strict=True)
    ):
        _bind_source_array_alias(collection, f"{prefix}/entity/{degree}/keys", keys)
        _bind_source_array_alias(collection, f"{prefix}/entity/{degree}/ids", identifiers)
        _bind_source_array_alias(collection, f"{prefix}/entity/{degree}/owners", owners)


def _collect_owner_array_aliases(
    value: Any, path: str, collection: _SourceArrayAliasCollection, /
) -> None:
    match value:
        case _result.CollectiveMeshEvidence() | InitialCollectiveMeshEvidence():
            _collect_collective_array_aliases(value, path, collection)
        case FiniteElementTopologyTransfer():
            for name in (
                "source_coordinates",
                "target_coordinates",
                "source_measures",
                "target_measures",
            ):
                _bind_source_array_alias(
                    collection,
                    f"transfer/{value.transfer_id}/claims/{name}",
                    object.__getattribute__(value, name),
                )
        case _restart_distribution.SimplexRestartRepack():
            for key, array in jax.tree_util.tree_flatten_with_path(value)[0]:
                _bind_source_array_alias(
                    collection,
                    f"restart/{path}/{jax.tree_util.keystr(key)}",
                    array,
                )
        case _cell_mesh.CellMeshStorage():
            prefix = f"collective/{value.evidence_id}"
            for name, array in value.logical_arrays:
                _bind_source_array_alias(collection, f"{prefix}/bank/{name}", array)
            if value.geometry_projection is not None:
                for name, array in value.geometry_projection.projected_arrays:
                    _bind_source_array_alias(
                        collection, f"{prefix}/geometry_projection/{name}", array
                    )
        case _scope.MeshScopeProjection():
            prefix = f"collective/{value.evidence_id}"
            _collect_publication_aliases(value.publication, prefix, collection)
            _bind_source_array_alias(
                collection,
                f"{prefix}/scope/{value.membership_name}/universe",
                value.global_ids,
            )
            _bind_source_array_alias(
                collection,
                f"{prefix}/scope/{value.membership_name}/members",
                value.members,
            )
        case organization_records_owner.MeshAttribute():
            projection = value.scope.projection
            if projection is not None:
                _bind_source_array_alias(
                    collection,
                    f"collective/{projection.evidence_id}/scope/{projection.membership_name}/values",
                    value.global_values,
                )
        case _:
            pass


def _visit_source_array_aliases(
    value: Any, path: str, collection: _SourceArrayAliasCollection, /
) -> None:
    if isinstance(value, (jax.Array, np.ndarray)) or id(value) in collection.visited:
        return
    collection.visited.add(id(value))
    _collect_owner_array_aliases(value, path, collection)
    if is_dataclass(value):
        for member in fields(value):
            _visit_source_array_aliases(
                object.__getattribute__(value, member.name),
                f"{path}.{member.name}",
                collection,
            )
        return
    match value:
        case tuple():
            for index, child in enumerate(value):
                _visit_source_array_aliases(child, f"{path}[{index}]", collection)
        case Mapping():
            for name, child in value.items():
                _visit_source_array_aliases(child, f"{path}.{name}", collection)
        case _:
            return


@checked
def meshing_source_shared_array_aliases(
    closure: MeshingSourceClosure, /
) -> tuple[dict[str, jax.Array], dict[int, str]]:
    """Collect shared banks by producer object identity before serialization."""
    collection = _SourceArrayAliasCollection({}, {}, set())
    # Subscription-based role lookup need not construct a PyTree bound method
    # and flatten the retained source DAG merely to read its mapping metadata.
    target = Mapping.get(closure, "collective_target")
    if isinstance(target, _result.CellMeshingResult):
        if target.collective_evidence is None:
            raise ValueError(
                "source_closure.collective_target must retain collective evidence for shared arrays; got missing evidence"
            )
        _visit_source_array_aliases(
            target.collective_evidence,
            "collective_target.collective_evidence",
            collection,
        )
        accepted = Mapping.get(closure, "accepted_data", {})
        for name, array in accepted.get("fields", {}).items():
            _bind_source_array_alias(
                collection,
                f"collective/{target.collective_evidence.evidence_id}/field/{name}",
                array,
            )
    _visit_source_array_aliases(closure, "source_closure", collection)
    return dict(sorted(collection.arrays.items())), collection.identities


def _local_source_comparison_values(
    value: jax.Array | np.ndarray, /
) -> jax.Array | np.ndarray:
    """Expose CPU shard bytes without lowering a device reshape."""
    if isinstance(value, jax.Array):
        devices = value.devices()
        if (
            not value.is_fully_addressable
            or len(devices) != 1
            or next(iter(devices)).platform != "cpu"
        ):
            return value.reshape(-1)
    return np.asarray(value).reshape(-1)


@dataclass(frozen=True, slots=True, init=False)
class MeshingSourceValidation:
    "common-epoch admission reused by owner-local checkpoint phases."

    source_logical_arrays: Mapping[str, jax.Array]
    source_array_identities: frozenset[int]
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

        epochs: dict[
            str, _result.CollectiveMeshEvidence | InitialCollectiveMeshEvidence
        ] = {}
        active: set[int] = set()

        def consume(target: _result.CellMeshingResult) -> None:
            if id(target) in active:
                raise ValueError(
                    "source_closure.consume.stored_fields must match its declared source identity and lineage; got Scientific checkpoint source epochs contain a cyclic predecessor"
                )
            active.add(id(target))
            evidence = target.collective_evidence
            match evidence:
                case _result.CollectiveMeshEvidence():
                    consume(evidence.source)
                    if evidence.evidence_id not in epochs:
                        if target.mesh.storage is None:
                            raise ValueError(
                                "source_closure.consume.stored_fields must retain its declared source type and fields; got Collective source admission requires its numerical storage"
                            )
                        validate_partitioned_mesh_evidence(target.mesh.storage, evidence)
                        epochs[evidence.evidence_id] = evidence
                case InitialCollectiveMeshEvidence():
                    if evidence.evidence_id not in epochs:
                        evidence.require_current()
                        epochs[evidence.evidence_id] = evidence
                case _:
                    pass
            active.remove(id(target))

        for closure in closures:
            target = closure.get("collective_target")
            if isinstance(target, _result.CellMeshingResult):
                consume(target)
        object.__setattr__(
            self, "source_logical_arrays", MappingProxyType(dict(source_logical_arrays))
        )
        object.__setattr__(
            self,
            "source_array_identities",
            frozenset(id(array) for array in source_logical_arrays.values()),
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
                "source_closure.require_binding.stored_fields must retain its declared numerical inventory; got Source admission belongs to a different scientific bank"
            )
        if any(
            arrays[name] is not value
            for name, value in self.source_logical_arrays.items()
        ):
            raise ValueError(
                "source_closure.require_binding.stored_fields must retain its declared numerical inventory; got Source admission lost its immutable common-bank object references"
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
        left = _local_source_comparison_values(first)
        right = _local_source_comparison_values(second)
        for start in range(0, size, max(1, step)):
            stop = min(size, start + step)
            if not np.array_equal(
                np.asarray(jax.device_get(left[start:stop])).view(np.uint8),
                np.asarray(jax.device_get(right[start:stop])).view(np.uint8),
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
                "source_closure.require_epoch.stored_fields must match its declared source identity and lineage; got An owner record does not belong to the mathematically admitted common epoch"
            )
        if not self.equivalent(admitted.logical_arrays, evidence.logical_arrays):
            raise ValueError(
                "source_closure.require_epoch.stored_fields must match its declared source identity and lineage; got Owner numerical source banks differ from the admitted common epoch"
            )

    def array_equal(
        self, value: jax.Array | np.ndarray, other: jax.Array | np.ndarray, /
    ) -> bool:
        """Authenticate admitted bank objects and compare owner-local bytes."""
        if value.shape != other.shape or value.dtype != other.dtype:
            return False
        if not isinstance(value, jax.Array):
            return self._equal_local_values(value, other)
        if not isinstance(other, jax.Array):
            return False
        if (
            not value.is_fully_addressable
            and id(value) not in self.source_array_identities
        ):
            raise ValueError(
                "source_closure.equivalent.stored_fields must retain its declared numerical inventory; got A shared source field lacks its common-bank object binding"
            )
        if value is other:
            return True
        first_shards = value.addressable_shards
        second_shards = other.addressable_shards
        aligned = len(first_shards) == len(second_shards) and all(
            a.index == b.index for a, b in zip(first_shards, second_shards, strict=True)
        )
        if aligned:
            return all(
                self._equal_local_values(a.data, b.data)
                for a, b in zip(first_shards, second_shards, strict=True)
            )
        if not value.is_fully_addressable or not other.is_fully_addressable:
            return False
        if self.maximum_chunk_bytes is None:
            return logical_array_value_collection_digest(
                {"value": value}
            ) == logical_array_value_collection_digest({"value": other})
        return logical_array_value_collection_digest(
            {"value": value}, maximum_chunk_bytes=self.maximum_chunk_bytes
        ) == logical_array_value_collection_digest(
            {"value": other}, maximum_chunk_bytes=self.maximum_chunk_bytes
        )

    def equivalent(
        self,
        first: Any,
        second: Any,
        /,
        *,
        limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ) -> bool:

        pairs = model_recipe_array_pairs(first, second, limits=limits)
        if pairs is None:
            return False
        return all(self.array_equal(value, other) for value, other in pairs)


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


@checked
def native_family_audit_policy(
    source: NativeGenerationSource,
    specification: _contracts.SurfaceMeshingSpec | _contracts.VolumeMeshingSpec,
    mesh: CellMesh,
    /,
) -> _audit.CellMeshAuditPolicy:
    """Return the audit policy for the declared source family."""
    disposition = _audit.CellMeshAuditDisposition
    match source:
        case (
            _native_sources.NativeStructuredSource() | _native_sources.NativeSweepSource()
        ):
            return _audit.CellMeshAuditPolicy(
                watertight_boundary=disposition.REJECT
                if mesh.topological_dimension == 3
                else disposition.SKIP,
            )
        case _native_sources.NativeSurfaceSource():
            match specification:
                case _contracts.SurfaceMeshingSpec():
                    is_closed = closed(
                        compile_surface_domain(source.domain, specification)
                    )
                case _contracts.VolumeMeshingSpec():
                    raise TypeError(
                        "source_closure.generation_specification must be SurfaceMeshingSpec for a surface source; got VolumeMeshingSpec"
                    )
                case _ as invalid:
                    assert_never(invalid)
            return _audit.CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=disposition.REJECT if is_closed else disposition.SKIP,
            )
        case _native_sources.NativeLayerCoreSource():
            return publication_policy("layer_core").audit_policy
        case (
            _native_sources.NativePlanarSource()
            | _native_sources.NativePlcSource()
            | _native_sources.NativeMappedHexSource()
            | _native_sources.NativePolyhedralSource()
            | _native_sources.NativeSurfaceEnvelopeSource()
            | _native_sources.NativeImplicitSource()
            | CompartmentMeshingSource()
        ):
            return _audit.CellMeshAuditPolicy(
                require_complete_association=True, watertight_boundary=disposition.REJECT
            )
        case _native_periodic.NativePeriodicSource():
            return _audit.CellMeshAuditPolicy()
        case _ as invalid:
            assert_never(invalid)


def _require_registered_compartment(
    source: CompartmentMeshingSource,
    specification: _contracts.VolumeMeshingSpec,
    result: _result.CellMeshingResult,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    "Replay the occupied-image/material and independently authored outer source."

    source.validate_source_integrity()
    evidence = result.region_evidence
    if evidence is None or result.certification is None:
        raise ValueError(
            "source_closure.report.request.domain must retain its declared source type and fields; got Image registration requires the material/source evidence"
        )
    evidence.require_source(source.compartments)
    evidence.require_current(
        result.mesh, result.zones, result.patches, geometry=result.geometry
    )
    request = result.certification.request
    if not source_content_equal(request.domain, evidence.domain, limits=limits):
        raise ValueError(
            "source_closure.report.request.domain must match its retained source binding; got Image certificate lost its material domain"
        )
    regions = np.asarray(
        [evidence.domain.region_ids.index(region) for region in evidence.cell_region_ids],
        dtype=np.int64,
    )
    if request.cell_regions is None or not np.array_equal(request.cell_regions, regions):
        raise ValueError(
            "source_closure.report.request.domain must match its retained source binding; got Image certificate lost its cell-to-material assignments"
        )
    renewal = revalidate_region_evidence(
        result,
        result.mesh,
        result.geometry,
        result.zones,
        result.patches,
        compartment_source=source,
        certificate_limits=evidence.certificate_limits,
        limits=specification.limits,
    )
    if not source_content_equal(evidence, renewal.region_evidence, limits=limits):
        raise ValueError(
            "source_closure.report.request.domain must match its retained source binding; got Image material/interface/outer-source renewal differs from its scientific binding"
        )
    if not source_content_equal(
        result.zones, renewal.zones, limits=limits
    ) or not source_content_equal(result.patches, renewal.patches, limits=limits):
        raise ValueError(
            "source_closure.report.request.domain must match its retained source binding; got Image registration lost its material-zone and interface organization"
        )


def _require_registered_plc_geometry(
    source: _native_sources.NativePlcSource,
    specification: _contracts.VolumeMeshingSpec,
    result: _result.CellMeshingResult,
    /,
) -> None:
    "Bind coordinates to independent source rows and their bounds."
    exact = result.geometry.exact_source
    if not isinstance(exact, ExactPlcCellGeometrySource):
        return
    if (exact.domain_source_id, exact.domain_source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        raise ValueError(
            "source_closure.geometry.exact_source must match its declared source identity and lineage; got Registered PLC geometry must retain its source identity and revision"
        )
    with NativeExecutionScope(specification.limits.native_limits()) as scope:
        prepared = _volume_generation.prepare_plc_source(
            source.complex,
            source.source_id,
            source.source_revision,
            result.coordinate_contract,
            limits=specification.limits,
            scope=scope,
        )
    transfer = prepared.association_transfer
    triangles, edges = (
        np.asarray(transfer.triangle_vertices),
        np.asarray(transfer.edge_vertices),
    )
    diagonals = (
        _volume_generation.facet_diagonals(triangles, prepared.input_polygons, edges)[0]
        if source.complex.boundary == "fixed"
        else np.empty((0, 2), dtype=np.int32)
    )
    expected = _volume_generation.plc_source_rows(
        source.complex,
        specification,
        triangles,
        prepared.input_polygons,
        edges,
        diagonals,
    )
    exact.require_domain(
        expected.points, expected.faces, source.source_id, source.source_revision
    )
    for name, declared in zip(
        (
            "source_points",
            "source_triangles",
            "source_triangle_ids",
            "source_triangle_bounds",
            "source_segments",
            "source_segment_ids",
            "source_segment_bounds",
        ),
        (
            expected.points,
            expected.faces,
            expected.triangle_facets,
            expected.face_tolerances,
            expected.segments,
            expected.segment_ids,
            expected.segment_tolerances,
        ),
        strict=True,
    ):
        actual = np.asarray(object.__getattribute__(exact, name))
        same = actual.shape == declared.shape and (
            np.array_equal(actual.view(np.uint64), declared.view(np.uint64))
            if declared.dtype == np.dtype(np.float64)
            else np.array_equal(actual, declared)
        )
        if not same:
            raise ValueError(
                f"source_closure.geometry.exact_source must retain its declared numerical inventory; got Registered PLC geometry changed its {name} bank."
            )


def _require_registered_layer_source(
    request: MeshCertificationInputs,
    source: _native_sources.NativeLayerCoreSource,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    "Keep fidelity separate from its declared approximation coverage."

    if not source_content_equal(request.source, source.fidelity_source, limits=limits):
        raise ValueError(
            "source_closure.report.request.source must match its retained source binding; got Layer certificate must retain its independent fidelity source"
        )
    original = source.source_domain
    match original:
        case domain_coverage_owner.PiecewiseLinearDomain():
            if not source_content_equal(request.domain, original, limits=limits):
                raise ValueError(
                    "source_closure.report.request.source must match its retained source binding; got Layer certificate must retain its independently authored represented domain"
                )
        case domain.MeshingDomain():
            query = request.source
            match query:
                case domain_boundary_owner.MeshingDomainBoundarySource():
                    if not source_content_equal(query.domain, original, limits=limits):
                        raise ValueError(
                            "source_closure.report.request.source must match its retained source binding; got Layer certificate query must retain its independently authored parametric domain"
                        )
                case _:
                    raise ValueError(
                        "source_closure.report.request.source must retain its declared source type and fields; got Layer certificate requires its parametric domain query"
                    )
        case None:
            pass
        case _:
            raise TypeError(
                "source_closure.report.request.source must retain its declared source type and fields; got Layer source domain must retain its declared represented or parametric owner"
            )


def _require_registered_envelope(
    source: _native_sources.NativeSurfaceEnvelopeSource,
    specification: _contracts.VolumeMeshingSpec,
    result: _result.CellMeshingResult,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:

    report = result.certification
    if report is None:
        raise ValueError(
            "source_closure.report.request.domain must retain its declared source type and fields; got Registered envelope volume requires its source certificate"
        )
    original = source.plc_source
    declared = _volume_generation.declared_plc_domain(
        original.complex, original.source_id
    )
    if not source_content_equal(report.request.domain, declared, limits=limits):
        raise ValueError(
            "source_closure.report.request.domain must match its retained source binding; got Envelope volume differs from its independently authored repaired PLC domain"
        )
    _require_registered_plc_geometry(original, specification, result)
    evidence = source.envelope.evidence
    stage = _trace.MeshingStageReport(
        _trace.MeshingStageKind.TOPOLOGY_REPAIR,
        _trace.MeshingStageStatus.PASSED,
        input_ids=(
            evidence.source_id,
            evidence.source_revision,
            evidence.source_geometry_id,
            evidence.policy_id,
        ),
        output_ids=(
            source.source_id,
            source.source_revision,
            evidence.repaired_model_id,
            evidence.evidence_id,
        ),
        created_count=evidence.repaired_topology.faces,
        deleted_count=evidence.source_topology.faces,
    )
    if result.trace.stages[:1] != (stage,):
        raise ValueError(
            "source_closure.report.request.domain must match its retained source binding; got Envelope volume lost its source repair evidence"
        )
    achieved = dict(result.compliance.achieved)
    for name in (
        "source_to_repaired_upper",
        "repaired_to_source_upper",
        "source_inclusion_margin",
        "interpolation_error",
    ):
        if achieved.get(f"envelope:{name}") != object.__getattribute__(evidence, name):
            raise ValueError(
                f"source_closure.report.request.domain must match its retained source binding; got Envelope volume changed its {name} evidence."
            )


class _SourceEvidenceContext(StrictModule):
    mesh: CellMesh
    geometry: CellGeometrySpec
    audit: CellMeshAuditReport
    limits: ArrayArchiveLimits


def _reconstruct_source_evidence(
    value: Any,
    context: _SourceEvidenceContext,
    /,
    *,
    bindings: Mapping[str, Any] | None = None,
) -> None:
    supplied = {} if bindings is None else bindings
    match value:
        case global_embedding_owner.GlobalEmbeddingCertificate():
            rebuilt = (
                global_embedding_owner.GlobalEmbeddingCertificate.from_stored_fields(
                    value
                )
            )
        case domain_coverage_owner.DomainCoverageCertificate():
            rebuilt = domain_coverage_owner.DomainCoverageCertificate.from_stored_fields(
                value, supplied["domain"]
            )
        case source_fidelity_owner.SourceFidelityCertificate():
            rebuilt = source_fidelity_owner.SourceFidelityCertificate.from_stored_fields(
                value
            )
        case _:
            parameters = signature(type(value)).parameters
            rebuilt = reconstruct_stored_fields(
                value,
                bindings={
                    name: argument
                    for name, argument in supplied.items()
                    if name in parameters
                },
            )
    if not content_equal(value, rebuilt, limits=context.limits):
        raise ValueError(
            f"{type(value).__name__}.stored_fields must match its reconstructed evidence; got changed content"
        )


def _validate_embedding_evidence(
    report: MeshCertificationReport, context: _SourceEvidenceContext, /
) -> None:
    embedding = report.embedding
    if embedding is None:
        return
    if type(embedding) is not global_embedding_owner.GlobalEmbeddingCertificate:
        raise TypeError(
            f"MeshCertificationReport.embedding must have the registered embedding owner; got {type(embedding).__name__}"
        )
    embedding.binding.require(context.mesh, context.geometry)
    _reconstruct_source_evidence(
        embedding.binding,
        context,
        bindings={
            "mesh": context.mesh,
            "geometry": context.geometry,
            "limits": report.request.limits,
        },
    )
    for finding in embedding.findings:
        _reconstruct_source_evidence(finding, context)
    if embedding.binding.limits_id != report.request.limits.limits_id:
        raise ValueError(
            "MeshCertificationReport.embedding.limits_id must match its certificate controls; got different identities"
        )
    for name, maximum in (
        ("source_expression_work_units", report.request.limits.maximum_work_units),
        ("source_expression_peak_bytes", report.request.limits.maximum_scratch_bytes),
    ):
        quantity = object.__getattribute__(embedding, name)
        if type(quantity) is not int or not 0 <= quantity <= maximum:
            raise ValueError(
                f"MeshCertificationReport.embedding.{name} must lie in [0, {maximum}]; got {quantity}"
            )
    _reconstruct_source_evidence(embedding, context)


def _validate_coverage_evidence(
    report: MeshCertificationReport, context: _SourceEvidenceContext, /
) -> None:
    coverage = report.coverage
    if coverage is None:
        return
    if (
        type(coverage) is not domain_coverage_owner.DomainCoverageCertificate
        or report.embedding is None
    ):
        raise TypeError(
            "MeshCertificationReport.coverage must retain its embedding and domain certificate owners; got missing or incorrect owner"
        )
    coverage.binding.require(context.mesh, context.geometry)
    _reconstruct_source_evidence(
        coverage.binding,
        context,
        bindings={
            "mesh": context.mesh,
            "geometry": context.geometry,
            "limits": report.request.limits,
        },
    )
    for finding in coverage.findings:
        _reconstruct_source_evidence(finding, context)
    if coverage.embedding_certificate_id != report.embedding.certificate_id:
        raise ValueError(
            "MeshCertificationReport.coverage.embedding_certificate_id must match its embedding evidence; got a different dependency"
        )
    _reconstruct_source_evidence(
        coverage, context, bindings={"domain": report.request.domain}
    )


def _validate_report_evidence(
    report: MeshCertificationReport, context: _SourceEvidenceContext, /
) -> None:
    report.request.validate_source_integrity()
    if type(report.request.limits) is not MeshCertificateLimits:
        raise TypeError(
            f"MeshCertificationReport.request.limits must have the registered limits owner; got {type(report.request.limits).__name__}"
        )
    _reconstruct_source_evidence(report.request.limits, context)
    _reconstruct_source_evidence(report.schedule, context)
    report.require_passed()
    _validate_embedding_evidence(report, context)
    _validate_coverage_evidence(report, context)
    dependencies = {
        "global_embedding": report.embedding,
        "domain_coverage": report.coverage,
    }
    for outcome in report.outcomes:
        if type(outcome) is not MeshCertificationOutcome:
            raise TypeError(
                f"MeshCertificationReport.outcomes must retain registered outcome records; got {type(outcome).__name__}"
            )
        _reconstruct_source_evidence(outcome, context)
        dependency = dependencies.get(outcome.check)
        if dependency is not None and outcome.evidence_id != dependency.certificate_id:
            raise ValueError(
                f"MeshCertificationReport.outcomes.{outcome.check}.evidence_id must match its certificate; got {outcome.evidence_id}"
            )
    _reconstruct_source_evidence(
        report,
        context,
        bindings={
            "mesh": context.mesh,
            "geometry": context.geometry,
            "audit": context.audit,
        },
    )


def _compare_source_evidence_fields(
    left: Any, right: Any, omitted: frozenset[str], limits: ArrayArchiveLimits, /
) -> None:
    if type(left) is not type(right):
        raise TypeError(
            f"source_evidence.owner must remain the same; got {type(left).__name__} and {type(right).__name__}"
        )
    for member in fields(left):
        if member.name not in omitted and not content_equal(
            object.__getattribute__(left, member.name),
            object.__getattribute__(right, member.name),
            limits=limits,
        ):
            raise ValueError(
                f"source_evidence.{type(left).__name__}.{member.name} must retain its scientific content; got changed content"
            )


def _compare_coverage_evidence(
    left: domain_coverage_owner.DomainCoverageCertificate,
    right: domain_coverage_owner.DomainCoverageCertificate,
    limits: ArrayArchiveLimits,
    /,
) -> None:
    if len(left.premise_certificate_ids) != len(right.premise_certificate_ids):
        raise ValueError(
            "source_evidence.coverage.premise_certificate_ids must retain its dependency count; got a different count"
        )
    _compare_source_evidence_fields(
        left,
        right,
        frozenset(
            {
                "embedding_certificate_id",
                "premise_certificate_ids",
                "source_expression_work_units",
                "source_expression_required_work_units",
                "source_expression_peak_bytes",
                "certificate_id",
            }
        ),
        limits,
    )


def _compare_report_certificate_evidence(
    original: MeshCertificationReport,
    renewed: MeshCertificationReport,
    limits: ArrayArchiveLimits,
    /,
) -> None:
    if (original.embedding is None) != (renewed.embedding is None) or (
        original.coverage is None
    ) != (renewed.coverage is None):
        raise ValueError(
            "source_evidence.certificates must retain the scheduled certificate owners; got different owners"
        )
    if original.embedding is not None and renewed.embedding is not None:
        _compare_source_evidence_fields(
            original.embedding,
            renewed.embedding,
            frozenset(
                {
                    "source_expression_work_units",
                    "source_expression_peak_bytes",
                    "certificate_id",
                }
            ),
            limits,
        )
    if original.coverage is not None and renewed.coverage is not None:
        _compare_coverage_evidence(original.coverage, renewed.coverage, limits)
    if (original.fidelity is None) != (renewed.fidelity is None):
        raise ValueError(
            "source_evidence.fidelity must retain its scheduled certificate owner; got different owners"
        )
    if original.fidelity is not None and renewed.fidelity is not None:
        _compare_source_evidence_fields(
            original.fidelity,
            renewed.fidelity,
            frozenset({"domain_coverage", "certificate_id"}),
            limits,
        )
        old_domain, new_domain = (
            original.fidelity.domain_coverage,
            renewed.fidelity.domain_coverage,
        )
        if (old_domain is None) != (new_domain is None):
            raise ValueError(
                "source_evidence.fidelity.domain_coverage must retain its certificate owner; got different owners"
            )
        if old_domain is not None and new_domain is not None:
            _compare_coverage_evidence(old_domain, new_domain, limits)


@checked
def require_same_source_evidence(
    original: MeshCertificationReport,
    renewed: MeshCertificationReport,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    audit: CellMeshAuditReport,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> tuple[MeshCertificationReport, MeshCertificationReport]:
    """Authenticate renewed scientific content separately from execution history.

    Both reports are reconstructed before comparing their measured outcomes.
    Historical work quantities and dependent certificate IDs remain unchanged
    in the reports but do not define equality of their scientific content.
    """
    register_meshing_source_artifacts()
    context = _SourceEvidenceContext(mesh, geometry, audit, limits)
    _validate_report_evidence(original, context)
    _validate_report_evidence(renewed, context)
    _compare_source_evidence_fields(
        original,
        renewed,
        frozenset({"embedding", "coverage", "fidelity", "outcomes", "report_id"}),
        limits,
    )
    _compare_report_certificate_evidence(original, renewed, limits)
    if len(original.outcomes) != len(renewed.outcomes):
        raise ValueError(
            "source_evidence.outcomes must retain the scheduled outcome count; got a different count"
        )
    for old, new in zip(original.outcomes, renewed.outcomes, strict=True):
        omitted = frozenset({"outcome_id"})
        if old.check in ("global_embedding", "domain_coverage", "source_fidelity"):
            omitted |= {"evidence_id"}
        _compare_source_evidence_fields(old, new, omitted, limits)
    return original, renewed


class _RegisteredGeneration(StrictModule):
    result: CellMeshingResult
    request: MeshCertificationInputs
    report: MeshCertificationReport
    volume_specification: _contracts.VolumeMeshingSpec | None

    def require_volume_specification(self) -> _contracts.VolumeMeshingSpec:
        """Return volume controls retained by the native-family validation."""
        specification = self.volume_specification
        if specification is None:
            raise RuntimeError(
                "source_closure.generation_options must retain its validated volume request; got missing volume specification"
            )
        return specification


def _validate_registered_generation(
    part: MeshPart,
    source: NativeGenerationSource,
    specification: _contracts.SurfaceMeshingSpec | _contracts.VolumeMeshingSpec,
    options: NativeMeshingOptions | None,
    /,
) -> _RegisteredGeneration:
    if type(source) not in native_generation_source_types():
        raise TypeError(
            "source_closure.generation_options must match its retained source binding; got Generation source must retain its native family"
        )
    volume = type(source) in (
        _native_sources.NativePlcSource,
        _native_sources.NativePolyhedralSource,
        _native_sources.NativeMappedHexSource,
        _native_sources.NativeSweepSource,
        _native_sources.NativeLayerCoreSource,
        _native_sources.NativeSurfaceEnvelopeSource,
        CompartmentMeshingSource,
    )
    if type(specification) not in (
        (_contracts.VolumeMeshingSpec,)
        if volume
        else (_contracts.SurfaceMeshingSpec, _contracts.VolumeMeshingSpec)
    ):
        raise TypeError(
            "source_closure.generation_options must match its retained source binding; got Generation hard request must retain its native family contract"
        )
    if (
        type(source)
        in (_native_sources.NativePlanarSource, _native_sources.NativeSurfaceSource)
        and type(specification) is not _contracts.SurfaceMeshingSpec
    ):
        raise TypeError(
            "source_closure.generation_options must retain its declared source type and fields; got Native surface generation requires SurfaceMeshingSpec"
        )
    match specification:
        case _contracts.VolumeMeshingSpec():
            scope = specification.boundary_scope
            volume_specification = specification
        case _contracts.SurfaceMeshingSpec():
            scope = specification.scope
            volume_specification = None
        case _ as invalid:
            assert_never(invalid)
    if (scope.source_id, scope.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        raise ValueError(
            "source_closure.generation_options must match its declared source identity and lineage; got generation request must bind its authored source revision"
        )
    match part.carrier:
        case CellMeshingResult() as result:
            report = result.certification
            if report is None:
                raise ValueError(
                    "source_closure.generation_options must retain its declared source type and fields; got Native generation requires the whole source-certified CellMeshingResult"
                )
        case _:
            raise ValueError(
                "source_closure.generation_options must retain its declared source type and fields; got Native generation requires the whole source-certified CellMeshingResult"
            )
    if result.compliance.specification_id != specification.specification_id:
        raise ValueError(
            "source_closure.generation_options must retain its declared source type and fields; got Registered carrier lost its hard generation request"
        )
    binding = result.trace.binding
    if binding is None or (binding.source_id, binding.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        raise ValueError(
            "source_closure.generation_options must match its declared source identity and lineage; got Registered native result must retain its generation source revision"
        )
    if options is not None:
        if type(options) is not NativeMeshingOptions:
            raise TypeError(
                "source_closure.generation_options must retain its declared source type and fields; got Native generation options require their algorithm declaration"
            )

        # This prepares real source-specific controls and rejects unsupported
        # family/target/option combinations. No live plan enters the archive.
        plan = NativeMeshingProvider(options).plan(
            source,
            specification,
            coordinate_contract=result.coordinate_contract,
        )
        provenance = json.loads(result.provenance.content_json)
        if (
            provenance.get("kind") != "mapping"
            or dict(provenance["items"]).get("plan") != plan.plan_id
        ):
            raise ValueError(
                "source_closure.generation_options must match its retained source binding; got Native generation options differ from the source-bound generation plan"
            )
    elif type(source) is not _native_sources.NativePlanarSource:
        raise ValueError(
            "source_closure.generation_options must match its retained source binding; got Native family registration must retain its generation options"
        )
    return _RegisteredGeneration(result, report.request, report, volume_specification)


def _require_registered_structured_source(
    source: _native_sources.NativeStructuredSource | _native_sources.NativeSweepSource,
    specification: _contracts.SurfaceMeshingSpec | _contracts.VolumeMeshingSpec,
    result: CellMeshingResult,
    request: MeshCertificationInputs,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    if not source_content_equal(request.source, source.fidelity_source, limits=limits):
        raise ValueError(
            "source_closure.require_registered_structured_source.stored_fields must match its retained source binding; got Structured certificate must retain its independent authored boundary query"
        )
    if not source_content_equal(request.domain, source.domain, limits=limits):
        raise ValueError(
            "source_closure.require_registered_structured_source.stored_fields must match its retained source binding; got Structured certificate must bind its independently declared represented domain"
        )


def _require_registered_mapped_source(
    source: _native_sources.NativeMappedHexSource,
    specification: _contracts.SurfaceMeshingSpec | _contracts.VolumeMeshingSpec,
    result: CellMeshingResult,
    request: MeshCertificationInputs,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    if not source_content_equal(request.domain, source.domain, limits=limits):
        raise ValueError(
            "source_closure.require_registered_mapped_source.stored_fields must match its retained source binding; got Mapped certificate must bind its independent source coordinate maps"
        )
    if type(
        request.source
    ) is not source_fidelity_owner.MappedDomainBoundarySource or not source_content_equal(
        request.source.domain, source.domain, limits=limits
    ):
        raise ValueError(
            "source_closure.require_registered_mapped_source.stored_fields must match its retained source binding; got Mapped certificate must retain its mapped boundary binding"
        )


def _require_registered_mapped_layer_source(
    source: _native_sources.NativeLayerCoreSource,
    specification: _contracts.VolumeMeshingSpec,
    result: CellMeshingResult,
    request: MeshCertificationInputs,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    mapped_domain = source.mapped_domain
    if mapped_domain is None:
        raise ValueError(
            "source_closure.require_registered_mapped_layer_source.stored_fields must match its retained source binding; got Mapped layer certificate must retain its independent source roots"
        )
    if not source_content_equal(request.domain, mapped_domain, limits=limits):
        raise ValueError(
            "source_closure.require_registered_mapped_layer_source.stored_fields must match its retained source binding; got Mapped layer certificate must retain its independent source roots"
        )
    match source.fidelity_source, request.source:
        case (
            source_fidelity_owner.MappedDomainBoundarySource() as fidelity,
            source_fidelity_owner.MappedDomainBoundarySource() as query,
        ):
            pass
        case _:
            raise ValueError(
                "source_closure.require_registered_mapped_layer_source.stored_fields must match its retained source binding; got Mapped layer certificate must retain its independent fidelity source and explicit target-region renewal"
            )
    if (
        not source_content_equal(query.domain, fidelity.domain, limits=limits)
        or not source_content_equal(query.limits, fidelity.limits, limits=limits)
        or query.covering_radius != fidelity.covering_radius
        or not np.array_equal(
            np.asarray(fidelity.cell_regions),
            np.asarray(mapped_domain.cell_regions),
        )
        or request.cell_regions is None
        or not np.array_equal(
            np.asarray(query.cell_regions),
            np.asarray(request.cell_regions),
        )
    ):
        raise ValueError(
            "source_closure.require_registered_mapped_layer_source.stored_fields must match its retained source binding; got Mapped layer certificate must retain its independent fidelity source and explicit target-region renewal"
        )
    transfer = _layer_core_association.prepare_native_layer_association_transfer(
        source, specification, result
    )
    transfer.source_associations(result)


def _require_registered_surface_source(
    source: _native_sources.NativeSurfaceSource,
    specification: _contracts.SurfaceMeshingSpec | _contracts.VolumeMeshingSpec,
    result: CellMeshingResult,
    request: MeshCertificationInputs,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    if type(
        request.source
    ) is not MeshingDomainBoundarySource or not source_content_equal(
        request.source.domain, source.domain, limits=limits
    ):
        raise ValueError(
            "source_closure.require_registered_surface_source.stored_fields must match its retained source binding; got Surface certificate must retain its source chart domain"
        )


def _require_registered_plc_source(
    source: _native_sources.NativePlcSource | _native_sources.NativePolyhedralSource,
    specification: _contracts.VolumeMeshingSpec,
    result: CellMeshingResult,
    request: MeshCertificationInputs,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    if (
        request.domain is None
        or request.domain.source_id != source.source_id
        or request.domain.region_ids != source.complex.region_ids
    ):
        raise ValueError(
            "source_closure.require_registered_plc_source.stored_fields must match its retained source binding; got PLC certificate must bind its authored source domain and regions"
        )
    declared = _volume_generation.declared_plc_domain(source.complex, source.source_id)
    if not source_content_equal(request.domain, declared, limits=limits):
        raise ValueError(
            "source_closure.require_registered_plc_source.stored_fields must match its retained source binding; got PLC certificate differs from its independent source polygon domain"
        )
    match source:
        case _native_sources.NativePlcSource():
            _require_registered_plc_geometry(source, specification, result)
        case _native_sources.NativePolyhedralSource():
            pass
        case _ as invalid:
            assert_never(invalid)


def _require_registered_implicit_source(
    source: NativeImplicitSource,
    result: CellMeshingResult,
    request: MeshCertificationInputs,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    """Bind the retained implicit field to the independently certified query."""
    query = request.source
    match query:
        case source_fidelity_owner.ImplicitProjectionBoundarySource():
            query.profile.require_bound(source.geometry, result.coordinate_contract)
            geometry = query.profile.geometry
        case source_fidelity_owner.ImplicitBoundarySource():
            geometry = query.geometry
        case _:
            raise ValueError(
                "source_closure.implicit_source must retain its implicit boundary query; got another source type"
            )
    if (query.source_id, query.source_revision) != (
        source.source_id,
        source.source_revision,
    ) or not source_content_equal(geometry, source.geometry, limits=limits):
        raise ValueError(
            "source_closure.implicit_source must bind the retained field and revision; got a foreign geometry or source identity"
        )


def _bind_registered_generation(
    context: _RegisteredGeneration,
    source: NativeGenerationSource,
    specification: _contracts.SurfaceMeshingSpec | _contracts.VolumeMeshingSpec,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    result, request = context.result, context.request
    validate_restored_meshing_source_bindings(
        request.source, request, context.report, limits=limits
    )
    for association in result.associations:
        match source:
            case _native_sources.NativeLayerCoreSource():
                validate_layer_core_association(
                    request, association, source, mesh=result.mesh
                )
            case _:
                validate_association_binding(
                    request, association, generation_source=source
                )
    match source:
        case (
            _native_sources.NativeStructuredSource() | _native_sources.NativeSweepSource()
        ):
            _require_registered_structured_source(
                source, specification, result, request, limits=limits
            )
        case _native_sources.NativeMappedHexSource():
            _require_registered_mapped_source(
                source, specification, result, request, limits=limits
            )
        case _native_sources.NativeLayerCoreSource() if source.mapped_domain is not None:
            _require_registered_mapped_layer_source(
                source,
                context.require_volume_specification(),
                result,
                request,
                limits=limits,
            )
        case _native_sources.NativeLayerCoreSource():
            _require_registered_layer_source(request, source, limits=limits)
        case _native_sources.NativeSurfaceSource():
            _require_registered_surface_source(
                source, specification, result, request, limits=limits
            )
        case _native_sources.NativeSurfaceEnvelopeSource():
            _require_registered_envelope(
                source, context.require_volume_specification(), result, limits=limits
            )
        case CompartmentMeshingSource():
            _require_registered_compartment(
                source, context.require_volume_specification(), result, limits=limits
            )
        case NativeImplicitSource():
            _require_registered_implicit_source(source, result, request, limits=limits)
        case _native_sources.NativePlcSource() | _native_sources.NativePolyhedralSource():
            _require_registered_plc_source(
                source,
                context.require_volume_specification(),
                result,
                request,
                limits=limits,
            )
        case (
            _native_sources.NativePlanarSource() | _native_periodic.NativePeriodicSource()
        ):
            # Their domain binding is validated by the shared request checks.
            return
        case _ as invalid:
            assert_never(invalid)


def _renew_registered_generation(
    context: _RegisteredGeneration,
    source: NativeGenerationSource,
    specification: _contracts.SurfaceMeshingSpec | _contracts.VolumeMeshingSpec,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> tuple[MeshCertificationReport, MeshCertificationReport]:
    result, request = context.result, context.request
    policy = native_family_audit_policy(source, specification, result.mesh)
    if policy.policy_id != result.audit.policy_id:
        raise ValueError(
            "source_closure.renew_registered_generation.stored_fields must match its retained source binding; got Native carrier lost its source-family audit policy"
        )
    audit = _audit.audit_cell_mesh(
        result.mesh,
        result.geometry,
        policy=policy,
        boundary=result.boundary,
        patches=result.patches,
        associations=result.associations,
        attributes=result.attributes,
        zones=result.zones,
        labels=result.labels,
    )
    audit.require_passed()
    audit.require_decided()
    if not source_content_equal(result.audit, audit, limits=limits):
        raise ValueError(
            "source_closure.renew_registered_generation.stored_fields must match its retained source binding; got Stored native audit differs from its fresh family audit"
        )
    require_same_source_evidence(
        context.report,
        context.report,
        result.mesh,
        result.geometry,
        audit,
        limits=limits,
    )
    fresh = recertify_restored_meshing_source(
        request, result.mesh, result.geometry, audit, archive_limits=limits
    )
    return require_same_source_evidence(
        context.report,
        fresh,
        result.mesh,
        result.geometry,
        audit,
        limits=limits,
    )


@checked
def validate_registered_native_part(
    part: MeshPart,
    source: NativeGenerationSource,
    specification: _contracts.SurfaceMeshingSpec | _contracts.VolumeMeshingSpec,
    options: NativeMeshingOptions | None = None,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> tuple[MeshCertificationReport, MeshCertificationReport]:
    """Bind retained generation inputs to a fresh family audit and source evidence."""
    context = _validate_registered_generation(part, source, specification, options)
    _bind_registered_generation(context, source, specification, limits=limits)
    return _renew_registered_generation(context, source, specification, limits=limits)
