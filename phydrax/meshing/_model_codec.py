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
    _read_array_archive,
    array_collection_digest,
    array_payload_digest,
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    read_array_archive,
    write_array_archive,
)
from .._fingerprint import canonical_fingerprint
from .._model._artifacts import artifact_value, artifact_value_id
from .._model._structure import (
    _canonical_recipe_payload,
    _expand_shared_immutable_model_archive_recipe,
    _model_from_shared_immutable_array_recipe,
    _pack_owned_model_array_tree,
    _pack_projected_model_array_tree,
    _recipe_array_nodes,
    _require_model_array_content,
    _row_record_recipe_types,
    _same_array_bytes,
    AdaptiveImplicitBoundarySourceArchiveRecipe,
    AffinePlcBoundarySourceArchiveRecipe,
    AffinePlcDomainArchiveRecipe,
    AffinePlcGeometrySourceArchiveRecipe,
    AnalyticImplicitProfileArchiveRecipe,
    CADGraphArchiveRecipe,
    CADRevisionArchiveRecipe,
    CADTransactionArchiveRecipe,
    CompartmentMeshingSourceArchiveRecipe,
    DistributedFiniteElementTransferArchiveRecipe,
    expand_shared_model_archive_recipe,
    FiniteElementAcceptedStateArchiveRecipe,
    FiniteElementFieldBanksArchiveRecipe,
    FiniteElementHPEpochArchiveRecipe,
    FiniteVolumeOversetPartSpecArchiveRecipe,
    FiniteVolumeOversetPolicyArchiveRecipe,
    FiniteVolumeOwnerLocalCellStateBankArchiveRecipe,
    FiniteVolumeOwnerLocalRepartitionContextArchiveRecipe,
    FiniteVolumeRuntimeStateArchiveRecipe,
    ImageFieldSpecArchiveRecipe,
    ImageIndexAffineArchiveRecipe,
    ImageQualityFlagArchiveRecipe,
    ImageUncertaintyArchiveRecipe,
    LabelVolumeArchiveRecipe,
    LayerCoreAcceptedAdaptationArchiveRecipe,
    LayerCoreAcceptedResultArchiveRecipe,
    LayerCoreFidelitySourceReferenceArchiveRecipe,
    LayerCoreSourceReferenceArchiveRecipe,
    MappedCoreFacetAuthorityArchiveRecipe,
    MappedReferenceBoundarySourceArchiveRecipe,
    MeshingDomainArchiveRecipe,
    MeshingDomainBoundarySourceArchiveRecipe,
    MixedAdaptationPolicyArchiveRecipe,
    model_archive_projection,
    model_archive_root_projection,
    model_from_array_recipe,
    model_structure_matches_recipe,
    model_structure_recipe,
    NativeImplicitBoundaryQueryArchiveRecipe,
    NativeImplicitSourceArchiveRecipe,
    NativeLayerCoreAuthoringArchiveRecipe,
    NativeLayerMappedBoundaryArchiveRecipe,
    NativeMappedHexSourceArchiveRecipe,
    NativePlanarMotionSourceArchiveRecipe,
    NativePlanarSourceArchiveRecipe,
    NativePlcSourceArchiveRecipe,
    OversetRegistrationArchiveRecipe,
    pack_model_array_tree,
    ParametricCurveBoundarySourceArchiveRecipe,
    PartitionOperandArchiveRecipe,
    PartitionPlanArchiveRecipe,
    PERIODIC_FINITE_ELEMENT_TRANSFER_ROOT_ENTRY_IDS,
    PeriodicFiniteElementTransferArchiveRecipe,
    PiecewiseLinearBoundarySourceArchiveRecipe,
    PiecewiseLinearComplexArchiveRecipe,
    PlanarMotionAdvanceArchiveRecipe,
    PlcCurveAuthorityArchiveRecipe,
    PreparedSphereChartDeformationArchiveRecipe,
    PreparedSurfaceSourceSupportArchiveRecipe,
    ReconstructedTriangleBoundarySourceArchiveRecipe,
    restore_row_record_recipes,
    share_model_archive_recipe,
    SourceBoundaryFaceImageWitnessArchiveRecipe,
    SourceBoundaryImagePullbackArchiveRecipe,
    SparseCoordinateOperatorArchiveRecipe,
    SphereMaterialCellAtlasArchiveRecipe,
    SurfaceChartBoundarySourceArchiveRecipe,
    SurfaceEnvelopeSourceArchiveRecipe,
    SurfaceNativeRestrictionBoundarySourceArchiveRecipe,
    SurfaceSourceChartsArchiveRecipe,
    SurfaceSubdivisionBoundarySourceArchiveRecipe,
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
# so composite models admit 4,096 leaves (plus the manifest). Occurrence-qualified
# source identities and exact evidence also occupy manifest space: a complete
# layer/core model can exceed the generic 1 MiB allowance without large arrays.
# Bound that model metadata separately at 16 MiB; central-directory, member,
# aggregate-byte, dtype and shape limits retain their conservative defaults.
MODEL_ARCHIVE_LIMITS = ArrayArchiveLimits(
    max_manifest_nesting=64, max_members=4097, max_manifest_bytes=16_777_216
)


def _same_model_content(left: Any, right: Any, /) -> bool:
    """Compare canonical recipes and array payloads, streaming fidelity fields."""
    from ..geometry._source_fidelity_certificate import SourceFidelityCertificate

    if (
        type(left) is SourceFidelityCertificate
        and type(right) is SourceFidelityCertificate
    ):
        from ._archive_layer_columns import _COLUMNS, _encode_column

        columns = _COLUMNS["SourceFidelityCertificate"]
        for field in dataclasses.fields(left):
            a = object.__getattribute__(left, field.name)
            b = object.__getattribute__(right, field.name)
            if field.name in columns:
                a = _encode_column(a, columns[field.name])
                b = _encode_column(b, columns[field.name])
            if not _same_model_content(
                model_archive_projection(a), model_archive_projection(b)
            ):
                return False
        return True
    left = model_archive_root_projection(left)
    right = model_archive_root_projection(right)
    if type(left) is not type(right):
        return False
    recipe = model_structure_recipe(left, path="compare")
    if _canonical_recipe_payload(recipe) != _canonical_recipe_payload(
        model_structure_recipe(right, path="compare")
    ):
        return False
    # These are existing operands, not an archive admission. A caller may have
    # admitted them under a larger policy than the generic archive defaults.
    left_arrays = _pack_projected_model_array_tree(
        left, recipe, prefix="compare", limits=None
    )
    right_arrays = _pack_projected_model_array_tree(
        right, recipe, prefix="compare", limits=None
    )
    return left_arrays.keys() == right_arrays.keys() and all(
        array_payload_digest(left_arrays[key]) == array_payload_digest(right_arrays[key])
        for key in left_arrays
    )


def _hp_epoch_archive_recipe(value: Any, /) -> FiniteElementHPEpochArchiveRecipe:
    """Capture an epoch's concrete prepared field layouts, not their callbacks."""
    import numpy as np

    from ..discretization.fem._hp_runtime import FiniteElementHPTraceConstraintPlan

    fields = None
    if value.discretization is not None:
        topology = value.topology
        layouts = []
        for field, dofs, elements in zip(
            value.discretization.field_spaces,
            value.discretization.dof_maps,
            value.discretization.elements,
            strict=True,
        ):
            continuity = {element.continuity for element in elements}
            if continuity not in ({"conforming"}, {"discontinuous"}):
                raise ValueError("hp field has unsupported element continuity.")
            conformity = "L2" if continuity == {"discontinuous"} else "H1"
            degrees = tuple(
                tuple(
                    int(np.unique(np.asarray(element.reference_nodes)[:, axis]).size - 1)
                    for axis in range(topology.dimension)
                )
                for element in elements
            )
            layouts.append(
                (
                    field.name,
                    conformity,
                    dofs.component_shape,
                    degrees,
                    tuple(x.element_id for x in elements),
                )
            )
        fields = tuple(layouts)
    if any(
        type(plan) is not FiniteElementHPTraceConstraintPlan
        for _, plan in value.constraints
    ):
        raise TypeError("hp epoch contains an unsupported trace constraint.")
    return FiniteElementHPEpochArchiveRecipe(
        value.mesh,
        value.topology,
        value.geometry,
        fields,
        value.epoch_id,
        value.worksets.plan_id,
        value.interfaces.plan_id,
        None if value.discretization is None else value.discretization.prepared_id,
        tuple((name, plan.plan_id) for name, plan in value.constraints),
    )


def _restore_hp_epoch_archive_recipe(
    item: FiniteElementHPEpochArchiveRecipe, restore: Callable[[Any], Any], /
) -> Any:
    import numpy as np

    from ..discretization.fem._generic import FiniteElementFieldSpec, FiniteElementPlan
    from ..discretization.fem._high_order import ReferenceNodalFamily
    from ..discretization.fem._hp_runtime import (
        _hp_trace_constraint_for_field,
        certify_finite_element_hp_geometry,
        finite_element_hp_interface_plan,
        FiniteElementHPEpoch,
        FiniteElementHPGeometry,
        FiniteElementHPTraceConstraintPlan,
    )
    from ..discretization.fem._reference import FiniteElementSpec

    topology = restore(item.topology)
    stored_geometry = item.geometry
    if type(stored_geometry) is not FiniteElementHPGeometry:
        raise TypeError("hp epoch requires its canonical geometry owner.")
    geometry = FiniteElementHPGeometry(
        topology,
        stored_geometry.cell_vertices,
        stored_geometry.reference_lower,
        stored_geometry.reference_upper,
    )
    if not _same_model_content(geometry, stored_geometry):
        raise ValueError("Stored hp geometry differs from its constructor.")
    interfaces = finite_element_hp_interface_plan(topology, geometry)
    if not certify_finite_element_hp_geometry(topology, geometry, interfaces).passed:
        raise ValueError("Stored hp geometry failed its certificate.")
    mesh = restore(item.mesh)
    slots = {
        int(identifier): slot
        for slot, identifier in enumerate(np.asarray(topology.cell_global_ids))
        if bool(np.asarray(topology.active)[slot])
    }
    for block in mesh.blocks:
        if block.cell_kind != topology.cell_kind or not np.all(block.vertex_valid):
            raise ValueError("Stored hp mesh has an incompatible cell block.")
        try:
            block_slots = [slots[int(identifier)] for identifier in block.global_ids]
        except KeyError as error:
            raise ValueError("Stored hp mesh names an unknown active cell.") from error
        if not np.array_equal(
            np.asarray(mesh.coordinates)[np.asarray(block.vertices)],
            np.asarray(geometry.cell_vertices)[block_slots],
        ):
            raise ValueError("Stored hp mesh differs from its forest geometry.")
    if item.fields is None:
        epoch = FiniteElementHPEpoch(mesh, topology, geometry, interfaces)
    else:
        specs = []
        for name, conformity, components, degrees, identifiers in item.fields:
            if conformity not in ("H1", "L2") or len(degrees) != len(mesh.blocks):
                raise ValueError("Stored hp field layout is invalid.")
            elements = {}
            for block, degree, identifier in zip(
                mesh.blocks, degrees, identifiers, strict=True
            ):
                element = ReferenceNodalFamily(
                    topology.cell_kind, degree
                ).finite_element()
                if conformity == "L2":
                    entities: list[tuple[tuple[int, ...], ...]] = [
                        tuple(() for _ in level) for level in element.entity_dofs
                    ]
                    entities[-1] = (tuple(range(element.local_dof_count)),)
                    element = FiniteElementSpec(
                        "DiscontinuousTensorProductLagrange",
                        element.cell_kind,
                        element.degree,
                        element.reference_nodes,
                        tuple(entities),
                        value_spec=element.value_spec,
                        continuity="discontinuous",
                        representation=element.representation,
                        tabulator=element.tabulate,
                        tabulator_id=f"discontinuous:{element.element_id}",
                    )
                if element.element_id != identifier:
                    raise ValueError("Stored hp field element identity changed.")
                elements[block.name] = element
            specs.append(
                FiniteElementFieldSpec(name, elements, component_shape=components)
            )
        discretization = FiniteElementPlan(mesh, tuple(specs)).prepare()
        slots_by_id = {
            int(identifier): slot
            for slot, identifier in enumerate(np.asarray(topology.cell_global_ids))
            if bool(np.asarray(topology.active)[slot])
        }
        active_slots = np.asarray(
            [
                slots_by_id[int(identifier)]
                for block in mesh.blocks
                for identifier in block.global_ids
            ],
            dtype=np.int32,
        )
        constraints = tuple(
            (
                name,
                _hp_trace_constraint_for_field(
                    topology, interfaces, discretization, active_slots, index
                ),
            )
            for index, (name, conformity, _, _, _) in enumerate(item.fields)
            if conformity == "H1"
        )
        if any(
            type(plan) is not FiniteElementHPTraceConstraintPlan
            for _, plan in constraints
        ):
            raise TypeError("hp trace preparation returned an unsupported owner.")
        epoch = FiniteElementHPEpoch(
            mesh,
            topology,
            geometry,
            interfaces,
            discretization=discretization,
            constraints=constraints,
        )
    if (
        epoch.epoch_id != item.epoch_id
        or epoch.worksets.plan_id != item.workset_id
        or epoch.interfaces.plan_id != item.interface_id
        or (None if epoch.discretization is None else epoch.discretization.prepared_id)
        != item.prepared_id
        or tuple((name, getattr(plan, "plan_id")) for name, plan in epoch.constraints)
        != item.constraint_ids
    ):
        raise ValueError("Stored hp epoch identity differs from canonical preparation.")
    return epoch


def _same_implicit_archive(left: Any, right: Any, /) -> bool:
    """Compare implicit owner inputs directly, including every stored byte."""
    from ..geometry._source_boundary_records import SourceBoundaryChartWitness
    from ..geometry.implicit._adaptive_discovery import (
        AdaptiveImplicitSurfaceArchiveRecipe,
    )

    if type(left) not in (
        AdaptiveImplicitSurfaceArchiveRecipe,
        AdaptiveImplicitBoundarySourceArchiveRecipe,
        NativeImplicitBoundaryQueryArchiveRecipe,
        SourceBoundaryChartWitness,
    ) or type(right) is not type(left):
        return False
    if type(left) is SourceBoundaryChartWitness and not left.implicit_cell_keys.size:
        return False
    if left is right:
        return True
    recipe = model_structure_recipe(left, path="compare")
    if _canonical_recipe_payload(recipe) != _canonical_recipe_payload(
        model_structure_recipe(right, path="compare")
    ):
        return False
    left_arrays = _pack_projected_model_array_tree(
        left, recipe, prefix="compare", limits=None
    )
    right_arrays = _pack_projected_model_array_tree(
        right, recipe, prefix="compare", limits=None
    )
    return left_arrays.keys() == right_arrays.keys() and all(
        _same_array_bytes(left_arrays[key], right_arrays[key]) for key in left_arrays
    )


def _register_model_archive_types() -> None:
    """Register built-in scientific models supported by the generic recipe codec."""
    from .._archive_columns import (
        ExactTensorArchiveRecipe,
        IntegerRowsArchiveRecipe,
        StaticColumnArchiveRecipe,
        TextColumnArchiveRecipe,
        TextRowsArchiveRecipe,
    )
    from .._differentiation import (
        DerivativeContract,
        DerivativeRegularity,
        DerivativeRoute,
        DerivativeSurface,
        GradientLevel,
        SurfaceDerivative,
    )
    from .._geometry_predicates import PredicateMode
    from .._identity import SemanticProvenance
    from .._meshcore import PlcSearchBudget
    from .._model._artifacts import register_artifact_value
    from .._physical import SpatialCoordinateContract
    from ..discretization._adaptive_simplex import AdaptiveSimplexStatus
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
    from ..discretization._exact_algebraic_geometry import ExactPeriodicAlgebraicSource
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
        ReferenceCellPartition,
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
    from ..discretization.fem._hp import FiniteElementHPLineage, FiniteElementHPTopology
    from ..discretization.fem._hp_runtime import (
        FiniteElementHPGeometry,
        FiniteElementHPRefinementResult,
    )
    from ..discretization.fem._reference import FiniteElementSpec
    from ..discretization.fem._topology_transfer import FiniteElementTopologyTransfer
    from ..discretization.finite_volume._owner_local_remap import (
        OwnerLocalConservativeRemapEvidence,
        PreparedOwnerLocalConservativeRemap,
    )
    from ..discretization.finite_volume._unstructured_remap import (
        UnstructuredConservativeRemapPlan,
        UnstructuredRemapReport,
    )
    from ..equations._finite_element_material import (
        MaterialCheckpointPayload,
        MaterialSiteId,
        MaterialSiteOwnership,
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
        NestedGeometryOverlapEvidence,
    )
    from ..geometry._compartments import (
        CompartmentDefinition,
        CompartmentInterfaceDefinition,
    )
    from ..geometry._contracts import CompiledGeometry, GeometryTolerance
    from ..geometry._curve_boundary import (
        ParametricCurveBoundarySource,
        SegmentChartCurve,
    )
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
    from ..geometry._nested_restriction_embedding import (
        NestedRestrictionEmbeddingEvidence,
        NestedRestrictionPremise,
    )
    from ..geometry._periodic_mapped_restriction import PeriodicMappedRestrictionEvidence
    from ..geometry._periodic_restriction import PeriodicRestrictionPremise
    from ..geometry._piecewise_linear_boundary import PiecewiseLinearDomainBoundaryOwner
    from ..geometry._planar_coverage import FacetOverlap, PlcSourcePointWitness
    from ..geometry._planar_embedding import PlanarEmbedding
    from ..geometry._source_boundary_records import (
        SourceBoundaryChartCover,
        SourceBoundaryChartWitness,
        SourceBoundaryDistance,
        SourceBoundaryDistanceWitness,
        SourceBoundaryFaceImageWitness,
        SourceBoundaryImagePullback,
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
    from ..geometry.analytic._expressions import _UnionKernel
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
    from ..geometry.brep._projection_contracts import (
        BRepProjectionPolicy,
        BRepProjectionResult,
    )
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
    from ..geometry.multiregion_surface._contracts import (
        MultiRegionSurfaceCapacityEvidence,
        MultiRegionSurfaceCapacityPlan,
        MultiRegionSurfaceCounts,
        MultiRegionSurfaceEvidence,
        MultiRegionSurfaceStatus,
    )
    from ..geometry.multiregion_surface._topology import MultiRegionSurfaceTopology
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
    from ..geometry.reconstruction._sheet import AuthoredTriangleSheetArchiveRecipe
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
    from ..imaging._asset import ImageFieldSpec
    from ..imaging._core import (
        DeidentificationEvidence,
        ImageAxisConvention,
        ImageIndexAffine,
        LabelDefinition,
        LabelOntology,
        VoxelReference,
    )
    from ..interchange._report import (
        AdapterCapability,
        AdapterFormatProfile,
        AdapterLoss,
        AdapterNegotiationResult,
        AdapterReport,
        AdapterRequirement,
        AdapterStatus,
        AdapterWaiver,
    )
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
    from ..measurement._asset import (
        AcquisitionIdentity,
        DataOrigin,
        DataStage,
        DerivationRecord,
    )
    from ..measurement._field import (
        IndependentStandardUncertainty,
        QualityFlag,
        SamplingSemantics,
        SpatialSamplingKind,
    )
    from ..measurement._quantity import QuantitySpec, ValueKind, ValueLayout
    from ..measurement._time import TemporalSampling, TemporalSamplingKind
    from ..meshing._association import (
        AssociationPropagationPolicy,
        GeometryAssociation,
        GeometryAssociationArchiveRecipe,
        GeometryAssociationBankArchiveRecipe,
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
        BlockInterfaceControl,
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
        TransfiniteCurveControl,
        TransfiniteSurfaceControl,
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
    from ..meshing._scope import MeshingEntityKind, MeshingScope, MeshingScopeSide
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
    from ..qualification._reference import ReferenceArtifactManifest
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
    from ._adaptation import (
        HPAdaptationEvidence,
        MarkedMeshAdaptation,
        MeshAdaptationResult,
        MeshAdaptationRoute,
        MeshAdaptationStatus,
    )
    from ._archive_layer_columns import (
        ExactExpressionRowsArchiveRecipe,
        ImplicitChartRowsArchiveRecipe,
        LayerRecordArchiveRecipe,
    )
    from ._assembly import MeshCarrierKind, MeshPart
    from ._audit import CellMeshAuditDisposition, CellMeshAuditPolicy, CellMeshAuditReport
    from ._bisection import BisectionEvidence, BisectionHierarchy
    from ._boundary_layer import (
        BoundaryLayerEvidence,
        BoundaryLayerMesh,
        BoundaryLayerPolicy,
    )
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
    from ._device_metric import DeviceMetricEvidence
    from ._device_tetra_metric import DeviceTetraMetricEvidence
    from ._distribution import (
        MeshDistribution,
        MeshDistributionTransition,
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
    from ._dual_extraction import (
        _RestrictedCoordinateMap,
        SurfaceCrossField,
        SurfaceQuadFieldArchiveRecipe,
    )
    from ._hex_cut_generation import _CutCoordinateMap
    from ._hex_cut_preparation import BalancedCutGridPlan
    from ._hex_dominant import _HexDominantCoordinateMap
    from ._hex_frame_records import HexFrameIntegerLayout
    from ._hex_schedule import NativeHexGridSchedule
    from ._layer_core_carrier_witness import NativeLayerCoreCarrierWitness
    from ._layer_core_root_geometry import LayerCoreRootGeometry
    from ._layer_core_subdivision import (
        NearestWallPointEvidence,
        OverlapSourceStencils,
        RefinedColumnCertification,
        RefinedColumnEvidence,
    )
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
    from ._local_metric import LocalMetricEvidence
    from ._mixed_adaptation import (
        _RetainedCoordinateBank,
        MixedActiveCell,
        MixedAdaptationEvidence,
        MixedAdaptationHierarchy,
        MixedCellContentTransfer,
        MixedForestRoot,
        MixedLayerColumns,
        MixedSibling,
    )
    from ._motion import MeshMotionMonitorPolicy
    from ._optimization import MeshQualityObjective
    from ._periodic import (
        PeriodicPointOrbits,
        PeriodicQuotientHistory,
        PeriodicRefinementEdgeLock,
    )
    from ._plc_source import PiecewiseLinearComplex
    from ._polyhedral_adaptation import (
        _PolyhedralExactCoordinateSource,
        PolyhedralAdaptationEvidence,
        PolyhedralAdaptationHierarchy,
        PolyhedralAdaptationOperation,
    )
    from ._polyhedral_generation import NativePolyhedralSchedule
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
    from ._structured import TransfiniteBlock
    from ._surface_metric import (
        SurfaceMetricGeometryRelation,
        SurfaceMetricGeometryWitness,
        SurfaceMetricOutcome,
        SurfaceMetricStatus,
    )
    from ._sweep import SweepControl, SweepMapKind
    from ._tetra_metric import (
        TetraMetricEvidence,
        TetraMetricStatus,
        TetraMetricUnmetEdge,
        TetraMetricUnmetKind,
    )
    from ._topology_edit import (
        CellTopologyEdit,
        EntityRelations,
        PeriodicTopologyEditWitness,
        PolyhedralTopologyEditBlock,
        PrescribedEntityIds,
        SharedFaceWitnesses,
        TopologyEditBlock,
    )
    from ._trace import (
        MeshingDiagnostic,
        MeshingDiagnosticSeverity,
        MeshingStageKind,
        MeshingStageReport,
        MeshingStageStatus,
        MeshingTrace,
    )
    from ._volume_generation import NativeVolumeSchedule
    from .providers._mmg import (
        MmgAdaptationResult,
        MmgFieldTransfer,
        MmgReference,
        MmgReferenceRetention,
        MmgSessionEvidence,
    )
    from .providers._native_options import NativeMeshingOptions
    from .providers._native_periodic import NativePeriodicSource
    from .providers._native_schedules import NativeCurveSchedule, NativeSurfaceSchedule
    from .providers._native_sources import NativeStructuredSource, NativeSweepSource
    from .providers._omega_h import (
        OmegaHAdaptationEvidence,
        OmegaHAdaptationResult,
        OmegaHClassification,
        OmegaHFieldEvidence,
        OmegaHFieldTransfer,
        OmegaHOptions,
        OmegaHPartition,
        OmegaHTransferredField,
    )

    types = (
        ("phydrax.geometry", CompartmentDefinition),
        ("phydrax.geometry", CompartmentInterfaceDefinition),
        ("phydrax.imaging", ImageFieldSpec),
        ("phydrax.imaging", DeidentificationEvidence),
        ("phydrax.imaging", ImageAxisConvention),
        ("phydrax.imaging", ImageIndexAffine),
        ("phydrax.imaging", LabelDefinition),
        ("phydrax.imaging", LabelOntology),
        ("phydrax.imaging", VoxelReference),
        ("phydrax.measurement", AcquisitionIdentity),
        ("phydrax.measurement", DataOrigin),
        ("phydrax.measurement", DataStage),
        ("phydrax.measurement", DerivationRecord),
        ("phydrax.measurement", IndependentStandardUncertainty),
        ("phydrax.measurement", QualityFlag),
        ("phydrax.measurement", SamplingSemantics),
        ("phydrax.measurement", SpatialSamplingKind),
        ("phydrax.measurement", QuantitySpec),
        ("phydrax.measurement", ValueKind),
        ("phydrax.measurement", ValueLayout),
        ("phydrax.measurement", TemporalSampling),
        ("phydrax.measurement", TemporalSamplingKind),
        ("phydrax.qualification", ReferenceArtifactManifest),
        ("phydrax", DerivativeContract),
        ("phydrax", DerivativeRegularity),
        ("phydrax", DerivativeRoute),
        ("phydrax", DerivativeSurface),
        ("phydrax", GradientLevel),
        ("phydrax", SurfaceDerivative),
        ("phydrax.geometry", CompiledGeometry),
        ("phydrax.geometry", GeometryTolerance),
        ("phydrax.geometry.design", DesignState),
        ("phydrax.geometry.design", ParameterBinding),
        ("phydrax.geometry.design", ParameterId),
        ("phydrax.geometry.design", ParameterSchema),
        ("phydrax.geometry.design", ParameterSpec),
        ("phydrax.geometry.analytic.internal", _BallKernel),
        ("phydrax.geometry.analytic.internal", _TorusKernel),
        ("phydrax.geometry.analytic.internal", _UnionKernel),
        ("phydrax.geometry.implicit", AdaptiveImplicitSurfaceArchiveRecipe),
        ("phydrax.geometry.implicit", AdaptiveImplicitSurfacePolicy),
        ("phydrax.geometry.implicit", ImplicitChartPremise),
        ("phydrax.geometry.implicit", ImplicitChartRestrictionEvidence),
        ("phydrax.geometry.implicit", ImplicitChartRows),
        ("phydrax.geometry.reconstruction", ReconstructionReport),
        ("phydrax.geometry.reconstruction", ReconstructedGeometrySourceArchiveRecipe),
        ("phydrax.geometry.reconstruction", AuthoredTriangleSheetArchiveRecipe),
        ("phydrax.geometry.reconstruction.internal", _ReconstructedTrianglePatch),
        ("phydrax.geometry.reconstruction.internal", SurfaceEnvelopeReproduction),
        ("phydrax.model.archive", AdaptiveImplicitBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", AffinePlcBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", AffinePlcDomainArchiveRecipe),
        ("phydrax.model.archive", AffinePlcGeometrySourceArchiveRecipe),
        ("phydrax.model.archive", LabelVolumeArchiveRecipe),
        ("phydrax.model.archive", CompartmentMeshingSourceArchiveRecipe),
        ("phydrax.model.archive", ImageFieldSpecArchiveRecipe),
        ("phydrax.model.archive", ImageIndexAffineArchiveRecipe),
        ("phydrax.model.archive", ImageQualityFlagArchiveRecipe),
        ("phydrax.model.archive", ImageUncertaintyArchiveRecipe),
        ("phydrax.model.archive", MeshingDomainArchiveRecipe),
        ("phydrax.model.archive", MeshingDomainBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", NativeImplicitBoundaryQueryArchiveRecipe),
        ("phydrax.model.archive", NativeImplicitSourceArchiveRecipe),
        ("phydrax.model.archive", AnalyticImplicitProfileArchiveRecipe),
        ("phydrax.model.archive", ParametricCurveBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", ReconstructedTriangleBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", PiecewiseLinearComplexArchiveRecipe),
        ("phydrax.model.archive", SurfaceSourceChartsArchiveRecipe),
        ("phydrax.model.archive", PreparedSurfaceSourceSupportArchiveRecipe),
        ("phydrax.model.archive", SurfaceNativeRestrictionBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", SurfaceChartBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", SurfaceSubdivisionBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", SphereMaterialCellAtlasArchiveRecipe),
        ("phydrax.model.archive", PreparedSphereChartDeformationArchiveRecipe),
        ("phydrax.model.archive", SurfaceEnvelopeSourceArchiveRecipe),
        ("phydrax.model.archive", PiecewiseLinearBoundarySourceArchiveRecipe),
        ("phydrax.model.archive", PlcCurveAuthorityArchiveRecipe),
        ("phydrax.model.archive", NativePlcSourceArchiveRecipe),
        ("phydrax.model.archive", NativePlanarSourceArchiveRecipe),
        ("phydrax.model.archive", NativePlanarMotionSourceArchiveRecipe),
        ("phydrax.model.archive", PlanarMotionAdvanceArchiveRecipe),
        ("phydrax.model.archive", NativeLayerCoreAuthoringArchiveRecipe),
        ("phydrax.model.archive", LayerCoreAcceptedResultArchiveRecipe),
        ("phydrax.model.archive", LayerCoreAcceptedAdaptationArchiveRecipe),
        ("phydrax.model.archive", FiniteElementHPEpochArchiveRecipe),
        ("phydrax.model.archive", LayerCoreSourceReferenceArchiveRecipe),
        ("phydrax.model.archive", LayerCoreFidelitySourceReferenceArchiveRecipe),
        ("phydrax.model.archive", SourceBoundaryImagePullbackArchiveRecipe),
        ("phydrax.model.archive", SourceBoundaryFaceImageWitnessArchiveRecipe),
        ("phydrax.model.archive", NativeLayerMappedBoundaryArchiveRecipe),
        ("phydrax.model.archive", MixedAdaptationPolicyArchiveRecipe),
        ("phydrax.model.archive", OversetRegistrationArchiveRecipe),
        ("phydrax.model.archive", NativeMappedHexSourceArchiveRecipe),
        ("phydrax.model.archive", MappedCoreFacetAuthorityArchiveRecipe),
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
        ("phydrax.geometry", MultiRegionSurfaceCapacityEvidence),
        ("phydrax.geometry", MultiRegionSurfaceCapacityPlan),
        ("phydrax.geometry", MultiRegionSurfaceCounts),
        ("phydrax.geometry", MultiRegionSurfaceEvidence),
        ("phydrax.geometry", MultiRegionSurfaceStatus),
        ("phydrax.geometry", MultiRegionSurfaceTopology),
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
        ("phydrax.geometry", SegmentChartCurve),
        ("phydrax.geometry", PatchCurveUse),
        ("phydrax.geometry", PatchPoleUse),
        ("phydrax.geometry", PiecewiseLinearDomain),
        ("phydrax.geometry", PiecewiseLinearDomainBoundaryOwner),
        ("phydrax.geometry", SourceBoundaryChartCover),
        ("phydrax.geometry", SourceBoundaryChartWitness),
        ("phydrax.geometry", SourceBoundaryFaceImageWitness),
        ("phydrax.geometry", SourceBoundaryImagePullback),
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
        ("phydrax.geometry.brep", BRepProjectionPolicy),
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
        ("phydrax.discretization", ExactPeriodicAlgebraicSource),
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
        ("phydrax.meshing.internal", _HexDominantCoordinateMap),
        ("phydrax.meshing", NativeHexGridSchedule),
        ("phydrax.meshing", HexFrameIntegerLayout),
        ("phydrax.meshing.internal", _RestrictedCoordinateMap),
        ("phydrax.meshing.internal", _ExactCoordinateBankMap),
        ("phydrax.meshing.internal", _RetainedCoordinateBank),
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
        ("phydrax.discretization.fem", FiniteElementHPLineage),
        ("phydrax.discretization.fem", FiniteElementHPTopology),
        ("phydrax.discretization.fem", FiniteElementHPGeometry),
        ("phydrax.discretization.fem", FiniteElementHPRefinementResult),
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
        ("phydrax.equations", MaterialCheckpointPayload),
        ("phydrax.equations", MaterialSiteId),
        ("phydrax.equations", MaterialSiteOwnership),
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
        ("phydrax.geometry", NestedGeometryOverlapEvidence),
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
        ("phydrax.geometry", NestedRestrictionEmbeddingEvidence),
        ("phydrax.geometry", NestedRestrictionPremise),
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
        ("phydrax.meshing", NativeLayerCoreCarrierWitness),
        ("phydrax.meshing", LayerCoreRootGeometry),
        ("phydrax.meshing", NearestWallPointEvidence),
        ("phydrax.meshing", OverlapSourceStencils),
        ("phydrax.meshing", RefinedColumnCertification),
        ("phydrax.meshing", RefinedColumnEvidence),
        ("phydrax.meshing", BoundaryLayerCollisionPolicy),
        ("phydrax.meshing", BoundaryLayerEvidence),
        ("phydrax.meshing", BoundaryLayerMesh),
        ("phydrax.meshing", BoundaryLayerPolicy),
        ("phydrax.meshing", MeshMotionMonitorPolicy),
        ("phydrax.meshing", MeshQualityObjective),
        ("phydrax.meshing", SimplexForestRecords),
        ("phydrax.meshing", PeriodicSimplexForest),
        ("phydrax.meshing", OwnerLocalMigrationState),
        ("phydrax.meshing", CellMeshTransition),
        ("phydrax.meshing", SurfaceQuadFieldArchiveRecipe),
        ("phydrax.meshing", SurfaceCrossField),
        ("phydrax.meshing", EntityLineage),
        ("phydrax.meshing", EntityLineageKind),
        ("phydrax.meshing", MeshLineage),
        ("phydrax.meshing", MeshTransitionKind),
        ("phydrax.meshing", VertexInterpolationStencil),
        ("phydrax.meshing", MixedActiveCell),
        ("phydrax.meshing", MixedAdaptationHierarchy),
        ("phydrax.meshing", MixedLayerColumns),
        ("phydrax.meshing", MixedAdaptationEvidence),
        ("phydrax.meshing", MixedCellContentTransfer),
        ("phydrax.meshing", MeshAdaptationResult),
        ("phydrax.meshing", HPAdaptationEvidence),
        ("phydrax.meshing", MeshAdaptationRoute),
        ("phydrax.meshing", MixedForestRoot),
        ("phydrax.meshing", MixedSibling),
        ("phydrax.discretization", ReferenceCellPartition),
        ("phydrax.meshing", MarkedMeshAdaptation),
        ("phydrax.meshing", MeshAdaptationStatus),
        ("phydrax.meshing", BisectionEvidence),
        ("phydrax.meshing", BisectionHierarchy),
        ("phydrax.meshing", LocalMetricEvidence),
        ("phydrax.meshing", DeviceMetricEvidence),
        ("phydrax.discretization", AdaptiveSimplexStatus),
        ("phydrax.meshing", TetraMetricEvidence),
        ("phydrax.meshing", TetraMetricStatus),
        ("phydrax.meshing", TetraMetricUnmetEdge),
        ("phydrax.meshing", TetraMetricUnmetKind),
        ("phydrax.meshing", DeviceTetraMetricEvidence),
        ("phydrax.meshing", MeshDistributionTransition),
        ("phydrax.meshing", MmgAdaptationResult),
        ("phydrax.meshing", MmgFieldTransfer),
        ("phydrax.meshing", MmgReference),
        ("phydrax.meshing", MmgReferenceRetention),
        ("phydrax.meshing", MmgSessionEvidence),
        ("phydrax.meshing", OmegaHAdaptationResult),
        ("phydrax.meshing", OmegaHAdaptationEvidence),
        ("phydrax.meshing", OmegaHOptions),
        ("phydrax.meshing", OmegaHFieldEvidence),
        ("phydrax.meshing", OmegaHFieldTransfer),
        ("phydrax.meshing", OmegaHClassification),
        ("phydrax.meshing", OmegaHPartition),
        ("phydrax.meshing", OmegaHTransferredField),
        ("phydrax.meshing", SurfaceMetricOutcome),
        ("phydrax.meshing", SurfaceMetricGeometryWitness),
        ("phydrax.meshing", SurfaceMetricStatus),
        ("phydrax.meshing", SurfaceMetricGeometryRelation),
        ("phydrax.meshing", CellTopologyEdit),
        ("phydrax.meshing", TopologyEditBlock),
        ("phydrax.meshing", PolyhedralTopologyEditBlock),
        ("phydrax.meshing", EntityRelations),
        ("phydrax.meshing", PrescribedEntityIds),
        ("phydrax.meshing", SharedFaceWitnesses),
        ("phydrax.meshing", PeriodicTopologyEditWitness),
        ("phydrax.interchange", AdapterReport),
        ("phydrax.interchange", AdapterStatus),
        ("phydrax.interchange", AdapterLoss),
        ("phydrax.interchange", AdapterFormatProfile),
        ("phydrax.interchange", AdapterRequirement),
        ("phydrax.interchange", AdapterCapability),
        ("phydrax.interchange", AdapterWaiver),
        ("phydrax.interchange", AdapterNegotiationResult),
        ("phydrax.meshing", PolyhedralAdaptationEvidence),
        ("phydrax.meshing", PolyhedralAdaptationHierarchy),
        ("phydrax.meshing", PolyhedralAdaptationOperation),
        ("phydrax.meshing.internal", _PolyhedralExactCoordinateSource),
        (
            "phydrax.discretization.finite_volume",
            UnstructuredConservativeRemapPlan,
        ),
        ("phydrax.discretization.finite_volume", UnstructuredRemapReport),
        ("phydrax.meshing", PeriodicQuotientHistory),
        ("phydrax.meshing", PeriodicRefinementEdgeLock),
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
        ("phydrax.meshing", BlockInterfaceControl),
        ("phydrax.meshing", TransfiniteCurveControl),
        ("phydrax.meshing", TransfiniteSurfaceControl),
        ("phydrax.meshing", TransfiniteBlock),
        ("phydrax.meshing", NativeStructuredSource),
        ("phydrax.meshing", SweepControl),
        ("phydrax.meshing", SweepMapKind),
        ("phydrax.meshing", NativeSweepSource),
        ("phydrax.meshing", ProtectedFeature),
        ("phydrax.meshing", MeshingEntityKind),
        ("phydrax.meshing", MeshingScope),
        ("phydrax.meshing", MeshingScopeSide),
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
        ("phydrax.meshing", GeometryAssociationBankArchiveRecipe),
        ("phydrax.meshing", AssociationPropagationPolicy),
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
        ("phydrax.meshing", NativeMeshingOptions),
        ("phydrax.meshing", NativePolyhedralSchedule),
        ("phydrax.meshing", NativeVolumeSchedule),
        ("phydrax._meshcore", PlcSearchBudget),
        ("phydrax.meshing", NativeCurveSchedule),
        ("phydrax.meshing", NativeSurfaceSchedule),
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
        ("phydrax.solver", FiniteElementFieldBanksArchiveRecipe),
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


def _rebuild_compiled_analytic_geometry(value: object, /) -> Any:
    import numpy as np

    from .._fingerprint import array_tree_fingerprint
    from ..geometry._contracts import CompiledGeometry, GeometryKernel, GeometryTolerance
    from ..geometry.analytic._expressions import _UnionKernel
    from ..geometry.analytic._extended import _TorusKernel
    from ..geometry.analytic._primitives import _BallKernel
    from ..geometry.design._schema import (
        DesignState,
        ParameterBinding,
        ParameterId,
        ParameterSchema,
        ParameterSpec,
    )

    if not isinstance(value, CompiledGeometry):
        raise TypeError("Analytic implicit profile requires a compiled geometry owner.")
    stored_schema = value.state.schema
    if value.schema != stored_schema:
        raise ValueError(
            "Compiled analytic geometry has inconsistent state and schema owners."
        )
    schema = ParameterSchema(
        tuple(
            ParameterSpec(
                ParameterId(spec.parameter_id.feature_id, spec.parameter_id.name),
                spec.shape,
                spec.dtype,
                spec.role,
                spec.physical_scale,
                spec.bounds,
                spec.trainable,
            )
            for spec in stored_schema.specs
        )
    )

    def rebuild_binding(binding: ParameterBinding, /) -> ParameterBinding:
        parameter_id = ParameterId(
            binding.parameter_id.feature_id,
            binding.parameter_id.name,
        )
        if (
            binding.index < 0
            or binding.index >= len(schema.specs)
            or schema.specs[binding.index].parameter_id != parameter_id
        ):
            raise ValueError(
                "Compiled analytic parameter bindings differ from their schema."
            )
        return ParameterBinding(parameter_id, binding.index)

    def rebuild_kernel(kernel_owner: GeometryKernel, /) -> GeometryKernel:
        if type(kernel_owner) is _BallKernel:
            return _BallKernel(
                rebuild_binding(kernel_owner.center),
                rebuild_binding(kernel_owner.radius),
                dimension=kernel_owner.dimension,
                source_id=kernel_owner.source_id,
            )
        if type(kernel_owner) is _TorusKernel:
            return _TorusKernel(
                rebuild_binding(kernel_owner.center),
                rebuild_binding(kernel_owner.major),
                rebuild_binding(kernel_owner.minor),
                rebuild_binding(kernel_owner.angle),
                full=kernel_owner.full,
                source_id=kernel_owner.source_id,
            )
        if type(kernel_owner) is _UnionKernel:
            if type(kernel_owner.children) is not tuple or len(kernel_owner.children) < 2:
                raise ValueError(
                    "Compiled union must retain at least two geometry children."
                )
            children = tuple(rebuild_kernel(child) for child in kernel_owner.children)
            if any(
                child.ambient_dimension != children[0].ambient_dimension
                for child in children
            ):
                raise ValueError(
                    "Compiled union children must share an ambient dimension."
                )
            return _UnionKernel(children)
        raise TypeError("Analytic implicit profile uses an unsupported geometry kernel.")

    kernel = rebuild_kernel(value.kernel)
    state = DesignState(schema, value.state.values)
    tolerance = GeometryTolerance(value.tolerance.absolute, value.tolerance.relative)
    geometry = CompiledGeometry(kernel, state, tolerance=tolerance)
    if array_tree_fingerprint(geometry) != array_tree_fingerprint(value):
        raise ValueError("Compiled analytic source fields differ after reconstruction.")
    if not bool(np.asarray(geometry.validity().accepted)):
        raise ValueError(
            "Reconstructed analytic geometry does not have accepted validity."
        )
    return geometry


def _restore_owner_local_dependencies(
    value: Any, restore: Callable[[Any], Any], /
) -> Any:
    """Rebuild executable dependencies before the owner replays geometry."""
    import jax

    from .._model._structure import (
        _FiniteElementSpecArchiveProjection,
        SparseCoordinateOperatorArchiveRecipe,
    )
    from ..linalg._operator_pairing import OperatorPairingArchiveRecipe
    from ..linalg._spaces import ArraySpaceArchiveRecipe
    from ._distribution_forest import PeriodicSimplexForest

    recipe_types = (
        ArraySpaceArchiveRecipe,
        _FiniteElementSpecArchiveProjection,
        OperatorPairingArchiveRecipe,
        SparseCoordinateOperatorArchiveRecipe,
        PeriodicSimplexForest,
    )
    return jax.tree_util.tree_map(
        lambda node: restore(node) if type(node) in recipe_types else node,
        value,
        is_leaf=lambda node: type(node) in recipe_types,
    )


def restore_model_archive_value(
    value: Any,
    /,
    *,
    execution_group: Any | None = None,
) -> Any:
    """Rebuild projected FE/linalg models through their canonical constructors."""
    import inspect
    from collections.abc import Mapping

    import numpy as np

    from .._meshcore import PlcSearchBudget
    from .._model._structure import _FiniteElementSpecArchiveProjection
    from .._physical import SpatialCoordinateContract
    from ..discretization._cell_geometry_transition_types import (
        CellGeometryTransition,
        CellGeometryTransitionEvidence,
    )
    from ..discretization._cell_mesh import CellMesh
    from ..discretization._exact_algebraic_geometry import ExactPeriodicAlgebraicSource
    from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
    from ..discretization._mapped_power_geometry import MappedPowerCellGeometrySource
    from ..discretization._nested_reference_types import (
        ReferenceCellMapWitnesses,
        ReferenceCellPartition,
        validate_reference_cell_map_witnesses,
        validate_reference_cell_partition,
    )
    from ..discretization.fem._de_rham import (
        _distribute_transfer_result,
        DistributedFiniteElementTransfer,
        FiniteElementTransferResult,
        PeriodicFiniteElementTransfer,
        SparseCommutingEvidence,
        SparseTransferEntryEvidence,
    )
    from ..discretization.fem._hp import FiniteElementHPLineage, FiniteElementHPTopology
    from ..discretization.fem._hp_runtime import (
        FiniteElementHPGeometry,
        FiniteElementHPRefinementResult,
    )
    from ..equations._finite_element_material import (
        MaterialCheckpointPayload,
        MaterialSiteId,
        MaterialSiteOwnership,
        MaterialState,
        MaterialTransaction,
    )
    from ..geometry._affine_plc_boundary import AffinePlcBoundarySource
    from ..geometry._affine_plc_domain import AffinePlcDomain
    from ..geometry._affine_plc_geometry import AffinePlcGeometrySource
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
    from ..geometry._compartments import CompartmentMeshingSource
    from ..geometry._curve_boundary import (
        ParametricCurveBoundarySource,
        SegmentChartCurve,
    )
    from ..geometry._domain_boundary import MeshingDomainBoundarySource
    from ..geometry._domain_coverage import PiecewiseLinearDomain
    from ..geometry._planar_coverage import PlcSourcePointWitness
    from ..geometry._planar_embedding import PlanarEmbedding
    from ..geometry._source_boundary_records import SurfaceEnvelopeDistanceWitness
    from ..geometry._surface_source_archive import (
        _domain,
        _geometry,
        restore_surface_source_charts,
    )
    from ..geometry._surface_source_support import (
        PreparedSurfaceSourceSupport,
        SurfaceNativeRestrictionBoundarySource,
        SurfaceSourceCharts,
        SurfaceSourceRootAtlas,
    )
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
    from ..geometry.reconstruction._core import ReconstructedGeometrySourceArchiveRecipe
    from ..geometry.reconstruction._sheet import AuthoredTriangleSheetArchiveRecipe
    from ..geometry.simplicial._regions import PlanarMeshRegion, SegmentMesh
    from ..geometry.surface._model import SurfaceModel
    from ..imaging._asset import ImageFieldSpec
    from ..imaging._core import ImageIndexAffine, LabelVolume, MedicalImageAsset
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
    from ..measurement._field import (
        IndependentStandardUncertainty,
        QualityFlag,
        SamplingSemantics,
    )
    from ..measurement._quantity import QuantitySpec, ValueLayout
    from ..measurement._time import TemporalSampling
    from ..meshing._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
    from ..meshing._controls import PatchControl
    from ..meshing._scope import MeshingScope, MeshingScopeSide
    from ..sparse._linear import SparseCoordinateOperator
    from ..sparse._relation import EdgeRelation
    from ..units._unit import UnitDefinition
    from ._adaptation import (
        HPAdaptationEvidence,
        MarkedMeshAdaptation,
        MeshAdaptationResult,
    )
    from ._controls import (
        BlockInterfaceControl,
        LayerSchedule,
        TransfiniteCurveControl,
        TransfiniteSurfaceControl,
    )
    from ._distribution_forest import PeriodicSimplexForest, SimplexForestRecords
    from ._distribution_migration import OwnerLocalMigrationState
    from ._mixed_adaptation import (
        _require_retained_bank_origin,
        MixedAdaptationHierarchy,
        MixedLayerColumns,
    )
    from ._motion_source import NativePlanarMotionSource
    from ._periodic import PeriodicQuotientHistory, PeriodicRefinementEdgeLock
    from ._plc_source import PiecewiseLinearComplex
    from ._polyhedral_generation import NativePolyhedralSchedule
    from ._result import CellMeshingResult
    from ._structured import TransfiniteBlock
    from ._sweep import SweepControl
    from ._volume_generation import NativeVolumeSchedule
    from .providers._native_options import NativeMeshingOptions
    from .providers._native_schedules import NativeCurveSchedule, NativeSurfaceSchedule
    from .providers._native_sources import (
        NativePlanarSource,
        NativePlcSource,
        NativeStructuredSource,
        NativeSweepSource,
    )

    _register_model_archive_types()
    in_composition = False
    pending_graphs: list[Any] = []
    pending_selections: list[Any] = []
    pending_sided_scopes: list[MeshingScope] = []

    def same_content(left: Any, right: Any) -> bool:
        left = model_archive_root_projection(left)
        right = model_archive_root_projection(right)
        left_recipe = model_structure_recipe(left)
        if _canonical_recipe_payload(left_recipe) != _canonical_recipe_payload(
            model_structure_recipe(right)
        ):
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
        if type(item) is FiniteElementHPEpochArchiveRecipe:
            return _restore_hp_epoch_archive_recipe(item, restore)
        if type(item) in (
            LayerCoreAcceptedResultArchiveRecipe,
            LayerCoreAcceptedAdaptationArchiveRecipe,
            NativeLayerCoreAuthoringArchiveRecipe,
            OversetRegistrationArchiveRecipe,
        ):
            # These are declarative replay inputs; their endpoint owners are
            # reconstructed together by the generation/adaptation replay APIs.
            return item
        from ._dual_extraction import (
            restore_surface_quad_field_event,
            SurfaceCrossField,
            SurfaceQuadFieldArchiveRecipe,
        )

        if type(item) is SurfaceQuadFieldArchiveRecipe:
            item = restore_row_record_recipes(item)
            source = CellMesh.from_stored_fields(item.source)
            geometry = _geometry(item.source_geometry, source)
            return restore_surface_quad_field_event(item, source, geometry)
        if type(item) is SurfaceCrossField:
            return SurfaceCrossField.from_stored_fields(item)
        if type(item) is SurfaceMeshingSpec:
            return item if in_composition else SurfaceMeshingSpec.from_stored_fields(item)
        from ..geometry._mapped_reference_domain import MappedReferenceDomain

        if type(item) is MappedReferenceDomain:
            from ..geometry._surface_source_archive import _mapped_reference

            return _mapped_reference(item)
        from ..geometry.implicit._adaptive_discovery import (
            AdaptiveImplicitBoundarySource,
            AdaptiveImplicitSurfaceArchiveRecipe,
            restore_adaptive_implicit_surface,
        )
        from ..geometry.implicit._policy import AdaptiveImplicitSurfacePolicy

        if type(item) is AdaptiveImplicitSurfaceArchiveRecipe:
            geometry = _rebuild_compiled_analytic_geometry(item.geometry)
            stored_policy = item.policy
            if type(stored_policy) is not AdaptiveImplicitSurfacePolicy:
                raise TypeError("Adaptive implicit source lost its discovery policy.")
            policy = AdaptiveImplicitSurfacePolicy(
                enclosure=stored_policy.enclosure,
                relative_cell_diameter=stored_policy.relative_cell_diameter,
                root_tolerance=stored_policy.root_tolerance,
                allow_approximate_zero_set=stored_policy.allow_approximate_zero_set,
                allow_nonsmooth_field=stored_policy.allow_nonsmooth_field,
            )
            if not _same_model_content(policy, stored_policy):
                raise ValueError("Stored adaptive implicit policy changed.")
            return restore_adaptive_implicit_surface(item, geometry, policy)
        if type(item) is AdaptiveImplicitBoundarySourceArchiveRecipe:
            surface = restore(item.surface)
            query = AdaptiveImplicitBoundarySource(
                surface, frame_id=item.frame_id or None
            )
            if (
                query.source_id != item.expected_source_id
                or query.source_revision != item.expected_source_revision
                or query.frame_id != item.frame_id
            ):
                raise ValueError("Stored adaptive implicit query changed.")
            query.require_current()
            return query
        if type(item) is AnalyticImplicitProfileArchiveRecipe:
            from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile

            geometry = _rebuild_compiled_analytic_geometry(item.geometry)
            contract = SpatialCoordinateContract.from_dict(
                item.coordinate_contract.to_dict()
            )
            profile = AnalyticImplicitProfile(
                geometry,
                contract,
                tube_radius=item.tube_radius,
                cover_radius=item.cover_radius,
                source_id=item.source_id,
                source_revision=item.source_revision,
            )
            if profile.profile_id != item.expected_profile_id:
                raise ValueError("Stored analytic implicit profile changed.")
            return profile
        if type(item) is NativeImplicitBoundaryQueryArchiveRecipe:
            from .providers._native_implicit import _NativeImplicitBoundaryQuery

            query = _NativeImplicitBoundaryQuery(
                restore(item.query),
                item.source_id,
                item.source_revision,
                item.frame_id,
                authoring_profile=restore(item.authoring_profile),
            )
            if query.query_id != item.expected_query_id:
                raise ValueError("Stored native implicit query changed.")
            query.require_current()
            return query
        if type(item) is NativeImplicitSourceArchiveRecipe:
            from ..discretization._tensor_support import PreparedTensorGrid
            from .providers._native_sources import NativeImplicitSource

            grid = item.grid
            if grid is not None:
                if type(grid) is not PreparedTensorGrid:
                    raise TypeError("Native implicit source lost its tensor grid owner.")
                grid = PreparedTensorGrid(
                    grid.axes,
                    axis_names=grid.axis_names,
                    plan_id=grid.plan_id,
                    prepared_id=grid.prepared_id,
                )
                if not _same_model_content(grid, item.grid):
                    raise ValueError("Stored native implicit grid changed.")
            if item.discovery is not None and (
                type(item.discovery) is not AdaptiveImplicitSurfaceArchiveRecipe
            ):
                raise TypeError("Native implicit source lost its discovery recipe.")
            if item.authoring_profile is not None and (
                type(item.authoring_profile) is not AnalyticImplicitProfileArchiveRecipe
            ):
                raise TypeError("Native implicit source lost its authoring profile.")
            discovery = restore(item.discovery)
            source = NativeImplicitSource(
                _rebuild_compiled_analytic_geometry(item.geometry),
                item.source_id,
                item.source_revision,
                grid=grid,
                discovery=discovery,
                authoring_profile=restore(item.authoring_profile),
            )
            if source.binding_id != item.expected_binding_id:
                raise ValueError("Stored native implicit source changed.")
            return source
        if type(item) is LayerSchedule:
            return LayerSchedule.from_stored_fields(item)
        if type(item) is SweepControl:
            schedule = LayerSchedule.from_stored_fields(item.schedule)
            from ._sweep import SweepMapKind

            frames = item.frames if item.kind is SweepMapKind.FRAMES else None
            translations = item.translations if item.kind is SweepMapKind.FRAMES else None
            rebuilt = SweepControl(
                item.kind,
                schedule,
                item.map_id,
                origin=item.origin,
                axis=item.axis,
                frames=frames,
                translations=translations,
                closed=item.closed,
                tolerance=item.tolerance,
            )
            if not _same_model_content(rebuilt, item):
                raise ValueError("Stored sweep control changed.")
            return rebuilt
        if type(item) is SurfaceEnvelopeDistanceWitness:
            # The kernel-run record is evidence of the writer process only; a
            # restored witness reproduces its envelope by rerunning the kernel.
            envelope_witness = SurfaceEnvelopeDistanceWitness(
                mesh_id=item.mesh_id,
                geometry_id=item.geometry_id,
                topology_id=item.topology_id,
                source_id=item.source_id,
                source_revision=item.source_revision,
                source_query_id=item.source_query_id,
                offset=item.offset,
                spacing=item.spacing,
                certificate_spacing=item.certificate_spacing,
                envelope_mode=item.envelope_mode,
                bounds=np.asarray(item.bounds),
                envelope_vertices=np.asarray(item.envelope_vertices),
                envelope_triangles=np.asarray(item.envelope_triangles),
                target_vertex_global_ids=np.asarray(item.target_vertex_global_ids),
                target_facet_global_ids=np.asarray(item.target_facet_global_ids),
                derived_source=None
                if item.derived_source is None
                else restore(item.derived_source),
                derived_witness=None
                if item.derived_witness is None
                else restore(item.derived_witness),
            )
            if envelope_witness.witness_id != item.witness_id:
                raise ValueError("Stored surface-envelope witness changed.")
            return envelope_witness
        if type(item) is AuthoredTriangleSheetArchiveRecipe:
            from ..geometry.reconstruction._sheet import AuthoredTriangleSheet

            sheet = AuthoredTriangleSheet(
                np.asarray(item.vertices, dtype=np.float64),
                np.asarray(item.triangles, dtype=np.int64),
                feature_id=item.feature_id,
            )
            if (sheet.source_id, sheet.source_revision) != (
                item.expected_source_id,
                item.expected_source_revision,
            ):
                raise ValueError("Stored authored triangle sheet changed its identity.")
            return sheet
        if type(item) is ReconstructedGeometrySourceArchiveRecipe:
            from ..geometry.reconstruction._core import ReconstructedGeometrySource

            reconstructed = ReconstructedGeometrySource(
                restore(item.source),
                restore(item.report),
                persistent_vertex_ids=np.asarray(
                    item.persistent_vertex_ids, dtype=np.int64
                ),
                persistent_edge_ids=np.asarray(item.persistent_edge_ids, dtype=np.int64),
                persistent_facet_ids=np.asarray(
                    item.persistent_facet_ids, dtype=np.int64
                ),
                persistent_region_id=item.persistent_region_id,
            )
            if (reconstructed.source_id, reconstructed.source_revision) != (
                item.expected_source_id,
                item.expected_source_revision,
            ):
                raise ValueError("Stored reconstructed geometry source changed.")
            return reconstructed
        if type(item) is ReconstructedTriangleBoundarySourceArchiveRecipe:
            from ..geometry.reconstruction._boundary import (
                ReconstructedTriangleBoundarySource,
            )

            triangle_query = ReconstructedTriangleBoundarySource(restore(item.source))
            if (triangle_query.source_id, triangle_query.source_revision) != (
                item.expected_source_id,
                item.expected_source_revision,
            ):
                raise ValueError("Stored reconstructed triangle query changed.")
            return triangle_query
        if type(item) is SurfaceEnvelopeSourceArchiveRecipe:
            from ..geometry.reconstruction._envelope import parse_envelope_mode
            from ._surface_envelope import SurfaceEnvelopeSource

            envelope = SurfaceEnvelopeSource(
                restore(item.original),
                source_id=item.source_id,
                coordinate_contract=restore(item.coordinate_contract),
                offset=item.offset,
                spacing=item.spacing,
                certificate_spacing=item.certificate_spacing,
                maximum_deviation=item.maximum_deviation,
                envelope_mode=parse_envelope_mode(item.envelope_mode),
            )
            if (envelope.source_revision, envelope.envelope_id) != (
                item.expected_source_revision,
                item.expected_envelope_id,
            ):
                raise ValueError("Stored surface envelope does not recompute.")
            return envelope
        if type(item) is PiecewiseLinearBoundarySourceArchiveRecipe:
            from ..geometry._piecewise_linear_boundary import (
                PiecewiseLinearBoundarySource,
                PiecewiseLinearDomainBoundaryOwner,
            )

            if type(item.owner) in (
                PiecewiseLinearDomainBoundaryOwner,
                PlcCurveAuthorityArchiveRecipe,
                SurfaceEnvelopeSourceArchiveRecipe,
            ):
                owner = restore(item.owner)
                query = PiecewiseLinearBoundarySource(
                    owner,
                    item.entity_dimension,
                    entity_ids=item.entity_ids,
                )
                if query.source_query_id != item.expected_source_query_id:
                    raise ValueError("Stored PWL domain query changed.")
                query.require_current()
                return query
        if type(item) is PlcCurveAuthorityArchiveRecipe:
            from .providers._native_plc_geometry import _PlcCurveAuthority

            owner = _PlcCurveAuthority(
                restore(item.source), restore(item.reference_domain)
            )
            if owner.domain.domain_id != item.expected_domain_id:
                raise ValueError("Stored PLC curve authority changed its exact domain.")
            return owner
        from ..geometry._piecewise_linear_boundary import (
            PiecewiseLinearDomainBoundaryOwner,
        )

        if type(item) is PiecewiseLinearDomainBoundaryOwner:
            owner = PiecewiseLinearDomainBoundaryOwner(restore(item.domain))
            if not _same_model_content(owner, item):
                raise ValueError("Stored PWL domain boundary owner changed.")
            return owner
        if type(item) is NativeSweepSource:
            from ..discretization._cell_geometry import CellGeometryRestrictionSource

            profile = CellMesh.from_stored_fields(item.profile)
            geometry = _geometry(item.profile_geometry, profile)
            domain = restore(item.domain)
            from ..geometry._mapped_reference_boundary import (
                MappedReferenceBoundarySource,
            )
            from ..geometry._piecewise_linear_boundary import (
                PiecewiseLinearBoundarySource,
            )

            query = restore(item.fidelity_source)
            if (
                type(query)
                in (PiecewiseLinearBoundarySource, MappedReferenceBoundarySource)
                and domain is not None
            ):
                if not _same_model_content(query.domain, domain):
                    raise ValueError("Stored sweep query binds a different domain.")
                domain = query.domain
            restriction = item.root_correspondence
            if restriction is not None:
                if type(domain) is not MappedReferenceDomain:
                    raise TypeError(
                        "Sweep root correspondence requires its mapped domain."
                    )
                maps = {}
                for name, witness in restriction.block_parent_reference_maps.items():
                    if witness is None:
                        maps[name] = None
                        continue
                    if type(witness) is not ReferenceCellMapWitnesses:
                        raise TypeError("Sweep restriction lost its reference maps.")
                    rebuilt_witness = ReferenceCellMapWitnesses(
                        witness.target_cell_ids,
                        witness.source_cell_ids,
                        witness.target_cell_kind,
                        witness.source_cell_kind,
                        witness.source_reference_vertices,
                        witness.source_reference_exact_vertices,
                        witness.source_reference_chart_arguments,
                    )
                    validate_reference_cell_map_witnesses(rebuilt_witness)
                    maps[name] = rebuilt_witness
                carrier = (
                    None
                    if restriction.carrier_geometry is None
                    else _geometry(restriction.carrier_geometry, domain.reference_mesh)
                )
                rebuilt_restriction = CellGeometryRestrictionSource(
                    restriction.source_geometry_id,
                    restriction.source_topology_id,
                    restriction.block_parent_cell_ids,
                    restriction.block_parent_vertex_ids,
                    block_source_blocks=restriction.block_source_blocks,
                    carrier_publication=restriction.carrier_publication,
                    block_parent_reference_maps=maps,
                    carrier_geometry=carrier,
                )
                if not _same_model_content(rebuilt_restriction, restriction):
                    raise ValueError("Stored sweep root correspondence changed.")
                restriction = rebuilt_restriction
            rebuilt = NativeSweepSource(
                profile,
                restore(item.control),
                item.source_id,
                item.source_revision,
                fidelity_source=query,
                maximum_deviation=item.maximum_deviation,
                domain=domain,
                block_regions=item.block_regions,
                domain_vertex_map=item.domain_vertex_map,
                profile_geometry=geometry,
                root_correspondence=restriction,
            )
            if not _same_model_content(rebuilt, item):
                raise ValueError("Stored sweep source changed.")
            return rebuilt
        if type(item) in (
            TransfiniteCurveControl,
            TransfiniteSurfaceControl,
            BlockInterfaceControl,
            TransfiniteBlock,
            NativeStructuredSource,
        ):
            if type(item) is TransfiniteCurveControl:
                curve = (
                    restore(item.curve)
                    if type(item.curve) is SegmentChartCurve
                    else _reconstruct_brep_graph(item.curve)
                )
                rebuilt = TransfiniteCurveControl(
                    curve,
                    item.source_id,
                    item.parameter_range,
                    item.intervals,
                    reversed=item.reversed,
                )
            elif type(item) is TransfiniteSurfaceControl:
                rebuilt = TransfiniteSurfaceControl(
                    _reconstruct_brep_graph(item.surface),
                    item.source_id,
                    item.parameter_box,
                    item.intervals,
                    permutation=item.permutation,
                    flips=item.flips,
                )
            elif type(item) is BlockInterfaceControl:
                rebuilt = BlockInterfaceControl(
                    item.first_block,
                    item.first_face,
                    item.second_block,
                    item.second_face,
                    item.permutation,
                    item.flips,
                    conforming=item.conforming,
                    tolerance=item.tolerance,
                )
            elif type(item) is TransfiniteBlock:
                rebuilt = TransfiniteBlock(
                    item.name,
                    restore(item.boundaries),
                    tolerance=item.tolerance,
                )
            else:
                from ..geometry._mapped_reference_boundary import (
                    MappedReferenceBoundarySource,
                )
                from ..geometry._piecewise_linear_boundary import (
                    PiecewiseLinearBoundarySource,
                )

                query = restore(item.fidelity_source)
                domain = restore(item.domain)
                if (
                    type(query)
                    in (PiecewiseLinearBoundarySource, MappedReferenceBoundarySource)
                    and domain is not None
                ):
                    if not _same_model_content(query.domain, domain):
                        raise ValueError(
                            "Stored structured query binds a different domain."
                        )
                    domain = query.domain
                rebuilt = NativeStructuredSource(
                    restore(item.blocks),
                    restore(item.interfaces),
                    item.source_id,
                    item.source_revision,
                    fidelity_source=query,
                    maximum_deviation=item.maximum_deviation,
                    domain=domain,
                    block_regions=item.block_regions,
                )
            if not _same_model_content(rebuilt, item):
                raise ValueError("Stored structured source or control changed.")
            return rebuilt
        if type(item) is CellMeshingResult:
            return CellMeshingResult.from_stored_fields(restore_row_record_recipes(item))
        if type(item) is MeshAdaptationResult:
            return MeshAdaptationResult.from_stored_fields(
                restore_row_record_recipes(item)
            )
        if type(item) is ImageIndexAffineArchiveRecipe:
            contract = SpatialCoordinateContract.from_dict(
                item.coordinate_contract.to_dict()
            )
            affine = ImageIndexAffine(
                item.matrix,
                item.source_frame_id,
                contract,
                item.axis_convention,
                item.voxel_reference,
                item.provenance,
            )
            if affine.affine_id != item.expected_affine_id:
                raise ValueError("Stored image index affine identity changed.")
            return affine
        if type(item) is ImageFieldSpecArchiveRecipe:
            quantity = item.quantity
            unit = UnitDefinition.from_dict(quantity.unit.to_dict())
            renewed_quantity = QuantitySpec(
                quantity.namespace,
                quantity.name,
                quantity.quantity_kind,
                unit,
                quantity.compatibility_key,
                quantity.axes,
                quantity.sign_convention,
                quantity.support_association,
                quantity.reference_configuration,
            )
            layout = item.layout
            renewed_layout = ValueLayout(
                layout.kind,
                layout.component_shape,
                layout.component_labels,
                layout.component_frame_id,
            )
            interval_unit = (
                None
                if item.interval_unit is None
                else UnitDefinition.from_dict(item.interval_unit.to_dict())
            )
            temporal = TemporalSampling(
                item.temporal_kind,
                item.interval_bounds,
                interval_unit,
                item.temporal_origin,
            )
            sampling = SamplingSemantics(
                item.spatial_kind, temporal, item.footprint_id, item.normalization
            )
            spec = ImageFieldSpec(renewed_quantity, renewed_layout, sampling)
            if (
                renewed_quantity.quantity_id != quantity.quantity_id
                or renewed_quantity.compatibility_id != quantity.compatibility_id
                or renewed_layout.layout_id != layout.layout_id
                or temporal.temporal_sampling_id != item.expected_temporal_id
                or sampling.sampling_id != item.expected_sampling_id
                or spec.spec_id != item.expected_spec_id
            ):
                raise ValueError("Stored image field specification changed.")
            return spec
        if type(item) is ImageUncertaintyArchiveRecipe:
            value = IndependentStandardUncertainty(
                item.values, UnitDefinition.from_dict(item.unit.to_dict())
            )
            if value.uncertainty_id != item.expected_uncertainty_id:
                raise ValueError("Stored image uncertainty changed.")
            return value
        if type(item) is ImageQualityFlagArchiveRecipe:
            value = QualityFlag(item.name, item.mask, item.meaning)
            if value.flag_id != item.expected_flag_id:
                raise ValueError("Stored image quality flag changed.")
            return value
        if type(item) is ImageIndexAffine:
            contract = SpatialCoordinateContract.from_dict(
                item.coordinate_contract.to_dict()
            )
            affine = ImageIndexAffine(
                item.matrix,
                item.source_frame_id,
                contract,
                item.axis_convention,
                item.voxel_reference,
                item.provenance,
            )
            if affine.affine_id != item.affine_id:
                raise ValueError("Stored image index affine identity changed.")
            return affine
        if type(item) is LabelVolumeArchiveRecipe:
            asset = MedicalImageAsset(**restore(item.asset_fields))
            if asset.content_id != item.expected_asset_id:
                raise ValueError("Stored categorical image asset identity changed.")
            labels = LabelVolume(asset, item.ontology)
            if labels.label_volume_id != item.expected_label_volume_id:
                raise ValueError("Stored categorical label volume identity changed.")
            return LabelVolume.from_stored_fields(labels)
        if type(item) is CompartmentMeshingSourceArchiveRecipe:
            labels = restore(item.labels)
            surface = restore_row_record_recipes(item.outer_surface)
            if type(surface) is not SurfaceModel:
                raise TypeError("Stored compartment source lost its outer surface owner.")
            stored_geometry = restore_row_record_recipes(item.outer_geometry)
            geometry = (
                None
                if stored_geometry is None
                else _geometry(stored_geometry, restore(surface.mesh))
            )
            return CompartmentMeshingSource.from_archive_inputs(
                labels,
                item.compartments,
                item.interfaces,
                surface,
                item.interpretation,
                geometry,
                expected_source_id=item.expected_source_id,
                expected_source_revision=item.expected_source_revision,
                expected_scope_ids=item.expected_scope_ids,
            )
        if type(item) is PiecewiseLinearDomain:
            domain = PiecewiseLinearDomain(
                item.vertices,
                item.facets,
                item.facet_regions,
                item.region_ids,
                source_id=item.source_id,
                region_occurrence_paths=item.region_occurrence_paths,
            )
            if (
                domain.domain_id != item.domain_id
                or domain.source_revision != item.source_revision
            ):
                raise ValueError("Stored piecewise-linear domain changed.")
            return domain
        if type(item) is MeshingDomainArchiveRecipe:
            return _domain(item)
        if type(item) is MeshingDomainBoundarySourceArchiveRecipe:
            domain = _domain(item.domain)
            query = MeshingDomainBoundarySource(
                domain,
                item.patches,
                resolution=item.resolution,
                boundary_occurrences=item.boundary_occurrences,
                boundary_cell_ids=item.boundary_cell_ids,
                boundary_patch_ids=item.boundary_patch_ids,
            )
            if query.source_scope_id != item.expected_source_scope_id:
                raise ValueError("Stored meshing-domain boundary scope changed.")
            query.require_current()
            return query
        if type(item) is SurfaceSourceRootAtlas:
            atlas_domain = restore(item.domain)
            atlas_mesh = CellMesh.from_stored_fields(
                restore_row_record_recipes(item.mesh)
            )
            atlas_geometry = _geometry(
                restore_row_record_recipes(item.geometry), atlas_mesh
            )
            atlas = SurfaceSourceRootAtlas(
                atlas_domain,
                atlas_mesh,
                atlas_geometry,
                item.root_parameters,
                item.root_patches,
                item.coordinate_contract,
                root_source_cell_ids=item.root_source_cell_ids,
                root_parameter_bounds=item.root_parameter_bounds,
            )
            if (
                atlas.atlas_id != item.atlas_id
                or atlas.root_cell_ids != item.root_cell_ids
                or (atlas.source_id, atlas.source_revision)
                != (item.source_id, item.source_revision)
            ):
                raise ValueError("Stored surface root atlas does not reproduce.")
            return atlas
        if type(item) is PreparedSurfaceSourceSupportArchiveRecipe:
            domain = restore(item.domain)
            root_query = restore(item.root_boundary_source)
            original = restore(item.original)
            support = PreparedSurfaceSourceSupport(
                domain, original, item.root_parameters, root_query
            )
            if support.support_id != item.expected_support_id:
                raise ValueError("Stored surface source support does not reproduce.")
            return support
        if type(item) is SurfaceNativeRestrictionBoundarySourceArchiveRecipe:
            root = restore(item.root)
            support = restore(item.support)
            mesh = CellMesh.from_stored_fields(restore_row_record_recipes(item.mesh))
            geometry = _geometry(restore_row_record_recipes(item.geometry), mesh)
            query = SurfaceNativeRestrictionBoundarySource(root, support, mesh, geometry)
            if (query.source_id, query.source_revision, query.source_scope_id) != (
                item.source_id,
                item.source_revision,
                item.source_scope_id,
            ):
                raise ValueError("Stored surface restriction source does not reproduce.")
            return query
        if type(item) is NativePlcSourceArchiveRecipe and item.fidelity_source is None:
            from ..geometry._surface_source_archive import _mapped_reference

            mapped = (
                None
                if item.mapped_domain is None
                else _mapped_reference(item.mapped_domain)
            )
            source = NativePlcSource(
                restore(item.complex),
                item.source_id,
                item.source_revision,
                mapped_domain=mapped,
                maximum_deviation=item.maximum_deviation,
                plc_vertex_root_global_ids=item.plc_vertex_root_global_ids,
            )
            if source.binding_id != item.expected_binding_id:
                raise ValueError("Stored native PLC source identity changed.")
            if mapped is not None:
                source.validate_restored()
            return source
        if type(item) is NativeMappedHexSourceArchiveRecipe:
            from ..geometry._surface_source_archive import _mapped_reference
            from .providers._native_sources import NativeMappedHexSource

            reference = restore(item.reference)
            if type(reference) is not NativePlcSource:
                raise TypeError("Mapped source lost its native PLC reference owner.")
            domain = _mapped_reference(item.domain)
            source = NativeMappedHexSource(
                reference,
                domain,
                plc_vertex_root_global_ids=item.plc_vertex_root_global_ids,
            )
            if (
                source.source_id != item.expected_source_id
                or source.source_revision != item.expected_source_revision
                or source.binding_id != item.expected_binding_id
                or source.root_relation.relation_id != item.expected_relation_id
            ):
                raise ValueError(
                    "Mapped source differs from its retained root authority."
                )
            source.validate_restored()
            return source
        if type(item) is MappedReferenceBoundarySourceArchiveRecipe:
            from ..geometry._mapped_reference_boundary import (
                MappedPlcRootAuthority,
                MappedReferenceBoundarySource,
            )

            authority = restore(item.authority)
            if not isinstance(authority, MappedPlcRootAuthority):
                raise TypeError("Mapped query lost its native root authority.")
            query = MappedReferenceBoundarySource(
                authority,
                face_scope_keys=item.face_scope_keys,
                edge_scope_keys=item.edge_scope_keys,
            )
            if query.source_query_id != item.expected_source_query_id:
                raise ValueError("Mapped query differs from its retained source scope.")
            query.require_current()
            return query
        if type(item) is AffinePlcBoundarySourceArchiveRecipe:
            query = AffinePlcBoundarySource(
                restore(item.owner), restore(item.domain), item.facet_rows
            )
            if query.source_query_id != item.expected_source_query_id:
                raise ValueError("Stored affine PLC boundary query changed.")
            return query
        if type(item) is PlcSourcePointWitness:
            witness = PlcSourcePointWitness(
                item.vertex_global_ids,
                item.vertex_kinds,
                item.vertex_entities,
                item.vertex_parameters,
                segments=item.segments,
                faces=item.faces,
                face_groups=item.face_groups,
                face_planes=item.face_planes,
                face_plane_tolerances=item.face_plane_tolerances,
                domain_face_rows=item.domain_face_rows,
                domain_id=item.domain_id,
                vertex_deviations=item.vertex_deviations,
            )
            if witness.witness_id != item.witness_id:
                raise ValueError("Stored PLC source-point witness identity changed.")
            return witness
        if type(item) is AffinePlcDomainArchiveRecipe:
            reference = item.reference_domain
            if type(reference) is not PiecewiseLinearDomain:
                raise TypeError("Affine PLC source lost its reference domain.")
            domain = PiecewiseLinearDomain(
                reference.vertices,
                reference.facets,
                reference.facet_regions,
                reference.region_ids,
                source_id=reference.source_id,
                region_occurrence_paths=reference.region_occurrence_paths,
            )
            if (
                domain.domain_id != reference.domain_id
                or domain.source_revision != reference.source_revision
            ):
                raise ValueError("Stored affine PLC reference domain changed.")
            authority = None if item.authority is None else restore(item.authority)
            if authority is not None and type(authority) is not CompartmentMeshingSource:
                raise TypeError(
                    "Affine PLC authority must be a replayed categorical source."
                )
            affine = AffinePlcDomain(
                domain,
                item.matrix,
                item.offset,
                source_revision=item.source_revision,
                authority=authority,
            )
            if affine.domain_id != item.expected_domain_id:
                raise ValueError("Stored affine PLC map identity changed.")
            return affine
        if type(item) is AffinePlcGeometrySourceArchiveRecipe:
            geometry_source = AffinePlcGeometrySource(
                restore(item.domain),
                restore(item.reference_mesh),
                witness=restore(item.witness),
            )
            if (
                geometry_source.source_id != item.expected_source_id
                or geometry_source.source_revision != item.expected_source_revision
                or geometry_source.source_map_id != item.expected_source_map_id
            ):
                raise ValueError("Stored affine PLC geometry source changed.")
            geometry_source.source_coordinates()
            return geometry_source
        if type(item) is SegmentChartCurve:
            curve = SegmentChartCurve(item.start, item.end, item.first, item.last)
            if (
                curve.origin_exact != item.origin_exact
                or curve.direction_exact != item.direction_exact
            ):
                raise ValueError("Stored segment chart exact coefficients changed.")
            return curve
        if type(item) is ParametricCurveBoundarySourceArchiveRecipe:
            curves = tuple(
                restore(curve)
                if type(curve) is SegmentChartCurve
                else _reconstruct_brep_graph(curve)
                for curve in item.curves
            )
            query = ParametricCurveBoundarySource(
                curves,
                item.parameter_ranges,
                source_id=item.source_id,
                source_revision=item.source_revision,
                curve_entity_ids=item.curve_entity_ids,
                occurrence_paths=item.occurrence_paths,
                chart_ids=item.chart_ids,
                endpoint_source_keys=item.endpoint_source_keys,
                endpoint_point_bounds=item.endpoint_point_bounds,
                endpoint_parameter_bounds=item.endpoint_parameter_bounds,
            )
            if query.query_id != item.expected_query_id:
                raise ValueError("Stored source curve query does not reproduce.")
            query.require_current()
            return query
        if type(item) is NativePlanarSourceArchiveRecipe:
            edges = np.asarray(item.edges)
            offsets = np.asarray(item.loop_offsets)
            if (
                edges.ndim != 2
                or edges.shape[1] != 2
                or edges.dtype.kind not in "iu"
                or offsets.ndim != 1
                or offsets.dtype.kind not in "iu"
                or len(offsets) < 2
                or offsets[0] != 0
                or offsets[-1] != len(edges)
                or np.any(np.diff(offsets) < 3)
            ):
                raise ValueError("Stored planar source loops are malformed.")
            loops = tuple(
                tuple(int(index) for index in edges[int(a) : int(b), 0])
                for a, b in zip(offsets[:-1], offsets[1:], strict=True)
            )
            region = PlanarMeshRegion(item.vertices, loops, feature_id=item.source_id)
            if not np.array_equal(np.asarray(region.edges), edges) or not np.array_equal(
                np.asarray(region.loop_offsets), offsets
            ):
                raise ValueError(
                    "Stored planar source edges do not form their declared loops."
                )
            embedded_inputs = (
                item.embedded_vertices,
                item.embedded_edges,
                item.embedded_source_id,
                item.embedded_vertex_entity_ids,
                item.embedded_edge_entity_ids,
            )
            if all(value is None for value in embedded_inputs):
                embedded = None
            elif any(value is None for value in embedded_inputs):
                raise ValueError("Stored embedded source strata are incomplete.")
            else:
                embedded = SegmentMesh(
                    item.embedded_vertices,
                    item.embedded_edges,
                    source_id=item.embedded_source_id,
                )
            source = NativePlanarSource(
                region,
                item.declared_source_revision,
                embedded=embedded,
                region_scope_entity_ids=item.region_scope_entity_ids,
                vertex_entity_ids=item.vertex_entity_ids,
                edge_entity_ids=item.edge_entity_ids,
                embedded_vertex_entity_ids=item.embedded_vertex_entity_ids,
                embedded_edge_entity_ids=item.embedded_edge_entity_ids,
            )
            if source.binding_id != item.expected_binding_id:
                raise ValueError("Stored planar source binding does not reproduce.")
            return source
        if type(item) is NativePlanarMotionSourceArchiveRecipe:
            source = restore(item.source)
            motion = NativePlanarMotionSource(source, item.translation)
            if (
                motion.transition_id != item.expected_transition_id
                or motion.successor.binding_id != item.expected_successor_binding_id
            ):
                raise ValueError("Stored planar source motion does not reproduce.")
            return motion
        if type(item) is MaterialSiteId:
            site = MaterialSiteId(
                item.key,
                domain_id=item.domain_id,
                cell_id=item.cell_id,
                quadrature_rule_id=item.quadrature_rule_id,
                reference_site_id=item.reference_site_id,
            )
            if site.site_id != item.site_id:
                raise ValueError("Stored material site identity does not reproduce.")
            return site
        if type(item) is MaterialSiteOwnership:
            if len(item.site_ids) != len(item.owners):
                raise ValueError("Stored material ownership rows differ in length.")
            ownership = MaterialSiteOwnership(
                tuple(
                    (restore(site), owner)
                    for site, owner in zip(item.site_ids, item.owners, strict=True)
                ),
                part_count=item.part_count,
            )
            if (
                ownership.ownership_id != item.ownership_id
                or tuple(site.key for site in ownership.site_ids)
                != tuple(site.key for site in item.site_ids)
                or ownership.owners != item.owners
            ):
                raise ValueError("Stored material ownership does not reproduce.")
            return ownership
        if type(item) is MaterialState:
            state = MaterialState(
                restore(item.site_id),
                item.model_id,
                item.committed,
                trial=item.trial,
                state_version=item.state_version,
            )
            if state.layout_id != item.layout_id or state.state_id != item.state_id:
                raise ValueError("Stored material state identity does not reproduce.")
            return state
        if type(item) is MaterialTransaction:
            transaction = MaterialTransaction(
                tuple(restore(state) for state in item.states),
                ownership=None if item.ownership is None else restore(item.ownership),
                owner_rank=item.owner_rank,
            )
            if (
                transaction.layout_id != item.layout_id
                or transaction.transaction_id != item.transaction_id
                or tuple(state.site_id.key for state in transaction.states)
                != tuple(state.site_id.key for state in item.states)
            ):
                raise ValueError(
                    "Stored material transaction identity does not reproduce."
                )
            return transaction
        if type(item) is MaterialCheckpointPayload:
            payload = MaterialCheckpointPayload(restore(item.state), plan_id=item.plan_id)
            if (
                payload.layout_id != item.layout_id
                or payload.payload_id != item.payload_id
            ):
                raise ValueError("Stored material checkpoint payload does not reproduce.")
            return payload
        if type(item) is BRepModel:
            return _reconstruct_brep_graph(item)
        if type(item) is MeshingScope:
            if not item.side_records:
                return MeshingScope.from_stored_fields(item)
            if not in_composition:
                raise ValueError("Sided scope requires a replayed source model.")
            pending_sided_scopes.append(item)
            return item
        if type(item) is SurfaceSourceChartsArchiveRecipe:
            return restore_surface_source_charts(item)
        if type(item) is PiecewiseLinearComplexArchiveRecipe:
            item = item.decode()
        if type(item) is PiecewiseLinearComplex:
            stored_charts = item.cad_source_charts
            if stored_charts is None:
                return PiecewiseLinearComplex.from_stored_fields(item)
            if type(stored_charts) is SurfaceSourceChartsArchiveRecipe:
                charts = restore_surface_source_charts(stored_charts)
            elif type(stored_charts) is SurfaceSourceCharts:
                stored_charts.require_current()
                charts = stored_charts
            else:
                raise TypeError("Stored CAD PLC requires canonical source charts.")
            if (
                item.cad_embedding is None
                or item.cad_fidelity is None
                or item.cad_fidelity_tolerance is None
            ):
                raise ValueError("Stored CAD PLC is missing independent surface proofs.")
            return PiecewiseLinearComplex.from_stored_fields(
                item, cad_source_charts=charts
            )
        if type(item) in (
            MeshingScopeSide,
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
        if type(item) is NativeCurveSchedule:
            schedule = NativeCurveSchedule(
                quadrature_intervals=item.quadrature_intervals,
                quadrature_order=item.quadrature_order,
                fidelity_samples=item.fidelity_samples,
            )
            if schedule.schedule_id != item.schedule_id:
                raise ValueError("Stored native curve schedule changed.")
            return schedule
        if type(item) is NativeSurfaceSchedule:
            schedule = NativeSurfaceSchedule(
                quality_angle_degrees=item.quality_angle_degrees,
                spacing_fraction=item.spacing_fraction,
                minimum_edge_samples=item.minimum_edge_samples,
                normal_deviation_aim=item.normal_deviation_aim,
                curve_schedule=restore(item.curve_schedule),
            )
            if schedule.schedule_id != item.schedule_id:
                raise ValueError("Stored native surface schedule changed.")
            return schedule
        if type(item) is NativeVolumeSchedule:
            schedule = NativeVolumeSchedule(
                radius_edge_bound=item.radius_edge_bound,
                minimum_dihedral_degrees=item.minimum_dihedral_degrees,
                shape_strength=item.shape_strength,
                plc_search_budget=PlcSearchBudget(*item.plc_search_budget),
            )
            if schedule.schedule_id != item.schedule_id:
                raise ValueError("Stored native volume schedule changed.")
            return schedule
        if type(item) is NativePolyhedralSchedule:
            schedule = NativePolyhedralSchedule(
                item.maximum_cell_diameter,
                item.target_site_volumes,
                item.relative_volume_tolerance,
                item.feature_tolerance,
                item.protected_site_ids,
                item.require_every_site,
                item.lloyd_steps,
            )
            if schedule.schedule_id != item.schedule_id:
                raise ValueError("Stored native polyhedral schedule changed.")
            return schedule
        if type(item) is NativeMeshingOptions:
            if (
                item.route not in ("layer_core", "image_material_tetrahedral")
                or (
                    item.route == "image_material_tetrahedral"
                    and item.polyhedral_schedule is not None
                )
                or any(
                    value is not None
                    for value in (
                        item.implicit_policy,
                        item.curve_schedule,
                        item.surface_schedule,
                        item.structured_schedule,
                        item.hex_core_fraction,
                        item.grid_schedule,
                    )
                )
            ):
                raise ValueError("Stored native options need their route-specific owner.")
            options = NativeMeshingOptions(
                item.route,
                volume_schedule=restore(item.volume_schedule),
                polyhedral_schedule=restore(item.polyhedral_schedule),
            )
            if options.options_id != item.options_id:
                raise ValueError("Stored native meshing options changed.")
            return options
        if type(item) is MixedLayerColumns:
            columns = MixedLayerColumns(
                item.cell_ids,
                item.column_ids,
                item.interval_indices,
                hard_first_thickness=item.hard_first_thickness,
                axial_refinement=item.axial_refinement,
                allow_schedule_change=item.allow_schedule_change,
            )
            if columns.columns_id != item.columns_id:
                raise ValueError("Stored mixed layer columns identity changed.")
            return columns
        if type(item) is PeriodicRefinementEdgeLock:
            raise ValueError("A periodic edge lock requires its source mesh owner.")
        if type(item) is PeriodicQuotientHistory:
            history = PeriodicQuotientHistory(
                item.identification_id,
                item.source_mesh_id,
                item.entity_keys_by_degree,
                item.allocator_next_ids,
                item.occurrences_by_degree,
                item.vertex_actions_by_id,
            )
            if (
                history.history_id != item.history_id
                or history.entity_keys_by_degree != item.entity_keys_by_degree
                or history.occurrences_by_degree != item.occurrences_by_degree
                or history.vertex_actions_by_id != item.vertex_actions_by_id
            ):
                raise ValueError("Stored periodic quotient history changed.")
            return history
        if type(item) in (FiniteElementHPGeometry, FiniteElementHPRefinementResult):
            raise ValueError("HP geometry and refinement require their source owner.")
        if type(item) is HPAdaptationEvidence:
            raise ValueError("HP adaptation evidence requires its result endpoints.")
        if type(item) is FiniteElementHPTopology:
            rebuilt = FiniteElementHPTopology(
                item.cell_kind,
                item.topology_id,
                item.cell_global_ids,
                item.allocated,
                item.active,
                item.cell_degrees,
                root_cell_ids=item.root_cell_ids,
                path_codes=item.path_codes,
                levels=item.levels,
                parent_slots=item.parent_slots,
                child_slots=item.child_slots,
                child_valid=item.child_valid,
            )
            if not same_content(rebuilt, item):
                raise ValueError("Stored hp topology differs from its constructor.")
            return rebuilt
        if type(item) is FiniteElementHPLineage:
            codes = np.asarray(item.relation_codes)
            if codes.ndim != 1 or np.any((codes < 0) | (codes > 2)):
                raise ValueError("Stored hp lineage relation codes are invalid.")
            names = ("unchanged", "refinement", "coarsening")
            rebuilt = FiniteElementHPLineage(
                item.source_topology_id,
                item.target_topology_id,
                item.source_capacity,
                item.target_capacity,
                item.source_slots,
                item.target_slots,
                tuple(names[int(code)] for code in codes),
                valid=item.valid,
            )
            if not same_content(rebuilt, item):
                raise ValueError("Stored hp lineage differs from its constructor.")
            return rebuilt
        if type(item) is MixedAdaptationHierarchy:
            hierarchy = MixedAdaptationHierarchy(
                item.siblings,
                item.active_cells,
                item.forest_roots,
                item.vertex_points,
                item.retired_entities,
                item.layer_policy,
                item.next_cell_id,
                item.next_vertex_id,
                restore(item.periodic_quotient_history),
                item.active_lifted_topology_id,
            )
            if hierarchy.hierarchy_id != item.hierarchy_id or not _same_model_content(
                hierarchy, item
            ):
                raise ValueError("Stored mixed adaptation hierarchy identity changed.")
            return hierarchy
        if type(item) is MarkedMeshAdaptation:
            if item.periodic_edge_lock is not None:
                raise ValueError("Periodic edge locks require their own archive owner.")
            if item.hierarchy is not None and type(item.hierarchy) not in (
                MixedAdaptationHierarchy,
                FiniteElementHPEpochArchiveRecipe,
            ):
                raise ValueError("Stored marked request has an unsupported hierarchy.")
            request = MarkedMeshAdaptation(
                item.refine_cell_ids,
                item.coarsen_cell_ids,
                hierarchy=restore(item.hierarchy),
                layer_columns=restore(item.layer_columns),
                inherited_cell_markers=item.inherited_cell_markers,
            )
            if request.request_id != item.request_id:
                raise ValueError("Stored marked adaptation request changed.")
            return request
        if type(item) is ReferenceCellMapWitnesses:
            witness = ReferenceCellMapWitnesses(
                item.target_cell_ids,
                item.source_cell_ids,
                item.target_cell_kind,
                item.source_cell_kind,
                item.source_reference_vertices,
                item.source_reference_exact_vertices,
                item.source_reference_chart_arguments,
            )
            validate_reference_cell_map_witnesses(witness)
            return witness
        if type(item) is ReferenceCellPartition:
            partition = ReferenceCellPartition(
                tuple(restore(group) for group in item.maps)
            )
            validate_reference_cell_partition(partition)
            return partition
        if type(item) is CellGeometryTransition:
            source = restore(item.source_mesh)
            target = restore(item.target_mesh)
            stored_evidence = item.evidence
            evidence = CellGeometryTransitionEvidence(
                stored_evidence.kind,
                reference_map_count=stored_evidence.reference_map_count,
                refinement_map_count=stored_evidence.refinement_map_count,
                coarsening_map_count=stored_evidence.coarsening_map_count,
                continuity_residual=stored_evidence.continuity_residual,
                approximation_bound=stored_evidence.approximation_bound,
            )
            if evidence.evidence_id != stored_evidence.evidence_id:
                raise ValueError("Stored geometry transition evidence changed.")
            source_geometry = _geometry(item.source_geometry, source)
            target_geometry = _geometry(item.target_geometry, target)
            _require_retained_bank_origin(source_geometry, target_geometry)
            transition = CellGeometryTransition(
                source,
                source_geometry,
                target,
                target_geometry,
                evidence=evidence,
                reference_map=restore(item.reference_map),
                refinement=restore(item.refinement),
                coarsening=restore(item.coarsening),
                coarsening_partition=restore(item.coarsening_partition),
            )
            if transition.transition_id != item.transition_id:
                raise ValueError("Stored geometry transition identity changed.")
            return transition
        if type(item) is ExactPeriodicAlgebraicSource:
            source = ExactPeriodicAlgebraicSource.from_stored_fields(item)
            if not _same_model_content(source, item):
                raise ValueError(
                    "Stored exact periodic algebraic source differs from its constructor."
                )
            return source
        if type(item) is ExactPlcCellGeometrySource:
            source = ExactPlcCellGeometrySource.from_stored_fields(item)
            if not _same_model_content(source, item):
                raise ValueError("Stored exact PLC source differs from its constructor.")
            return source
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
            scope_start = len(pending_sided_scopes)
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
            models = tuple(
                entry.value if type(entry.value) is BRepModel else entry.value.model
                for entry in entries
                if type(entry.value) in (BRepModel, BRepPartitionResult)
            )
            visited: set[int] = set()
            nested: list[object] = [entry.value for entry in entries]
            while nested:
                value = nested.pop()
                if id(value) in visited:
                    continue
                visited.add(id(value))
                if type(value) is MeshingScope:
                    if value.side_records and all(
                        value is not item for item in pending_sided_scopes[scope_start:]
                    ):
                        pending_sided_scopes.append(value)
                    continue
                if type(value) in (BRepModel, BRepPartitionResult, Composition):
                    continue
                if dataclasses.is_dataclass(value) and not isinstance(value, type):
                    nested.extend(
                        getattr(value, field.name) for field in dataclasses.fields(value)
                    )
                elif isinstance(value, (tuple, list, set, frozenset)):
                    nested.extend(value)
                elif isinstance(value, Mapping):
                    nested.extend(value.values())
            validated: dict[int, MeshingScope] = {}
            scope_owners: dict[int, BRepModel] = {}
            for scope in pending_sided_scopes[scope_start:]:
                owners = tuple(
                    model
                    for model in models
                    if (model.source_id, model.source_revision)
                    == (scope.source_id, scope.source_revision)
                )
                if not owners or any(
                    model.model_id != owners[0].model_id for model in owners
                ):
                    raise ValueError(
                        "Sided scope requires one replayed current source model."
                    )
                validated[id(scope)] = MeshingScope.from_stored_fields(
                    scope, source_model=owners[0]
                )
                scope_owners[id(scope)] = owners[0]

            def rebuilt_entry_value(value: Any) -> Any:
                if id(value) in validated:
                    return validated[id(value)]
                if type(value) is PatchControl and value.scope.side_records:
                    return PatchControl.from_stored_fields(
                        value, source_model=scope_owners[id(value.scope)]
                    )
                if type(value) in (SurfaceMeshingSpec, VolumeMeshingSpec):
                    primary = (
                        value.scope
                        if type(value) is SurfaceMeshingSpec
                        else value.boundary_scope
                    )
                    owners = tuple(
                        model
                        for model in models
                        if (model.source_id, model.source_revision)
                        == (primary.source_id, primary.source_revision)
                    )
                    if owners:
                        return type(value).from_stored_fields(
                            value, source_model=owners[0]
                        )
                    if type(value) is SurfaceMeshingSpec:
                        return SurfaceMeshingSpec.from_stored_fields(value)
                return value

            entries = tuple(
                CompositionEntry(
                    rebuilt_entry_value(entry.value),
                    entry_id=entry.entry_id,
                    role=entry.role,
                    owner_id=entry.owner_id,
                    structure_id=entry.structure_id,
                    revision_id=entry.revision_id,
                    semantics_id=entry.semantics_id,
                    dependencies=entry.dependencies,
                )
                for entry in entries
            )
            del pending_sided_scopes[scope_start:]
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
    return _read_model_archive(path, limits=limits, immutable_arrays=False)


def _read_model_archive(
    path: str | Path,
    /,
    *,
    limits: ArrayArchiveLimits | None,
    immutable_arrays: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _register_model_archive_types()
    manifest, members = (
        _read_array_archive(path, limits=limits, immutable_arrays=True)
        if immutable_arrays
        else read_array_archive(path, limits=limits)
    )
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
        if immutable_arrays:
            recipe, arrays = _expand_shared_immutable_model_archive_recipe(
                shared_recipe, members, _MODEL_ARRAYS, limits
            )
        else:
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
    *,
    immutable_arrays: bool,
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
    restore_arrays = (
        _model_from_shared_immutable_array_recipe
        if immutable_arrays
        else model_from_array_recipe
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
                value = restore_arrays(
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
    immutable_arrays: bool = False,
) -> dict[str, Any]:
    """Decode selected registered field paths for prerequisite validation.

    The complete archive is admitted first. The returned fields are stored
    inputs, not certified authority; the owner must replay their mathematics.

    ``immutable_arrays`` preserves authenticated shared array ownership and
    charges each immutable owner once, as the full model reader does. Returned
    numerical buffers then remain read-only. Mutable reads retain occurrence
    accounting and independent buffers.
    """
    _register_model_archive_types()
    _require_model_type(expected_type)
    if not isinstance(immutable_arrays, bool):
        raise TypeError("immutable_arrays must be a bool.")
    if (
        not fields
        or any(
            type(name) is not str or any(not part for part in name.split("."))
            for name in fields
        )
        or len(set(fields)) != len(fields)
    ):
        raise ValueError("Selected model fields must be distinct nonempty names.")
    manifest, arrays = _read_model_archive(
        path, limits=limits, immutable_arrays=immutable_arrays
    )
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
        _validate_archive_columns(
            recipe, payloads, limits, immutable_arrays=immutable_arrays
        )
        restore_arrays = (
            _model_from_shared_immutable_array_recipe
            if immutable_arrays
            else model_from_array_recipe
        )
        selected = {}
        for name in fields:
            subtree = subtrees[name]
            subnodes = _recipe_array_nodes(subtree)
            subarrays = {
                f"{_MODEL_ARRAYS}/{index:06d}": payloads[id(node)]
                for index, node in enumerate(subnodes)
            }
            raw = restore_arrays(
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


def _model_archive_content_id(manifest: dict[str, Any], arrays: dict[str, Any], /) -> str:
    return canonical_fingerprint(
        {
            "kind": _MODEL_KIND,
            "model_recipe": manifest["model_recipe"],
            "arrays": array_collection_digest(arrays),
        }
    )


def model_archive_content_id(
    path: str | Path, /, *, limits: ArrayArchiveLimits | None
) -> str:
    """Return the canonical content ID of an admitted native model archive."""
    manifest, arrays = _read_model_archive(path, limits=limits, immutable_arrays=True)
    return _model_archive_content_id(manifest, arrays)


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
    arrays = _pack_owned_model_array_tree(
        model_archive_root_projection(archive_model),
        recipe,
        prefix=_MODEL_ARRAYS,
        limits=limits,
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
    expected_content_id: str | None = None,
) -> Any:
    """Restore a root recipe, then delegate scientific reconstruction to its owner."""
    _register_model_archive_types()
    _require_model_type(expected_type)
    if not callable(restore):
        raise TypeError("restore must be the owning model's reconstruction function.")
    manifest, arrays = _read_model_archive(path, limits=limits, immutable_arrays=True)
    if expected_content_id is not None and (
        not isinstance(expected_content_id, str)
        or _model_archive_content_id(manifest, arrays) != expected_content_id
    ):
        raise ArrayArchiveCorruptionError("Native model archive content ID differs.")
    recipe = manifest["model_recipe"]
    try:
        declared_type = artifact_value(recipe["type"])
        _require_model_type(declared_type)
        if declared_type is not expected_type:
            raise ValueError("Native model archive has a different registered root type.")
        raw = _model_from_shared_immutable_array_recipe(
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
        _require_model_array_content(
            model_archive_root_projection(rebuilt), recipe, arrays, prefix=_MODEL_ARRAYS
        )
    except ArrayArchiveCorruptionError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ArrayArchiveCorruptionError(
            "Native model archive is incompatible with its registered recipe."
        ) from error
    return rebuilt


def overset_registration_recipe(registration: Any, /) -> OversetRegistrationArchiveRecipe:
    """Capture the serial registration and source partition, excluding search caches."""
    from ._overset import OversetRegistration

    if type(registration) is not OversetRegistration:
        raise TypeError("Expected the actual overset registration owner.")
    if registration.assembly.owner_local or registration.assembly.couplings:
        raise ValueError("This archive requires an independently owned serial assembly.")
    return OversetRegistrationArchiveRecipe(
        tuple((part.name, part.part_id) for part in registration.assembly.parts),
        tuple(model_archive_projection(spec) for spec in registration.specs),
        model_archive_projection(registration.policy),
        registration.epoch,
        registration.predecessor_id,
        registration.connectivity_id,
        registration.partition_priority,
        model_archive_projection(registration.partition_sources),
        registration.assembly.assembly_id,
    )


def restore_overset_registration(
    recipe: OversetRegistrationArchiveRecipe, parts: tuple[Any, ...], /
) -> Any:
    """Bind declarative overset registration to independently restored accepted parts.

    The caller must invoke ``prepare()`` to rebuild and verify the actual
    connectivity and source-bound exact partition against the stored identity.
    """
    from ..solver._finite_volume_archive_context import (
        _finite_volume_archive_overset_part_spec,
        _finite_volume_archive_overset_policy,
    )
    from ._assembly import MeshAssembly, MeshPart
    from ._overset import OversetRegistration

    if type(recipe) is not OversetRegistrationArchiveRecipe:
        raise TypeError("Expected a canonical overset registration recipe.")
    if not isinstance(parts, tuple) or any(type(part) is not MeshPart for part in parts):
        raise TypeError(
            "Overset registration requires independently restored mesh parts."
        )
    by_name = {part.name: part for part in parts}
    if len(by_name) != len(parts) or set(by_name) != {
        name for name, _ in recipe.part_bindings
    }:
        raise ValueError("Stored overset part inventory differs from the restored parts.")
    ordered = tuple(by_name[name] for name, _ in recipe.part_bindings)
    if tuple((part.name, part.part_id) for part in ordered) != recipe.part_bindings:
        raise ValueError("Stored overset part revisions differ from the restored parts.")
    assembly = MeshAssembly(ordered)
    if assembly.assembly_id != recipe.assembly_id:
        raise ValueError("Stored overset assembly identity changed.")
    specifications = []
    for stored in recipe.specifications:
        if type(stored) is not FiniteVolumeOversetPartSpecArchiveRecipe:
            raise TypeError("Stored overset specification lacks its canonical recipe.")
        renewed = stored._replace(
            wall=restore_model_archive_value(stored.wall),
            boundary=restore_model_archive_value(stored.boundary),
            excluded=restore_model_archive_value(stored.excluded),
        )
        specifications.append(_finite_volume_archive_overset_part_spec(renewed))
    policy = _finite_volume_archive_overset_policy(recipe.policy)
    sources = restore_model_archive_value(recipe.partition_sources)
    return OversetRegistration(
        assembly,
        tuple(specifications),
        policy,
        recipe.epoch,
        recipe.predecessor_id,
        recipe.connectivity_id,
        partition_priority=recipe.partition_priority,
        partition_sources=sources,
    )


def planar_motion_advance_recipe(
    monitor: Any,
    source: Any,
    coordinates: Any,
    advance: Any,
    /,
    *,
    boundary_residual: float,
    fixed_vertices: Any | None,
    numeric_version: str,
    adaptation_policy: Any | None,
    remesh_metric: Any | None,
    prior_transition: Any | None,
    source_transition: Any,
) -> PlanarMotionAdvanceArchiveRecipe:
    """Capture the actual operands of one accepted planar source translation."""
    import numpy as np

    from ._motion import MeshMotionAdvance, MeshMotionDecision, MeshMotionMonitor
    from ._motion_source import NativePlanarMotionSource

    if (
        adaptation_policy is not None
        or remesh_metric is not None
        or prior_transition is not None
        or type(monitor) is not MeshMotionMonitor
        or type(advance) is not MeshMotionAdvance
        or advance.decision is not MeshMotionDecision.ACCEPT_MOTION
        or not advance.accepted
        or type(advance.source_transition) is not NativePlanarMotionSource
        or type(source_transition) is not NativePlanarMotionSource
        or source_transition.transition_id != advance.source_transition.transition_id
        or advance.source_result_id != source.result_id
        or advance.source_mesh_id != source.mesh.mesh_id
        or monitor.reference.mesh_id != source.mesh.mesh_id
        or advance.result is None
        or advance.result.certification is None
        or advance.geometry_transition is None
        or advance.relocation is not None
        or advance.adaptation is not None
        or advance.material_transition is not None
        or monitor.policy.relocation_termination is not None
        or len(advance.assessments) != 1
        or float(boundary_residual) != advance.assessments[0].boundary_residual
        or not np.array_equal(
            np.asarray(coordinates), np.asarray(advance.result.mesh.coordinates)
        )
    ):
        raise ValueError("Planar motion archive requires its actual accepted proposal.")
    return PlanarMotionAdvanceArchiveRecipe(
        source_transition,
        monitor.policy,
        coordinates,
        float(boundary_residual),
        fixed_vertices,
        numeric_version,
        source.result_id,
        monitor.monitor_id,
        advance.advance_id,
        advance.result.result_id,
        advance.geometry_transition.transition_id,
        advance.result.certification.inputs.fidelity_witness,
    )


def restore_planar_motion_advance(
    recipe: PlanarMotionAdvanceArchiveRecipe, source: Any, /
) -> Any:
    """Rebuild the monitor and execute a retained translated source proposal."""
    import numpy as np

    from ..discretization._cell_geometry_validity import CellValidityPolicy
    from ._motion import (
        advance_mesh_motion,
        MeshMotionDecision,
        MeshMotionMonitor,
        MeshMotionMonitorPolicy,
    )
    from ._motion_source import NativePlanarMotionSource

    if type(recipe) is not PlanarMotionAdvanceArchiveRecipe:
        raise TypeError("Expected canonical planar motion inputs.")
    if source.result_id != recipe.source_result_id:
        raise ValueError("Planar motion source differs from the accepted predecessor.")
    stored = recipe.monitor_policy
    if (
        type(stored) is not MeshMotionMonitorPolicy
        or stored.relocation_termination is not None
    ):
        raise TypeError("Stored motion policy requires its canonical owner.")
    validity = CellValidityPolicy(
        relative_determinant_floor=stored.validity.relative_determinant_floor,
        relative_planarity_tolerance=stored.validity.relative_planarity_tolerance,
    )
    if validity.policy_id != stored.validity.policy_id:
        raise ValueError("Stored motion validity policy changed.")
    policy = MeshMotionMonitorPolicy(
        relocation_jacobian_ratio=stored.relocation_jacobian_ratio,
        remesh_jacobian_ratio=stored.remesh_jacobian_ratio,
        relocation_quality_ratio=stored.relocation_quality_ratio,
        remesh_quality_ratio=stored.remesh_quality_ratio,
        relocation_displacement_fraction=stored.relocation_displacement_fraction,
        remesh_displacement_fraction=stored.remesh_displacement_fraction,
        maximum_boundary_residual=stored.maximum_boundary_residual,
        relocation_objective=stored.relocation_objective,
        validity=validity,
        accept_valid_nonconverged_relocation=stored.accept_valid_nonconverged_relocation,
    )
    if policy.policy_id != stored.policy_id:
        raise ValueError("Stored motion monitor policy changed.")
    monitor = MeshMotionMonitor(source.mesh, policy=policy)
    if monitor.monitor_id != recipe.monitor_id:
        raise ValueError("Rebuilt motion monitor does not bind the predecessor.")
    transition = recipe.source_transition
    if type(transition) is NativePlanarMotionSourceArchiveRecipe:
        transition = restore_model_archive_value(transition)
    elif type(transition) is NativePlanarMotionSource:
        transition = NativePlanarMotionSource.from_stored_fields(transition)
    else:
        raise TypeError("Stored planar motion lost its source transition.")
    advance = advance_mesh_motion(
        monitor,
        source,
        np.asarray(recipe.proposal_coordinates),
        boundary_residual=recipe.boundary_residual,
        fixed_vertices=None
        if recipe.fixed_vertices is None
        else np.asarray(recipe.fixed_vertices),
        numeric_version=recipe.numeric_version,
        source_transition=transition,
    )
    if (
        advance.decision is not MeshMotionDecision.ACCEPT_MOTION
        or not advance.accepted
        or advance.advance_id != recipe.advance_id
        or advance.result is None
        or advance.result.result_id != recipe.target_result_id
        or advance.result.certification is None
        or advance.geometry_transition is None
        or advance.geometry_transition.transition_id != recipe.geometry_transition_id
        or not _same_model_content(
            advance.result.certification.inputs.fidelity_witness, recipe.fidelity_witness
        )
    ):
        raise ValueError("Replayed planar motion differs from its accepted advance.")
    return advance


def layer_core_authoring_recipe(
    source: Any, control: Any, policy: Any, /
) -> NativeLayerCoreAuthoringArchiveRecipe:
    """Capture actual advancing-layer state and its independent authoring inputs."""
    from ._boundary_layer import BoundaryLayerPolicy
    from ._controls import BoundaryLayerControl
    from .providers._native_sources import NativeLayerCoreSource

    if type(source) is not NativeLayerCoreSource:
        raise TypeError("Layer/core archive requires the canonical source owner.")
    if (
        type(control) is not BoundaryLayerControl
        or type(policy) is not BoundaryLayerPolicy
    ):
        raise TypeError("Layer/core archive requires its actual control and policy.")
    if (
        source.layers.control_id != control.control_id
        or source.layers.policy_id != policy.policy_id
    ):
        raise ValueError("Layer source does not bind the supplied authoring inputs.")
    if (source.layer_control is None) != (source.layer_policy is None):
        raise ValueError("Layer source must retain control and policy together.")
    if source.layer_control is not None and (
        source.layer_control.control_id != control.control_id
        or source.layer_policy.policy_id != policy.policy_id
    ):
        raise ValueError("Retained layer controls differ from the authoring inputs.")
    if source.core_power_source is not None:
        raise ValueError("Power-mapped layer sources require their own input recipe.")
    mapped = source.mapped_domain is not None
    if mapped != (source.reference_source is not None):
        raise ValueError("Mapped layer authoring requires both original owners.")
    if mapped:
        from ._layer_core_source_restore import validate_mapped_layer_core_source

        validate_mapped_layer_core_source(source)
        reference = source.reference_source
        if any(
            value is not None
            for value in (
                source.layers.wall_association,
                source.source_boundary_scope,
                source.source_wall_charts,
                source.source_boundary_charts,
                source.source_boundary_schedule,
                source.source_wall_projection,
                source.source_wall_association_policy,
                reference.source_domain,
                reference.source_boundary_scope,
                reference.source_wall_charts,
                reference.source_boundary_charts,
                reference.source_boundary_schedule,
                reference.source_wall_projection,
                reference.source_wall_association_policy,
                reference.fidelity_source,
                reference.maximum_deviation,
                reference.core_power_source,
            )
        ):
            raise ValueError("Mapped layer reference has unsupported source owners.")
        if (
            (reference.layer_control is None) != (source.layer_control is None)
            or (reference.layer_policy is None) != (source.layer_policy is None)
            or (
                reference.layer_control is not None
                and (
                    reference.layer_control.control_id != control.control_id
                    or reference.layer_policy.policy_id != policy.policy_id
                )
            )
        ):
            raise ValueError("Mapped layer reference has different authoring controls.")
    if (
        source.layers.wall_association is None
        and source.source_boundary_charts is not None
    ):
        from ._layer_core_topology import validate_closed_boundary_source

        validate_closed_boundary_source(source)
    elif any(
        value is not None
        for value in (
            source.layers.wall_association,
            source.source_wall_charts,
            source.source_boundary_charts,
            source.source_wall_projection,
            source.source_wall_association_policy,
        )
    ) and (
        source.layers.wall_association is None
        or source.source_wall_charts is None
        or source.source_boundary_charts is None
    ):
        raise ValueError(
            "Charted layer authoring requires both source atlases and wall association."
        )
    projection = source.source_wall_projection
    domain = model_archive_projection(source.source_domain)
    query = model_archive_projection(source.fidelity_source)
    if mapped:
        if (
            domain is not None
            or type(query) is not MappedReferenceBoundarySourceArchiveRecipe
        ):
            raise TypeError(
                "Mapped layer source requires its full mapped boundary authority."
            )
        authority = query.authority
        if (
            type(authority) is not NativePlcSourceArchiveRecipe
            or authority.mapped_domain is None
            or authority.fidelity_source is not None
        ):
            raise TypeError(
                "Mapped layer query requires its original full PLC authority."
            )
        query = NativeLayerMappedBoundaryArchiveRecipe(
            authority.complex,
            authority.source_id,
            authority.source_revision,
            authority.maximum_deviation,
            authority.plc_vertex_root_global_ids,
            authority.expected_binding_id,
            query.face_scope_keys,
            query.edge_scope_keys,
            query.expected_source_query_id,
        )
    elif (
        type(domain) is not MeshingDomainArchiveRecipe
        or type(query) is not MeshingDomainBoundarySourceArchiveRecipe
    ):
        raise TypeError("Layer/core authoring requires an exact meshing-domain query.")
    return NativeLayerCoreAuthoringArchiveRecipe(
        source.layers.source_wall,
        source.layers,
        control,
        policy,
        source.layer_control is not None,
        source.complex,
        domain,
        query,
        source.mapped_domain,
        None if source.reference_source is None else source.reference_source.binding_id,
        source.layers.wall_association,
        source.source_boundary_scope,
        model_archive_projection(source.source_wall_charts),
        model_archive_projection(source.source_boundary_charts),
        source.source_boundary_schedule,
        None if projection is None else projection.policy,
        None if projection is None else projection.embedding,
        None if projection is None else projection.projection_id,
        source.source_wall_association_policy,
        source.source_id,
        source.source_revision,
        source.cap_refinement,
        source.noncap_refinement,
        source.vertex_layer_ids,
        source.cap_polygon_ids,
        source.layer_regions,
        source.region_ids,
        source.region_entity_ids,
        source.region_occurrence_paths,
        source.core_region_map,
        source.maximum_deviation,
        source.core_facet_source_ids,
        source.core_facet_source_occurrence_paths,
        source.core_vertex_representatives,
        source.core_vertex_shifts,
        source.core_seam_polygon_pairs,
        source.layers.result_id,
        source.binding_id,
    )


def restore_boundary_layer_state(
    stored_layers: Any, stored_control: Any, stored_policy: Any, /
) -> tuple[Any, Any, Any]:
    """Reconstruct and independently reprove an actual retained advancing layer."""
    from copy import copy

    from ..discretization._cell_geometry_validity import (
        CellValidityCertificate,
        CellValidityPolicy,
    )
    from ..discretization._cell_mesh import CellMesh
    from ..geometry._surface_source_archive import _geometry
    from ._association import GeometryAssociation
    from ._boundary_layer import (
        _reprove_retained_boundary_layers,
        BoundaryLayerEvidence,
        BoundaryLayerMesh,
        BoundaryLayerPolicy,
    )
    from ._controls import BoundaryLayerControl

    if type(stored_policy) is not BoundaryLayerPolicy:
        raise TypeError("Stored layer policy has the wrong owner.")
    validity = CellValidityPolicy(
        relative_determinant_floor=stored_policy.validity.relative_determinant_floor,
        relative_planarity_tolerance=stored_policy.validity.relative_planarity_tolerance,
    )
    if validity.policy_id != stored_policy.validity.policy_id:
        raise ValueError("Stored layer validity policy changed.")
    policy = BoundaryLayerPolicy(
        proximity_samples=stored_policy.proximity_samples,
        proximity_fraction=stored_policy.proximity_fraction,
        curvature_fraction=stored_policy.curvature_fraction,
        medial_tolerance=stored_policy.medial_tolerance,
        reduction_factor=stored_policy.reduction_factor,
        simplex_cap=stored_policy.simplex_cap,
        validity=validity,
    )
    if policy.policy_id != stored_policy.policy_id:
        raise ValueError("Stored layer policy changed.")
    if type(stored_control) is not BoundaryLayerControl:
        raise TypeError("Stored layer control has the wrong owner.")
    control = BoundaryLayerControl.from_stored_fields(stored_control)
    stored_layers = restore_row_record_recipes(stored_layers)
    if type(stored_layers) is not BoundaryLayerMesh:
        raise TypeError("Stored layer source lost its actual accepted layer state.")
    wall = restore_model_archive_value(stored_layers.source_wall)
    wall_association = restore_row_record_recipes(
        model_archive_projection(stored_layers.wall_association)
    )
    if wall_association is not None and type(wall_association) is not GeometryAssociation:
        raise TypeError("Stored wall association lost its canonical owner.")
    if not _same_model_content(
        stored_layers.source_wall, wall
    ) or not _same_model_content(stored_layers.wall_association, wall_association):
        raise ValueError(
            "Stored layer source wall or association differs from its authoring."
        )
    layer_mesh = CellMesh.from_stored_fields(stored_layers.mesh)
    cap = (
        None
        if stored_layers.cap is None
        else CellMesh.from_stored_fields(stored_layers.cap)
    )
    obstacles = (
        None
        if stored_layers.obstacles is None
        else CellMesh.from_stored_fields(stored_layers.obstacles)
    )
    geometry = _geometry(stored_layers.validity.geometry, layer_mesh)
    if not _same_model_content(stored_layers.validity.geometry, geometry):
        raise ValueError("Stored layer validity geometry changed.")
    stored_validity = copy(stored_layers.validity)
    object.__setattr__(stored_validity, "geometry", geometry)
    certificate = CellValidityCertificate.from_stored_fields(
        stored_validity,
        layer_mesh,
        geometry,
        policy=policy.validity,
    )
    evidence = BoundaryLayerEvidence(
        **{
            field.name: getattr(stored_layers.evidence, field.name)
            for field in dataclasses.fields(stored_layers.evidence)
            if field.name != "evidence_id"
        }
    )
    if not _same_model_content(evidence, stored_layers.evidence):
        raise ValueError("Stored layer evidence fields changed.")
    positional = {
        "mesh",
        "cap",
        "wall_vertices",
        "cap_vertices",
        "layer_index",
        "validity",
        "evidence",
        "source_wall",
        "wall_association",
        "obstacles",
        "result_id",
    }
    layer_fields = {
        field.name: getattr(stored_layers, field.name)
        for field in dataclasses.fields(stored_layers)
        if field.name not in positional
    }
    layers = BoundaryLayerMesh(
        layer_mesh,
        cap,
        stored_layers.wall_vertices,
        stored_layers.cap_vertices,
        stored_layers.layer_index,
        certificate,
        evidence,
        source_wall=wall,
        wall_association=wall_association,
        obstacles=obstacles,
        **layer_fields,
    )
    if not _same_model_content(layers, stored_layers):
        raise ValueError(
            "Stored actual layer state differs from its canonical constructors."
        )
    layers = _reprove_retained_boundary_layers(
        layers,
        wall,
        control,
        policy=policy,
        wall_association=wall_association,
        obstacles=obstacles,
    )
    return layers, control, policy


def restore_layer_core_authoring(
    recipe: NativeLayerCoreAuthoringArchiveRecipe, /, *, specification: Any | None = None
) -> Any:
    """Restore the actual retained layers and their fixed-core source authority."""
    import numpy as np

    from ..geometry._domain_boundary import MeshingDomainBoundarySource
    from ..geometry._planar_embedding import PlanarEmbedding
    from ..geometry._surface_source_archive import _domain, restore_surface_source_charts
    from ..geometry.brep._projection import prepare_brep_projection
    from ..geometry.brep._projection_contracts import BRepProjectionPolicy
    from ._association import AssociationPropagationPolicy
    from ._contracts import VolumeMeshingSpec
    from ._layer_core_topology import validate_staged_layer_domain
    from ._scope import MeshingScope
    from .providers._native_sources import NativeLayerCoreSource

    _register_model_archive_types()

    if type(recipe) is not NativeLayerCoreAuthoringArchiveRecipe:
        raise TypeError("Expected exact layer/core authoring inputs.")
    if type(recipe.source_controls_bound) is not bool:
        raise TypeError("Stored source control binding must be boolean.")
    layers, control, policy = restore_boundary_layer_state(
        recipe.layers,
        recipe.control,
        recipe.policy,
    )
    wall = layers.source_wall
    wall_association = layers.wall_association
    if (
        layers.result_id != recipe.expected_layers_id
        or not _same_model_content(recipe.wall, wall)
        or not _same_model_content(recipe.wall_association, wall_association)
    ):
        raise ValueError(
            "Stored layer source wall or association differs from its authoring."
        )
    complex_ = restore_model_archive_value(recipe.complex)
    mapped = recipe.mapped_domain is not None
    if mapped:
        from ..geometry._surface_source_archive import _mapped_reference

        if recipe.source_domain is not None or not isinstance(
            recipe.reference_binding_id, str
        ):
            raise ValueError(
                "Mapped layer source lost its original reference declaration."
            )
        if type(recipe.fidelity_source) is not NativeLayerMappedBoundaryArchiveRecipe:
            raise TypeError("Mapped layer source lost its exact full boundary query.")
        if any(
            value is not None
            for value in (
                wall_association,
                recipe.source_boundary_scope,
                recipe.wall_charts,
                recipe.boundary_charts,
                recipe.boundary_schedule,
                recipe.wall_projection_policy,
                recipe.wall_projection_embedding,
                recipe.wall_projection_id,
                recipe.wall_association_policy,
            )
        ):
            raise ValueError("Mapped layer reference has unsupported wall source owners.")
        domain = None
        mapped_domain = _mapped_reference(recipe.mapped_domain)
        from ..geometry._mapped_reference_boundary import (
            MappedPlcRootAuthority,
            MappedReferenceBoundarySource,
        )
        from .providers._native_sources import NativePlcSource

        stored_query = recipe.fidelity_source
        authority = NativePlcSource(
            restore_model_archive_value(stored_query.complex),
            stored_query.source_id,
            stored_query.source_revision,
            mapped_domain=mapped_domain,
            maximum_deviation=stored_query.maximum_deviation,
            plc_vertex_root_global_ids=np.asarray(
                stored_query.plc_vertex_root_global_ids
            ),
        )
        if authority.binding_id != stored_query.authority_binding_id:
            raise ValueError("Original mapped PLC authority binding changed.")
        authority.validate_restored()
        if not isinstance(authority, MappedPlcRootAuthority):
            raise TypeError("Mapped PLC source lost its root authority.")
        query = MappedReferenceBoundarySource(
            authority,
            face_scope_keys=stored_query.face_scope_keys,
            edge_scope_keys=stored_query.edge_scope_keys,
        )
        if query.source_query_id != stored_query.query_id:
            raise ValueError("Original mapped boundary query scope changed.")
        query.require_current()
    else:
        if recipe.reference_binding_id is not None:
            raise ValueError("Unmapped layer source cannot name a mapped reference.")
        if (
            type(recipe.source_domain) is not MeshingDomainArchiveRecipe
            or type(recipe.fidelity_source)
            is not MeshingDomainBoundarySourceArchiveRecipe
        ):
            raise TypeError("Stored source lost its exact domain and query recipes.")
        if not _same_model_content(recipe.source_domain, recipe.fidelity_source.domain):
            raise ValueError("Stored source query names a different domain.")
        domain = _domain(recipe.source_domain)
        mapped_domain = None
        stored_query = recipe.fidelity_source
        query = MeshingDomainBoundarySource(
            domain,
            stored_query.patches,
            resolution=stored_query.resolution,
            boundary_occurrences=None
            if stored_query.boundary_occurrences is None
            else np.asarray(stored_query.boundary_occurrences),
            boundary_cell_ids=None
            if stored_query.boundary_cell_ids is None
            else np.asarray(stored_query.boundary_cell_ids),
            boundary_patch_ids=None
            if stored_query.boundary_patch_ids is None
            else np.asarray(stored_query.boundary_patch_ids),
        )
        if query.source_scope_id != stored_query.expected_source_scope_id:
            raise ValueError("Stored source boundary scope changed.")
        query.require_current()

    def restore_charts(value: Any, /) -> Any:
        if value is None:
            return None
        if type(value) is not SurfaceSourceChartsArchiveRecipe:
            raise TypeError("Stored layer charts have the wrong owner.")
        return restore_surface_source_charts(value)

    wall_charts = restore_charts(recipe.wall_charts)
    boundary_charts = restore_charts(recipe.boundary_charts)
    boundary_schedule = restore_model_archive_value(recipe.boundary_schedule)
    wall_projection = None
    projection_inputs = (recipe.wall_projection_policy, recipe.wall_projection_id)
    if any(value is not None for value in projection_inputs):
        if (
            type(recipe.wall_projection_policy) is not BRepProjectionPolicy
            or not isinstance(recipe.wall_projection_id, str)
            or domain is None
            or domain.brep_authority is None
        ):
            raise ValueError("Stored wall projection lost its original model or policy.")
        old = recipe.wall_projection_policy
        projection_policy = BRepProjectionPolicy(
            ambiguity_tolerance=old.ambiguity_tolerance,
            parametric_tolerance=old.parametric_tolerance,
            classifier_tolerance=old.classifier_tolerance,
        )
        if projection_policy.policy_id != old.policy_id:
            raise ValueError("Stored wall projection policy changed.")
        embedding = recipe.wall_projection_embedding
        if embedding is not None:
            if type(embedding) is not PlanarEmbedding:
                raise TypeError("Stored wall projection embedding has the wrong owner.")
            rebuilt_embedding = PlanarEmbedding(
                embedding.origin, embedding.x_axis, embedding.y_axis, embedding.normal
            )
            if rebuilt_embedding.embedding_id != embedding.embedding_id:
                raise ValueError("Stored wall projection embedding changed.")
            embedding = rebuilt_embedding
        wall_projection = prepare_brep_projection(
            domain.brep_authority, policy=projection_policy, embedding=embedding
        )
        if wall_projection.projection_id != recipe.wall_projection_id:
            raise ValueError("Stored original wall projection changed.")
    elif recipe.wall_projection_embedding is not None:
        raise ValueError("Wall projection embedding has no original projection.")
    association_policy = recipe.wall_association_policy
    if association_policy is not None:
        if type(association_policy) is not AssociationPropagationPolicy:
            raise TypeError("Stored wall association policy has the wrong owner.")
        old = association_policy
        association_policy = AssociationPropagationPolicy(
            classification_tolerance=old.classification_tolerance,
            maximum_residual=old.maximum_residual,
        )
        if association_policy.policy_id != old.policy_id:
            raise ValueError("Stored wall association policy changed.")
    if (
        recipe.source_boundary_scope is not None
        and type(recipe.source_boundary_scope) is not MeshingScope
    ):
        raise TypeError("Stored boundary scope has the wrong owner.")
    boundary_scope = (
        None
        if recipe.source_boundary_scope is None
        else MeshingScope.from_stored_fields(
            recipe.source_boundary_scope,
            source_model=None if domain is None else domain.brep_authority,
        )
    )
    source_inputs: dict[str, Any] = dict(
        vertex_layer_ids=recipe.vertex_layer_ids,
        cap_polygon_ids=recipe.cap_polygon_ids,
        layer_regions=recipe.layer_regions,
        region_ids=recipe.region_ids,
        region_entity_ids=recipe.region_entity_ids,
        region_occurrence_paths=recipe.region_occurrence_paths,
        core_region_map=recipe.core_region_map,
        source_domain=domain,
        source_boundary_scope=boundary_scope,
        source_wall_charts=wall_charts,
        source_boundary_charts=boundary_charts,
        source_boundary_schedule=boundary_schedule,
        source_wall_projection=wall_projection,
        source_wall_association_policy=association_policy,
        maximum_deviation=recipe.maximum_deviation,
        core_facet_source_ids=recipe.core_facet_source_ids,
        core_facet_source_occurrence_paths=recipe.core_facet_source_occurrence_paths,
        core_vertex_representatives=recipe.core_vertex_representatives,
        core_vertex_shifts=recipe.core_vertex_shifts,
        cap_refinement=recipe.cap_refinement,
        noncap_refinement=recipe.noncap_refinement,
        layer_control=control if recipe.source_controls_bound else None,
        layer_policy=policy if recipe.source_controls_bound else None,
        core_seam_polygon_pairs=recipe.core_seam_polygon_pairs,
    )
    reference = None
    if mapped:
        reference_inputs = dict(source_inputs)
        reference_inputs["maximum_deviation"] = None
        reference = NativeLayerCoreSource(
            layers, complex_, recipe.source_id, recipe.source_revision, **reference_inputs
        )
        if reference.binding_id != recipe.reference_binding_id:
            raise ValueError("Rebuilt original layer reference binding changed.")
    source = NativeLayerCoreSource(
        layers,
        complex_,
        recipe.source_id,
        recipe.source_revision,
        **source_inputs,
        fidelity_source=query,
        reference_source=reference,
        mapped_domain=mapped_domain,
    )
    if source.binding_id != recipe.expected_binding_id:
        raise ValueError("Rebuilt layer/core source binding changed.")
    if mapped:
        from ._layer_core_source_restore import validate_mapped_layer_core_source

        validate_mapped_layer_core_source(source)
    elif layers.wall_association is None and boundary_charts is not None:
        from ._layer_core_topology import validate_closed_boundary_source

        validate_closed_boundary_source(source)
    elif wall_charts is None and boundary_charts is None:
        validate_staged_layer_domain(source)
    else:
        from ._layer_core_brep import restrict_wall_charts
        from ._layer_wall_source_replay import (
            validate_full_boundary_charts,
            validate_wall_source,
        )

        if wall_charts is None or boundary_charts is None:
            raise ValueError(
                "Charted source requires both original boundary and wall atlases."
            )
        if not _same_model_content(wall_charts, boundary_charts):
            restricted = restrict_wall_charts(boundary_charts, wall)
            # Inward-wall construction and general restriction retain the same
            # source strata in different association tuple orders. Preserve the
            # stored order, but compare every actual owner, not only its ID.
            derived = {item.association_id: item for item in restricted.associations}
            if (
                len(derived) != len(restricted.associations)
                or len(wall_charts.associations) != len(derived)
                or len({item.association_id for item in wall_charts.associations})
                != len(derived)
                or any(
                    item.association_id not in derived
                    or not _same_model_content(item, derived[item.association_id])
                    for item in wall_charts.associations
                )
                or any(
                    not _same_model_content(
                        getattr(restricted, field.name), getattr(wall_charts, field.name)
                    )
                    for field in dataclasses.fields(wall_charts)
                    if field.name not in ("associations", "source_chart_id")
                )
            ):
                raise ValueError(
                    "Stored wall atlas is not the exact original boundary restriction."
                )

        if type(specification) is not VolumeMeshingSpec:
            raise TypeError(
                "Charted source replay requires its actual volume specification."
            )
        specification = VolumeMeshingSpec.from_stored_fields(specification)
        tolerances = tuple(
            value
            for value in (source.maximum_deviation, specification.maximum_deviation)
            if value is not None
        )
        if tolerances:
            tolerance = min(tolerances)
        elif domain is None:
            raise ValueError("Charted layer source lost its domain.")
        elif domain.accuracy == "exact_represented":
            # Exact represented source fidelity proves zero; no control is added.
            tolerance = 0.0
        else:
            raise ValueError(
                "Nonexact source fidelity requires an authored deviation tolerance."
            )
        validate_full_boundary_charts(source, tolerance)
        validate_wall_source(source, tolerance)
    return source


def restore_mapped_layer_core_boundary_source(recipe: Any, source: Any, /) -> Any:
    """Rebind a scoped internal-facet query to its restored original layer source."""
    from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
    from ._layer_core_facet_authority import MappedCoreFacetAuthority
    from ._layer_core_source_restore import validate_mapped_layer_core_source
    from .providers._native_sources import NativeLayerCoreSource

    if (
        type(recipe) is not MappedReferenceBoundarySourceArchiveRecipe
        or type(recipe.authority) is not MappedCoreFacetAuthorityArchiveRecipe
        or type(source) is not NativeLayerCoreSource
    ):
        raise TypeError("Mapped internal-facet query requires its original layer owner.")
    stored = recipe.authority
    if (
        stored.source_id != source.source_id
        or stored.source_revision != source.source_revision
        or stored.source_binding_id != source.binding_id
    ):
        raise ValueError("Mapped internal-facet query belongs to another layer source.")
    validate_mapped_layer_core_source(source)
    authority = MappedCoreFacetAuthority(source, stored.facet_rows)
    if (
        authority.binding_id != stored.expected_binding_id
        or authority.root_relation.relation_id != stored.expected_relation_id
    ):
        raise ValueError(
            "Mapped internal-facet relation changed from the original source."
        )
    query = MappedReferenceBoundarySource(
        authority,
        face_scope_keys=recipe.face_scope_keys,
        edge_scope_keys=recipe.edge_scope_keys,
    )
    if query.source_query_id != recipe.expected_source_query_id:
        raise ValueError("Mapped internal-facet query scope changed.")
    query.require_current()
    return query


def _require_layer_root_transfer(
    result: Any, transfer: Any, /, *, computed_root: Any | None = None
) -> None:
    root = result.layer_core_root_geometry
    if root is None:
        if computed_root is not None:
            raise ValueError("A computed layer root requires its actual retained root.")
        if (
            result.layer_core_overlap_stencils is not None
            or result.layer_core_column_certification is not None
        ):
            raise ValueError("Layer overlap evidence requires its original root owner.")
        return
    from ._layer_core_association_transfer import NativeLayerCoreAssociationTransfer
    from ._layer_core_subdivision import (
        require_descendant_overlap_source_provenance,
        require_refined_column_certification,
    )

    if type(transfer) is not NativeLayerCoreAssociationTransfer:
        raise TypeError(
            "Layer root evidence requires its original source-bound transfer."
        )
    if computed_root is not None:
        from ._layer_core_root_geometry import ComputedLayerCoreRoot

        if type(computed_root) is not ComputedLayerCoreRoot:
            raise TypeError("Layer root replay requires a computed root proof.")
    if transfer.source.mapped_domain is not None:
        if (
            root.reference_mesh is None
            or result.layer_core_overlap_stencils is not None
            or result.layer_core_column_certification is not None
        ):
            raise ValueError(
                "Mapped layer endpoints require their actual reference root."
            )
        # Endpoint replay still checks the complete current mapped geometry and
        # associations; a fresh root proof may be reused only for exact operands.
        if computed_root is None:
            transfer.validate_source(result)
        else:
            transfer._validate_source(result, computed_root=computed_root)
        return
    if computed_root is None:
        root.require_current(transfer.source, transfer.specification)
    else:
        computed_root.require(
            transfer.source, transfer.specification, root.mesh, root.geometry, root
        )
    require_descendant_overlap_source_provenance(transfer.source, result)
    if result.layer_core_column_certification is None:
        raise ValueError("Layer descendant lost its current column certification.")
    require_refined_column_certification(
        transfer.source,
        result.mesh,
        result.geometry,
        transfer.specification,
        result.layer_core_column_certification,
    )


def _require_layer_result_recipe_inputs(
    result: Any, source: Any, specification: Any, /
) -> None:
    """Require the actual accepted endpoint and its typed authoring inputs."""
    from ._contracts import VolumeMeshingSpec
    from ._result import CellMeshingResult
    from .providers._native_sources import NativeLayerCoreSource

    if (
        type(result) is not CellMeshingResult
        or type(source) is not NativeLayerCoreSource
        or type(specification) is not VolumeMeshingSpec
        or result.certification is None
        or not result.certification.passed
    ):
        raise TypeError(
            "Layer snapshot requires its actual certified result and authoring."
        )


def layer_core_accepted_result_recipe(
    result: Any, source: Any, specification: Any, /
) -> LayerCoreAcceptedResultArchiveRecipe:
    """Retain an actual accepted result with its original source/request binding."""
    from ._layer_core_association_transfer import NativeLayerCoreAssociationTransfer

    _require_layer_result_recipe_inputs(result, source, specification)
    transfer = NativeLayerCoreAssociationTransfer(source, specification)
    if result.layer_core_root_geometry is None or source.mapped_domain is None:
        transfer.validate_source(result)
    _require_layer_root_transfer(result, transfer)
    return LayerCoreAcceptedResultArchiveRecipe(
        result, source.binding_id, specification.specification_id
    )


def _restore_layer_overlap(value: Any, /) -> Any:
    from ._layer_core_subdivision import OverlapSourceStencils

    if value is None:
        return None
    if type(value) is not OverlapSourceStencils:
        raise TypeError("Stored layer overlap evidence has the wrong owner.")
    rebuilt = OverlapSourceStencils(
        value.source_binding_id, value.target_mesh_id, value.rows
    )
    if not _same_model_content(rebuilt, value):
        raise ValueError("Stored layer overlap stencil identity changed.")
    return rebuilt


def _restore_layer_column(value: Any, /) -> Any:
    from ._layer_core_subdivision import (
        NearestWallPointEvidence,
        RefinedColumnCertification,
        RefinedColumnEvidence,
    )

    if value is None:
        return None
    if type(value) is not RefinedColumnCertification:
        raise TypeError("Stored layer column evidence has the wrong owner.")
    columns = tuple(
        RefinedColumnEvidence(
            column.source_columns,
            column.target_vertices,
            tuple(
                NearestWallPointEvidence(row.squared_distance, row.supports)
                for row in column.wall_distances
            ),
            column.maximum_stretch_squared,
        )
        for column in value.columns
    )
    rebuilt = RefinedColumnCertification(
        value.source_binding_id,
        value.specification_id,
        value.control_id,
        value.policy_id,
        columns,
    )
    if not _same_model_content(rebuilt, value):
        raise ValueError("Stored layer column certification changed.")
    return rebuilt


def restore_layer_core_root_geometry(
    root: Any, source: Any, specification: Any, /
) -> Any:
    """Independently reconstruct and reprove an actual retained layer root."""
    return _restore_layer_core_computed_root(root, source, specification).root


def _restore_layer_core_computed_root(
    root: Any, source: Any, specification: Any, /
) -> Any:
    """Reconstruct the root and retain its freshly computed operation-local proof."""
    from ..discretization._cell_mesh import CellMesh
    from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
    from ..geometry._surface_source_archive import _geometry
    from ._contracts import VolumeMeshingSpec
    from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry

    _register_model_archive_types()

    if type(root) is not LayerCoreRootGeometry:
        raise TypeError("Stored layer root has the wrong owner.")
    root_specification = VolumeMeshingSpec.from_stored_fields(root.specification)
    if not _same_model_content(root_specification, specification):
        raise ValueError("Stored layer root binds another volume request.")
    root_mesh = CellMesh.from_stored_fields(root.mesh)
    root_geometry = _geometry(root.geometry, root_mesh)
    reference_mesh = (
        None
        if root.reference_mesh is None
        else CellMesh.from_stored_fields(root.reference_mesh)
    )
    if (reference_mesh is None) != (root.reference_geometry is None):
        raise ValueError("Stored mapped layer root lost its reference geometry owner.")
    reference_geometry = (
        None
        if reference_mesh is None
        else _geometry(root.reference_geometry, reference_mesh)
    )
    computed = ComputedLayerCoreRoot(
        source,
        specification,
        root_mesh,
        root_geometry,
        ExactPlcCellGeometrySource.from_stored_fields(root.overlap_exact_geometry),
        _restore_layer_overlap(root.overlap_stencils),
        _restore_layer_column(root.column_certification),
        restore_model_archive_value(root.layer_root_partition),
        reference_mesh=reference_mesh,
        reference_geometry=reference_geometry,
        reference_cell_regions=root.reference_cell_regions,
    )
    if not _same_model_content(computed.root, root):
        raise ValueError("Stored layer root fields or identity changed.")
    return computed


def restore_layer_core_accepted_result(
    recipe: LayerCoreAcceptedResultArchiveRecipe, source: Any, specification: Any, /
) -> Any:
    """Independently reprove a retained result against its source and request.

    Archive content IDs authenticate bytes, not mathematical provenance. Root,
    carrier support, associations and acceptance are revalidated from the actual
    reconstructed scientific inputs.
    """
    result, _ = _restore_layer_core_accepted_result(recipe, source, specification, None)
    return result


def _require_layer_endpoint_coordinate_origin(
    geometry: Any, root: Any | None, source_geometry: Any | None, /
) -> None:
    """Check a retained bank against an actual root or transition source, if bound."""
    from ._mixed_adaptation import _require_retained_bank_origin, _RetainedCoordinateBank

    if type(geometry.source_map) is not _RetainedCoordinateBank:
        return
    origin = root.geometry if root is not None else source_geometry
    if origin is not None:
        _require_retained_bank_origin(origin, geometry)


def _restore_layer_core_accepted_result(
    recipe: LayerCoreAcceptedResultArchiveRecipe,
    source: Any,
    specification: Any,
    computed_root: Any | None,
    /,
    *,
    computed_embedding_out: list[Any] | None = None,
    validated_chart_parent_embedding: Any | None = None,
    source_geometry: Any | None = None,
) -> tuple[Any, Any | None]:
    """Restore an endpoint, optionally using this operation's exact root proof."""
    from ..discretization._cell_mesh import CellMesh
    from ..geometry._source_boundary_records import SourceBoundaryQuery
    from ..geometry._surface_source_archive import (
        _geometry,
        restore_surface_source_charts,
    )
    from ._contracts import VolumeMeshingSpec
    from ._layer_core_association_transfer import NativeLayerCoreAssociationTransfer
    from ._layer_core_carrier_witness import NativeLayerCoreCarrierWitness
    from ._layer_curving_records import LayerCurvingSourceEvidence
    from ._result import CellMeshingResult
    from .providers._native_sources import NativeLayerCoreSource

    if type(recipe) is not LayerCoreAcceptedResultArchiveRecipe:
        raise TypeError("Expected a retained layer/core accepted-result recipe.")
    if (
        type(source) is not NativeLayerCoreSource
        or type(specification) is not VolumeMeshingSpec
    ):
        raise TypeError(
            "Layer result replay requires its actual source and request owners."
        )
    if (
        source.binding_id != recipe.source_binding_id
        or specification.specification_id != recipe.specification_id
    ):
        raise ValueError("Stored layer result belongs to another source or request.")
    if computed_root is None:
        specification = VolumeMeshingSpec.from_stored_fields(specification)
    else:
        from ._layer_core_root_geometry import ComputedLayerCoreRoot

        if type(computed_root) is not ComputedLayerCoreRoot:
            raise TypeError("Layer endpoint replay requires a computed root proof.")
        computed_root.require(
            source,
            specification,
            computed_root.mesh,
            computed_root.geometry,
            computed_root.root,
        )
    stored = restore_row_record_recipes(model_archive_projection(recipe.result))
    if type(stored) is not CellMeshingResult:
        raise TypeError("Layer snapshot lost its actual accepted result.")

    carrier = stored.layer_core_carrier_witness
    if carrier is not None:
        if type(carrier) is not NativeLayerCoreCarrierWitness:
            raise TypeError("Stored layer carrier inventory has the wrong owner.")
        rebuilt_carrier = NativeLayerCoreCarrierWitness(source, specification)
        if not _same_model_content(carrier, rebuilt_carrier):
            raise ValueError("Stored native carrier inventory changed.")
        carrier = rebuilt_carrier
    overlap = _restore_layer_overlap(stored.layer_core_overlap_stencils)
    column = _restore_layer_column(stored.layer_core_column_certification)
    stored_root = stored.layer_core_root_geometry
    if stored_root is None:
        if computed_root is not None:
            raise ValueError("Stored layer endpoint lost its proved root.")
        root = None
    elif computed_root is None:
        computed_root = _restore_layer_core_computed_root(
            stored_root, source, specification
        )
        root = computed_root.root
    else:
        from ..geometry._mesh_certificates import _same_tree_content

        root = computed_root.root
        if not _same_tree_content(
            model_archive_root_projection(stored_root),
            model_archive_root_projection(root),
        ) or not _same_model_content(stored_root, root):
            raise ValueError("Stored layer endpoint has a different proved root.")
    curving = stored.layer_curving_evidence
    if curving is not None:
        if type(curving) is not LayerCurvingSourceEvidence:
            raise TypeError("Stored layer curving evidence has the wrong owner.")
        query_recipe = curving.source_query
        if (
            type(query_recipe) is MappedReferenceBoundarySourceArchiveRecipe
            and type(query_recipe.authority) is MappedCoreFacetAuthorityArchiveRecipe
        ):
            query = restore_mapped_layer_core_boundary_source(query_recipe, source)
        else:
            query = restore_model_archive_value(query_recipe)
        if not isinstance(query, SourceBoundaryQuery):
            raise TypeError("Stored layer curving evidence lost its source query.")
        charts_recipe = curving.source_surface_charts
        if charts_recipe is None:
            charts = None
        elif type(charts_recipe) is SurfaceSourceChartsArchiveRecipe:
            charts = restore_surface_source_charts(charts_recipe)
            if source.source_wall_charts is None or not _same_model_content(
                charts, source.source_wall_charts
            ):
                raise ValueError(
                    "Stored layer curving atlas differs from its source authority."
                )
        else:
            raise TypeError("Stored layer curving evidence lost its source atlas.")
        curving_mesh = CellMesh.from_stored_fields(stored.mesh)
        curving_geometry = _geometry(stored.geometry, curving_mesh)
        curving = LayerCurvingSourceEvidence.from_stored_fields(
            curving,
            curving_mesh,
            curving_geometry,
            source,
            query,
            charts,
        )
    result = CellMeshingResult.from_stored_fields(
        stored,
        layer_core_carrier_witness=carrier,
        layer_curving_evidence=curving,
        layer_core_overlap_stencils=overlap,
        layer_core_column_certification=column,
        layer_core_root_geometry=root,
        layer_core_feature_source=source,
        computed_root=computed_root,
        _computed_embedding_out=computed_embedding_out,
        _validated_chart_parent_embedding=validated_chart_parent_embedding,
    )
    _require_layer_endpoint_coordinate_origin(result.geometry, root, source_geometry)
    if root is not None and source.mapped_domain is not None:
        # The final result constructor has already replayed all mapped source
        # associations. Keep the external source/root binding checks here.
        if (
            result.layer_core_root_geometry is not root
            or result.certification is None
            or result.certification.inputs.feature_source is not source
            or root.specification is not specification
            or root.reference_mesh is None
            or result.layer_core_overlap_stencils is not None
            or result.layer_core_column_certification is not None
            or computed_root is None
        ):
            raise ValueError("Restored mapped endpoint lost its actual source or root.")
        computed_root.require(source, specification, root.mesh, root.geometry, root)
    else:
        transfer = NativeLayerCoreAssociationTransfer(source, specification)
        if result.layer_core_root_geometry is None:
            transfer.validate_source(result)
        else:
            transfer._validate_source(result, computed_root=computed_root)
        _require_layer_root_transfer(result, transfer, computed_root=computed_root)
    return result, computed_root


def mixed_adaptation_policy_recipe(policy: Any, /) -> MixedAdaptationPolicyArchiveRecipe:
    """Capture a mixed policy without serializing its live association capability."""
    from ._adaptation import MeshAdaptationPolicy, MeshAdaptationRoute

    if (
        type(policy) is not MeshAdaptationPolicy
        or policy.route is not MeshAdaptationRoute.NATIVE_MIXED
    ):
        raise TypeError("Expected the actual native mixed adaptation policy.")
    if policy.association_transfer is None or any(
        value is not None
        for value in (
            policy.device_policy,
            policy.provider,
            policy.provider_options,
            policy.distribution,
            policy.partition_policy,
        )
    ):
        raise ValueError("Mixed policy requires a source-bound association owner.")
    return MixedAdaptationPolicyArchiveRecipe(
        policy.protected_scopes,
        policy.compatibility.value,
        policy.predicate_mode.value,
        policy.relocation,
        policy.audit_policy,
        policy.association_transfer.transfer_id,
        policy.policy_id,
    )


def restore_mixed_adaptation_policy(
    recipe: MixedAdaptationPolicyArchiveRecipe, association_transfer: Any, /
) -> Any:
    """Rebind stored policy inputs to a freshly reconstructed association owner."""
    from .._geometry_predicates import PredicateMode
    from ..discretization._cell_geometry_validity import CellValidityPolicy
    from ._adaptation import MeshAdaptationPolicy, MeshAdaptationRoute
    from ._audit import CellMeshAuditPolicy
    from ._bisection import BisectionCompatibility

    if type(recipe) is not MixedAdaptationPolicyArchiveRecipe:
        raise TypeError("Expected canonical mixed adaptation policy inputs.")
    if (
        getattr(association_transfer, "transfer_id", None)
        != recipe.association_transfer_id
    ):
        raise ValueError("Rebuilt association transfer does not match the stored policy.")
    stored = recipe.audit_policy
    if type(stored) is not CellMeshAuditPolicy:
        raise TypeError("Stored adaptation audit policy has the wrong owner.")
    validity = CellValidityPolicy(
        relative_determinant_floor=stored.validity_policy.relative_determinant_floor,
        relative_planarity_tolerance=stored.validity_policy.relative_planarity_tolerance,
    )
    if validity.policy_id != stored.validity_policy.policy_id:
        raise ValueError("Stored audit validity policy changed.")
    names = (
        "unused_entities",
        "require_complete_association",
        "minimum_measure",
        "minimum_mean_ratio",
        "maximum_aspect_ratio",
        "coincident_vertices",
        "coincident_vertex_tolerance",
        "duplicate_cells",
        "nonmanifold",
        "inconsistent_orientation",
        "watertight_boundary",
        "self_intersection",
        "invalid_geometry",
        "unresolved",
    )
    audit = CellMeshAuditPolicy(
        **{name: getattr(stored, name) for name in names}, validity_policy=validity
    )
    if audit.policy_id != stored.policy_id:
        raise ValueError("Stored adaptation audit policy changed.")
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_MIXED,
        protected_scopes=tuple(
            restore_model_archive_value(x) for x in recipe.protected_scopes
        ),
        compatibility=BisectionCompatibility(recipe.compatibility),
        predicate_mode=PredicateMode(recipe.predicate_mode),
        relocation=recipe.relocation,
        audit_policy=audit,
        association_transfer=association_transfer,
    )
    if policy.policy_id != recipe.expected_policy_id:
        raise ValueError("Rebuilt mixed adaptation policy identity changed.")
    return policy


def _restore_layer_geometry_transition(stored: Any, source: Any, target: Any, /) -> Any:
    """Reprove retained exact geometry maps against the actual accepted endpoints."""
    from ..discretization._cell_geometry_transition_types import (
        CellGeometryTransition,
        CellGeometryTransitionEvidence,
    )
    from ._mixed_adaptation import _require_retained_bank_origin

    if type(stored) is not CellGeometryTransition:
        raise TypeError("Layer adaptation lost its exact geometry transition.")
    original = stored.evidence
    evidence = CellGeometryTransitionEvidence(
        original.kind,
        reference_map_count=original.reference_map_count,
        refinement_map_count=original.refinement_map_count,
        coarsening_map_count=original.coarsening_map_count,
        continuity_residual=original.continuity_residual,
        approximation_bound=original.approximation_bound,
    )
    if evidence.evidence_id != original.evidence_id:
        raise ValueError("Stored geometry transition evidence changed.")
    _require_retained_bank_origin(source.geometry, target.geometry)
    rebuilt = CellGeometryTransition(
        source.mesh,
        source.geometry,
        target.mesh,
        target.geometry,
        evidence=evidence,
        reference_map=restore_model_archive_value(stored.reference_map),
        refinement=restore_model_archive_value(stored.refinement),
        coarsening=restore_model_archive_value(stored.coarsening),
        coarsening_partition=restore_model_archive_value(stored.coarsening_partition),
    )
    if rebuilt.transition_id != stored.transition_id or not _same_model_content(
        rebuilt, stored
    ):
        raise ValueError("Stored geometry transition differs from its actual endpoints.")
    return rebuilt


def _require_layer_adaptation_recipe_inputs(prepared: Any, result: Any, /) -> None:
    """Require the real mixed adaptation and its exact declarative request."""
    from ._adaptation import (
        MarkedMeshAdaptation,
        MeshAdaptationResult,
        PreparedMeshAdaptation,
    )
    from ._mixed_adaptation import MixedAdaptationHierarchy

    if (
        type(prepared) is not PreparedMeshAdaptation
        or type(result) is not MeshAdaptationResult
        or type(prepared.request) is not MarkedMeshAdaptation
        or prepared.prior_transition is not None
        or prepared.material_reference is not None
        or result.distribution is not None
        or result.prepared_id != prepared.prepared_id
        or result.source.result_id != prepared.source.result_id
        or type(result.hierarchy) is not MixedAdaptationHierarchy
        or result.transition is None
        or result.transition.geometry_transition is None
    ):
        raise ValueError("Layer snapshot requires its actual bound mixed adaptation.")


def layer_core_accepted_adaptation_recipe(
    prepared: Any, result: Any, /
) -> LayerCoreAcceptedAdaptationArchiveRecipe:
    """Retain the actual mixed result and its original declarative request."""
    _require_layer_adaptation_recipe_inputs(prepared, result)
    _require_layer_root_transfer(result.target, prepared.policy.association_transfer)
    return LayerCoreAcceptedAdaptationArchiveRecipe(
        result,
        prepared.request,
        mixed_adaptation_policy_recipe(prepared.policy),
        result.source.result_id,
        result.target.result_id,
        result.result_id,
    )


def _layer_core_accepted_lifecycle_recipes(
    initial: Any, prepared: Any, result: Any, source: Any, specification: Any, /
) -> tuple[
    LayerCoreAcceptedResultArchiveRecipe,
    LayerCoreAcceptedAdaptationArchiveRecipe,
    LayerCoreAcceptedResultArchiveRecipe,
]:
    """Reprove shared mapped root once and each actual endpoint independently."""
    from ._layer_core_association_transfer import NativeLayerCoreAssociationTransfer
    from ._layer_core_root_geometry import ComputedLayerCoreRoot

    _require_layer_result_recipe_inputs(initial, source, specification)
    _require_layer_adaptation_recipe_inputs(prepared, result)
    _require_layer_result_recipe_inputs(result.target, source, specification)
    if (
        initial is not result.source and not _same_model_content(initial, result.source)
    ) or (
        initial is not prepared.source
        and not _same_model_content(initial, prepared.source)
    ):
        raise ValueError(
            "Layer adaptation does not retain its actual accepted source endpoint."
        )
    transfer = prepared.policy.association_transfer
    root = initial.layer_core_root_geometry
    if (
        source.mapped_domain is None
        or root is None
        or result.target.layer_core_root_geometry is not root
        or type(transfer) is not NativeLayerCoreAssociationTransfer
        or transfer.source is not source
        or transfer.specification is not specification
    ):
        return (
            layer_core_accepted_result_recipe(initial, source, specification),
            layer_core_accepted_adaptation_recipe(prepared, result),
            layer_core_accepted_result_recipe(result.target, source, specification),
        )
    canonical_transfer = NativeLayerCoreAssociationTransfer(source, specification)
    if transfer.transfer_id != canonical_transfer.transfer_id:
        raise ValueError(
            "Layer adaptation association transfer changed from its source owner."
        )
    for endpoint in (initial, result.target):
        if (
            root.reference_mesh is None
            or endpoint.layer_core_overlap_stencils is not None
            or endpoint.layer_core_column_certification is not None
        ):
            raise ValueError(
                "Mapped layer endpoints require their actual reference root."
            )
    computed_root = ComputedLayerCoreRoot.reprove(source, specification, root)
    transfer._validate_source(result.target, computed_root=computed_root)
    transfer._validate_source(initial, computed_root=computed_root)
    return (
        LayerCoreAcceptedResultArchiveRecipe(
            initial, source.binding_id, specification.specification_id
        ),
        LayerCoreAcceptedAdaptationArchiveRecipe(
            result,
            prepared.request,
            mixed_adaptation_policy_recipe(prepared.policy),
            result.source.result_id,
            result.target.result_id,
            result.result_id,
        ),
        LayerCoreAcceptedResultArchiveRecipe(
            result.target, source.binding_id, specification.specification_id
        ),
    )


def restore_layer_core_accepted_adaptation(
    recipe: LayerCoreAcceptedAdaptationArchiveRecipe,
    source: Any,
    target: Any,
    association_transfer: Any,
    /,
) -> Any:
    """Reprove the actual retained mixed transition without executing its request."""
    _require_layer_root_transfer(target, association_transfer)
    return _restore_layer_core_accepted_adaptation(
        recipe, source, target, association_transfer, None
    )


def _restore_layer_core_accepted_adaptation(
    recipe: LayerCoreAcceptedAdaptationArchiveRecipe,
    source: Any,
    target: Any,
    association_transfer: Any,
    computed_root: Any | None,
    /,
) -> Any:
    from ._adaptation import MeshAdaptationResult

    if type(recipe) is not LayerCoreAcceptedAdaptationArchiveRecipe:
        raise TypeError("Expected a retained layer/core adaptation recipe.")
    if (
        source.result_id != recipe.source_result_id
        or target.result_id != recipe.target_result_id
    ):
        raise ValueError("Stored mixed adaptation names different accepted endpoints.")
    raw = restore_row_record_recipes(model_archive_projection(recipe.result))
    if type(raw) is not MeshAdaptationResult or raw.transition is None:
        raise TypeError("Stored layer adaptation lost its actual transition owner.")
    if (
        raw.result_id != recipe.expected_result_id
        or not _same_model_content(raw.source, source)
        or not _same_model_content(raw.target, target)
    ):
        raise ValueError(
            "Stored mixed adaptation differs from its authenticated endpoints."
        )
    request = restore_model_archive_value(recipe.request)
    if type(recipe.policy) is not MixedAdaptationPolicyArchiveRecipe:
        raise TypeError("Stored adaptation policy has the wrong owner.")
    policy = restore_mixed_adaptation_policy(recipe.policy, association_transfer)
    hierarchy = restore_model_archive_value(raw.hierarchy)
    geometry = _restore_layer_geometry_transition(
        raw.transition.geometry_transition, source, target
    )
    restored = MeshAdaptationResult.from_stored_fields(
        raw,
        source=source,
        target=target,
        policy=policy,
        request=request,
        geometry_transition=geometry,
        hierarchy=hierarchy,
        expected_result_id=recipe.expected_result_id,
        _computed_layer_root=computed_root,
    )
    return restored


def _restore_layer_core_accepted_lifecycle(
    initial_recipe: LayerCoreAcceptedResultArchiveRecipe,
    refined_recipe: LayerCoreAcceptedResultArchiveRecipe,
    adaptation_recipe: LayerCoreAcceptedAdaptationArchiveRecipe,
    source: Any,
    specification: Any,
    /,
) -> tuple[Any, Any, Any, Any]:
    """Restore independently checked endpoints with one operation-local root proof."""
    from ._layer_core_association_transfer import NativeLayerCoreAssociationTransfer
    from ._result import CellMeshingResult

    parent_embeddings: list[Any] = []
    initial, computed_root = _restore_layer_core_accepted_result(
        initial_recipe,
        source,
        specification,
        None,
        computed_embedding_out=parent_embeddings,
    )
    if initial.result_id != adaptation_recipe.source_result_id:
        raise ValueError("Stored mixed adaptation names a different source endpoint.")
    operation_specification = (
        specification if computed_root is None else computed_root.specification
    )
    domain = source.mapped_domain
    stored_target = refined_recipe.result
    if type(stored_target) is not CellMeshingResult:
        raise TypeError("Layer snapshot lost its actual accepted result.")
    restriction = stored_target.geometry.restriction_source
    parent_embedding = (
        parent_embeddings[0]
        if parent_embeddings
        and domain is not None
        and restriction is not None
        and (
            restriction.source_geometry_id != domain.reference_geometry_id
            or restriction.source_topology_id != domain.reference_mesh.topology_id
        )
        else None
    )
    target, target_root = _restore_layer_core_accepted_result(
        refined_recipe,
        source,
        operation_specification,
        computed_root,
        validated_chart_parent_embedding=parent_embedding,
        source_geometry=initial.geometry,
    )
    if computed_root is not None and target_root is not computed_root:
        raise ValueError("Layer endpoints did not retain the same computed root.")
    transfer = NativeLayerCoreAssociationTransfer(source, operation_specification)
    if computed_root is not None and domain is not None:
        root = target.layer_core_root_geometry
        if (
            root is not computed_root.root
            or transfer.source is not source
            or transfer.specification is not operation_specification
            or target.certification is None
            or target.certification.inputs.feature_source is not source
            or target.certification.inputs.source is not source.fidelity_source
            or target.certification.inputs.fidelity_witness is None
            or target.certification.inputs.layer_core_root_geometry is not root
            or root.specification is not operation_specification
            or root.reference_mesh is None
            or target.layer_core_overlap_stencils is not None
            or target.layer_core_column_certification is not None
        ):
            raise ValueError("Restored mapped transition lost its actual source or root.")
        computed_root.require(
            source, operation_specification, root.mesh, root.geometry, root
        )
    else:
        _require_layer_root_transfer(target, transfer, computed_root=computed_root)
    adaptation = _restore_layer_core_accepted_adaptation(
        adaptation_recipe, initial, target, transfer, computed_root
    )
    return initial, target, adaptation, transfer


def finite_element_field_banks_recipe(
    space: Any, layouts: tuple[Any, ...], fields: tuple[Any, ...], /
) -> FiniteElementFieldBanksArchiveRecipe:
    """Capture typed field coefficients against an already prepared FE space."""
    from ..solver._finite_element_schedule import FiniteElementAcceptedFieldLayout

    if any(type(layout) is not FiniteElementAcceptedFieldLayout for layout in layouts):
        raise TypeError("Field banks require canonical accepted FE layouts.")
    recipe = FiniteElementFieldBanksArchiveRecipe(
        tuple(layout.recipe() for layout in layouts),
        tuple(fields),
        space.mesh.topology_id,
        space.prepared_id,
    )
    restore_finite_element_field_banks(recipe, space)
    return recipe


def restore_finite_element_field_banks(
    recipe: FiniteElementFieldBanksArchiveRecipe, space: Any, /
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Bind stored field layouts and coefficients to an independently prepared space."""
    import numpy as np

    from ..discretization.fem._generic import FiniteElementDiscretization
    from ..solver._finite_element_schedule import (
        FiniteElementAcceptedFieldLayout,
        FiniteElementAcceptedFieldLayoutRecipe,
    )

    if (
        type(recipe) is not FiniteElementFieldBanksArchiveRecipe
        or type(space) is not FiniteElementDiscretization
    ):
        raise TypeError("Field-bank restore needs its canonical recipe and FE space.")
    if (
        recipe.topology_id != space.mesh.topology_id
        or recipe.prepared_id != space.prepared_id
    ):
        raise ValueError("Stored field banks do not bind the current FE space.")
    layout_values = []
    for item in recipe.field_layouts:
        if type(item) is not FiniteElementAcceptedFieldLayoutRecipe:
            raise TypeError("Stored FE field layouts have the wrong owner.")
        layout_values.append(FiniteElementAcceptedFieldLayout.from_recipe(item))
    layouts = tuple(layout_values)
    if len(layouts) != len(space.dof_maps) or len(recipe.fields) != len(layouts):
        raise ValueError("Stored FE fields and prepared layout counts differ.")
    for layout, field, field_space, dofs, elements in zip(
        layouts,
        recipe.fields,
        space.field_spaces,
        space.dof_maps,
        space.elements,
        strict=True,
    ):
        if (
            layout.name != field_space.name
            or layout.storage_kind != "canonical_dofs"
            or layout.periodic_quotient
        ):
            raise ValueError("Stored FE field name or storage does not bind the layout.")
        expected = layout.field_spec.resolve(space.mesh)
        if tuple(element.element_id for element in expected) != tuple(
            element.element_id for element in elements
        ):
            raise ValueError(
                "Stored FE reference elements differ from the prepared space."
            )
        if np.asarray(field).shape != (dofs.global_dof_count, *layout.component_shape):
            raise ValueError("Stored FE coefficient shape differs from its DOF layout.")
    return layouts, recipe.fields


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
    "layer_core_accepted_adaptation_recipe",
    "restore_layer_core_accepted_adaptation",
    "layer_core_accepted_result_recipe",
    "restore_layer_core_accepted_result",
    "restore_layer_core_root_geometry",
    "overset_registration_recipe",
    "restore_overset_registration",
    "planar_motion_advance_recipe",
    "restore_planar_motion_advance",
    "model_archive_content_id",
    "mixed_adaptation_policy_recipe",
    "restore_mixed_adaptation_policy",
    "finite_element_field_banks_recipe",
    "restore_finite_element_field_banks",
    "layer_core_authoring_recipe",
    "restore_layer_core_authoring",
    "read_model_archive",
    "read_model_archive_fields",
    "read_model_artifact",
    "write_model_archive",
    "write_model_artifact",
]
