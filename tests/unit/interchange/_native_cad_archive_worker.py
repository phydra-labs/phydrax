from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn, TYPE_CHECKING, TypeVar


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tests._support.layer_curving_fixtures import (
    make_layer_curving_archive_case,
    make_mapped_layer_curving_archive_case,
)
from tests._support.volume_material_map_fixtures import (
    make_curved_material_refinement,
    make_material_coverage_pair,
    make_material_source,
    reference_map_witness,
    ROOT_REFERENCE_TETRAHEDRON,
)


if TYPE_CHECKING:
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.discretization._cell_geometry import (
        ScalarCoordinateElement,
    )
    from phydrax.discretization._cell_geometry_transition_types import (
        CellGeometryTransition,
    )
    from phydrax.discretization._fv_precision import FiniteVolumePrecisionPolicy
    from phydrax.discretization._nested_reference_types import (
        NestedReferenceWitnesses,
        ReferenceCellMapWitnesses,
    )
    from phydrax.discretization._sphere_chart_records import (
        PreparedSphereChartDeformation,
    )
    from phydrax.discretization._surface_chart_records import (
        PreparedSurfaceChartDeformation,
        PreparedSurfaceChartPiece,
    )
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.discretization.fem._de_rham import (
        FiniteElementTransferResult,
        SparseCommutingEvidence,
        SparseTransferEntryEvidence,
    )
    from phydrax.discretization.finite_volume._owner_local_remap import (
        PreparedOwnerLocalConservativeRemap,
    )
    from phydrax.geometry._mapped_reference_boundary import MappedReferenceBoundarySource
    from phydrax.geometry._sphere_material_atlas import SphereMaterialCellAtlas
    from phydrax.geometry._surface_source_support import (
        PreparedSurfaceSourceSupport,
        SurfaceSourceRootAtlas,
    )
    from phydrax.geometry.implicit._source_charts import (
        ImplicitRootChartEvidence,
        Interval,
        RootEdgeTriangle,
        SourceCellKey,
    )
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.linalg._spaces import AbstractVectorSpace
    from phydrax.meshing._boundary_layer import BoundaryLayerEvidence
    from phydrax.meshing._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
    from phydrax.meshing._layer_curving import LayerCurvingConstructionResult
    from phydrax.meshing._layer_curving_records import (
        LayerColumnProjectionIdentity,
        LayerCurvingImplicitRootTrace,
        LayerCurvingMappedChartTrace,
        LayerCurvingProjectionTrace,
        LayerCurvingSourceEvidence,
    )
    from phydrax.meshing._lineage import (
        EntityLineage,
        MeshLineage,
    )
    from phydrax.meshing._surface_boundary_sources import (
        SurfaceChartBoundarySource,
        SurfaceSubdivisionBoundarySource,
    )
    from phydrax.meshing.providers._native_sources import NativePlcSource
    from phydrax.solver._finite_volume_runtime import FiniteVolumeValidatedRuntimeState

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._interpolation import TMesh2D
from phydrax._model._structure import (
    AdaptiveImplicitBoundarySourceArchiveRecipe,
    AnalyticImplicitProfileArchiveRecipe,
    FiniteElementAcceptedStateArchiveRecipe,
    FiniteVolumeRuntimeStateArchiveRecipe,
    MappedReferenceBoundarySourceArchiveRecipe,
    MeshingDomainArchiveRecipe,
    MeshingDomainBoundarySourceArchiveRecipe,
    NativeImplicitBoundaryQueryArchiveRecipe,
    NativeMappedHexSourceArchiveRecipe,
    NativePlcSourceArchiveRecipe,
    ParametricCurveBoundarySourceArchiveRecipe,
    PiecewiseLinearBoundarySourceArchiveRecipe,
    PiecewiseLinearComplexArchiveRecipe,
    PreparedSphereChartDeformationArchiveRecipe,
    PreparedSurfaceSourceSupportArchiveRecipe,
    ReconstructedTriangleBoundarySourceArchiveRecipe,
    SphereMaterialCellAtlasArchiveRecipe,
    SurfaceChartBoundarySourceArchiveRecipe,
    SurfaceEnvelopeSourceArchiveRecipe,
    SurfaceNativeRestrictionBoundarySourceArchiveRecipe,
    SurfaceSourceChartsArchiveRecipe,
    SurfaceSubdivisionBoundarySourceArchiveRecipe,
)
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._axis import AxisDiscretization
from phydrax.discretization._axis_domain import AxisDomain
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    CellVertexGeometryElement,
    PerCellRestrictedCellGeometryElement,
    RationalComposedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import (
    CoordinateSourceBank,
    RationalPolynomial,
)
from phydrax.discretization._exact_algebraic_geometry import (
    coefficient_payload,
    RadicalElement,
)
from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from phydrax.discretization._exact_power_geometry import ExactPowerCellGeometrySource
from phydrax.discretization._mapped_power_geometry import MappedPowerCellGeometrySource
from phydrax.discretization._partition import CellPartition
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import PeriodicIsometryGroup
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.discretization._surface_chart_records import (
    SurfaceChartWitness,
    SurfaceRootMeshCellBinding,
)
from phydrax.discretization._tensor_support import PreparedTensorGrid
from phydrax.discretization.fem import (
    FiniteElementDofOwnership,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    form_element,
)
from phydrax.discretization.fem._de_rham import (
    DistributedFiniteElementTransfer,
    PeriodicFiniteElementTransfer,
)
from phydrax.discretization.fem._distributed import finite_element_dof_identity_plan
from phydrax.discretization.fem._reference import FiniteElementSpec, lagrange_element
from phydrax.equations._finite_element_material import (
    MaterialSiteId,
    MaterialState,
    MaterialTransaction,
)
from phydrax.exterior._form_type import FormType, FormValueSpec
from phydrax.geometry._atlas import TrimDomain
from phydrax.geometry._contracts import CompiledGeometry, GeometryTolerance
from phydrax.geometry._curve_boundary import ParametricCurveBoundarySource
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._domain_certification import DomainCoverageCertificate
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._mesh_certificates import _same_tree_content
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSourceAccuracy,
    MeshingSurfacePatch,
    PatchBoundaryUse,
    PatchCurveUse,
    PatchPoleUse,
)
from phydrax.geometry._piecewise_linear_boundary import PiecewiseLinearBoundarySource
from phydrax.geometry._planar_embedding import PlanarEmbedding
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryDistance,
    SourceBoundaryDistanceWitness,
    SourceBoundaryPointWitness,
    SourceBoundaryQuery,
    SourceBoundaryRationalChartBank,
    SourceBoundarySamples,
    SurfaceEnvelopeDistanceWitness,
)
from phydrax.geometry._source_fidelity import certify_source_fidelity
from phydrax.geometry._source_fidelity_certificate import (
    SourceFidelityCertificate,
    SourceFidelityWitness,
)
from phydrax.geometry._surface_source_support import (
    SurfaceSourceCharts,
)
from phydrax.geometry._volume_material_deformation import (
    MaterialReferenceDeclaration,
    MaterialReferencePolicy,
    VolumeMaterialDeformation,
)
from phydrax.geometry._volume_material_map import (
    EulerianVolumeOverlapChain,
    EulerianVolumeOverlapMap,
    MaterialSourceMapCapability,
    VolumeMaterialChain,
    VolumeMaterialMap,
)
from phydrax.geometry.analytic._extended import _TorusKernel
from phydrax.geometry.analytic._primitives import _BallKernel, Sphere
from phydrax.geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from phydrax.geometry.brep._boolean_core import AffinePCurve, PeriodicPCurve
from phydrax.geometry.brep._carrier_pieces import carrier_payload
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_offset,
    brep_sphere,
    BRepTessellationPolicy,
)
from phydrax.geometry.brep._intersection import (
    ConicParameterEndpoint,
    NativePeriodEndpoint,
)
from phydrax.geometry.brep._intersection_curve import CurveTrimSegment, SurfaceRegion
from phydrax.geometry.brep._model import (
    _certify_edge_face_gap_bounds,
    BRepEntityId,
    BRepGeometry,
    BRepImportReport,
    BRepModel,
)
from phydrax.geometry.brep._offset_result import BRepOffsetResult
from phydrax.geometry.brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    ExactLineCurve,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SurfaceMappedCurve,
    TorusPatch,
)
from phydrax.geometry.brep._placed import PlacedCurve, PlacedSurface
from phydrax.geometry.brep._projection import prepare_brep_projection
from phydrax.geometry.brep._projection_contracts import BRepProjectionResult
from phydrax.geometry.brep._root_bindings import (
    BRepConicVertex,
    BRepCurveSurfaceLift,
    BRepPlacedVertex,
    BRepVertexRoot,
)
from phydrax.geometry.brep._tspline import TSplineSurfacePatch
from phydrax.geometry.design._schema import (
    DesignState,
    ParameterBinding,
    ParameterId,
    ParameterSchema,
    ParameterSpec,
)
from phydrax.geometry.implicit._adaptive_discovery import (
    AdaptiveImplicitBoundarySource,
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfaceArchiveRecipe,
    discover_adaptive_implicit_surface,
    ImplicitVolumeClass,
)
from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
from phydrax.geometry.implicit._policy import AdaptiveImplicitSurfacePolicy
from phydrax.geometry.implicit._restricted_delaunay_coverage import (
    RestrictedDelaunayBoundaryWitness,
    RestrictedDelaunayCoverageTransportWitness,
)
from phydrax.geometry.reconstruction._boundary import ReconstructedTriangleBoundarySource
from phydrax.geometry.reconstruction._core import (
    reconstruct_dem_region,
    ReconstructedGeometrySource,
    ReconstructedGeometrySourceArchiveRecipe,
    ReconstructionReport,
)
from phydrax.geometry.simplicial._regions import MeshRegion
from phydrax.geometry.surface._contracts import (
    SurfaceInterface,
    SurfaceMetadata,
    SurfaceOrientationRepair,
    SurfaceSelection,
)
from phydrax.geometry.surface._model import SurfaceModel
from phydrax.interchange._cad_archive import load_brep_archive, save_brep_archive
from phydrax.lifecycle._composition_rebind import (
    Composition,
    CompositionDependency,
    CompositionEntry,
)
from phydrax.meshing._assembly import MeshPart
from phydrax.meshing._association import GeometryAssociation, GeometryAssociationKind
from phydrax.meshing._audit import CellMeshAuditPolicy
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._certification_inputs import MeshCertificationInputs
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import ProtectedFeature
from phydrax.meshing._curving import _ExactCoordinateBankMap
from phydrax.meshing._distribution import MeshDistribution, MeshPartitionKind
from phydrax.meshing._dual_extraction import _RestrictedCoordinateMap
from phydrax.meshing._hex_cut_generation import (
    _CutCoordinateMap,
    generate_balanced_cut_hexes,
)
from phydrax.meshing._hex_cut_preparation import (
    _HexFrameIntegerLayout,
    BalancedCutGridPlan,
    prepare_balanced_cut_grid,
)
from phydrax.meshing._hex_generation import extract_volume_hexes
from phydrax.meshing._hex_schedule import NativeHexGridSchedule
from phydrax.meshing._lineage import CellMeshTransition, VertexInterpolationStencil
from phydrax.meshing._model_codec import (
    MODEL_ARCHIVE_LIMITS,
    read_model_archive_fields,
    read_model_artifact,
    restore_model_archive_value,
    write_model_artifact,
)
from phydrax.meshing._native_construction import NativeMeshOrganization
from phydrax.meshing._organization import (
    MeshAttribute,
    MeshLabel,
    MeshPatch,
    MeshZone,
    RegionBoundaryEvidence,
)
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._scope import _SourceEntityScopes, MeshingEntityKind, MeshingScope
from phydrax.meshing._surface_envelope import SurfaceEnvelopeSource
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native_implicit import prepare_native_implicit
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.solver._finite_element_schedule import (
    FiniteElementAcceptedElementRecipe,
    FiniteElementAcceptedFieldLayout,
    FiniteElementAcceptedFieldLayoutRecipe,
    FiniteElementAcceptedState,
    FiniteElementFieldTransferSemantics,
    FiniteElementRestartManifest,
    FiniteElementRestartStateAssociation,
    FiniteElementRestartStateLayout,
    FiniteElementRestartStateRole,
)
from phydrax.units import UnitDefinition


def _fraction_point3(value: object, /) -> tuple[Fraction, Fraction, Fraction]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("Exact BRep points must retain three rational coordinates.")
    x, y, z = value
    if (
        not isinstance(x, Fraction)
        or not isinstance(y, Fraction)
        or not isinstance(z, Fraction)
    ):
        raise TypeError("Exact BRep point coordinates must remain Fractions.")
    return x, y, z


def _square_model(contract: SpatialCoordinateContract, /) -> BRepModel:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    exact_points: tuple[tuple[Fraction, Fraction, Fraction], ...] = tuple(
        (Fraction(int(point[0])), Fraction(int(point[1])), Fraction(int(point[2])))
        for point in points
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    curves = tuple(
        ExactLineCurve(exact_points[first], exact_points[last]) for first, last in edges
    )
    pcurves = (
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((1.0, 0.0), (0.0, 1.0)),
        LineCurve((1.0, 1.0), (-1.0, 0.0)),
        LineCurve((0.0, 1.0), (0.0, -1.0)),
    )
    placed_affine = BRepAffinePlacement.from_matrix(
        (
            (Fraction(1), Fraction(0), Fraction(0), Fraction(-1, 2)),
            (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
        )
    )
    placed_root = BRepVertexRoot(
        BRepPlacedVertex(
            np.asarray((0.5, 0.0, 0.0), dtype=np.float64),
            np.eye(3, dtype=np.float64),
            np.asarray((-0.5, 0.0, 0.0), dtype=np.float64),
            "archive-test:square:vertex-0",
            affine=placed_affine,
        )
    )
    geometry = BRepGeometry(
        vertex_points=points,
        vertex_exact_points=exact_points,
        vertex_roots=(placed_root, None, None, None),
        edge_endpoint_roots=((None, None),) * 4,
        coedge_endpoint_roots=((None, None),) * 4,
        curves=curves,
        edge_curves=(0, 1, 2, 3),
        edge_ranges=np.tile(np.asarray((0.0, 1.0)), (4, 1)),
        edge_vertices=edges,
        pcurves=pcurves,
        coedge_edges=(0, 1, 2, 3),
        coedge_senses=(1, 1, 1, 1),
        face_loops=(((0, 1, 2, 3),),),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    report = BRepImportReport(
        source_id="archive-test:square",
        source_digest="a" * 64,
        source_format="native",
        coordinate_contract=contract,
        import_policy_id="f" * 64,
        num_solids=0,
        num_faces=1,
        num_edges=4,
        num_vertices=4,
    )
    return BRepModel(
        geometry=geometry,
        patches=(PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),),
        parameter_bounds=np.asarray((((0.0, 0.0), (1.0, 1.0)),)),
        orientation=np.asarray((1.0,)),
        physical_tags=("profile",),
        coordinate_contract=contract,
        report=report,
    )


def _bspline_surface_model(contract: SpatialCoordinateContract, /) -> BRepModel:
    square = _square_model(contract)
    exact_control_points = (
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
        ),
        (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(1), Fraction(0)),
        ),
    )
    exact_weights = ((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1)))
    exact_knots = (Fraction(0), Fraction(0), Fraction(1), Fraction(1))
    patch = BSplineSurfacePatch(
        (
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        ),
        ((1.0, 1.0), (1.0, 1.0)),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        1,
        1,
        exact_control_points=exact_control_points,
        exact_weights=exact_weights,
        exact_u_knots=exact_knots,
        exact_v_knots=exact_knots,
        control_point_ids=((101, 102), (103, 104)),
    )
    return BRepModel(
        geometry=square.geometry,
        patches=(patch,),
        parameter_bounds=square.parameter_bounds,
        orientation=square.orientation,
        physical_tags=square.physical_tags,
        coordinate_contract=contract,
        report=square.report,
    )


def _offset_ruled_surface_model(contract: SpatialCoordinateContract, /) -> BRepModel:
    exact_points = (
        (Fraction(0), Fraction(-1, 3), Fraction(0)),
        (Fraction(1), Fraction(-1, 3), Fraction(0)),
        (Fraction(1), Fraction(2, 3), Fraction(0)),
        (Fraction(0), Fraction(2, 3), Fraction(0)),
    )
    points = np.asarray(
        [[float(value) for value in row] for row in exact_points], dtype=np.float64
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    geometry = BRepGeometry(
        vertex_points=points,
        vertex_exact_points=exact_points,
        curves=tuple(
            ExactLineCurve(exact_points[first], exact_points[last])
            for first, last in edges
        ),
        edge_curves=(0, 1, 2, 3),
        edge_ranges=np.tile(np.asarray((0.0, 1.0)), (4, 1)),
        edge_vertices=edges,
        pcurves=(
            LineCurve((0.0, 0.0), (1.0, 0.0)),
            LineCurve((1.0, 0.0), (0.0, 1.0)),
            LineCurve((1.0, 1.0), (-1.0, 0.0)),
            LineCurve((0.0, 1.0), (0.0, -1.0)),
        ),
        coedge_edges=(0, 1, 2, 3),
        coedge_senses=(1, 1, 1, 1),
        face_loops=(((0, 1, 2, 3),),),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    report = BRepImportReport(
        source_id="archive-offset-ruled-source",
        source_digest="d" * 64,
        source_format="native",
        coordinate_contract=contract,
        import_policy_id="e" * 64,
        num_solids=0,
        num_faces=1,
        num_edges=4,
        num_vertices=4,
    )
    patch = RuledSurface(
        OffsetCurve(
            LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            Fraction(1, 3),
            direction=(0.0, 0.0, 1.0),
        ),
        LineCurve(
            (0.0, 2.0 / 3.0, 0.0),
            (1.0, 0.0, 0.0),
            origin_exact=(Fraction(0), Fraction(2, 3), Fraction(0)),
            direction_exact=(Fraction(1), Fraction(0), Fraction(0)),
        ),
    )
    return BRepModel(
        geometry=geometry,
        patches=(patch,),
        parameter_bounds=np.asarray((((0.0, 0.0), (1.0, 1.0)),)),
        orientation=np.asarray((1.0,)),
        physical_tags=("offset-ruled",),
        coordinate_contract=contract,
        report=report,
    )


def _boolean_curve_carrier_model(contract: SpatialCoordinateContract, /) -> BRepModel:
    source = _square_model(contract)
    geometry = source.geometry
    exact_curve = BSplineCurve(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        (1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        1,
        exact_control_points=(
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0), Fraction(0)),
        ),
        exact_weights=(Fraction(1), Fraction(1)),
        exact_knots=(Fraction(0), Fraction(0), Fraction(1), Fraction(1)),
    )
    curve_carriers = (exact_curve, *geometry.curves[1:])
    patch = source.patches[0]
    identity = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    conic_root = ConicParameterRoot(
        CircleCurve(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            1.0,
        ),
        Fraction(1),
        Fraction(0),
    )
    conic_pcurve = ConicRootLinePCurve(conic_root, Fraction(1), Fraction(-1))
    pcurves = (
        AffinePCurve(
            geometry.pcurves[0],
            np.asarray(identity, dtype=object),
            np.asarray((Fraction(0), Fraction(0)), dtype=object),
        ),
        PeriodicPCurve(geometry.pcurves[1], patch, (0, 0)),
        geometry.pcurves[2],
        conic_pcurve,
    )
    conic_point_values = conic_root.exact_point()
    if len(conic_point_values) != 3:
        raise ValueError("Conic carrier root must be a three-dimensional point.")
    conic_point = (
        conic_point_values[0],
        conic_point_values[1],
        conic_point_values[2],
    )
    conic_vertex_root = BRepVertexRoot(BRepConicVertex(conic_root))
    edge_endpoint_roots = (
        (
            None,
            ConicParameterEndpoint(
                conic_root, curve_carriers[0], Fraction(1), conic_point
            ),
        ),
        (
            ConicParameterEndpoint(
                conic_root, geometry.curves[1], Fraction(0), conic_point
            ),
            None,
        ),
        *geometry.edge_endpoint_roots[2:],
    )
    coedge_endpoint_roots = (
        (None, ConicParameterEndpoint(conic_root, pcurves[0], Fraction(1), conic_point)),
        (ConicParameterEndpoint(conic_root, pcurves[1], Fraction(0), conic_point), None),
        *geometry.coedge_endpoint_roots[2:],
    )
    trim_domain = TrimDomain.from_curve_loops(
        (
            tuple(
                CurveTrimSegment(
                    pcurve, 0.0, 1.0, first_root=roots[0], last_root=roots[1]
                )
                for pcurve, roots in zip(pcurves, coedge_endpoint_roots, strict=True)
            ),
        )
    )
    rebuilt_geometry = BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=geometry.vertex_exact_points,
        vertex_roots=(
            geometry.vertex_roots[0],
            conic_vertex_root,
            *geometry.vertex_roots[2:],
        ),
        edge_endpoint_roots=edge_endpoint_roots,
        coedge_endpoint_roots=coedge_endpoint_roots,
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=geometry.edge_face_gap_bounds,
        curves=curve_carriers,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=geometry.assembly_graph,
    )
    return BRepModel(
        geometry=rebuilt_geometry,
        patches=source.patches,
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        physical_tags=source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        report=source.report,
        trim_domains=(trim_domain,),
    )


def _surface_mapped_square(contract: SpatialCoordinateContract, /) -> BRepModel:
    square = _square_model(contract)
    source = square.geometry
    mapped_curve = SurfaceMappedCurve(square.patches[0], source.pcurves[0])
    geometry = BRepGeometry(
        vertex_points=source.vertex_points,
        vertex_exact_points=source.vertex_exact_points,
        vertex_roots=source.vertex_roots,
        edge_endpoint_roots=source.edge_endpoint_roots,
        coedge_endpoint_roots=source.coedge_endpoint_roots,
        edge_vertex_gap_bounds=source.edge_vertex_gap_bounds,
        edge_face_gap_bounds=source.edge_face_gap_bounds,
        curves=(mapped_curve, *source.curves[1:]),
        edge_curves=source.edge_curves,
        edge_ranges=source.edge_ranges,
        edge_vertices=source.edge_vertices,
        pcurves=source.pcurves,
        coedge_edges=source.coedge_edges,
        coedge_senses=source.coedge_senses,
        face_loops=source.face_loops,
        shell_faces=source.shell_faces,
        shell_orientations=source.shell_orientations,
        solid_shells=source.solid_shells,
        assembly_graph=source.assembly_graph,
    )
    return BRepModel(
        geometry=geometry,
        patches=square.patches,
        parameter_bounds=square.parameter_bounds,
        orientation=square.orientation,
        physical_tags=square.physical_tags,
        coordinate_contract=square.coordinate_contract,
        report=square.report,
        trim_domains=square.trim_domains,
    )


def _placed_box(contract: SpatialCoordinateContract, /) -> tuple[BRepModel, BRepModel]:
    box = brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=contract)
    translated = BRepAffinePlacement.from_matrix(
        (
            (1, 0, 0, Fraction(3, 2)),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    graph = BRepAssemblyGraph(
        roots=(BRepAssemblyUse("pair", definition_id="two-copies"),),
        definitions=(
            BRepAssemblyDefinition(
                "two-copies",
                (
                    BRepAssemblyUse("left", solid_index=0),
                    BRepAssemblyUse("right", placement=translated, solid_index=0),
                ),
            ),
        ),
        solid_count=1,
    )
    source = box.geometry
    source_affine = BRepAffinePlacement.from_matrix(
        (
            (Fraction(1), Fraction(0), Fraction(0), Fraction(1, 3)),
            (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
        )
    )
    rotation = np.eye(3, dtype=np.float64)
    translation = np.asarray((float(Fraction(1, 3)), 0.0, 0.0), dtype=np.float64)
    if (
        any(root is not None for root in source.vertex_roots)
        or any(root is not None for pair in source.edge_endpoint_roots for root in pair)
        or any(root is not None for pair in source.coedge_endpoint_roots for root in pair)
    ):
        raise ValueError(
            "The placed-box archive fixture requires an unrooted source box."
        )
    exact_points: tuple[tuple[Fraction, Fraction, Fraction] | None, ...] = tuple(
        None if point is None else _fraction_point3(source_affine.apply(point))
        for point in source.vertex_exact_points
    )
    vertex_points = np.asarray(source.vertex_points) @ rotation.T + translation
    for index, point in enumerate(exact_points):
        if point is not None:
            vertex_points[index] = np.asarray(tuple(float(value) for value in point))
    placed_curves = tuple(
        PlacedCurve(curve, rotation, translation, affine=source_affine)
        for curve in source.curves
    )
    placed_patches = tuple(
        PlacedSurface(patch, rotation, translation, affine=source_affine)
        for patch in box.patches
    )
    geometry = BRepGeometry(
        vertex_points=vertex_points,
        vertex_exact_points=exact_points,
        vertex_roots=source.vertex_roots,
        edge_endpoint_roots=source.edge_endpoint_roots,
        coedge_endpoint_roots=source.coedge_endpoint_roots,
        edge_vertex_gap_bounds=source.edge_vertex_gap_bounds,
        edge_face_gap_bounds=source.edge_face_gap_bounds,
        curves=placed_curves,
        edge_curves=source.edge_curves,
        edge_ranges=source.edge_ranges,
        edge_vertices=source.edge_vertices,
        pcurves=source.pcurves,
        coedge_edges=source.coedge_edges,
        coedge_senses=source.coedge_senses,
        face_loops=source.face_loops,
        shell_faces=source.shell_faces,
        shell_orientations=source.shell_orientations,
        solid_shells=source.solid_shells,
        assembly_graph=graph,
    )
    placed = BRepModel(
        geometry=geometry,
        patches=placed_patches,
        parameter_bounds=box.parameter_bounds,
        orientation=box.orientation,
        physical_tags=box.physical_tags,
        coordinate_contract=box.coordinate_contract,
        report=box.report,
        trim_domains=box.trim_domains,
    )
    return box, placed


def _refined_rational_patch(contract: SpatialCoordinateContract, /) -> BRepModel:
    source_mesh = TMesh2D.tensor_product(
        (Fraction(0), Fraction(1)), (Fraction(0), Fraction(1))
    )
    refinement = source_mesh.refine((source_mesh.cells[0].cell_id,))
    # Non-dyadic, yet inside the exact predicates' [2**-120, 2**120] coordinate range.
    tiny = Fraction(1, 3 * 2**100)
    # Greville control points reproduce the identity chart, so the square's z=0
    # edges lift exactly except for the ``tiny`` lift of the first control point.
    exact_points = tuple(
        (
            sum(anchor.local_knots.u[1:4], Fraction(0)) / 3,
            sum(anchor.local_knots.v[1:4], Fraction(0)) / 3,
            tiny if index == 0 else Fraction(0),
        )
        for index, anchor in enumerate(source_mesh.anchors)
    )
    controls = np.asarray(
        [[float(value) for value in point] for point in exact_points],
        dtype=np.float64,
    )
    exact_weights = (Fraction(1, 3),) * source_mesh.coefficient_count
    weights = np.full(
        (source_mesh.coefficient_count,), float(Fraction(1, 3)), dtype=np.float64
    )
    patch = TSplineSurfacePatch(
        refinement.refined_mesh,
        controls,
        weights,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
        source_mesh=source_mesh,
        transfer_plans=(refinement.transfer_plan,),
    )
    square = _square_model(contract)
    geometry = square.geometry
    # Declare the recomputed exact face-gap bound of every coedge.
    face_gaps = _certify_edge_face_gap_bounds(
        geometry,
        (patch,),
        parameter_bounds=square.parameter_bounds,
        exact_unsupported=True,
        certified_degenerate_coedges=True,
    )
    if face_gaps is None or max(face_gaps) > float(tiny) * 2.0:
        raise ValueError("The lifted T-spline corner must keep a tiny face gap.")
    declared = BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=geometry.vertex_exact_points,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=face_gaps,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=geometry.assembly_graph,
    )
    return BRepModel(
        geometry=declared,
        patches=(patch,),
        parameter_bounds=square.parameter_bounds,
        orientation=square.orientation,
        physical_tags=square.physical_tags,
        coordinate_contract=square.coordinate_contract,
        report=square.report,
        trim_domains=square.trim_domains,
    )


def _tspline_archive_state(model: BRepModel, /) -> dict[str, object]:
    if len(model.patches) != 1 or not isinstance(model.patches[0], TSplineSurfacePatch):
        raise TypeError("Cold T-spline archive requires one native TSplineSurfacePatch.")
    patch = model.patches[0]
    parameters = np.asarray((0.37, 0.41), dtype=np.float64)
    parameter_array = jnp.asarray(parameters, dtype=jnp.float64)
    point = np.asarray(jax.jit(lambda value: patch.evaluate(value))(parameter_array))
    gradient = np.asarray(
        jax.jit(jax.jacfwd(lambda value: patch.evaluate(value)))(parameter_array)
    )
    return {
        "source_revision": model.source_revision,
        "source_definition_id": patch.source_definition_id,
        "source_authority_id": patch.source_authority_id,
        "source_numeric_bank_id": patch.source_numeric_bank_id,
        "source_mesh_id": patch.source_mesh.mesh_id,
        "target_mesh_id": patch.mesh.mesh_id,
        "source_basis_certificate_id": patch.source_basis_certificate.certificate_id,
        "source_mesh": _tspline_mesh_archive_record(patch.source_mesh),
        "target_mesh": _tspline_mesh_archive_record(patch.mesh),
        "exact_source_control_points": _fraction_rows_for_archive(
            patch.exact_source_control_points
        ),
        "exact_source_weights": [
            _fraction_pair_for_archive(value) for value in patch.exact_source_weights
        ],
        "exact_target_homogeneous": _fraction_rows_for_archive(
            patch.exact_target_homogeneous
        ),
        "parameters": parameters.tolist(),
        "point": point.tolist(),
        "gradient": gradient.tolist(),
    }


def _write_tspline_cold_archive(directory: Path, /) -> None:
    model = _refined_rational_patch(SpatialCoordinateContract.si())
    state = _tspline_archive_state(model)
    save_brep_archive(model, directory / "refined.phx")
    print(json.dumps(state, sort_keys=True))


def _read_tspline_cold_archive(
    directory: Path,
    expected: dict[str, object],
    /,
) -> None:
    model = load_brep_archive(directory / "refined.phx")
    state = _tspline_archive_state(model)
    if state != expected:
        raise ValueError(
            "Cold T-spline source/target identity or numerical continuation changed."
        )
    print("tspline-cold-restored")


def _plane_cylinder_section_pcurve_fixture() -> PlaneCylinderSectionPCurve:
    phase = ConicParameterRoot(
        CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0),
        Fraction(3, 5),
        Fraction(4, 5),
    )
    return PlaneCylinderSectionPCurve(
        (Fraction(0), Fraction(1, 7)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1, 3)),
        (Fraction(0), Fraction(2, 5)),
        parameter_range=(0.0, 2.0 * np.pi),
        angle_offset_root=phase,
    )


def _plane_cylinder_section_value(
    curve: PlaneCylinderSectionPCurve, parameters: jax.Array, /
) -> jax.Array:
    return curve.evaluate(parameters)


@eqx.filter_jit
def _compiled_plane_cylinder_section_value(
    curve: PlaneCylinderSectionPCurve, parameters: jax.Array, /
) -> jax.Array:
    return _plane_cylinder_section_value(curve, parameters)


@eqx.filter_jit
def _compiled_plane_cylinder_section_gradient(
    curve: PlaneCylinderSectionPCurve, parameter: jax.Array, /
) -> jax.Array:
    return jax.jacfwd(_plane_cylinder_section_value, argnums=1)(curve, parameter)


def _plane_cylinder_section_pcurve_state(
    curve: PlaneCylinderSectionPCurve, /
) -> dict[str, object]:
    parameters = np.asarray((0.19, 1.37, 3.41, 5.79), dtype=np.float64)
    points = np.asarray(
        _compiled_plane_cylinder_section_value(curve, jnp.asarray(parameters))
    )
    gradient = np.asarray(
        _compiled_plane_cylinder_section_gradient(
            curve, jnp.asarray(0.73, dtype=jnp.float64)
        )
    )
    phase = curve.angle_offset_root
    if phase is None:
        raise ValueError("Plane-cylinder section fixture requires its exact phase root.")
    plane_x = (
        phase.cosine_exact * curve.cosine_exact[1]
        - phase.sine_exact * curve.sine_exact[1]
    )
    plane_y = (
        phase.sine_exact * curve.cosine_exact[1]
        + phase.cosine_exact * curve.sine_exact[1]
    )
    points_x = np.cos(points[:, 0])
    points_y = np.sin(points[:, 0])
    plane_residual = (
        points[:, 1]
        - float(curve.offset_exact[1])
        - float(plane_x) * points_x
        - float(plane_y) * points_y
    )
    np.testing.assert_allclose(plane_residual, 0.0, rtol=0.0, atol=3e-15)
    return {
        "identity_payload": _json_payload(curve.identity_payload()),
        "offset_exact": [
            _exact_scalar_pair_for_archive(value) for value in curve.offset_exact
        ],
        "linear_exact": [
            _exact_scalar_pair_for_archive(value) for value in curve.linear_exact
        ],
        "cosine_exact": [
            _exact_scalar_pair_for_archive(value) for value in curve.cosine_exact
        ],
        "sine_exact": [
            _exact_scalar_pair_for_archive(value) for value in curve.sine_exact
        ],
        "phase_root_id": None if phase is None else phase.root_id,
        "phase_root_payload": None
        if phase is None
        else _json_payload(phase.identity_payload()),
        "source_plane_exact": {
            "offset": _exact_scalar_pair_for_archive(curve.offset_exact[1]),
            "x_coefficient": _exact_scalar_pair_for_archive(plane_x),
            "y_coefficient": _exact_scalar_pair_for_archive(plane_y),
        },
        "parameter_range": [float(value).hex() for value in curve.parameter_range],
        "angle_offset": float(np.asarray(curve.angle_offset)).hex(),
        "numeric_shadows": {
            "offset": np.asarray(curve.offset, dtype=np.float64).tolist(),
            "linear": np.asarray(curve.linear, dtype=np.float64).tolist(),
            "cosine": np.asarray(curve.cosine, dtype=np.float64).tolist(),
            "sine": np.asarray(curve.sine, dtype=np.float64).tolist(),
        },
        "parameters": parameters.tolist(),
        "points": points.tolist(),
        "gradient": gradient.tolist(),
    }


def _write_plane_cylinder_section_pcurve(directory: Path, /) -> None:
    curve = _plane_cylinder_section_pcurve_fixture()
    write_model_artifact(
        directory / "plane-cylinder-section-pcurve.phx", curve, limits=None
    )
    print(json.dumps(_plane_cylinder_section_pcurve_state(curve), sort_keys=True))


def _read_plane_cylinder_section_pcurve(
    directory: Path, expected: dict[str, object], /
) -> None:
    from phydrax.interchange._cad_archive import _reconstruct_brep_graph

    curve = read_model_artifact(
        directory / "plane-cylinder-section-pcurve.phx",
        expected_type=PlaneCylinderSectionPCurve,
        restore=_reconstruct_brep_graph,
        limits=None,
    )
    if type(curve) is not PlaneCylinderSectionPCurve:
        raise TypeError("Cold plane-cylinder p-curve archive restored another root type.")
    if _plane_cylinder_section_pcurve_state(curve) != expected:
        raise ValueError(
            "Cold plane-cylinder p-curve identity or numerical continuation changed."
        )
    print("plane-cylinder-section-pcurve-cold-restored")


def _brep_curve_surface_lift_fixture() -> BRepCurveSurfaceLift:
    source = _square_model(SpatialCoordinateContract.si())
    patch = source.patches[0]
    source_region = SurfaceRegion(
        patch, np.asarray(((0.25, 0.0), (0.75, 1.0)), dtype=np.float64)
    )
    return BRepCurveSurfaceLift(
        source.geometry.curves[0],
        patch,
        source.geometry.pcurves[0],
        0.25,
        0.75,
        source_region=source_region,
    )


def _brep_curve_surface_lift_state(lift: BRepCurveSurfaceLift, /) -> dict[str, object]:
    parameters = jnp.asarray((0.31, 0.5, 0.69), dtype=jnp.float64)
    curve_points = np.asarray(lift.curve.evaluate(parameters))
    pcurve_points = np.asarray(lift.pcurve.evaluate(parameters))
    surface_points = np.asarray(lift.surface.evaluate(jnp.asarray(pcurve_points)))
    np.testing.assert_allclose(curve_points, surface_points, rtol=0.0, atol=1e-15)
    return {
        "root_id": lift.root_id,
        "identity_payload": _json_payload(lift.identity_payload()),
        "parameter_box": [
            [float(value).hex() for value in row] for row in lift.parameter_box
        ],
        "first": float(lift.first).hex(),
        "last": float(lift.last).hex(),
        "parameters": np.asarray(parameters).tolist(),
        "curve_points": curve_points.tolist(),
        "pcurve_points": pcurve_points.tolist(),
        "surface_points": surface_points.tolist(),
    }


def _write_brep_curve_surface_lift(directory: Path, /) -> None:
    lift = _brep_curve_surface_lift_fixture()
    write_model_artifact(directory / "brep-curve-surface-lift.phx", lift, limits=None)
    print(json.dumps(_brep_curve_surface_lift_state(lift), sort_keys=True))


def _read_brep_curve_surface_lift(
    directory: Path, expected: dict[str, object], /
) -> None:
    from phydrax.interchange._cad_archive import _reconstruct_brep_graph

    lift = read_model_artifact(
        directory / "brep-curve-surface-lift.phx",
        expected_type=BRepCurveSurfaceLift,
        restore=_reconstruct_brep_graph,
        limits=None,
    )
    if type(lift) is not BRepCurveSurfaceLift:
        raise TypeError("Cold B-Rep lift archive restored another root type.")
    if _brep_curve_surface_lift_state(lift) != expected:
        raise ValueError("Cold B-Rep lift changed its retained source chart or identity.")
    print("brep-curve-surface-lift-cold-restored")


def _sphere_chamfer_conic_root_model() -> BRepModel:
    from phydrax.geometry.brep._fillet import brep_chamfer
    from tests.unit.geometry.test_brep_blend import _fixture

    source, edge = _fixture("closed-sphere")
    result = brep_chamfer(
        source,
        (source.edge_ids[edge],),
        (0.25, 0.375),
        include_parameter_range=True,
    )
    if result.status != "certified" or result.model is None:
        raise ValueError(f"Sphere chamfer did not certify: {result.reason}")
    return result.model


def _sphere_chamfer_conic_root_state(model: BRepModel, /) -> dict[str, object]:
    """Exact sphere-side chamfer rails: Revolution-mapped conic-root p-curves."""
    rail_records = []
    for patch in model.patches:
        if not isinstance(patch, RuledSurface):
            continue
        rail = patch.second
        if not (
            isinstance(rail, SurfaceMappedCurve)
            and isinstance(rail.surface, RevolutionSurface)
            and isinstance(rail.pcurve, AffinePCurve)
        ):
            continue
        conic = rail.pcurve.curve
        if not isinstance(conic, ConicRootLinePCurve):
            continue
        rail_records.append(
            _json_payload(
                {
                    "phase_shift_exact": coefficient_payload(conic.phase_shift_exact),
                    "height_origin_exact": coefficient_payload(conic.height_origin_exact),
                    "height_direction_exact": coefficient_payload(
                        conic.height_direction_exact
                    ),
                    "identity_payload": rail.identity_payload(),
                }
            )
        )
    if not rail_records:
        raise ValueError("Sphere chamfer lost its exact conic-root revolution rail.")
    return {
        "model_id": model.model_id,
        "source_revision": model.source_revision,
        "rails": rail_records,
    }


def _torus_exact_archive_state(patch: TorusPatch, /) -> dict[str, object]:
    if type(patch) is not TorusPatch:
        raise TypeError("Exact torus archive requires a TorusPatch.")
    identity = patch.identity_payload()
    expected_fields = {
        "kind",
        "center",
        "first_axis",
        "second_axis",
        "axis",
        "major_radius",
        "minor_radius",
    }
    if set(identity) != expected_fields:
        raise ValueError(
            "TorusPatch identity omitted an authoritative exact source field."
        )
    return {"torus_patch_identity": _json_payload(identity)}


def _write_torus_exact_source_archive(directory: Path, /) -> None:
    from phydrax.geometry.brep._patches import CircleCurve
    from phydrax.geometry.brep._placed import PlacedCurve, PlacedSurface

    identity = BRepAffinePlacement.identity()
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros((3,), dtype=np.float64)
    patch = TorusPatch(
        (0.1, -0.5, 1.25),
        (0.6, 0.8, 0.0),
        (0.0, 0.0, 1.0),
        (0.8, -0.6, 0.0),
        float(Fraction(21, 10)),
        float(Fraction(1, 5)),
        center_exact=(Fraction(1, 10), Fraction(-1, 2), Fraction(5, 4)),
        first_axis_exact=(Fraction(3, 5), Fraction(4, 5), Fraction(0)),
        second_axis_exact=(Fraction(0), Fraction(0), Fraction(1)),
        axis_exact=(Fraction(4, 5), Fraction(-3, 5), Fraction(0)),
        major_radius_exact=Fraction(21, 10),
        minor_radius_exact=Fraction(1, 5),
    )
    placed_surface = PlacedSurface(patch, rotation, translation, affine=identity)
    circle = CircleCurve(
        (0.1, -0.5, 1.25),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        0.1,
        center_exact=(Fraction(1, 10), Fraction(-1, 2), Fraction(5, 4)),
        first_axis_exact=(Fraction(1), Fraction(0), Fraction(0)),
        second_axis_exact=(Fraction(0), Fraction(0), Fraction(1)),
        radius_exact=Fraction(1, 10),
    )
    placed_curve = PlacedCurve(circle, rotation, translation, affine=identity)
    write_model_artifact(directory / "exact-torus-source.phx", patch, limits=None)
    write_model_artifact(
        directory / "placed-torus-surface.phx", placed_surface, limits=None
    )
    write_model_artifact(directory / "placed-exact-circle.phx", placed_curve, limits=None)
    state = _torus_exact_archive_state(patch)
    state["placed_surface_identity"] = _json_payload(placed_surface.identity_payload())
    state["placed_curve_identity"] = _json_payload(placed_curve.identity_payload())
    state["placed_torus_patch_identity"] = _json_payload(patch.identity_payload())
    print(json.dumps(state, sort_keys=True))


def _read_torus_exact_source_archive(
    directory: Path, expected: dict[str, object], /
) -> None:
    from phydrax.geometry.brep._patches import CircleCurve
    from phydrax.geometry.brep._placed import PlacedCurve, PlacedSurface
    from phydrax.interchange._cad_archive import _reconstruct_brep_graph

    patch = read_model_artifact(
        directory / "exact-torus-source.phx",
        expected_type=TorusPatch,
        restore=_reconstruct_brep_graph,
        limits=None,
    )
    placed_surface = read_model_artifact(
        directory / "placed-torus-surface.phx",
        expected_type=PlacedSurface,
        restore=_reconstruct_brep_graph,
        limits=None,
    )
    placed_curve = read_model_artifact(
        directory / "placed-exact-circle.phx",
        expected_type=PlacedCurve,
        restore=_reconstruct_brep_graph,
        limits=None,
    )
    if (
        type(placed_surface.definition) is not TorusPatch
        or type(placed_curve.definition) is not CircleCurve
    ):
        raise TypeError("Cold placed carriers lost their exact Torus/Circle definitions.")
    state = _torus_exact_archive_state(patch)
    state["placed_surface_identity"] = _json_payload(placed_surface.identity_payload())
    state["placed_curve_identity"] = _json_payload(placed_curve.identity_payload())
    state["placed_torus_patch_identity"] = _json_payload(
        placed_surface.definition.identity_payload()
    )
    if state != expected:
        raise ValueError(
            "Cold TorusPatch or placed conic archive changed its exact coefficient identity."
        )
    print("exact-torus-source-cold-restored")


def _write_sphere_chamfer_conic_root_archive(directory: Path, /) -> None:
    model = _sphere_chamfer_conic_root_model()
    save_brep_archive(model, directory / "sphere-chamfer.phx")
    print(json.dumps(_sphere_chamfer_conic_root_state(model), sort_keys=True))


def _read_sphere_chamfer_conic_root_archive(
    directory: Path, expected: dict[str, object], /
) -> None:
    model = load_brep_archive(directory / "sphere-chamfer.phx")
    if _sphere_chamfer_conic_root_state(model) != expected:
        raise ValueError("Cold sphere chamfer changed its exact conic-root source rail.")
    print("sphere-chamfer-conic-root-cold-restored")


def _parallel_cylinder_chamfer_model() -> BRepModel:
    from phydrax.geometry.brep._fillet import brep_chamfer
    from tests.unit.geometry.test_brep_blend import _fixture

    source, edge = _fixture("parallel-cylinders")
    result = brep_chamfer(
        source,
        (source.edge_ids[edge],),
        (0.25, 0.375),
        include_parameter_range=True,
    )
    if result.status != "certified" or result.model is None:
        raise ValueError(f"Parallel-cylinder chamfer did not certify: {result.reason}")
    return result.model


def _parallel_cylinder_chamfer_state(model: BRepModel, /) -> dict[str, object]:
    return {
        "model_id": model.model_id,
        "source_revision": model.source_revision,
    }


def _write_parallel_cylinder_chamfer_archive(directory: Path, /) -> None:
    model = _parallel_cylinder_chamfer_model()
    save_brep_archive(model, directory / "parallel-cylinder-chamfer.phx")
    print(json.dumps(_parallel_cylinder_chamfer_state(model), sort_keys=True))


def _read_parallel_cylinder_chamfer_archive(
    directory: Path, expected: dict[str, object], /
) -> None:
    model = load_brep_archive(directory / "parallel-cylinder-chamfer.phx")
    if _parallel_cylinder_chamfer_state(model) != expected:
        raise ValueError("Cold parallel-cylinder chamfer changed its source root.")
    print("parallel-cylinder-chamfer-cold-restored")


def _orthogonal_quadric_boolean_model() -> BRepModel:
    from phydrax.geometry.brep import brep_cylinder
    from phydrax.geometry.brep._boolean import boolean_brep
    from tests.unit.geometry.test_brep_boolean import _COORDINATES

    first = brep_cylinder(
        1.0,
        4.0,
        base_center=(0.0, 0.0, -2.0),
        coordinate_contract=_COORDINATES,
    )
    second = brep_cylinder(
        0.8,
        4.0,
        base_center=(-2.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        coordinate_contract=_COORDINATES,
    )
    result = boolean_brep(first, second, "intersection")
    if result.model is None:
        raise ValueError("Orthogonal quadric intersection did not retain its BRep model.")
    return result.model


def _orthogonal_quadric_boolean_state(model: BRepModel, /) -> dict[str, object]:
    from phydrax.geometry.brep._intersection_curve import IntersectionCurve

    geometry = model.geometry
    if geometry is None:
        raise ValueError("Orthogonal quadric Boolean lost its BRep geometry.")
    curves = tuple(
        curve for curve in geometry.curves if isinstance(curve, IntersectionCurve)
    )
    if not curves:
        raise ValueError(
            "Orthogonal quadric Boolean lost its source intersection curves."
        )
    shifts = tuple(
        np.asarray(curve.node_period_shifts, dtype=np.int64).tolist() for curve in curves
    )
    # Closed loops may close at the far end of a native period box: their deck
    # is the closure turn between the first and last node enclosures.
    closures = tuple(curve.closure_period_shifts().tolist() for curve in curves)
    if not any(
        np.any(np.asarray(shift)) or np.any(np.asarray(closure))
        for shift, closure in zip(shifts, closures, strict=True)
    ):
        raise ValueError(
            "Orthogonal quadric intersection lost its nonzero exact deck shifts."
        )
    return {
        "model_id": model.model_id,
        "geometry_id": geometry.geometry_id,
        "source_revision": model.source_revision,
        "branches": [
            {
                "branch_id": curve.branch_id,
                "node_period_shifts": shift,
                "closure_period_shifts": closure,
            }
            for curve, shift, closure in zip(curves, shifts, closures, strict=True)
        ],
    }


def _write_orthogonal_quadric_boolean_archive(directory: Path, /) -> None:
    model = _orthogonal_quadric_boolean_model()
    save_brep_archive(model, directory / "orthogonal-quadric-boolean.phx")
    print(json.dumps(_orthogonal_quadric_boolean_state(model), sort_keys=True))


def _read_orthogonal_quadric_boolean_archive(
    directory: Path, expected: dict[str, object], /
) -> None:
    model = load_brep_archive(directory / "orthogonal-quadric-boolean.phx")
    if _orthogonal_quadric_boolean_state(model) != expected:
        raise ValueError(
            "Cold quadric Boolean changed its exact intersection period shifts."
        )
    print("orthogonal-quadric-boolean-cold-restored")


def _offset_sphere(source: BRepModel, /) -> BRepOffsetResult:
    result = brep_offset(
        source,
        1.0 / 3.0,
        source_id="archive-offset-sphere",
    )
    if result.source_model_id != source.source_revision:
        raise ValueError("The sphere offset result lost its exact source model identity.")
    model = result.model
    patch = model.patches[0]
    if not isinstance(patch, OffsetSurface) or not isinstance(
        patch.base, RevolutionSurface
    ):
        raise TypeError("The offset fixture must retain its native sphere revolution.")
    if (
        result.evidence.volume_status != "unresolved"
        or result.evidence.volume_enclosure is not None
    ):
        raise ValueError(
            "The smooth sphere offset must retain unresolved volume evidence."
        )
    return result


def _sphere_revolution_witness_payload(witness: object, /) -> object:
    if witness is None:
        return None
    if not isinstance(witness, tuple) or len(witness) != 4:
        raise TypeError("Sphere revolution witness must retain its four exact fields.")
    center, radius, axis, frame_sign = witness
    return {
        "center": [str(value) for value in center],
        "radius": str(radius),
        "axis": [str(value) for value in axis],
        "frame_sign": frame_sign,
    }


def _json_payload(value: object, /) -> object:
    return json.loads(json.dumps(value, sort_keys=True))


def _root_endpoint_payload(root: object | None, /) -> object:
    if root is None:
        return None
    if isinstance(root, ConicParameterEndpoint):
        return {
            "kind": "conic-parameter",
            "root_id": root.root.root_id,
            "endpoint_id": root.root_id,
            "identity": _json_payload(root.identity_payload()),
        }
    if not isinstance(root, NativePeriodEndpoint):
        raise TypeError("Sphere offset archive encountered an unsupported endpoint root.")
    return {
        "kind": "native-period",
        "axis": root.axis,
        "rational": [root.rational.numerator, root.rational.denominator],
        "turns": [root.turns.numerator, root.turns.denominator],
        "curve": _json_payload(carrier_payload(root.curve)),
        "patch": None
        if root.patch is None
        else _json_payload(carrier_payload(root.patch)),
    }


def _root_endpoint_arrays(geometry: BRepGeometry, /) -> dict[str, object]:
    return {
        "edges": [
            [edge, [_root_endpoint_payload(root) for root in roots]]
            for edge, roots in enumerate(geometry.edge_endpoint_roots)
        ],
        "coedges": [
            [coedge, [_root_endpoint_payload(root) for root in roots]]
            for coedge, roots in enumerate(geometry.coedge_endpoint_roots)
        ],
    }


def _conic_vertex_payload(vertex: int, root: BRepConicVertex, /) -> dict[str, object]:
    return {
        "vertex": vertex,
        "root_id": root.root_id,
        "conic_root_id": root.root.root_id,
        "conic_root": _json_payload(root.root.identity_payload()),
        "identity": _json_payload(root.identity_payload()),
        "offset_exact": [str(value) for value in root.offset_exact],
        "point_exact": [str(value) for value in root.point_exact],
    }


def _sphere_offset_root_state(
    source: BRepModel, target: BRepModel, /
) -> dict[str, object]:
    if len(source.patches) != 1 or not isinstance(source.patches[0], RevolutionSurface):
        raise TypeError("Sphere offset source lost its canonical RevolutionSurface.")
    if len(target.patches) != 1 or not isinstance(target.patches[0], OffsetSurface):
        raise TypeError("Sphere offset target lost its canonical OffsetSurface.")
    source_patch = source.patches[0]
    target_patch = target.patches[0]
    if not isinstance(target_patch.base, RevolutionSurface):
        raise TypeError("Sphere offset target lost its RevolutionSurface base.")
    source_witness = source_patch.exact_sphere_revolution_witness()
    target_witness = target_patch.exact_sphere_revolution_witness()
    if source_witness is None or target_witness is None:
        raise ValueError("Sphere offset lost its recomputed exact revolution witness.")
    if carrier_payload(target_patch.base) != carrier_payload(
        source_patch
    ) or target_witness != (*source_witness[:3], target_patch.frame_sign):
        raise ValueError(
            "Offset surface no longer retains its exact source sphere patch."
        )
    source_profile = source_patch.curve
    if not isinstance(source_profile, CircleCurve):
        raise TypeError("Sphere offset source profile is not its canonical circle.")
    source_geometry = source.geometry
    target_geometry = target.geometry
    profile_edges = tuple(
        edge
        for edge, curve_index in enumerate(source_geometry.edge_curves)
        if curve_index >= 0
        and carrier_payload(source_geometry.curves[curve_index])
        == carrier_payload(source_profile)
    )
    if len(profile_edges) != 1:
        raise ValueError("Sphere offset source has no unique profile edge carrier.")
    profile_edge = profile_edges[0]
    target_profile_edge_index = target_geometry.edge_curves[profile_edge]
    if target_profile_edge_index < 0:
        raise ValueError("Sphere offset target lost its profile edge carrier.")
    source_vertices: dict[int, BRepConicVertex] = {}
    for vertex, root in enumerate(source_geometry.vertex_roots):
        if not isinstance(root, BRepVertexRoot) or not isinstance(
            root.primary, BRepConicVertex
        ):
            raise TypeError("Sphere offset source lost an exact conic pole root.")
        primary = root.primary
        if (
            carrier_payload(primary.root.conic) != carrier_payload(source_profile)
            or any(primary.offset_exact)
            or primary.root.exact_point() != primary.point_exact
        ):
            raise ValueError("Sphere offset source conic pole root is stale.")
        source_vertices[vertex] = primary
    if len(source_vertices) != 2:
        raise ValueError("Sphere offset source must retain both exact pole roots.")
    center, radius, axis, _ = source_witness
    expected_poles = {
        tuple(center[index] + sign * radius * axis[index] for index in range(3))
        for sign in (-1, 1)
    }
    if {root.point_exact for root in source_vertices.values()} != expected_poles:
        raise ValueError("Sphere offset source pole roots differ from its exact witness.")

    def require_conic_endpoint(
        endpoint: object,
        vertex: int,
        carrier: AbstractCurve,
        /,
    ) -> ConicParameterEndpoint:
        if not isinstance(endpoint, ConicParameterEndpoint):
            raise TypeError("Sphere profile edge lost its conic parameter endpoint.")
        pole = source_vertices[vertex]
        if (
            endpoint.root.root_id != pole.root.root_id
            or endpoint.point_exact != pole.point_exact
            or carrier_payload(endpoint.carrier) != carrier_payload(carrier)
        ):
            raise ValueError(
                "Sphere profile endpoint is incompatible with its pole root."
            )
        return endpoint

    for endpoint_index, endpoint in enumerate(
        source_geometry.edge_endpoint_roots[profile_edge]
    ):
        vertex = source_geometry.edge_vertices[profile_edge][endpoint_index]
        require_conic_endpoint(endpoint, vertex, source_profile)
    profile_coedges = tuple(
        coedge
        for coedge, edge in enumerate(source_geometry.coedge_edges)
        if edge == profile_edge
    )
    for coedge in profile_coedges:
        pcurve = source_geometry.pcurves[coedge]
        for endpoint_index, endpoint in enumerate(
            source_geometry.coedge_endpoint_roots[coedge]
        ):
            vertex = source_geometry.edge_vertices[profile_edge][endpoint_index]
            require_conic_endpoint(endpoint, vertex, pcurve)

    distance = Fraction.from_float(1.0 / 3.0)
    if target_patch.distance_exact != distance:
        raise ValueError("Sphere offset did not retain the exact binary64 distance.")
    target_vertices: dict[int, BRepConicVertex] = {}
    for vertex, root in enumerate(target_geometry.vertex_roots):
        if not isinstance(root, BRepVertexRoot) or not isinstance(
            root.primary, BRepConicVertex
        ):
            raise TypeError("Sphere offset target lost an exact conic pole root.")
        primary = root.primary
        source_root = source_vertices[vertex]
        if (
            primary.root.root_id != source_root.root.root_id
            or primary.root.identity_payload() != source_root.root.identity_payload()
        ):
            raise ValueError(
                "Sphere offset target pole lost its source conic root identity."
            )
        direction = tuple(
            (point - coordinate) / radius
            for point, coordinate in zip(source_root.point_exact, center, strict=True)
        )
        expected_offset = tuple(
            target_patch.frame_sign * distance * component for component in direction
        )
        expected_point = tuple(
            point + offset
            for point, offset in zip(
                source_root.point_exact, expected_offset, strict=True
            )
        )
        if (
            primary.offset_exact != expected_offset
            or primary.point_exact != expected_point
        ):
            raise ValueError(
                "Sphere offset target pole does not retain its exact signed offset."
            )
        target_vertices[vertex] = primary

    target_profile_carrier = target_geometry.curves[target_profile_edge_index]
    target_profile_roots: list[ConicParameterEndpoint] = []
    for endpoint_index, endpoint in enumerate(
        target_geometry.edge_endpoint_roots[profile_edge]
    ):
        if not isinstance(endpoint, ConicParameterEndpoint):
            raise TypeError("Sphere offset target profile lost its conic endpoint.")
        vertex = target_geometry.edge_vertices[profile_edge][endpoint_index]
        source_endpoint = source_geometry.edge_endpoint_roots[profile_edge][
            endpoint_index
        ]
        if not isinstance(source_endpoint, ConicParameterEndpoint):
            raise TypeError("Sphere offset source profile lost its conic endpoint.")
        conic = endpoint.root.conic
        if not isinstance(conic, CircleCurve):
            raise TypeError(
                "Sphere offset target endpoint lost its exact offset meridian."
            )
        source_vertex = source_vertices[vertex]
        if (
            endpoint.parameter_exact != source_endpoint.parameter_exact
            or endpoint.root.cosine_exact != source_vertex.root.cosine_exact
            or endpoint.root.sine_exact != source_vertex.root.sine_exact
            or endpoint.root.turns != source_vertex.root.turns
            or conic.center_exact != source_profile.center_exact
            or conic.first_axis_exact != source_profile.first_axis_exact
            or conic.second_axis_exact != source_profile.second_axis_exact
            or conic.radius_exact != radius + target_patch.frame_sign * distance
            or endpoint.point_exact != target_vertices[vertex].point_exact
            or carrier_payload(endpoint.carrier)
            != carrier_payload(target_profile_carrier)
        ):
            raise ValueError(
                "Sphere offset target endpoint lost its exact conic lineage."
            )
        target_profile_roots.append(endpoint)
    for coedge in profile_coedges:
        pcurve = target_geometry.pcurves[coedge]
        for endpoint_index, endpoint in enumerate(
            target_geometry.coedge_endpoint_roots[coedge]
        ):
            if not isinstance(endpoint, ConicParameterEndpoint):
                raise TypeError(
                    "Sphere offset target profile coedge lost its conic endpoint."
                )
            vertex = target_geometry.edge_vertices[profile_edge][endpoint_index]
            source_endpoint = source_geometry.coedge_endpoint_roots[coedge][
                endpoint_index
            ]
            if not isinstance(source_endpoint, ConicParameterEndpoint):
                raise TypeError(
                    "Sphere offset source profile coedge lost its conic endpoint."
                )
            if (
                endpoint.parameter_exact != source_endpoint.parameter_exact
                or endpoint.root.identity_payload()
                != target_profile_roots[endpoint_index].root.identity_payload()
                or endpoint.point_exact != target_vertices[vertex].point_exact
                or carrier_payload(endpoint.carrier) != carrier_payload(pcurve)
            ):
                raise ValueError(
                    "Sphere offset target coedge lost its exact conic lineage."
                )

    def require_period_roots(
        geometry: BRepGeometry, patch: AbstractSurfacePatch, /
    ) -> None:
        for edge, curve_index in enumerate(geometry.edge_curves):
            if curve_index >= 0:
                continue
            roots = geometry.edge_endpoint_roots[edge]
            if not all(isinstance(root, NativePeriodEndpoint) for root in roots):
                raise TypeError("Sphere degenerate edge lost its native period roots.")
            start, end = roots
            assert isinstance(start, NativePeriodEndpoint)
            assert isinstance(end, NativePeriodEndpoint)
            coedge = next(
                coedge
                for coedge, edge_index in enumerate(geometry.coedge_edges)
                if edge_index == edge
            )
            pcurve = geometry.pcurves[coedge]
            if not (
                start.axis == end.axis == 0
                and start.rational == end.rational == 0
                and start.turns == 0
                and end.turns == 1
                and carrier_payload(start.curve) == carrier_payload(pcurve)
                and carrier_payload(end.curve) == carrier_payload(pcurve)
                and start.patch is not None
                and end.patch is not None
                and carrier_payload(start.patch) == carrier_payload(patch)
                and carrier_payload(end.patch) == carrier_payload(patch)
            ):
                raise ValueError(
                    "Sphere degenerate edge lost its exact native period binding."
                )
            coedge_roots = geometry.coedge_endpoint_roots[coedge]
            if any(root is not None for root in coedge_roots):
                if not all(
                    isinstance(root, NativePeriodEndpoint) for root in coedge_roots
                ):
                    raise TypeError(
                        "Sphere degenerate coedge lost its native period roots."
                    )
                coedge_start, coedge_end = coedge_roots
                assert isinstance(coedge_start, NativePeriodEndpoint)
                assert isinstance(coedge_end, NativePeriodEndpoint)
                if not (
                    coedge_start.axis == coedge_end.axis == 0
                    and coedge_start.rational == coedge_end.rational == 0
                    and coedge_start.turns == 0
                    and coedge_end.turns == 1
                    and coedge_start.patch is not None
                    and carrier_payload(coedge_start.patch) == carrier_payload(patch)
                    and coedge_end.patch is not None
                    and carrier_payload(coedge_end.patch) == carrier_payload(patch)
                ):
                    raise ValueError(
                        "Sphere degenerate coedge lost its exact period binding."
                    )

    require_period_roots(source_geometry, source_patch)
    require_period_roots(target_geometry, target_patch)
    return {
        "source_patch": _json_payload(carrier_payload(source_patch)),
        "target_patch": _json_payload(carrier_payload(target_patch)),
        "source_witness": _sphere_revolution_witness_payload(source_witness),
        "target_witness": _sphere_revolution_witness_payload(target_witness),
        "source_vertices": [
            _conic_vertex_payload(vertex, root)
            for vertex, root in sorted(source_vertices.items())
        ],
        "target_vertices": [
            _conic_vertex_payload(vertex, root)
            for vertex, root in sorted(target_vertices.items())
        ],
        "source_endpoints": _root_endpoint_arrays(source_geometry),
        "target_endpoints": _root_endpoint_arrays(target_geometry),
    }


def _write_revolution_offset_archive(directory: Path, /) -> None:
    contract = SpatialCoordinateContract.si()
    source = brep_sphere(
        1.0, coordinate_contract=contract, source_id="archive-revolution-sphere"
    )
    offset_result = brep_offset(
        source, 0.125, source_id="archive-revolution-sphere-offset"
    )
    if (
        offset_result.evidence.volume_status != "unresolved"
        or offset_result.evidence.volume_enclosure is not None
    ):
        raise ValueError(
            "The smooth sphere offset must retain unresolved volume evidence."
        )
    model = offset_result.model
    if not isinstance(model, BRepModel):
        raise ValueError("Native sphere offset did not produce its source carrier model.")
    patch = model.patches[0]
    if not isinstance(patch, OffsetSurface) or not isinstance(
        patch.base, RevolutionSurface
    ):
        raise TypeError("Sphere offset fixture lost its RevolutionSurface carrier.")
    witness = patch.exact_sphere_revolution_witness()
    if witness is None:
        raise ValueError("Sphere offset fixture omitted its exact revolution witness.")
    save_brep_archive(model, directory / "revolution-offset.phx")
    print(
        json.dumps(
            {
                "source_revision": model.source_revision,
                "witness": _sphere_revolution_witness_payload(witness),
                "parameter_bounds": np.asarray(model.parameter_bounds).tolist(),
            }
        )
    )


def _read_revolution_offset_archive(
    directory: Path, expected: dict[str, object], /
) -> None:
    model = load_brep_archive(directory / "revolution-offset.phx")
    if len(model.patches) != 1 or not isinstance(model.patches[0], OffsetSurface):
        raise TypeError("Cold sphere offset lost its canonical OffsetSurface patch.")
    patch = model.patches[0]
    if not isinstance(patch.base, RevolutionSurface):
        raise TypeError("Cold sphere offset lost its RevolutionSurface source.")
    source_witness = patch.base.exact_sphere_revolution_witness()
    stored_witness = patch.exact_sphere_revolution_witness()
    if source_witness is None or stored_witness is None:
        raise ValueError("Cold sphere offset lost its exact source revolution witness.")
    expected_witness = (*source_witness[:3], patch.frame_sign)
    point = np.asarray(patch.evaluate(jnp.asarray((0.37, 0.61), dtype=jnp.float64)))
    if (
        stored_witness != expected_witness
        or model.source_revision != expected["source_revision"]
        or _sphere_revolution_witness_payload(stored_witness) != expected["witness"]
        or np.asarray(model.parameter_bounds).tolist() != expected["parameter_bounds"]
    ):
        raise ValueError(
            "Cold sphere offset did not recompute its exact revolution witness."
        )
    np.testing.assert_allclose(np.linalg.norm(point), 1.125, rtol=0.0, atol=3.0e-15)
    print("revolution-offset-cold-restored")


def _curved_volume_material_chain() -> tuple[
    VolumeMaterialChain,
    CellMesh,
    CellGeometrySpec,
    CellMesh,
    CellGeometrySpec,
    MaterialReferenceDeclaration,
    tuple[VolumeMaterialDeformation, VolumeMaterialDeformation],
]:
    (
        root_mesh,
        root_geometry,
        source_geometry,
        refined_mesh,
        refined_geometry,
        refined_reference_map,
        witness,
    ) = make_curved_material_refinement()
    if witness is None:
        raise ValueError("The curved material fixture requires a common-root witness.")
    coverage_domain, root_coverage, refined_coverage = make_material_coverage_pair(
        root_mesh,
        root_geometry,
        refined_mesh,
        refined_geometry,
        region_id="aluminum",
        occurrence_path=("assembly-1", "occurrence-9"),
    )
    source_map = make_material_source(coverage_domain)
    root_reference_map = reference_map_witness((0,), (7,), (ROOT_REFERENCE_TETRAHEDRON,))
    reference = MaterialReferenceDeclaration(
        root_mesh,
        root_geometry,
        source_map,
        ("solid-cell",),
        ("aluminum",),
        root_coverage,
        frame_id="material-frame-4",
        policy=MaterialReferencePolicy.ACCEPTED_CONFIGURATION,
        occurrence_paths=(("assembly-1", "occurrence-9"),),
    )
    refinement_deformation = VolumeMaterialDeformation(
        reference,
        root_mesh,
        source_geometry,
        refined_mesh,
        refined_geometry,
        source_reference_maps={"root": root_reference_map},
        target_reference_maps={refined_mesh.blocks[0].name: refined_reference_map},
        source_coverage=root_coverage,
        target_coverage=refined_coverage,
        source_frame_id=reference.frame_id,
        target_frame_id=reference.frame_id,
    )
    coarsening_deformation = VolumeMaterialDeformation(
        reference,
        refined_mesh,
        refined_geometry,
        root_mesh,
        root_geometry,
        source_reference_maps={refined_mesh.blocks[0].name: refined_reference_map},
        target_reference_maps={"root": root_reference_map},
        source_coverage=refined_coverage,
        target_coverage=root_coverage,
        source_frame_id=reference.frame_id,
        target_frame_id=reference.frame_id,
    )
    refined = VolumeMaterialMap(
        root_mesh,
        source_geometry,
        refined_mesh,
        refined_geometry,
        refinement_deformation,
        ("solid-cell",),
        ("solid-cell", "solid-cell"),
        ("aluminum",),
        ("aluminum", "aluminum"),
        source_coverage=root_coverage,
        target_coverage=refined_coverage,
    )
    coarsened = VolumeMaterialMap(
        refined_mesh,
        refined_geometry,
        root_mesh,
        root_geometry,
        coarsening_deformation,
        ("solid-cell", "solid-cell"),
        ("solid-cell",),
        ("aluminum", "aluminum"),
        ("aluminum",),
        source_coverage=refined_coverage,
        target_coverage=root_coverage,
    )
    return (
        VolumeMaterialChain((refined, coarsened)),
        root_mesh,
        root_geometry,
        refined_mesh,
        refined_geometry,
        reference,
        (refinement_deformation, coarsening_deformation),
    )


def _rebuild_compiled_analytic_geometry(value: object, /) -> CompiledGeometry:
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

    kernel_owner = value.kernel
    if type(kernel_owner) is _BallKernel:
        kernel = _BallKernel(
            rebuild_binding(kernel_owner.center),
            rebuild_binding(kernel_owner.radius),
            dimension=kernel_owner.dimension,
            source_id=kernel_owner.source_id,
        )
    elif type(kernel_owner) is _TorusKernel:
        kernel = _TorusKernel(
            rebuild_binding(kernel_owner.center),
            rebuild_binding(kernel_owner.major),
            rebuild_binding(kernel_owner.minor),
            rebuild_binding(kernel_owner.angle),
            full=kernel_owner.full,
            source_id=kernel_owner.source_id,
        )
    else:
        raise TypeError("Analytic implicit profile uses an unsupported geometry kernel.")
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


def _rebuild_compiled_sphere_geometry(value: CompiledGeometry, /) -> CompiledGeometry:
    if not isinstance(value, CompiledGeometry) or type(value.kernel) is not _BallKernel:
        raise TypeError(
            "The adaptive archive fixture requires the registered sphere kernel."
        )
    return _rebuild_compiled_analytic_geometry(value)


def _rebuild_adaptive_surface_policy(
    value: AdaptiveImplicitSurfacePolicy, /
) -> AdaptiveImplicitSurfacePolicy:
    if not isinstance(value, AdaptiveImplicitSurfacePolicy):
        raise TypeError("Adaptive implicit archive policy has the wrong registered type.")
    return AdaptiveImplicitSurfacePolicy(
        enclosure=value.enclosure,
        relative_cell_diameter=value.relative_cell_diameter,
        root_tolerance=value.root_tolerance,
        minimum_face_area=value.minimum_face_area,
        allow_approximate_zero_set=value.allow_approximate_zero_set,
        allow_nonsmooth_field=value.allow_nonsmooth_field,
    )


def _rebuild_adaptive_implicit_source(
    value: AdaptiveImplicitSurfaceArchiveRecipe, /
) -> tuple[AdaptiveImplicitSurfaceArchiveRecipe, AdaptiveImplicitSurface]:
    if not isinstance(value, AdaptiveImplicitSurfaceArchiveRecipe):
        raise TypeError(
            "Adaptive implicit archive payload has the wrong source recipe type."
        )
    geometry = _rebuild_compiled_sphere_geometry(value.geometry)
    policy = _rebuild_adaptive_surface_policy(value.policy)
    recipe = AdaptiveImplicitSurfaceArchiveRecipe(
        geometry,
        value.domain,
        policy,
        value.source_id,
    )
    surface = discover_adaptive_implicit_surface(
        recipe.geometry,
        domain=recipe.domain,
        policy=recipe.policy,
        source_id=recipe.source_id,
    )
    if surface.region_ids != ("interior",):
        raise ValueError(
            "Recomputed adaptive source is not an established certification domain."
        )
    return recipe, surface


def _adaptive_implicit_surface_revision(surface: AdaptiveImplicitSurface, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "adaptive-implicit-surface-revision",
            "source_id": surface.source_id,
            "domain_id": surface.domain_id,
            "evidence_id": surface.evidence.evidence_id,
            "cover_id": None if surface.cover is None else surface.cover.cover_id,
            "topology": None
            if surface.topology is None
            else array_tree_fingerprint(surface.topology),
            "volume_query_id": surface.volume.query_id,
            "mesh": None
            if surface.mesh is None
            else array_tree_fingerprint(surface.mesh),
            "scaffold": array_tree_fingerprint(surface.scaffold),
            "feature_edges": array_tree_fingerprint(surface.feature_edges),
            "boundary_witness_boxes": array_tree_fingerprint(
                surface.boundary_witness_boxes
            ),
            "boundary_witnesses": array_tree_fingerprint(surface.boundary_witnesses),
            "boundary_errors": array_tree_fingerprint(surface.boundary_errors),
        }
    )


def _adaptive_implicit_archive_fixture() -> tuple[
    AdaptiveImplicitSurfaceArchiveRecipe, AdaptiveImplicitSurface
]:
    source_id = "archive-implicit-sphere"
    geometry = Sphere(
        (0.37, 0.13, -0.36),
        0.04,
        feature_id=source_id,
    ).compile()
    domain = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    policy = AdaptiveImplicitSurfacePolicy(relative_cell_diameter=0.005)
    surface = discover_adaptive_implicit_surface(
        geometry,
        domain=domain,
        policy=policy,
        source_id=source_id,
    )
    if surface.region_ids != ("interior",):
        raise ValueError(
            "The adaptive archive fixture requires a certified sphere domain."
        )
    return surface.archive_recipe, surface


def _reconstructed_source_archive_fixture() -> tuple[
    ReconstructedGeometrySourceArchiveRecipe, ReconstructedGeometrySource
]:
    dem = reconstruct_dem_region(
        np.asarray(
            ((0.02, 0.04, 0.03), (0.05, 0.09, 0.06), (0.01, 0.03, 0.02)),
            dtype=np.float64,
        ),
        recenter=False,
        tolerance=0.0,
        extrude_depth=0.75,
        feature_id="archive-reconstructed-dem",
    )
    original_source = dem.source
    if type(original_source) is not MeshRegion:
        raise TypeError("The reconstructed source fixture requires a MeshRegion.")
    vertices = np.asarray(original_source.vertices)
    faces = np.asarray(original_source.faces, dtype=np.int32)
    original_edges = np.asarray(original_source.triangle_mesh.topology.edges)
    vertex_ids = np.arange(vertices.shape[0], dtype=np.int64) + 1000
    edge_ids = np.arange(original_edges.shape[0], dtype=np.int64) + 2000
    facet_ids = np.arange(faces.shape[0], dtype=np.int64) + 3000
    region_id = 4000
    original = ReconstructedGeometrySource(
        original_source,
        dem.report,
        persistent_vertex_ids=vertex_ids,
        persistent_edge_ids=edge_ids,
        persistent_facet_ids=facet_ids,
        persistent_region_id=region_id,
    )

    vertex_permutation = np.roll(np.arange(vertices.shape[0], dtype=np.int32), 1)
    old_vertex_for_new = np.empty_like(vertex_permutation)
    old_vertex_for_new[vertex_permutation] = np.arange(vertices.shape[0], dtype=np.int32)
    facet_permutation = np.roll(np.arange(faces.shape[0], dtype=np.int32), 1)
    reordered_mesh = MeshRegion(
        vertices[vertex_permutation],
        old_vertex_for_new[faces[facet_permutation]],
        feature_id=original_source.feature_id,
    )
    edge_id_by_vertices = {
        tuple(sorted((int(edge[0]), int(edge[1])))): int(edge_id)
        for edge, edge_id in zip(original_edges, edge_ids, strict=True)
    }
    reordered_edges = np.asarray(reordered_mesh.triangle_mesh.topology.edges)
    reordered_edge_ids = np.asarray(
        [
            edge_id_by_vertices[
                tuple(
                    sorted(
                        (
                            int(vertex_permutation[int(edge[0])]),
                            int(vertex_permutation[int(edge[1])]),
                        )
                    )
                )
            ]
            for edge in reordered_edges
        ],
        dtype=np.int64,
    )
    reordered = ReconstructedGeometrySource(
        reordered_mesh,
        dem.report,
        persistent_vertex_ids=vertex_ids[vertex_permutation],
        persistent_edge_ids=reordered_edge_ids,
        persistent_facet_ids=facet_ids[facet_permutation],
        persistent_region_id=region_id,
    )
    if (
        reordered.source_id != original.source_id
        or reordered.source_revision != original.source_revision
    ):
        raise ValueError(
            "Persistent reconstruction identity changed with mesh row order."
        )
    return reordered.archive_recipe, reordered


def _continuation_composition(
    contract: SpatialCoordinateContract, /
) -> tuple[Composition, FiniteElementRestartManifest]:
    source = _square_model(contract)
    (
        volume_chain,
        volume_root_mesh,
        volume_root_geometry,
        volume_refined_mesh,
        volume_refined_geometry,
        material_reference,
        material_deformations,
    ) = _curved_volume_material_chain()
    shape = 3
    vertex_rows = np.arange(shape * shape, dtype=np.int64).reshape((shape, shape))
    lattice = np.stack(
        np.meshgrid(np.arange(shape), np.arange(shape), indexing="ij"), axis=-1
    ).reshape((-1, 2))
    coordinates = lattice.astype(np.float64) / 2.0
    representatives = vertex_rows[
        np.arange(shape)[:, None] % 2, np.arange(shape)[None, :] % 2
    ].reshape(-1)
    shifts = np.stack(
        np.meshgrid(np.arange(shape) // 2, np.arange(shape) // 2, indexing="ij"),
        axis=-1,
    ).reshape((-1, 2))
    triangles = []
    for i in range(shape - 1):
        for j in range(shape - 1):
            a = int(vertex_rows[i, j])
            b = int(vertex_rows[i + 1, j])
            c = int(vertex_rows[i + 1, j + 1])
            d = int(vertex_rows[i, j + 1])
            triangles.extend(((a, b, c), (a, c, d)))
    block = CellBlock(
        "triangles",
        "triangle",
        np.asarray(triangles, dtype=np.int32),
        global_ids=np.arange(500, 508, dtype=np.int64),
    )
    lifted = CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=np.arange(shape * shape, dtype=np.int64),
    )
    periodic_cell = PeriodicCell(np.eye(2, dtype=np.float64))
    topology = PeriodicMeshTopology(lifted, periodic_cell, representatives, shifts)
    mesh = CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=np.arange(shape * shape, dtype=np.int64),
        periodic_topology=topology,
    )
    coordinate_element = RationalComposedCellGeometryElement(
        lagrange_element("triangle", 1),
        "triangle",
        (
            RationalPolynomial({(1, 0): Fraction(1, 3)}, {(0, 0): Fraction(1, 3)}),
            RationalPolynomial({(0, 1): Fraction(1, 3)}, {(0, 0): Fraction(1, 3)}),
        ),
    )
    geometry = CellGeometrySpec(
        {block.name: coordinate_element},
        {block.name: block.vertices},
        coordinates,
    )
    result = certify_cell_mesh(mesh, contract, geometry=geometry)
    mesh, geometry = result.mesh, result.geometry
    periodic = mesh.periodic_topology
    if periodic is None:
        raise TypeError("Periodic continuation fixture lost its quotient topology.")
    part = MeshPart("periodic-cell-mesh", result)
    distribution = MeshDistribution(
        part,
        CellPartition(np.zeros((mesh.blocks[0].cell_count,), dtype=np.int32), 1),
        halo_width=0,
        partition_kind=MeshPartitionKind.PROVIDER,
        partition_provenance="archive-test-owner",
    )
    scalar_field_spec = FiniteElementFieldSpec(
        "temperature", lagrange_element("triangle", 1)
    )
    scalar_layout = FiniteElementAcceptedFieldLayout(
        scalar_field_spec,
        transfer_semantics=FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION,
    )
    circulation_field_spec = FiniteElementFieldSpec(
        "circulation", form_element("triangle", 1, 1, proxy="circulation")
    )
    circulation_layout = FiniteElementAcceptedFieldLayout(
        circulation_field_spec,
        periodic_quotient=False,
    )
    fe_discretization = FiniteElementPlan(
        mesh,
        (scalar_field_spec, circulation_field_spec),
        coordinate_spec=geometry,
    ).prepare()
    field = np.linspace(
        0.25, 1.25, fe_discretization.dof_maps[0].global_dof_count, dtype=np.float64
    )
    edge_field = np.linspace(
        0.5, 1.5, fe_discretization.dof_maps[1].global_dof_count, dtype=np.float64
    )
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    cell_owner_ranks = np.zeros((cell_ids.size,), dtype=np.int32)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    vertex_owner_ranks = np.zeros((vertex_ids.size,), dtype=np.int32)
    field_values: list[jax.Array] = []
    field_ownership: list[FiniteElementDofOwnership] = []
    for field_index, (layout, values) in enumerate(
        zip(
            (scalar_layout, circulation_layout),
            (field, edge_field),
            strict=True,
        )
    ):
        identity = finite_element_dof_identity_plan(
            fe_discretization,
            field_index,
            field_layout_id=layout.layout_id,
            distribution_id=distribution.distribution_id,
            topology_id=mesh.topology_id,
            geometry_id=cell_geometry_id(geometry),
            local_rank=0,
            part_count=1,
            cell_ids=cell_ids,
            cell_owner_ranks=cell_owner_ranks,
            vertex_ids=vertex_ids,
            vertex_owner_ranks=vertex_owner_ranks,
            periodic_quotient=layout.periodic_quotient,
            coordinate_spec=geometry,
        )
        sharded_values, ownership = identity.shard_values(values)
        field_values.append(sharded_values)
        field_ownership.append(ownership)
    material_site = MaterialSiteId("steel:integration-site:0")
    materials = MaterialTransaction(
        (
            MaterialState(
                material_site,
                "linear-elastic:steel",
                np.asarray((2.5, 3.5), dtype=np.float64),
                trial=np.asarray((2.75, 3.75), dtype=np.float64),
                state_version=4,
            ),
        )
    )
    state = FiniteElementAcceptedState(
        tuple(field_values),
        np.asarray(0.125, dtype=np.float64),
        7,
        mesh.topology_id,
        "fe-prepared:mesh-0",
        "fe-compiled:mesh-0",
        field_layouts=(scalar_layout, circulation_layout),
        materials=materials,
        schedule_cursor=3,
        state_version=5,
        field_ownership=tuple(field_ownership),
    )
    rng_key = jax.random.key(2026)
    restart_manifest = FiniteElementRestartManifest(
        state,
        auxiliary_state=(
            ("temperature_history", field_values[0] * 0.5),
            ("time_history", np.asarray((0.0, 0.0625), dtype=np.float64)),
        ),
        integrator_state=(
            ("previous_step", np.asarray((0.125,), dtype=np.float64)),
            ("rng", rng_key),
        ),
        auxiliary_layouts=(
            FiniteElementRestartStateLayout(
                "temperature_history",
                FiniteElementRestartStateRole.HISTORY,
                FiniteElementRestartStateAssociation.OWNED_ROWS,
                ownership=field_ownership[0],
            ),
            FiniteElementRestartStateLayout(
                "time_history",
                FiniteElementRestartStateRole.HISTORY,
                FiniteElementRestartStateAssociation.REPLICATED,
            ),
        ),
        integrator_layouts=(
            FiniteElementRestartStateLayout(
                "previous_step",
                FiniteElementRestartStateRole.HISTORY,
                FiniteElementRestartStateAssociation.REPLICATED,
            ),
            FiniteElementRestartStateLayout(
                "rng",
                FiniteElementRestartStateRole.RNG,
                FiniteElementRestartStateAssociation.REPLICATED,
                prng_impl=str(jax.random.key_impl(rng_key)),
            ),
        ),
        material_layouts=(
            FiniteElementRestartStateLayout(
                "steel:integration-site:0",
                FiniteElementRestartStateRole.MODEL_STATE,
                FiniteElementRestartStateAssociation.REPLICATED,
            ),
        ),
    )

    source_entry = CompositionEntry(
        source,
        entry_id="cad/source",
        role="topology",
        owner_id="phydrax.geometry.brep",
        structure_id=source.geometry.geometry_id,
        revision_id=source.source_revision,
        semantics_id="native-source-brep",
    )
    mesh_entry = CompositionEntry(
        mesh,
        entry_id="mesh/topology",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=mesh.topology_id,
        revision_id=mesh.mesh_id,
        semantics_id="cell-mesh",
    )
    geometry_entry = CompositionEntry(
        geometry,
        entry_id="mesh/geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=geometry.geometry_layout_id,
        revision_id=mesh.geometry_id,
        semantics_id="cell-coordinate-map",
        dependencies=(mesh_entry.binding("structure"),),
    )
    result_entry = CompositionEntry(
        result,
        entry_id="mesh/result",
        role="discretization",
        owner_id="phydrax.meshing",
        structure_id=mesh.topology_id,
        revision_id=result.result_id,
        semantics_id="accepted-cell-meshing-result",
        dependencies=(
            mesh_entry.binding("structure"),
            geometry_entry.binding("structure"),
        ),
    )
    state_entry = CompositionEntry(
        state,
        entry_id="solver/state",
        role="physical-state",
        owner_id="phydrax.solver.finite-element",
        structure_id=mesh.topology_id,
        revision_id=state.accepted_id,
        semantics_id="accepted-fe-state",
        dependencies=(
            mesh_entry.binding("structure"),
            geometry_entry.binding("structure"),
            result_entry.binding("revision"),
        ),
    )
    material_entry = CompositionEntry(
        materials,
        entry_id="solver/materials",
        role="model-state",
        owner_id="phydrax.equations.material",
        structure_id=materials.layout_id,
        revision_id=materials.transaction_id,
        semantics_id="constitutive-material-state",
        dependencies=(state_entry.binding("structure"),),
    )
    periodic_entry = CompositionEntry(
        periodic,
        entry_id="mesh/periodicity",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=periodic.periodic_topology_id,
        revision_id=periodic.periodic_topology_id,
        semantics_id="periodic-mesh-quotient",
        dependencies=(mesh_entry.binding("structure"),),
    )
    distribution_entry = CompositionEntry(
        distribution,
        entry_id="mesh/distribution",
        role="discretization",
        owner_id="phydrax.meshing",
        structure_id=distribution.part.part_id,
        revision_id=distribution.distribution_id,
        semantics_id="distributed-mesh-ownership",
        dependencies=(result_entry.binding("revision"),),
    )
    volume_root_mesh_entry = CompositionEntry(
        volume_root_mesh,
        entry_id="material/root-mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=volume_root_mesh.topology_id,
        revision_id=volume_root_mesh.mesh_id,
        semantics_id="curved-material-root-mesh",
    )
    volume_root_geometry_entry = CompositionEntry(
        volume_root_geometry,
        entry_id="material/root-geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=volume_root_geometry.geometry_layout_id,
        revision_id=cell_geometry_id(volume_root_geometry),
        semantics_id="curved-material-root-coordinate-map",
        dependencies=(
            volume_root_mesh_entry.binding("structure"),
            volume_root_mesh_entry.binding("revision"),
        ),
    )
    volume_refined_mesh_entry = CompositionEntry(
        volume_refined_mesh,
        entry_id="material/refined-mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=volume_refined_mesh.topology_id,
        revision_id=volume_refined_mesh.mesh_id,
        semantics_id="curved-material-refined-mesh",
    )
    volume_refined_geometry_entry = CompositionEntry(
        volume_refined_geometry,
        entry_id="material/refined-geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=volume_refined_geometry.geometry_layout_id,
        revision_id=cell_geometry_id(volume_refined_geometry),
        semantics_id="curved-material-refined-coordinate-map",
        dependencies=(
            volume_refined_mesh_entry.binding("structure"),
            volume_refined_mesh_entry.binding("revision"),
        ),
    )
    material_reference_entry = CompositionEntry(
        material_reference,
        entry_id="material/reference-declaration",
        role="transfer",
        owner_id="phydrax.geometry",
        structure_id=volume_root_geometry.geometry_layout_id,
        revision_id=material_reference.declaration_id,
        semantics_id="material-reference-configuration",
        dependencies=(
            volume_root_mesh_entry.binding("structure"),
            volume_root_mesh_entry.binding("revision"),
            volume_root_geometry_entry.binding("structure"),
            volume_root_geometry_entry.binding("revision"),
        ),
    )
    refinement_deformation_entry = CompositionEntry(
        material_deformations[0],
        entry_id="material/refinement-deformation",
        role="transfer",
        owner_id="phydrax.geometry",
        structure_id=material_reference.declaration_id,
        revision_id=material_deformations[0].deformation_id,
        semantics_id="material-reference-deformation",
        dependencies=(
            material_reference_entry.binding("revision"),
            volume_root_mesh_entry.binding("structure"),
            volume_root_mesh_entry.binding("revision"),
            volume_root_geometry_entry.binding("structure"),
            volume_root_geometry_entry.binding("revision"),
            volume_refined_mesh_entry.binding("structure"),
            volume_refined_mesh_entry.binding("revision"),
            volume_refined_geometry_entry.binding("structure"),
            volume_refined_geometry_entry.binding("revision"),
        ),
    )
    coarsening_deformation_entry = CompositionEntry(
        material_deformations[1],
        entry_id="material/coarsening-deformation",
        role="transfer",
        owner_id="phydrax.geometry",
        structure_id=material_reference.declaration_id,
        revision_id=material_deformations[1].deformation_id,
        semantics_id="material-reference-deformation",
        dependencies=(
            material_reference_entry.binding("revision"),
            volume_refined_mesh_entry.binding("structure"),
            volume_refined_mesh_entry.binding("revision"),
            volume_refined_geometry_entry.binding("structure"),
            volume_refined_geometry_entry.binding("revision"),
            volume_root_mesh_entry.binding("structure"),
            volume_root_mesh_entry.binding("revision"),
            volume_root_geometry_entry.binding("structure"),
            volume_root_geometry_entry.binding("revision"),
        ),
    )
    volume_chain_entry = CompositionEntry(
        volume_chain,
        entry_id="material/volume-chain",
        role="transfer",
        owner_id="phydrax.geometry",
        structure_id=volume_chain.source_geometry_id,
        revision_id=volume_chain.chain_id,
        semantics_id="curved-volume-material-chain",
        dependencies=(
            volume_root_mesh_entry.binding("structure"),
            volume_root_mesh_entry.binding("revision"),
            volume_root_geometry_entry.binding("structure"),
            volume_root_geometry_entry.binding("revision"),
            volume_refined_mesh_entry.binding("structure"),
            volume_refined_mesh_entry.binding("revision"),
            volume_refined_geometry_entry.binding("structure"),
            volume_refined_geometry_entry.binding("revision"),
            material_reference_entry.binding("revision"),
            refinement_deformation_entry.binding("revision"),
            coarsening_deformation_entry.binding("revision"),
        ),
    )
    composition = Composition(
        (
            source_entry,
            mesh_entry,
            geometry_entry,
            result_entry,
            state_entry,
            material_entry,
            periodic_entry,
            distribution_entry,
            volume_root_mesh_entry,
            volume_root_geometry_entry,
            volume_refined_mesh_entry,
            volume_refined_geometry_entry,
            material_reference_entry,
            refinement_deformation_entry,
            coarsening_deformation_entry,
            volume_chain_entry,
        ),
        boundary_id="accepted-step:7",
    )
    return composition, restart_manifest


def _rebuild_periodic_identification(
    value: object, /
) -> PeriodicCell | PeriodicIsometryGroup:
    if isinstance(value, PeriodicCell):
        return _rebuild_as(value, PeriodicCell)
    if type(value) is PeriodicIsometryGroup:
        group = PeriodicIsometryGroup(value.generators)
        if (
            group.group_id != value.group_id
            or group.orders != value.orders
            or group.linear_orders != value.linear_orders
            or group.translation_periods != value.translation_periods
        ):
            raise ValueError(
                "Stored periodic isometry group differs from its exact generators."
            )
        return group
    raise TypeError("Stored periodic identification has no registered constructor.")


def _rebuild_periodic_topology(
    value: PeriodicMeshTopology,
    lifted: CellMesh,
    /,
    *,
    actual_geometry: CellGeometrySpec | None = None,
) -> PeriodicMeshTopology:
    return PeriodicMeshTopology.from_stored_fields(
        value,
        lifted,
        actual_geometry=actual_geometry,
    )


def _rebuild_cell_geometry_restriction_source(
    value: CellGeometryRestrictionSource, /
) -> CellGeometryRestrictionSource:
    carrier_geometry = (
        None
        if value.carrier_geometry is None
        else _rebuild_as(value.carrier_geometry, CellGeometrySpec)
    )
    reference_maps = {
        name: (
            None if witness is None else _rebuild_reference_cell_map_witnesses(witness)
        )
        for name, witness in value.block_parent_reference_maps.items()
    }
    rebuilt = CellGeometryRestrictionSource(
        value.source_geometry_id,
        value.source_topology_id,
        value.block_parent_cell_ids,
        value.block_parent_vertex_ids,
        block_source_blocks=value.block_source_blocks,
        carrier_publication=value.carrier_publication,
        block_parent_reference_maps=reference_maps,
        carrier_geometry=carrier_geometry,
    )
    if rebuilt.restriction_source_id != value.restriction_source_id:
        raise ValueError("Cell geometry restriction source identity changed.")
    return rebuilt


def _fraction_tuple_for_archive(value: object, /) -> tuple[Fraction, ...]:
    if not isinstance(value, tuple):
        raise TypeError("Frame-grid knot coordinates must remain exact tuples.")
    knots: list[Fraction] = []
    for knot in value:
        if not isinstance(knot, Fraction):
            raise TypeError("Frame-grid knots must remain exact Fraction values.")
        knots.append(knot)
    return tuple(knots)


def _frame_axis_knots_for_archive(
    value: object, /
) -> tuple[tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]], ...]:
    if not isinstance(value, tuple):
        raise TypeError("Frame-grid block knot banks must remain exact tuples.")
    result: list[
        tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]]
    ] = []
    for axes in value:
        if not isinstance(axes, tuple) or len(axes) != 3:
            raise TypeError("Each frame-grid block must retain three axis knot vectors.")
        u, v, w = axes
        result.append(
            (
                _fraction_tuple_for_archive(u),
                _fraction_tuple_for_archive(v),
                _fraction_tuple_for_archive(w),
            )
        )
    return tuple(result)


def _rebuild_hex_frame_integer_layout(value: object, /) -> _HexFrameIntegerLayout:
    from phydrax.meshing._hex_frame_records import HexFrameIntegerLayout

    if type(value) is not HexFrameIntegerLayout:
        raise TypeError("Stored frame-grid layout has an unexpected concrete type.")
    layout = HexFrameIntegerLayout(
        value.cell_global_ids,
        value.cell_block_indices,
        value.cell_integer_coordinates,
        np.asarray(value.block_axes, dtype=np.float64),
        np.asarray(value.block_origins, dtype=np.float64),
        _frame_axis_knots_for_archive(value.block_axis_knots),
        value.face_connections,
        value.face_transitions,
        value.face_offsets,
        value.boundary_face_parity,
        value.source_complex_id,
        value.source_mesh_id,
        value.source_topology_id,
        value.schedule_id,
        value.frame_id,
    )
    if not isinstance(layout, _HexFrameIntegerLayout):
        raise TypeError(
            "Rebuilt frame-grid layout does not satisfy its exact source protocol."
        )
    if layout.layout_id != value.layout_id:
        raise ValueError(
            "Stored frame-grid layout identity differs from its exact charts."
        )
    return layout


def _rebuild_cut_coordinate_map(
    value: _CutCoordinateMap, target_mesh: CellMesh, /
) -> _CutCoordinateMap:
    if type(value) is not _CutCoordinateMap:
        raise TypeError("Stored cut-coordinate map has an unexpected concrete type.")
    source = restore_model_archive_value(value.source_complex)
    if type(source) is not PiecewiseLinearComplex:
        raise TypeError("Cut-coordinate replay did not renew its PLC source.")
    source_mesh = _rebuild_continuation_value(value.source_mesh)
    if not isinstance(source_mesh, CellMesh):
        raise TypeError("Cut-coordinate replay did not renew its source mesh.")
    source_geometry = _rebuild_geometry_for_mesh(value.source_geometry, source_mesh)
    if not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("Cut-coordinate replay did not renew its source geometry.")
    frame_layout = (
        None
        if value.plan.frame_layout is None
        else _rebuild_hex_frame_integer_layout(value.plan.frame_layout)
    )
    schedule = NativeHexGridSchedule(
        value.schedule.route,
        minimum_scaled_jacobian=value.schedule.minimum_scaled_jacobian,
        minimum_mean_ratio=value.schedule.minimum_mean_ratio,
        maximum_aspect_ratio=value.schedule.maximum_aspect_ratio,
    )
    if schedule.schedule_id != value.schedule.schedule_id:
        raise ValueError("Stored native hex schedule identity changed.")
    plan = BalancedCutGridPlan(
        value.plan.axis_knots,
        source,
        source_mesh,
        schedule,
        value.plan.target_size,
        source_id=value.source_id,
        source_revision=value.source_revision,
        frame_axes=np.asarray(value.plan.frame_axes, dtype=np.float64),
        frame_origin=np.asarray(value.plan.frame_origin, dtype=np.float64),
        frame_id=value.plan.frame_id,
        frame_layout=frame_layout,
    )
    if (
        plan.plan_id != value.plan.plan_id
        or plan.source_complex_id != value.plan.source_complex_id
        or plan.source_mesh_id != value.plan.source_mesh_id
        or plan.source_topology_id != value.plan.source_topology_id
        or plan.source_vertex_ids != value.plan.source_vertex_ids
        or plan.schedule_id != value.plan.schedule_id
    ):
        raise ValueError(
            "Stored balanced-grid plan does not replay from its exact inputs."
        )
    rebuilt = _CutCoordinateMap(
        value.source_id,
        value.source_revision,
        source,
        plan,
        schedule,
        source_mesh,
        source_geometry,
        value.cell_regions,
        target_mesh,
        value.node_keys,
        value.exact_vertices,
    )
    if (
        rebuilt.source_map_id != value.source_map_id
        or not np.array_equal(rebuilt.rounded_vertices, value.rounded_vertices)
        or rebuilt.target_mesh_id != value.target_mesh_id
        or rebuilt.target_topology_id != value.target_topology_id
        or not np.array_equal(rebuilt.cell_regions, value.cell_regions)
        or rebuilt.node_keys != value.node_keys
        or rebuilt.exact_vertices != value.exact_vertices
    ):
        raise ValueError(
            "Stored cut-coordinate map does not reproduce its exact replay data."
        )
    return _CutCoordinateMap.from_stored_fields(rebuilt, target_mesh)


def _rebuild_restricted_coordinate_map(
    value: _RestrictedCoordinateMap, target_mesh: CellMesh, /
) -> _RestrictedCoordinateMap:
    if type(value) is not _RestrictedCoordinateMap:
        raise TypeError("Stored restricted coordinates have an unexpected concrete type.")
    source_mesh = _rebuild_continuation_value(value.source_mesh)
    if not isinstance(source_mesh, CellMesh):
        raise TypeError("Restricted coordinate replay did not renew its source mesh.")
    source_geometry = _rebuild_geometry_for_mesh(value.source_geometry, source_mesh)
    if not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("Restricted coordinate replay did not renew its source geometry.")
    rebuilt = _RestrictedCoordinateMap(source_mesh, source_geometry, target_mesh)
    if (
        rebuilt.source_mesh_id != value.source_mesh_id
        or rebuilt.source_topology_id != value.source_topology_id
        or rebuilt.target_mesh_id != value.target_mesh_id
        or rebuilt.target_topology_id != value.target_topology_id
        or rebuilt.target_geometry_id != value.target_geometry_id
        or rebuilt.source_id != value.source_id
        or rebuilt.source_revision != value.source_revision
        or rebuilt.source_map_id != value.source_map_id
        or not np.array_equal(rebuilt.rounded_vertices, value.rounded_vertices)
    ):
        raise ValueError(
            "Stored restricted coordinate map does not renew from its exact source owners."
        )
    return _RestrictedCoordinateMap.from_stored_fields(rebuilt, target_mesh=target_mesh)


def _rebuild_reference_cell_map_witnesses(value: object, /) -> ReferenceCellMapWitnesses:
    from phydrax.discretization._nested_reference_types import (
        ReferenceCellMapWitnesses,
        validate_reference_cell_map_witnesses,
    )

    if type(value) is not ReferenceCellMapWitnesses:
        raise TypeError("Stored reference-map evidence has an unexpected concrete type.")
    witness = ReferenceCellMapWitnesses(
        value.target_cell_ids,
        value.source_cell_ids,
        value.target_cell_kind,
        value.source_cell_kind,
        value.source_reference_vertices,
        value.source_reference_exact_vertices,
        value.source_reference_chart_arguments,
    )
    validate_reference_cell_map_witnesses(witness)
    return witness


def _decode_finite_element_projections[T](value: T, /) -> T:
    """Restore nested FiniteElementSpec archive projections inside a raw record.

    Raw coordinate maps keep these projections, so their recomputed IDs and tree
    content differ from the live maps; the owner's restore then re-proves the
    decoded record against its stored IDs.
    """
    from phydrax._model._structure import _FiniteElementSpecArchiveProjection

    return jax.tree_util.tree_map(
        lambda node: (
            node.rebuild() if type(node) is _FiniteElementSpecArchiveProjection else node
        ),
        value,
        is_leaf=lambda node: type(node) is _FiniteElementSpecArchiveProjection,
    )


def _rebuild_domain_coverage_certificate(
    value: object,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> DomainCoverageCertificate:
    from phydrax.geometry.implicit._restricted_delaunay_coverage import (
        RestrictedDelaunayCoverageTransportWitness,
    )

    if (
        not isinstance(value, DomainCoverageCertificate)
        or type(value) is not DomainCoverageCertificate
    ):
        raise TypeError("Stored material coverage has an unexpected concrete type.")
    value = _decode_finite_element_projections(value)
    domain = _rebuild_continuation_value(value.domain)
    if not isinstance(
        domain,
        (
            PiecewiseLinearDomain,
            MappedReferenceDomain,
            MeshingDomain,
            AdaptiveImplicitSurface,
            AnalyticImplicitProfile,
        ),
    ):
        raise TypeError("Coverage certificate lost its canonical domain owner.")
    source = (
        None if value.source is None else _rebuild_as(value.source, SourceBoundaryQuery)
    )
    source_witness = (
        None
        if value.source_coverage_witness is None
        else _rebuild_continuation_value(value.source_coverage_witness)
    )
    if source_witness is not None and not isinstance(
        source_witness,
        (RestrictedDelaunayBoundaryWitness, RestrictedDelaunayCoverageTransportWitness),
    ):
        raise TypeError(
            "Coverage certificate lost its typed restricted-Delaunay witness."
        )
    embedding = GlobalEmbeddingCertificate.from_stored_fields(value.embedding)
    fidelity = None
    if value.fidelity is not None:
        fidelity_source = (
            source
            if source is not None
            else _rebuild_as(value.fidelity.source, SourceBoundaryQuery)
        )
        fidelity_witness = _rebuild_continuation_value(value.fidelity.witness)
        if not isinstance(
            fidelity_witness,
            (
                SurfaceChartWitness,
                SourceBoundaryChartWitness,
                SourceBoundaryPointWitness,
                SourceBoundaryDistanceWitness,
                SurfaceEnvelopeDistanceWitness,
            ),
        ):
            raise TypeError("Coverage fidelity lost its typed source-bound witness.")
        fidelity = certify_source_fidelity(
            mesh,
            geometry,
            fidelity_source,
            tolerance=value.fidelity.tolerance,
            maximum_normal_deviation=value.fidelity.maximum_normal_deviation,
            witness=fidelity_witness,
        )
        if fidelity.certificate_id != value.fidelity.certificate_id:
            raise ValueError(
                "Stored source-fidelity proof does not renew from its source query."
            )
        if source is None:
            source = fidelity_source
    value.require_current(
        mesh,
        geometry,
        domain,
        value.cell_regions,
        embedding=embedding,
        source=source,
        fidelity=fidelity,
        source_coverage_witness=source_witness,
    )
    rebuilt = DomainCoverageCertificate(
        mesh,
        geometry,
        domain,
        value.cell_regions,
        embedding=embedding,
        source=source,
        fidelity=fidelity,
        source_coverage_witness=source_witness,
    )
    if rebuilt.certificate_id != value.certificate_id:
        raise ValueError(
            "Stored material coverage identity changed on cold reconstruction."
        )
    return rebuilt


def _rebuild_cell_geometry_transition(
    value: object,
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
) -> CellGeometryTransition:
    from phydrax.discretization._cell_geometry_transition_types import (
        CellGeometryTransition,
        CellGeometryTransitionEvidence,
    )
    from phydrax.discretization._nested_reference_types import (
        NestedReferenceWitnesses,
        validate_nested_reference_witnesses,
    )

    if (
        not isinstance(value, CellGeometryTransition)
        or type(value) is not CellGeometryTransition
    ):
        raise TypeError(
            "Stored material coordinate transition has an unexpected concrete type."
        )
    kind = value.evidence.kind
    if kind == "nested_restriction":
        evidence_kind = "nested_restriction"
    elif kind == "coarsening_interpolation":
        evidence_kind = "coarsening_interpolation"
    elif kind == "parent_restoration":
        evidence_kind = "parent_restoration"
    elif kind == "vertex_displacement":
        evidence_kind = "vertex_displacement"
    elif kind == "source_realization":
        evidence_kind = "source_realization"
    elif kind == "bounded_reconstruction":
        evidence_kind = "bounded_reconstruction"
    else:
        raise ValueError("Stored geometry transition has an unknown evidence kind.")
    evidence = CellGeometryTransitionEvidence(
        evidence_kind,
        reference_map_count=value.evidence.reference_map_count,
        refinement_map_count=value.evidence.refinement_map_count,
        coarsening_map_count=value.evidence.coarsening_map_count,
        continuity_residual=value.evidence.continuity_residual,
        approximation_bound=value.evidence.approximation_bound,
    )
    if evidence.evidence_id != value.evidence.evidence_id:
        raise ValueError("Stored geometry transition evidence identity changed.")
    reference_map = value.reference_map
    if isinstance(reference_map, tuple):
        reference_map = tuple(
            _rebuild_reference_cell_map_witnesses(item) for item in reference_map
        )
    elif reference_map is not None:
        reference_map = _rebuild_reference_cell_map_witnesses(reference_map)

    def nested_witness(item: object) -> NestedReferenceWitnesses | None:
        if item is None:
            return None
        if type(item) is not NestedReferenceWitnesses:
            raise TypeError(
                "Stored nested geometry ancestry has an unexpected concrete type."
            )
        rebuilt = NestedReferenceWitnesses(
            item.fine_cell_ids,
            item.coarse_cell_ids,
            item.fine_reference_vertices,
            item.fine_reference_homogeneous,
            item.fine_reference_exact_vertices,
        )
        validate_nested_reference_witnesses(rebuilt)
        return rebuilt

    transition = CellGeometryTransition(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        evidence=evidence,
        reference_map=reference_map,
        refinement=nested_witness(value.refinement),
        coarsening=nested_witness(value.coarsening),
    )
    if transition.transition_id != value.transition_id:
        raise ValueError(
            "Stored geometry transition does not reproduce from endpoint owners."
        )
    return transition


def _rebuild_boundary_layer_evidence(value: object, /) -> BoundaryLayerEvidence:
    from phydrax.meshing._boundary_layer import BoundaryLayerEvidence
    from phydrax.meshing._controls import BoundaryLayerCollisionPolicy

    if (
        not isinstance(value, BoundaryLayerEvidence)
        or type(value) is not BoundaryLayerEvidence
    ):
        raise TypeError("Stored boundary-layer evidence has an unexpected concrete type.")
    policy = BoundaryLayerCollisionPolicy(value.collision_policy.value)
    rebuilt = BoundaryLayerEvidence(
        requested_thicknesses=value.requested_thicknesses,
        achieved_thicknesses=value.achieved_thicknesses,
        minimum_thicknesses=value.minimum_thicknesses,
        maximum_thicknesses=value.maximum_thicknesses,
        achieved_growth_rates=value.achieved_growth_rates,
        layer_active=value.layer_active,
        column_count=value.column_count,
        fan_column_count=value.fan_column_count,
        corner_patch_count=value.corner_patch_count,
        convex_ridge_count=value.convex_ridge_count,
        concave_ridge_count=value.concave_ridge_count,
        rim_vertex_count=value.rim_vertex_count,
        minimum_visibility=value.minimum_visibility,
        maximum_stretch=value.maximum_stretch,
        collision_policy=policy,
        predicted_collision_vertex_count=value.predicted_collision_vertex_count,
        detected_collision_count=value.detected_collision_count,
        resolution_iterations=value.resolution_iterations,
        reduced_vertex_count=value.reduced_vertex_count,
        terminated_vertex_count=value.terminated_vertex_count,
        merged_vertex_count=value.merged_vertex_count,
        minimum_scale=value.minimum_scale,
        cell_counts=value.cell_counts,
        certified_valid_count=value.certified_valid_count,
    )
    if rebuilt.evidence_id != value.evidence_id:
        raise ValueError("Stored boundary-layer evidence identity changed.")
    return rebuilt


def _rebuild_layer_column_projection_identity(
    value: object, /
) -> LayerColumnProjectionIdentity:
    from phydrax.meshing._layer_curving_records import LayerColumnProjectionIdentity

    if (
        not isinstance(value, LayerColumnProjectionIdentity)
        or type(value) is not LayerColumnProjectionIdentity
    ):
        raise TypeError("Stored layer column identity has an unexpected concrete type.")
    rebuilt = LayerColumnProjectionIdentity(
        value.layer_indices,
        value.source_wall_cell_ids,
        value.front_row_ids,
        value.front_row_identities,
        value.source_column_ids,
        value.source_column_identities,
        value.source_column_root_vertex_ids,
    )
    if rebuilt.identity_id != value.identity_id:
        raise ValueError("Stored layer column/root identity does not reproduce.")
    return rebuilt


def _rebuild_layer_curving_projection_trace(
    value: object, /
) -> LayerCurvingProjectionTrace:
    from phydrax.meshing._layer_curving_records import LayerCurvingProjectionTrace

    if (
        not isinstance(value, LayerCurvingProjectionTrace)
        or type(value) is not LayerCurvingProjectionTrace
    ):
        raise TypeError("Stored layer projection trace has an unexpected concrete type.")
    column_identity = _rebuild_layer_column_projection_identity(value.column_identity)
    trace = LayerCurvingProjectionTrace(
        value.source_id,
        value.source_revision,
        value.frame_id,
        value.geometry_node_rows,
        value.target_cell_global_ids,
        value.target_local_dofs,
        value.source_root_cell_ids,
        value.source_piece_ids,
        column_identity,
        value.requested_entity_ids,
        value.returned_entity_ids,
        value.requested_occurrence_paths,
        value.returned_occurrence_paths,
        value.queries,
        value.parameters,
        value.projected_points,
        value.distance_lower_bounds,
        value.residuals,
        value.status,
        value.returned_source_ids,
        value.returned_source_revisions,
        value.returned_frame_ids,
        value.failures,
        normals=value.normals,
        tangents=value.tangents,
        chart_witness_rows=value.chart_witness_rows,
        target_reference_points=value.target_reference_points,
    )
    if trace.trace_id != value.trace_id:
        raise ValueError("Stored layer projection trace identity changed.")
    return trace


def _rebuild_implicit_root_chart_evidence(value: object, /) -> ImplicitRootChartEvidence:
    from phydrax.geometry.implicit._source_charts import ImplicitRootChartEvidence

    field_names = {
        "row",
        "target_cell_global_id",
        "target_entity_global_id",
        "target_local_entity_index",
        "source_cell_key",
        "root_edge_keys",
        "status",
        "field_residual_bound",
        "target_to_root_lower_bound",
        "target_to_root_bound",
        "source_normal_deviation_bounds",
        "chart_jacobian_lower",
        "chart_injectivity_margin",
        "source_gradient_lower",
        "detail",
    }
    if type(value) is not dict:
        raise TypeError(
            "Stored implicit root-chart evidence must be a primitive mapping."
        )
    fields: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Implicit root-chart evidence field names must be strings.")
        fields[key] = item
    if set(fields) != field_names:
        raise TypeError(
            "Stored implicit root-chart evidence has an invalid primitive projection."
        )

    def integer(name: str, /) -> int:
        result = fields[name]
        if type(result) is not int:
            raise TypeError(f"Implicit root-chart {name} must be an integer.")
        return result

    def real(name: str, /) -> float:
        result = fields[name]
        if type(result) is not float:
            raise TypeError(f"Implicit root-chart {name} must be a float.")
        return result

    def text(name: str, /) -> str:
        result = fields[name]
        if not isinstance(result, str):
            raise TypeError(f"Implicit root-chart {name} must be text.")
        return result

    def corner(raw: object, /) -> tuple[int, int, int]:
        if type(raw) is not tuple or len(raw) != 3:
            raise TypeError("Implicit root-edge corners must be integer triples.")
        x, y, z = raw
        if type(x) is not int or type(y) is not int or type(z) is not int:
            raise TypeError("Implicit root-edge corners must contain exact integers.")
        return x, y, z

    def edge(raw: object, /) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        if type(raw) is not tuple or len(raw) != 2:
            raise TypeError("Implicit root edges must retain two endpoint corners.")
        first, second = raw
        return corner(first), corner(second)

    source_key_value = fields["source_cell_key"]
    if type(source_key_value) is not tuple or len(source_key_value) != 4:
        raise TypeError("Implicit source cell keys must be integer quadruples.")
    key_a, key_b, key_c, key_d = source_key_value
    if (
        type(key_a) is not int
        or type(key_b) is not int
        or type(key_c) is not int
        or type(key_d) is not int
    ):
        raise TypeError("Implicit source cell keys must contain exact integers.")
    source_cell_key: SourceCellKey = (key_a, key_b, key_c, key_d)

    edge_values = fields["root_edge_keys"]
    if type(edge_values) is not tuple or len(edge_values) != 3:
        raise TypeError("Implicit root-edge triangles must retain three edges.")
    edge_a, edge_b, edge_c = edge_values
    root_edge_keys: RootEdgeTriangle = (edge(edge_a), edge(edge_b), edge(edge_c))

    interval_values = fields["source_normal_deviation_bounds"]
    if type(interval_values) is not tuple or len(interval_values) != 2:
        raise TypeError("Implicit source-normal deviation bounds must be an interval.")
    interval_lower, interval_upper = interval_values
    if type(interval_lower) is not float or type(interval_upper) is not float:
        raise TypeError("Implicit source-normal deviation endpoints must be floats.")
    normal_deviation: Interval = (interval_lower, interval_upper)
    return ImplicitRootChartEvidence(
        row=integer("row"),
        target_cell_global_id=integer("target_cell_global_id"),
        target_entity_global_id=integer("target_entity_global_id"),
        target_local_entity_index=integer("target_local_entity_index"),
        source_cell_key=source_cell_key,
        root_edge_keys=root_edge_keys,
        status=text("status"),
        field_residual_bound=real("field_residual_bound"),
        target_to_root_lower_bound=real("target_to_root_lower_bound"),
        target_to_root_bound=real("target_to_root_bound"),
        source_normal_deviation_bounds=normal_deviation,
        chart_jacobian_lower=real("chart_jacobian_lower"),
        chart_injectivity_margin=real("chart_injectivity_margin"),
        source_gradient_lower=real("source_gradient_lower"),
        detail=text("detail"),
    )


def _rebuild_layer_curving_implicit_trace(
    value: object, /
) -> LayerCurvingImplicitRootTrace:
    from phydrax.meshing._layer_curving_records import LayerCurvingImplicitRootTrace

    if (
        not isinstance(value, LayerCurvingImplicitRootTrace)
        or type(value) is not LayerCurvingImplicitRootTrace
    ):
        raise TypeError(
            "Stored implicit layer root trace has an unexpected concrete type."
        )
    chart_evidence = tuple(
        _rebuild_implicit_root_chart_evidence(item) for item in value.chart_evidence
    )
    column_identity = _rebuild_layer_column_projection_identity(value.column_identity)
    trace = LayerCurvingImplicitRootTrace(
        value.source_id,
        value.source_revision,
        value.frame_id,
        value.map_witness_id,
        value.root_chart_witness_id,
        value.geometry_node_rows,
        value.target_cell_global_ids,
        value.target_local_dofs,
        value.chart_rows,
        value.target_reference_points,
        value.target_points,
        value.source_points,
        value.displacement_upper,
        value.root_error,
        value.source_cell_keys,
        value.root_edge_keys,
        value.statuses,
        chart_evidence,
        column_identity,
    )
    if trace.trace_id != value.trace_id:
        raise ValueError("Stored implicit layer root trace identity changed.")
    return trace


def _rebuild_layer_curving_mapped_trace(value: object, /) -> LayerCurvingMappedChartTrace:
    from phydrax.meshing._layer_curving_records import LayerCurvingMappedChartTrace

    if (
        not isinstance(value, LayerCurvingMappedChartTrace)
        or type(value) is not LayerCurvingMappedChartTrace
    ):
        raise TypeError("Stored mapped layer trace has an unexpected concrete type.")
    column_identity = _rebuild_layer_column_projection_identity(value.column_identity)
    trace = LayerCurvingMappedChartTrace(
        value.source_id,
        value.source_revision,
        value.frame_id,
        value.query_id,
        value.geometry_node_rows,
        value.target_cell_global_ids,
        value.target_local_dofs,
        value.chart_witness_rows,
        value.target_reference_points,
        value.root_reference_points,
        value.source_parameters,
        value.source_chart_ids,
        value.source_entity_ids,
        value.occurrence_paths,
        value.root_face_global_ids,
        value.root_cell_global_ids,
        value.root_local_face_indices,
        value.target_points,
        value.source_points,
        value.distance_lower_bounds,
        value.distance_upper_bounds,
        value.normals,
        value.tangents,
        column_identity,
    )
    if trace.trace_id != value.trace_id:
        raise ValueError("Stored mapped layer trace identity changed.")
    return trace


def _rebuild_layer_curving_source_evidence(
    value: object,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    require_current: bool = True,
) -> LayerCurvingSourceEvidence:
    from phydrax.discretization._nested_reference_types import ReferenceCellMapWitnesses
    from phydrax.geometry._source_boundary_records import (
        SourceBoundaryProjectionQuery,
        SourceBoundaryQuery,
    )
    from phydrax.meshing._layer_curving_records import LayerCurvingSourceEvidence

    if (
        not isinstance(value, LayerCurvingSourceEvidence)
        or type(value) is not LayerCurvingSourceEvidence
    ):
        raise TypeError("Stored layer source evidence has an unexpected concrete type.")
    source_query = _rebuild_continuation_value(value.source_query)
    if not isinstance(source_query, SourceBoundaryQuery):
        raise TypeError("Layer source evidence lost its source-bound query.")
    if value.projection_trace is not None and not isinstance(
        source_query, SourceBoundaryProjectionQuery
    ):
        raise TypeError("Projection trace requires a projection-capable source query.")
    charts = None
    if value.source_surface_charts is not None:
        stored_charts = _rebuild_continuation_value(value.source_surface_charts)
        if not isinstance(stored_charts, SurfaceSourceCharts) or not isinstance(
            source_query, MeshingDomainBoundarySource
        ):
            raise TypeError(
                "Layer source evidence lost its reconstructed SurfaceSourceCharts authority."
            )
        charts = SurfaceSourceCharts(
            stored_charts.domain,
            stored_charts.mesh,
            stored_charts.geometry,
            stored_charts.coordinate_contract,
            stored_charts.root_parameters,
            stored_charts.root_patches,
            stored_charts.associations,
            source_query,
            source_root_cell_ids=stored_charts.source_root_cell_ids,
            source_piece_ids=stored_charts.source_piece_ids,
            root_parameter_exact=stored_charts.root_parameter_exact,
        )
        if charts.source_chart_id != stored_charts.source_chart_id:
            raise ValueError(
                "Layer source query differs from its retained surface-chart authority."
            )
    projection_trace = (
        None
        if value.projection_trace is None
        else _rebuild_layer_curving_projection_trace(value.projection_trace)
    )
    implicit_root_trace = (
        None
        if value.implicit_root_trace is None
        else _rebuild_layer_curving_implicit_trace(value.implicit_root_trace)
    )
    mapped_chart_trace = (
        None
        if value.mapped_chart_trace is None
        else _rebuild_layer_curving_mapped_trace(value.mapped_chart_trace)
    )
    mapped_reference_maps = tuple(
        _rebuild_reference_cell_map_witnesses(item)
        for item in value.mapped_reference_maps
    )
    if not all(type(item) is ReferenceCellMapWitnesses for item in mapped_reference_maps):
        raise TypeError(
            "Layer source evidence lost its exact mapped reference-map banks."
        )
    implicit_chart_evidence = tuple(
        _rebuild_implicit_root_chart_evidence(item)
        for item in value.implicit_chart_evidence
    )
    chart_witness = _rebuild_continuation_value(value.chart_witness)
    if not isinstance(chart_witness, SourceBoundaryChartWitness):
        raise TypeError(
            "Layer source evidence lost its reconstructed exact chart witness."
        )
    _validate_implicit_source_chart_witness(source_query, chart_witness)
    fidelity = certify_source_fidelity(
        mesh,
        geometry,
        source_query,
        tolerance=value.fidelity.tolerance,
        maximum_normal_deviation=value.fidelity.maximum_normal_deviation,
        witness=chart_witness,
    )
    if fidelity.certificate_id != value.fidelity.certificate_id:
        raise ValueError("Stored layer source-fidelity certificate does not reproduce.")
    layer_evidence = _rebuild_boundary_layer_evidence(value.layer_evidence)
    chart_column_identity = _rebuild_layer_column_projection_identity(
        value.chart_column_identity
    )
    implicit_root_chart_witness = (
        None
        if value.implicit_root_chart_witness is None
        else _rebuild_continuation_value(value.implicit_root_chart_witness)
    )
    if implicit_root_chart_witness is not None and not isinstance(
        implicit_root_chart_witness, SourceBoundaryChartWitness
    ):
        raise TypeError("Layer source evidence lost its implicit root-chart witness.")
    implicit_root_cell_vertex_global_ids = value.implicit_root_cell_vertex_global_ids
    evidence = LayerCurvingSourceEvidence(
        charts,
        source_query,
        value.layer_result_id,
        layer_evidence,
        chart_witness,
        fidelity,
        value.minimum_scaled_jacobian,
        value.maximum_distortion,
        chart_column_identity=chart_column_identity,
        projection_trace=projection_trace,
        implicit_root_trace=implicit_root_trace,
        mapped_chart_trace=mapped_chart_trace,
        mapped_reference_maps=mapped_reference_maps,
        frame_id=value.frame_id,
        implicit_map_witness_id=value.implicit_map_witness_id,
        implicit_chart_evidence=implicit_chart_evidence,
        implicit_root_chart_witness=implicit_root_chart_witness,
        implicit_root_cell_vertex_global_ids=implicit_root_cell_vertex_global_ids,
    )
    if evidence.evidence_id != value.evidence_id:
        raise ValueError("Stored layer source evidence identity changed.")
    if require_current:
        evidence.revalidate(mesh, geometry)
    return evidence


def _rebuild_entity_lineage(value: object, /) -> EntityLineage:
    from phydrax.meshing._lineage import EntityLineage

    if type(value) is not EntityLineage:
        raise TypeError("Stored entity lineage has an unexpected concrete type.")
    rebuilt = EntityLineage(
        value.dimension,
        value.source_entity_set_id,
        value.target_entity_set_id,
        value.source_global_ids,
        value.target_global_ids,
        value.relation_kinds,
        created_target_ids=value.created_target_ids,
        deleted_source_ids=value.deleted_source_ids,
    )
    if rebuilt.lineage_id != value.lineage_id:
        raise ValueError("Stored entity lineage identity changed.")
    return rebuilt


def _rebuild_mesh_lineage(value: object, /) -> MeshLineage:
    from phydrax.meshing._lineage import MeshLineage

    if type(value) is not MeshLineage:
        raise TypeError("Stored mesh lineage has an unexpected concrete type.")
    entities = tuple(_rebuild_entity_lineage(item) for item in value.entities)
    lineage = MeshLineage(value.source_topology_id, value.target_topology_id, entities)
    if lineage.lineage_id != value.lineage_id:
        raise ValueError("Stored mesh lineage identity changed.")
    return lineage


def _rebuild_vertex_interpolation_stencil(value: object, /) -> VertexInterpolationStencil:
    from phydrax.meshing._lineage import VertexInterpolationStencil

    if type(value) is not VertexInterpolationStencil:
        raise TypeError(
            "Stored vertex interpolation stencil has an unexpected concrete type."
        )
    stencil = VertexInterpolationStencil(
        value.source_entity_set_id,
        value.target_entity_set_id,
        value.target_global_ids,
        value.source_global_ids,
        value.weights,
        value.valid,
        preserves_constants=value.preserves_constants,
        exact_weights=value.exact_weights,
    )
    if stencil.stencil_id != value.stencil_id:
        raise ValueError("Stored vertex interpolation stencil identity changed.")
    return stencil


def _rebuild_cell_mesh_transition(value: object, /) -> CellMeshTransition:
    from phydrax.meshing._lineage import CellMeshTransition, MeshTransitionKind

    if type(value) is not CellMeshTransition:
        raise TypeError("Stored cell mesh transition has an unexpected concrete type.")
    target = _rebuild_as(value.target, CellMeshingResult)
    lineage = _rebuild_mesh_lineage(value.lineage)
    vertex_stencil = (
        None
        if value.vertex_stencil is None
        else _rebuild_vertex_interpolation_stencil(value.vertex_stencil)
    )
    geometry_transition = None
    if value.geometry_transition is not None:
        source_mesh = _rebuild_as(value.geometry_transition.source_mesh, CellMesh)
        source_geometry = _rebuild_geometry_for_mesh(
            value.geometry_transition.source_geometry, source_mesh
        )
        if not isinstance(source_mesh, CellMesh) or not isinstance(
            source_geometry, CellGeometrySpec
        ):
            raise TypeError(
                "Cell mesh transition lost its geometry-transition source owners."
            )
        geometry_transition = _rebuild_cell_geometry_transition(
            value.geometry_transition,
            source_mesh,
            source_geometry,
            target.mesh,
            target.geometry,
        )
    volume_material_chain = (
        None
        if value.volume_material_chain is None
        else _rebuild_as(value.volume_material_chain, VolumeMaterialChain)
    )
    material_reference = (
        None
        if value.material_reference is None
        else _rebuild_as(value.material_reference, MaterialReferenceDeclaration)
    )
    if material_reference is not None and not isinstance(
        material_reference, MaterialReferenceDeclaration
    ):
        raise TypeError("Cell mesh transition lost its material reference declaration.")
    transition_kind = MeshTransitionKind(value.transition_kind.value)
    transition = CellMeshTransition(
        value.source_mesh_id,
        value.source_topology_id,
        target,
        lineage,
        transition_kind,
        vertex_stencil=vertex_stencil,
        geometry_transition=geometry_transition,
        volume_material_chain=volume_material_chain,
        material_reference=material_reference,
    )
    if transition.transition_id != value.transition_id:
        raise ValueError(
            "Stored cell mesh transition does not reproduce from its owner fields."
        )
    return transition


def _rebuild_piecewise_boundary_query_recipe(
    value: object, owner: object, /
) -> PiecewiseLinearBoundarySource:
    if type(value) is not PiecewiseLinearBoundarySourceArchiveRecipe:
        raise TypeError("Stored PWL boundary evidence has an unexpected concrete type.")
    if not isinstance(owner, SurfaceEnvelopeSource):
        raise TypeError("PWL boundary query recipe lost its SurfaceEnvelopeSource owner.")
    if (
        owner.source_id != value.owner.source_id
        or owner.source_revision != value.owner.expected_source_revision
        or owner.envelope_id != value.owner.expected_envelope_id
    ):
        raise ValueError("PWL query recipe changed its retained envelope owner.")
    entity_ids = np.asarray(value.entity_ids)
    query = PiecewiseLinearBoundarySource(
        owner,
        value.entity_dimension,
        entity_ids=entity_ids,
    )
    if (
        query.entity_dimension != value.entity_dimension
        or query.source_query_id != value.expected_source_query_id
        or not np.array_equal(query.entity_ids, entity_ids)
    ):
        raise ValueError(
            "PWL boundary query ancestry differs from its reconstructed owner."
        )
    return query


def _rebuild_native_plc_source_archive_recipe(
    value: NativePlcSourceArchiveRecipe,
    /,
    *,
    mapped_domain_override: MappedReferenceDomain | None = None,
) -> NativePlcSource:
    from phydrax.meshing.providers._native_sources import NativePlcSource

    if type(value) is not NativePlcSourceArchiveRecipe:
        raise TypeError(
            "Native PLC source reconstruction requires its exact archive recipe."
        )
    complex_ = _rebuild_as(value.complex, PiecewiseLinearComplex)
    stored_domain = (
        None
        if value.mapped_domain is None
        else _rebuild_as(value.mapped_domain, MappedReferenceDomain)
    )
    if mapped_domain_override is None:
        mapped_domain = stored_domain
    else:
        if (
            stored_domain is None
            or stored_domain.domain_id != mapped_domain_override.domain_id
            or stored_domain.coordinate_contract.spatial_id
            != mapped_domain_override.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Native PLC reference differs from its mapped-domain authority."
            )
        mapped_domain = mapped_domain_override
    fidelity_source = (
        None
        if value.fidelity_source is None
        else _rebuild_as(value.fidelity_source, SourceBoundaryQuery)
    )
    plc_vertex_root_global_ids = (
        None
        if value.plc_vertex_root_global_ids is None
        else np.asarray(value.plc_vertex_root_global_ids, dtype=np.int64)
    )
    source = NativePlcSource(
        complex_,
        value.source_id,
        value.source_revision,
        mapped_domain=mapped_domain,
        fidelity_source=fidelity_source,
        maximum_deviation=value.maximum_deviation,
        plc_vertex_root_global_ids=plc_vertex_root_global_ids,
    )
    if source.binding_id != value.expected_binding_id:
        raise ValueError("Native PLC source binding differs from its exact owner recipe.")
    return source


def _finite_element_transfer_metadata(
    value: FiniteElementTransferResult, /
) -> dict[str, object]:
    from phydrax._model._structure import model_structure_recipe
    from phydrax.discretization.fem._de_rham import FiniteElementTransferResult
    from phydrax.linalg._operator_pairing import OperatorPairing
    from phydrax.linalg._pairings import DiagonalPairing, EuclideanPairing
    from phydrax.linalg._spaces import ArraySpace
    from phydrax.sparse._linear import SparseCoordinateOperator
    from phydrax.sparse._relation import EdgeRelation

    if type(value) is not FiniteElementTransferResult:
        raise TypeError(
            "FE transfer archive metadata requires the canonical result type."
        )

    def pairing_payload(space: AbstractVectorSpace, /) -> dict[str, object]:
        if not isinstance(space, ArraySpace):
            raise TypeError(
                "FE archive fixture requires canonical array coefficient spaces."
            )
        pairing = space.pairing
        if type(pairing) is DiagonalPairing:
            return {
                "kind": "diagonal",
                "pairing_id": pairing.pairing_id,
                "weights": np.asarray(pairing.weights).tolist(),
            }
        if type(pairing) is EuclideanPairing:
            return {"kind": "euclidean", "pairing_id": pairing.pairing_id}
        if type(pairing) is OperatorPairing:
            prepared = pairing.prepared_inverse
            if pairing.inverse_action is not None or prepared is None:
                raise TypeError(
                    "FE archive Riesz pairing requires a re-preparable solver."
                )
            return {
                "kind": "operator",
                "pairing_id": pairing.pairing_id,
                "operator_id": pairing.operator.operator_id,
                "solve_policy_id": canonical_fingerprint(
                    model_structure_recipe(prepared.template.plan.policy)
                ),
            }
        raise TypeError("FE archive fixture has an unsupported coefficient pairing.")

    operators = []
    for index, operator in enumerate(value.complex_map.maps):
        if not isinstance(operator, SparseCoordinateOperator):
            raise TypeError("FE transfer archive lost a sparse coordinate map.")
        relation = operator.relation
        if not isinstance(relation, EdgeRelation):
            raise TypeError("FE archive fixture requires edge-relation sparse maps.")
        source_probe = np.linspace(0.25, 1.25, operator.source.size, dtype=np.float64)
        target_probe = np.linspace(-0.75, 0.5, operator.target.size, dtype=np.float64)
        operators.append(
            {
                "degree": value.complex_map.first_degree + index,
                "operator_id": operator.operator_id,
                "source_space_id": operator.source.space_id,
                "target_space_id": operator.target.space_id,
                "relation_source_indices": np.asarray(relation.source_indices).tolist(),
                "relation_target_indices": np.asarray(relation.target_indices).tolist(),
                "relation_valid": np.asarray(relation.valid).tolist(),
                "coefficients": np.asarray(operator.coefficients).tolist(),
                "source_pairing": pairing_payload(operator.source),
                "target_pairing": pairing_payload(operator.target),
                "source_probe": source_probe.tolist(),
                "mapped_probe": np.asarray(
                    jax.jit(lambda sparse_operator, values: sparse_operator.mv(values))(
                        operator, jnp.asarray(source_probe)
                    )
                ).tolist(),
                "target_probe": target_probe.tolist(),
                "transposed_probe": np.asarray(
                    jax.jit(
                        lambda sparse_operator, values: sparse_operator.transpose_mv(
                            values
                        )
                    )(operator, jnp.asarray(target_probe))
                ).tolist(),
            }
        )

    def entry_payload(item: SparseTransferEntryEvidence, /) -> dict[str, object]:
        return {
            "degree": item.degree,
            "source_size": item.source_size,
            "target_size": item.target_size,
            "target_indices": np.asarray(item.target_indices).tolist(),
            "source_indices": np.asarray(item.source_indices).tolist(),
            "absolute_error_bounds": np.asarray(item.absolute_error_bounds).tolist(),
            "exact_integral_counts": np.asarray(item.exact_integral_counts).tolist(),
            "enclosed_integral_counts": np.asarray(
                item.enclosed_integral_counts
            ).tolist(),
        }

    def residual_payload(item: SparseCommutingEvidence, /) -> dict[str, object]:
        return {
            "kind": item.kind,
            "degree": item.degree,
            "source_size": item.source_size,
            "target_size": item.target_size,
            "target_indices": np.asarray(item.target_indices).tolist(),
            "source_indices": np.asarray(item.source_indices).tolist(),
            "observed_defects": np.asarray(item.observed_defects).tolist(),
            "integration_error_bounds": np.asarray(
                item.integration_error_bounds
            ).tolist(),
            "roundoff_error_bounds": np.asarray(item.roundoff_error_bounds).tolist(),
        }

    return {
        "transfer_id": value.transfer_id,
        "evidence_id": value.evidence_id,
        "map_id": value.complex_map.map_id,
        "source_complex_id": value.complex_map.source.complex_id,
        "target_complex_id": value.complex_map.target.complex_id,
        "first_degree": value.complex_map.first_degree,
        "degree_offset": value.complex_map.degree_offset,
        "integration_policy_id": (
            None
            if value.integration_policy is None
            else value.integration_policy.policy_id
        ),
        "exact_integral_count": value.exact_integral_count,
        "enclosed_integral_count": value.enclosed_integral_count,
        "operators": operators,
        "entry_evidence": [entry_payload(item) for item in value.entry_evidence],
        "commuting_evidence": [
            residual_payload(item) for item in value.commuting_evidence
        ],
        "constraint_evidence": [
            residual_payload(item) for item in value.constraint_evidence
        ],
    }


def _write_finite_element_transfer_archive(directory: Path, /) -> None:
    from tests._support.rational_form_transfer import make_rational_form_transfer_case

    result = make_rational_form_transfer_case()
    write_model_artifact(directory / "finite-element-transfer.phx", result, limits=None)
    print(json.dumps(_finite_element_transfer_metadata(result)))


def _read_finite_element_transfer_archive(
    path: Path, expected: dict[str, object], /
) -> None:
    from phydrax.discretization.fem._de_rham import FiniteElementTransferResult
    from phydrax.meshing._model_codec import restore_model_archive_value

    result = read_model_artifact(
        path,
        expected_type=FiniteElementTransferResult,
        restore=restore_model_archive_value,
        limits=None,
    )
    if not isinstance(result, FiniteElementTransferResult):
        raise TypeError("FE transfer cold archive lost its typed result.")
    if _finite_element_transfer_metadata(result) != expected:
        raise ValueError(
            "FE transfer cold restore differs from its exact writer payload."
        )
    print("finite-element-transfer-cold-restored")


def _vertex_interpolation_stencil_archive_fixture() -> VertexInterpolationStencil:
    return VertexInterpolationStencil(
        "archive-source-vertices",
        "archive-target-vertices",
        np.asarray((101,), dtype=np.int64),
        np.asarray(((11, 29),), dtype=np.int64),
        np.asarray(((1.0 / 3.0, 2.0 / 3.0),), dtype=np.float64),
        np.asarray(((True, True),), dtype=np.bool_),
        exact_weights=((Fraction(1, 3), Fraction(2, 3)),),
        preserves_constants=True,
    )


def _vertex_interpolation_stencil_metadata(
    value: VertexInterpolationStencil, /
) -> dict[str, object]:
    return {
        "stencil_id": value.stencil_id,
        "source_entity_set_id": value.source_entity_set_id,
        "target_entity_set_id": value.target_entity_set_id,
        "target_global_ids": np.asarray(value.target_global_ids, dtype=np.int64).tolist(),
        "source_global_ids": np.asarray(value.source_global_ids, dtype=np.int64).tolist(),
        "weights": np.asarray(value.weights, dtype=np.float64).tolist(),
        "valid": np.asarray(value.valid, dtype=np.bool_).tolist(),
        "exact_weights": [
            [[weight.numerator, weight.denominator] for weight in row]
            for row in value.exact_weights
        ],
        "preserves_constants": value.preserves_constants,
    }


def _write_vertex_interpolation_stencil_archive(directory: Path, /) -> None:
    stencil = _vertex_interpolation_stencil_archive_fixture()
    write_model_artifact(
        directory / "vertex-interpolation-stencil.phx", stencil, limits=None
    )
    print(json.dumps(_vertex_interpolation_stencil_metadata(stencil), sort_keys=True))


def _read_vertex_interpolation_stencil_archive(
    path: Path, expected: dict[str, object], /
) -> None:
    stencil = read_model_artifact(
        path,
        expected_type=VertexInterpolationStencil,
        restore=_rebuild_vertex_interpolation_stencil,
        limits=None,
    )
    if type(stencil) is not VertexInterpolationStencil:
        raise TypeError("Vertex-stencil archive lost its exact owner.")
    if _vertex_interpolation_stencil_metadata(stencil) != expected:
        raise ValueError("Vertex-stencil archive changed its exact source coefficients.")
    print("vertex-interpolation-stencil-cold-restored")


def _periodic_distributed_finite_element_transfer_case() -> tuple[
    CellMesh,
    tuple[str, int, str, int, str, int],
    PeriodicFiniteElementTransfer,
    DistributedFiniteElementTransfer,
]:
    from phydrax.discretization.fem._de_rham import FiniteElementDeRhamComplex
    from tests.unit.discretization.test_periodic_fem_transfer import (
        _periodic_tetrahedral_cover,
    )

    settings = ("trimmed", 1, "trimmed", 2, "relative", 2)
    mesh = _periodic_tetrahedral_cover((True, False, False))
    source = FiniteElementDeRhamComplex(mesh, family=settings[0], order=settings[1])
    target = FiniteElementDeRhamComplex(mesh, family=settings[2], order=settings[3])
    periodic = source.transfer_periodic_quotient(target, boundary=settings[4])
    source_owners = tuple(
        np.arange(space.size, dtype=np.int32) % settings[5]
        for space in periodic.complex_map.source.spaces
    )
    target_owners = tuple(
        np.arange(space.size, dtype=np.int32) % settings[5]
        for space in periodic.complex_map.target.spaces
    )
    distributed = periodic.distributed(source_owners, target_owners, settings[5])
    return mesh, settings, periodic, distributed


def _periodic_distributed_transfer_composition(
    mesh: CellMesh,
    settings: tuple[str, int, str, int, str, int],
    periodic: PeriodicFiniteElementTransfer,
    distributed: DistributedFiniteElementTransfer,
    /,
) -> Composition:
    settings_id = canonical_fingerprint(
        {"kind": "periodic-finite-element-transfer-settings", "settings": settings}
    )
    mesh_entry = CompositionEntry(
        mesh,
        entry_id="mesh/periodic-topology",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=mesh.topology_id,
        revision_id=mesh.mesh_id,
        semantics_id="periodic-finite-element-cover-mesh",
    )
    settings_entry = CompositionEntry(
        settings,
        entry_id="transfer/settings",
        role="model-parameter",
        owner_id="phydrax.discretization.fem",
        structure_id=settings_id,
        revision_id=settings_id,
        semantics_id="periodic-finite-element-transfer-settings",
        dependencies=(
            mesh_entry.binding("structure"),
            mesh_entry.binding("revision"),
        ),
    )
    periodic_entry = CompositionEntry(
        periodic,
        entry_id="transfer/periodic",
        role="transfer",
        owner_id="phydrax.discretization.fem",
        structure_id=periodic.complex_map.map_id,
        revision_id=periodic.transfer_id,
        semantics_id="periodic-finite-element-quotient-transfer",
        dependencies=(
            mesh_entry.binding("structure"),
            mesh_entry.binding("revision"),
            settings_entry.binding("semantics"),
        ),
    )
    distributed_entry = CompositionEntry(
        distributed,
        entry_id="transfer/distributed",
        role="transfer",
        owner_id="phydrax.discretization.fem",
        structure_id=distributed.complex_map.map_id,
        revision_id=distributed.transfer_id,
        semantics_id="distributed-finite-element-quotient-transfer",
        dependencies=(
            periodic_entry.binding("revision"),
            settings_entry.binding("semantics"),
        ),
    )
    return Composition(
        (mesh_entry, settings_entry, periodic_entry, distributed_entry),
        boundary_id="periodic-distributed-fe-transfer:archive",
    )


def _periodic_distributed_transfer_metadata(
    periodic: PeriodicFiniteElementTransfer,
    distributed: DistributedFiniteElementTransfer,
    /,
) -> dict[str, object]:
    if (
        distributed.transfer_result.evidence_id
        != periodic.quotient_transfer_result.evidence_id
    ):
        raise ValueError(
            "Distributed FE transfer lost the periodic quotient evidence graph."
        )
    degree_payloads = []
    for degree in range(periodic.complex_map.source.top_degree + 1):
        source_space = periodic.complex_map.source.space(degree)
        target_space = periodic.complex_map.target.space(degree)
        source_probe = np.linspace(0.25, 1.25, source_space.size, dtype=np.float64)
        target_probe = np.linspace(-0.75, 0.5, target_space.size, dtype=np.float64)
        quotient_map = periodic.complex_map.map(degree)
        compiled_forward = jax.jit(lambda values: quotient_map.mv(values))(
            jnp.asarray(source_probe)
        )
        compiled_transpose = jax.jit(lambda values: quotient_map.transpose_mv(values))(
            jnp.asarray(target_probe)
        )
        distributed_map = distributed.map(degree)
        local_forward = distributed_map.serial_reference(source_probe)
        local_transpose = distributed_map.serial_transpose_reference(target_probe)
        np.testing.assert_allclose(
            np.asarray(compiled_forward),
            np.asarray(local_forward),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(compiled_transpose),
            np.asarray(local_transpose),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        lifted = periodic.lift_target(degree, compiled_forward)
        round_trip = periodic.collapse_target(degree, lifted)
        np.testing.assert_allclose(
            np.asarray(round_trip),
            np.asarray(compiled_forward),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        degree_payloads.append(
            {
                "degree": degree,
                "quotient_map_id": quotient_map.operator_id,
                "distributed_map_id": distributed_map.transfer_id,
                "source_owner": np.asarray(distributed_map.source_owner).tolist(),
                "target_owner": np.asarray(distributed_map.target_owner).tolist(),
                "forward": np.asarray(compiled_forward).tolist(),
                "transpose": np.asarray(compiled_transpose).tolist(),
                "lift_round_trip": np.asarray(round_trip).tolist(),
            }
        )
    return {
        "periodic_transfer_id": periodic.transfer_id,
        "periodic_map_id": periodic.complex_map.map_id,
        "lifted_transfer_id": periodic.lifted_transfer_result.transfer_id,
        "lifted_evidence_id": periodic.lifted_transfer_result.evidence_id,
        "quotient_transfer_id": periodic.quotient_transfer_result.transfer_id,
        "quotient_evidence_id": periodic.quotient_transfer_result.evidence_id,
        "boundary": periodic.boundary,
        "source_orbit_map_ids": [list(row) for row in periodic.source_orbit_map_ids],
        "target_orbit_map_ids": [list(row) for row in periodic.target_orbit_map_ids],
        "source_active_indices": [list(row) for row in periodic.source_active_indices],
        "target_active_indices": [list(row) for row in periodic.target_active_indices],
        "source_closure_defects": np.asarray(periodic.source_closure_defects).tolist(),
        "target_closure_defects": np.asarray(periodic.target_closure_defects).tolist(),
        "source_nilpotency_defects": np.asarray(
            periodic.source_nilpotency_defects
        ).tolist(),
        "target_nilpotency_defects": np.asarray(
            periodic.target_nilpotency_defects
        ).tolist(),
        "deck_descent_defects": np.asarray(periodic.deck_descent_defects).tolist(),
        "deck_descent_error_bounds": np.asarray(
            periodic.deck_descent_error_bounds
        ).tolist(),
        "commuting_defects": np.asarray(periodic.commuting_defects).tolist(),
        "commuting_error_bounds": np.asarray(periodic.commuting_error_bounds).tolist(),
        "distributed_transfer_id": distributed.transfer_id,
        "distributed_degree_map_ids": [
            item.transfer_id for item in distributed.degree_maps
        ],
        "degrees": degree_payloads,
    }


def _write_periodic_distributed_fe_transfer_archive(directory: Path, /) -> None:
    mesh, settings, periodic, distributed = (
        _periodic_distributed_finite_element_transfer_case()
    )
    composition = _periodic_distributed_transfer_composition(
        mesh, settings, periodic, distributed
    )
    write_model_artifact(
        directory / "periodic-distributed-fe-transfer.phx",
        composition,
        limits=None,
    )
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "mesh_id": mesh.mesh_id,
                "settings": list(settings),
                **_periodic_distributed_transfer_metadata(periodic, distributed),
            }
        )
    )


def _read_periodic_distributed_fe_transfer_archive(
    path: Path, expected: dict[str, object], /
) -> None:
    from phydrax.meshing._model_codec import restore_model_archive_value

    composition = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=None,
    )
    mesh = composition.value("mesh/periodic-topology")
    settings = composition.value("transfer/settings")
    if not isinstance(mesh, CellMesh) or not isinstance(settings, tuple):
        raise TypeError("Cold periodic FE archive lost its mesh or settings owner.")
    periodic = composition.value("transfer/periodic")
    distributed = composition.value("transfer/distributed")
    if not isinstance(periodic, PeriodicFiniteElementTransfer) or not isinstance(
        distributed, DistributedFiniteElementTransfer
    ):
        raise TypeError("Cold periodic/distributed FE archive lost its typed owners.")
    observed = {
        "composition_id": composition.composition_id,
        "mesh_id": mesh.mesh_id,
        "settings": list(settings),
        **_periodic_distributed_transfer_metadata(periodic, distributed),
    }
    if observed != expected:
        raise ValueError(
            "Cold periodic/distributed FE transfer differs from its source evidence."
        )
    print("periodic-distributed-fe-transfer-cold-restored")


_Rebuilt = TypeVar("_Rebuilt")


def _rebuild_as(value: object, expected_type: type[_Rebuilt], /) -> _Rebuilt:
    rebuilt = _rebuild_continuation_value(value)
    if not isinstance(rebuilt, expected_type):
        raise TypeError(
            f"Stored value rebuilt as {type(rebuilt).__name__}, expected {expected_type.__name__}."
        )
    return rebuilt


def _rebuild_tuple_as(
    value: object, expected_type: type[_Rebuilt], /
) -> tuple[_Rebuilt, ...]:
    if not isinstance(value, tuple):
        raise TypeError("Stored sequence owner must retain its exact tuple structure.")
    return tuple(_rebuild_as(item, expected_type) for item in value)


def _exact_coordinate_bank_payload(bank: CoordinateSourceBank | None, /) -> object:
    if bank is None:
        return None
    return json.loads(
        json.dumps(
            [[coefficient_payload(value) for value in row] for row in bank],
            sort_keys=True,
        )
    )


def _rebuild_geometry_for_mesh(value: object, source_mesh: object, /) -> CellGeometrySpec:
    if not isinstance(source_mesh, CellMesh) or type(source_mesh) is not CellMesh:
        raise TypeError("CellGeometrySpec restoration requires its current CellMesh.")
    geometry = _rebuild_continuation_value(value, source_mesh=source_mesh)
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Stored source geometry did not rebuild as CellGeometrySpec.")
    return geometry


def _periodic_actual_geometry(
    stored_mesh: CellMesh, geometry: CellGeometrySpec, /
) -> CellGeometrySpec | None:
    topology = stored_mesh.periodic_topology
    if topology is None:
        return None
    if (
        topology._actual_geometry is not None
        or topology.coordinate_source_map_id is not None
    ):
        return geometry
    return None


def _rebuild_mesh_geometry_pair(
    stored_mesh: object, stored_geometry: object, /
) -> tuple[CellMesh, CellGeometrySpec]:
    if not isinstance(stored_mesh, CellMesh) or type(stored_mesh) is not CellMesh:
        raise TypeError("Stored geometry pair requires exact CellMesh.")
    geometry = _rebuild_geometry_for_mesh(stored_geometry, stored_mesh)
    mesh = CellMesh.from_stored_fields(
        stored_mesh,
        actual_geometry=_periodic_actual_geometry(stored_mesh, geometry),
    )
    renewed_geometry = _rebuild_geometry_for_mesh(stored_geometry, mesh)
    if (
        renewed_geometry.geometry_layout_id != geometry.geometry_layout_id
        or cell_geometry_id(renewed_geometry) != cell_geometry_id(geometry)
    ):
        raise ValueError(
            "Stored mesh/geometry source context changed during reattachment."
        )
    return mesh, renewed_geometry


def _validate_implicit_source_chart_witness(
    source_query: SourceBoundaryQuery,
    chart_witness: SourceBoundaryChartWitness,
    /,
) -> None:
    if not isinstance(source_query, AdaptiveImplicitBoundarySource):
        return
    from phydrax.geometry._source_identity import source_entity_id

    zero_set_entity_id = source_entity_id(
        source_query.source_id,
        source_query.source_revision,
        "implicit-surface",
        "zero-set",
    )
    if (
        chart_witness.source_id != source_query.source_id
        or chart_witness.source_revision != source_query.source_revision
        or any(
            entity_id != zero_set_entity_id
            for entity_id in chart_witness.source_entity_ids
        )
        or any(chart_witness.occurrence_paths)
    ):
        raise ValueError(
            "Implicit source chart witness lost its canonical zero-set entity key."
        )


def _rebuild_coverage_domain(
    value: object,
    /,
) -> (
    PiecewiseLinearDomain
    | MappedReferenceDomain
    | MeshingDomain
    | AdaptiveImplicitSurface
    | AnalyticImplicitProfile
):
    domain = _rebuild_continuation_value(value)
    if not isinstance(
        domain,
        (
            PiecewiseLinearDomain,
            MappedReferenceDomain,
            MeshingDomain,
            AdaptiveImplicitSurface,
            AnalyticImplicitProfile,
        ),
    ):
        raise TypeError("Stored certification domain has no registered source owner.")
    return domain


def _rebuild_source_fidelity_witness(value: object, /) -> SourceFidelityWitness:
    witness = _rebuild_continuation_value(value)
    if not isinstance(
        witness,
        (
            SurfaceChartWitness,
            SourceBoundaryChartWitness,
            SourceBoundaryPointWitness,
            SourceBoundaryDistanceWitness,
            SurfaceEnvelopeDistanceWitness,
        ),
    ):
        raise TypeError("Stored fidelity witness has no registered proof owner.")
    return witness


def _rebuild_scoped_fidelity_input(
    value: object, /
) -> tuple[SourceBoundaryQuery, SourceFidelityWitness, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("Scoped fidelity input must retain its exact three-field tuple.")
    query = _rebuild_as(value[0], SourceBoundaryQuery)
    witness = _rebuild_source_fidelity_witness(value[1])
    tolerance = value[2]
    if isinstance(tolerance, bool) or not isinstance(tolerance, float):
        raise TypeError("Scoped fidelity tolerance must remain a finite float.")
    return query, witness, tolerance


def _rebuild_restricted_coverage_witness(
    value: object, /
) -> RestrictedDelaunayBoundaryWitness | RestrictedDelaunayCoverageTransportWitness:
    witness = _rebuild_continuation_value(value)
    if not isinstance(
        witness,
        (RestrictedDelaunayBoundaryWitness, RestrictedDelaunayCoverageTransportWitness),
    ):
        raise TypeError("Stored source coverage witness has no registered proof owner.")
    return witness


def _rebuild_mesh_certification_premises(
    value: object, /
) -> tuple[MeshCertificationInputs, CellMeshAuditPolicy, str]:
    """Rebuild a stored report's premises; acceptance is recomputed once by the caller."""
    from phydrax.meshing._certification import MeshCertificationReport
    from phydrax.meshing._quality import CellQualityReport

    if type(value) is not MeshCertificationReport:
        raise TypeError("Stored mesh certification has an unexpected concrete type.")
    stored_quality = CellQualityReport.from_stored_fields(value.audit.quality)
    inputs = _rebuild_as(value.inputs, MeshCertificationInputs)
    if isinstance(inputs.source, AdaptiveImplicitBoundarySource) and isinstance(
        inputs.fidelity_witness, SourceBoundaryChartWitness
    ):
        _validate_implicit_source_chart_witness(inputs.source, inputs.fidelity_witness)
    audit_policy = _rebuild_as(value.audit_policy, CellMeshAuditPolicy)
    return inputs, audit_policy, stored_quality.report_id


def _meshing_source_accuracy(value: str, /) -> MeshingSourceAccuracy:
    if value == "exact_represented":
        return "exact_represented"
    if value == "bounded_approximation":
        return "bounded_approximation"
    if value == "certified_enclosure":
        return "certified_enclosure"
    if value == "sampled_interpretation":
        return "sampled_interpretation"
    if value == "repaired_envelope":
        return "repaired_envelope"
    raise ValueError(
        "Stored meshing-domain accuracy is not a registered source semantic."
    )


def _rebuild_patch_boundary_use(value: object, /) -> PatchBoundaryUse:
    if type(value) is PatchCurveUse:
        return PatchCurveUse(
            value.curve,
            _rebuild_as(value.pcurve, AbstractCurve),
            value.first,
            value.last,
            sense=value.sense,
        )
    if type(value) is PatchPoleUse:
        return PatchPoleUse(
            value.corner,
            _rebuild_as(value.pcurve, AbstractCurve),
            value.first,
            value.last,
            sense=value.sense,
        )
    raise TypeError("Stored patch boundary use has no registered concrete constructor.")


def _rebuild_scalar_coordinate_element(value: object, /) -> ScalarCoordinateElement:
    from phydrax.discretization._cell_geometry import ScalarCoordinateElement

    element = _rebuild_continuation_value(value)
    if not isinstance(element, ScalarCoordinateElement):
        raise TypeError(
            "Stored coordinate source does not satisfy the scalar element contract."
        )
    return element


def _array_archive_input(value: object, /) -> np.ndarray | jax.Array:
    if isinstance(value, (np.ndarray, jax.Array)):
        return value
    if isinstance(value, np.generic):
        return np.asarray(value)
    raise TypeError("Stored accepted FE state values must retain numeric array payloads.")


def _expected_object_mapping(value: object, /) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError("Worker expectation record must remain a JSON object.")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Worker expectation keys must remain strings.")
        result[key] = item
    return result


def _expected_sequence(value: object, /) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("Worker expectation sequence must remain a JSON array.")
    return tuple(value)


def _expected_array(value: object, /) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        items = tuple(_expected_array(item) for item in value)
        if not items:
            return np.asarray((), dtype=np.float64)
        try:
            return np.stack(items)
        except ValueError as error:
            raise ValueError("Worker numeric expectation must be rectangular.") from error
    if isinstance(value, (bool, int, float)):
        return np.asarray(value)
    raise TypeError("Worker numeric expectation must contain only JSON numbers.")


def _rebuild_cell_validity_certificate(
    value: object, mesh: CellMesh, geometry: CellGeometrySpec, /
) -> CellValidityCertificate:
    import dataclasses

    from phydrax._model._structure import _restore_dataclass

    if type(value) is not CellValidityCertificate:
        raise TypeError("Stored chart validity proof has an unexpected concrete type.")
    policy = _rebuild_as(value.policy, CellValidityPolicy)
    # The stored proof carries its coordinate map in archive form; renew it on the
    # same mesh, require the exact renewed operands, then recompute every proof.
    stored_mesh = None if value.mesh is None else _rebuild_as(value.mesh, CellMesh)
    stored_geometry = _rebuild_geometry_for_mesh(
        value.geometry, mesh if stored_mesh is None else stored_mesh
    )
    if not _same_tree_content(stored_geometry, geometry) or (
        stored_mesh is not None and not _same_tree_content(stored_mesh, mesh)
    ):
        raise ValueError("Stored chart validity proof names different operands.")
    fields = {
        field.name: object.__getattribute__(value, field.name)
        for field in dataclasses.fields(value)
    }
    fields.update(
        geometry=geometry,
        mesh=None if stored_mesh is None else mesh,
        policy=policy,
    )
    return CellValidityCertificate.from_stored_fields(
        _restore_dataclass(CellValidityCertificate, fields),
        None if stored_mesh is None else mesh,
        geometry,
        policy=policy,
    )


def _rebuild_embedding_certificate(
    value: object,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    /,
) -> GlobalEmbeddingCertificate:
    if type(value) is not GlobalEmbeddingCertificate:
        raise TypeError("Stored chart embedding proof has an unexpected concrete type.")
    periodic = mesh.periodic_topology
    if (value.periodic is None) != (periodic is None) or (
        periodic is not None
        and value.periodic is not None
        and periodic.periodic_topology_id != value.periodic.periodic_topology_id
    ):
        raise ValueError(
            "Stored chart embedding proof changes periodic endpoint identity."
        )
    certificate = certify_global_embedding(
        mesh,
        geometry,
        validity,
        junction_vertices=value.binding.junction_vertices,
        periodic=periodic,
    )
    if certificate.certificate_id != value.certificate_id:
        raise ValueError(
            "Stored chart embedding proof differs from endpoint recomputation."
        )
    return certificate


def _rebuild_source_fidelity_certificate(
    value: object,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    source_record: object,
    /,
    *,
    prepared_parent: object | None = None,
) -> SourceFidelityCertificate:
    if type(value) is not SourceFidelityCertificate:
        raise TypeError("Stored source fidelity proof has an unexpected concrete type.")
    from phydrax.geometry._surface_source_support import _same_tree_content

    if not _same_tree_content(value.source, source_record):
        raise ValueError(
            "Stored source fidelity query differs from its reconstructed root."
        )
    witness = (
        None if value.witness is None else _rebuild_source_fidelity_witness(value.witness)
    )
    from phydrax.geometry.implicit._chart_restriction import (
        ImplicitChartPremise,
        ValidatedImplicitChartParent,
    )

    chart_parent = None
    validated_parent = None
    if value.chart_premise is not None:
        retained = value.chart_premise
        if type(retained) is not ImplicitChartPremise:
            raise TypeError("Stored chart restriction lost its exact parent premise.")
        if prepared_parent is None:
            parent_mesh = _rebuild_as(retained.parent_mesh, CellMesh)
            parent_geometry = _rebuild_geometry_for_mesh(
                retained.parent_geometry, parent_mesh
            )
            parent_certificate = _rebuild_source_fidelity_certificate(
                retained.parent_certificate,
                parent_mesh,
                parent_geometry,
                source,
                source_record,
            )
        elif type(prepared_parent) is ValidatedImplicitChartParent:
            validated_parent = prepared_parent
            parent_mesh = prepared_parent.mesh
            parent_geometry = prepared_parent.geometry
            parent_certificate = prepared_parent.certificate
        else:
            raise TypeError("Prepared chart parent must be a validated host artifact.")
        transition = (
            None
            if retained.transition is None
            else _rebuild_cell_geometry_transition(
                retained.transition, parent_mesh, parent_geometry, mesh, geometry
            )
        )
        chart_parent = ImplicitChartPremise(
            parent_mesh, parent_geometry, parent_certificate, transition
        )
    certificate = certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=value.tolerance,
        maximum_normal_deviation=value.maximum_normal_deviation,
        witness=witness,
        chart_parent=chart_parent,
        validated_chart_parent=validated_parent,
    )
    if (
        certificate.certificate_id != value.certificate_id
        or not _same_tree_content(certificate.chart_rows, value.chart_rows)
        or not _same_tree_content(certificate.chart_restriction, value.chart_restriction)
    ):
        raise ValueError(
            "Stored source fidelity proof does not reproduce its exact query."
        )
    return certificate


def _write_implicit_chart_restriction(directory: Path, /) -> None:
    from phydrax.meshing._adaptation import (
        execute_mesh_adaptation,
        MarkedMeshAdaptation,
        MeshAdaptationPolicy,
        MeshAdaptationRoute,
        prepare_mesh_adaptation,
    )
    from phydrax.meshing._bisection import BisectionCompatibility
    from phydrax.meshing._implicit_association_transfer import ImplicitAssociationTransfer
    from phydrax.meshing.providers._native_sources import NativeImplicitSource
    from tests.unit.meshing.test_source_association_transfer import (
        _implicit_surface_result,
        _profile,
    )

    source = _implicit_surface_result(
        _profile("restriction-sphere"), relative_cell_diameter=0.4
    )
    assert source.certification is not None
    native_source = source.certification.inputs.feature_source
    assert isinstance(native_source, NativeImplicitSource)
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(np.asarray(source.mesh.entity_set(2).entity_ids)),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
                association_transfer=ImplicitAssociationTransfer(native_source),
            ),
        )
    )
    assert refined.target.certification is not None
    certificate = refined.target.certification.fidelity
    assert certificate is not None and certificate.chart_restriction is not None
    write_model_artifact(
        directory / "implicit-chart-restriction.phx",
        certificate,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    print(json.dumps({"certificate_id": certificate.certificate_id}))


def _read_implicit_chart_restriction(path: Path, expected: dict[str, str], /) -> None:
    from phydrax.geometry._surface_source_support import _same_tree_content
    from phydrax.geometry.implicit._chart_restriction import (
        ImplicitChartPremise,
        ValidatedImplicitChartParent,
    )

    fields = read_model_archive_fields(
        path,
        expected_type=SourceFidelityCertificate,
        fields=(
            "source",
            "chart_premise.parent_mesh",
            "chart_premise.parent_geometry",
            "chart_premise.parent_certificate",
        ),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    source_record = fields["source"]
    stored_parent_mesh = fields["chart_premise.parent_mesh"]
    stored_parent_geometry = fields["chart_premise.parent_geometry"]
    stored_parent_certificate = fields["chart_premise.parent_certificate"]
    del fields
    source = _rebuild_continuation_value(source_record)
    if not isinstance(source, SourceBoundaryQuery):
        raise TypeError("Cold restriction source lost its query contract.")
    parent_mesh = _rebuild_as(stored_parent_mesh, CellMesh)
    parent_geometry = _rebuild_geometry_for_mesh(stored_parent_geometry, parent_mesh)
    parent_certificate = _rebuild_source_fidelity_certificate(
        stored_parent_certificate, parent_mesh, parent_geometry, source, source_record
    )
    prepared_parent = ValidatedImplicitChartParent(
        parent_mesh, parent_geometry, source, parent_certificate
    )

    def rebuild(value: object) -> SourceFidelityCertificate:
        nonlocal \
            source_record, \
            stored_parent_mesh, \
            stored_parent_geometry, \
            stored_parent_certificate
        if type(value) is not SourceFidelityCertificate:
            raise TypeError("Cold restriction root must be SourceFidelityCertificate.")
        retained = value.chart_premise
        if (
            type(retained) is not ImplicitChartPremise
            or not _same_tree_content(value.source, source_record)
            or not _same_tree_content(retained.parent_mesh, stored_parent_mesh)
            or not _same_tree_content(retained.parent_geometry, stored_parent_geometry)
            or not _same_tree_content(
                retained.parent_certificate, stored_parent_certificate
            )
        ):
            raise ValueError("Archive parent premises changed between admitted reads.")
        # The second admitted archive has been compared field-for-field to the
        # prevalidated parent. Its raw prerequisite copies no longer serve replay.
        del (
            source_record,
            stored_parent_mesh,
            stored_parent_geometry,
            stored_parent_certificate,
        )
        mesh = _rebuild_as(value.binding.mesh, CellMesh)
        geometry = _rebuild_geometry_for_mesh(value.binding.geometry, mesh)
        return _rebuild_source_fidelity_certificate(
            value, mesh, geometry, source, value.source, prepared_parent=prepared_parent
        )

    certificate = read_model_artifact(
        path,
        expected_type=SourceFidelityCertificate,
        restore=rebuild,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert certificate.certificate_id == expected["certificate_id"]
    assert certificate.chart_restriction is not None
    print(json.dumps({"certificate_id": certificate.certificate_id}))


def _write_exact_tensor_quad_fidelity(directory: Path, /) -> None:
    from phydrax.meshing.providers import NativeMeshingProvider
    from tools.meshing_qualification.scenarios.native_family_matrix import (
        _CONTRACT,
        _quarter_surface_case,
    )

    case = _quarter_surface_case(4)
    result = (
        NativeMeshingProvider(case.options)
        .plan(case.source, case.specification, coordinate_contract=_CONTRACT)
        .execute()
    )
    if result.certification is None or result.certification.fidelity is None:
        raise ValueError("Exact tensor quad result has no fidelity certificate.")
    certificate = result.certification.fidelity
    write_model_artifact(
        directory / "exact-tensor-quad-fidelity.phx", certificate, limits=None
    )
    write_model_artifact(
        directory / "exact-tensor-quad-quality.phx",
        result.quality.evaluation,
        limits=None,
    )
    print(
        json.dumps(
            {
                "certificate_id": certificate.certificate_id,
                "quality_id": result.quality.report_id,
                "measure_exact": result.quality.evaluation.measure_exact,
            }
        )
    )


def _read_exact_tensor_quad_fidelity(path: Path, expected: dict[str, object], /) -> None:
    def rebuild(value: object) -> SourceFidelityCertificate:
        if type(value) is not SourceFidelityCertificate:
            raise TypeError("Cold exact tensor root must be SourceFidelityCertificate.")
        mesh = _rebuild_as(value.binding.mesh, CellMesh)
        geometry = _rebuild_geometry_for_mesh(value.binding.geometry, mesh)
        source = _rebuild_continuation_value(value.source)
        if not isinstance(source, SourceBoundaryQuery):
            raise TypeError("Cold exact tensor source lost its query contract.")
        return _rebuild_source_fidelity_certificate(
            value, mesh, geometry, source, value.source
        )

    certificate = read_model_artifact(
        path, expected_type=SourceFidelityCertificate, restore=rebuild, limits=None
    )
    if certificate.certificate_id != expected["certificate_id"]:
        raise ValueError("Cold exact tensor certificate identity changed.")
    if certificate.mesh_to_source_upper != 0.0 or certificate.source_to_mesh_upper != 0.0:
        raise ValueError("Cold exact tensor fidelity did not reproduce zero deviation.")
    from phydrax.meshing._quality import (
        _with_certified_mapped_measures,
        CellQualityEvaluation,
        CellQualityReport,
        evaluate_cell_quality,
    )

    def rebuild_quality(value: object) -> CellQualityEvaluation:
        if type(value) is not CellQualityEvaluation:
            raise TypeError("Cold mapped quality lost its evaluation owner.")
        mesh = certificate.binding.mesh
        geometry = certificate.binding.geometry
        validity = certify_cell_geometry_validity(geometry, mesh=mesh)
        rebuilt = _with_certified_mapped_measures(
            evaluate_cell_quality(mesh), mesh, geometry, validity
        )
        if rebuilt.evaluation_id != value.evaluation_id:
            raise ValueError("Cold mapped quality changed its exact integration inputs.")
        return rebuilt

    evaluation = read_model_artifact(
        path.with_name("exact-tensor-quad-quality.phx"),
        expected_type=CellQualityEvaluation,
        restore=rebuild_quality,
        limits=None,
    )
    quality = CellQualityReport(evaluation)
    if (
        quality.report_id != expected["quality_id"]
        or quality.evaluation.measure_exact is not expected["measure_exact"]
    ):
        raise ValueError("Cold mapped quality changed its identity or exactness status.")
    print(
        json.dumps(
            {
                "certificate_id": certificate.certificate_id,
                "quality_id": quality.report_id,
                "measure_exact": quality.evaluation.measure_exact,
            }
        )
    )


def _rebuild_surface_source_root_atlas(value: object, /) -> SurfaceSourceRootAtlas:
    from phydrax.geometry._surface_source_support import SurfaceSourceRootAtlas

    if type(value) is not SurfaceSourceRootAtlas:
        raise TypeError("Stored surface root atlas has an unexpected concrete type.")
    domain = _rebuild_as(value.domain, MeshingDomain)
    mesh = _rebuild_as(value.mesh, CellMesh)
    geometry = _rebuild_geometry_for_mesh(value.geometry, mesh)
    contract = _rebuild_as(value.coordinate_contract, SpatialCoordinateContract)
    atlas = SurfaceSourceRootAtlas(
        domain,
        mesh,
        geometry,
        value.root_parameters,
        value.root_patches,
        contract,
        root_source_cell_ids=value.root_source_cell_ids,
        root_parameter_bounds=value.root_parameter_bounds,
    )
    if atlas.atlas_id != value.atlas_id:
        raise ValueError(
            "Stored surface root atlas differs from exact source recomputation."
        )
    return atlas


def _rebuild_prepared_surface_support(value: object, /) -> PreparedSurfaceSourceSupport:
    from phydrax.geometry._surface_source_support import (
        PreparedSurfaceSourceSupport,
        SurfaceSourceRootAtlas,
    )

    if type(value) is not PreparedSurfaceSourceSupportArchiveRecipe:
        raise TypeError(
            "Stored prepared surface support has an unexpected archive recipe."
        )
    domain = _rebuild_as(value.domain, MeshingDomain)
    original = _rebuild_continuation_value(value.original)
    if not isinstance(original, (CellMeshingResult, SurfaceSourceRootAtlas)):
        raise TypeError("Prepared surface support lost its accepted root owner.")
    root_boundary = _rebuild_as(value.root_boundary_source, MeshingDomainBoundarySource)
    support = PreparedSurfaceSourceSupport(
        domain, original, value.root_parameters, root_boundary
    )
    # The identity binds every recomputed root bank (cells, atlas cells, source
    # pieces, exact chart bounds, parameters, patches, vertices, coordinates).
    if support.support_id != value.expected_support_id:
        raise ValueError(
            "Prepared surface support differs from its reconstructed root charts."
        )
    return support


def _rebuild_surface_subdivision_source(
    value: object, /
) -> SurfaceSubdivisionBoundarySource:
    from phydrax.meshing._surface_boundary_sources import (
        _boundary_occurrence_identity,
        SurfaceSubdivisionBoundarySource,
    )

    if type(value) is not SurfaceSubdivisionBoundarySourceArchiveRecipe:
        raise TypeError(
            "Stored surface subdivision source has an unexpected recipe type."
        )
    root = _rebuild_as(value.root, MeshingDomainBoundarySource)
    support = _rebuild_prepared_surface_support(value.support)
    lineages = tuple(_rebuild_mesh_lineage(item) for item in value.lineages)
    vertex_stencils = tuple(
        _rebuild_vertex_interpolation_stencil(item) for item in value.vertex_stencils
    )
    targets = _rebuild_tuple_as(value.targets, CellMesh)
    if not targets:
        raise ValueError("Surface subdivision archive requires a nonempty target chain.")
    geometry = _rebuild_geometry_for_mesh(value.geometry, targets[-1])
    source = SurfaceSubdivisionBoundarySource(
        root, support, lineages, targets, vertex_stencils, geometry
    )
    source.require_current()
    if (
        source.source_id != value.source_id
        or source.source_revision != value.source_revision
        or _boundary_occurrence_identity(source.current_boundary_source)
        != value.expected_boundary_occurrence_id
    ):
        raise ValueError("Surface subdivision query differs from its exact root lineage.")
    return source


def _rebuild_sphere_material_atlas(value: object, /) -> SphereMaterialCellAtlas:
    from phydrax.geometry._sphere_material_atlas import SphereMaterialCellAtlas

    if type(value) is not SphereMaterialCellAtlasArchiveRecipe:
        raise TypeError("Stored sphere atlas has an unexpected recipe type.")
    profile = _rebuild_as(value.source_profile, AnalyticImplicitProfile)
    mesh = _rebuild_as(value.source_mesh, CellMesh)
    geometry = _rebuild_geometry_for_mesh(value.source_geometry, mesh)
    validity = _rebuild_cell_validity_certificate(value.source_validity, mesh, geometry)
    embedding = _rebuild_embedding_certificate(
        value.source_embedding, mesh, geometry, validity
    )
    atlas = SphereMaterialCellAtlas(profile, mesh, geometry, validity, embedding)
    if atlas.atlas_id != value.expected_atlas_id:
        raise ValueError(
            "Sphere material atlas differs from its recomputed source owners."
        )
    return atlas


def _rebuild_sphere_chart_deformation(value: object, /) -> PreparedSphereChartDeformation:
    from phydrax.geometry._sphere_material_atlas import prepare_sphere_chart_deformation

    if type(value) is not PreparedSphereChartDeformationArchiveRecipe:
        raise TypeError("Stored sphere deformation has an unexpected recipe type.")
    source_atlas = _rebuild_sphere_material_atlas(value.source_atlas)
    target_atlas = _rebuild_sphere_material_atlas(value.target_atlas)
    deformation = prepare_sphere_chart_deformation(source_atlas, target_atlas)
    if deformation.deformation_id != value.expected_deformation_id:
        raise ValueError("Sphere deformation does not reproduce its exact atlas overlap.")
    return deformation


def _rebuild_surface_chart_piece(value: object, /) -> PreparedSurfaceChartPiece:
    from phydrax.discretization._surface_chart_records import PreparedSurfaceChartPiece

    if type(value) is not PreparedSurfaceChartPiece:
        raise TypeError("Stored surface-chart piece has an unexpected concrete type.")
    return PreparedSurfaceChartPiece(
        value.source_row,
        value.target_row,
        value.source_cell_global_id,
        value.target_cell_global_id,
        value.exact_source_reference_vertices,
        value.exact_target_reference_vertices,
        value.source_reference_area,
        value.target_reference_area,
        value.native_boundary_labels,
        value.exact_native_source_polygon,
        value.native_polygon_vertex_indices,
        value.piece_id,
    )


def _rebuild_surface_chart_occurrence(value: object, /) -> object:
    from phydrax.discretization._surface_chart_records import (
        PreparedSurfaceChartOccurrence,
    )

    if type(value) is not PreparedSurfaceChartOccurrence:
        raise TypeError(
            "Stored surface-chart occurrence has an unexpected concrete type."
        )
    return PreparedSurfaceChartOccurrence(
        value.patch,
        value.geometry_entity_id,
        value.occurrence_path,
        value.source_rows,
        value.target_rows,
        value.exact_source_material_charts,
        value.exact_target_material_charts,
        tuple(_rebuild_surface_chart_piece(item) for item in value.pieces),
        tuple(_rebuild_surface_chart_piece(item) for item in value.root_pieces),
        value.material_root_witness_id,
        value.displacement_bounds,
        value.source_domain_coverage,
        value.target_domain_coverage,
    )


def _rebuild_surface_chart_deformation(
    value: object, /
) -> PreparedSurfaceChartDeformation:
    from phydrax._model._structure import _restore_dataclass
    from phydrax.discretization._surface_chart_records import (
        PreparedSurfaceChartDeformation,
    )

    if type(value) is not PreparedSurfaceChartDeformation:
        raise TypeError("Stored surface deformation has an unexpected concrete type.")
    source_mesh = _rebuild_as(value.source_mesh, CellMesh)
    target_mesh = _rebuild_as(value.target_mesh, CellMesh)
    source_geometry = _rebuild_geometry_for_mesh(value.source_geometry, source_mesh)
    target_geometry = _rebuild_geometry_for_mesh(value.target_geometry, target_mesh)
    source_validity = _rebuild_cell_validity_certificate(
        value.source_validity, source_mesh, source_geometry
    )
    target_validity = _rebuild_cell_validity_certificate(
        value.target_validity, target_mesh, target_geometry
    )
    source_embedding = _rebuild_embedding_certificate(
        value.source_embedding, source_mesh, source_geometry, source_validity
    )
    target_embedding = _rebuild_embedding_certificate(
        value.target_embedding, target_mesh, target_geometry, target_validity
    )
    source_fidelity_value = value.source_fidelity_certificate
    target_fidelity_value = value.target_fidelity_certificate
    source_fidelity_query = _rebuild_continuation_value(source_fidelity_value.source)
    target_fidelity_query = _rebuild_continuation_value(target_fidelity_value.source)
    if not isinstance(source_fidelity_query, SourceBoundaryQuery) or not isinstance(
        target_fidelity_query, SourceBoundaryQuery
    ):
        raise TypeError("Stored surface deformation lost an endpoint fidelity query.")
    source_fidelity = _rebuild_source_fidelity_certificate(
        source_fidelity_value,
        source_mesh,
        source_geometry,
        source_fidelity_query,
        source_fidelity_value.source,
    )
    target_fidelity = _rebuild_source_fidelity_certificate(
        target_fidelity_value,
        target_mesh,
        target_geometry,
        target_fidelity_query,
        target_fidelity_value.source,
    )
    root_cover = (
        None
        if value.material_root_chart_cover is None
        else _rebuild_as(value.material_root_chart_cover, MeshingDomainBoundarySource)
    )
    target_cover = (
        None
        if value.target_chart_cover is None
        else _rebuild_as(value.target_chart_cover, MeshingDomainBoundarySource)
    )
    fields = {
        "source_witness": _rebuild_as(value.source_witness, SurfaceChartWitness),
        "target_witness": _rebuild_as(value.target_witness, SurfaceChartWitness),
        "material_root_witness": _rebuild_as(
            value.material_root_witness, SurfaceChartWitness
        ),
        "material_root_chart_cover": root_cover,
        "target_chart_cover": target_cover,
        "source_mesh": source_mesh,
        "target_mesh": target_mesh,
        "source_geometry": source_geometry,
        "target_geometry": target_geometry,
        "occurrences": tuple(
            _rebuild_surface_chart_occurrence(item) for item in value.occurrences
        ),
        "source_geometry_id": value.source_geometry_id,
        "target_geometry_id": value.target_geometry_id,
        "source_topology_id": value.source_topology_id,
        "target_topology_id": value.target_topology_id,
        "domain_id": value.domain_id,
        "source_fidelity_bounds": value.source_fidelity_bounds,
        "target_fidelity_bounds": value.target_fidelity_bounds,
        "source_polynomial_chord_bounds": value.source_polynomial_chord_bounds,
        "source_corner_enclosure_bounds": value.source_corner_enclosure_bounds,
        "source_continuous_chord_bounds": value.source_continuous_chord_bounds,
        "source_cell_measures": value.source_cell_measures,
        "target_cell_measures": value.target_cell_measures,
        "source_measure_errors": value.source_measure_errors,
        "target_measure_errors": value.target_measure_errors,
        "source_validity": source_validity,
        "target_validity": target_validity,
        "source_embedding": source_embedding,
        "target_embedding": target_embedding,
        "source_fidelity_certificate": source_fidelity,
        "target_fidelity_certificate": target_fidelity,
        "maximum_displacement_bound": value.maximum_displacement_bound,
        "deformation_id": value.deformation_id,
    }
    rebuilt = _restore_dataclass(PreparedSurfaceChartDeformation, fields)
    rebuilt.from_stored_fields(rebuilt)
    return rebuilt


def _rebuild_surface_chart_boundary_source(
    value: object, /
) -> SurfaceChartBoundarySource:
    from phydrax.discretization._sphere_chart_records import (
        PreparedSphereChartDeformation,
    )
    from phydrax.discretization._surface_chart_records import (
        PreparedSurfaceChartDeformation,
    )
    from phydrax.geometry._surface_source_support import PreparedSurfaceSourceSupport
    from phydrax.meshing._surface_boundary_sources import (
        SurfaceChartBoundarySource,
        SurfaceSubdivisionBoundarySource,
    )

    if type(value) is not SurfaceChartBoundarySourceArchiveRecipe:
        raise TypeError("Stored surface chart query has an unexpected archive recipe.")
    root = _rebuild_continuation_value(value.root)
    if not isinstance(
        root, (MeshingDomainBoundarySource, SurfaceSubdivisionBoundarySource)
    ):
        raise TypeError("Surface chart query lost its canonical root boundary owner.")
    support = _rebuild_prepared_surface_support(value.support)
    if not isinstance(support, PreparedSurfaceSourceSupport):
        raise TypeError("Surface chart query lost its prepared support owner.")
    if isinstance(value.deformation, PreparedSphereChartDeformationArchiveRecipe):
        deformation = _rebuild_sphere_chart_deformation(value.deformation)
    elif type(value.deformation) is PreparedSurfaceChartDeformation:
        deformation = _rebuild_surface_chart_deformation(value.deformation)
    else:
        raise TypeError("Surface chart query lost its registered deformation owner.")
    if not isinstance(
        deformation, (PreparedSurfaceChartDeformation, PreparedSphereChartDeformation)
    ):
        raise TypeError("Surface chart deformation has no registered exact owner.")
    predecessor_fidelity = _rebuild_source_fidelity_certificate(
        value.predecessor_fidelity,
        deformation.source_mesh,
        deformation.source_geometry,
        root,
        value.root,
    )
    query = SurfaceChartBoundarySource(root, support, deformation, predecessor_fidelity)
    query.require_current()
    sphere_profile_id = sphere_profile_revision = None
    source_atlas_id = target_atlas_id = None
    if type(deformation) is PreparedSphereChartDeformation:
        source_atlas_id = deformation.source_atlas.atlas_id
        target_atlas_id = deformation.target_atlas.atlas_id
        sphere_profile_id = deformation.source_atlas.source_profile.source_id
        sphere_profile_revision = deformation.source_atlas.source_profile.source_revision
    root_scope_id = (
        root.source_scope_id
        if isinstance(root, MeshingDomainBoundarySource)
        else root.current_boundary_source.source_scope_id
    )
    if (
        query.source_query_id != value.expected_source_query_id
        or query.source_id != value.expected_source_id
        or query.source_revision != value.expected_source_revision
        or root_scope_id != value.expected_source_scope_id
        or query.domain.domain_id != value.expected_domain_id
        or query.support.support_id != value.expected_support_id
        or query.deformation.domain_id != value.expected_deformation_domain_id
        or query.deformation.deformation_id != value.expected_deformation_id
        or query.predecessor_fidelity.certificate_id
        != value.expected_predecessor_fidelity_id
        or sphere_profile_id != value.expected_source_profile_id
        or sphere_profile_revision != value.expected_source_profile_revision
        or source_atlas_id != value.expected_source_atlas_id
        or target_atlas_id != value.expected_target_atlas_id
    ):
        raise ValueError(
            "Surface chart query identity differs from its recomputed owners."
        )
    return query


def _rebuild_reconstructed_triangle_patch(value: object, /) -> object:
    from phydrax.geometry.reconstruction._core import _ReconstructedTrianglePatch

    if type(value) is not _ReconstructedTrianglePatch:
        raise TypeError("Stored reconstructed triangle patch has an unexpected type.")
    corners = value.exact_corners
    if (
        not isinstance(corners, tuple)
        or len(corners) != 3
        or any(
            not isinstance(point, tuple)
            or len(point) != 3
            or any(not isinstance(coordinate, Fraction) for coordinate in point)
            for point in corners
        )
    ):
        raise TypeError(
            "Reconstructed triangle corners must retain their exact Fraction bank."
        )
    rebuilt = _ReconstructedTrianglePatch(
        np.asarray(
            [[float(coordinate) for coordinate in point] for point in corners],
            dtype=np.float64,
        )
    )
    if rebuilt.exact_corners != corners or not np.array_equal(
        np.asarray(rebuilt.corners), np.asarray(value.corners)
    ):
        raise ValueError(
            "Reconstructed triangle patch does not renew from exact authored corners."
        )
    return rebuilt


def _rebuild_continuation_value(
    value: object,
    /,
    *,
    source_mesh: CellMesh | None = None,
    actual_geometry: CellGeometrySpec | None = None,
) -> object:
    if (
        type(value).__module__ == "phydrax.geometry.reconstruction._core"
        and type(value).__name__ == "_ReconstructedTrianglePatch"
    ):
        return _rebuild_reconstructed_triangle_patch(value)
    if type(value) is SurfaceChartBoundarySourceArchiveRecipe:
        return _rebuild_surface_chart_boundary_source(value)
    if type(value) is SurfaceSubdivisionBoundarySourceArchiveRecipe:
        return _rebuild_surface_subdivision_source(value)
    if type(value) is SphereMaterialCellAtlasArchiveRecipe:
        return _rebuild_sphere_material_atlas(value)
    if type(value) is PreparedSphereChartDeformationArchiveRecipe:
        return _rebuild_sphere_chart_deformation(value)
    if (
        type(value).__module__ == "phydrax.geometry._surface_source_support"
        and type(value).__name__ == "SurfaceSourceRootAtlas"
    ):
        return _rebuild_surface_source_root_atlas(value)
    if type(value) is PreparedSurfaceSourceSupportArchiveRecipe:
        return _rebuild_prepared_surface_support(value)
    if type(value) is SurfaceNativeRestrictionBoundarySourceArchiveRecipe:
        from phydrax.geometry._surface_source_support import (
            SurfaceNativeRestrictionBoundarySource,
        )

        root = _rebuild_as(value.root, MeshingDomainBoundarySource)
        support = _rebuild_prepared_surface_support(value.support)
        mesh = _rebuild_as(value.mesh, CellMesh)
        geometry = _rebuild_geometry_for_mesh(value.geometry, mesh)
        source = SurfaceNativeRestrictionBoundarySource(root, support, mesh, geometry)
        if (
            source.source_id != value.source_id
            or source.source_revision != value.source_revision
            or source.source_scope_id != value.source_scope_id
        ):
            raise ValueError("Cold native restriction source changed its identity.")
        return source
    if (
        type(value).__module__ == "phydrax.discretization._surface_chart_records"
        and type(value).__name__ == "PreparedSurfaceChartDeformation"
    ):
        return _rebuild_surface_chart_deformation(value)
    if (
        type(value).__module__ == "phydrax._model._structure"
        and type(value).__name__
        in {
            "SparseCoordinateOperatorArchiveRecipe",
            "_FiniteElementSpecArchiveProjection",
        }
    ) or type(value).__module__ in {
        "phydrax.sparse._relation",
        "phydrax.linalg._spaces",
        "phydrax.linalg._operator_pairing",
        "phydrax.linalg._properties",
        "phydrax.linalg._complexes",
        "phydrax.discretization.fem._de_rham",
        "phydrax.geometry._common_root_volume_overlap",
    }:
        return restore_model_archive_value(value)
    if isinstance(value, SurfaceMetadata):
        metadata = SurfaceMetadata(
            source_id=value.source_id,
            source_revision=value.source_revision,
            coordinate_contract=_rebuild_as(
                value.coordinate_contract, SpatialCoordinateContract
            ),
            provenance=value.provenance,
            cell_tags=value.cell_tags,
        )
        if metadata.metadata_id != value.metadata_id:
            raise ValueError("Surface metadata identity changed from its source fields.")
        return metadata
    if isinstance(value, SurfaceSelection):
        selection = SurfaceSelection(
            value.name,
            value.cell_global_ids,
            cell_entity_set_id=value.cell_entity_set_id,
            role=value.role,
        )
        if selection.selection_id != value.selection_id:
            raise ValueError("Surface selection identity changed from its entity scope.")
        return selection
    if isinstance(value, SurfaceInterface):
        support = _rebuild_as(value.support, SurfaceSelection)
        interface = SurfaceInterface(
            value.name,
            support,
            minus_region=value.minus_region,
            plus_region=value.plus_region,
        )
        if interface.interface_id != value.interface_id:
            raise ValueError("Surface interface identity changed from its support.")
        return interface
    if isinstance(value, SurfaceOrientationRepair):
        repair = SurfaceOrientationRepair(
            source_face_indices=value.source_face_indices,
            orientation_signs=value.orientation_signs,
            component_ids=value.component_ids,
            source_topology_id=value.source_topology_id,
            repaired_topology_id=value.repaired_topology_id,
        )
        if repair.repair_id != value.repair_id:
            raise ValueError("Surface orientation repair identity changed.")
        return repair
    if isinstance(value, SurfaceModel):
        mesh = _rebuild_as(value.mesh, CellMesh)
        metadata = _rebuild_as(value.metadata, SurfaceMetadata)
        model = SurfaceModel(
            mesh,
            metadata,
            selections=_rebuild_tuple_as(value.selections, SurfaceSelection),
            interfaces=_rebuild_tuple_as(value.interfaces, SurfaceInterface),
            orientation_repair=(
                None
                if value.orientation_repair is None
                else _rebuild_as(value.orientation_repair, SurfaceOrientationRepair)
            ),
        )
        if model.model_id != value.model_id:
            raise ValueError(
                "Surface model identity differs from its reconstructed owners."
            )
        return model
    if type(value) in (PiecewiseLinearComplex, PiecewiseLinearComplexArchiveRecipe):
        return restore_model_archive_value(value)
    if type(value) is NativeHexGridSchedule:
        schedule = NativeHexGridSchedule(
            value.route,
            minimum_scaled_jacobian=value.minimum_scaled_jacobian,
            minimum_mean_ratio=value.minimum_mean_ratio,
            maximum_aspect_ratio=value.maximum_aspect_ratio,
        )
        if schedule.schedule_id != value.schedule_id:
            raise ValueError("Stored native hex schedule identity changed.")
        return schedule
    if isinstance(value, PiecewiseLinearDomain):
        domain = PiecewiseLinearDomain(
            value.vertices,
            value.facets,
            value.facet_regions,
            value.region_ids,
            source_id=value.source_id,
            region_occurrence_paths=value.region_occurrence_paths,
        )
        if (
            domain.source_revision != value.source_revision
            or domain.domain_id != value.domain_id
        ):
            raise ValueError("Piecewise-linear coverage domain identity changed.")
        return domain
    if isinstance(value, MappedReferenceDomain):
        reference_mesh = _rebuild_as(value.reference_mesh, CellMesh)
        source_geometry = _rebuild_geometry_for_mesh(
            value.source_geometry, reference_mesh
        )
        reference_domain = _rebuild_as(value.reference_domain, PiecewiseLinearDomain)
        domain = MappedReferenceDomain(
            reference_domain,
            reference_mesh,
            source_geometry,
            value.cell_regions,
            source_id=value.source_id,
            source_revision=value.source_revision,
            coordinate_contract=_rebuild_as(
                value.coordinate_contract, SpatialCoordinateContract
            ),
            root_vertex_occurrence_paths=value.root_vertex_occurrence_paths,
            root_cell_occurrence_paths=value.root_cell_occurrence_paths,
            reference_domain_vertex_root_ids=(
                None
                if value.reference_domain_vertex_root_ids is None
                else np.asarray(value.reference_domain_vertex_root_ids, dtype=np.int64)
            ),
        )
        if domain.domain_id != value.domain_id:
            raise ValueError("Mapped reference coverage domain identity changed.")
        return domain
    if type(value) is SourceBoundaryDistance:
        return SourceBoundaryDistance(value.lower, value.upper, value.semantics)
    if type(value) is SourceBoundarySamples:
        samples = SourceBoundarySamples(
            value.points,
            value.covering_radius,
            value.point_error,
            value.semantics,
            complete=value.complete,
        )
        if not np.array_equal(samples.points, value.points):
            raise ValueError("Source boundary sample coordinates changed on restoration.")
        return samples
    if type(value) is RestrictedDelaunayBoundaryWitness:
        witness = RestrictedDelaunayBoundaryWitness(
            mesh_id=value.mesh_id,
            geometry_id=value.geometry_id,
            topology_id=value.topology_id,
            profile_id=value.profile_id,
            source_id=value.source_id,
            source_revision=value.source_revision,
            domain_id=value.domain_id,
            distance_witness_id=value.distance_witness_id,
        )
        if witness.witness_id != value.witness_id:
            raise ValueError(
                "Restricted Delaunay witness identity changed on restoration."
            )
        return witness
    if type(value) is RestrictedDelaunayCoverageTransportWitness:
        root_witness = _rebuild_as(value.root_witness, RestrictedDelaunayBoundaryWitness)
        root_distance_witness = _rebuild_as(
            value.root_distance_witness, SourceBoundaryDistanceWitness
        )
        transitions = []
        for stored_transition in value.transitions:
            source_mesh = _rebuild_as(stored_transition.source_mesh, CellMesh)
            source_geometry = _rebuild_geometry_for_mesh(
                stored_transition.source_geometry, source_mesh
            )
            target_mesh = _rebuild_as(stored_transition.target_mesh, CellMesh)
            target_geometry = _rebuild_geometry_for_mesh(
                stored_transition.target_geometry, target_mesh
            )
            transitions.append(
                _rebuild_cell_geometry_transition(
                    stored_transition,
                    source_mesh,
                    source_geometry,
                    target_mesh,
                    target_geometry,
                )
            )
        witness = RestrictedDelaunayCoverageTransportWitness(
            root_witness=root_witness,
            root_distance_witness=root_distance_witness,
            transitions=tuple(transitions),
            stage_cell_regions=value.stage_cell_regions,
            target_distance_witness_id=value.target_distance_witness_id,
        )
        if witness.witness_id != value.witness_id:
            raise ValueError(
                "Restricted coverage transport identity changed on restoration."
            )
        return witness
    if isinstance(value, SurfaceChartWitness):
        root_mesh_cell_bindings = value.root_mesh_cell_bindings
        if not isinstance(root_mesh_cell_bindings, tuple) or any(
            binding is not None and type(binding) is not SurfaceRootMeshCellBinding
            for binding in root_mesh_cell_bindings
        ):
            raise TypeError("Stored surface witness has malformed root-mesh bindings.")
        witness = SurfaceChartWitness(
            geometry_id=value.geometry_id,
            topology_id=value.topology_id,
            domain_id=value.domain_id,
            cell_kind=value.cell_kind,
            cell_global_ids=value.cell_global_ids,
            patches=value.patches,
            charts=value.charts,
            source_root_cell_ids=value.source_root_cell_ids,
            source_piece_ids=value.source_piece_ids,
            geometry_entity_ids=value.geometry_entity_ids,
            occurrence_paths=value.occurrence_paths,
            chart_rational_tokens=value.chart_rational_tokens,
            root_mesh_cell_bindings=root_mesh_cell_bindings,
        )
        if witness.witness_id != value.witness_id:
            raise ValueError(
                "Surface chart witness identity differs from its source ancestry."
            )
        return witness
    if isinstance(value, SourceBoundaryChartWitness):
        witness = SourceBoundaryChartWitness(
            mesh_id=value.mesh_id,
            geometry_id=value.geometry_id,
            topology_id=value.topology_id,
            source_id=value.source_id,
            source_revision=value.source_revision,
            source_query_id=value.source_query_id,
            chart_dimension=value.chart_dimension,
            reference_dimension=value.reference_dimension,
            cell_global_ids=value.cell_global_ids,
            entity_global_ids=value.entity_global_ids,
            local_entity_indices=value.local_entity_indices,
            target_reference_vertices=value.target_reference_vertices,
            target_reference_rational_tokens=value.target_reference_rational_tokens,
            target_face_vertex_global_ids=value.target_face_vertex_global_ids,
            source_chart_ids=value.source_chart_ids,
            source_entity_ids=value.source_entity_ids,
            occurrence_paths=value.occurrence_paths,
            source_parameters=value.source_parameters,
            source_parameter_rational_tokens=value.source_parameter_rational_tokens,
            orientations=value.orientations,
            implicit_cell_keys=value.implicit_cell_keys,
            implicit_root_edge_keys=value.implicit_root_edge_keys,
            cad_plc=(
                None if value.cad_plc is None
                else _rebuild_as(value.cad_plc, PiecewiseLinearComplex)
            ),
            cad_curve_association=value.cad_curve_association,
            cad_root_cell_ids=value.cad_root_cell_ids,
        )
        if witness.witness_id != value.witness_id:
            raise ValueError("Source boundary chart witness identity changed.")
        return witness
    if isinstance(value, SourceBoundaryPointWitness):
        witness = SourceBoundaryPointWitness(
            mesh_id=value.mesh_id,
            geometry_id=value.geometry_id,
            topology_id=value.topology_id,
            source_id=value.source_id,
            source_revision=value.source_revision,
            source_query_id=value.source_query_id,
            vertex_global_ids=value.vertex_global_ids,
            source_entity_ids=value.source_entity_ids,
            occurrence_paths=value.occurrence_paths,
        )
        if witness.witness_id != value.witness_id:
            raise ValueError("Source boundary point witness identity changed.")
        return witness
    if type(value) is SourceBoundaryDistanceWitness:
        witness = SourceBoundaryDistanceWitness(
            mesh_id=value.mesh_id,
            geometry_id=value.geometry_id,
            topology_id=value.topology_id,
            source_id=value.source_id,
            source_revision=value.source_revision,
            source_query_id=value.source_query_id,
            sample_count=value.sample_count,
            cover_radius=value.cover_radius,
            target_facet_global_ids=value.target_facet_global_ids,
        )
        if witness.witness_id != value.witness_id:
            raise ValueError("Source boundary distance witness identity changed.")
        return witness
    if type(value) is SurfaceEnvelopeDistanceWitness:
        witness = SurfaceEnvelopeDistanceWitness(
            mesh_id=value.mesh_id,
            geometry_id=value.geometry_id,
            topology_id=value.topology_id,
            source_id=value.source_id,
            source_revision=value.source_revision,
            source_query_id=value.source_query_id,
            offset=value.offset,
            spacing=value.spacing,
            certificate_spacing=value.certificate_spacing,
            bounds=value.bounds,
            envelope_vertices=value.envelope_vertices,
            envelope_triangles=value.envelope_triangles,
            target_vertex_global_ids=value.target_vertex_global_ids,
            target_facet_global_ids=value.target_facet_global_ids,
        )
        if witness.witness_id != value.witness_id:
            raise ValueError("Surface envelope distance witness identity changed.")
        return witness
    if type(value) is SurfaceEnvelopeSourceArchiveRecipe:
        original = _rebuild_continuation_value(value.original)
        contract = _rebuild_continuation_value(value.coordinate_contract)
        if not isinstance(original, ReconstructedGeometrySource):
            raise TypeError(
                "Surface envelope recipe lost its reconstructed source owner."
            )
        if not isinstance(contract, SpatialCoordinateContract):
            raise TypeError("Surface envelope recipe lost its coordinate contract.")
        source = SurfaceEnvelopeSource(
            original,
            source_id=value.source_id,
            coordinate_contract=contract,
            offset=value.offset,
            spacing=value.spacing,
            certificate_spacing=value.certificate_spacing,
            maximum_deviation=value.maximum_deviation,
        )
        if type(source) is not SurfaceEnvelopeSource:
            raise TypeError(
                "Surface envelope constructor returned an unexpected concrete owner."
            )
        if (
            source.source_revision != value.expected_source_revision
            or source.envelope_id != value.expected_envelope_id
        ):
            raise ValueError(
                "Surface envelope source does not recompute from its retained source."
            )
        return source
    if type(value) is PiecewiseLinearBoundarySourceArchiveRecipe:
        owner = _rebuild_continuation_value(value.owner)
        return _rebuild_piecewise_boundary_query_recipe(value, owner)
    if type(value) is NativePlcSourceArchiveRecipe:
        return _rebuild_native_plc_source_archive_recipe(value)
    if isinstance(value, NativeMappedHexSourceArchiveRecipe):
        if type(value) is not NativeMappedHexSourceArchiveRecipe:
            raise TypeError("Mapped hex source recipe has an unexpected concrete type.")
        from phydrax.meshing.providers._native_sources import NativeMappedHexSource

        reference_recipe = value.reference
        if not isinstance(reference_recipe, NativePlcSourceArchiveRecipe):
            raise TypeError(
                "Mapped hex source recipe lost its exact native PLC reference."
            )
        domain = _rebuild_as(value.domain, MappedReferenceDomain)
        if reference_recipe.mapped_domain is None:
            reference = _rebuild_native_plc_source_archive_recipe(reference_recipe)
        else:
            reference = _rebuild_native_plc_source_archive_recipe(
                reference_recipe, mapped_domain_override=domain
            )
        root_global_ids = np.asarray(value.plc_vertex_root_global_ids, dtype=np.int64)
        source = NativeMappedHexSource(
            reference,
            domain,
            plc_vertex_root_global_ids=root_global_ids,
        )
        if (
            source.source_id != value.expected_source_id
            or source.source_revision != value.expected_source_revision
            or source.binding_id != value.expected_binding_id
            or source.root_relation.relation_id != value.expected_relation_id
        ):
            raise ValueError(
                "Mapped hex source relation differs from its exact authority recipe."
            )
        source.validate_restored()
        return source
    if type(value) is MappedReferenceBoundarySourceArchiveRecipe:
        from phydrax.geometry._mapped_reference_boundary import (
            MappedPlcRootAuthority,
            MappedReferenceBoundarySource,
        )

        authority = _rebuild_continuation_value(value.authority)
        if not isinstance(authority, MappedPlcRootAuthority):
            raise TypeError(
                "Mapped boundary query recipe lost its native PLC root authority."
            )
        query = MappedReferenceBoundarySource(
            authority,
            face_scope_keys=value.face_scope_keys,
            edge_scope_keys=value.edge_scope_keys,
        )
        if query.source_query_id != value.expected_source_query_id:
            raise ValueError(
                "Mapped boundary query identity differs from its exact root authority."
            )
        query.require_current()
        return query
    if isinstance(value, AnalyticImplicitProfileArchiveRecipe):
        geometry = _rebuild_compiled_analytic_geometry(value.geometry)
        contract = _rebuild_as(value.coordinate_contract, SpatialCoordinateContract)
        if not isinstance(contract, SpatialCoordinateContract):
            raise TypeError("Analytic implicit profile lost its coordinate contract.")
        profile = AnalyticImplicitProfile(
            geometry,
            contract,
            tube_radius=value.tube_radius,
            cover_radius=value.cover_radius,
            source_id=value.source_id,
            source_revision=value.source_revision,
        )
        if (
            profile.source_id != value.source_id
            or profile.source_revision != value.source_revision
            or profile.profile_id != value.expected_profile_id
        ):
            raise ValueError(
                "Analytic implicit source profile identity changed on cold reconstruction."
            )
        return profile
    if type(value) is AdaptiveImplicitSurfaceArchiveRecipe:
        return _rebuild_adaptive_implicit_source(value)[1]
    if isinstance(value, AdaptiveImplicitBoundarySourceArchiveRecipe):
        # The stored surface recipe rebuilds (and revalidates) as its surface.
        surface = _rebuild_as(value.surface, AdaptiveImplicitSurface)
        query = AdaptiveImplicitBoundarySource(surface, frame_id=value.frame_id or None)
        if (
            query.source_id != value.expected_source_id
            or query.source_revision != value.expected_source_revision
            or query.frame_id != value.frame_id
        ):
            raise ValueError("Adaptive implicit boundary query identity changed.")
        return query
    if isinstance(value, ReconstructedGeometrySourceArchiveRecipe):
        return _rebuild_reconstructed_source(value)[1]
    if isinstance(value, ReconstructedTriangleBoundarySourceArchiveRecipe):
        source = _rebuild_continuation_value(value.source)
        if not isinstance(source, ReconstructedGeometrySource):
            raise TypeError("Reconstructed triangle query requires its source owner.")
        query = ReconstructedTriangleBoundarySource(source)
        if (query.source_id, query.source_revision) != (
            value.expected_source_id,
            value.expected_source_revision,
        ):
            raise ValueError("Reconstructed triangle query identity changed.")
        return query
    if isinstance(value, MeshingDomainArchiveRecipe):
        if type(value.corner_points_authored) is not bool:
            raise TypeError("MeshingDomain corner-point provenance must be boolean.")
        patches = _rebuild_tuple_as(value.patches, MeshingSurfacePatch)
        curves = _rebuild_tuple_as(value.curves, MeshingDomainCurve)
        regions = _rebuild_tuple_as(value.regions, MeshingDomainRegion)
        brep_authority = None
        if value.brep_authority is not None:
            candidate = _rebuild_continuation_value(value.brep_authority)
            if not isinstance(candidate, BRepModel):
                raise TypeError(
                    "Meshing-domain recipe lost its registered BRep authority."
                )
            brep_authority = candidate
        domain = MeshingDomain(
            patches,
            curves,
            value.corner_count,
            source_id=value.source_id,
            source_revision=value.source_revision,
            regions=regions,
            tolerance=value.tolerance,
            accuracy=_meshing_source_accuracy(value.accuracy),
            source_indices=value.source_indices,
            source_occurrences=value.source_occurrences,
            region_source_indices=value.region_source_indices,
            region_source_occurrences=value.region_source_occurrences,
            source_kinds=value.source_kinds,
            authority_id=value.authority_id,
            brep_authority=brep_authority,
            corner_points=(
                None
                if not value.corner_points_authored
                else _array_archive_input(value.corner_points)
            ),
        )
        if not np.array_equal(
            np.asarray(domain.corner_points), np.asarray(value.corner_points)
        ):
            raise ValueError(
                "MeshingDomain corner points changed during source reconstruction."
            )
        if domain.domain_id != value.expected_domain_id:
            raise ValueError("MeshingDomain identity differs from its source recipe.")
        return domain
    if isinstance(value, MeshingDomainBoundarySourceArchiveRecipe):
        domain = _rebuild_continuation_value(value.domain)
        if not isinstance(domain, MeshingDomain):
            raise TypeError("Meshing-domain boundary query requires its source domain.")
        query = MeshingDomainBoundarySource(
            domain,
            value.patches,
            resolution=value.resolution,
            boundary_occurrences=(
                None
                if value.boundary_occurrences is None
                else _array_archive_input(value.boundary_occurrences)
            ),
            boundary_cell_ids=(
                None
                if value.boundary_cell_ids is None
                else _array_archive_input(value.boundary_cell_ids)
            ),
            boundary_patch_ids=(
                None
                if value.boundary_patch_ids is None
                else _array_archive_input(value.boundary_patch_ids)
            ),
        )
        if query.source_scope_id != value.expected_source_scope_id:
            raise ValueError("Meshing-domain boundary scope identity changed.")
        query.require_current()
        return query
    if isinstance(value, SurfaceSourceChartsArchiveRecipe):
        from phydrax.geometry._surface_source_archive import restore_surface_source_charts

        return restore_surface_source_charts(value)
    if isinstance(value, NativeImplicitBoundaryQueryArchiveRecipe):
        query = _rebuild_continuation_value(value.query)
        if not isinstance(query, AdaptiveImplicitBoundarySource):
            raise TypeError(
                "Native implicit query recipe lost its source boundary query."
            )
        from phydrax.meshing.providers._native_implicit import (
            _NativeImplicitBoundaryQuery,
        )

        profile = (
            None
            if value.authoring_profile is None
            else _rebuild_as(value.authoring_profile, AnalyticImplicitProfile)
        )
        native_query = _NativeImplicitBoundaryQuery(
            query,
            value.source_id,
            value.source_revision,
            value.frame_id,
            authoring_profile=profile,
        )
        if (
            native_query.source_id != value.source_id
            or native_query.source_revision != value.source_revision
            or native_query.frame_id != value.frame_id
            or native_query.query_id != value.expected_query_id
        ):
            raise ValueError("Native implicit boundary query identity changed.")
        return native_query
    if isinstance(value, (PatchCurveUse, PatchPoleUse)):
        return _rebuild_patch_boundary_use(value)
    if isinstance(value, MeshingSurfacePatch):
        surface = _rebuild_as(value.surface, AbstractSurfacePatch)
        loops = tuple(
            tuple(_rebuild_patch_boundary_use(use) for use in loop)
            for loop in value.loops
        )
        return MeshingSurfacePatch(
            surface,
            loops,
            orientation=value.orientation,
        )
    if isinstance(value, MeshingDomainCurve):
        return MeshingDomainCurve(value.start, value.end)
    if isinstance(value, MeshingDomainRegion):
        return MeshingDomainRegion(value.name, value.boundary)
    if isinstance(value, MeshingScope):
        scope = MeshingScope(
            value.source_id,
            value.source_revision,
            value.entity_kind,
            value.entity_dimension,
            value.entity_set_id,
            value.entity_ids,
        )
        if scope.scope_id != value.scope_id:
            raise ValueError("MeshingScope identity differs from its exact source IDs.")
        return scope
    if isinstance(value, MeshPatch):
        return MeshPatch(
            value.name,
            _rebuild_as(value.scope, MeshingScope),
            connected=value.connected,
            adjacent_zone_ids=value.adjacent_zone_ids,
            source_adjacent_region_ids=value.source_adjacent_region_ids,
            source_adjacent_region_occurrence_paths=(
                value.source_adjacent_region_occurrence_paths
            ),
        )
    if isinstance(value, MeshZone):
        return MeshZone(
            value.name,
            value.role,
            _rebuild_as(value.scope, MeshingScope),
            material_id=value.material_id,
            region_role=value.region_role,
        )
    if isinstance(value, MeshLabel):
        return MeshLabel(value.name, _rebuild_as(value.scope, MeshingScope))
    if isinstance(value, MeshAttribute):
        return MeshAttribute(
            value.name,
            value.role,
            _rebuild_as(value.scope, MeshingScope),
            value.values,
            unit=(
                None if value.unit is None else _rebuild_as(value.unit, UnitDefinition)
            ),
        )
    if isinstance(value, PlanarEmbedding):
        embedding = PlanarEmbedding(
            value.origin,
            value.x_axis,
            value.y_axis,
            value.normal,
        )
        if embedding.embedding_id != value.embedding_id:
            raise ValueError("Planar embedding identity differs from its source frame.")
        return embedding
    if isinstance(value, ProtectedFeature):
        return ProtectedFeature(
            _rebuild_as(value.scope, MeshingScope),
            value.feature_kind,
            maximum_deviation=value.maximum_deviation,
            hard=value.hard,
        )
    if isinstance(value, GeometryAssociation):
        # Rows without a source stratum are stored as -1; when no row names one
        # the constructor's absent-strata form reproduces exactly those banks.
        strata_absent = value.association_kind is not GeometryAssociationKind.BREP and (
            bool(np.all(np.asarray(value.source_dimensions) == -1))
            and bool(np.all(np.asarray(value.source_indices) == -1))
        )
        return GeometryAssociation(
            value.association_kind,
            value.source_id,
            value.source_revision,
            value.target_entity_set_id,
            value.target_global_ids,
            value.source_entity_ids,
            value.residuals,
            resolved=value.resolved,
            ambiguous=value.ambiguous,
            exact=value.exact,
            source_dimensions=None if strata_absent else value.source_dimensions,
            source_indices=None if strata_absent else value.source_indices,
            source_roles=value.source_roles,
            source_occurrence_paths=value.source_occurrence_paths,
            source_region_sides=value.source_region_sides,
            source_parameter_intervals=value.source_parameter_intervals,
            source_parameter_points=value.source_parameter_points,
            parameters=value.parameters,
            orientations=value.orientations,
            parent_dimensions=value.parent_dimensions,
            parent_ids=value.parent_ids,
            parent_association_id=value.parent_association_id,
            provenance=value.provenance,
        )
    if isinstance(value, ParametricCurveBoundarySourceArchiveRecipe):
        query = ParametricCurveBoundarySource(
            _rebuild_tuple_as(value.curves, AbstractCurve),
            value.parameter_ranges,
            source_id=value.source_id,
            source_revision=value.source_revision,
            curve_entity_ids=value.curve_entity_ids,
            occurrence_paths=value.occurrence_paths,
            chart_ids=value.chart_ids,
        )
        if query.query_id != value.expected_query_id:
            raise ValueError(
                "Parametric curve query identity differs from its source recipe."
            )
        query.require_current()
        return query
    if isinstance(value, LineCurve):
        return LineCurve(value.origin, value.direction)
    if isinstance(value, ExactLineCurve):
        return ExactLineCurve(value.first_exact, value.last_exact)
    if isinstance(value, (AbstractCurve, AbstractSurfacePatch, BRepGeometry)):
        from phydrax.interchange._cad_archive import _reconstruct_brep_graph

        return _reconstruct_brep_graph(value)
    if isinstance(value, SpatialCoordinateContract):
        from phydrax.interchange._cad_archive import _reconstruct_brep_graph

        return _reconstruct_brep_graph(value)
    if isinstance(value, FormType):
        return FormType(
            value.dimension,
            value.degree,
            twist=value.twist,
            fiber_shape=value.fiber_shape,
            ambient_dimension=value.ambient_dimension,
        )
    if isinstance(value, FormValueSpec):
        return FormValueSpec(_rebuild_as(value.form_type, FormType), proxy=value.proxy)
    if isinstance(value, FiniteElementSpec):
        return FiniteElementSpec(
            value.family,
            value.cell_kind,
            value.degree,
            value.reference_nodes,
            value.entity_dofs,
            value_spec=_rebuild_as(value.value_spec, FormValueSpec),
            continuity=value.continuity,
            representation=value.representation,
            tabulator=value.tabulator,
            tabulator_id=value.tabulator_id,
            form_basis=value.form_basis,
        )
    if type(value) is CellVertexGeometryElement:
        return CellVertexGeometryElement(value.cell_kind, value.local_dof_count)
    from phydrax.discretization._cell_geometry import SplineCellGeometryElement

    if isinstance(value, SplineCellGeometryElement):
        rebuilt = SplineCellGeometryElement(
            value.anchor_ids,
            value.parameter_bounds,
            value.exact_extraction_operator,
            value.weights,
            value.source_id,
            value.source_revision,
            degrees=value.degrees,
            exact_weights=value.exact_weights,
        )
        if rebuilt.element_id != value.element_id:
            raise ValueError("Stored spline element differs from its exact extraction.")
        return rebuilt
    if isinstance(value, PerCellRestrictedCellGeometryElement):
        from phydrax.discretization._cell_geometry import _require_stored_chart_term_types

        source_element = _rebuild_scalar_coordinate_element(value.source_element)
        # Retain one validated expression row per exact repeated chart template.
        # Re-expanding every row as dictionaries before the constructor defeats
        # its own call-local canonical sharing on large refined blocks.
        chart_templates = {}
        charts = []
        for row, terms in enumerate(value.chart_coefficients):
            _require_stored_chart_term_types(terms)
            chart = chart_templates.get(terms)
            if chart is None:
                chart = value.chart_arguments(row)
                chart_templates[terms] = chart
            charts.append(chart)
        return PerCellRestrictedCellGeometryElement.from_stored_fields(
            value, source_element, charts
        )
    if isinstance(value, RationalComposedCellGeometryElement):
        source_element = _rebuild_scalar_coordinate_element(value.source_element)
        return RationalComposedCellGeometryElement(
            source_element, value.cell_kind, value.chart_arguments
        )
    if isinstance(value, BRepModel):
        from phydrax.interchange._cad_archive import _reconstruct_brep_graph

        return _reconstruct_brep_graph(value)
    if isinstance(value, MaterialSourceMapCapability):
        return MaterialSourceMapCapability.revalidate(value)
    if isinstance(value, MaterialReferenceDeclaration):
        root_mesh = _rebuild_as(value.root_mesh, CellMesh)
        root_geometry = _rebuild_geometry_for_mesh(value.root_geometry, root_mesh)
        source_map = _rebuild_as(value.source_map, MaterialSourceMapCapability)
        if not isinstance(root_mesh, CellMesh) or not isinstance(
            root_geometry, CellGeometrySpec
        ):
            raise TypeError(
                "Material reference declaration requires rebuilt root owners."
            )
        if not isinstance(source_map, MaterialSourceMapCapability):
            raise TypeError(
                "Material reference declaration lost its source-map capability."
            )
        root_coverage = _rebuild_domain_coverage_certificate(
            value.root_coverage, root_mesh, root_geometry
        )
        return MaterialReferenceDeclaration.revalidate(
            value, root_mesh, root_geometry, source_map, root_coverage
        )
    if isinstance(value, VolumeMaterialDeformation):
        from phydrax.geometry._common_root_volume_overlap import (
            MaterialReferenceVolumeWitness,
        )

        reference = _rebuild_as(value.reference, MaterialReferenceDeclaration)
        source_mesh = _rebuild_as(value.source_mesh, CellMesh)
        source_geometry = _rebuild_geometry_for_mesh(value.source_geometry, source_mesh)
        target_mesh = _rebuild_as(value.target_mesh, CellMesh)
        target_geometry = _rebuild_geometry_for_mesh(value.target_geometry, target_mesh)
        if not isinstance(reference, MaterialReferenceDeclaration):
            raise TypeError(
                "Material deformation lost its rebuilt reference declaration."
            )
        if not isinstance(source_mesh, CellMesh):
            raise TypeError("Material deformation lost its source mesh.")
        if not isinstance(target_mesh, CellMesh):
            raise TypeError("Material deformation lost its target mesh.")
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("Material deformation lost its source geometry.")
        if not isinstance(target_geometry, CellGeometrySpec):
            raise TypeError("Material deformation lost its target geometry.")
        source_coverage = _rebuild_domain_coverage_certificate(
            value.source_coverage, source_mesh, source_geometry
        )
        target_coverage = _rebuild_domain_coverage_certificate(
            value.target_coverage, target_mesh, target_geometry
        )
        coordinate_transition = (
            None
            if value.coordinate_transition is None
            else _rebuild_cell_geometry_transition(
                value.coordinate_transition,
                source_mesh,
                source_geometry,
                target_mesh,
                target_geometry,
            )
        )
        stored_witness = value.reference_witness
        if not isinstance(stored_witness, MaterialReferenceVolumeWitness):
            raise TypeError(
                "Material deformation lost its material-reference volume witness."
            )
        rebuilt_witness = stored_witness.revalidate(
            reference.root_mesh,
            reference.root_geometry,
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
        )
        if (
            rebuilt_witness.source_chart_witnesses
            != stored_witness.source_chart_witnesses
            or rebuilt_witness.target_chart_witnesses
            != stored_witness.target_chart_witnesses
        ):
            raise ValueError(
                "Stored material chart witnesses differ from their reconstructed maps."
            )
        return VolumeMaterialDeformation.revalidate(
            value,
            reference,
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            source_coverage=source_coverage,
            target_coverage=target_coverage,
            coordinate_transition=coordinate_transition,
        )
    if isinstance(value, VolumeMaterialChain):
        maps = _rebuild_tuple_as(value.maps, VolumeMaterialMap)
        return VolumeMaterialChain.revalidate(value, maps)
    if isinstance(value, VolumeMaterialMap):
        source_mesh = _rebuild_as(value.source_mesh, CellMesh)
        source_geometry = _rebuild_geometry_for_mesh(value.source_geometry, source_mesh)
        target_mesh = _rebuild_as(value.target_mesh, CellMesh)
        target_geometry = _rebuild_geometry_for_mesh(value.target_geometry, target_mesh)
        reference = _rebuild_as(value.reference, MaterialReferenceDeclaration)
        if not isinstance(reference, MaterialReferenceDeclaration):
            raise TypeError(
                "Volume material map requires a rebuilt reference declaration."
            )
        if not isinstance(source_mesh, CellMesh):
            raise TypeError("Volume material map lost its source mesh.")
        if not isinstance(target_mesh, CellMesh):
            raise TypeError("Volume material map lost its target mesh.")
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("Volume material map lost its source geometry.")
        if not isinstance(target_geometry, CellGeometrySpec):
            raise TypeError("Volume material map lost its target geometry.")
        source_coverage = _rebuild_domain_coverage_certificate(
            value.source_coverage, source_mesh, source_geometry
        )
        target_coverage = _rebuild_domain_coverage_certificate(
            value.target_coverage, target_mesh, target_geometry
        )
        coordinate_transition = (
            None
            if value.deformation.coordinate_transition is None
            else _rebuild_cell_geometry_transition(
                value.deformation.coordinate_transition,
                source_mesh,
                source_geometry,
                target_mesh,
                target_geometry,
            )
        )
        return VolumeMaterialMap.revalidate(
            value,
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            root_mesh=reference.root_mesh,
            root_geometry=reference.root_geometry,
            root_coverage=reference.root_coverage,
            source_coverage=source_coverage,
            target_coverage=target_coverage,
            coordinate_transition=coordinate_transition,
        )
    if isinstance(value, EulerianVolumeOverlapChain):
        maps = _rebuild_tuple_as(value.maps, EulerianVolumeOverlapMap)
        return EulerianVolumeOverlapChain.revalidate(value, maps)
    if isinstance(value, EulerianVolumeOverlapMap):
        source_mesh = _rebuild_as(value.source_mesh, CellMesh)
        source_geometry = _rebuild_geometry_for_mesh(value.source_geometry, source_mesh)
        target_mesh = _rebuild_as(value.target_mesh, CellMesh)
        target_geometry = _rebuild_geometry_for_mesh(value.target_geometry, target_mesh)
        if not isinstance(source_mesh, CellMesh):
            raise TypeError("Eulerian overlap map lost its source mesh.")
        if not isinstance(target_mesh, CellMesh):
            raise TypeError("Eulerian overlap map lost its target mesh.")
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("Eulerian overlap map lost its source geometry.")
        if not isinstance(target_geometry, CellGeometrySpec):
            raise TypeError("Eulerian overlap map lost its target geometry.")
        root_mesh: CellMesh | None = None
        root_geometry: CellGeometrySpec | None = None
        if value.common_root is not None:
            root_mesh = _rebuild_as(value.common_root.root_mesh, CellMesh)
            root_geometry = _rebuild_geometry_for_mesh(
                value.common_root.root_geometry, root_mesh
            )
            if not isinstance(root_geometry, CellGeometrySpec):
                raise TypeError("Eulerian overlap map lost its common-root geometry.")
        source_coverage = _rebuild_domain_coverage_certificate(
            value.source_coverage, source_mesh, source_geometry
        )
        target_coverage = _rebuild_domain_coverage_certificate(
            value.target_coverage, target_mesh, target_geometry
        )
        return EulerianVolumeOverlapMap.revalidate(
            value,
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            root_mesh=root_mesh,
            root_geometry=root_geometry,
            source_coverage=source_coverage,
            target_coverage=target_coverage,
        )
    if isinstance(value, CellValidityPolicy):
        policy = CellValidityPolicy(
            relative_determinant_floor=value.relative_determinant_floor,
            relative_planarity_tolerance=value.relative_planarity_tolerance,
        )
        if policy.policy_id != value.policy_id:
            raise ValueError("Cell validity policy identity changed.")
        return policy
    if isinstance(value, CellMeshAuditPolicy):
        policy = CellMeshAuditPolicy(
            unused_entities=value.unused_entities,
            require_complete_association=value.require_complete_association,
            minimum_measure=value.minimum_measure,
            minimum_mean_ratio=value.minimum_mean_ratio,
            maximum_aspect_ratio=value.maximum_aspect_ratio,
            coincident_vertices=value.coincident_vertices,
            coincident_vertex_tolerance=value.coincident_vertex_tolerance,
            duplicate_cells=value.duplicate_cells,
            nonmanifold=value.nonmanifold,
            inconsistent_orientation=value.inconsistent_orientation,
            watertight_boundary=value.watertight_boundary,
            self_intersection=value.self_intersection,
            invalid_geometry=value.invalid_geometry,
            unresolved=value.unresolved,
            validity_policy=(
                None
                if value.validity_policy is None
                else _rebuild_as(value.validity_policy, CellValidityPolicy)
            ),
        )
        if policy.policy_id != value.policy_id:
            raise ValueError("Cell mesh audit policy identity changed.")
        return policy
    if isinstance(value, MeshCertificationSchedule):
        schedule = MeshCertificationSchedule(value.route)
        if schedule.schedule_id != value.schedule_id:
            raise ValueError("Mesh certification schedule identity changed.")
        return schedule
    if isinstance(value, MeshCertificationInputs):
        if source_mesh is None:
            mesh, geometry = _rebuild_mesh_geometry_pair(value.mesh, value.geometry)
        else:
            if (
                source_mesh.mesh_id != value.mesh.mesh_id
                or source_mesh.topology_id != value.mesh.topology_id
            ):
                raise ValueError(
                    "Mesh certification inputs differ from the current mesh context."
                )
            mesh = source_mesh
            geometry = _rebuild_geometry_for_mesh(value.geometry, mesh)
        if actual_geometry is not None and (
            geometry.geometry_layout_id != actual_geometry.geometry_layout_id
            or cell_geometry_id(geometry) != cell_geometry_id(actual_geometry)
        ):
            raise ValueError(
                "Mesh certification geometry differs from its current source context."
            )
        schedule = _rebuild_as(value.schedule, MeshCertificationSchedule)
        domain = None if value.domain is None else _rebuild_coverage_domain(value.domain)
        # One archived definition (an analytic profile serving as domain,
        # fidelity and feature source) restores to one owner object, as the
        # route authored it; the archive stores equal subtrees once.
        profile = domain if isinstance(domain, AnalyticImplicitProfile) else None
        source = (
            None
            if value.source is None
            else profile
            if profile is not None and _same_tree_content(value.source, value.domain)
            else _rebuild_as(value.source, SourceBoundaryQuery)
        )
        inputs = MeshCertificationInputs(
            mesh,
            geometry,
            schedule,
            source_id=value.source_id,
            source_revision=value.source_revision,
            domain=domain,
            cell_regions=value.cell_regions,
            source=source,
            fidelity_tolerance=value.fidelity_tolerance,
            maximum_normal_deviation=value.maximum_normal_deviation,
            junction_vertices=value.junction_vertices,
            protected_features=(
                None
                if value.protected_features is None
                else _rebuild_tuple_as(value.protected_features, ProtectedFeature)
            ),
            scoped_fidelity=tuple(
                _rebuild_scoped_fidelity_input(item) for item in value.scoped_fidelity
            ),
            fidelity_witness=(
                None
                if value.fidelity_witness is None
                else _rebuild_source_fidelity_witness(value.fidelity_witness)
            ),
            feature_source=(
                None
                if value.feature_source is None
                else profile
                if profile is not None
                and _same_tree_content(value.feature_source, value.domain)
                else _rebuild_as(value.feature_source, _SourceEntityScopes)
            ),
            source_coverage_witness=(
                None
                if value.source_coverage_witness is None
                else _rebuild_restricted_coverage_witness(value.source_coverage_witness)
            ),
        )
        if inputs.request_id != value.request_id:
            raise ValueError(
                "Mesh certification inputs differ from their retained source evidence."
            )
        return inputs
    if isinstance(value, CellMeshingResult):
        if source_mesh is None:
            mesh, geometry = _rebuild_mesh_geometry_pair(value.mesh, value.geometry)
        else:
            if (
                source_mesh.mesh_id != value.mesh.mesh_id
                or source_mesh.topology_id != value.mesh.topology_id
            ):
                raise ValueError(
                    "CellMeshingResult differs from the current mesh context."
                )
            mesh = source_mesh
            geometry = _rebuild_geometry_for_mesh(value.geometry, mesh)
        contract = _rebuild_as(value.coordinate_contract, SpatialCoordinateContract)
        if actual_geometry is not None and (
            geometry.geometry_layout_id != actual_geometry.geometry_layout_id
            or cell_geometry_id(geometry) != cell_geometry_id(actual_geometry)
        ):
            raise ValueError(
                "CellMeshingResult geometry differs from its current source context."
            )
        if not isinstance(mesh, CellMesh):
            raise TypeError("Certified mesh result must rebuild a CellMesh.")
        if not isinstance(contract, SpatialCoordinateContract):
            raise TypeError("Certified mesh result must rebuild its coordinate contract.")
        if not isinstance(geometry, CellGeometrySpec):
            raise TypeError("Certified mesh result must rebuild its CellGeometrySpec.")
        boundary = (
            None if value.boundary is None else _rebuild_as(value.boundary, SurfaceModel)
        )
        patches = _rebuild_tuple_as(value.patches, MeshPatch)
        zones = _rebuild_tuple_as(value.zones, MeshZone)
        labels = _rebuild_tuple_as(value.labels, MeshLabel)
        attributes = _rebuild_tuple_as(value.attributes, MeshAttribute)
        associations = _rebuild_tuple_as(value.associations, GeometryAssociation)
        surface_source = (
            None
            if value.surface_source is None
            else _rebuild_as(value.surface_source, SurfaceSourceCharts)
        )
        region_boundary_evidence = tuple(
            RegionBoundaryEvidence(
                mesh,
                _rebuild_as(item.source_scope, MeshingScope),
                item.source_region_id,
                item.control_id,
                item.material_id,
                item.role,
                _rebuild_as(item.boundary_label, MeshLabel),
                item.patch_sides,
                source_occurrence_path=item.source_occurrence_path,
            )
            for item in value.region_boundary_evidence
        )
        planar_embedding = (
            None
            if value.planar_embedding is None
            else _rebuild_as(value.planar_embedding, PlanarEmbedding)
        )
        premises = (
            None
            if value.certification is None
            else _rebuild_mesh_certification_premises(value.certification)
        )
        certification_inputs = None if premises is None else premises[0]
        audit_policy = None if premises is None else premises[1]
        if value.certification is not None and premises is not None:
            # Identity-determined claims are compared before any proof replay so a
            # tampered report is refused without recomputing acceptance.
            value.certification.require_premises(
                premises[0],
                audit_policy=premises[1],
                organization=NativeMeshOrganization(
                    boundary=boundary,
                    patches=patches,
                    zones=zones,
                    labels=labels,
                    attributes=attributes,
                    associations=associations,
                ),
            )
        layer_curving_evidence = (
            None
            if value.layer_curving_evidence is None
            else _rebuild_layer_curving_source_evidence(
                value.layer_curving_evidence,
                mesh,
                geometry,
                require_current=False,
            )
        )
        certified_result = certify_cell_mesh(
            mesh,
            contract,
            geometry=geometry,
            audit_policy=audit_policy,
            boundary=boundary,
            certification_inputs=certification_inputs,
            patches=patches,
            zones=zones,
            labels=labels,
            attributes=attributes,
            associations=associations,
            surface_source=surface_source,
            region_boundary_evidence=region_boundary_evidence,
            planar_embedding=planar_embedding,
        )
        # Restore-time validation: certify_cell_mesh recomputes acceptance once
        # from the rebuilt premises; later revalidation of this report reuses it.
        if value.certification is not None and (
            premises is None
            or certified_result.certification is None
            or certified_result.certification.report_id != value.certification.report_id
            or certified_result.certification.audit.quality.report_id != premises[2]
        ):
            raise ValueError(
                "Stored certification report differs from recomputed premises, quality, or organization."
            )
        if layer_curving_evidence is None:
            result = certified_result
        else:
            result = CellMeshingResult(
                certified_result.mesh,
                certified_result.geometry,
                certified_result.coordinate_contract,
                certified_result.audit,
                certified_result.quality,
                certified_result.compliance,
                certified_result.trace,
                certified_result.provider,
                certified_result.runtime,
                certified_result.derivative_mode,
                certified_result.provenance,
                boundary=certified_result.boundary,
                patches=certified_result.patches,
                zones=certified_result.zones,
                labels=certified_result.labels,
                attributes=certified_result.attributes,
                associations=certified_result.associations,
                adapter_reports=certified_result.adapter_reports,
                surface_source=certified_result.surface_source,
                region_boundary_evidence=certified_result.region_boundary_evidence,
                planar_embedding=certified_result.planar_embedding,
                certification=certified_result.certification,
                layer_curving_evidence=layer_curving_evidence,
            )
        if result.result_id != value.result_id:
            raise ValueError(
                "Stored cell meshing result differs from its revalidated owner evidence."
            )
        return result
    if isinstance(value, CellMeshTransition):
        return _rebuild_cell_mesh_transition(value)
    if type(value) is CellMesh:
        if source_mesh is not None:
            if (
                source_mesh.mesh_id != value.mesh_id
                or source_mesh.topology_id != value.topology_id
            ):
                raise ValueError("Stored CellMesh differs from the current mesh context.")
            return source_mesh
        return CellMesh.from_stored_fields(value, actual_geometry=actual_geometry)
    if isinstance(value, CellGeometryRestrictionSource):
        return _rebuild_cell_geometry_restriction_source(value)
    if type(value) is ExactPowerCellGeometrySource:
        domain_source = value.domain_source
        if type(domain_source) in (PiecewiseLinearComplex, PiecewiseLinearComplexArchiveRecipe):
            domain_source = restore_model_archive_value(domain_source)
        elif type(domain_source) is MappedReferenceDomain:
            domain_source = _rebuild_as(domain_source, MappedReferenceDomain)
        elif domain_source is not None:
            raise TypeError(
                f"No exact-power domain source constructor is registered for {type(domain_source).__name__}."
            )
        target_periodic_cell: PeriodicCell | None = None
        if (
            value.periodic_source_context is not None
            and isinstance(source_mesh, CellMesh)
            and source_mesh.periodic_topology is not None
        ):
            target_cell = source_mesh.periodic_topology.cell
            if not isinstance(target_cell, PeriodicCell):
                raise TypeError("Periodic power target mesh lost its PeriodicCell owner.")
            target_periodic_cell = target_cell
        return ExactPowerCellGeometrySource.from_stored_fields(
            value,
            domain_source=domain_source,
            target_periodic_cell=target_periodic_cell,
        )
    if type(value) is MappedPowerCellGeometrySource:
        reference_source = _rebuild_continuation_value(value.reference_source)
        if type(reference_source) is not ExactPowerCellGeometrySource:
            raise TypeError("Mapped power restoration lost its exact reference source.")
        return MappedPowerCellGeometrySource.from_stored_fields(
            value, reference_source=reference_source
        )
    if type(value) is _CutCoordinateMap:
        if not isinstance(source_mesh, CellMesh):
            raise ValueError(
                "Cut-coordinate replay requires its exact target mesh owner."
            )
        return _rebuild_cut_coordinate_map(value, source_mesh)
    if type(value) is _RestrictedCoordinateMap:
        if not isinstance(source_mesh, CellMesh):
            raise ValueError(
                "Restricted-coordinate replay requires its exact target mesh owner."
            )
        return _rebuild_restricted_coordinate_map(value, source_mesh)
    if isinstance(value, CellGeometrySpec):
        from phydrax.discretization._cell_geometry import (
            CellGeometryCoordinateMap,
            CellGeometryElement,
        )
        from phydrax.geometry._surface_source_support import _SplineSurfaceCoordinateMap

        source_map = value.source_map
        if source_map is not None:
            if not isinstance(source_mesh, CellMesh):
                raise ValueError(
                    "Source-bound coordinate maps require their exact target mesh owner."
                )
            if isinstance(source_map, ExactPlcCellGeometrySource):
                if type(source_map) is not ExactPlcCellGeometrySource:
                    raise TypeError("Stored exact PLC map has a noncanonical owner type.")
                source_map = ExactPlcCellGeometrySource.from_stored_fields(source_map)
            elif isinstance(source_map, ExactPowerCellGeometrySource):
                if type(source_map) is not ExactPowerCellGeometrySource:
                    raise TypeError(
                        "Stored exact power map has a noncanonical owner type."
                    )
                source_map = _rebuild_continuation_value(
                    source_map, source_mesh=source_mesh
                )
                if not isinstance(source_map, CellGeometryCoordinateMap):
                    raise TypeError(
                        "Exact power restoration lost its coordinate-map contract."
                    )
            elif isinstance(source_map, _CutCoordinateMap):
                if type(source_map) is not _CutCoordinateMap:
                    raise TypeError("Stored cut map has a noncanonical owner type.")
                source_map = _rebuild_cut_coordinate_map(source_map, source_mesh)
            elif type(source_map) is MappedPowerCellGeometrySource:
                source_map = _rebuild_continuation_value(source_map)
                if not isinstance(source_map, CellGeometryCoordinateMap):
                    raise TypeError(
                        "Mapped power restoration lost its coordinate-map contract."
                    )
            elif isinstance(source_map, _RestrictedCoordinateMap):
                if type(source_map) is not _RestrictedCoordinateMap:
                    raise TypeError(
                        "Stored restricted map has a noncanonical owner type."
                    )
                source_map = _rebuild_restricted_coordinate_map(source_map, source_mesh)
            elif type(source_map) is _ExactCoordinateBankMap:
                source_map = _ExactCoordinateBankMap.from_stored_fields(
                    source_map, mesh=source_mesh
                )
            elif type(source_map) is _SplineSurfaceCoordinateMap:
                original_map = source_map
                source_map = _SplineSurfaceCoordinateMap(
                    original_map.source_id,
                    original_map.source_revision,
                    original_map.rounded_vertices,
                    original_map.exact_coordinates,
                    value.coordinates,
                    original_map.coefficient_keys,
                    original_map.coefficient_authorities,
                )
                if source_map.source_map_id != original_map.source_map_id:
                    raise ValueError(
                        "Stored spline source map differs from its exact bank."
                    )
            else:
                raise TypeError(
                    f"No explicit coordinate-map archive constructor is registered for {type(source_map).__name__}."
                )
        restriction_source = (
            None
            if value.restriction_source is None
            else _rebuild_cell_geometry_restriction_source(value.restriction_source)
        )
        elements: dict[str, object] = {}
        for name, element in zip(value.block_names, value.elements, strict=True):
            elements[name] = (
                _rebuild_continuation_value(element)
                if isinstance(element, CellGeometryElement)
                else element
            )
        return CellGeometrySpec.from_stored_fields(
            value,
            source_mesh=source_mesh if source_map is not None else None,
            source_map=source_map,
            restriction_source=restriction_source,
            elements=elements,
        )
    if isinstance(value, MaterialSiteId):
        return MaterialSiteId(value.key)
    if isinstance(value, MaterialState):
        return MaterialState(
            _rebuild_as(value.site_id, MaterialSiteId),
            value.model_id,
            value.committed,
            trial=value.trial,
            state_version=value.state_version,
        )
    if isinstance(value, MaterialTransaction):
        return MaterialTransaction(
            tuple(_rebuild_as(state, MaterialState) for state in value.states)
        )
    if isinstance(value, FiniteElementAcceptedElementRecipe):
        return FiniteElementAcceptedElementRecipe(
            value.recipe_kind,
            value.family,
            value.cell_kind,
            value.degree,
            value.continuity,
            value.representation,
            value.value_spec,
            value.form_identity,
            value.reference_nodes,
            value.entity_dofs,
            value.tabulator_id,
            value.element_id,
        )
    if isinstance(value, FiniteElementAcceptedFieldLayoutRecipe):
        return FiniteElementAcceptedFieldLayoutRecipe(
            value.name,
            value.block_names,
            value.component_shape,
            value.storage_kind,
            tuple(
                _rebuild_as(item, FiniteElementAcceptedElementRecipe)
                for item in value.elements
            ),
            value.periodic_quotient,
            value.transfer_semantics,
            value.field_spec_id,
            value.layout_id,
        )
    if isinstance(value, FiniteElementAcceptedFieldLayout):
        return FiniteElementAcceptedFieldLayout.from_recipe(
            _rebuild_as(value.recipe(), FiniteElementAcceptedFieldLayoutRecipe)
        )
    if type(value) is FiniteElementAcceptedStateArchiveRecipe:
        return FiniteElementAcceptedState.from_archive_recipe(value)
    if type(value) is FiniteElementAcceptedState:
        return FiniteElementAcceptedState.from_stored_fields(value)
    if type(value) is FiniteElementDofOwnership:
        return FiniteElementDofOwnership.from_stored_fields(value)
    if type(value) is FiniteElementRestartStateLayout:
        return FiniteElementRestartStateLayout.from_stored_fields(value)
    if type(value) is FiniteElementRestartManifest:
        return FiniteElementRestartManifest.from_stored_fields(value)
    if isinstance(value, PeriodicCell):
        return PeriodicCell(
            value.vectors,
            origin=value.origin,
            periodic_axes=value.periodic_axes,
            maximum_condition_number=value.certified_condition_number,
        )
    if isinstance(value, AxisDomain):
        return AxisDomain(
            value.kind,
            lower=value.lower,
            upper=value.upper,
            direction=value.direction,
        )
    if isinstance(value, AxisDiscretization):
        return AxisDiscretization(
            nodes=value.nodes,
            quad_weights=value.quad_weights,
            basis=value.basis,
            domain=_rebuild_as(value.domain, AxisDomain),
            primary_entity=value.primary_entity,
            lower_endpoint_included=value.lower_endpoint_included,
            upper_endpoint_included=value.upper_endpoint_included,
            active=value.active,
            level=value.level,
            parent_interval=value.parent_interval,
        )
    if isinstance(value, PreparedTensorGrid):
        return PreparedTensorGrid(
            tuple(_rebuild_as(axis, AxisDiscretization) for axis in value.axes),
            axis_names=value.axis_names,
            plan_id=value.plan_id,
            embedding_id=value.support.embedding_id,
        )
    if isinstance(value, MeshPart):
        carrier = _rebuild_as(value.carrier, CellMeshingResult)
        coordinate_contract = _rebuild_as(
            value.coordinate_contract, SpatialCoordinateContract
        )
        return MeshPart(
            value.name,
            carrier,
            coordinate_contract=coordinate_contract,
        )
    if isinstance(value, CellPartition):
        return CellPartition(value.cell_owner, value.part_count)
    if isinstance(value, MeshDistribution):
        part = _rebuild_as(value.part, MeshPart)
        partition = _rebuild_as(value.partition, CellPartition)
        return MeshDistribution(
            part,
            partition,
            cell_global_ids=value.cell_global_ids,
            halo_global_ids=value.halo_global_ids,
            split_factors=value.split_factors,
            halo_width=value.halo_width,
            cell_weights=value.cell_weights,
            partition_kind=value.evidence.kind,
            partition_provenance=value.evidence.provenance,
            partition_feasible=value.evidence.feasible,
        )
    if isinstance(value, tuple):
        return tuple(_rebuild_continuation_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float, np.ndarray, jax.Array)):
        return value
    raise TypeError(
        f"No explicit continuation constructor is registered for {type(value).__name__}."
    )


def _rebuild_finite_volume_precision_policy(
    value: object, /
) -> FiniteVolumePrecisionPolicy:
    from phydrax._precision import PrecisionResolution
    from phydrax.discretization._fv_precision import FiniteVolumePrecisionPolicy

    if not isinstance(value, FiniteVolumePrecisionPolicy):
        raise TypeError("Stored FV precision policy has the wrong owner type.")
    if type(value) is not FiniteVolumePrecisionPolicy:
        raise TypeError("Stored FV precision policy has the wrong concrete owner type.")
    resolution = None
    if value.resolution_id is not None:
        resolution = PrecisionResolution(
            value.request,
            value.provider,
            dict(value.request.requested),
        )
        if resolution.resolution_id != value.resolution_id:
            raise ValueError("Stored FV precision resolution identity changed.")
    policy = FiniteVolumePrecisionPolicy(
        value.storage_dtype,
        reconstruction_dtype=value.reconstruction_dtype,
        flux_dtype=value.flux_dtype,
        reduction_dtype=value.reduction_dtype,
        output_dtype=value.output_dtype,
        checkpoint_dtype=value.checkpoint_dtype,
        resolution=resolution,
    )
    if policy.policy_id != value.policy_id:
        raise ValueError("Stored FV precision policy does not rebuild canonically.")
    return policy


def _rebuild_finite_volume_runtime_state_archive_recipe(
    value: object,
    source_runtime: object,
    /,
) -> FiniteVolumeValidatedRuntimeState:
    from phydrax.solver._finite_volume_content import (
        FiniteVolumeConservativeContentState,
    )
    from phydrax.solver._finite_volume_runtime import (
        FiniteVolumeRuntimeState,
        FiniteVolumeValidatedRuntimeState,
        PreparedFiniteVolumeRuntime,
    )
    from phydrax.solver._finite_volume_topology_events import (
        FiniteVolumeTopologyEventJournal,
    )

    if not isinstance(value, FiniteVolumeRuntimeStateArchiveRecipe):
        raise TypeError("Stored FV runtime state must use its canonical archive recipe.")
    if type(value) is not FiniteVolumeRuntimeStateArchiveRecipe:
        raise TypeError("Stored FV runtime state recipe has the wrong concrete type.")
    if not isinstance(source_runtime, PreparedFiniteVolumeRuntime):
        raise TypeError(
            "Restoring an FV state requires its current prepared source runtime."
        )
    stored_content = value.content_state
    if not isinstance(stored_content, FiniteVolumeConservativeContentState):
        raise TypeError("Stored FV archive lost its canonical conservative content.")
    if type(stored_content) is not FiniteVolumeConservativeContentState:
        raise TypeError("Stored FV archive content has the wrong concrete owner type.")
    precision = _rebuild_finite_volume_precision_policy(stored_content.precision)
    content = FiniteVolumeConservativeContentState(
        stored_content.conservative_content,
        stored_content.effective_cell_volumes,
        stored_content.active_cell_mask,
        stored_content.time,
        topology_epoch_id=stored_content.topology_epoch_id,
        geometry_family_id=stored_content.geometry_family_id,
        geometry_layout_id=stored_content.geometry_layout_id,
        geometry_version=stored_content.geometry_version,
        evidence_policy_id=stored_content.evidence_policy_id,
        evidence_version=stored_content.evidence_version,
        precision=precision,
        owned_cell_mask=stored_content.owned_cell_mask,
    )
    journal = FiniteVolumeTopologyEventJournal.from_archive_record(
        dict(value.topology_journal_record),
        dict(value.topology_journal_arrays),
    )
    sliding_plan = source_runtime.sliding_plan
    sliding_coupling = None
    if sliding_plan is None:
        if value.sliding_coupling_id is not None:
            raise ValueError("Stored FV state requires a missing source sliding plan.")
    else:
        if value.sliding_coupling_id is None:
            raise ValueError(
                "Stored sliding FV state omitted its exact coupling identity."
            )
        sliding_coupling = sliding_plan.coupling(value.sliding_shift)
        if sliding_coupling.coupling_id != value.sliding_coupling_id:
            raise ValueError(
                "Stored FV sliding coupling differs from the current source plan."
            )
    state = FiniteVolumeRuntimeState(
        content,
        journal,
        value.step_size,
        accepted_step=value.accepted_step,
        last_status=value.last_status,
        controller_state=value.controller_state,
        integrator_state=value.integrator_state,
        output_cursor=value.output_cursor,
        sliding_coupling=sliding_coupling,
        sliding_shift=value.sliding_shift,
        sliding_event_id=value.sliding_event_id,
    )
    restored = source_runtime.validate_restored_accepted_state(
        state,
        value.source_cell_ids,
        expected_runtime_id=value.expected_runtime_id,
        expected_epoch_id=value.expected_epoch_id,
        expected_state_id=value.expected_state_id,
    )
    if not isinstance(restored, FiniteVolumeValidatedRuntimeState):
        raise TypeError("FV source-state validation returned no production token.")
    return restored


def _canonical_dependencies(
    dependencies: tuple[CompositionDependency, ...], /
) -> tuple[CompositionDependency, ...]:
    """Authored bindings in the canonical (entry_id, facet) order an entry stores."""
    return tuple(sorted(dependencies, key=lambda item: (item.entry_id, item.facet)))


def _rebuild_composition(value: object, /) -> Composition:
    if not isinstance(value, Composition):
        raise TypeError("value must be the Composition root type.")
    entries = {entry.entry_id: entry for entry in value.entries}
    volume_root_mesh = _rebuild_as(entries["material/root-mesh"].value, CellMesh)
    volume_root_geometry = _rebuild_geometry_for_mesh(
        entries["material/root-geometry"].value, volume_root_mesh
    )
    volume_refined_mesh = _rebuild_as(entries["material/refined-mesh"].value, CellMesh)
    volume_refined_geometry = _rebuild_geometry_for_mesh(
        entries["material/refined-geometry"].value, volume_refined_mesh
    )
    stored_volume_chain = _rebuild_as(
        entries["material/volume-chain"].value, VolumeMaterialChain
    )
    stored_reference = _rebuild_as(
        entries["material/reference-declaration"].value, MaterialReferenceDeclaration
    )
    stored_refinement_deformation = _rebuild_as(
        entries["material/refinement-deformation"].value, VolumeMaterialDeformation
    )
    stored_coarsening_deformation = _rebuild_as(
        entries["material/coarsening-deformation"].value, VolumeMaterialDeformation
    )
    if not all(
        isinstance(item, CellMesh) for item in (volume_root_mesh, volume_refined_mesh)
    ) or not all(
        isinstance(item, CellGeometrySpec)
        for item in (volume_root_geometry, volume_refined_geometry)
    ):
        raise TypeError(
            "Volume material entries require independently rebuilt mesh and geometry owners."
        )
    if (
        not isinstance(stored_volume_chain, VolumeMaterialChain)
        or len(stored_volume_chain.maps) != 2
    ):
        raise TypeError(
            "Volume material chain entry must contain the expected two-map continuation."
        )
    if not isinstance(stored_reference, MaterialReferenceDeclaration):
        raise TypeError(
            "Material reference entry must contain its explicit declaration owner."
        )
    if not all(
        isinstance(item, VolumeMaterialDeformation)
        for item in (stored_refinement_deformation, stored_coarsening_deformation)
    ):
        raise TypeError(
            "Material deformation entries must contain explicit deformation owners."
        )
    source_map = _rebuild_as(stored_reference.source_map, MaterialSourceMapCapability)
    root_coverage = _rebuild_domain_coverage_certificate(
        stored_reference.root_coverage, volume_root_mesh, volume_root_geometry
    )
    material_reference = MaterialReferenceDeclaration.revalidate(
        stored_reference,
        volume_root_mesh,
        volume_root_geometry,
        source_map,
        root_coverage,
    )
    refinement_source_coverage = _rebuild_domain_coverage_certificate(
        stored_refinement_deformation.source_coverage,
        volume_root_mesh,
        volume_root_geometry,
    )
    refinement_target_coverage = _rebuild_domain_coverage_certificate(
        stored_refinement_deformation.target_coverage,
        volume_refined_mesh,
        volume_refined_geometry,
    )
    coarsening_source_coverage = _rebuild_domain_coverage_certificate(
        stored_coarsening_deformation.source_coverage,
        volume_refined_mesh,
        volume_refined_geometry,
    )
    coarsening_target_coverage = _rebuild_domain_coverage_certificate(
        stored_coarsening_deformation.target_coverage,
        volume_root_mesh,
        volume_root_geometry,
    )
    refinement_transition = (
        None
        if stored_refinement_deformation.coordinate_transition is None
        else _rebuild_cell_geometry_transition(
            stored_refinement_deformation.coordinate_transition,
            volume_root_mesh,
            volume_root_geometry,
            volume_refined_mesh,
            volume_refined_geometry,
        )
    )
    coarsening_transition = (
        None
        if stored_coarsening_deformation.coordinate_transition is None
        else _rebuild_cell_geometry_transition(
            stored_coarsening_deformation.coordinate_transition,
            volume_refined_mesh,
            volume_refined_geometry,
            volume_root_mesh,
            volume_root_geometry,
        )
    )
    material_deformations = (
        VolumeMaterialDeformation.revalidate(
            stored_refinement_deformation,
            material_reference,
            volume_root_mesh,
            volume_root_geometry,
            volume_refined_mesh,
            volume_refined_geometry,
            source_coverage=refinement_source_coverage,
            target_coverage=refinement_target_coverage,
            coordinate_transition=refinement_transition,
        ),
        VolumeMaterialDeformation.revalidate(
            stored_coarsening_deformation,
            material_reference,
            volume_refined_mesh,
            volume_refined_geometry,
            volume_root_mesh,
            volume_root_geometry,
            source_coverage=coarsening_source_coverage,
            target_coverage=coarsening_target_coverage,
            coordinate_transition=coarsening_transition,
        ),
    )
    rebuilt_volume_maps = (
        VolumeMaterialMap.revalidate(
            stored_volume_chain.maps[0],
            volume_root_mesh,
            volume_root_geometry,
            volume_refined_mesh,
            volume_refined_geometry,
            root_mesh=volume_root_mesh,
            root_geometry=volume_root_geometry,
            root_coverage=root_coverage,
            source_coverage=refinement_source_coverage,
            target_coverage=refinement_target_coverage,
            coordinate_transition=refinement_transition,
        ),
        VolumeMaterialMap.revalidate(
            stored_volume_chain.maps[1],
            volume_refined_mesh,
            volume_refined_geometry,
            volume_root_mesh,
            volume_root_geometry,
            root_mesh=volume_root_mesh,
            root_geometry=volume_root_geometry,
            root_coverage=root_coverage,
            source_coverage=coarsening_source_coverage,
            target_coverage=coarsening_target_coverage,
            coordinate_transition=coarsening_transition,
        ),
    )
    if any(
        rebuilt_map.deformation.deformation_id != deformation.deformation_id
        for rebuilt_map, deformation in zip(
            rebuilt_volume_maps, material_deformations, strict=True
        )
    ):
        raise ValueError("Material map and independent deformation entries disagree.")
    volume_chain = VolumeMaterialChain.revalidate(
        stored_volume_chain, rebuilt_volume_maps
    )
    if (
        volume_chain.source_mesh_id != volume_root_mesh.mesh_id
        or volume_chain.target_mesh_id != volume_root_mesh.mesh_id
        or volume_chain.source_geometry_id != cell_geometry_id(volume_root_geometry)
        or volume_chain.target_geometry_id != cell_geometry_id(volume_root_geometry)
        or volume_chain.reference.declaration_id != material_reference.declaration_id
    ):
        raise ValueError(
            "Curved material chain endpoints differ from the independent composition owners."
        )
    mesh, geometry = _rebuild_mesh_geometry_pair(
        entries["mesh/topology"].value,
        entries["mesh/geometry"].value,
    )
    stored_result = entries["mesh/result"].value
    if not isinstance(stored_result, CellMeshingResult):
        raise TypeError("Composition mesh result must rebuild from its canonical owner.")
    if type(stored_result) is not CellMeshingResult:
        raise TypeError("Composition mesh result has a noncanonical owner type.")
    rebuilt_result = _rebuild_continuation_value(
        stored_result,
        source_mesh=mesh,
        actual_geometry=_periodic_actual_geometry(stored_result.mesh, geometry),
    )
    if not isinstance(rebuilt_result, CellMeshingResult):
        raise TypeError("Composition mesh result did not rebuild as CellMeshingResult.")
    if type(rebuilt_result) is not CellMeshingResult:
        raise TypeError("Composition mesh result rebuilt with a noncanonical type.")
    result = rebuilt_result
    if (
        result.mesh.mesh_id != mesh.mesh_id
        or result.mesh.topology_id != mesh.topology_id
        or result.geometry.geometry_layout_id != geometry.geometry_layout_id
    ):
        raise ValueError(
            "Mesh, geometry, and accepted result entries are not the same realization."
        )
    _, geometry_routes, geometry_coordinates = geometry.resolve(mesh)
    _, result_routes, result_coordinates = result.geometry.resolve(result.mesh)
    if (
        not np.array_equal(np.asarray(geometry_coordinates), np.asarray(mesh.coordinates))
        or not np.array_equal(
            np.asarray(result_coordinates), np.asarray(geometry_coordinates)
        )
        or any(
            not np.array_equal(np.asarray(route), np.asarray(block.vertices))
            for route, block in zip(geometry_routes, mesh.blocks, strict=True)
        )
        or any(
            not np.array_equal(np.asarray(route), np.asarray(block.vertices))
            for route, block in zip(result_routes, result.mesh.blocks, strict=True)
        )
    ):
        raise ValueError(
            "CellGeometrySpec coordinates or routes differ from the certified mesh."
        )
    periodic_source = entries["mesh/periodicity"].value
    if not isinstance(periodic_source, PeriodicMeshTopology):
        raise TypeError("Composition periodicity entry must be PeriodicMeshTopology.")
    lifted_ids = {
        degree: ids
        for degree, ids in enumerate(periodic_source.lifted_entity_ids)
        if degree > 0
    }
    lifted_mesh = CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        entity_global_ids=lifted_ids,
        numeric_version=mesh.numeric_version,
    )
    periodic = _rebuild_periodic_topology(
        periodic_source,
        lifted_mesh,
        actual_geometry=(
            geometry
            if periodic_source._actual_geometry is not None
            or periodic_source.coordinate_source_map_id is not None
            else None
        ),
    )
    if (
        mesh.periodic_topology is None
        or mesh.periodic_topology.periodic_topology_id != periodic.periodic_topology_id
        or result.mesh.periodic_topology is None
        or result.mesh.periodic_topology.periodic_topology_id
        != periodic.periodic_topology_id
    ):
        raise ValueError("Composition periodicity entry does not match its mesh result.")
    rebuilt_values = {
        "mesh/result": result,
        "mesh/topology": mesh,
        "mesh/geometry": geometry,
        "mesh/periodicity": periodic,
        "material/root-mesh": volume_root_mesh,
        "material/root-geometry": volume_root_geometry,
        "material/refined-mesh": volume_refined_mesh,
        "material/refined-geometry": volume_refined_geometry,
        "material/reference-declaration": material_reference,
        "material/refinement-deformation": material_deformations[0],
        "material/coarsening-deformation": material_deformations[1],
        "material/volume-chain": volume_chain,
    }
    rebuilt_values.update(
        {
            entry.entry_id: _rebuild_continuation_value(entry.value)
            for entry in value.entries
            if entry.entry_id not in rebuilt_values
        }
    )
    source = rebuilt_values["cad/source"]
    state = rebuilt_values["solver/state"]
    materials = rebuilt_values["solver/materials"]
    distribution = rebuilt_values["mesh/distribution"]
    if (
        not isinstance(source, BRepModel)
        or not isinstance(state, FiniteElementAcceptedState)
        or not isinstance(materials, MaterialTransaction)
        or not isinstance(distribution, MeshDistribution)
    ):
        raise TypeError("Composition state entries did not rebuild their owning records.")
    state_materials = state.materials
    if state_materials is None:
        raise ValueError("Accepted finite-element state lost its material transaction.")
    if (
        state_materials.transaction_id != materials.transaction_id
        or not isinstance(distribution.part.carrier, CellMeshingResult)
        or distribution.part.carrier.result_id != result.result_id
    ):
        raise ValueError(
            "Accepted material or distribution state is stale against the mesh result."
        )
    identities = (
        ("cad/source", source.geometry.geometry_id, source.source_revision),
        ("mesh/topology", mesh.topology_id, mesh.mesh_id),
        ("mesh/geometry", geometry.geometry_layout_id, mesh.geometry_id),
        ("mesh/result", result.mesh.topology_id, result.result_id),
        ("solver/state", state.topology_id, state.accepted_id),
        ("solver/materials", materials.layout_id, materials.transaction_id),
        (
            "mesh/periodicity",
            periodic.periodic_topology_id,
            periodic.periodic_topology_id,
        ),
        (
            "mesh/distribution",
            distribution.part.part_id,
            distribution.distribution_id,
        ),
        (
            "material/root-mesh",
            volume_root_mesh.topology_id,
            volume_root_mesh.mesh_id,
        ),
        (
            "material/root-geometry",
            volume_root_geometry.geometry_layout_id,
            cell_geometry_id(volume_root_geometry),
        ),
        (
            "material/refined-mesh",
            volume_refined_mesh.topology_id,
            volume_refined_mesh.mesh_id,
        ),
        (
            "material/refined-geometry",
            volume_refined_geometry.geometry_layout_id,
            cell_geometry_id(volume_refined_geometry),
        ),
        (
            "material/reference-declaration",
            volume_root_geometry.geometry_layout_id,
            material_reference.declaration_id,
        ),
        (
            "material/refinement-deformation",
            material_reference.declaration_id,
            material_deformations[0].deformation_id,
        ),
        (
            "material/coarsening-deformation",
            material_reference.declaration_id,
            material_deformations[1].deformation_id,
        ),
        (
            "material/volume-chain",
            volume_chain.source_geometry_id,
            volume_chain.chain_id,
        ),
    )
    if any(
        entries[entry_id].structure_id != structure_id
        or entries[entry_id].revision_id != revision_id
        for entry_id, structure_id, revision_id in identities
    ):
        raise ValueError(
            "Composition entry bindings differ from recomputed science identities."
        )
    expected_material_metadata = (
        (
            "material/root-mesh",
            "topology",
            "phydrax.discretization",
            "curved-material-root-mesh",
        ),
        (
            "material/root-geometry",
            "discretization",
            "phydrax.discretization",
            "curved-material-root-coordinate-map",
        ),
        (
            "material/refined-mesh",
            "topology",
            "phydrax.discretization",
            "curved-material-refined-mesh",
        ),
        (
            "material/refined-geometry",
            "discretization",
            "phydrax.discretization",
            "curved-material-refined-coordinate-map",
        ),
        (
            "material/reference-declaration",
            "transfer",
            "phydrax.geometry",
            "material-reference-configuration",
        ),
        (
            "material/refinement-deformation",
            "transfer",
            "phydrax.geometry",
            "material-reference-deformation",
        ),
        (
            "material/coarsening-deformation",
            "transfer",
            "phydrax.geometry",
            "material-reference-deformation",
        ),
        (
            "material/volume-chain",
            "transfer",
            "phydrax.geometry",
            "curved-volume-material-chain",
        ),
    )
    if any(
        (
            entries[entry_id].role,
            entries[entry_id].owner_id,
            entries[entry_id].semantics_id,
        )
        != (role, owner, semantics)
        for entry_id, role, owner, semantics in expected_material_metadata
    ):
        raise ValueError("Curved material composition role or owner metadata changed.")
    expected_material_dependencies = (
        (
            "material/root-geometry",
            (
                entries["material/root-mesh"].binding("structure"),
                entries["material/root-mesh"].binding("revision"),
            ),
        ),
        (
            "material/refined-geometry",
            (
                entries["material/refined-mesh"].binding("structure"),
                entries["material/refined-mesh"].binding("revision"),
            ),
        ),
        (
            "material/reference-declaration",
            (
                entries["material/root-mesh"].binding("structure"),
                entries["material/root-mesh"].binding("revision"),
                entries["material/root-geometry"].binding("structure"),
                entries["material/root-geometry"].binding("revision"),
            ),
        ),
        (
            "material/refinement-deformation",
            (
                entries["material/reference-declaration"].binding("revision"),
                entries["material/root-mesh"].binding("structure"),
                entries["material/root-mesh"].binding("revision"),
                entries["material/root-geometry"].binding("structure"),
                entries["material/root-geometry"].binding("revision"),
                entries["material/refined-mesh"].binding("structure"),
                entries["material/refined-mesh"].binding("revision"),
                entries["material/refined-geometry"].binding("structure"),
                entries["material/refined-geometry"].binding("revision"),
            ),
        ),
        (
            "material/coarsening-deformation",
            (
                entries["material/reference-declaration"].binding("revision"),
                entries["material/refined-mesh"].binding("structure"),
                entries["material/refined-mesh"].binding("revision"),
                entries["material/refined-geometry"].binding("structure"),
                entries["material/refined-geometry"].binding("revision"),
                entries["material/root-mesh"].binding("structure"),
                entries["material/root-mesh"].binding("revision"),
                entries["material/root-geometry"].binding("structure"),
                entries["material/root-geometry"].binding("revision"),
            ),
        ),
        (
            "material/volume-chain",
            (
                entries["material/root-mesh"].binding("structure"),
                entries["material/root-mesh"].binding("revision"),
                entries["material/root-geometry"].binding("structure"),
                entries["material/root-geometry"].binding("revision"),
                entries["material/refined-mesh"].binding("structure"),
                entries["material/refined-mesh"].binding("revision"),
                entries["material/refined-geometry"].binding("structure"),
                entries["material/refined-geometry"].binding("revision"),
                entries["material/reference-declaration"].binding("revision"),
                entries["material/refinement-deformation"].binding("revision"),
                entries["material/coarsening-deformation"].binding("revision"),
            ),
        ),
    )
    if any(
        entries[entry_id].dependencies != _canonical_dependencies(dependencies)
        for entry_id, dependencies in expected_material_dependencies
    ):
        raise ValueError(
            "Curved material dependency bindings differ from their endpoints."
        )
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
                CompositionDependency(
                    dependency.entry_id, dependency.facet, dependency.bound_id
                )
                for dependency in entry.dependencies
            ),
        )
        for entry in value.entries
    )
    return Composition(rebuilt_entries, boundary_id=value.boundary_id)


def _rebuild_layer_curving_composition(value: object, /) -> Composition:
    if not isinstance(value, Composition) or set(value.entry_ids) != {
        "layer/mesh",
        "layer/geometry",
        "layer/result",
    }:
        raise TypeError(
            "Layer-curving archive requires its mesh, geometry, and accepted result entries."
        )
    entries = {entry.entry_id: entry for entry in value.entries}
    mesh = _rebuild_continuation_value(entries["layer/mesh"].value)
    if not isinstance(mesh, CellMesh):
        raise TypeError("Layer-curving mesh entry must rebuild a CellMesh.")
    geometry = _rebuild_geometry_for_mesh(entries["layer/geometry"].value, mesh)
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Layer-curving geometry entry must rebuild a CellGeometrySpec.")
    result = _rebuild_continuation_value(entries["layer/result"].value)
    if not isinstance(result, CellMeshingResult) or result.layer_curving_evidence is None:
        raise TypeError("Layer-curving result lost its accepted source evidence.")
    if (
        result.mesh.mesh_id != mesh.mesh_id
        or result.mesh.topology_id != mesh.topology_id
        or result.geometry.geometry_layout_id != geometry.geometry_layout_id
        or cell_geometry_id(result.geometry) != cell_geometry_id(geometry)
    ):
        raise ValueError(
            "Layer-curving result differs from its independent mesh/geometry entries."
        )
    identities = (
        ("layer/mesh", mesh.topology_id, mesh.mesh_id),
        ("layer/geometry", geometry.geometry_layout_id, cell_geometry_id(geometry)),
        ("layer/result", result.mesh.topology_id, result.result_id),
    )
    if any(
        entries[entry_id].structure_id != structure_id
        or entries[entry_id].revision_id != revision_id
        for entry_id, structure_id, revision_id in identities
    ):
        raise ValueError(
            "Layer-curving composition identities differ from recomputed owners."
        )
    metadata = (
        ("layer/mesh", "topology", "phydrax.discretization", "layer-curving-target-mesh"),
        (
            "layer/geometry",
            "discretization",
            "phydrax.discretization",
            "layer-curving-target-coordinate-map",
        ),
        (
            "layer/result",
            "discretization",
            "phydrax.meshing",
            "layer-curving-result",
        ),
    )
    if any(
        (
            entries[entry_id].role,
            entries[entry_id].owner_id,
            entries[entry_id].semantics_id,
        )
        != (role, owner, semantics)
        for entry_id, role, owner, semantics in metadata
    ):
        raise ValueError("Layer-curving composition role or owner metadata changed.")
    dependencies = (
        (
            "layer/geometry",
            (
                entries["layer/mesh"].binding("structure"),
                entries["layer/mesh"].binding("revision"),
            ),
        ),
        (
            "layer/result",
            (
                entries["layer/mesh"].binding("structure"),
                entries["layer/mesh"].binding("revision"),
                entries["layer/geometry"].binding("structure"),
                entries["layer/geometry"].binding("revision"),
            ),
        ),
    )
    if any(
        entries[entry_id].dependencies != _canonical_dependencies(expected)
        for entry_id, expected in dependencies
    ):
        raise ValueError("Layer-curving result dependencies differ from endpoint owners.")
    values = {
        "layer/mesh": mesh,
        "layer/geometry": geometry,
        "layer/result": result,
    }
    rebuilt_entries = tuple(
        CompositionEntry(
            values[entry.entry_id],
            entry_id=entry.entry_id,
            role=entry.role,
            owner_id=entry.owner_id,
            structure_id=entry.structure_id,
            revision_id=entry.revision_id,
            semantics_id=entry.semantics_id,
            dependencies=tuple(
                CompositionDependency(
                    dependency.entry_id, dependency.facet, dependency.bound_id
                )
                for dependency in entry.dependencies
            ),
        )
        for entry in value.entries
    )
    return Composition(rebuilt_entries, boundary_id=value.boundary_id)


def _write_layer_curving_evidence(directory: Path, /) -> None:
    _write_layer_curving_archive_case(
        directory,
        make_layer_curving_archive_case(),
        "layer-curving-source-evidence.phx",
        "layer-curving:archive",
    )


def _write_mapped_layer_curving_evidence(directory: Path, /) -> None:
    _write_layer_curving_archive_case(
        directory,
        make_mapped_layer_curving_archive_case(),
        "mapped-layer-curving-source-evidence.phx",
        "mapped-layer-curving:archive",
    )


def _write_restricted_layer_curving_evidence(directory: Path, /) -> None:
    from tests._support.layer_curving_fixtures import (
        make_restricted_layer_curving_archive_case,
    )

    _write_layer_curving_archive_case(
        directory,
        make_restricted_layer_curving_archive_case(),
        "restricted-layer-curving-source-evidence.phx",
        "restricted-layer-curving:archive",
    )


def _write_implicit_layer_curving_evidence(directory: Path, /) -> None:
    from tests._support.implicit_layer_curving_fixtures import (
        make_implicit_layer_curving_archive_case,
    )

    _write_layer_curving_archive_case(
        directory,
        make_implicit_layer_curving_archive_case(),
        "implicit-layer-curving-source-evidence.phx",
        "implicit-layer-curving:archive",
    )


def _fraction_pair_for_archive(value: Fraction, /) -> list[int]:
    if not isinstance(value, Fraction):
        raise TypeError("Archive fixture requires an exact rational coefficient.")
    return [value.numerator, value.denominator]


def _fraction_rows_for_archive(
    rows: tuple[tuple[Fraction, ...], ...], /
) -> list[list[list[int]]]:
    return [[_fraction_pair_for_archive(value) for value in row] for row in rows]


def _exact_scalar_pair_for_archive(value: Fraction | RadicalElement, /) -> list[object]:
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if not isinstance(value, RadicalElement):
        raise TypeError("Archive fixture requires an exact algebraic coefficient.")
    if type(value) is not RadicalElement:
        raise TypeError("Archive fixture requires the canonical RadicalElement owner.")
    radical = RadicalElement.from_stored_fields(value)
    if not radical.terms:
        return [0, 1]
    if len(radical.terms) == 1 and radical.terms[0][0] == 1:
        coefficient = radical.terms[0][1]
        return [coefficient.numerator, coefficient.denominator]
    terms = [
        [radicand, coefficient.numerator, coefficient.denominator]
        for radicand, coefficient in radical.terms
    ]
    payload: list[object] = ["radical", terms]
    return payload


def _exact_scalar_rows_for_archive(
    rows: tuple[tuple[Fraction | RadicalElement, ...], ...], /
) -> list[list[list[object]]]:
    return [[_exact_scalar_pair_for_archive(value) for value in row] for row in rows]


def _tspline_mesh_archive_record(mesh: TMesh2D, /) -> dict[str, object]:
    return {
        "mesh_id": mesh.mesh_id,
        "degree": list(mesh.degree),
        "parameter_domain": [
            _fraction_pair_for_archive(value) for value in mesh.parameter_domain
        ],
        "vertices": [
            [
                vertex.vertex_id,
                [_fraction_pair_for_archive(value) for value in vertex.parameter],
            ]
            for vertex in mesh.vertices
        ],
        "edges": [
            [edge.edge_id, list(edge.vertex_ids), list(edge.cell_ids), edge.axis]
            for edge in mesh.edges
        ],
        "cells": [
            [
                cell.cell_id,
                [_fraction_pair_for_archive(value) for value in cell.parameter_bounds],
                list(cell.vertex_ids),
                list(cell.edge_ids),
                cell.level,
                cell.parent_id,
                list(cell.refinement_address),
            ]
            for cell in mesh.cells
        ],
        "anchors": [
            [
                anchor.anchor_id,
                [_fraction_pair_for_archive(value) for value in anchor.parameter],
                [_fraction_pair_for_archive(value) for value in anchor.local_knots.u],
                [_fraction_pair_for_archive(value) for value in anchor.local_knots.v],
            ]
            for anchor in mesh.anchors
        ],
        "t_junction_extensions": [
            [
                extension.junction_vertex_id,
                extension.axis,
                extension.missing_direction,
                list(extension.segment),
                extension.backward_crossings,
                extension.forward_crossings,
            ]
            for extension in mesh.t_junction_extensions
        ],
        "cell_lineage": [
            [cell_id, list(address)] for cell_id, address in mesh.cell_lineage
        ],
        "reserved_cell_ids": list(mesh.reserved_cell_ids),
        "reserved_anchor_ids": list(mesh.reserved_anchor_ids),
    }


def _surface_root_binding_tokens(witness: SurfaceChartWitness, /) -> list[object]:
    return [
        None
        if binding is None
        else [
            binding.root_geometry_id,
            binding.root_topology_id,
            binding.root_cell_global_id,
        ]
        for binding in witness.root_mesh_cell_bindings
    ]


def _mapped_reference_map_bindings_for_archive(
    reference_maps: tuple[ReferenceCellMapWitnesses, ...], /
) -> list[list[list[object]]]:
    from phydrax.discretization._nested_reference import nested_cell_reference_maps
    from phydrax.discretization._nested_reference_types import (
        reference_chart_arguments_identity,
    )

    return [
        [
            [
                mapping.target_cell_id,
                mapping.source_cell_id,
                None
                if mapping.arguments is None
                else reference_chart_arguments_identity((mapping.arguments,)),
            ]
            for mapping in nested_cell_reference_maps(reference_map)
        ]
        for reference_map in reference_maps
    ]


def _write_layer_curving_archive_case(
    directory: Path,
    curving_result: LayerCurvingConstructionResult,
    archive_name: str,
    boundary_id: str,
    /,
) -> None:
    from phydrax.meshing._layer_curving import LayerCurvingConstructionResult

    if type(curving_result) is not LayerCurvingConstructionResult:
        raise TypeError(
            "Layer archive fixture must retain its canonical construction result."
        )
    mesh = curving_result.construction.mesh
    geometry = curving_result.candidate_geometry
    evidence = curving_result.source_evidence
    contract = SpatialCoordinateContract.si()
    if not isinstance(geometry, CellGeometrySpec) or evidence is None:
        raise TypeError(
            "The layer-curving fixture must retain accepted geometry and source evidence."
        )
    base_result = certify_cell_mesh(mesh, contract, geometry=geometry)
    result = CellMeshingResult(
        base_result.mesh,
        base_result.geometry,
        base_result.coordinate_contract,
        base_result.audit,
        base_result.quality,
        base_result.compliance,
        base_result.trace,
        base_result.provider,
        base_result.runtime,
        base_result.derivative_mode,
        base_result.provenance,
        boundary=base_result.boundary,
        patches=base_result.patches,
        zones=base_result.zones,
        labels=base_result.labels,
        attributes=base_result.attributes,
        associations=base_result.associations,
        adapter_reports=base_result.adapter_reports,
        surface_source=base_result.surface_source,
        region_boundary_evidence=base_result.region_boundary_evidence,
        planar_embedding=base_result.planar_embedding,
        certification=base_result.certification,
        layer_curving_evidence=evidence,
    )
    mesh = result.mesh
    geometry = result.geometry
    evidence = result.layer_curving_evidence
    if evidence is None:
        raise TypeError("The certified result lost its layer-curving source evidence.")
    mesh_entry = CompositionEntry(
        mesh,
        entry_id="layer/mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=mesh.topology_id,
        revision_id=mesh.mesh_id,
        semantics_id="layer-curving-target-mesh",
    )
    geometry_entry = CompositionEntry(
        geometry,
        entry_id="layer/geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=geometry.geometry_layout_id,
        revision_id=cell_geometry_id(geometry),
        semantics_id="layer-curving-target-coordinate-map",
        dependencies=(
            mesh_entry.binding("structure"),
            mesh_entry.binding("revision"),
        ),
    )
    result_entry = CompositionEntry(
        result,
        entry_id="layer/result",
        role="discretization",
        owner_id="phydrax.meshing",
        structure_id=mesh.topology_id,
        revision_id=result.result_id,
        semantics_id="layer-curving-result",
        dependencies=(
            mesh_entry.binding("structure"),
            mesh_entry.binding("revision"),
            geometry_entry.binding("structure"),
            geometry_entry.binding("revision"),
        ),
    )
    composition = Composition(
        (mesh_entry, geometry_entry, result_entry),
        boundary_id=boundary_id,
    )
    path = directory / archive_name
    write_model_artifact(path, composition, limits=None)
    print(
        json.dumps(
            _layer_curving_archive_state(composition, mesh, geometry, result, evidence),
            sort_keys=True,
        )
    )


def _layer_curving_archive_state(
    composition: Composition,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    result: CellMeshingResult,
    evidence: LayerCurvingSourceEvidence,
    /,
) -> dict[str, object]:
    """JSON-normalized writer/reader state of one layer-curving archive."""
    mapped_trace = evidence.mapped_chart_trace
    mapped_reference_map_bindings = (
        None
        if not evidence.mapped_reference_maps
        else _mapped_reference_map_bindings_for_archive(evidence.mapped_reference_maps)
    )
    state = _json_payload(
        {
            "composition_id": composition.composition_id,
            "cell_result_id": result.result_id,
            "mesh_id": mesh.mesh_id,
            "geometry_id": cell_geometry_id(geometry),
            "source_id": evidence.source_id,
            "source_revision": evidence.source_revision,
            "frame_id": evidence.frame_id,
            "source_chart_id": evidence.source_chart_id,
            "layer_result_id": evidence.layer_result_id,
            "layer_evidence_id": evidence.layer_evidence.evidence_id,
            "evidence_id": evidence.evidence_id,
            "fidelity_certificate_id": evidence.fidelity.certificate_id,
            "trace_id": (
                None
                if evidence.projection_trace is None
                else evidence.projection_trace.trace_id
            ),
            "chart_witness_rows": (
                None
                if evidence.projection_trace is None
                else evidence.projection_trace.chart_witness_rows.tolist()
            ),
            "geometry_node_rows": (
                None
                if evidence.projection_trace is None
                else evidence.projection_trace.geometry_node_rows.tolist()
            ),
            "target_reference_points": (
                None
                if evidence.projection_trace is None
                else _exact_scalar_rows_for_archive(
                    evidence.projection_trace.target_reference_points
                )
            ),
            "mapped_trace_id": (None if mapped_trace is None else mapped_trace.trace_id),
            "mapped_query_id": (None if mapped_trace is None else mapped_trace.query_id),
            "mapped_geometry_node_rows": (
                None if mapped_trace is None else mapped_trace.geometry_node_rows.tolist()
            ),
            "mapped_chart_witness_rows": (
                None if mapped_trace is None else mapped_trace.chart_witness_rows.tolist()
            ),
            "mapped_target_reference_points": (
                None
                if mapped_trace is None
                else _exact_scalar_rows_for_archive(mapped_trace.target_reference_points)
            ),
            "mapped_root_reference_points": (
                None
                if mapped_trace is None
                else _exact_scalar_rows_for_archive(mapped_trace.root_reference_points)
            ),
            "mapped_source_parameters": (
                None
                if mapped_trace is None
                else _exact_scalar_rows_for_archive(mapped_trace.source_parameters)
            ),
            "mapped_source_chart_ids": (
                None if mapped_trace is None else list(mapped_trace.source_chart_ids)
            ),
            "mapped_source_entity_ids": (
                None if mapped_trace is None else list(mapped_trace.source_entity_ids)
            ),
            "mapped_occurrence_paths": (
                None
                if mapped_trace is None
                else [list(path) for path in mapped_trace.occurrence_paths]
            ),
            "mapped_root_face_global_ids": (
                None
                if mapped_trace is None
                else np.asarray(mapped_trace.root_face_global_ids).tolist()
            ),
            "mapped_root_cell_global_ids": (
                None
                if mapped_trace is None
                else np.asarray(mapped_trace.root_cell_global_ids).tolist()
            ),
            "mapped_root_local_face_indices": (
                None
                if mapped_trace is None
                else np.asarray(mapped_trace.root_local_face_indices).tolist()
            ),
            "mapped_target_points": (
                None
                if mapped_trace is None
                else np.asarray(mapped_trace.target_points).tolist()
            ),
            "mapped_source_points": (
                None
                if mapped_trace is None
                else np.asarray(mapped_trace.source_points).tolist()
            ),
            "mapped_distance_lower_bounds": (
                None
                if mapped_trace is None
                else np.asarray(mapped_trace.distance_lower_bounds).tolist()
            ),
            "mapped_distance_upper_bounds": (
                None
                if mapped_trace is None
                else np.asarray(mapped_trace.distance_upper_bounds).tolist()
            ),
            "mapped_normals": (
                None
                if mapped_trace is None
                else np.asarray(mapped_trace.normals).tolist()
            ),
            "mapped_tangents": (
                None
                if mapped_trace is None
                else np.asarray(mapped_trace.tangents).tolist()
            ),
            "mapped_column_identity_id": (
                None if mapped_trace is None else mapped_trace.column_identity.identity_id
            ),
            "mapped_reference_map_bindings": mapped_reference_map_bindings,
            "implicit_trace_id": (
                None
                if evidence.implicit_root_trace is None
                else evidence.implicit_root_trace.trace_id
            ),
            "implicit_map_witness_id": evidence.implicit_map_witness_id,
            "chart_witness_id": evidence.chart_witness.witness_id,
            "chart_target_reference_tokens": _json_payload(
                evidence.chart_witness.target_reference_rational_tokens
            ),
            "chart_source_parameter_tokens": _json_payload(
                evidence.chart_witness.source_parameter_rational_tokens
            ),
            "column_identity_id": evidence.column_identity.identity_id,
            "chart_column_identity_id": evidence.chart_column_identity.identity_id,
            "chart_column_row_count": evidence.chart_column_identity.row_count,
            "source_root_cell_ids": evidence.source_root_cell_ids.tolist(),
            "source_piece_ids": [list(row) for row in evidence.source_piece_ids],
            "source_wall_cell_ids": evidence.column_identity.source_wall_cell_ids.tolist(),
            "front_row_identities": list(evidence.column_identity.front_row_identities),
            "implicit_root_chart_witness_id": (
                None
                if evidence.implicit_root_chart_witness is None
                else evidence.implicit_root_chart_witness.witness_id
            ),
            "implicit_root_cell_vertex_global_ids": (
                None
                if evidence.implicit_root_cell_vertex_global_ids is None
                else np.asarray(evidence.implicit_root_cell_vertex_global_ids).tolist()
            ),
        }
    )
    if not isinstance(state, dict):
        raise TypeError("Layer-curving archive state must be a JSON object.")
    return state


def _read_layer_curving_evidence(path: Path, expected: dict[str, object], /) -> None:
    composition = read_model_artifact(
        path,
        expected_type=Composition,
        restore=_rebuild_layer_curving_composition,
        limits=None,
    )
    mesh = composition.value("layer/mesh")
    geometry = composition.value("layer/geometry")
    result = composition.value("layer/result")
    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Layer-curving cold restore lost its endpoint owners.")
    if not isinstance(result, CellMeshingResult) or result.layer_curving_evidence is None:
        raise TypeError("Layer-curving cold restore lost its accepted result evidence.")
    evidence = result.layer_curving_evidence
    state = _layer_curving_archive_state(composition, mesh, geometry, result, evidence)
    if state != expected:
        changed = sorted(
            name
            for name in set(state) | set(expected)
            if state.get(name) != expected.get(name)
        )
        raise ValueError(
            f"Layer-curving cold restore differs from its writer evidence: {changed}."
        )
    evidence.revalidate(mesh, geometry)
    print("layer-curving-evidence-cold-restored")


def _read_restricted_layer_curving_evidence(
    path: Path, expected: dict[str, object], /
) -> None:
    _read_layer_curving_evidence(path, expected)


def _material_reference_seed_transition() -> CellMeshTransition:
    from phydrax.meshing._lineage import (
        CellMeshTransition,
        EntityLineage,
        EntityLineageKind,
        MeshLineage,
        MeshTransitionKind,
    )

    root_mesh, root_geometry, *_ = make_curved_material_refinement()
    result = certify_cell_mesh(
        root_mesh,
        SpatialCoordinateContract.si(),
        geometry=root_geometry,
    )
    root_mesh = result.mesh
    root_geometry = result.geometry
    domain, root_coverage, _ = make_material_coverage_pair(
        root_mesh,
        root_geometry,
        root_mesh,
        root_geometry,
        region_id="aluminum",
        occurrence_path=("assembly-1", "occurrence-9"),
    )
    source_map = make_material_source(domain)
    reference = MaterialReferenceDeclaration(
        root_mesh,
        root_geometry,
        source_map,
        ("solid-cell",),
        ("aluminum",),
        root_coverage,
        frame_id="material-frame-seed",
        policy=MaterialReferencePolicy.ACCEPTED_CONFIGURATION,
        occurrence_paths=(("assembly-1", "occurrence-9"),),
    )
    lineages = []
    for dimension in range(root_mesh.topological_dimension + 1):
        entity_set = root_mesh.entity_set(dimension)
        identifiers = np.asarray(entity_set.entity_ids, dtype=np.int64)
        lineages.append(
            EntityLineage(
                dimension,
                entity_set.entity_set_id,
                entity_set.entity_set_id,
                identifiers,
                identifiers,
                np.full(
                    identifiers.shape,
                    int(EntityLineageKind.PRESERVED),
                    dtype=np.int32,
                ),
            )
        )
    lineage = MeshLineage(root_mesh.topology_id, root_mesh.topology_id, tuple(lineages))
    transition = CellMeshTransition(
        root_mesh.mesh_id,
        root_mesh.topology_id,
        result,
        lineage,
        MeshTransitionKind.REMESH,
        material_reference=reference,
    )
    if transition.volume_material_chain is not None:
        raise ValueError("Seed transition unexpectedly contains material maps.")
    return transition


def _rebuild_material_reference_seed_composition(value: object, /) -> Composition:
    from phydrax.meshing._lineage import CellMeshTransition

    if not isinstance(value, Composition) or value.entry_ids != ("transition/seed",):
        raise TypeError(
            "Material-reference seed archive requires its Composition transition entry."
        )
    entry = value.entries[0]
    transition = _rebuild_continuation_value(entry.value)
    if not isinstance(transition, CellMeshTransition):
        raise TypeError(
            "Material-reference seed entry must rebuild a CellMeshTransition."
        )
    material_reference = transition.material_reference
    if material_reference is None:
        raise ValueError("Seed transition lost its material reference declaration.")
    if transition.volume_material_chain is not None:
        raise ValueError("Seed transition unexpectedly acquired material maps.")
    if (
        entry.structure_id != transition.target.mesh.topology_id
        or entry.revision_id != transition.transition_id
        or entry.role != "transfer"
        or entry.owner_id != "phydrax.meshing"
        or entry.semantics_id != "material-reference-seed-transition"
        or entry.dependencies
    ):
        raise ValueError("Material-reference seed composition binding changed.")
    rebuilt_entry = CompositionEntry(
        transition,
        entry_id=entry.entry_id,
        role=entry.role,
        owner_id=entry.owner_id,
        structure_id=entry.structure_id,
        revision_id=entry.revision_id,
        semantics_id=entry.semantics_id,
        dependencies=(),
    )
    return Composition((rebuilt_entry,), boundary_id=value.boundary_id)


def _write_material_reference_seed_transition(directory: Path, /) -> None:
    transition = _material_reference_seed_transition()
    material_reference = transition.material_reference
    if material_reference is None:
        raise ValueError("Seed transition omitted its material reference declaration.")
    entry = CompositionEntry(
        transition,
        entry_id="transition/seed",
        role="transfer",
        owner_id="phydrax.meshing",
        structure_id=transition.target.mesh.topology_id,
        revision_id=transition.transition_id,
        semantics_id="material-reference-seed-transition",
    )
    composition = Composition((entry,), boundary_id="material-reference-seed:archive")
    path = directory / "material-reference-seed-transition.phx"
    write_model_artifact(path, composition, limits=None)
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "transition_id": transition.transition_id,
                "material_reference_id": material_reference.declaration_id,
                "root_mesh_id": material_reference.root_mesh.mesh_id,
                "target_result_id": transition.target.result_id,
                "lineage_id": transition.lineage.lineage_id,
                "volume_material_chain": None,
            }
        )
    )


def _read_material_reference_seed_transition(
    path: Path, expected: dict[str, object], /
) -> None:
    from phydrax.meshing._lineage import CellMeshTransition

    composition = read_model_artifact(
        path,
        expected_type=Composition,
        restore=_rebuild_material_reference_seed_composition,
        limits=None,
    )
    transition = composition.value("transition/seed")
    if not isinstance(transition, CellMeshTransition):
        raise TypeError("Seed transition archive lost its CellMeshTransition entry.")
    if (
        transition.material_reference is None
        or transition.volume_material_chain is not None
    ):
        raise ValueError(
            "Cold seed transition lost its declaration or invented material maps."
        )
    if (
        composition.composition_id != expected["composition_id"]
        or transition.transition_id != expected["transition_id"]
        or transition.material_reference.declaration_id
        != expected["material_reference_id"]
        or transition.material_reference.root_mesh.mesh_id != expected["root_mesh_id"]
        or transition.target.result_id != expected["target_result_id"]
        or transition.lineage.lineage_id != expected["lineage_id"]
        or transition.source_mesh_id != transition.material_reference.root_mesh.mesh_id
        or transition.target.mesh.mesh_id
        != transition.material_reference.root_mesh.mesh_id
    ):
        raise ValueError(
            "Cold seed transition differs from its independent material reference owner."
        )
    print("material-reference-seed-transition-cold-restored")


def _rebuild_adaptive_implicit_composition(value: object, /) -> Composition:
    if not isinstance(value, Composition) or len(value.entries) != 1:
        raise TypeError(
            "Adaptive implicit archive requires one Composition source entry."
        )
    entry = value.entries[0]
    if entry.entry_id != "geometry/adaptive-implicit-domain":
        raise ValueError("Adaptive implicit archive entry identity changed.")
    stored_recipe = entry.value
    if not isinstance(stored_recipe, AdaptiveImplicitSurfaceArchiveRecipe):
        raise TypeError("Adaptive implicit archive entry must retain its source recipe.")
    recipe, surface = _rebuild_adaptive_implicit_source(stored_recipe)
    if (
        entry.role != "topology"
        or entry.owner_id != "phydrax.geometry.implicit"
        or entry.structure_id != surface.domain_id
        or entry.revision_id != _adaptive_implicit_surface_revision(surface)
        or entry.semantics_id != "adaptive-implicit-certification-domain-source"
        or entry.dependencies
    ):
        raise ValueError(
            "Adaptive implicit source binding differs from recomputed proof data."
        )
    kernel = surface.geometry.kernel
    if not isinstance(kernel, _BallKernel):
        raise TypeError(
            "Adaptive archive classification requires its sphere source kernel."
        )
    center = np.asarray(
        surface.geometry.state.value(kernel.center), dtype=np.float64
    ).reshape((3,))
    classification = surface.volume.classify_points(
        jnp.asarray(
            np.stack((center, np.zeros((3,), dtype=np.float64))), dtype=jnp.float64
        )
    )
    if not np.array_equal(
        np.asarray(classification.classes),
        np.asarray((int(ImplicitVolumeClass.INSIDE), int(ImplicitVolumeClass.OUTSIDE))),
    ):
        raise ValueError(
            "Cold adaptive implicit source no longer supports its volume query."
        )
    rebuilt_entry = CompositionEntry(
        recipe,
        entry_id=entry.entry_id,
        role=entry.role,
        owner_id=entry.owner_id,
        structure_id=entry.structure_id,
        revision_id=entry.revision_id,
        semantics_id=entry.semantics_id,
        dependencies=(),
    )
    return Composition((rebuilt_entry,), boundary_id=value.boundary_id)


def _rebuild_reconstruction_report(
    value: ReconstructionReport, /
) -> ReconstructionReport:
    if type(value) is not ReconstructionReport:
        raise TypeError("Reconstruction provenance must retain its exact report type.")
    return ReconstructionReport(
        source_kind=value.source_kind,
        algorithm=value.algorithm,
        input_digest=value.input_digest,
        input_points=value.input_points,
        retained_points=value.retained_points,
        output_vertices=value.output_vertices,
        output_cells=value.output_cells,
        connected_components=value.connected_components,
        watertight=value.watertight,
        winding_consistent=value.winding_consistent,
        recenter_offset=value.recenter_offset,
        parameters=value.parameters,
        warnings=value.warnings,
        source_product_id=value.source_product_id,
        euler_characteristic=value.euler_characteristic,
        normal_evidence=value.normal_evidence,
        poisson_evidence=value.poisson_evidence,
        deviation_evidence=value.deviation_evidence,
        outlier_evidence=value.outlier_evidence,
        thin_feature_evidence=value.thin_feature_evidence,
        component_evidence=value.component_evidence,
        coverage_evidence=value.coverage_evidence,
        repair_interpretation=value.repair_interpretation,
    )


def _rebuild_reconstructed_source(
    value: ReconstructedGeometrySourceArchiveRecipe,
    /,
) -> tuple[ReconstructedGeometrySourceArchiveRecipe, ReconstructedGeometrySource]:
    if type(value) is not ReconstructedGeometrySourceArchiveRecipe:
        raise TypeError("Reconstructed geometry must retain its explicit archive recipe.")
    if not isinstance(value.source, MeshRegion) or type(value.source) is not MeshRegion:
        raise TypeError(
            "The native archive currently reconstructs MeshRegion sources explicitly."
        )
    stored_source = value.source
    source = MeshRegion(
        jnp.asarray(stored_source.vertices),
        jnp.asarray(stored_source.faces),
        feature_id=stored_source.feature_id,
    )
    report = _rebuild_reconstruction_report(value.report)
    rebuilt = ReconstructedGeometrySource(
        source,
        report,
        persistent_vertex_ids=value.persistent_vertex_ids,
        persistent_edge_ids=value.persistent_edge_ids,
        persistent_facet_ids=value.persistent_facet_ids,
        persistent_region_id=value.persistent_region_id,
    )
    if (
        rebuilt.source_id != value.expected_source_id
        or rebuilt.source_revision != value.expected_source_revision
        or rebuilt.persistent_region_id != value.persistent_region_id
        or not np.array_equal(rebuilt.persistent_vertex_ids, value.persistent_vertex_ids)
        or not np.array_equal(rebuilt.persistent_edge_ids, value.persistent_edge_ids)
        or not np.array_equal(rebuilt.persistent_facet_ids, value.persistent_facet_ids)
    ):
        raise ValueError("Reconstructed source identity or persistent row IDs changed.")
    return rebuilt.archive_recipe, rebuilt


def _rebuild_reconstructed_source_composition(value: object, /) -> Composition:
    if not isinstance(value, Composition) or len(value.entries) != 1:
        raise TypeError("Reconstructed source archive requires one Composition entry.")
    entry = value.entries[0]
    if entry.entry_id != "geometry/reconstructed-source":
        raise ValueError("Reconstructed source Composition entry identity changed.")
    recipe, source = _rebuild_reconstructed_source(entry.value)
    if (
        entry.role != "topology"
        or entry.owner_id != "phydrax.geometry.reconstruction"
        or entry.structure_id != source.source_id
        or entry.revision_id != source.source_revision
        or entry.semantics_id != "reconstructed-mesh-source"
        or entry.dependencies
    ):
        raise ValueError(
            "Reconstructed source entry does not match recomputed source identity."
        )
    if source.meshing_domain is None:
        raise ValueError(
            "Cold reconstructed mesh source did not rebuild its MeshingDomain."
        )
    # Persistent IDs are the domain's authored source indices; numeric scope IDs
    # are the canonical source-bank IDs that resolve back to the same rows.
    for dimension, identifiers in enumerate(
        (
            source.persistent_vertex_ids,
            source.persistent_edge_ids,
            source.persistent_facet_ids,
        )
    ):
        if source.meshing_domain.source_indices[dimension] != tuple(
            int(identifier) for identifier in identifiers
        ):
            raise ValueError("Persistent reconstructed entity ID changed its source row.")
        for row in range(identifiers.size):
            scope_id = source.scope_entity_id_for_mesh_row(dimension, row)
            if source.source_row_for_scope(dimension, scope_id) != row:
                raise ValueError(
                    "Reconstructed scope ID no longer resolves to its source row."
                )
    if not isinstance(source.source, MeshRegion):
        raise TypeError("Cold reconstructed source lost its native MeshRegion authority.")
    mesh_region = source.source
    top_count = np.asarray(mesh_region.vertices).shape[0] // 2
    inside = np.mean(np.asarray(mesh_region.vertices)[:top_count], axis=0)
    inside[2] -= 0.35
    outside = inside + np.asarray((4.0, 0.0, 0.0))
    classified = source.compile().contains(
        jnp.asarray(np.stack((inside, outside)), dtype=jnp.float64)
    )
    if not np.array_equal(np.asarray(classified), np.asarray((True, False))):
        raise ValueError(
            "Cold reconstructed geometry no longer supports its native region query."
        )
    rebuilt_entry = CompositionEntry(
        recipe,
        entry_id=entry.entry_id,
        role=entry.role,
        owner_id=entry.owner_id,
        structure_id=entry.structure_id,
        revision_id=entry.revision_id,
        semantics_id=entry.semantics_id,
        dependencies=(),
    )
    return Composition((rebuilt_entry,), boundary_id=value.boundary_id)


def _reconstructed_triangle_corner_bank(
    source: ReconstructedGeometrySource,
    domain: MeshingDomain,
    /,
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    from phydrax.geometry.reconstruction._core import _ReconstructedTrianglePatch

    if not isinstance(source.source, MeshRegion):
        raise TypeError(
            "Reconstructed triangle corners require their original MeshRegion."
        )
    vertices = np.asarray(source.source.vertices, dtype=np.float64)
    faces = np.asarray(source.source.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(domain.patches) != faces.shape[0]:
        raise ValueError(
            "Reconstructed source face rows do not align with meshing patches."
        )
    corners: list[tuple[tuple[Fraction, ...], ...]] = []
    for row, patch in enumerate(domain.patches):
        surface = patch.surface
        if type(surface) is not _ReconstructedTrianglePatch:
            raise TypeError(
                "Reconstructed MeshingDomain lost its exact triangle chart owner."
            )
        expected = tuple(
            tuple(
                Fraction.from_float(float(vertices[int(faces[row][local_index]), axis]))
                for axis in range(3)
            )
            for local_index in range(3)
        )
        if surface.exact_corners != expected:
            raise ValueError(
                "Reconstructed triangle exact corners changed authored face order."
            )
        corners.append(surface.exact_corners)
    return tuple(corners)


def _reconstructed_domain_composition(
    source: ReconstructedGeometrySource,
    domain: MeshingDomain,
    /,
) -> Composition:
    if (
        source.meshing_domain is None
        or source.meshing_domain.domain_id != domain.domain_id
    ):
        raise ValueError("Reconstructed MeshingDomain differs from its source owner.")
    source_entry = CompositionEntry(
        source.archive_recipe,
        entry_id="geometry/reconstructed-source",
        role="topology",
        owner_id="phydrax.geometry.reconstruction",
        structure_id=source.source_id,
        revision_id=source.source_revision,
        semantics_id="reconstructed-mesh-source",
    )
    domain_entry = CompositionEntry(
        domain,
        entry_id="geometry/reconstructed-domain",
        role="topology",
        owner_id="phydrax.geometry",
        structure_id=domain.domain_id,
        revision_id=domain.source_revision,
        semantics_id="reconstructed-triangle-meshing-domain",
        dependencies=(source_entry.binding("revision"),),
    )
    return Composition(
        (source_entry, domain_entry),
        boundary_id="reconstructed-source-meshing-domain",
    )


def _reconstructed_domain_metadata(
    source: ReconstructedGeometrySource,
    domain: MeshingDomain,
    composition: Composition,
    /,
) -> dict[str, object]:
    corners = _reconstructed_triangle_corner_bank(source, domain)
    return {
        "domain_composition_id": composition.composition_id,
        "domain_id": domain.domain_id,
        "domain_patch_count": len(domain.patches),
        "exact_corner_bank": [
            [
                [[coordinate.numerator, coordinate.denominator] for coordinate in point]
                for point in patch
            ]
            for patch in corners
        ],
    }


def _rebuild_reconstructed_domain_composition(
    value: object,
    expected: dict[str, object],
    /,
) -> Composition:
    expected_entries = {
        "geometry/reconstructed-source",
        "geometry/reconstructed-domain",
    }
    if not isinstance(value, Composition) or set(value.entry_ids) != expected_entries:
        raise TypeError(
            "Reconstructed domain archive requires its exact source and domain entries."
        )
    entries = {entry.entry_id: entry for entry in value.entries}
    source = _rebuild_continuation_value(entries["geometry/reconstructed-source"].value)
    domain = _rebuild_continuation_value(entries["geometry/reconstructed-domain"].value)
    if not isinstance(source, ReconstructedGeometrySource) or not isinstance(
        domain, MeshingDomain
    ):
        raise TypeError(
            "Cold reconstructed domain lost its native source or MeshingDomain."
        )
    if (
        source.meshing_domain is None
        or source.meshing_domain.domain_id != domain.domain_id
    ):
        raise ValueError(
            "Cold reconstructed domain is not derived from its retained source."
        )
    _reconstructed_triangle_corner_bank(source, domain)
    rebuilt = _reconstructed_domain_composition(source, domain)
    for stored, current in zip(value.entries, rebuilt.entries, strict=True):
        if (
            stored.entry_id != current.entry_id
            or stored.role != current.role
            or stored.owner_id != current.owner_id
            or stored.structure_id != current.structure_id
            or stored.revision_id != current.revision_id
            or stored.semantics_id != current.semantics_id
            or stored.dependencies != current.dependencies
        ):
            raise ValueError(
                "Reconstructed domain Composition binding changed on restore."
            )
    metadata = _reconstructed_domain_metadata(source, domain, rebuilt)
    if (
        metadata != {key: expected[key] for key in metadata}
        or rebuilt.composition_id != value.composition_id
    ):
        raise ValueError("Reconstructed domain exact corner bank changed on restore.")
    return rebuilt


def _write_reconstructed_source(directory: Path, /) -> None:
    recipe, source = _reconstructed_source_archive_fixture()
    entry = CompositionEntry(
        recipe,
        entry_id="geometry/reconstructed-source",
        role="topology",
        owner_id="phydrax.geometry.reconstruction",
        structure_id=source.source_id,
        revision_id=source.source_revision,
        semantics_id="reconstructed-mesh-source",
    )
    composition = Composition((entry,), boundary_id="reconstructed-source-archive")
    write_model_artifact(directory / "reconstructed-source.phx", composition, limits=None)
    domain = source.meshing_domain
    if domain is None:
        raise ValueError("Reconstructed source fixture must retain its MeshingDomain.")
    domain_composition = _reconstructed_domain_composition(source, domain)
    domain_metadata = _reconstructed_domain_metadata(source, domain, domain_composition)
    write_model_artifact(
        directory / "reconstructed-meshing-domain.phx",
        domain_composition,
        limits=None,
    )
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "persistent_region_id": source.persistent_region_id,
                "vertex_ids": source.persistent_vertex_ids.tolist(),
                "edge_ids": source.persistent_edge_ids.tolist(),
                "facet_ids": source.persistent_facet_ids.tolist(),
                **domain_metadata,
            }
        )
    )


def _read_reconstructed_source(path: Path, expected: dict[str, object], /) -> None:
    composition = read_model_artifact(
        path,
        expected_type=Composition,
        restore=_rebuild_reconstructed_source_composition,
        limits=None,
    )
    if len(composition.entries) != 1:
        raise ValueError("Cold reconstructed source Composition changed its entry count.")
    entry = composition.entries[0]
    recipe = entry.value
    if (
        composition.composition_id != expected["composition_id"]
        or entry.structure_id != expected["source_id"]
        or entry.revision_id != expected["source_revision"]
        or not isinstance(recipe, ReconstructedGeometrySourceArchiveRecipe)
        or recipe.expected_source_id != expected["source_id"]
        or recipe.expected_source_revision != expected["source_revision"]
        or recipe.persistent_region_id != expected["persistent_region_id"]
        or recipe.persistent_vertex_ids.tolist() != expected["vertex_ids"]
        or recipe.persistent_edge_ids.tolist() != expected["edge_ids"]
        or recipe.persistent_facet_ids.tolist() != expected["facet_ids"]
    ):
        raise ValueError("Cold reconstructed source identities or stable IDs changed.")
    read_model_artifact(
        path.parent / "reconstructed-meshing-domain.phx",
        expected_type=Composition,
        restore=lambda value: _rebuild_reconstructed_domain_composition(value, expected),
        limits=None,
    )
    print("reconstructed-source-cold-restored")


def _piecewise_boundary_query_fixture() -> tuple[
    ReconstructedGeometrySource,
    SurfaceEnvelopeSource,
    tuple[PiecewiseLinearBoundarySource, ...],
]:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=np.int32)
    region = MeshRegion(
        jnp.asarray(vertices),
        jnp.asarray(faces),
        feature_id="archive-envelope-tetrahedron",
    )
    report = ReconstructionReport(
        source_kind="point_cloud",
        algorithm="screened-poisson-reconstruction",
        input_digest="a" * 64,
        input_points=4,
        retained_points=4,
        output_vertices=4,
        output_cells=4,
        connected_components=1,
        watertight=True,
        winding_consistent=True,
        recenter_offset=(0.0, 0.0, 0.0),
        parameters=(("screening", "4.0"),),
    )
    original_edges = np.asarray(region.triangle_mesh.topology.edges)
    original = ReconstructedGeometrySource(
        region,
        report,
        persistent_vertex_ids=np.arange(vertices.shape[0], dtype=np.int64) + 1000,
        persistent_edge_ids=np.arange(original_edges.shape[0], dtype=np.int64) + 2000,
        persistent_facet_ids=np.arange(faces.shape[0], dtype=np.int64) + 3000,
        persistent_region_id=4000,
    )
    source = SurfaceEnvelopeSource(
        original,
        source_id=f"{original.source_id}:unsigned-distance-envelope",
        coordinate_contract=SpatialCoordinateContract.si(),
        offset=1.5,
        spacing=0.4,
        certificate_spacing=0.4,
        maximum_deviation=3.0,
    )
    queries = tuple(
        PiecewiseLinearBoundarySource(source, dimension) for dimension in (2, 1, 0)
    )
    return original, source, queries


def _rebuild_piecewise_boundary_query_composition(value: object, /) -> Composition:
    query_ids = tuple(f"source/boundary-query-{dimension}" for dimension in (2, 1, 0))
    if not isinstance(value, Composition) or len(value.entries) != 2 + len(query_ids):
        raise TypeError(
            "PWL boundary query archive requires one source and all three strata."
        )
    entries = {entry.entry_id: entry for entry in value.entries}
    original = _rebuild_continuation_value(entries["source/original"].value)
    source = _rebuild_continuation_value(entries["source/envelope"].value)
    queries = {
        entry_id: _rebuild_piecewise_boundary_query_recipe(
            entries[entry_id].value, source
        )
        for entry_id in query_ids
    }
    if not isinstance(original, ReconstructedGeometrySource):
        raise TypeError("PWL boundary continuation lost its reconstructed source owner.")
    if not isinstance(source, SurfaceEnvelopeSource):
        raise TypeError("PWL boundary continuation lost its native envelope source.")
    if (
        source.original.source_id != original.source_id
        or source.original.source_revision != original.source_revision
    ):
        raise ValueError("PWL envelope no longer binds to its reconstructed source.")
    for query in queries.values():
        if not isinstance(query, PiecewiseLinearBoundarySource):
            raise TypeError("PWL boundary continuation lost one of its query owners.")
        query_owner = query.owner
        if not isinstance(query_owner, SurfaceEnvelopeSource):
            raise TypeError("PWL boundary query lost its reconstructed envelope owner.")
        if (
            query_owner.envelope_id != source.envelope_id
            or query.domain.domain_id != source.domain.domain_id
        ):
            raise ValueError(
                "PWL boundary query differs from its recomputed envelope source."
            )
    rebuilt_values = {
        "source/original": original,
        "source/envelope": source,
        **queries,
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
            dependencies=entry.dependencies,
        )
        for entry in value.entries
    )
    composition = Composition(rebuilt_entries, boundary_id=value.boundary_id)
    if composition.composition_id != value.composition_id:
        raise ValueError(
            "PWL boundary Composition identity changed on cold reconstruction."
        )
    return composition


def _write_piecewise_boundary_query(directory: Path, /) -> None:
    original, source, queries = _piecewise_boundary_query_fixture()
    original_entry = CompositionEntry(
        original,
        entry_id="source/original",
        role="topology",
        owner_id="phydrax.geometry.reconstruction",
        structure_id=original.source_id,
        revision_id=original.source_revision,
        semantics_id="reconstructed-mesh-source",
    )
    source_entry = CompositionEntry(
        source,
        entry_id="source/envelope",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=source.envelope_id,
        revision_id=source.source_revision,
        semantics_id="native-surface-envelope",
        dependencies=(
            original_entry.binding("structure"),
            original_entry.binding("revision"),
        ),
    )
    query_entries = tuple(
        CompositionEntry(
            query,
            entry_id=f"source/boundary-query-{query.entity_dimension}",
            role="topology",
            owner_id="phydrax.geometry",
            structure_id=query.source_query_id,
            revision_id=query.source_revision,
            semantics_id="piecewise-linear-boundary-query",
            dependencies=(
                source_entry.binding("structure"),
                source_entry.binding("revision"),
            ),
        )
        for query in queries
    )
    composition = Composition(
        (original_entry, source_entry, *query_entries),
        boundary_id="piecewise-linear-boundary-query-continuation",
    )
    write_model_artifact(directory / "pwl-boundary-query.phx", composition, limits=None)
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "envelope_id": source.envelope_id,
                "queries": tuple(
                    {
                        "dimension": query.entity_dimension,
                        "source_query_id": query.source_query_id,
                        "entity_ids": query.entity_ids.tolist(),
                        "source_rows": query.source_rows.tolist(),
                        "entity_keys": query.source_entity_keys,
                        "occurrences": query.source_occurrence_paths,
                        "chart_ids": query.chart_ids,
                    }
                    for query in queries
                ),
            }
        )
    )


def _read_piecewise_boundary_query(path: Path, expected: dict[str, object], /) -> None:
    composition = read_model_artifact(
        path,
        expected_type=Composition,
        restore=_rebuild_piecewise_boundary_query_composition,
        limits=None,
    )
    entries = {entry.entry_id: entry for entry in composition.entries}
    source = entries["source/envelope"].value
    queries = tuple(
        entries[f"source/boundary-query-{dimension}"].value for dimension in (2, 1, 0)
    )
    expected_queries = tuple(
        _expected_object_mapping(record)
        for record in _expected_sequence(expected["queries"])
    )
    if (
        composition.composition_id != expected["composition_id"]
        or source.source_id != expected["source_id"]
        or source.source_revision != expected["source_revision"]
        or source.envelope_id != expected["envelope_id"]
        or len(queries) != len(expected_queries)
        or any(
            query.entity_dimension != record["dimension"]
            or query.source_query_id != record["source_query_id"]
            or query.entity_ids.tolist() != record["entity_ids"]
            or query.source_rows.tolist() != record["source_rows"]
            or query.source_entity_keys != _expected_sequence(record["entity_keys"])
            or query.source_occurrence_paths
            != tuple(
                _expected_sequence(path)
                for path in _expected_sequence(record["occurrences"])
            )
            or query.chart_ids != _expected_sequence(record["chart_ids"])
            for query, record in zip(queries, expected_queries, strict=True)
        )
    ):
        raise ValueError(
            "Cold PWL boundary queries changed their retained source strata."
        )
    print("pwl-boundary-query-cold-restored")


def _write_parametric_curve_query(directory: Path, /) -> None:
    source = _square_model(SpatialCoordinateContract.si())
    query = ParametricCurveBoundarySource(
        source.geometry.pcurves,
        ((0.0, 1.0),) * len(source.geometry.pcurves),
        source_id=source.source_id,
        source_revision=source.source_revision,
        curve_entity_ids=tuple(f"square-edge:{index}" for index in range(4)),
        occurrence_paths=(("square-face:0",),) * 4,
        chart_ids=tuple(f"square-chart:{index}" for index in range(4)),
    )
    query.require_current()
    source_entry = CompositionEntry(
        source,
        entry_id="cad/source",
        role="topology",
        owner_id="phydrax.geometry.brep",
        structure_id=source.geometry.geometry_id,
        revision_id=source.source_revision,
        semantics_id="native-source-brep",
    )
    query_entry = CompositionEntry(
        query,
        entry_id="geometry/curve-query",
        role="topology",
        owner_id="phydrax.geometry",
        structure_id=query.query_id,
        revision_id=query.source_revision,
        semantics_id="parametric-curve-boundary-query",
        dependencies=(source_entry.binding("revision"),),
    )
    composition = Composition(
        (source_entry, query_entry), boundary_id="parametric-curve-query-archive"
    )
    write_model_artifact(
        directory / "parametric-curve-query.phx", composition, limits=None
    )
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "query_id": query.query_id,
            }
        )
    )


def _rebuild_parametric_curve_query_composition(value: object, /) -> Composition:
    if not isinstance(value, Composition) or len(value.entries) != 2:
        raise TypeError("Curve-query archive requires one source and one query entry.")
    entries = {entry.entry_id: entry for entry in value.entries}
    source_entry = entries.get("cad/source")
    query_entry = entries.get("geometry/curve-query")
    if source_entry is None or query_entry is None:
        raise ValueError("Curve-query archive is missing its source or query owner.")
    source = _rebuild_continuation_value(source_entry.value)
    query = _rebuild_continuation_value(query_entry.value)
    if not isinstance(source, BRepModel) or not isinstance(
        query, ParametricCurveBoundarySource
    ):
        raise TypeError("Curve-query archive owners did not rebuild their native values.")
    if (
        source_entry.role != "topology"
        or source_entry.owner_id != "phydrax.geometry.brep"
        or source_entry.structure_id != source.geometry.geometry_id
        or source_entry.revision_id != source.source_revision
        or source_entry.semantics_id != "native-source-brep"
        or query_entry.role != "topology"
        or query_entry.owner_id != "phydrax.geometry"
        or query_entry.structure_id != query.query_id
        or query_entry.revision_id != query.source_revision
        or query_entry.semantics_id != "parametric-curve-boundary-query"
        or query.source_id != source.source_id
        or query.source_revision != source.source_revision
        or query_entry.dependencies != (source_entry.binding("revision"),)
    ):
        raise ValueError("Curve-query source binding differs from recomputed owners.")
    query.require_current()
    measured = query.boundary_distance(np.asarray(((0.5, 0.1),), dtype=np.float64))
    if not (
        float(measured.lower[0]) <= 0.1 <= float(measured.upper[0])
        and float(measured.upper[0]) <= 0.1000000001
    ):
        raise ValueError(
            "Cold curve source no longer supports its native distance query."
        )
    rebuilt_entries = (
        CompositionEntry(
            source,
            entry_id=source_entry.entry_id,
            role=source_entry.role,
            owner_id=source_entry.owner_id,
            structure_id=source_entry.structure_id,
            revision_id=source_entry.revision_id,
            semantics_id=source_entry.semantics_id,
        ),
        CompositionEntry(
            query,
            entry_id=query_entry.entry_id,
            role=query_entry.role,
            owner_id=query_entry.owner_id,
            structure_id=query_entry.structure_id,
            revision_id=query_entry.revision_id,
            semantics_id=query_entry.semantics_id,
            dependencies=(source_entry.binding("revision"),),
        ),
    )
    return Composition(rebuilt_entries, boundary_id=value.boundary_id)


def _read_parametric_curve_query(path: Path, expected: dict[str, str], /) -> None:
    composition = read_model_artifact(
        path,
        expected_type=Composition,
        restore=_rebuild_parametric_curve_query_composition,
        limits=None,
    )
    query = composition.value("geometry/curve-query")
    if (
        composition.composition_id != expected["composition_id"]
        or query.source_id != expected["source_id"]
        or query.source_revision != expected["source_revision"]
        or query.query_id != expected["query_id"]
    ):
        raise ValueError("Cold parametric curve source identity changed.")
    print("parametric-curve-query-cold-restored")


def _write_adaptive_implicit(directory: Path, /) -> None:
    recipe, surface = _adaptive_implicit_archive_fixture()
    entry = CompositionEntry(
        recipe,
        entry_id="geometry/adaptive-implicit-domain",
        role="topology",
        owner_id="phydrax.geometry.implicit",
        structure_id=surface.domain_id,
        revision_id=_adaptive_implicit_surface_revision(surface),
        semantics_id="adaptive-implicit-certification-domain-source",
    )
    composition = Composition((entry,), boundary_id="adaptive-implicit-archive")
    write_model_artifact(directory / "implicit-domain.phx", composition, limits=None)
    cover = surface.cover
    if cover is None:
        raise ValueError(
            "Adaptive implicit archive fixture requires its genuine computed cover."
        )
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "domain_id": surface.domain_id,
                "evidence_id": surface.evidence.evidence_id,
                "surface_revision": entry.revision_id,
                "cover_id": cover.cover_id,
                "source_id": surface.source_id,
            }
        )
    )


def _read_adaptive_implicit(path: Path, expected: dict[str, str], /) -> None:
    composition = read_model_artifact(
        path,
        expected_type=Composition,
        restore=_rebuild_adaptive_implicit_composition,
        limits=None,
    )
    entry = composition.entries[0]
    if (
        composition.composition_id != expected["composition_id"]
        or entry.structure_id != expected["domain_id"]
        or entry.revision_id != expected["surface_revision"]
        or entry.owner_id != "phydrax.geometry.implicit"
        or not isinstance(entry.value, AdaptiveImplicitSurfaceArchiveRecipe)
        or entry.value.source_id != expected["source_id"]
    ):
        raise ValueError("Cold adaptive implicit Composition identities changed.")
    print("implicit-cold-restored")


def _source_coverage_archive_fixture() -> tuple[
    AnalyticImplicitProfile, CellMeshingResult
]:
    source_id = "archive-restricted-coverage-sphere"
    source_revision = "archive-restricted-coverage-sphere-r1"
    coordinate_contract = SpatialCoordinateContract.si()
    geometry = Sphere((0.0, 0.0, 0.0), 0.75, feature_id=source_id).compile()
    source = AnalyticImplicitProfile(
        geometry,
        coordinate_contract,
        tube_radius=0.5,
        source_id=source_id,
        source_revision=source_revision,
    )
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        maximum_deviation=0.8,
    )
    construction = prepare_native_implicit(
        source,
        specification,
        NativeMeshingOptions(
            "implicit_restricted_delaunay",
            volume_schedule=NativeVolumeSchedule(),
        ),
        coordinate_contract,
    ).execute()
    request = construction.certification
    if request.source_coverage_witness is None:
        raise ValueError("Restricted coverage fixture omitted its native witness.")
    inputs = MeshCertificationInputs(
        construction.mesh,
        construction.geometry,
        request.schedule,
        source_id=request.source_id,
        source_revision=request.source_revision,
        domain=request.domain,
        cell_regions=request.cell_regions,
        source=request.fidelity_source,
        fidelity_tolerance=request.fidelity_tolerance,
        maximum_normal_deviation=request.maximum_normal_deviation,
        junction_vertices=request.junction_vertices,
        protected_features=request.protected_features,
        scoped_fidelity=request.scoped_fidelity,
        fidelity_witness=request.fidelity_witness,
        feature_source=request.feature_source,
        source_coverage_witness=request.source_coverage_witness,
    )
    organization = construction.organization
    result = certify_cell_mesh(
        construction.mesh,
        coordinate_contract,
        geometry=construction.geometry,
        boundary=organization.boundary,
        certification_inputs=inputs,
        patches=organization.patches,
        zones=organization.zones,
        labels=organization.labels,
        attributes=organization.attributes,
        associations=organization.associations,
        surface_source=organization.surface_source,
        region_boundary_evidence=organization.region_boundary_evidence,
        planar_embedding=construction.planar_embedding,
        embedding=request.global_embedding,
    )
    if result.certification is None:
        raise ValueError("Restricted coverage fixture lost computed certification.")
    return source, result


def _source_coverage_composition(
    source: AnalyticImplicitProfile, result: CellMeshingResult, /
) -> Composition:
    mesh = result.mesh
    geometry = result.geometry
    source_entry = CompositionEntry(
        source,
        entry_id="implicit/source",
        role="topology",
        owner_id="phydrax.geometry.implicit",
        structure_id=source.profile_id,
        revision_id=source.source_revision,
        semantics_id="analytic-implicit-source-profile",
    )
    mesh_entry = CompositionEntry(
        mesh,
        entry_id="mesh/topology",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=mesh.topology_id,
        revision_id=mesh.mesh_id,
        semantics_id="cell-mesh",
    )
    geometry_entry = CompositionEntry(
        geometry,
        entry_id="mesh/geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=geometry.geometry_layout_id,
        revision_id=mesh.geometry_id,
        semantics_id="cell-coordinate-map",
        dependencies=(mesh_entry.binding("structure"),),
    )
    result_entry = CompositionEntry(
        result,
        entry_id="mesh/result",
        role="discretization",
        owner_id="phydrax.meshing",
        structure_id=mesh.topology_id,
        revision_id=result.result_id,
        semantics_id="accepted-cell-meshing-result",
        dependencies=(
            source_entry.binding("revision"),
            mesh_entry.binding("structure"),
            geometry_entry.binding("structure"),
        ),
    )
    return Composition(
        (source_entry, mesh_entry, geometry_entry, result_entry),
        boundary_id="native-source-coverage-continuation",
    )


def _write_source_coverage_artifact(directory: Path, /) -> None:
    source, result = _source_coverage_archive_fixture()
    composition = _source_coverage_composition(source, result)
    write_model_artifact(
        directory / "source-coverage.phx", composition, limits=MODEL_ARCHIVE_LIMITS
    )
    report = result.certification
    if report is None or report.inputs.source_coverage_witness is None:
        raise ValueError("Restricted coverage report omitted its witness.")
    if report.coverage is None:
        raise ValueError("Restricted coverage report omitted its material certificate.")
    query_points = np.asarray(((0.75, 0.0, 0.0), (1.1, 0.2, -0.1)))
    distance = source.boundary_distance(query_points)
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "profile_id": source.profile_id,
                "mesh_id": result.mesh.mesh_id,
                "geometry_id": cell_geometry_id(result.geometry),
                "request_id": report.inputs.request_id,
                "report_id": report.report_id,
                "result_id": result.result_id,
                "witness_id": report.inputs.source_coverage_witness.witness_id,
                "coverage_region_occurrence_paths": [
                    list(path) for path in report.coverage.region_occurrence_paths
                ],
                "query_points": query_points.tolist(),
                "distance_lower": np.asarray(distance.lower).tolist(),
                "distance_upper": np.asarray(distance.upper).tolist(),
            }
        )
    )


def _rebuild_source_coverage_composition(
    value: object, expected: dict[str, object], /
) -> Composition:
    if not isinstance(value, Composition):
        raise TypeError("Source-coverage continuation requires a Composition root.")
    entries = {entry.entry_id: entry for entry in value.entries}
    expected_ids = {"implicit/source", "mesh/topology", "mesh/geometry", "mesh/result"}
    if set(entries) != expected_ids:
        raise ValueError(
            "Source-coverage Composition entries differ from its restart schema."
        )
    source = _rebuild_continuation_value(entries["implicit/source"].value)
    mesh = _rebuild_continuation_value(entries["mesh/topology"].value)
    geometry = _rebuild_geometry_for_mesh(entries["mesh/geometry"].value, mesh)
    result = _rebuild_continuation_value(entries["mesh/result"].value)
    if (
        not isinstance(source, AnalyticImplicitProfile)
        or not isinstance(mesh, CellMesh)
        or not isinstance(geometry, CellGeometrySpec)
        or not isinstance(result, CellMeshingResult)
        or result.certification is None
    ):
        raise TypeError("Cold source-coverage continuation lost a native model owner.")
    report = result.certification
    witness = report.inputs.source_coverage_witness
    if witness is None:
        raise ValueError("Cold source-coverage continuation lost its proof input.")
    report.require_passed()
    # The restore recomputed this report from the rebuilt premises in this
    # process; replaying it again on the same objects is determined by their
    # identity, so only the binding to the restored mesh and map is required.
    report.inputs.require_current(
        result.mesh,
        result.geometry,
        domain=report.inputs.domain,
        source=report.inputs.source,
    )
    if report.coverage is None:
        raise ValueError("Cold restricted coverage lost its computed domain certificate.")
    if not isinstance(report.inputs.domain, AnalyticImplicitProfile):
        raise TypeError(
            "Cold restricted coverage lost its analytic profile domain owner."
        )
    coverage_source = report.inputs.source
    if coverage_source is None:
        raise TypeError("Cold restricted coverage lost its source boundary query.")
    query_points = np.asarray(_expected_array(expected["query_points"]), dtype=np.float64)
    distance = source.boundary_distance(query_points)
    np.testing.assert_array_equal(
        np.asarray(distance.lower), _expected_array(expected["distance_lower"])
    )
    np.testing.assert_array_equal(
        np.asarray(distance.upper), _expected_array(expected["distance_upper"])
    )
    if (
        source.profile_id != expected["profile_id"]
        or mesh.mesh_id != expected["mesh_id"]
        or result.mesh.mesh_id != mesh.mesh_id
        or cell_geometry_id(geometry) != expected["geometry_id"]
        or cell_geometry_id(result.geometry) != expected["geometry_id"]
        or report.inputs.request_id != expected["request_id"]
        or report.report_id != expected["report_id"]
        or result.result_id != expected["result_id"]
        or witness.witness_id != expected["witness_id"]
        or [list(path) for path in report.coverage.region_occurrence_paths]
        != expected["coverage_region_occurrence_paths"]
        or report.inputs.domain.profile_id != source.profile_id
        or coverage_source.source_id != source.source_id
        or coverage_source.source_revision != source.source_revision
    ):
        raise ValueError(
            "Cold restricted coverage continuation changed its source or proof identity."
        )
    rebuilt_entries = tuple(
        CompositionEntry(
            rebuilt,
            entry_id=entries[entry_id].entry_id,
            role=entries[entry_id].role,
            owner_id=entries[entry_id].owner_id,
            structure_id=entries[entry_id].structure_id,
            revision_id=entries[entry_id].revision_id,
            semantics_id=entries[entry_id].semantics_id,
            dependencies=entries[entry_id].dependencies,
        )
        for entry_id, rebuilt in (
            ("implicit/source", source),
            ("mesh/topology", mesh),
            ("mesh/geometry", geometry),
            ("mesh/result", result),
        )
    )
    composition = Composition(rebuilt_entries, boundary_id=value.boundary_id)
    if composition.composition_id != expected["composition_id"]:
        raise ValueError("Cold source-coverage Composition identity changed.")
    return composition


def _read_source_coverage_artifact(
    directory: Path, expected: dict[str, object], /
) -> None:
    composition = read_model_artifact(
        directory / "source-coverage.phx",
        expected_type=Composition,
        restore=lambda value: _rebuild_source_coverage_composition(value, expected),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    if composition.composition_id != expected["composition_id"]:
        raise ValueError("Cold source-coverage Composition identity changed.")
    print("source-coverage-cold-restored")


def _region_boundary_archive_fixture() -> CellMeshingResult:
    coordinates = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "triangle",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int64),
                global_ids=np.asarray((41,), dtype=np.int64),
            ),
        ),
    )
    boundary_scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        1,
        mesh.entity_set(1).entity_set_id,
        mesh.entity_set(1).entity_ids,
    )
    source_scope = MeshingScope(
        "authored-sheet",
        "sheet-revision",
        MeshingEntityKind.GEOMETRY,
        2,
        "authored-regions",
        np.asarray((7,), dtype=np.int64),
    )
    source_path: tuple[str, ...] = ()
    patch = MeshPatch(
        "wall",
        boundary_scope,
        source_adjacent_region_ids=("sheet-region", None),
        source_adjacent_region_occurrence_paths=(source_path, None),
    )
    label = MeshLabel("region-boundary", boundary_scope)
    evidence = RegionBoundaryEvidence(
        mesh,
        source_scope,
        "sheet-region",
        "region-control",
        None,
        None,
        label,
        ((patch.patch_id, 1),),
        source_occurrence_path=source_path,
    )
    result = certify_cell_mesh(
        mesh,
        SpatialCoordinateContract.si(),
        patches=(patch,),
        labels=(label,),
        region_boundary_evidence=(evidence,),
        planar_embedding=PlanarEmbedding(
            (3.0, 4.0, 5.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    )
    return result


def _region_boundary_archive_metadata(result: CellMeshingResult, /) -> dict[str, object]:
    if len(result.patches) != 1 or len(result.region_boundary_evidence) != 1:
        raise ValueError(
            "Region boundary fixture must retain one patch and one evidence record."
        )
    patch = result.patches[0]
    evidence = result.region_boundary_evidence[0]
    return {
        "result_id": result.result_id,
        "patch_id": patch.patch_id,
        "evidence_id": evidence.evidence_id,
        "source_id": evidence.source_scope.source_id,
        "source_revision": evidence.source_scope.source_revision,
        "source_scope_id": evidence.source_scope.scope_id,
        "source_entity_set_id": evidence.source_scope.entity_set_id,
        "source_region_id": evidence.source_region_id,
        "source_occurrence_path": list(evidence.source_occurrence_path),
        "control_id": evidence.control_id,
        "patch_source_adjacent_region_ids": list(patch.source_adjacent_region_ids or ()),
        "patch_source_adjacent_region_occurrence_paths": [
            None if path is None else list(path)
            for path in (patch.source_adjacent_region_occurrence_paths or ())
        ],
    }


def _write_region_boundary_archive(directory: Path, /) -> None:
    result = _region_boundary_archive_fixture()
    write_model_artifact(directory / "region-boundary-evidence.phx", result, limits=None)
    print(json.dumps(_region_boundary_archive_metadata(result), sort_keys=True))


def _read_region_boundary_archive(
    directory: Path,
    expected: dict[str, object],
    /,
) -> None:
    result = read_model_artifact(
        directory / "region-boundary-evidence.phx",
        expected_type=CellMeshingResult,
        restore=_rebuild_continuation_value,
        limits=None,
    )
    if (
        not isinstance(result, CellMeshingResult)
        or len(result.region_boundary_evidence) != 1
    ):
        raise TypeError("Cold region boundary archive lost its accepted evidence owner.")
    evidence = result.region_boundary_evidence[0]
    evidence.require_current(result.mesh, result.labels, result.patches)
    if _region_boundary_archive_metadata(result) != expected:
        raise ValueError("Cold region boundary occurrence identity changed.")
    print("region-boundary-evidence-cold-restored")


def _write_certification_artifact(directory: Path, /) -> None:
    mesh = CellMesh(
        np.asarray(((0.0,), (1.0,)), dtype=np.float64),
        (
            CellBlock(
                "curve",
                "interval",
                np.asarray(((0, 1),), dtype=np.int32),
                global_ids=np.asarray((41,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray((17, 23), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    inputs = MeshCertificationInputs(
        mesh,
        geometry,
        MeshCertificationSchedule("curve"),
        source_id="archive-test:curve-source",
        source_revision="archive-test:curve-revision",
    )
    result = certify_cell_mesh(
        mesh,
        SpatialCoordinateContract.si(),
        geometry=geometry,
        certification_inputs=inputs,
    )
    if result.certification is None:
        raise ValueError("Certification archive fixture must retain computed acceptance.")
    write_model_artifact(directory / "certification.phx", result, limits=None)
    print(
        json.dumps(
            {
                "mesh_id": mesh.mesh_id,
                "topology_id": mesh.topology_id,
                "geometry_id": cell_geometry_id(geometry),
                "request_id": inputs.request_id,
                "report_id": result.certification.report_id,
                "result_id": result.result_id,
            }
        )
    )


def _read_certification_artifact(directory: Path, expected: dict[str, str], /) -> None:
    result = read_model_artifact(
        directory / "certification.phx",
        expected_type=CellMeshingResult,
        restore=_rebuild_continuation_value,
        limits=None,
    )
    if not isinstance(result, CellMeshingResult) or result.certification is None:
        raise TypeError("Cold certification archive lost its computed result report.")
    report = result.certification
    report.require_passed()
    report.revalidate(result.mesh, result.geometry)
    if (
        result.mesh.mesh_id != expected["mesh_id"]
        or result.mesh.topology_id != expected["topology_id"]
        or cell_geometry_id(result.geometry) != expected["geometry_id"]
        or report.inputs.request_id != expected["request_id"]
        or report.report_id != expected["report_id"]
        or result.result_id != expected["result_id"]
    ):
        raise ValueError(
            "Cold certification continuation changed its source or result identity."
        )
    print("certification-cold-restored")


def _archive_tetra_source(source_key: str, /) -> PiecewiseLinearComplex:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    polygons = (
        np.asarray((0, 2, 1), dtype=np.int64),
        np.asarray((0, 1, 3), dtype=np.int64),
        np.asarray((0, 3, 2), dtype=np.int64),
        np.asarray((1, 2, 3), dtype=np.int64),
    )
    return PiecewiseLinearComplex(
        points,
        polygons,
        np.arange(4, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (4, 1)),
        (f"{source_key}:bulk",),
        vertex_ids=tuple(f"{source_key}:vertex:{index}" for index in range(4)),
        vertex_occurrence_paths=tuple(
            (source_key, "vertex", str(index)) for index in range(4)
        ),
        facet_ids=tuple(f"{source_key}:facet:{index}" for index in range(4)),
        facet_occurrence_paths=tuple((source_key, "shell") for _ in range(4)),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((source_key, "region", "bulk"),),
        boundary="conforming",
    )


def _two_solid_occurrence_model(contract: SpatialCoordinateContract, /) -> BRepModel:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    exact_points = tuple(
        (
            Fraction(int(row[0])),
            Fraction(int(row[1])),
            Fraction(int(row[2])),
        )
        for row in points
    )
    edges = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3))
    curves = tuple(
        ExactLineCurve(exact_points[start], exact_points[end]) for start, end in edges
    )
    pcurves = (
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((1.0, 0.0), (-1.0, 1.0)),
        LineCurve((0.0, 1.0), (0.0, -1.0)),
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((1.0, 0.0), (-1.0, 1.0)),
        LineCurve((0.0, 1.0), (0.0, -1.0)),
    )
    assembly = BRepAssemblyGraph(
        roots=(
            BRepAssemblyUse("left", solid_index=0),
            BRepAssemblyUse("right", solid_index=1),
        ),
        definitions=(),
        solid_count=2,
    )
    geometry = BRepGeometry(
        vertex_points=points,
        vertex_exact_points=exact_points,
        curves=curves,
        edge_curves=tuple(range(6)),
        edge_ranges=np.tile(np.asarray((0.0, 1.0)), (6, 1)),
        edge_vertices=edges,
        pcurves=pcurves,
        coedge_edges=tuple(range(6)),
        coedge_senses=(1,) * 6,
        face_loops=(((0, 1, 2),), ((3, 4, 5),)),
        shell_faces=((0,), (1,)),
        shell_orientations=((1,), (1,)),
        solid_shells=((0,), (1,)),
        assembly_graph=assembly,
    )
    source_id = "archive-qualified-entity-source"
    report = BRepImportReport(
        source_id=source_id,
        source_digest="b" * 64,
        source_format="native",
        coordinate_contract=contract,
        import_policy_id="c" * 64,
        num_solids=2,
        num_faces=2,
        num_edges=6,
        num_vertices=6,
    )
    return BRepModel(
        geometry=geometry,
        patches=(
            PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            PlanePatch((2.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ),
        parameter_bounds=np.asarray(
            (((0.0, 0.0), (1.0, 1.0)), ((0.0, 0.0), (1.0, 1.0))),
            dtype=np.float64,
        ),
        orientation=np.asarray((1.0, 1.0), dtype=np.float64),
        physical_tags=("left-face", "right-face"),
        coordinate_contract=contract,
        report=report,
    )


def _brep_assembly_placement_evidence_model() -> BRepModel:
    from phydrax.geometry.brep._constructors import brep_box
    from phydrax.interchange._cad import CadImportPolicy
    from phydrax.interchange._cad_carriers import RigidPlacement
    from phydrax.interchange._cad_placement_evidence import (
        certify_placed_model,
        source_placement_evidence,
    )
    from tests.unit.interchange.test_cad_placement_evidence import _model_with_graph

    contract = SpatialCoordinateContract.si()
    definition = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        coordinate_contract=contract,
    )
    rotation = RigidPlacement.from_source_directions(
        np.asarray((0.8, 0.6, 0.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        np.asarray((4.0, -3.0, 2.0), dtype=np.float64),
        "archive-placement-evidence",
    )
    evidence = source_placement_evidence(rotation)
    if evidence is None:
        raise ValueError("Placement evidence fixture has no source pose error.")
    placement = BRepAffinePlacement.from_matrix(
        (
            (*rotation.rotation[0], rotation.translation[0]),
            (*rotation.rotation[1], rotation.translation[1]),
            (*rotation.rotation[2], rotation.translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    graph = BRepAssemblyGraph(
        roots=(
            BRepAssemblyUse(
                "placed-box",
                placement,
                solid_index=0,
                placement_evidence=evidence,
            ),
        ),
        definitions=(),
        solid_count=1,
    )
    source = _model_with_graph(definition, graph)
    certified = certify_placed_model(
        source,
        CadImportPolicy(contract, maximum_placement_deviation=1.0e-12),
    )
    if type(certified) is not BRepModel:
        raise TypeError("Placement evidence certification did not return BRepModel.")
    return certified


def _brep_assembly_placement_evidence_summary(model: BRepModel, /) -> dict[str, object]:
    from phydrax.geometry.brep._assembly import BRepPlacementEvidence

    graph = model.geometry.assembly_graph
    if not isinstance(graph, BRepAssemblyGraph):
        raise TypeError("Placement evidence archive lost its BRepAssemblyGraph.")
    if type(graph) is not BRepAssemblyGraph:
        raise TypeError("Placement evidence archive has a noncanonical assembly graph.")
    if len(graph.roots) != 1 or len(graph.occurrence_evidence) != 1:
        raise ValueError("Placement evidence archive requires one certified occurrence.")
    root_evidence = graph.roots[0].placement_evidence
    if not isinstance(root_evidence, BRepPlacementEvidence):
        raise TypeError("Placement evidence archive lost its root-use error bound.")
    if type(root_evidence) is not BRepPlacementEvidence:
        raise TypeError("Placement evidence archive has a noncanonical root-use bound.")
    occurrences = tuple(model.geometry.iter_occurrences())
    if len(occurrences) != 1:
        raise ValueError("Placement evidence archive requires one realized occurrence.")
    occurrence = occurrences[0]
    occurrence_evidence = occurrence.placement_evidence
    if not isinstance(occurrence_evidence, BRepPlacementEvidence):
        raise TypeError("Placement evidence archive lost its occurrence error bound.")
    if type(occurrence_evidence) is not BRepPlacementEvidence:
        raise TypeError("Placement evidence archive has a noncanonical occurrence bound.")
    return {
        "model_id": model.model_id,
        "source_revision": model.source_revision,
        "geometry_id": model.geometry.geometry_id,
        "root_use_evidence": _json_payload(root_evidence.payload()),
        "occurrence_evidence": [
            [list(path), _json_payload(evidence.payload())]
            for path, evidence in graph.occurrence_evidence
        ],
        "occurrences": [
            {
                "path": list(occurrence.path),
                "solid_index": occurrence.solid_index,
                "placement": _json_payload(occurrence.placement.identity_payload()),
                "placement_evidence": _json_payload(occurrence_evidence.payload()),
            }
            for occurrence in occurrences
        ],
    }


def _write_brep_assembly_placement_evidence_archive(directory: Path, /) -> None:
    model = _brep_assembly_placement_evidence_model()
    path = directory / "brep-assembly-placement-evidence.phx"
    save_brep_archive(model, path, ingress_limits=None)
    print(json.dumps(_brep_assembly_placement_evidence_summary(model), sort_keys=True))


def _read_brep_assembly_placement_evidence_archive(
    directory: Path, expected: dict[str, object], /
) -> None:
    model = load_brep_archive(
        directory / "brep-assembly-placement-evidence.phx",
        ingress_limits=None,
    )
    observed = _brep_assembly_placement_evidence_summary(model)
    if observed != expected:
        raise ValueError("Cold BRep assembly placement evidence changed.")
    print("brep-assembly-placement-evidence-cold-restored")


def _brep_entity_composition(entity_index: int = 0, /) -> Composition:
    model = _two_solid_occurrence_model(SpatialCoordinateContract.si())
    entity = BRepEntityId(
        model.source_id,
        model.source_revision,
        "face",
        entity_index,
        ("left",),
    )
    source_entry = CompositionEntry(
        model,
        entry_id="cad/source",
        role="topology",
        owner_id="phydrax.geometry.brep",
        structure_id=model.geometry.geometry_id,
        revision_id=model.source_revision,
        semantics_id="native-source-brep",
    )
    entity_entry = CompositionEntry(
        entity,
        entry_id="cad/qualified-face",
        role="topology",
        owner_id="phydrax.geometry.brep",
        structure_id=model.geometry.geometry_id,
        revision_id=model.source_revision,
        semantics_id="occurrence-qualified-brep-entity",
        dependencies=(source_entry.binding("revision"),),
    )
    return Composition(
        (source_entry, entity_entry),
        boundary_id="occurrence-qualified-brep-entity-continuation",
    )


def _rebuild_brep_entity_composition(value: object, /) -> Composition:
    if not isinstance(value, Composition):
        raise TypeError("BRep entity continuation requires a Composition root.")
    entries = {entry.entry_id: entry for entry in value.entries}
    if set(entries) != {"cad/source", "cad/qualified-face"}:
        raise ValueError(
            "BRep entity Composition entries differ from its restart schema."
        )
    from phydrax.interchange._cad_archive import _reconstruct_brep_graph

    model, entity = _reconstruct_brep_graph(
        (entries["cad/source"].value, entries["cad/qualified-face"].value)
    )
    if not isinstance(model, BRepModel) or not isinstance(entity, BRepEntityId):
        raise TypeError("Cold BRep entity continuation lost its canonical owner.")
    rebuilt_entries = tuple(
        CompositionEntry(
            rebuilt,
            entry_id=entries[entry_id].entry_id,
            role=entries[entry_id].role,
            owner_id=entries[entry_id].owner_id,
            structure_id=entries[entry_id].structure_id,
            revision_id=entries[entry_id].revision_id,
            semantics_id=entries[entry_id].semantics_id,
            dependencies=entries[entry_id].dependencies,
        )
        for entry_id, rebuilt in (("cad/source", model), ("cad/qualified-face", entity))
    )
    composition = Composition(rebuilt_entries, boundary_id=value.boundary_id)
    if composition.composition_id != value.composition_id:
        raise ValueError("Cold BRep entity Composition identity changed.")
    return composition


def _write_brep_entity_composition(directory: Path, /) -> None:
    composition = _brep_entity_composition()
    incidence = composition.value("cad/source").geometry.qualified_entity_incidence(
        composition.value("cad/source").source_id,
        composition.value("cad/source").source_revision,
    )
    entity = composition.value("cad/qualified-face")
    if not any(
        relation.container == entity or relation.member == entity
        for relation in incidence
    ):
        raise ValueError("BRep occurrence fixture does not contain its selected face.")
    write_model_artifact(
        directory / "qualified-brep-entity.phx", composition, limits=None
    )
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "source_id": entity.source_id,
                "source_revision": entity.source_revision,
                "entity_index": entity.index,
                "occurrence_path": list(entity.occurrence_path),
            }
        )
    )


def _read_brep_entity_composition(
    directory: Path, expected: dict[str, object], /
) -> None:
    composition = read_model_artifact(
        directory / "qualified-brep-entity.phx",
        expected_type=Composition,
        restore=_rebuild_brep_entity_composition,
        limits=None,
    )
    entity = composition.value("cad/qualified-face")
    if (
        composition.composition_id != expected["composition_id"]
        or entity.source_id != expected["source_id"]
        or entity.source_revision != expected["source_revision"]
        or entity.index != expected["entity_index"]
        or list(entity.occurrence_path) != expected["occurrence_path"]
    ):
        raise ValueError(
            "Cold occurrence-qualified BRep entity changed source membership."
        )
    print("qualified-brep-entity-cold-restored")


def _rebuild_brep_projection_result(
    value: object, source: BRepModel, /
) -> BRepProjectionResult:
    if type(value) is not BRepProjectionResult:
        raise TypeError("Stored projection evidence has an unexpected concrete type.")
    if (
        value.source_id != source.source_id
        or value.source_revision != source.source_revision
        or value.frame_id != source.coordinate_contract.spatial_id
    ):
        raise ValueError(
            "Stored BRep projection result changes its source or frame binding."
        )
    projection = prepare_brep_projection(source)
    rebuilt = projection.project_boundary(
        value.queries,
        occurrence_paths=value.occurrence_paths,
    )
    fields = (
        "dimensions",
        "indices",
        "queries",
        "points",
        "parameters",
        "residuals",
        "distance_lower_bounds",
        "status",
        "normals",
        "tangents",
    )
    for name in fields:
        stored = np.asarray(object.__getattribute__(value, name))
        current = np.asarray(object.__getattribute__(rebuilt, name))
        if stored.dtype.kind in "fc" or current.dtype.kind in "fc":
            equal = np.array_equal(stored, current, equal_nan=True)
        else:
            equal = np.array_equal(stored, current)
        if not equal:
            raise ValueError(
                f"Stored BRep projection {name} differs from cold source recomputation."
            )
    if (
        rebuilt.result_id != value.result_id
        or rebuilt.entity_ids() != value.entity_ids()
        or rebuilt.frame_id != value.frame_id
    ):
        raise ValueError(
            "Stored BRep projection identity differs from its recomputed rows."
        )
    return rebuilt


def _rebuild_brep_projection_composition(value: object, /) -> Composition:
    if not isinstance(value, Composition) or len(value.entries) != 2:
        raise TypeError(
            "BRep projection archive requires a source and projection result."
        )
    entries = {entry.entry_id: entry for entry in value.entries}
    if set(entries) != {"cad/source", "cad/projection"}:
        raise ValueError("BRep projection Composition entries differ from its schema.")
    source = _rebuild_continuation_value(entries["cad/source"].value)
    if not isinstance(source, BRepModel):
        raise TypeError("BRep projection result lost its retained BRepModel source.")
    projection = _rebuild_brep_projection_result(entries["cad/projection"].value, source)
    source_entry = entries["cad/source"]
    projection_entry = entries["cad/projection"]
    if (
        source_entry.structure_id != source.geometry.geometry_id
        or source_entry.revision_id != source.source_revision
        or projection_entry.structure_id != projection.result_id
        or projection_entry.revision_id != source.source_revision
        or projection_entry.dependencies
        != _canonical_dependencies(
            (source_entry.binding("structure"), source_entry.binding("revision"))
        )
    ):
        raise ValueError(
            "BRep projection Composition bindings differ from its renewed source."
        )
    rebuilt_entries = (
        CompositionEntry(
            source,
            entry_id=source_entry.entry_id,
            role=source_entry.role,
            owner_id=source_entry.owner_id,
            structure_id=source_entry.structure_id,
            revision_id=source_entry.revision_id,
            semantics_id=source_entry.semantics_id,
            dependencies=source_entry.dependencies,
        ),
        CompositionEntry(
            projection,
            entry_id=projection_entry.entry_id,
            role=projection_entry.role,
            owner_id=projection_entry.owner_id,
            structure_id=projection_entry.structure_id,
            revision_id=projection_entry.revision_id,
            semantics_id=projection_entry.semantics_id,
            dependencies=projection_entry.dependencies,
        ),
    )
    composition = Composition(rebuilt_entries, boundary_id=value.boundary_id)
    if composition.composition_id != value.composition_id:
        raise ValueError("Cold BRep projection Composition identity changed.")
    return composition


def _write_brep_projection_composition(directory: Path, /) -> None:
    source = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="archive-projection-box",
    )
    result = prepare_brep_projection(source).project_boundary(
        np.asarray(((1.1, 0.5, 0.5), (0.5, 1.1, 0.5)), dtype=np.float64)
    )
    source_entry = CompositionEntry(
        source,
        entry_id="cad/source",
        role="topology",
        owner_id="phydrax.geometry.brep",
        structure_id=source.geometry.geometry_id,
        revision_id=source.source_revision,
        semantics_id="native-source-brep",
    )
    projection_entry = CompositionEntry(
        result,
        entry_id="cad/projection",
        role="observation",
        owner_id="phydrax.geometry.brep",
        structure_id=result.result_id,
        revision_id=source.source_revision,
        semantics_id="source-bound-brep-projection",
        dependencies=(
            source_entry.binding("structure"),
            source_entry.binding("revision"),
        ),
    )
    composition = Composition(
        (source_entry, projection_entry),
        boundary_id="source-bound-brep-projection-continuation",
    )
    write_model_artifact(directory / "brep-projection.phx", composition, limits=None)
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "frame_id": result.frame_id,
                "result_id": result.result_id,
                "entity_ids": result.entity_ids(),
                "occurrence_paths": result.occurrence_paths,
            }
        )
    )


def _read_brep_projection_composition(path: Path, expected: dict[str, object], /) -> None:
    composition = read_model_artifact(
        path,
        expected_type=Composition,
        restore=_rebuild_brep_projection_composition,
        limits=None,
    )
    source = composition.value("cad/source")
    result = composition.value("cad/projection")
    if (
        composition.composition_id != expected["composition_id"]
        or source.source_id != expected["source_id"]
        or source.source_revision != expected["source_revision"]
        or result.frame_id != expected["frame_id"]
        or result.result_id != expected["result_id"]
        or result.entity_ids() != _expected_sequence(expected["entity_ids"])
        or result.occurrence_paths
        != tuple(
            _expected_sequence(path)
            for path in _expected_sequence(expected["occurrence_paths"])
        )
    ):
        raise ValueError(
            "Cold BRep projection did not renew from its retained source owner."
        )
    print("brep-projection-cold-restored")


def _reference_chart_arguments_composition() -> tuple[
    Composition, CellGeometrySpec, ReferenceCellMapWitnesses
]:
    from phydrax.discretization._nested_reference_types import ReferenceCellMapWitnesses

    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    block = CellBlock(
        "triangles",
        "triangle",
        np.asarray(((0, 1, 2),), dtype=np.int32),
        global_ids=np.asarray((51,), dtype=np.int64),
    )
    mesh = CellMesh(
        points,
        (block,),
        vertex_global_ids=np.asarray((101, 102, 103), dtype=np.int64),
    )
    source_geometry = CellGeometrySpec.affine(mesh)
    exact_corners = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    rational_x = RationalPolynomial({(1, 0): Fraction(1)}, {(0, 0): Fraction(1)})
    polynomial_y = RationalPolynomial({(0, 1): Fraction(1)}, {(0, 0): Fraction(1)})
    witness = ReferenceCellMapWitnesses(
        np.asarray((0,), dtype=np.int64),
        np.asarray(block.global_ids, dtype=np.int64),
        "triangle",
        "triangle",
        np.asarray([[[float(value) for value in point] for point in exact_corners]]),
        (exact_corners,),
        ((rational_x, polynomial_y),),
    )
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        mesh.topology_id,
        {block.name: block.global_ids},
        {block.name: block.vertices},
        block_parent_reference_maps={block.name: witness},
    )
    target_geometry = CellGeometrySpec(
        dict(zip(source_geometry.block_names, source_geometry.elements, strict=True)),
        dict(
            zip(source_geometry.block_names, source_geometry.geometry_dofs, strict=True)
        ),
        source_geometry.coordinates,
        restriction_source=restriction,
    )
    mesh_entry = CompositionEntry(
        mesh,
        entry_id="mesh/topology",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=mesh.topology_id,
        revision_id=mesh.mesh_id,
        semantics_id="cell-mesh",
    )
    source_geometry_entry = CompositionEntry(
        source_geometry,
        entry_id="geometry/source",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=source_geometry.geometry_layout_id,
        revision_id=cell_geometry_id(source_geometry),
        semantics_id="root-coordinate-map",
        dependencies=(mesh_entry.binding("structure"),),
    )
    target_geometry_entry = CompositionEntry(
        target_geometry,
        entry_id="geometry/target",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=target_geometry.geometry_layout_id,
        revision_id=cell_geometry_id(target_geometry),
        semantics_id="reference-restricted-coordinate-map",
        dependencies=(
            mesh_entry.binding("structure"),
            source_geometry_entry.binding("structure"),
        ),
    )
    composition = Composition(
        (mesh_entry, source_geometry_entry, target_geometry_entry),
        boundary_id="exact-reference-chart-arguments",
    )
    return composition, target_geometry, witness


def _rebuild_reference_chart_arguments_composition(value: object, /) -> Composition:
    from phydrax.discretization._nested_reference_types import (
        exact_reference_cell_vertices,
        ReferenceCellMapWitnesses,
        validate_reference_cell_map_witnesses,
    )

    if not isinstance(value, Composition) or len(value.entries) != 3:
        raise TypeError(
            "Reference chart archive requires mesh and both coordinate-map owners."
        )
    entries = {entry.entry_id: entry for entry in value.entries}
    mesh = _rebuild_continuation_value(entries["mesh/topology"].value)
    source_geometry = _rebuild_geometry_for_mesh(entries["geometry/source"].value, mesh)
    target_geometry = _rebuild_geometry_for_mesh(entries["geometry/target"].value, mesh)
    if not isinstance(mesh, CellMesh):
        raise TypeError("Reference chart archive lost its mesh owner.")
    if not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("Reference chart archive lost its source coordinate owner.")
    if not isinstance(target_geometry, CellGeometrySpec):
        raise TypeError("Reference chart archive lost its target coordinate owner.")
    restriction = target_geometry.restriction_source
    if restriction is None:
        raise ValueError("Reference chart target lost its restriction ancestry.")
    witness = restriction.block_parent_reference_maps["triangles"]
    if type(witness) is not ReferenceCellMapWitnesses:
        raise TypeError("Reference chart target lost its typed witness bank.")
    validate_reference_cell_map_witnesses(witness)
    arguments = witness.source_reference_chart_arguments
    if arguments is None:
        raise ValueError("Reference chart target lost its exact argument expressions.")
    expressions = arguments[0]
    rational_x, polynomial_y = expressions
    exact_vertices = witness.source_reference_exact_vertices
    if exact_vertices is None or not exact_vertices:
        raise ValueError(
            "Reference chart witness lost its exact source reference vertices."
        )
    if (
        not isinstance(rational_x, RationalPolynomial)
        or type(rational_x) is not RationalPolynomial
        or dict(rational_x.numerator) != {(1, 0): Fraction(1)}
        or dict(rational_x.denominator) != {(0, 0): Fraction(1)}
        or not isinstance(polynomial_y, RationalPolynomial)
        or type(polynomial_y) is not RationalPolynomial
        or dict(polynomial_y.numerator) != {(0, 1): Fraction(1)}
        or dict(polynomial_y.denominator) != {(0, 0): Fraction(1)}
        or exact_reference_cell_vertices(witness) != (exact_vertices[0],)
    ):
        raise ValueError(
            "Cold reference chart expressions lost their exact polynomial coefficients."
        )
    rebuilt_entries = tuple(
        CompositionEntry(
            rebuilt,
            entry_id=entries[entry_id].entry_id,
            role=entries[entry_id].role,
            owner_id=entries[entry_id].owner_id,
            structure_id=entries[entry_id].structure_id,
            revision_id=entries[entry_id].revision_id,
            semantics_id=entries[entry_id].semantics_id,
            dependencies=entries[entry_id].dependencies,
        )
        for entry_id, rebuilt in (
            ("mesh/topology", mesh),
            ("geometry/source", source_geometry),
            ("geometry/target", target_geometry),
        )
    )
    composition = Composition(rebuilt_entries, boundary_id=value.boundary_id)
    if composition.composition_id != value.composition_id:
        raise ValueError("Exact reference chart Composition identity changed.")
    return composition


def _write_reference_chart_arguments(directory: Path, /) -> None:
    composition, geometry, witness = _reference_chart_arguments_composition()
    write_model_artifact(
        directory / "reference-chart-arguments.phx", composition, limits=None
    )
    arguments = witness.source_reference_chart_arguments
    if arguments is None:
        raise ValueError("Reference chart fixture lost its exact argument bank.")
    rational_x, polynomial_y = arguments[0]
    if type(rational_x) is not RationalPolynomial:
        raise TypeError("Reference chart fixture must retain its rational x expression.")
    if type(polynomial_y) is not RationalPolynomial:
        raise TypeError("Reference chart fixture must retain its rational y expression.")
    restriction_source = geometry.restriction_source
    if restriction_source is None:
        raise ValueError("Reference chart fixture lost its restriction source.")
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "source_geometry_id": cell_geometry_id(
                    composition.value("geometry/source")
                ),
                "target_geometry_id": cell_geometry_id(geometry),
                "restriction_source_id": restriction_source.restriction_source_id,
                "rational_numerator": [
                    [list(exponent), *_exact_scalar_pair_for_archive(value)]
                    for exponent, value in rational_x.numerator.items()
                ],
                "rational_denominator": [
                    [list(exponent), *_exact_scalar_pair_for_archive(value)]
                    for exponent, value in rational_x.denominator.items()
                ],
                "polynomial_y_numerator": [
                    [list(exponent), *_exact_scalar_pair_for_archive(value)]
                    for exponent, value in polynomial_y.numerator.items()
                ],
                "polynomial_y_denominator": [
                    [list(exponent), *_exact_scalar_pair_for_archive(value)]
                    for exponent, value in polynomial_y.denominator.items()
                ],
            }
        )
    )


def _read_reference_chart_arguments(path: Path, expected: dict[str, object], /) -> None:
    composition = read_model_artifact(
        path,
        expected_type=Composition,
        restore=_rebuild_reference_chart_arguments_composition,
        limits=None,
    )
    geometry = composition.value("geometry/target")
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Cold exact reference chart target lost its coordinate owner.")
    source_geometry = composition.value("geometry/source")
    if not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("Cold exact reference chart source lost its coordinate owner.")
    restriction_source = geometry.restriction_source
    if restriction_source is None:
        raise ValueError("Cold exact reference chart target lost its restriction source.")
    if (
        composition.composition_id != expected["composition_id"]
        or cell_geometry_id(source_geometry) != expected["source_geometry_id"]
        or cell_geometry_id(geometry) != expected["target_geometry_id"]
        or restriction_source.restriction_source_id != expected["restriction_source_id"]
    ):
        raise ValueError("Cold exact reference chart owners changed identity.")
    print("reference-chart-arguments-cold-restored")


def _mapped_reference_domain_fixture() -> MappedReferenceDomain:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    vertex_ids = np.asarray((100, 101, 102, 103), dtype=np.int64)
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "roots",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((71,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=vertex_ids,
    )
    reference_domain = PiecewiseLinearDomain(
        points,
        np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64),
        np.tile(np.asarray(((0, -1),), dtype=np.int64), (4, 1)),
        ("solid",),
        source_id="archive-reference-tetrahedron",
        region_occurrence_paths=((),),
    )
    return MappedReferenceDomain(
        reference_domain,
        mesh,
        CellGeometrySpec.affine(mesh),
        np.zeros((1,), dtype=np.int64),
        source_id="archive-mapped-reference-source",
        source_revision="archive-mapped-reference-r1",
        coordinate_contract=SpatialCoordinateContract.si(),
        reference_domain_vertex_root_ids=vertex_ids,
    )


def _piecewise_duplicate_region_occurrence_domain() -> PiecewiseLinearDomain:
    vertices = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (3.0, 1.0),
        ),
        dtype=np.float64,
    )
    facets = np.asarray(
        ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)),
        dtype=np.int64,
    )
    facet_regions = np.asarray(
        ((0, -1), (0, -1), (0, -1), (1, -1), (1, -1), (1, -1)),
        dtype=np.int64,
    )
    return PiecewiseLinearDomain(
        vertices,
        facets,
        facet_regions,
        ("solid", "solid"),
        source_id="archive-duplicate-region-occurrences",
        region_occurrence_paths=(("assembly", "left"), ("assembly", "right")),
    )


def _piecewise_region_occurrence_metadata(
    domain: PiecewiseLinearDomain, /
) -> dict[str, object]:
    return {
        "source_id": domain.source_id,
        "source_revision": domain.source_revision,
        "domain_id": domain.domain_id,
        "region_ids": list(domain.region_ids),
        "region_occurrence_paths": [
            list(path) for path in domain.region_occurrence_paths
        ],
        "exact_region_measures": [
            _fraction_pair_for_archive(measure)
            for measure in domain.exact_region_measures()
        ],
    }


def _write_piecewise_region_occurrence_archive(directory: Path, /) -> None:
    domain = _piecewise_duplicate_region_occurrence_domain()
    write_model_artifact(
        directory / "piecewise-region-occurrences.phx", domain, limits=None
    )
    print(json.dumps(_piecewise_region_occurrence_metadata(domain), sort_keys=True))


def _read_piecewise_region_occurrence_archive(
    directory: Path,
    expected: dict[str, object],
    /,
) -> None:
    domain = read_model_artifact(
        directory / "piecewise-region-occurrences.phx",
        expected_type=PiecewiseLinearDomain,
        restore=_rebuild_continuation_value,
        limits=None,
    )
    if not isinstance(domain, PiecewiseLinearDomain):
        raise TypeError(
            "Cold region occurrence archive did not restore its typed domain."
        )
    if _piecewise_region_occurrence_metadata(domain) != expected:
        raise ValueError("Cold duplicate-region occurrence identities changed.")
    print("piecewise-region-occurrences-cold-restored")


def _mapped_reference_domain_composition(domain: MappedReferenceDomain, /) -> Composition:
    reference_domain_entry = CompositionEntry(
        domain.reference_domain,
        entry_id="mapped/reference-domain",
        role="topology",
        owner_id="phydrax.geometry",
        structure_id=domain.reference_domain.domain_id,
        revision_id=domain.reference_domain.source_revision,
        semantics_id="piecewise-linear-reference-domain",
    )
    mesh_entry = CompositionEntry(
        domain.reference_mesh,
        entry_id="mapped/root-mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=domain.reference_mesh.topology_id,
        revision_id=domain.reference_mesh.mesh_id,
        semantics_id="cell-mesh",
    )
    geometry_entry = CompositionEntry(
        domain.source_geometry,
        entry_id="mapped/root-geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=domain.source_geometry.geometry_layout_id,
        revision_id=domain.reference_mesh.geometry_id,
        semantics_id="cell-coordinate-map",
        dependencies=(mesh_entry.binding("structure"),),
    )
    mapped_entry = CompositionEntry(
        domain,
        entry_id="mapped/source-domain",
        role="topology",
        owner_id="phydrax.geometry",
        structure_id=domain.domain_id,
        revision_id=domain.source_revision,
        semantics_id="mapped-reference-domain",
        dependencies=(
            reference_domain_entry.binding("structure"),
            mesh_entry.binding("structure"),
            geometry_entry.binding("structure"),
        ),
    )
    return Composition(
        (reference_domain_entry, mesh_entry, geometry_entry, mapped_entry),
        boundary_id="mapped-reference-domain-cold-continuation",
    )


def _write_mapped_reference_domain_artifact(directory: Path, /) -> None:
    domain = _mapped_reference_domain_fixture()
    composition = _mapped_reference_domain_composition(domain)
    write_model_artifact(
        directory / "mapped-reference-domain.phx", composition, limits=None
    )
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "domain_id": domain.domain_id,
                "source_id": domain.source_id,
                "source_revision": domain.source_revision,
                "spatial_id": domain.coordinate_contract.spatial_id,
                "reference_domain_id": domain.reference_domain.domain_id,
                "reference_domain_region_occurrence_paths": [
                    list(path) for path in domain.reference_domain.region_occurrence_paths
                ],
                "reference_mesh_id": domain.reference_mesh_id,
                "reference_geometry_id": domain.reference_geometry_id,
                "reference_domain_vertex_root_ids": list(
                    domain.reference_domain_vertex_root_ids or ()
                ),
                "exact_region_measures": [
                    _exact_scalar_pair_for_archive(measure)
                    for measure in domain.exact_region_measures()
                ],
            }
        )
    )


def _rebuild_mapped_reference_domain_composition(
    value: object, expected: dict[str, object], /
) -> Composition:
    if not isinstance(value, Composition):
        raise TypeError(
            "Mapped-reference-domain continuation requires a Composition root."
        )
    entries = {entry.entry_id: entry for entry in value.entries}
    expected_ids = {
        "mapped/reference-domain",
        "mapped/root-mesh",
        "mapped/root-geometry",
        "mapped/source-domain",
    }
    if set(entries) != expected_ids:
        raise ValueError(
            "Mapped-reference-domain Composition differs from its restart schema."
        )
    reference_domain = _rebuild_continuation_value(
        entries["mapped/reference-domain"].value
    )
    mesh = _rebuild_continuation_value(entries["mapped/root-mesh"].value)
    geometry = _rebuild_geometry_for_mesh(entries["mapped/root-geometry"].value, mesh)
    domain = _rebuild_continuation_value(entries["mapped/source-domain"].value)
    if (
        not isinstance(reference_domain, PiecewiseLinearDomain)
        or not isinstance(mesh, CellMesh)
        or not isinstance(geometry, CellGeometrySpec)
        or not isinstance(domain, MappedReferenceDomain)
    ):
        raise TypeError("Cold mapped-reference-domain restore lost a canonical owner.")
    measures = domain.exact_region_measures()
    if (
        domain.domain_id != expected["domain_id"]
        or domain.source_id != expected["source_id"]
        or domain.source_revision != expected["source_revision"]
        or domain.coordinate_contract.spatial_id != expected["spatial_id"]
        or domain.reference_domain.domain_id != expected["reference_domain_id"]
        or [list(path) for path in domain.reference_domain.region_occurrence_paths]
        != expected["reference_domain_region_occurrence_paths"]
        or domain.reference_mesh_id != expected["reference_mesh_id"]
        or domain.reference_geometry_id != expected["reference_geometry_id"]
        or domain.reference_domain_vertex_root_ids
        != _expected_sequence(expected["reference_domain_vertex_root_ids"])
        or [_exact_scalar_pair_for_archive(measure) for measure in measures]
        != expected["exact_region_measures"]
        or mesh.mesh_id != domain.reference_mesh.mesh_id
        or cell_geometry_id(geometry) != domain.reference_geometry_id
    ):
        raise ValueError(
            "Cold mapped-reference-domain continuation changed root ancestry."
        )
    rebuilt_values = {
        "mapped/reference-domain": reference_domain,
        "mapped/root-mesh": mesh,
        "mapped/root-geometry": geometry,
        "mapped/source-domain": domain,
    }
    rebuilt_entries = tuple(
        CompositionEntry(
            rebuilt_values[entry_id],
            entry_id=entries[entry_id].entry_id,
            role=entries[entry_id].role,
            owner_id=entries[entry_id].owner_id,
            structure_id=entries[entry_id].structure_id,
            revision_id=entries[entry_id].revision_id,
            semantics_id=entries[entry_id].semantics_id,
            dependencies=entries[entry_id].dependencies,
        )
        for entry_id in (
            "mapped/reference-domain",
            "mapped/root-mesh",
            "mapped/root-geometry",
            "mapped/source-domain",
        )
    )
    composition = Composition(rebuilt_entries, boundary_id=value.boundary_id)
    if composition.composition_id != expected["composition_id"]:
        raise ValueError("Mapped-reference-domain Composition identity changed.")
    return composition


def _read_mapped_reference_domain_artifact(
    directory: Path, expected: dict[str, object], /
) -> None:
    composition = read_model_artifact(
        directory / "mapped-reference-domain.phx",
        expected_type=Composition,
        restore=lambda value: _rebuild_mapped_reference_domain_composition(
            value, expected
        ),
        limits=None,
    )
    if composition.composition_id != expected["composition_id"]:
        raise ValueError("Mapped-reference-domain Composition identity changed.")
    print("mapped-reference-domain-cold-restored")


def _mapped_reference_boundary_query_case() -> MappedReferenceBoundarySource:
    from tests.unit.meshing.test_native_mapped_hex_routes import _mapped_cube_source

    authority, _ = _mapped_cube_source()
    query = authority.boundary_query()
    from phydrax.geometry._mapped_reference_boundary import MappedReferenceBoundarySource

    if not isinstance(query, MappedReferenceBoundarySource):
        raise TypeError("Mapped PLC source did not produce its canonical boundary query.")
    return query


def _write_mapped_reference_boundary_query(directory: Path, /) -> None:
    query = _mapped_reference_boundary_query_case()
    query.require_current()
    entry = CompositionEntry(
        query,
        entry_id="mapped/boundary-query",
        role="topology",
        owner_id="phydrax.geometry",
        structure_id=query.source_query_id,
        revision_id=query.source_revision,
        semantics_id="mapped-reference-boundary-query",
    )
    composition = Composition((entry,), boundary_id="mapped-boundary-query:cold")
    write_model_artifact(
        directory / "mapped-reference-boundary-query.phx", composition, limits=None
    )
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "source_query_id": query.source_query_id,
                "source_id": query.source_id,
                "source_revision": query.source_revision,
                "spatial_id": query.coordinate_contract.spatial_id,
                "domain_id": query.domain.domain_id,
                "authority_id": query.authority_id,
                "relation_id": query.relation_id,
                "face_chart_ids": list(query.face_chart_ids),
                "edge_chart_ids": list(query.edge_chart_ids),
            }
        )
    )


def _rebuild_mapped_reference_boundary_query_composition(
    value: object, expected: dict[str, object], /
) -> Composition:
    from phydrax.geometry._mapped_reference_boundary import MappedReferenceBoundarySource

    if not isinstance(value, Composition) or set(value.entry_ids) != {
        "mapped/boundary-query"
    }:
        raise TypeError("Mapped boundary-query archive requires its Composition owner.")
    entry = value.entries[0]
    query = _rebuild_continuation_value(entry.value)
    if not isinstance(query, MappedReferenceBoundarySource):
        raise TypeError(
            "Mapped boundary-query recipe did not restore its canonical query."
        )
    query.require_current()
    if (
        value.composition_id != expected["composition_id"]
        or query.source_query_id != expected["source_query_id"]
        or query.source_id != expected["source_id"]
        or query.source_revision != expected["source_revision"]
        or query.coordinate_contract.spatial_id != expected["spatial_id"]
        or query.domain.domain_id != expected["domain_id"]
        or query.coordinate_contract is not query.domain.coordinate_contract
        or query.authority_id != expected["authority_id"]
        or query.relation_id != expected["relation_id"]
        or list(query.face_chart_ids) != expected["face_chart_ids"]
        or list(query.edge_chart_ids) != expected["edge_chart_ids"]
        or entry.role != "topology"
        or entry.owner_id != "phydrax.geometry"
        or entry.structure_id != query.source_query_id
        or entry.revision_id != query.source_revision
        or entry.semantics_id != "mapped-reference-boundary-query"
        or entry.dependencies
    ):
        raise ValueError(
            "Cold mapped boundary query differs from its reconstructed source authority."
        )
    rebuilt_entry = CompositionEntry(
        query,
        entry_id=entry.entry_id,
        role=entry.role,
        owner_id=entry.owner_id,
        structure_id=entry.structure_id,
        revision_id=entry.revision_id,
        semantics_id=entry.semantics_id,
        dependencies=entry.dependencies,
    )
    composition = Composition((rebuilt_entry,), boundary_id=value.boundary_id)
    if composition.composition_id != value.composition_id:
        raise ValueError(
            "Mapped boundary-query Composition identity changed on reconstruction."
        )
    return composition


def _read_mapped_reference_boundary_query(
    path: Path, expected: dict[str, object], /
) -> None:
    read_model_artifact(
        path,
        expected_type=Composition,
        restore=lambda value: _rebuild_mapped_reference_boundary_query_composition(
            value, expected
        ),
        limits=None,
    )
    print("mapped-reference-boundary-query-cold-restored")


def _coordinate_map_composition() -> Composition:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    carrier = CellMesh.from_simplices(
        points,
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        dimension=3,
        block_name="power-carrier-tets",
        vertex_global_ids=np.asarray((31, 11, 41, 21), dtype=np.int64),
        cell_global_ids=np.asarray((105,), dtype=np.int64),
        numeric_version="power-carrier-v7",
    )
    restricted_cells = SimpleNamespace(
        vertices=points,
        vertex_site_offsets=np.asarray((0, 1, 2, 3, 4), dtype=np.int64),
        vertex_sites=np.zeros((4,), dtype=np.int32),
        vertex_carriers=np.asarray(
            ((0, -1, -1, -1), (1, -1, -1, -1), (2, -1, -1, -1), (3, -1, -1, -1)),
            dtype=np.int32,
        ),
        cell_sites=np.asarray((0,), dtype=np.int32),
        cell_regions=np.asarray((7,), dtype=np.int32),
    )
    power_source = ExactPowerCellGeometrySource(
        np.asarray(((0.2, 0.2, 0.2),), dtype=np.float64),
        np.asarray((0.0,), dtype=np.float64),
        carrier,
        restricted_cells,
        source_id="archive-power-source",
        source_revision="archive-power-r7",
        domain_source=_archive_tetra_source("archive-power-domain"),
    )
    power_output_mesh = CellMesh.from_simplices(
        points,
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        dimension=3,
        block_name="power-output-tets",
    )
    power_geometry = CellGeometrySpec.power(power_output_mesh, power_source)
    selected_rows = np.asarray((1, 2, 3), dtype=np.int64)
    selected_source = ExactPowerCellGeometrySource(
        power_source.site_points,
        power_source.site_weights,
        carrier,
        restricted_cells,
        source_id=power_source.source_id,
        source_revision=power_source.source_revision,
        domain_source=power_source.domain_source,
        exact_site_points=power_source.exact_site_points,
        output_vertex_indices=selected_rows,
    )
    selected_mesh = CellMesh.from_simplices(
        points[selected_rows],
        np.asarray(((0, 1, 2),), dtype=np.int64),
        dimension=2,
        block_name="power-selected-output-triangle",
        vertex_global_ids=np.asarray((11, 41, 21), dtype=np.int64),
    )
    selected_geometry = CellGeometrySpec.power(selected_mesh, selected_source)
    power_extraction = extract_volume_hexes(
        power_output_mesh, source_geometry=power_geometry
    )
    power_refined_geometry = power_extraction.geometry
    if not isinstance(power_refined_geometry, CellGeometrySpec):
        raise TypeError("Power extraction lost its exact restricted coordinate geometry.")
    if type(power_refined_geometry.source_map) is not _RestrictedCoordinateMap:
        raise TypeError(
            "Refined power-map fixture did not retain its restricted map owner."
        )

    from phydrax.meshing._hex_frame_records import HexFrameIntegerLayout
    from tests.unit.meshing.test_hex_cut_generation import _simplex_plc_and_fill

    cut_source, cut_fill = _simplex_plc_and_fill()
    cut_schedule = NativeHexGridSchedule(
        "frame_grid",
        minimum_scaled_jacobian=1.0e-8,
        minimum_mean_ratio=1.0e-8,
        maximum_aspect_ratio=1.0e8,
    )
    axis = tuple(Fraction(value, 4) for value in (-1, 1, 3, 5))
    frame_layout = HexFrameIntegerLayout(
        np.asarray(cut_fill.blocks[0].global_ids, dtype=np.int64),
        np.zeros((cut_fill.blocks[0].cell_count,), dtype=np.int64),
        np.zeros((cut_fill.blocks[0].cell_count, 3), dtype=np.int64),
        np.eye(3, dtype=np.float64)[None, :, :],
        np.zeros((1, 3), dtype=np.float64),
        ((axis, axis, axis),),
        np.empty((0, 4), dtype=np.int64),
        np.empty((0, 3, 3), dtype=np.int64),
        np.empty((0, 3), dtype=np.int64),
        np.ones((1,), dtype=np.int8),
        cut_source.complex_id,
        cut_fill.mesh_id,
        cut_fill.topology_id,
        cut_schedule.schedule_id,
        "archive-cut-frame",
    )
    if not isinstance(frame_layout, _HexFrameIntegerLayout):
        raise TypeError(
            "Archive cut fixture must retain its exact frame-layout protocol."
        )
    cut_plan = prepare_balanced_cut_grid(
        cut_source,
        cut_fill,
        cut_schedule,
        0.5,
        source_id="archive-cut-source",
        source_revision="archive-cut-r3",
        frame_layout=frame_layout,
    )
    cut_result = generate_balanced_cut_hexes(
        cut_source,
        cut_fill,
        np.asarray((0,), dtype=np.int64),
        cut_plan,
        cut_schedule,
    )
    if type(cut_result.geometry.source_map) is not _CutCoordinateMap:
        raise TypeError(
            "Cut-map archive fixture did not retain the native coordinate map."
        )

    power_domain_entry = CompositionEntry(
        power_source.domain_source,
        entry_id="power/domain",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=power_source.domain_source.complex_id,
        revision_id=power_source.domain_source.complex_id,
        semantics_id="piecewise-linear-power-domain",
    )
    power_carrier_entry = CompositionEntry(
        carrier,
        entry_id="power/carrier-mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=carrier.topology_id,
        revision_id=carrier.mesh_id,
        semantics_id="exact-power-carrier-mesh",
    )
    power_mesh_entry = CompositionEntry(
        power_output_mesh,
        entry_id="power/output-mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=power_output_mesh.topology_id,
        revision_id=power_output_mesh.mesh_id,
        semantics_id="exact-power-output-mesh",
    )
    power_geometry_entry = CompositionEntry(
        power_geometry,
        entry_id="power/geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=power_geometry.geometry_layout_id,
        revision_id=cell_geometry_id(power_geometry),
        semantics_id="exact-power-coordinate-map",
        dependencies=(
            power_domain_entry.binding("structure"),
            power_carrier_entry.binding("structure"),
            power_carrier_entry.binding("revision"),
            power_mesh_entry.binding("structure"),
        ),
    )
    selected_mesh_entry = CompositionEntry(
        selected_mesh,
        entry_id="power/selected-mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=selected_mesh.topology_id,
        revision_id=selected_mesh.mesh_id,
        semantics_id="selected-exact-power-output-mesh",
        dependencies=(
            power_domain_entry.binding("structure"),
            power_carrier_entry.binding("structure"),
        ),
    )
    selected_geometry_entry = CompositionEntry(
        selected_geometry,
        entry_id="power/selected-geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=selected_geometry.geometry_layout_id,
        revision_id=cell_geometry_id(selected_geometry),
        semantics_id="selected-row-exact-power-coordinate-map",
        dependencies=(
            selected_mesh_entry.binding("structure"),
            power_domain_entry.binding("structure"),
            power_carrier_entry.binding("structure"),
            power_carrier_entry.binding("revision"),
        ),
    )
    power_refined_mesh_entry = CompositionEntry(
        power_extraction.mesh,
        entry_id="power/refined-mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=power_extraction.mesh.topology_id,
        revision_id=power_extraction.mesh.mesh_id,
        semantics_id="refined-power-output-mesh",
        dependencies=(power_mesh_entry.binding("structure"),),
    )
    power_refined_geometry_entry = CompositionEntry(
        power_refined_geometry,
        entry_id="power/refined-geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=power_refined_geometry.geometry_layout_id,
        revision_id=cell_geometry_id(power_refined_geometry),
        semantics_id="restricted-power-coordinate-map",
        dependencies=(
            power_mesh_entry.binding("structure"),
            power_mesh_entry.binding("revision"),
            power_geometry_entry.binding("structure"),
            power_geometry_entry.binding("revision"),
        ),
    )
    cut_domain_entry = CompositionEntry(
        cut_source,
        entry_id="cut/domain",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=cut_source.complex_id,
        revision_id=cut_source.complex_id,
        semantics_id="piecewise-linear-cut-domain",
    )
    cut_fill_entry = CompositionEntry(
        cut_fill,
        entry_id="cut/fill-mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=cut_fill.topology_id,
        revision_id=cut_fill.mesh_id,
        semantics_id="balanced-cut-fill-mesh",
    )
    cut_mesh_entry = CompositionEntry(
        cut_result.mesh,
        entry_id="cut/output-mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=cut_result.mesh.topology_id,
        revision_id=cut_result.mesh.mesh_id,
        semantics_id="balanced-cut-output-mesh",
    )
    cut_geometry_entry = CompositionEntry(
        cut_result.geometry,
        entry_id="cut/geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=cut_result.geometry.geometry_layout_id,
        revision_id=cell_geometry_id(cut_result.geometry),
        semantics_id="exact-balanced-cut-coordinate-map",
        dependencies=(
            cut_domain_entry.binding("structure"),
            cut_fill_entry.binding("structure"),
            cut_fill_entry.binding("revision"),
            cut_mesh_entry.binding("structure"),
        ),
    )
    return Composition(
        (
            power_domain_entry,
            power_carrier_entry,
            power_mesh_entry,
            power_geometry_entry,
            selected_mesh_entry,
            selected_geometry_entry,
            power_refined_mesh_entry,
            power_refined_geometry_entry,
            cut_domain_entry,
            cut_fill_entry,
            cut_mesh_entry,
            cut_geometry_entry,
        ),
        boundary_id="native-coordinate-map-continuation",
    )


def _write_coordinate_map_artifact(directory: Path, /) -> None:
    composition = _coordinate_map_composition()
    write_model_artifact(directory / "coordinate-maps.phx", composition, limits=None)
    entries = {entry.entry_id: entry for entry in composition.entries}
    power_geometry = entries["power/geometry"].value
    selected_geometry = entries["power/selected-geometry"].value
    restricted_geometry = entries["power/refined-geometry"].value
    cut_geometry = entries["cut/geometry"].value
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "power_source_map_id": power_geometry.source_map.source_map_id,
                "power_geometry_id": cell_geometry_id(power_geometry),
                "selected_power_source_map_id": selected_geometry.source_map.source_map_id,
                "selected_power_geometry_id": cell_geometry_id(selected_geometry),
                "selected_power_output_vertex_indices": np.asarray(
                    selected_geometry.source_map.output_vertex_indices,
                    dtype=np.int64,
                ).tolist(),
                "selected_power_coordinates": [
                    [str(value) for value in row]
                    for row in selected_geometry.source_coordinates()
                ],
                "selected_power_exact_coordinate_bank": _exact_coordinate_bank_payload(
                    selected_geometry.exact_coordinate_bank
                ),
                "restricted_source_map_id": entries[
                    "power/refined-geometry"
                ].value.source_map.source_map_id,
                "restricted_geometry_id": cell_geometry_id(
                    entries["power/refined-geometry"].value
                ),
                "restricted_source_mesh_id": entries[
                    "power/refined-geometry"
                ].value.source_map.source_mesh_id,
                "restricted_target_mesh_id": entries[
                    "power/refined-geometry"
                ].value.source_map.target_mesh_id,
                "power_coordinates": [
                    [str(value) for value in row]
                    for row in power_geometry.source_coordinates()
                ],
                "power_exact_coordinate_bank": _exact_coordinate_bank_payload(
                    power_geometry.exact_coordinate_bank
                ),
                "restricted_exact_coordinate_bank": _exact_coordinate_bank_payload(
                    restricted_geometry.exact_coordinate_bank
                ),
                "cut_source_map_id": cut_geometry.source_map.source_map_id,
                "cut_geometry_id": cell_geometry_id(cut_geometry),
                "cut_coordinates": [
                    [str(value) for value in row]
                    for row in cut_geometry.source_coordinates()
                ],
                "cut_exact_coordinate_bank": _exact_coordinate_bank_payload(
                    cut_geometry.exact_coordinate_bank
                ),
            }
        )
    )


def _rebuild_coordinate_map_composition(value: object, /) -> Composition:
    if not isinstance(value, Composition):
        raise TypeError("Coordinate-map continuation requires a Composition root.")
    entries = {entry.entry_id: entry for entry in value.entries}
    expected_ids = {
        "power/domain",
        "power/carrier-mesh",
        "power/output-mesh",
        "power/geometry",
        "power/selected-mesh",
        "power/selected-geometry",
        "power/refined-mesh",
        "power/refined-geometry",
        "cut/domain",
        "cut/fill-mesh",
        "cut/output-mesh",
        "cut/geometry",
    }
    if set(entries) != expected_ids:
        raise ValueError(
            "Coordinate-map Composition entries differ from its explicit restart schema."
        )
    rebuilt_values: dict[str, object] = {
        "power/domain": restore_model_archive_value(entries["power/domain"].value),
        "power/carrier-mesh": _rebuild_continuation_value(
            entries["power/carrier-mesh"].value
        ),
        "power/output-mesh": _rebuild_continuation_value(
            entries["power/output-mesh"].value
        ),
        "power/selected-mesh": _rebuild_continuation_value(
            entries["power/selected-mesh"].value
        ),
        "power/refined-mesh": _rebuild_continuation_value(
            entries["power/refined-mesh"].value
        ),
        "cut/domain": restore_model_archive_value(entries["cut/domain"].value),
        "cut/fill-mesh": _rebuild_continuation_value(entries["cut/fill-mesh"].value),
        "cut/output-mesh": _rebuild_continuation_value(entries["cut/output-mesh"].value),
    }
    power_geometry = _rebuild_geometry_for_mesh(
        entries["power/geometry"].value, rebuilt_values["power/output-mesh"]
    )
    selected_geometry = _rebuild_geometry_for_mesh(
        entries["power/selected-geometry"].value,
        rebuilt_values["power/selected-mesh"],
    )
    power_refined_geometry = _rebuild_geometry_for_mesh(
        entries["power/refined-geometry"].value, rebuilt_values["power/refined-mesh"]
    )
    cut_geometry = _rebuild_geometry_for_mesh(
        entries["cut/geometry"].value, rebuilt_values["cut/output-mesh"]
    )
    rebuilt_values["power/geometry"] = power_geometry
    rebuilt_values["power/selected-geometry"] = selected_geometry
    rebuilt_values["power/refined-geometry"] = power_refined_geometry
    rebuilt_values["cut/geometry"] = cut_geometry
    if not all(
        isinstance(item, CellGeometrySpec)
        for item in (
            power_geometry,
            selected_geometry,
            power_refined_geometry,
            cut_geometry,
        )
    ):
        raise TypeError(
            "Coordinate-map Composition did not reconstruct its geometry owners."
        )
    power_map = power_geometry.source_map
    selected_map = selected_geometry.source_map
    restricted_map = power_refined_geometry.source_map
    cut_map = cut_geometry.source_map
    power_domain = rebuilt_values["power/domain"]
    cut_domain = rebuilt_values["cut/domain"]
    power_carrier_mesh = rebuilt_values["power/carrier-mesh"]
    power_output_mesh = rebuilt_values["power/output-mesh"]
    power_selected_mesh = rebuilt_values["power/selected-mesh"]
    power_refined_mesh = rebuilt_values["power/refined-mesh"]
    cut_fill_mesh = rebuilt_values["cut/fill-mesh"]
    cut_output_mesh = rebuilt_values["cut/output-mesh"]
    if not isinstance(power_domain, PiecewiseLinearComplex):
        raise TypeError("Coordinate-map Composition lost its power source domain.")
    if type(power_domain) is not PiecewiseLinearComplex:
        raise TypeError("Coordinate-map Composition has a noncanonical power domain.")
    if not isinstance(cut_domain, PiecewiseLinearComplex):
        raise TypeError("Coordinate-map Composition lost its cut source domain.")
    if type(cut_domain) is not PiecewiseLinearComplex:
        raise TypeError("Coordinate-map Composition has a noncanonical cut domain.")
    if (
        not isinstance(power_carrier_mesh, CellMesh)
        or not isinstance(power_output_mesh, CellMesh)
        or not isinstance(power_selected_mesh, CellMesh)
        or not isinstance(power_refined_mesh, CellMesh)
        or not isinstance(cut_fill_mesh, CellMesh)
        or not isinstance(cut_output_mesh, CellMesh)
    ):
        raise TypeError(
            "Coordinate-map Composition lost a required source or mesh owner."
        )
    if not isinstance(power_map, ExactPowerCellGeometrySource):
        raise TypeError("Coordinate-map composition lost its exact power map.")
    if type(power_map) is not ExactPowerCellGeometrySource:
        raise TypeError("Coordinate-map composition has a noncanonical power map.")
    if not isinstance(selected_map, ExactPowerCellGeometrySource):
        raise TypeError("Coordinate-map composition lost its selected power map.")
    if type(selected_map) is not ExactPowerCellGeometrySource:
        raise TypeError("Coordinate-map composition has a noncanonical selected map.")
    if not isinstance(restricted_map, _RestrictedCoordinateMap):
        raise TypeError("Coordinate-map composition lost its restricted map.")
    if type(restricted_map) is not _RestrictedCoordinateMap:
        raise TypeError("Coordinate-map composition has a noncanonical restricted map.")
    if not isinstance(cut_map, _CutCoordinateMap):
        raise TypeError("Coordinate-map composition lost its cut map.")
    if type(cut_map) is not _CutCoordinateMap:
        raise TypeError("Coordinate-map composition has a noncanonical cut map.")
    if not isinstance(power_map.domain_source, PiecewiseLinearComplex):
        raise TypeError("Exact power map lost its exact PLC source.")
    if type(power_map.domain_source) is not PiecewiseLinearComplex:
        raise TypeError("Exact power map has a noncanonical PLC source.")
    if not isinstance(selected_map.domain_source, PiecewiseLinearComplex):
        raise TypeError("Selected exact power map lost its PLC source.")
    if type(selected_map.domain_source) is not PiecewiseLinearComplex:
        raise TypeError("Selected exact power map has a noncanonical PLC source.")
    if (
        selected_map.domain_source.complex_id != power_domain.complex_id
        or not np.array_equal(
            np.asarray(selected_map.output_vertex_indices, dtype=np.int64),
            np.asarray((1, 2, 3), dtype=np.int64),
        )
        or power_map.domain_source.complex_id != power_domain.complex_id
        or restricted_map.source_mesh.mesh_id != power_output_mesh.mesh_id
        or cell_geometry_id(restricted_map.source_geometry)
        != cell_geometry_id(power_geometry)
        or cut_map.source_complex.complex_id != cut_domain.complex_id
        or cut_map.source_mesh.mesh_id != cut_fill_mesh.mesh_id
    ):
        raise ValueError(
            "Cold coordinate maps do not retain their independent source and mesh owners."
        )
    power_map.require_carrier(power_carrier_mesh)
    selected_map.require_carrier(power_carrier_mesh)
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
                CompositionDependency(
                    dependency.entry_id, dependency.facet, dependency.bound_id
                )
                for dependency in entry.dependencies
            ),
        )
        for entry in value.entries
    )
    composition = Composition(rebuilt_entries, boundary_id=value.boundary_id)
    # Stored geometry values hold archive projections of their FE elements; each
    # entry's revision is the identity of the geometry it was written from.
    if (
        composition.composition_id != value.composition_id
        or cell_geometry_id(power_geometry) != entries["power/geometry"].revision_id
        or cell_geometry_id(cut_geometry) != entries["cut/geometry"].revision_id
        or cell_geometry_id(power_refined_geometry)
        != entries["power/refined-geometry"].revision_id
        or cell_geometry_id(selected_geometry)
        != entries["power/selected-geometry"].revision_id
    ):
        raise ValueError("Cold coordinate-map Composition identity changed.")
    return composition


def _read_coordinate_map_artifact(
    directory: Path, expected: dict[str, object], /
) -> None:
    composition = read_model_artifact(
        directory / "coordinate-maps.phx",
        expected_type=Composition,
        restore=_rebuild_coordinate_map_composition,
        limits=None,
    )
    entries = {entry.entry_id: entry for entry in composition.entries}
    power_geometry = entries["power/geometry"].value
    selected_geometry = entries["power/selected-geometry"].value
    cut_geometry = entries["cut/geometry"].value
    refined_geometry = entries["power/refined-geometry"].value
    if (
        not isinstance(power_geometry, CellGeometrySpec)
        or not isinstance(selected_geometry, CellGeometrySpec)
        or not isinstance(cut_geometry, CellGeometrySpec)
        or not isinstance(refined_geometry, CellGeometrySpec)
    ):
        raise TypeError("Cold coordinate-map archive lost its geometry owners.")
    if any(
        type(geometry) is not CellGeometrySpec
        for geometry in (
            power_geometry,
            selected_geometry,
            cut_geometry,
            refined_geometry,
        )
    ):
        raise TypeError("Cold coordinate-map archive has a noncanonical geometry owner.")
    power_map = power_geometry.source_map
    if not isinstance(power_map, ExactPowerCellGeometrySource):
        raise TypeError("Cold coordinate-map archive lost its exact power source map.")
    if type(power_map) is not ExactPowerCellGeometrySource:
        raise TypeError("Cold coordinate-map archive has a noncanonical power map.")
    selected_map = selected_geometry.source_map
    if not isinstance(selected_map, ExactPowerCellGeometrySource):
        raise TypeError("Cold coordinate-map archive lost its selected power source map.")
    if type(selected_map) is not ExactPowerCellGeometrySource:
        raise TypeError("Cold coordinate-map archive has a noncanonical selected map.")
    restricted_map = refined_geometry.source_map
    if not isinstance(restricted_map, _RestrictedCoordinateMap):
        raise TypeError("Cold coordinate-map archive lost its restricted source map.")
    if type(restricted_map) is not _RestrictedCoordinateMap:
        raise TypeError("Cold coordinate-map archive has a noncanonical restricted map.")
    cut_map = cut_geometry.source_map
    if not isinstance(cut_map, _CutCoordinateMap):
        raise TypeError("Cold coordinate-map archive lost its cut source map.")
    if type(cut_map) is not _CutCoordinateMap:
        raise TypeError("Cold coordinate-map archive has a noncanonical cut map.")
    if (
        composition.composition_id != expected["composition_id"]
        or power_map.source_map_id != expected["power_source_map_id"]
        or cell_geometry_id(power_geometry) != expected["power_geometry_id"]
        or selected_map.source_map_id != expected["selected_power_source_map_id"]
        or cell_geometry_id(selected_geometry) != expected["selected_power_geometry_id"]
        or np.asarray(selected_map.output_vertex_indices, dtype=np.int64).tolist()
        != expected["selected_power_output_vertex_indices"]
        or [
            [str(value) for value in row]
            for row in selected_geometry.source_coordinates()
        ]
        != expected["selected_power_coordinates"]
        or _exact_coordinate_bank_payload(selected_geometry.exact_coordinate_bank)
        != expected["selected_power_exact_coordinate_bank"]
        or [[str(value) for value in row] for row in power_geometry.source_coordinates()]
        != expected["power_coordinates"]
        or _exact_coordinate_bank_payload(power_geometry.exact_coordinate_bank)
        != expected["power_exact_coordinate_bank"]
        or restricted_map.source_map_id != expected["restricted_source_map_id"]
        or cell_geometry_id(refined_geometry) != expected["restricted_geometry_id"]
        or _exact_coordinate_bank_payload(refined_geometry.exact_coordinate_bank)
        != expected["restricted_exact_coordinate_bank"]
        or restricted_map.source_mesh_id != expected["restricted_source_mesh_id"]
        or restricted_map.target_mesh_id != expected["restricted_target_mesh_id"]
        or cut_map.source_map_id != expected["cut_source_map_id"]
        or cell_geometry_id(cut_geometry) != expected["cut_geometry_id"]
        or [[str(value) for value in row] for row in cut_geometry.source_coordinates()]
        != expected["cut_coordinates"]
        or _exact_coordinate_bank_payload(cut_geometry.exact_coordinate_bank)
        != expected["cut_exact_coordinate_bank"]
    ):
        raise ValueError(
            "Cold coordinate-map continuation changed source identity or exact coordinates."
        )
    print("coordinate-map-cold-restored")


def _single_rank_owner_directory_group(
    group_id: str = "archive-periodic-power-owner-directory", /
) -> ExecutionGroup:
    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup

    devices = tuple(jax.local_devices())
    if not devices:
        raise RuntimeError("Owner-directory archive requires a local JAX device.")
    process = jax.process_index()
    device = devices[0]
    specification = ExecutionGroupSpec(
        group_id,
        (process,),
        ((process, device.id),),
    )
    return ExecutionGroup(specification, (device,))


def _periodic_power_map_composition() -> Composition:
    from phydrax.discretization._exact_power_geometry import (
        ExactPeriodicPowerSourceContext,
    )
    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.meshing._distribution import ShardedOwnerDirectoryWitness
    from phydrax.meshing._periodic import PeriodicPointOrbits
    from phydrax.meshing.providers._native_periodic_power import periodic_power_mesh

    cell = PeriodicCell(np.eye(3, dtype=np.float64))
    fractional = np.asarray(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)), dtype=np.float64)
    source = PeriodicPointOrbits(
        cell,
        np.asarray(cell.cartesian(fractional), dtype=np.float64),
        np.asarray((101, 202), dtype=np.int64),
    )
    construction = periodic_power_mesh(
        source, np.asarray((1.0 / 16.0, 0.0), dtype=np.float64)
    )
    mesh = construction.mesh
    geometry = construction.geometry
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Periodic power construction lost its exact source geometry.")
    source_map = geometry.source_map
    if not isinstance(source_map, ExactPowerCellGeometrySource):
        raise TypeError("Periodic power geometry lost its ExactPowerCellGeometrySource.")
    if type(source_map) is not ExactPowerCellGeometrySource:
        raise TypeError("Periodic power geometry has a noncanonical exact source map.")
    context = source_map.periodic_source_context
    if not isinstance(context, ExactPeriodicPowerSourceContext):
        raise TypeError("Periodic power geometry lost its typed periodic source context.")
    if type(context) is not ExactPeriodicPowerSourceContext:
        raise TypeError("Periodic power geometry has a noncanonical source context.")
    context.require_periodic_mesh(mesh)
    source_entry = CompositionEntry(
        source,
        entry_id="periodic-power/source",
        role="topology",
        owner_id="phydrax.meshing",
        structure_id=source.orbits_id,
        revision_id=source.orbits_id,
        semantics_id="authored-periodic-point-orbits",
    )
    mesh_entry = CompositionEntry(
        mesh,
        entry_id="periodic-power/mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=mesh.topology_id,
        revision_id=mesh.mesh_id,
        semantics_id="periodic-power-quotient-mesh",
        dependencies=(source_entry.binding("structure"),),
    )
    geometry_entry = CompositionEntry(
        geometry,
        entry_id="periodic-power/geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=geometry.geometry_layout_id,
        revision_id=cell_geometry_id(geometry),
        semantics_id="exact-periodic-power-coordinate-map",
        dependencies=(
            source_entry.binding("structure"),
            mesh_entry.binding("structure"),
        ),
    )
    source_ids = np.asarray(source.point_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    source_content_id = source.orbits_id
    witness = ShardedOwnerDirectoryWitness(
        _single_rank_owner_directory_group(),
        0,
        source_content_id,
        "periodic-power/owner-directory",
        source_ids,
        np.zeros(source_ids.shape, dtype=np.int32),
        np.ones(source_ids.shape, dtype=np.bool_),
        cell_ids,
        np.zeros(cell_ids.shape, dtype=np.int32),
        vertex_ids,
        np.zeros(vertex_ids.shape, dtype=np.int32),
        source_authority_id=source_entry.record_id,
        source_map_id=source_map.source_map_id,
        source_context_id=context.context_id,
    )
    return Composition(
        (
            source_entry,
            mesh_entry,
            geometry_entry,
            witness.composition_entry(),
        ),
        boundary_id="periodic-power-source-context-cold-restore",
    )


def _periodic_power_map_archive_metadata(
    composition: Composition, /
) -> dict[str, object]:
    from phydrax.discretization._exact_power_geometry import (
        ExactPeriodicPowerSourceContext,
    )
    from phydrax.meshing._distribution import ShardedOwnerDirectoryWitness
    from phydrax.meshing._periodic import PeriodicPointOrbits

    entries = {entry.entry_id: entry for entry in composition.entries}
    source = entries["periodic-power/source"].value
    mesh = entries["periodic-power/mesh"].value
    geometry = entries["periodic-power/geometry"].value
    witness = entries["periodic-power/owner-directory"].value
    if (
        not isinstance(source, PeriodicPointOrbits)
        or not isinstance(mesh, CellMesh)
        or not isinstance(geometry, CellGeometrySpec)
        or not isinstance(witness, ShardedOwnerDirectoryWitness)
    ):
        raise TypeError("Periodic power archive lost a typed source or geometry owner.")
    if type(source) is not PeriodicPointOrbits:
        raise TypeError("Periodic power source has a noncanonical owner type.")
    if type(mesh) is not CellMesh:
        raise TypeError("Periodic power mesh has a noncanonical owner type.")
    if type(geometry) is not CellGeometrySpec:
        raise TypeError("Periodic power geometry has a noncanonical owner type.")
    if type(witness) is not ShardedOwnerDirectoryWitness:
        raise TypeError("Periodic power directory has a noncanonical owner type.")
    source_map = geometry.source_map
    if not isinstance(source_map, ExactPowerCellGeometrySource):
        raise TypeError("Periodic power archive lost its exact coordinate map.")
    if type(source_map) is not ExactPowerCellGeometrySource:
        raise TypeError("Periodic power archive has a noncanonical exact source map.")
    context = source_map.periodic_source_context
    if not isinstance(context, ExactPeriodicPowerSourceContext):
        raise TypeError("Periodic power archive lost its typed source context.")
    if type(context) is not ExactPeriodicPowerSourceContext:
        raise TypeError("Periodic power archive has a noncanonical source context.")
    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError("Periodic power archive mesh lost its periodic topology.")
    return {
        "composition_id": composition.composition_id,
        "source_entry_id": entries["periodic-power/source"].record_id,
        "mesh_id": mesh.mesh_id,
        "periodic_topology_id": topology.periodic_topology_id,
        "geometry_id": cell_geometry_id(geometry),
        "source_map_id": source_map.source_map_id,
        "context_id": context.context_id,
        "cell_id": context.periodic_source.cell.cell_id,
        "orbits_id": context.periodic_source.orbits_id,
        "output_vertex_indices": np.asarray(
            source_map.output_vertex_indices, dtype=np.int64
        ).tolist(),
        "coordinates": [
            [str(value) for value in row] for row in geometry.source_coordinates()
        ],
        "exact_coordinate_bank": _exact_coordinate_bank_payload(
            geometry.exact_coordinate_bank
        ),
        "owner_directory_id": witness.directory_id,
        "owner_authority_id": witness.authority_id,
        "owner_group_id": witness.group_id,
        "owner_rank": witness.rank,
        "owner_part_count": witness.part_count,
        "owner_source_manifest_id": witness.source_manifest_id,
        "owner_local_source_map_id": witness.local_source_map_id,
        "owner_local_source_context_id": witness.local_source_context_id,
        "owner_source_point_ids": np.asarray(
            witness.local_source_point_ids, dtype=np.int64
        ).tolist(),
        "owner_cell_ids": np.asarray(witness.local_cell_ids, dtype=np.int64).tolist(),
        "owner_vertex_ids": np.asarray(witness.local_vertex_ids, dtype=np.int64).tolist(),
    }


def _write_periodic_power_map_archive(directory: Path, /) -> None:
    composition = _periodic_power_map_composition()
    write_model_artifact(directory / "periodic-power.phx", composition, limits=None)
    print(json.dumps(_periodic_power_map_archive_metadata(composition), sort_keys=True))


def _rebuild_periodic_power_map_composition(value: object, /) -> Composition:
    from phydrax.discretization._exact_power_geometry import (
        ExactPeriodicPowerSourceContext,
    )
    from phydrax.meshing._distribution import ShardedOwnerDirectoryWitness
    from phydrax.meshing._periodic import PeriodicPointOrbits

    expected_ids = {
        "periodic-power/source",
        "periodic-power/mesh",
        "periodic-power/geometry",
        "periodic-power/owner-directory",
    }
    if not isinstance(value, Composition):
        raise TypeError("Periodic power archive requires a Composition root.")
    entries = {entry.entry_id: entry for entry in value.entries}
    if set(entries) != expected_ids:
        raise ValueError(
            "Periodic power Composition entries differ from its restart schema."
        )
    mesh, geometry = _rebuild_mesh_geometry_pair(
        entries["periodic-power/mesh"].value,
        entries["periodic-power/geometry"].value,
    )
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Periodic power geometry lost its exact source context.")
    source_map = geometry.source_map
    if not isinstance(source_map, ExactPowerCellGeometrySource):
        raise TypeError("Periodic power geometry lost its exact coordinate map.")
    if type(source_map) is not ExactPowerCellGeometrySource:
        raise TypeError("Periodic power geometry has a noncanonical exact source map.")
    context = source_map.periodic_source_context
    if not isinstance(context, ExactPeriodicPowerSourceContext):
        raise TypeError("Periodic power geometry lost its typed source context.")
    if type(context) is not ExactPeriodicPowerSourceContext:
        raise TypeError("Periodic power geometry has a noncanonical source context.")
    source = context.periodic_source
    stored_source = entries["periodic-power/source"].value
    if (
        not isinstance(source, PeriodicPointOrbits)
        or not isinstance(stored_source, PeriodicPointOrbits)
        or source.orbits_id != stored_source.orbits_id
    ):
        raise ValueError("Periodic power map source differs from its source entry.")

    def rebind_entry(entry: CompositionEntry, entry_value: object, /) -> CompositionEntry:
        return CompositionEntry(
            entry_value,
            entry_id=entry.entry_id,
            role=entry.role,
            owner_id=entry.owner_id,
            structure_id=entry.structure_id,
            revision_id=entry.revision_id,
            semantics_id=entry.semantics_id,
            dependencies=tuple(
                CompositionDependency(
                    dependency.entry_id, dependency.facet, dependency.bound_id
                )
                for dependency in entry.dependencies
            ),
        )

    source_entry = rebind_entry(entries["periodic-power/source"], source)
    if source_entry.record_id != entries["periodic-power/source"].record_id:
        raise ValueError("Periodic power source entry changed its retained identity.")
    witness = ShardedOwnerDirectoryWitness.from_stored_fields(
        entries["periodic-power/owner-directory"].value,
        _single_rank_owner_directory_group(),
    )
    if (
        witness.source_manifest_id != source.orbits_id
        or witness.source_authority_id != source_entry.record_id
        or witness.local_source_map_id != source_map.source_map_id
        or witness.local_source_context_id != context.context_id
        or not np.array_equal(
            np.asarray(witness.local_source_point_ids, dtype=np.int64),
            np.asarray(source.point_ids, dtype=np.int64),
        )
        or not np.array_equal(
            np.asarray(witness.local_cell_ids, dtype=np.int64),
            np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64),
        )
        or not np.array_equal(
            np.asarray(witness.local_vertex_ids, dtype=np.int64),
            np.asarray(mesh.vertex_global_ids, dtype=np.int64),
        )
    ):
        raise ValueError("Sharded owner directory no longer matches its source and mesh.")
    source_map.require_periodic_mesh(mesh)
    mesh_entry = rebind_entry(entries["periodic-power/mesh"], mesh)
    geometry_entry = rebind_entry(entries["periodic-power/geometry"], geometry)
    directory_entry = witness.composition_entry()
    if directory_entry.record_id != entries["periodic-power/owner-directory"].record_id:
        raise ValueError(
            "Sharded owner-directory Composition entry changed its identity."
        )
    composition = Composition(
        (source_entry, mesh_entry, geometry_entry, directory_entry),
        boundary_id=value.boundary_id,
    )
    if (
        composition.composition_id != value.composition_id
        or mesh.mesh_id != entries["periodic-power/mesh"].value.mesh_id
        or cell_geometry_id(geometry)
        != cell_geometry_id(entries["periodic-power/geometry"].value)
    ):
        raise ValueError("Periodic power source context or mesh identity changed.")
    return composition


def _restore_periodic_power_without_mesh(value: object, /) -> NoReturn:
    if not isinstance(value, Composition):
        raise TypeError("Periodic power archive requires a Composition root.")
    entries = {entry.entry_id: entry for entry in value.entries}
    geometry = entries["periodic-power/geometry"].value
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Periodic power archive lost its exact geometry owner.")
    source_map = geometry.source_map
    if type(source_map) is not ExactPowerCellGeometrySource:
        raise TypeError("Periodic power geometry lost its exact source map.")
    if source_map.periodic_source_context is None:
        raise TypeError("Periodic power geometry lost its typed source context.")
    ExactPowerCellGeometrySource.from_stored_fields(source_map, target_periodic_cell=None)
    raise ValueError(
        "Periodic source context unexpectedly restored without its target mesh."
    )


def _read_periodic_power_map_without_mesh(directory: Path, /) -> NoReturn:
    read_model_artifact(
        directory / "periodic-power.phx",
        expected_type=Composition,
        restore=_restore_periodic_power_without_mesh,
        limits=None,
    )
    raise ValueError(
        "Periodic power archive unexpectedly restored without its target mesh."
    )


def _read_periodic_power_map_archive(
    directory: Path, expected: dict[str, object], /
) -> None:
    composition = read_model_artifact(
        directory / "periodic-power.phx",
        expected_type=Composition,
        restore=_rebuild_periodic_power_map_composition,
        limits=None,
    )
    if _periodic_power_map_archive_metadata(composition) != expected:
        raise ValueError("Cold periodic power map changed its exact source context.")
    print("periodic-power-source-context-cold-restored")


def _owner_local_remap_archive_fixture() -> tuple[
    OwnerLocalCellGeometry,
    PreparedOwnerLocalConservativeRemap,
    ExecutionGroup,
    VolumeMeshingSpec,
]:
    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.discretization.finite_volume._owner_local_remap import (
        prepare_owner_local_conservative_remap,
    )
    from phydrax.meshing import (
        MeshingEntityKind,
        MeshingScope,
        NativeMeshingOptions,
        NativeMeshingProvider,
        NativePeriodicSource,
        PeriodicPointOrbits,
    )
    from phydrax.meshing._assembly import MeshAssembly
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )

    cell = PeriodicCell(
        np.asarray(
            ((1.0, 0.125, 0.0), (0.0, 1.0, 0.25), (0.125, 0.0, 1.0)),
            dtype=np.float64,
        )
    )
    fractional = np.asarray(
        (
            (0.125, 0.1875, 0.25),
            (0.625, 0.125, 0.125),
            (0.1875, 0.625, 0.3125),
            (0.3125, 0.3125, 0.75),
            (0.75, 0.8125, 0.5625),
            (0.875, 0.4375, 0.9375),
        ),
        dtype=np.float64,
    )
    periodic_points = PeriodicPointOrbits(
        cell,
        np.asarray(cell.cartesian(fractional), dtype=np.float64),
        np.asarray((91, 4, 53, 17, 62, 31), dtype=np.int64),
    )
    source_id = "archive-owner-local-periodic-power"
    source_revision = "archive-owner-local-periodic-power-r1"
    source = NativePeriodicSource(
        periodic_points,
        source_id,
        source_revision,
        site_weights=np.asarray(
            (0.0, 0.015, -0.01, 0.02, 0.005, -0.015), dtype=np.float64
        ),
        point_owner_ranks=np.zeros(
            np.asarray(periodic_points.point_ids).shape, dtype=np.int32
        ),
    )
    scope = MeshingScope(
        source_id,
        source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        source.scope_entity_ids(3),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("polyhedron",)),
        ),
        scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(),
    )
    group = _single_rank_owner_directory_group()
    assembly = (
        NativeMeshingProvider(NativeMeshingOptions("periodic_power"))
        .plan(
            source,
            specification,
            coordinate_contract=SpatialCoordinateContract.si(),
        )
        .execute(execution_group=group)
    )
    if not isinstance(assembly, MeshAssembly) or len(assembly.parts) != 1:
        raise TypeError(
            "Owner-local periodic generation did not return one MeshAssembly part."
        )
    view = assembly.parts[0].carrier
    if not isinstance(view, OwnerLocalCellGeometry):
        raise TypeError(
            "Periodic generation did not return canonical owner-local geometry."
        )
    if type(view) is not OwnerLocalCellGeometry:
        raise TypeError(
            "Periodic generation returned a noncanonical local geometry owner."
        )
    remap = prepare_owner_local_conservative_remap(
        view, view, group, axis_name="archive_owner_local_remap"
    )
    return view, remap, group, specification


def _owner_local_periodic_adaptive_archive_fixture() -> tuple[
    OwnerLocalCellGeometry,
    SurfaceMeshingSpec,
    GraphPartitionPlan,
    ExecutionGroup,
]:
    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.meshing import (
        MeshingEntityKind,
        MeshingScope,
        NativeMeshingOptions,
        NativeMeshingProvider,
        NativePeriodicSource,
        PeriodicPointOrbits,
    )
    from phydrax.meshing._assembly import MeshAssembly
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        SurfaceMeshingSpec,
    )
    from phydrax.meshing._sizing import SizeCompliancePolicy, UniformSizeControl

    cell = PeriodicCell(np.asarray(((1.0, 0.2), (0.3, 1.1)), dtype=np.float64))
    fractional = np.asarray(
        ((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77)),
        dtype=np.float64,
    )
    periodic_points = PeriodicPointOrbits(
        cell,
        np.asarray(cell.cartesian(fractional), dtype=np.float64),
        np.asarray((91, 4, 53, 17), dtype=np.int64),
    )
    source_id = "archive-owner-local-periodic-adaptive"
    source_revision = "archive-owner-local-periodic-adaptive-r1"
    source = NativePeriodicSource(
        periodic_points,
        source_id,
        source_revision,
        point_owner_ranks=np.zeros(
            np.asarray(periodic_points.point_ids).shape, dtype=np.int32
        ),
    )
    surface_scope = MeshingScope(
        source_id,
        source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            2,
            CellFamilyPolicy(required=("triangle",)),
        ),
        surface_scope,
        size_controls=(UniformSizeControl(surface_scope, 0.35),),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
    )
    graph_partition_plan = GraphPartitionPlan(
        1,
        maximum_imbalance=1.0,
    )
    group = _single_rank_owner_directory_group("archive-periodic-adaptive-source")
    assembly = (
        NativeMeshingProvider(NativeMeshingOptions("periodic_delaunay"))
        .plan(
            source,
            specification,
            coordinate_contract=SpatialCoordinateContract.si(),
        )
        .execute(
            execution_group=group,
            graph_partition_plan=graph_partition_plan,
        )
    )
    if not isinstance(assembly, MeshAssembly) or len(assembly.parts) != 1:
        raise TypeError("Periodic adaptation did not return one owner-local mesh part.")
    view = assembly.parts[0].carrier
    if not isinstance(view, OwnerLocalCellGeometry):
        raise TypeError("Periodic adaptation did not return its typed accepted result.")
    if type(view) is not OwnerLocalCellGeometry or view.adaptive_result is None:
        raise TypeError(
            "Periodic adaptation did not return its canonical accepted result."
        )
    if (
        view.adaptive_result.specification_id != specification.specification_id
        or view.adaptive_result.graph_partition_plan is None
        or view.adaptive_result.graph_partition_plan.plan_id
        != graph_partition_plan.plan_id
    ):
        raise ValueError(
            "Periodic adaptive result lost its current specification or plan."
        )
    return view, specification, graph_partition_plan, group


def _owner_local_periodic_adaptive_archive_composition() -> tuple[
    Composition,
    OwnerLocalCellGeometry,
    SurfaceMeshingSpec,
    GraphPartitionPlan,
]:
    view, specification, graph_partition_plan, _ = (
        _owner_local_periodic_adaptive_archive_fixture()
    )
    specification_entry = CompositionEntry(
        specification,
        entry_id="owner-local-adaptive/specification",
        role="interface-route",
        owner_id="phydrax.meshing",
        structure_id=specification.target.target_id,
        revision_id=specification.specification_id,
        semantics_id="current-periodic-adaptive-meshing-specification",
    )
    plan_entry = CompositionEntry(
        graph_partition_plan,
        entry_id="owner-local-adaptive/graph-plan",
        role="prepared-graph",
        owner_id="phydrax.graph",
        structure_id=graph_partition_plan.plan_id,
        revision_id=graph_partition_plan.plan_id,
        semantics_id="current-owner-local-graph-partition-plan",
    )
    view_entry = CompositionEntry(
        view,
        entry_id="owner-local-adaptive/view",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=view.geometry_layout_id,
        revision_id=view.distribution_id,
        semantics_id="native-periodic-owner-local-adaptive-cell-geometry",
        dependencies=(
            specification_entry.binding("revision"),
            plan_entry.binding("revision"),
        ),
    )
    composition = Composition(
        (specification_entry, plan_entry, view_entry),
        boundary_id="owner-local-periodic-adaptive-cold-restore",
    )
    return composition, view, specification, graph_partition_plan


def _owner_local_periodic_adaptive_archive_metadata(
    composition: Composition,
    view: OwnerLocalCellGeometry,
    specification: SurfaceMeshingSpec,
    graph_partition_plan: GraphPartitionPlan,
    /,
) -> dict[str, object]:
    result = view.adaptive_result
    evidence = view.graph_migration_evidence
    if result is None or evidence is None:
        raise TypeError("Owner-local adaptive archive lost its accepted graph migration.")
    if evidence.result.plan_id != graph_partition_plan.plan_id:
        raise ValueError(
            "Graph migration evidence differs from its current partition plan."
        )
    return {
        "composition_id": composition.composition_id,
        "specification_id": specification.specification_id,
        "graph_partition_plan_id": graph_partition_plan.plan_id,
        "adaptive_result_id": result.result_id,
        "candidate_id": result.candidate_id,
        "adaptive_status": result.status.value,
        "graph_migration_evidence_id": evidence.evidence_id,
        "graph_partition_result_id": evidence.result.result_id,
        "graph_source_candidate_id": evidence.source_candidate_id,
        "graph_source_part_id": evidence.source_part_id,
        "graph_face_coverage_id": evidence.periodic_face_coverage_id,
        "graph_source_directory_id": evidence.source_directory_id,
        "graph_target_part_id": evidence.target_part_id,
        "graph_target_directory_id": evidence.target_directory_id,
        "graph_group_id": evidence.group_id,
        "reattachment_group_id": _single_rank_owner_directory_group(
            "archive-periodic-adaptive-target"
        ).spec.group_id,
        "graph_local_rank": evidence.local_rank,
        "graph_part_count": evidence.part_count,
        "graph_target_cell_ids": np.asarray(
            evidence.target_cell_ids, dtype=np.int64
        ).tolist(),
        "local_mesh_id": view.local_mesh_id,
        "local_topology_id": view.local_topology_id,
        "local_geometry_id": view.local_geometry_id,
        "periodic_topology_id": view.periodic_topology_id,
        "periodic_face_coverage_id": view.periodic_face_coverage_id,
        "part_id": view.part_id,
        "distribution_id": view.distribution_id,
        "owner_rank": view.owner_rank,
        "owned_cell_count": int(np.asarray(view.owned_cell_ids).size),
    }


def _write_owner_local_periodic_adaptive_archive(directory: Path, /) -> None:
    composition, view, specification, graph_partition_plan = (
        _owner_local_periodic_adaptive_archive_composition()
    )
    write_model_artifact(
        directory / "owner-local-periodic-adaptive.phx",
        composition,
        limits=None,
    )
    print(
        json.dumps(
            _owner_local_periodic_adaptive_archive_metadata(
                composition, view, specification, graph_partition_plan
            )
        )
    )


def _read_owner_local_periodic_adaptive_archive(
    directory: Path, expected: dict[str, object], /
) -> None:
    from functools import partial

    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.meshing._assembly import MeshAssembly
    from phydrax.meshing._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
    from phydrax.meshing._model_codec import restore_model_archive_value
    from phydrax.meshing.providers._native_periodic import (
        rebuild_owner_local_periodic_assembly_from_shards,
    )

    source_group = _single_rank_owner_directory_group("archive-periodic-adaptive-source")
    composition = read_model_artifact(
        directory / "owner-local-periodic-adaptive.phx",
        expected_type=Composition,
        restore=partial(
            restore_model_archive_value,
            execution_group=source_group,
        ),
        limits=None,
    )
    entries = {entry.entry_id: entry for entry in composition.entries}
    archived_view = entries["owner-local-adaptive/view"].value
    specification = entries["owner-local-adaptive/specification"].value
    graph_partition_plan = entries["owner-local-adaptive/graph-plan"].value
    if (
        not isinstance(archived_view, OwnerLocalCellGeometry)
        or type(archived_view) is not OwnerLocalCellGeometry
        or not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec))
        or type(specification) not in (SurfaceMeshingSpec, VolumeMeshingSpec)
        or not isinstance(graph_partition_plan, GraphPartitionPlan)
        or type(graph_partition_plan) is not GraphPartitionPlan
        or archived_view.adaptive_result is None
        or _owner_local_periodic_adaptive_archive_metadata(
            composition, archived_view, specification, graph_partition_plan
        )
        != expected
    ):
        raise ValueError("Cold owner-local shard archive changed its scientific proof.")
    archived_evidence = archived_view.graph_migration_evidence
    if archived_evidence is None:
        raise TypeError("Archived owner-local shard lacks its graph migration evidence.")
    archived_mesh, archived_geometry = archived_view.reconstruct_local_geometry()
    if archived_mesh is None or archived_geometry is None:
        raise TypeError("Archived owner-local source has no current local geometry.")
    current_group = _single_rank_owner_directory_group("archive-periodic-adaptive-target")

    def archived_shards() -> Iterator[OwnerLocalCellGeometry]:
        yield archived_view

    assembly = rebuild_owner_local_periodic_assembly_from_shards(
        archived_shards(),
        current_group,
        current_specification=specification,
        graph_partition_plan=graph_partition_plan,
        axis_name="archive_periodic_adaptive_reattach",
    )
    if not isinstance(assembly, MeshAssembly) or len(assembly.parts) != 1:
        raise TypeError("Changed-group cold reattachment did not return one local part.")
    current_view = assembly.parts[0].carrier
    if not isinstance(current_view, OwnerLocalCellGeometry):
        raise TypeError("Changed-group reattachment lost its local geometry view.")
    if type(current_view) is not OwnerLocalCellGeometry:
        raise TypeError("Changed-group reattachment has a noncanonical geometry view.")
    evidence = current_view.graph_migration_evidence
    current_result = current_view.adaptive_result
    if current_group.spec.group_id != expected["reattachment_group_id"]:
        raise ValueError(
            "Cold archive target group differs from the current execution group."
        )
    if evidence is None or current_result is None:
        raise TypeError("Changed-group reattachment lost its adaptive graph evidence.")
    if (
        current_view.source_id != archived_view.source_id
        or current_view.source_revision != archived_view.source_revision
        or current_view.source_topology_id != archived_view.source_topology_id
        or current_view.source_geometry_id != archived_view.source_geometry_id
        or current_view.periodic_topology_id != archived_view.periodic_topology_id
        or current_view.periodic_face_coverage_id
        != archived_view.periodic_face_coverage_id
        or not np.array_equal(
            np.asarray(evidence.target_cell_ids, dtype=np.int64),
            np.asarray(archived_evidence.target_cell_ids, dtype=np.int64),
        )
        or current_result.specification_id != specification.specification_id
        or current_result.graph_partition_plan is None
        or current_result.graph_partition_plan.plan_id != graph_partition_plan.plan_id
        or evidence.group_id != current_group.spec.group_id
        or evidence.group_id == expected["graph_group_id"]
        or evidence.result.plan_id != graph_partition_plan.plan_id
    ):
        raise ValueError(
            "Changed-group graph reattachment changed source/coverage proof."
        )
    current_mesh, current_geometry = current_view.reconstruct_local_geometry()
    if current_mesh is None or current_geometry is None:
        raise TypeError("Reattached owner-local source has no current local geometry.")
    print("owner-local-periodic-adaptive-reattached-cold-restored")


def _meshing_spec_archive_composition() -> tuple[
    Composition, SurfaceMeshingSpec, VolumeMeshingSpec, SurfaceMeshingSpec
]:
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        MeshQualityTarget,
        SurfaceMeshingSpec,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )
    from phydrax.meshing._controls import (
        BackgroundMetricControl,
        BoundaryLayerControl,
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
    from phydrax.meshing._metric import MeshMetricField
    from phydrax.meshing._organization import RegionRole
    from phydrax.meshing._sizing import (
        CurvatureSizeControl,
        SizeCombinationPolicy,
        SizeCompliancePolicy,
        SizeControlStrength,
        UniformSizeControl,
    )

    background_mesh = CellMesh(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        (
            CellBlock(
                "background",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
            ),
        ),
        numeric_version="archive-background-metric-r1",
    )
    background_scope = MeshingScope.whole(background_mesh, 0)
    background_metric = MeshMetricField(
        background_scope,
        np.broadcast_to(np.eye(3, dtype=np.float64), (3, 3, 3)),
        minimum_size=0.5,
        maximum_size=2.0,
        maximum_anisotropy=10.0,
    )
    background_control = BackgroundMetricControl(
        background_mesh,
        background_metric,
        SpatialCoordinateContract.si(),
        maximum_metric_edge_length=1.5,
    )

    surface_source = ("archive-spec-surface", "revision-1", MeshingEntityKind.GEOMETRY)
    surface_scope = MeshingScope(
        *surface_source,
        2,
        "surface-faces",
        np.asarray((50,), dtype=np.int64),
    )
    surface_vertex_scope = MeshingScope(
        *surface_source,
        0,
        "surface-vertices",
        np.asarray((1, 2, 3), dtype=np.int64),
    )
    surface_edge_scope = MeshingScope(
        *surface_source,
        1,
        "surface-edges",
        np.asarray((20, 21, 22), dtype=np.int64),
    )
    surface_periodic_target = MeshingScope(
        *surface_source,
        1,
        "surface-edges",
        np.asarray((23,), dtype=np.int64),
    )
    surface_periodic_transform = np.eye(4, dtype=np.float64)
    surface_specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            3,
            CellFamilyPolicy(required=("triangle",)),
        ),
        surface_scope,
        size_controls=(
            UniformSizeControl(surface_vertex_scope, 0.25),
            CurvatureSizeControl(
                surface_vertex_scope,
                0.4,
                strength=SizeControlStrength.SOFT,
                priority=2,
            ),
        ),
        protected_features=(
            ProtectedFeature(
                surface_edge_scope,
                FeatureKind.CURVE,
                maximum_deviation=0.005,
                hard=False,
            ),
        ),
        region_controls=(
            RegionControl(
                surface_scope,
                "surface-region",
                "surface-material",
                RegionRole.SOLID,
            ),
        ),
        patch_controls=(
            PatchControl(
                "surface-boundary",
                surface_edge_scope,
                ("surface-region",),
            ),
        ),
        periodic_constraints=(
            PeriodicConstraint(
                surface_edge_scope,
                surface_periodic_target,
                surface_periodic_transform,
            ),
        ),
        layer_controls=(
            BoundaryLayerControl(
                surface_edge_scope,
                LayerSchedule((0.05, 0.1)),
                route=BoundaryLayerRoute.PROVIDER,
                core_maximum_size=0.5,
            ),
        ),
        size_combination=SizeCombinationPolicy.EXPLICIT_PRIORITY,
        size_compliance=SizeCompliancePolicy(
            absolute_tolerance=0.01,
            relative_tolerance=0.02,
        ),
        maximum_deviation=0.05,
        maximum_normal_deviation=0.1,
        quality_target=MeshQualityTarget(minimum_angle=0.2, hard=False),
        background_metric=background_control,
    )

    volume_source = ("archive-spec-volume", "revision-2", MeshingEntityKind.GEOMETRY)
    volume_boundary_scope = MeshingScope(
        *volume_source,
        2,
        "volume-boundary",
        np.asarray((100, 101), dtype=np.int64),
    )
    volume_cell_scope = MeshingScope(
        *volume_source,
        3,
        "volume-cells",
        np.asarray((200,), dtype=np.int64),
    )
    volume_periodic_source = MeshingScope(
        *volume_source,
        2,
        "volume-boundary",
        np.asarray((100,), dtype=np.int64),
    )
    volume_periodic_target = MeshingScope(
        *volume_source,
        2,
        "volume-boundary",
        np.asarray((101,), dtype=np.int64),
    )
    volume_periodic_transform = np.eye(4, dtype=np.float64)
    volume_periodic_transform[0, 3] = 1.0
    volume_specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        volume_boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(
                volume_boundary_scope,
                0.5,
                minimum_size=0.1,
                maximum_size=1.0,
                maximum_growth_rate=2.0,
                strength=SizeControlStrength.HARD,
                priority=3,
            ),
        ),
        protected_features=(
            ProtectedFeature(
                volume_boundary_scope,
                FeatureKind.SURFACE,
                maximum_deviation=0.02,
            ),
        ),
        region_controls=(
            RegionControl(
                volume_cell_scope,
                "volume-region",
                "volume-material",
                RegionRole.SOLID,
            ),
        ),
        patch_controls=(
            PatchControl(
                "volume-boundary-patch",
                volume_boundary_scope,
                ("volume-region",),
            ),
        ),
        region_seeds=(
            RegionSeed(
                np.asarray((0.2, 0.2, 0.2), dtype=np.float64),
                "volume-region",
                "volume-material",
                RegionRole.SOLID,
            ),
        ),
        hole_seeds=(
            HoleSeed(
                np.asarray((0.8, 0.8, 0.8), dtype=np.float64),
                volume_cell_scope,
            ),
        ),
        periodic_constraints=(
            PeriodicConstraint(
                volume_periodic_source,
                volume_periodic_target,
                volume_periodic_transform,
            ),
        ),
        size_combination=SizeCombinationPolicy.EXPLICIT_PRIORITY,
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.1),
        maximum_deviation=0.1,
        deterministic=False,
    )

    planar_source = (
        "archive-spec-planar-surface",
        "revision-3",
        MeshingEntityKind.GEOMETRY,
    )
    planar_scope = MeshingScope(
        *planar_source,
        2,
        "planar-faces",
        np.asarray((300,), dtype=np.int64),
    )
    planar_specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            2,
            CellFamilyPolicy(required=("triangle",)),
        ),
        planar_scope,
        planar_embedding=PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        size_controls=(UniformSizeControl(planar_scope, 0.4),),
    )

    specifications = (
        (
            "meshing-specs/surface-background",
            surface_specification,
            "surface-background-metric-specification",
        ),
        (
            "meshing-specs/volume-controls",
            volume_specification,
            "volume-scientific-controls-specification",
        ),
        (
            "meshing-specs/planar-embedding",
            planar_specification,
            "planar-embedding-specification",
        ),
    )
    entries = tuple(
        CompositionEntry(
            specification,
            entry_id=entry_id,
            role="interface-route",
            owner_id="phydrax.meshing",
            structure_id=specification.target.target_id,
            revision_id=specification.specification_id,
            semantics_id=semantics_id,
        )
        for entry_id, specification, semantics_id in specifications
    )
    return (
        Composition(entries, boundary_id="canonical-meshing-spec-cold-restore"),
        surface_specification,
        volume_specification,
        planar_specification,
    )


def _meshing_spec_archive_metadata(
    composition: Composition,
    surface_specification: SurfaceMeshingSpec,
    volume_specification: VolumeMeshingSpec,
    planar_specification: SurfaceMeshingSpec,
    /,
) -> dict[str, object]:
    background = surface_specification.background_metric
    planar = planar_specification.planar_embedding
    if background is None or planar is None:
        raise TypeError(
            "Meshing spec archive lost its background metric or planar frame."
        )
    return {
        "composition_id": composition.composition_id,
        "surface_specification_id": surface_specification.specification_id,
        "background_metric_control_id": background.control_id,
        "background_mesh_id": background.mesh.mesh_id,
        "background_metric_id": background.metric.metric_id,
        "volume_specification_id": volume_specification.specification_id,
        "planar_specification_id": planar_specification.specification_id,
        "planar_embedding_id": planar.embedding_id,
    }


def _write_meshing_spec_archive(directory: Path, /) -> None:
    composition, surface_specification, volume_specification, planar_specification = (
        _meshing_spec_archive_composition()
    )
    write_model_artifact(directory / "meshing-specs.phx", composition, limits=None)
    print(
        json.dumps(
            _meshing_spec_archive_metadata(
                composition,
                surface_specification,
                volume_specification,
                planar_specification,
            )
        )
    )


def _rebuild_meshing_spec_archive_composition(value: object, /) -> Composition:
    if not isinstance(value, Composition):
        raise TypeError("Meshing spec archive requires a Composition root.")
    expected = {
        "meshing-specs/surface-background",
        "meshing-specs/volume-controls",
        "meshing-specs/planar-embedding",
    }
    entries = {entry.entry_id: entry for entry in value.entries}
    if set(entries) != expected:
        raise ValueError(
            "Meshing spec Composition entries differ from its restart schema."
        )
    from phydrax.meshing._contracts import SurfaceMeshingSpec, VolumeMeshingSpec

    surface = entries["meshing-specs/surface-background"].value
    volume = entries["meshing-specs/volume-controls"].value
    planar = entries["meshing-specs/planar-embedding"].value
    if type(surface) is not SurfaceMeshingSpec or type(planar) is not SurfaceMeshingSpec:
        raise TypeError("Meshing spec archive lost its typed surface specification.")
    if type(volume) is not VolumeMeshingSpec:
        raise TypeError("Meshing spec archive lost its typed volume specification.")
    restored_values = {
        "meshing-specs/surface-background": SurfaceMeshingSpec.from_stored_fields(
            surface
        ),
        "meshing-specs/volume-controls": VolumeMeshingSpec.from_stored_fields(volume),
        "meshing-specs/planar-embedding": SurfaceMeshingSpec.from_stored_fields(planar),
    }
    restored_entries = tuple(
        CompositionEntry(
            restored_values[entry.entry_id],
            entry_id=entry.entry_id,
            role=entry.role,
            owner_id=entry.owner_id,
            structure_id=entry.structure_id,
            revision_id=entry.revision_id,
            semantics_id=entry.semantics_id,
            dependencies=entry.dependencies,
        )
        for entry in value.entries
    )
    composition = Composition(restored_entries, boundary_id=value.boundary_id)
    if composition.composition_id != value.composition_id or any(
        restored_values[entry_id].specification_id != entries[entry_id].revision_id
        for entry_id in expected
    ):
        raise ValueError("Meshing spec archive changed a canonical scientific identity.")
    return composition


def _read_meshing_spec_archive(directory: Path, expected: dict[str, object], /) -> None:
    composition = read_model_artifact(
        directory / "meshing-specs.phx",
        expected_type=Composition,
        restore=_rebuild_meshing_spec_archive_composition,
        limits=None,
    )
    entries = {entry.entry_id: entry.value for entry in composition.entries}
    surface = entries["meshing-specs/surface-background"]
    volume = entries["meshing-specs/volume-controls"]
    planar = entries["meshing-specs/planar-embedding"]
    if _meshing_spec_archive_metadata(composition, surface, volume, planar) != expected:
        raise ValueError("Cold meshing spec restore changed its exact scientific inputs.")
    print("meshing-specs-cold-restored")


def _owner_local_remap_archive_composition() -> tuple[
    Composition,
    OwnerLocalCellGeometry,
    PreparedOwnerLocalConservativeRemap,
    VolumeMeshingSpec,
]:
    view, remap, _, specification = _owner_local_remap_archive_fixture()
    specification_entry = CompositionEntry(
        specification,
        entry_id="owner-local/specification",
        role="interface-route",
        owner_id="phydrax.meshing",
        structure_id=specification.target.target_id,
        revision_id=specification.specification_id,
        semantics_id="current-periodic-power-volume-specification",
    )
    view_entry = CompositionEntry(
        view,
        entry_id="owner-local/view",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=view.geometry_layout_id,
        revision_id=view.distribution_id,
        semantics_id="native-periodic-owner-local-cell-geometry",
        dependencies=(specification_entry.binding("revision"),),
    )
    remap_entry = CompositionEntry(
        remap,
        entry_id="owner-local/remap",
        role="discretization",
        owner_id="phydrax.discretization.finite_volume",
        structure_id=remap.refinement_id,
        revision_id=remap.remap_id,
        semantics_id="owner-local-conservative-overlap",
        dependencies=(view_entry.binding("structure"),),
    )
    composition = Composition(
        (specification_entry, view_entry, remap_entry),
        boundary_id="owner-local-periodic-remap-cold-restore",
    )
    return composition, view, remap, specification


def _owner_local_remap_archive_metadata(
    composition: Composition,
    view: OwnerLocalCellGeometry,
    remap: PreparedOwnerLocalConservativeRemap,
    specification: VolumeMeshingSpec,
    /,
) -> dict[str, object]:
    if view.specification_id != specification.specification_id:
        raise ValueError(
            "Owner-local periodic power view lost its current specification."
        )
    return {
        "composition_id": composition.composition_id,
        "specification_id": specification.specification_id,
        "view_specification_id": view.specification_id,
        "part_id": view.part_id,
        "local_mesh_id": view.local_mesh_id,
        "local_topology_id": view.local_topology_id,
        "local_geometry_id": view.local_geometry_id,
        "source_context_id": view.source_context_id,
        "source_map_id": view.source_map_id,
        "periodic_topology_id": view.periodic_topology_id,
        "periodic_face_coverage_id": view.periodic_face_coverage_id,
        "distribution_id": view.distribution_id,
        "group_id": remap.group_id,
        "owner_rank": view.owner_rank,
        "local_rank": remap.local_rank,
        "remap_id": remap.remap_id,
        "evidence_id": remap.evidence.evidence_id,
        "coverage_complete": remap.evidence.coverage_complete,
        "cell_ids": np.asarray(view.owned_cell_ids, dtype=np.int64).tolist(),
        "owned_cell_count": int(np.asarray(view.owned_cell_ids).size),
    }


def _write_owner_local_remap_archive(directory: Path, /) -> None:
    composition, view, remap, specification = _owner_local_remap_archive_composition()
    write_model_artifact(directory / "owner-local-remap.phx", composition, limits=None)
    print(
        json.dumps(
            _owner_local_remap_archive_metadata(composition, view, remap, specification)
        )
    )


def _read_owner_local_remap_archive(
    directory: Path, expected: dict[str, object], /
) -> None:
    from functools import partial

    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.discretization.finite_volume._owner_local_remap import (
        PreparedOwnerLocalConservativeRemap,
    )
    from phydrax.meshing._contracts import VolumeMeshingSpec
    from phydrax.meshing._model_codec import restore_model_archive_value

    current_group = _single_rank_owner_directory_group()

    composition = read_model_artifact(
        directory / "owner-local-remap.phx",
        expected_type=Composition,
        restore=partial(
            restore_model_archive_value,
            execution_group=current_group,
        ),
        limits=None,
    )
    entries = {entry.entry_id: entry for entry in composition.entries}
    view = entries["owner-local/view"].value
    remap = entries["owner-local/remap"].value
    specification = entries["owner-local/specification"].value
    if (
        not isinstance(view, OwnerLocalCellGeometry)
        or type(view) is not OwnerLocalCellGeometry
        or not isinstance(remap, PreparedOwnerLocalConservativeRemap)
        or type(remap) is not PreparedOwnerLocalConservativeRemap
        or not isinstance(specification, VolumeMeshingSpec)
        or type(specification) is not VolumeMeshingSpec
        or view.specification_id != specification.specification_id
        or _owner_local_remap_archive_metadata(composition, view, remap, specification)
        != expected
    ):
        raise ValueError("Cold owner-local remap changed its scientific source proof.")

    local_cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
    source_values = np.arange(local_cell_ids.size, dtype=np.float64)[:, None] + 1.0
    source_value_by_id = {
        int(identifier): float(source_values[row, 0])
        for row, identifier in enumerate(local_cell_ids.tolist())
    }
    expected_values = np.asarray(
        [
            source_value_by_id[int(identifier)]
            for identifier in np.asarray(remap.target_cell_ids, dtype=np.int64)
        ],
        dtype=np.float64,
    )
    expected_local_values = np.asarray(
        [source_value_by_id[int(identifier)] for identifier in local_cell_ids],
        dtype=np.float64,
    )
    action = remap.apply_averages(
        source_values,
        view,
        view,
        current_group,
        axis_name=remap.axis_name,
    )
    if (
        action.evidence_id != remap.evidence.evidence_id
        or not np.array_equal(
            np.asarray(action.local_target_cell_ids, dtype=np.int64), local_cell_ids
        )
        or not np.allclose(
            np.asarray(action.local_values, dtype=np.float64)[:, 0],
            expected_local_values,
            rtol=0.0,
            atol=remap.tolerance,
        )
        or not np.allclose(
            np.asarray(action.owned_values, dtype=np.float64)[:, 0],
            expected_values,
            rtol=0.0,
            atol=remap.tolerance,
        )
        or not np.allclose(
            np.asarray(action.conservation_residual, dtype=np.float64),
            0.0,
            rtol=0.0,
            atol=remap.tolerance,
        )
    ):
        raise ValueError(
            "Restored owner-local remap did not execute its accepted identity action."
        )
    print("owner-local-periodic-remap-cold-restored")


def _analytic_profile_archive_fixture() -> AnalyticImplicitProfile:
    source_id = "archive-analytic-profile"
    geometry = Sphere((0.37, 0.13, -0.36), 0.04, feature_id=source_id).compile()
    return AnalyticImplicitProfile(
        geometry,
        SpatialCoordinateContract.si(),
        source_id=source_id,
        source_revision="archive-analytic-profile-r1",
    )


def _write_analytic_profile_artifact(directory: Path, /) -> None:
    profile = _analytic_profile_archive_fixture()
    entry = CompositionEntry(
        profile,
        entry_id="implicit/profile",
        role="topology",
        owner_id="phydrax.geometry.implicit",
        structure_id=profile.profile_id,
        revision_id=profile.source_revision,
        semantics_id="analytic-implicit-source-profile",
    )
    composition = Composition((entry,), boundary_id="analytic-profile-cold-restore")
    write_model_artifact(directory / "analytic-profile.phx", composition, limits=None)
    points = np.asarray(
        ((0.37, 0.13, -0.36), (0.41, 0.13, -0.36), (0.47, 0.13, -0.36)),
        dtype=np.float64,
    )
    distance = profile.boundary_distance(points)
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "source_id": profile.source_id,
                "source_revision": profile.source_revision,
                "profile_id": profile.profile_id,
                "query_points": points.tolist(),
                "distance_lower": np.asarray(distance.lower).tolist(),
                "distance_upper": np.asarray(distance.upper).tolist(),
                "distance_semantics": distance.semantics,
            }
        )
    )


def _rebuild_analytic_profile_composition(value: object, /) -> Composition:
    if not isinstance(value, Composition) or len(value.entries) != 1:
        raise TypeError("Analytic profile archive requires one Composition source entry.")
    entry = value.entries[0]
    profile = _rebuild_continuation_value(entry.value)
    if not isinstance(profile, AnalyticImplicitProfile):
        raise TypeError("Analytic profile archive did not reconstruct its query owner.")
    rebuilt_entry = CompositionEntry(
        profile,
        entry_id=entry.entry_id,
        role=entry.role,
        owner_id=entry.owner_id,
        structure_id=entry.structure_id,
        revision_id=entry.revision_id,
        semantics_id=entry.semantics_id,
        dependencies=tuple(
            CompositionDependency(
                dependency.entry_id, dependency.facet, dependency.bound_id
            )
            for dependency in entry.dependencies
        ),
    )
    composition = Composition((rebuilt_entry,), boundary_id=value.boundary_id)
    if (
        composition.composition_id != value.composition_id
        or profile.profile_id != entry.structure_id
        or profile.source_revision != entry.revision_id
    ):
        raise ValueError(
            "Analytic profile Composition identity changed on reconstruction."
        )
    return composition


def _read_analytic_profile_artifact(
    directory: Path, expected: dict[str, object], /
) -> None:
    composition = read_model_artifact(
        directory / "analytic-profile.phx",
        expected_type=Composition,
        restore=_rebuild_analytic_profile_composition,
        limits=None,
    )
    profile = composition.entries[0].value
    if not isinstance(profile, AnalyticImplicitProfile):
        raise TypeError("Cold analytic profile archive lost its geometry query.")
    distance = profile.boundary_distance(
        np.asarray(_expected_array(expected["query_points"]), dtype=np.float64)
    )
    if (
        composition.composition_id != expected["composition_id"]
        or profile.source_id != expected["source_id"]
        or profile.source_revision != expected["source_revision"]
        or profile.profile_id != expected["profile_id"]
        or distance.semantics != expected["distance_semantics"]
        or not np.array_equal(distance.lower, _expected_array(expected["distance_lower"]))
        or not np.array_equal(distance.upper, _expected_array(expected["distance_upper"]))
    ):
        raise ValueError("Cold analytic implicit query changed its exact source profile.")
    print("analytic-profile-cold-restored")


def _source_chart_rational_witness() -> SourceBoundaryChartWitness:
    target_tokens = (
        (
            ((0, 1), (0, 1)),
            ((1, 1), (0, 1)),
            ((0, 1), (1, 1)),
        ),
    )
    source_tokens = (
        (
            ((0, 1), (0, 1)),
            ((1, 3), (0, 1)),
            ((0, 1), (1, 1)),
        ),
    )
    return SourceBoundaryChartWitness(
        mesh_id="archive-chart-mesh",
        geometry_id="archive-chart-geometry",
        topology_id="archive-chart-topology",
        source_id="archive-chart-source",
        source_revision="archive-chart-source-r1",
        source_query_id="archive-chart-query",
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=np.asarray((700,), dtype=np.int64),
        entity_global_ids=np.asarray((800,), dtype=np.int64),
        # A chart of the full reference dimension has no local boundary entity.
        local_entity_indices=np.asarray((-1,), dtype=np.int32),
        target_reference_vertices=np.asarray(
            (((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),), dtype=np.float64
        ),
        target_reference_rational_tokens=target_tokens,
        source_chart_ids=("archive-chart-0",),
        source_entity_ids=("archive-face-0",),
        occurrence_paths=((),),
        source_parameters=np.asarray(
            (((0.0, 0.0), (1.0 / 3.0, 0.0), (0.0, 1.0)),),
            dtype=np.float64,
        ),
        source_parameter_rational_tokens=source_tokens,
        orientations=np.asarray((1,), dtype=np.int8),
    )


def _write_source_chart_rational_artifact(directory: Path, /) -> None:
    witness = _source_chart_rational_witness()
    write_model_artifact(
        directory / "source-chart-rational-witness.phx", witness, limits=None
    )
    print(
        json.dumps(
            {
                "witness_id": witness.witness_id,
                "target_tokens": witness.target_reference_rational_tokens,
                "source_tokens": witness.source_parameter_rational_tokens,
            }
        )
    )


def _read_source_chart_rational_artifact(
    directory: Path, expected: dict[str, object], /
) -> None:
    witness = read_model_artifact(
        directory / "source-chart-rational-witness.phx",
        expected_type=SourceBoundaryChartWitness,
        restore=_rebuild_continuation_value,
        limits=None,
    )
    target_chart = witness.target_reference_chart(0)
    source_chart = witness.source_parameter_chart(0)

    def token_payload(
        bank: SourceBoundaryRationalChartBank, /
    ) -> list[list[list[list[int]]]]:
        return [[[list(token) for token in point] for point in row] for row in bank]

    expected_target_chart = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    expected_source_chart = (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    if (
        witness.witness_id != expected["witness_id"]
        or token_payload(witness.target_reference_rational_tokens)
        != expected["target_tokens"]
        or token_payload(witness.source_parameter_rational_tokens)
        != expected["source_tokens"]
        or target_chart != expected_target_chart
        or source_chart != expected_source_chart
    ):
        raise ValueError("Cold source chart witness lost exact rational chart banks.")
    print("source-chart-rational-cold-restored")


def _read_cad_chart_witness(path: Path, expected_id: str, /) -> None:
    witness = read_model_artifact(
        path,
        expected_type=SourceBoundaryChartWitness,
        restore=_rebuild_continuation_value,
        limits=None,
    )
    if witness.witness_id != expected_id or witness.cad_plc is None:
        raise ValueError("Cold CAD chart witness lost its source ancestry.")
    charts = witness.cad_plc.cad_source_charts
    if charts is None:
        raise ValueError("Cold CAD PLC lost its source charts.")
    occurrences = np.asarray(charts.boundary_occurrences)
    if (
        occurrences.dtype != np.int64
        or occurrences.ndim != 3
        or occurrences.shape[1:] != (charts.mesh.blocks[0].arity, 3)
        or occurrences.shape[0] not in (0, len(charts.cell_global_ids))
    ):
        raise ValueError("Cold CAD source edge-occurrence rows are malformed.")
    print("cad-chart-witness-cold-restored")


def _surface_result_for_brep(model: BRepModel, /) -> CellMeshingResult:
    from phydrax.meshing.providers._native_surface import prepare_native_surface
    from tests.unit.meshing.test_native_surface_routes import _surface_specification

    domain = MeshingDomain.from_brep(model)
    construction = prepare_native_surface(
        domain,
        _surface_specification(domain, maximum_deviation=None),
        NativeMeshingOptions("parametric_surface"),
        model.coordinate_contract,
    ).execute()
    surface_source = construction.organization.surface_source
    if not isinstance(surface_source, SurfaceSourceCharts):
        raise TypeError(
            "Accepted native B-Rep surface construction lost SurfaceSourceCharts."
        )
    result = certify_cell_mesh(
        construction.mesh,
        construction.coordinate_contract,
        geometry=construction.geometry,
        boundary=construction.organization.boundary,
        patches=construction.organization.patches,
        zones=construction.organization.zones,
        labels=construction.organization.labels,
        attributes=construction.organization.attributes,
        associations=construction.organization.associations,
        surface_source=surface_source,
        region_boundary_evidence=construction.organization.region_boundary_evidence,
    )
    if result.surface_source is None:
        raise TypeError("Accepted B-Rep surface result lost its source chart authority.")
    result.surface_source.require_root(result.mesh, result.geometry)
    return result


def _sphere_surface_chart_boundary_query_fixture() -> tuple[
    CellMeshingResult,
    CellMeshingResult,
    SurfaceChartBoundarySource,
]:
    from phydrax.geometry._sphere_material_atlas import (
        prepare_sphere_chart_deformation,
        SphereMaterialCellAtlas,
    )
    from phydrax.geometry._surface_source_support import PreparedSurfaceSourceSupport
    from phydrax.meshing._surface_boundary_sources import SurfaceChartBoundarySource

    contract = SpatialCoordinateContract.si()
    model = brep_sphere(
        1.0,
        coordinate_contract=contract,
        source_id="archive-sphere-chart-boundary",
    )
    result = _surface_result_for_brep(model)
    source_charts = result.surface_source
    if not isinstance(source_charts, SurfaceSourceCharts):
        raise TypeError("The accepted sphere result must retain native source charts.")
    domain = source_charts.domain
    root = MeshingDomainBoundarySource(domain, tuple(range(len(domain.patches))))
    support = PreparedSurfaceSourceSupport(
        domain, result, source_charts.root_parameters, root
    )
    profile_geometry = Sphere((0.0, 0.0, 0.0), 1.0, feature_id=model.source_id).compile()
    profile = AnalyticImplicitProfile(
        profile_geometry,
        contract,
        tube_radius=0.25,
        cover_radius=0.5,
        source_id=model.source_id,
        source_revision=model.source_revision,
    )
    validity = certify_cell_geometry_validity(result.geometry, mesh=result.mesh)
    embedding = certify_global_embedding(result.mesh, result.geometry, validity)
    atlas = SphereMaterialCellAtlas(
        profile,
        result.mesh,
        result.geometry,
        validity,
        embedding,
    )
    deformation = prepare_sphere_chart_deformation(atlas, atlas)
    fidelity = certify_source_fidelity(
        result.mesh,
        result.geometry,
        root,
        tolerance=1.0,
        witness=source_charts.chart_witness,
    )
    query = SurfaceChartBoundarySource(root, support, deformation, fidelity)
    query.require_current()
    if not query.boundary_chart_cover().complete:
        raise ValueError("The sphere chart query fixture must retain complete coverage.")
    if deformation.domain_id == domain.domain_id:
        raise ValueError(
            "Sphere material and source domains must remain distinct identities."
        )
    inputs = MeshCertificationInputs(
        result.mesh,
        result.geometry,
        MeshCertificationSchedule("surface"),
        source_id=model.source_id,
        source_revision=model.source_revision,
        source=query,
        fidelity_tolerance=1.0,
        fidelity_witness=source_charts.chart_witness,
    )
    accepted_result = certify_cell_mesh(
        result.mesh,
        contract,
        geometry=result.geometry,
        certification_inputs=inputs,
        surface_source=source_charts,
    )
    if accepted_result.certification is None:
        raise ValueError("Cold sphere query fixture lost computed certification.")
    accepted_result.certification.require_passed()
    query.require_target_binding(accepted_result.mesh, accepted_result.geometry)
    if accepted_result.certification.inputs.source is not query:
        raise ValueError(
            "Accepted result did not retain its query in certification inputs."
        )
    return result, accepted_result, query


def _surface_chart_boundary_query_composition(
    root_result: CellMeshingResult,
    result: CellMeshingResult,
    query: SurfaceChartBoundarySource,
    /,
) -> Composition:
    if root_result.mesh.mesh_id != result.mesh.mesh_id or cell_geometry_id(
        root_result.geometry
    ) != cell_geometry_id(result.geometry):
        raise ValueError(
            "Sphere source and accepted target must retain the same cell realization."
        )
    mesh_entry = CompositionEntry(
        result.mesh,
        entry_id="surface-query/mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=result.mesh.topology_id,
        revision_id=result.mesh.mesh_id,
        semantics_id="surface-query-root-cell-mesh",
    )
    geometry_entry = CompositionEntry(
        result.geometry,
        entry_id="surface-query/geometry",
        role="discretization",
        owner_id="phydrax.discretization",
        structure_id=result.geometry.geometry_layout_id,
        revision_id=result.mesh.geometry_id,
        semantics_id="surface-query-root-cell-geometry",
        dependencies=(mesh_entry.binding("structure"),),
    )
    source_result_entry = CompositionEntry(
        root_result,
        entry_id="surface-query/root-result",
        role="discretization",
        owner_id="phydrax.meshing",
        structure_id=root_result.mesh.topology_id,
        revision_id=root_result.result_id,
        semantics_id="surface-query-source-root-result",
        dependencies=(
            mesh_entry.binding("structure"),
            geometry_entry.binding("structure"),
        ),
    )
    query_entry = CompositionEntry(
        query,
        entry_id="surface-query/boundary",
        role="topology",
        owner_id="phydrax.geometry",
        structure_id=query.source_query_id,
        revision_id=query.source_revision,
        semantics_id="accepted-surface-chart-boundary-query",
        dependencies=(source_result_entry.binding("revision"),),
    )
    result_entry = CompositionEntry(
        result,
        entry_id="surface-query/result",
        role="discretization",
        owner_id="phydrax.meshing",
        structure_id=result.mesh.topology_id,
        revision_id=result.result_id,
        semantics_id="accepted-native-surface-result",
        dependencies=(
            mesh_entry.binding("structure"),
            geometry_entry.binding("structure"),
            query_entry.binding("revision"),
        ),
    )
    return Composition(
        (mesh_entry, geometry_entry, source_result_entry, query_entry, result_entry),
        boundary_id="accepted-sphere-surface-chart-query",
    )


def _surface_chart_boundary_query_metadata(
    root_result: CellMeshingResult,
    result: CellMeshingResult,
    query: SurfaceChartBoundarySource,
    composition: Composition,
    /,
) -> dict[str, str]:
    from phydrax.discretization._sphere_chart_records import (
        PreparedSphereChartDeformation,
    )

    deformation = query.deformation
    if not isinstance(deformation, PreparedSphereChartDeformation):
        raise TypeError("The sphere chart query must retain its sphere deformation.")
    cover = query.boundary_chart_cover()
    if not cover.complete or cover.semantics != "certified":
        raise ValueError(
            "The accepted sphere chart query must retain certified coverage."
        )
    return {
        "composition_id": composition.composition_id,
        "source_result_id": root_result.result_id,
        "result_id": result.result_id,
        "mesh_id": result.mesh.mesh_id,
        "geometry_id": cell_geometry_id(result.geometry),
        "source_query_id": query.source_query_id,
        "source_id": query.source_id,
        "source_revision": query.source_revision,
        "source_domain_id": query.domain.domain_id,
        "support_id": query.support.support_id,
        "predecessor_fidelity_id": query.predecessor_fidelity.certificate_id,
        "material_domain_id": deformation.domain_id,
        "profile_id": deformation.source_atlas.source_profile.profile_id,
        "source_atlas_id": deformation.source_atlas.atlas_id,
        "target_atlas_id": deformation.target_atlas.atlas_id,
        "deformation_id": deformation.deformation_id,
        "cover_id": cover.cover_id,
    }


def _write_surface_chart_boundary_query(directory: Path, /) -> None:
    root_result, result, query = _sphere_surface_chart_boundary_query_fixture()
    composition = _surface_chart_boundary_query_composition(root_result, result, query)
    expected = _surface_chart_boundary_query_metadata(
        root_result, result, query, composition
    )
    write_model_artifact(
        directory / "surface-chart-boundary-query.phx",
        composition,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    print(json.dumps(expected, sort_keys=True))


def _rebuild_surface_chart_boundary_query_composition(
    value: object,
    expected: dict[str, str],
    /,
) -> Composition:
    from phydrax.meshing._surface_boundary_sources import SurfaceChartBoundarySource

    expected_entries = {
        "surface-query/mesh",
        "surface-query/geometry",
        "surface-query/root-result",
        "surface-query/boundary",
        "surface-query/result",
    }
    if not isinstance(value, Composition) or set(value.entry_ids) != expected_entries:
        raise TypeError(
            "Cold sphere chart query must restore its exact Composition entries."
        )
    entries = {entry.entry_id: entry for entry in value.entries}
    root_result = _rebuild_continuation_value(entries["surface-query/root-result"].value)
    result = _rebuild_continuation_value(entries["surface-query/result"].value)
    query = _rebuild_continuation_value(entries["surface-query/boundary"].value)
    if (
        not isinstance(root_result, CellMeshingResult)
        or not isinstance(result, CellMeshingResult)
        or not isinstance(query, SurfaceChartBoundarySource)
    ):
        raise TypeError(
            "Cold sphere chart query lost its source, accepted result, or query."
        )
    if result.certification is None:
        raise ValueError(
            "Cold accepted sphere result lost its computed certification report."
        )
    report_source = result.certification.inputs.source
    if not isinstance(report_source, SurfaceChartBoundarySource):
        raise ValueError(
            "Cold accepted result did not retain its source query in certification inputs."
        )
    if report_source.source_query_id != query.source_query_id:
        raise ValueError(
            "Cold accepted result certification changed its source query identity."
        )
    result.certification.revalidate(result.mesh, result.geometry)
    result.certification.require_passed()
    query.require_target_binding(result.mesh, result.geometry)
    source_original = query.support.original
    if (
        not isinstance(source_original, CellMeshingResult)
        or source_original.result_id != root_result.result_id
    ):
        raise ValueError(
            "Cold sphere query support is not bound to its retained source result."
        )
    cover = query.boundary_chart_cover()
    if not cover.complete or cover.semantics != "certified":
        raise ValueError("Cold sphere chart query lost its certified boundary cover.")
    archived_mesh = _rebuild_as(entries["surface-query/mesh"].value, CellMesh)
    archived_geometry = _rebuild_geometry_for_mesh(
        entries["surface-query/geometry"].value, archived_mesh
    )
    if archived_mesh.mesh_id != result.mesh.mesh_id or cell_geometry_id(
        archived_geometry
    ) != cell_geometry_id(result.geometry):
        raise ValueError(
            "Cold sphere chart query root banks disagree with its accepted result."
        )
    rebuilt = _surface_chart_boundary_query_composition(root_result, result, query)
    for stored, current in zip(value.entries, rebuilt.entries, strict=True):
        if (
            stored.entry_id != current.entry_id
            or stored.role != current.role
            or stored.owner_id != current.owner_id
            or stored.structure_id != current.structure_id
            or stored.revision_id != current.revision_id
            or stored.semantics_id != current.semantics_id
            or stored.dependencies != current.dependencies
        ):
            raise ValueError("Cold sphere chart query Composition entry binding changed.")
    metadata = _surface_chart_boundary_query_metadata(root_result, result, query, rebuilt)
    if metadata != expected or rebuilt.composition_id != value.composition_id:
        raise ValueError(
            "Cold sphere chart query identity changed during reconstruction."
        )
    return rebuilt


def _read_surface_chart_boundary_query(
    directory: Path,
    expected: dict[str, str],
    /,
) -> None:
    from phydrax.meshing._surface_boundary_sources import SurfaceChartBoundarySource

    composition = read_model_artifact(
        directory / "surface-chart-boundary-query.phx",
        expected_type=Composition,
        restore=lambda value: _rebuild_surface_chart_boundary_query_composition(
            value, expected
        ),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    query = composition.entry("surface-query/boundary").value
    if not isinstance(query, SurfaceChartBoundarySource):
        raise TypeError("Cold sphere chart query did not restore as canonical authority.")
    cover = query.boundary_chart_cover()
    print(
        json.dumps(
            {
                **expected,
                "restored_complete": cover.complete,
                "restored_semantics": cover.semantics,
                "restored_chart_count": int(cover.simplices.shape[0]),
            },
            sort_keys=True,
        )
    )


def _read_surface_chart_witness(
    directory: Path,
    expected: dict[str, object],
    /,
    *,
    source_model: BRepModel | None = None,
) -> SurfaceChartWitness:
    model = (
        load_brep_archive(directory / "refined.phx")
        if source_model is None
        else source_model
    )
    source_result = _surface_result_for_brep(model)
    if source_result.surface_source is None:
        raise TypeError("Cold accepted surface result lost SurfaceSourceCharts.")
    source_witness = source_result.surface_source.chart_witness
    archived = read_model_artifact(
        directory / "surface-chart-witness.phx",
        expected_type=SurfaceChartWitness,
        restore=_rebuild_continuation_value,
        limits=None,
    )
    if not isinstance(archived, SurfaceChartWitness):
        raise TypeError("Cold source chart archive lost its typed ancestry witness.")
    if (
        archived.witness_id != expected["source_chart_witness_id"]
        or source_result.result_id != expected["source_surface_result_id"]
        or source_result.mesh.mesh_id != expected["source_surface_mesh_id"]
        or cell_geometry_id(source_result.geometry)
        != expected["source_surface_geometry_id"]
        or source_result.surface_source.source_chart_id
        != expected["source_surface_chart_id"]
        or archived.witness_id != source_witness.witness_id
        or tuple(int(value) for value in archived.source_root_cell_ids)
        != _expected_sequence(expected["source_root_cell_ids"])
        or archived.source_piece_ids
        != tuple(
            _expected_sequence(piece)
            for piece in _expected_sequence(expected["source_piece_ids"])
        )
        or _surface_root_binding_tokens(archived)
        != expected["source_root_mesh_cell_bindings"]
        or archived.root_mesh_cell_bindings != source_witness.root_mesh_cell_bindings
        or archived.chart_rational_tokens != source_witness.chart_rational_tokens
        or archived.exact_chart(0) != source_witness.exact_chart(0)
    ):
        raise ValueError(
            "Cold source chart witness lost its exact root-cell or piece ancestry."
        )
    return archived


def _write(directory: Path, /) -> None:
    contract = SpatialCoordinateContract.si()
    sphere = brep_sphere(1.0, coordinate_contract=contract)
    tessellated_box = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=contract,
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.2,
            angular_deflection=0.5,
            trim_samples_per_edge=3,
        ),
        source_id="archive-tessellated-box",
    )
    if tessellated_box.tessellation is None:
        raise TypeError("The native B-Rep archive fixture must retain its tessellation.")
    tessellation = tessellated_box.tessellation
    box, placed = _placed_box(contract)
    refined = _refined_rational_patch(contract)
    offset_result = _offset_sphere(sphere)
    offset = offset_result.model
    mapped = _surface_mapped_square(contract)
    boolean_carriers = _boolean_curve_carrier_model(contract)
    bspline_surface = _bspline_surface_model(contract)
    offset_ruled = _offset_ruled_surface_model(contract)
    models = {
        "sphere": sphere,
        "box": box,
        "placed": placed,
        "refined": refined,
        "offset": offset,
        "mapped": mapped,
        "boolean-carriers": boolean_carriers,
        "bspline-surface": bspline_surface,
        "offset-ruled": offset_ruled,
        "tessellated-box": tessellated_box,
    }
    for name, model in models.items():
        save_brep_archive(model, directory / f"{name}.phx")
    source_surface_result = _surface_result_for_brep(refined)
    if source_surface_result.surface_source is None:
        raise TypeError("The accepted T-spline surface result lost SurfaceSourceCharts.")
    source_chart_witness = source_surface_result.surface_source.chart_witness
    write_model_artifact(
        directory / "surface-chart-witness.phx", source_chart_witness, limits=None
    )
    continuation, restart_manifest = _continuation_composition(contract)
    write_model_artifact(directory / "continuation.phx", continuation, limits=None)
    write_model_artifact(
        directory / "finite-element-restart.phx", restart_manifest, limits=None
    )
    fe_state = continuation.value("solver/state")
    fe_ownership = fe_state.field_ownership
    auxiliary_layouts = restart_manifest.auxiliary_layouts
    integrator_layouts = restart_manifest.integrator_layouts
    material_layouts = restart_manifest.material_layouts
    if (
        fe_ownership is None
        or auxiliary_layouts is None
        or integrator_layouts is None
        or material_layouts is None
    ):
        raise TypeError("FE continuation archive lost explicit state ownership layouts.")
    restart_integrator_state = dict(restart_manifest.integrator_state)
    restart_rng_key = restart_integrator_state["rng"]
    tspline_patch = refined.patches[0]
    if not isinstance(tspline_patch, TSplineSurfacePatch):
        raise TypeError("The exact-transfer fixture must contain a T-spline patch.")
    tspline_parameters = np.asarray((0.37, 0.41), dtype=np.float64)
    tspline_point = np.asarray(
        jax.jit(lambda parameters: tspline_patch.evaluate(parameters))(
            jnp.asarray(tspline_parameters)
        )
    )
    tspline_gradient = np.asarray(
        jax.jit(jax.jacfwd(lambda parameters: tspline_patch.evaluate(parameters)))(
            jnp.asarray(tspline_parameters)
        )
    )
    offset_patch = offset.patches[0]
    if not isinstance(offset_patch, OffsetSurface):
        raise TypeError("Offset archive fixture lost its exact OffsetSurface carrier.")
    offset_root_state = _sphere_offset_root_state(sphere, offset)
    print(
        json.dumps(
            {
                "sphere_revision": sphere.source_revision,
                "sphere_origin": sphere.report.origin_revision,
                "sphere_geometry": sphere.geometry.geometry_id,
                "box_revision": box.source_revision,
                "placed_revision": placed.source_revision,
                "refined_revision": refined.source_revision,
                "mapped_revision": mapped.source_revision,
                "offset_revision": offset.source_revision,
                "boolean_carrier_revision": boolean_carriers.source_revision,
                "offset_frame_sign": offset_patch.frame_sign,
                "offset_source_model_id": offset_result.source_model_id,
                "offset_volume_status": offset_result.evidence.volume_status,
                "offset_volume_enclosure": None
                if offset_result.evidence.volume_enclosure is None
                else np.asarray(offset_result.evidence.volume_enclosure).tolist(),
                "offset_root_state": offset_root_state,
                "tessellation": {
                    "tessellation_id": tessellation.tessellation_id,
                    "source_revision": tessellated_box.source_revision,
                    "source_chart_witness_id": tessellation.source_chart_witness.witness_id,
                    "vertex_global_ids": tessellation.vertex_global_ids.tolist(),
                    "triangle_global_ids": tessellation.triangle_global_ids.tolist(),
                    "face_ids": tessellation.face_ids.tolist(),
                    "source_root_cell_ids": np.asarray(
                        tessellation.source_chart_witness.source_root_cell_ids
                    ).tolist(),
                    "root_mesh_cell_bindings": _surface_root_binding_tokens(
                        tessellation.source_chart_witness
                    ),
                    "source_piece_ids": [
                        list(row)
                        for row in tessellation.source_chart_witness.source_piece_ids
                    ],
                    "geometry_entity_ids": tessellation.source_chart_witness.geometry_entity_ids,
                    "occurrence_paths": tessellation.source_chart_witness.occurrence_paths,
                    "chart_rational_tokens": [
                        [
                            [
                                [int(numerator), int(denominator)]
                                for numerator, denominator in point
                            ]
                            for point in triangle
                        ]
                        for triangle in tessellation.source_chart_witness.chart_rational_tokens
                    ],
                },
                "tspline_parameters": tspline_parameters.tolist(),
                "tspline_point": tspline_point.tolist(),
                "tspline_gradient": tspline_gradient.tolist(),
                "tspline_source_mesh": _tspline_mesh_archive_record(
                    tspline_patch.source_mesh
                ),
                "tspline_target_mesh": _tspline_mesh_archive_record(tspline_patch.mesh),
                "tspline_exact_source_control_points": _fraction_rows_for_archive(
                    tspline_patch.exact_source_control_points
                ),
                "tspline_exact_source_weights": [
                    _fraction_pair_for_archive(weight)
                    for weight in tspline_patch.exact_source_weights
                ],
                "tspline_exact_target_homogeneous": _fraction_rows_for_archive(
                    tspline_patch.exact_target_homogeneous
                ),
                "tspline_source_mesh_id": tspline_patch.source_mesh.mesh_id,
                "tspline_target_mesh_id": tspline_patch.mesh.mesh_id,
                "tspline_source_cell_ids": [
                    cell.cell_id for cell in tspline_patch.source_mesh.cells
                ],
                "tspline_target_cell_ids": [
                    cell.cell_id for cell in tspline_patch.mesh.cells
                ],
                "tspline_source_cell_lineage": [
                    [cell_id, list(address)]
                    for cell_id, address in tspline_patch.source_mesh.cell_lineage
                ],
                "tspline_target_cell_lineage": [
                    [cell_id, list(address)]
                    for cell_id, address in tspline_patch.mesh.cell_lineage
                ],
                "tspline_source_extension_count": len(
                    tspline_patch.source_mesh.t_junction_extensions
                ),
                "tspline_target_extension_count": len(
                    tspline_patch.mesh.t_junction_extensions
                ),
                "source_surface_result_id": source_surface_result.result_id,
                "source_surface_mesh_id": source_surface_result.mesh.mesh_id,
                "source_surface_geometry_id": cell_geometry_id(
                    source_surface_result.geometry
                ),
                "source_surface_chart_id": source_surface_result.surface_source.source_chart_id,
                "source_chart_witness_id": source_chart_witness.witness_id,
                "source_root_cell_ids": np.asarray(
                    source_chart_witness.source_root_cell_ids
                ).tolist(),
                "source_root_mesh_cell_bindings": _surface_root_binding_tokens(
                    source_chart_witness
                ),
                "source_piece_ids": [
                    list(piece) for piece in source_chart_witness.source_piece_ids
                ],
                "continuation": {
                    "composition_id": continuation.composition_id,
                    "structure_id": continuation.structure_id,
                    "source_revision": continuation.value("cad/source").source_revision,
                    "result_id": continuation.value("mesh/result").result_id,
                    "mesh_id": continuation.value("mesh/topology").mesh_id,
                    "mesh_topology_id": continuation.value("mesh/topology").topology_id,
                    "geometry_layout_id": continuation.value(
                        "mesh/geometry"
                    ).geometry_layout_id,
                    "coordinate_element_id": continuation.value("mesh/geometry")
                    .elements[0]
                    .element_id,
                    "accepted_id": continuation.value("solver/state").accepted_id,
                    "field_layout_ids": [
                        layout.layout_id
                        for layout in continuation.value("solver/state").field_layouts
                    ],
                    "field_spec_ids": [
                        layout.field_spec.field_spec_id
                        for layout in continuation.value("solver/state").field_layouts
                    ],
                    "field_form_identities": [
                        layout.form_identity
                        for layout in continuation.value("solver/state").field_layouts
                    ],
                    "field_storage_kinds": [
                        layout.storage_kind
                        for layout in continuation.value("solver/state").field_layouts
                    ],
                    "field_periodic_quotients": [
                        layout.periodic_quotient
                        for layout in continuation.value("solver/state").field_layouts
                    ],
                    "field_transfer_semantics": [
                        layout.transfer_semantics.value
                        for layout in continuation.value("solver/state").field_layouts
                    ],
                    "field_ownership_ids": [item.ownership_id for item in fe_ownership],
                    "field_dof_keys": [
                        np.asarray(item.dof_keys, dtype=np.int64).tolist()
                        for item in fe_ownership
                    ],
                    "fe_field_values": [
                        np.asarray(item).tolist() for item in fe_state.fields
                    ],
                    "restart_auxiliary_names": [
                        name for name, _ in restart_manifest.auxiliary_state
                    ],
                    "restart_auxiliary_values": [
                        np.asarray(item).tolist()
                        for _, item in restart_manifest.auxiliary_state
                    ],
                    "restart_integrator_names": [
                        name for name, _ in restart_manifest.integrator_state
                    ],
                    "restart_previous_step": np.asarray(
                        dict(restart_manifest.integrator_state)["previous_step"]
                    ).tolist(),
                    "restart_manifest_id": restart_manifest.manifest_id,
                    "restart_auxiliary_layout_ids": [
                        layout.layout_id for layout in auxiliary_layouts
                    ],
                    "restart_integrator_layout_ids": [
                        layout.layout_id for layout in integrator_layouts
                    ],
                    "restart_material_layout_ids": [
                        layout.layout_id for layout in material_layouts
                    ],
                    "restart_rng_implementation": str(
                        jax.random.key_impl(restart_rng_key)
                    ),
                    "restart_rng_key_data": np.asarray(
                        jax.random.key_data(restart_rng_key), dtype=np.uint32
                    ).tolist(),
                    "material_id": continuation.value("solver/materials").transaction_id,
                    "periodic_id": continuation.value(
                        "mesh/periodicity"
                    ).periodic_topology_id,
                    "distribution_id": continuation.value(
                        "mesh/distribution"
                    ).distribution_id,
                    "volume_chain_id": continuation.value(
                        "material/volume-chain"
                    ).chain_id,
                    "volume_reference_id": continuation.value(
                        "material/reference-declaration"
                    ).declaration_id,
                    "volume_deformation_ids": [
                        continuation.value(
                            "material/refinement-deformation"
                        ).deformation_id,
                        continuation.value(
                            "material/coarsening-deformation"
                        ).deformation_id,
                    ],
                    "volume_map_ids": [
                        item.map_id
                        for item in continuation.value("material/volume-chain").maps
                    ],
                    "volume_root_mesh_id": continuation.value(
                        "material/root-mesh"
                    ).mesh_id,
                    "volume_root_geometry_id": cell_geometry_id(
                        continuation.value("material/root-geometry")
                    ),
                    "volume_refined_mesh_id": continuation.value(
                        "material/refined-mesh"
                    ).mesh_id,
                    "volume_refined_geometry_id": cell_geometry_id(
                        continuation.value("material/refined-geometry")
                    ),
                },
            }
        )
    )


def _read_continuation(path: Path, expected: dict[str, object], /) -> None:
    composition = read_model_artifact(
        path,
        expected_type=Composition,
        restore=_rebuild_composition,
        limits=None,
    )
    restart_manifest = read_model_artifact(
        path.with_name("finite-element-restart.phx"),
        expected_type=FiniteElementRestartManifest,
        restore=FiniteElementRestartManifest.from_stored_fields,
        limits=None,
    )
    assert composition.composition_id == expected["composition_id"]
    assert composition.structure_id == expected["structure_id"]

    source = composition.value("cad/source")
    assert isinstance(source, BRepModel)
    assert source.source_revision == expected["source_revision"]
    assert source.tessellation is None
    assert source.geometry.vertex_exact_points[0] == (
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    placed_root = source.geometry.vertex_roots[0]
    assert isinstance(placed_root, BRepVertexRoot)
    assert isinstance(placed_root.primary, BRepPlacedVertex)
    np.testing.assert_array_equal(
        np.asarray(placed_root.primary.source_point), (0.5, 0.0, 0.0)
    )
    assert placed_root.primary.source_entity_id == "archive-test:square:vertex-0"
    assert placed_root.primary.affine.matrix[0][3] == Fraction(-1, 2)
    assert (
        placed_root.primary.affine.apply((Fraction(1, 2), Fraction(0), Fraction(0)))
        == source.geometry.vertex_exact_points[0]
    )
    exact_curve = source.geometry.curves[0]
    assert isinstance(exact_curve, ExactLineCurve)
    assert exact_curve.evaluate_exact(Fraction(1, 2)) == (
        Fraction(1, 2),
        Fraction(0),
        Fraction(0),
    )
    point = np.asarray(
        source.evaluate_face_carrier(
            jnp.asarray((0,), dtype=jnp.int32),
            jnp.asarray(((0.5, 0.5),), dtype=jnp.float64),
        )
    )[0]
    np.testing.assert_array_equal(point, (0.5, 0.5, 0.0))

    mesh = composition.value("mesh/topology")
    geometry = composition.value("mesh/geometry")
    assert isinstance(mesh, CellMesh)
    assert isinstance(geometry, CellGeometrySpec)
    result = composition.value("mesh/result")
    assert isinstance(result, CellMeshingResult)
    assert result.result_id == expected["result_id"]
    assert result.mesh.mesh_id == mesh.mesh_id
    assert result.geometry.geometry_layout_id == geometry.geometry_layout_id
    assert mesh.mesh_id == expected["mesh_id"]
    assert mesh.topology_id == expected["mesh_topology_id"]
    assert geometry.geometry_layout_id == expected["geometry_layout_id"]
    _, routes, coordinates = geometry.resolve(mesh)
    coordinate_element = geometry.elements[0]
    assert isinstance(coordinate_element, RationalComposedCellGeometryElement)
    assert coordinate_element.element_id == expected["coordinate_element_id"]
    assert coordinate_element.chart_coefficients == (
        (
            (((1, 0), Fraction(1, 3)),),
            (((0, 0), Fraction(1, 3)),),
        ),
        (
            (((0, 1), Fraction(1, 3)),),
            (((0, 0), Fraction(1, 3)),),
        ),
    )
    reference = jnp.asarray((0.2, 0.3), dtype=jnp.float64)
    dofs = jnp.asarray(coordinates)[jnp.asarray(routes[0][0])]

    def mapped_point(point: jax.Array, /) -> jax.Array:
        basis, _ = coordinate_element.tabulate(point[None, :])
        return basis[0] @ dofs

    compiled_map = jax.jit(mapped_point)
    compiled_gradient = jax.jit(jax.jacfwd(mapped_point))
    mapped = compiled_map(reference)
    gradient = compiled_gradient(reference)
    np.testing.assert_allclose(np.asarray(mapped), (0.25, 0.15), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(gradient), ((0.5, 0.5), (0.0, 0.5)), rtol=0.0, atol=1e-14
    )
    host_mapped = geometry.evaluate(
        mesh, np.asarray((500,), dtype=np.int64), np.asarray(((0.2, 0.3),))
    )[0, 0]
    _, basis_gradient = coordinate_element.tabulate(reference[None, :])
    expected_gradient = np.asarray(dofs).T @ np.asarray(basis_gradient)[0]
    np.testing.assert_allclose(np.asarray(mapped), host_mapped, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(gradient), expected_gradient, rtol=0.0, atol=1e-14
    )
    np.testing.assert_array_equal(
        np.asarray(routes[0]), np.asarray(mesh.blocks[0].vertices)
    )
    np.testing.assert_array_equal(point[:2], np.asarray(mesh.coordinates)[4])

    state = composition.value("solver/state")
    materials = composition.value("solver/materials")
    assert type(state) is FiniteElementAcceptedState
    assert type(materials) is MaterialTransaction
    if state.field_ownership is None or restart_manifest.state.field_ownership is None:
        raise ValueError("Cold FE continuation lost explicit DOF ownership.")
    if (
        state.accepted_id != expected["accepted_id"]
        or restart_manifest.state.accepted_id != state.accepted_id
        or restart_manifest.manifest_id != expected["restart_manifest_id"]
    ):
        raise ValueError("Cold FE state/restart manifest identities diverged.")
    if (
        restart_manifest.auxiliary_layouts is None
        or restart_manifest.integrator_layouts is None
        or restart_manifest.material_layouts is None
    ):
        raise ValueError("Cold FE restart manifest lost its explicit state layouts.")
    assert [layout.layout_id for layout in state.field_layouts] == expected[
        "field_layout_ids"
    ]
    assert [
        layout.field_spec.field_spec_id for layout in state.field_layouts
    ] == expected["field_spec_ids"]
    assert [
        None if layout.form_identity is None else list(layout.form_identity)
        for layout in state.field_layouts
    ] == expected["field_form_identities"]
    assert [layout.storage_kind for layout in state.field_layouts] == expected[
        "field_storage_kinds"
    ]
    assert [layout.periodic_quotient for layout in state.field_layouts] == expected[
        "field_periodic_quotients"
    ]
    assert [
        layout.transfer_semantics.value for layout in state.field_layouts
    ] == expected["field_transfer_semantics"]
    assert [item.ownership_id for item in state.field_ownership] == expected[
        "field_ownership_ids"
    ]
    assert [
        np.asarray(item.dof_keys, dtype=np.int64).tolist()
        for item in state.field_ownership
    ] == expected["field_dof_keys"]
    assert [
        item.ownership_id for item in restart_manifest.state.field_ownership
    ] == expected["field_ownership_ids"]
    expected_fields = _expected_sequence(expected["fe_field_values"])
    assert len(state.fields) == len(expected_fields)
    for actual, stored in zip(state.fields, expected_fields, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), _expected_array(stored))
    assert state.materials is not None
    assert state.materials.transaction_id == expected["material_id"]
    assert [
        layout.layout_id for layout in restart_manifest.auxiliary_layouts
    ] == expected["restart_auxiliary_layout_ids"]
    assert [
        layout.layout_id for layout in restart_manifest.integrator_layouts
    ] == expected["restart_integrator_layout_ids"]
    assert [layout.layout_id for layout in restart_manifest.material_layouts] == expected[
        "restart_material_layout_ids"
    ]
    assert [name for name, _ in restart_manifest.auxiliary_state] == expected[
        "restart_auxiliary_names"
    ]
    expected_auxiliary_values = _expected_sequence(expected["restart_auxiliary_values"])
    for (_, actual), stored in zip(
        restart_manifest.auxiliary_state, expected_auxiliary_values, strict=True
    ):
        np.testing.assert_array_equal(np.asarray(actual), _expected_array(stored))
    integrator_state = dict(restart_manifest.integrator_state)
    np.testing.assert_array_equal(
        np.asarray(integrator_state["previous_step"]),
        _expected_array(expected["restart_previous_step"]),
    )
    rng_key = integrator_state["rng"]
    assert str(jax.random.key_impl(rng_key)) == expected["restart_rng_implementation"]
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(rng_key)),
        np.asarray(_expected_array(expected["restart_rng_key_data"]), dtype=np.uint32),
    )
    assert state.materials.transaction_id == expected["material_id"]
    np.testing.assert_array_equal(
        np.asarray(materials.state("steel:integration-site:0").committed), (2.5, 3.5)
    )
    np.testing.assert_array_equal(
        np.asarray(materials.state("steel:integration-site:0").trial), (2.75, 3.75)
    )

    periodic = composition.value("mesh/periodicity")
    assert isinstance(periodic, PeriodicMeshTopology)
    assert periodic.periodic_topology_id == expected["periodic_id"]
    assert mesh.periodic_topology is not None
    assert mesh.periodic_topology.periodic_topology_id == periodic.periodic_topology_id
    periodic_cell = periodic.cell
    if not isinstance(periodic_cell, PeriodicCell):
        raise TypeError("Cold field continuation requires its periodic lattice owner.")
    wrapped, shifts = periodic_cell.wrap(np.asarray((1.25, -0.25), dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(wrapped), (0.25, 0.75))
    np.testing.assert_array_equal(np.asarray(shifts), (1, -1))

    distribution = composition.value("mesh/distribution")
    assert isinstance(distribution, MeshDistribution)
    assert distribution.distribution_id == expected["distribution_id"]
    assert distribution.part.carrier.result_id == expected["result_id"]
    sample = np.arange(mesh.blocks[0].cell_count, dtype=np.float64) + 10.0
    np.testing.assert_array_equal(
        np.asarray(distribution.gather(0, sample)),
        sample[np.asarray(distribution.owned_rows[0])],
    )
    volume_root_mesh = composition.value("material/root-mesh")
    volume_root_geometry = composition.value("material/root-geometry")
    volume_refined_mesh = composition.value("material/refined-mesh")
    volume_refined_geometry = composition.value("material/refined-geometry")
    volume_chain = composition.value("material/volume-chain")
    material_reference = composition.value("material/reference-declaration")
    refinement_deformation = composition.value("material/refinement-deformation")
    coarsening_deformation = composition.value("material/coarsening-deformation")
    assert isinstance(volume_root_mesh, CellMesh)
    assert isinstance(volume_root_geometry, CellGeometrySpec)
    assert isinstance(volume_refined_mesh, CellMesh)
    assert isinstance(volume_refined_geometry, CellGeometrySpec)
    assert isinstance(volume_chain, VolumeMaterialChain)
    assert volume_chain.chain_id == expected["volume_chain_id"]
    assert [item.map_id for item in volume_chain.maps] == expected["volume_map_ids"]
    assert isinstance(material_reference, MaterialReferenceDeclaration)
    assert isinstance(refinement_deformation, VolumeMaterialDeformation)
    assert isinstance(coarsening_deformation, VolumeMaterialDeformation)
    assert material_reference.declaration_id == expected["volume_reference_id"]
    assert [
        refinement_deformation.deformation_id,
        coarsening_deformation.deformation_id,
    ] == expected["volume_deformation_ids"]
    assert material_reference.root_mesh.mesh_id == volume_root_mesh.mesh_id
    assert cell_geometry_id(material_reference.root_geometry) == cell_geometry_id(
        volume_root_geometry
    )
    assert volume_chain.reference.declaration_id == material_reference.declaration_id
    assert volume_chain.source_mesh_id == volume_root_mesh.mesh_id
    assert volume_chain.target_mesh_id == volume_root_mesh.mesh_id
    assert volume_chain.source_geometry_id == cell_geometry_id(volume_root_geometry)
    assert volume_chain.target_geometry_id == cell_geometry_id(volume_root_geometry)
    assert volume_root_mesh.mesh_id == expected["volume_root_mesh_id"]
    assert cell_geometry_id(volume_root_geometry) == expected["volume_root_geometry_id"]
    assert volume_refined_mesh.mesh_id == expected["volume_refined_mesh_id"]
    assert (
        cell_geometry_id(volume_refined_geometry)
        == expected["volume_refined_geometry_id"]
    )
    assert volume_chain.source_map.source_id == "native-cad-source"
    assert volume_chain.source_map.source_revision == "revision-4"
    assert volume_chain.source_map.occurrence_paths == (("assembly-1", "occurrence-9"),)
    refinement, coarsening = volume_chain.maps
    assert refinement.source_mesh.mesh_id == volume_root_mesh.mesh_id
    assert refinement.target_mesh.mesh_id == volume_refined_mesh.mesh_id
    assert coarsening.source_mesh.mesh_id == volume_refined_mesh.mesh_id
    assert coarsening.target_mesh.mesh_id == volume_root_mesh.mesh_id
    assert cell_geometry_id(refinement.source_geometry) == cell_geometry_id(
        volume_root_geometry
    )
    assert cell_geometry_id(refinement.target_geometry) == cell_geometry_id(
        volume_refined_geometry
    )
    assert cell_geometry_id(coarsening.source_geometry) == cell_geometry_id(
        volume_refined_geometry
    )
    assert cell_geometry_id(coarsening.target_geometry) == cell_geometry_id(
        volume_root_geometry
    )
    assert refinement.deformation.deformation_id == refinement_deformation.deformation_id
    assert coarsening.deformation.deformation_id == coarsening_deformation.deformation_id
    assert (
        refinement.deformation.reference_witness.root_mesh_id == volume_root_mesh.mesh_id
    )
    assert refinement.deformation.reference_witness.root_geometry_id == cell_geometry_id(
        volume_root_geometry
    )
    assert refinement_deformation.source_reference_maps[0][0] == "root"
    assert refinement_deformation.source_reference_maps[0][
        1
    ].source_reference_exact_vertices == (ROOT_REFERENCE_TETRAHEDRON,)
    assert refinement_deformation.target_reference_maps[0][0] == "refined"
    assert (
        refinement_deformation.target_reference_maps[0][1].source_reference_exact_vertices
        is not None
    )
    assert (
        coarsening_deformation.source_reference_maps[0][1].source_reference_exact_vertices
        is not None
    )
    initial_content = np.asarray(((3.25, 1.75),), dtype=np.float64)
    refinement_action = refinement.apply_material_extensive_with_evidence(initial_content)
    coarsening_action = coarsening.apply_material_extensive_with_evidence(
        refinement_action.values
    )
    continued_content = volume_chain.apply_material_extensive(initial_content)
    assert refinement_action.values.shape == (2, 2)
    assert coarsening_action.values.shape == initial_content.shape
    assert np.all(np.isfinite(np.asarray(continued_content)))
    tolerances = tuple(
        item.integration_policy.absolute_tolerance
        + item.integration_policy.relative_tolerance
        * float(np.max(np.abs(np.asarray(action.before_total))))
        for item, action in zip(
            volume_chain.maps,
            (refinement_action, coarsening_action),
            strict=True,
        )
    )
    assert np.all(np.asarray(refinement_action.absolute_defect) <= tolerances[0])
    assert np.all(np.asarray(coarsening_action.absolute_defect) <= tolerances[1])
    np.testing.assert_allclose(
        np.asarray(continued_content),
        initial_content,
        rtol=0.0,
        atol=sum(tolerances),
    )
    np.testing.assert_allclose(
        np.asarray(continued_content),
        np.asarray(coarsening_action.values),
        rtol=0.0,
        atol=0.0,
    )


def _read(directory: Path, expected: dict[str, object], /) -> None:
    if (
        expected["offset_volume_status"] != "unresolved"
        or expected["offset_volume_enclosure"] is not None
    ):
        raise ValueError("Sphere offset fixture must retain unresolved volume status.")
    sphere = load_brep_archive(directory / "sphere.phx")
    assert sphere.source_revision == expected["sphere_revision"]
    assert sphere.report.origin_revision == expected["sphere_origin"]
    assert sphere.geometry.geometry_id == expected["sphere_geometry"]
    assert sphere.tessellation is None
    assert isinstance(sphere.patches[0], RevolutionSurface)
    parameters = jnp.asarray(
        np.mean(np.asarray(sphere.parameter_bounds[0]), axis=0)[None, :],
        dtype=jnp.float64,
    )
    point = np.asarray(
        sphere.evaluate_face_carrier(jnp.asarray((0,), dtype=jnp.int32), parameters)
    )[0]
    np.testing.assert_allclose(np.linalg.norm(point), 1.0, rtol=0.0, atol=2e-14)
    assert sphere.geometry.curves and sphere.geometry.pcurves
    tessellated_box = load_brep_archive(directory / "tessellated-box.phx")
    tessellation = tessellated_box.tessellation
    if tessellation is None:
        raise ValueError("Cold B-Rep archive lost its retained tessellation.")
    witness = tessellation.source_chart_witness
    stored_tessellation_raw = expected.get("tessellation")
    if not isinstance(stored_tessellation_raw, dict):
        raise TypeError(
            "Cold tessellation evidence must retain its nested metadata object."
        )
    stored_tessellation = stored_tessellation_raw
    assert tessellated_box.source_revision == stored_tessellation["source_revision"]
    assert tessellation.source_revision == tessellated_box.source_revision
    assert tessellation.tessellation_id == stored_tessellation["tessellation_id"]
    assert witness.witness_id == stored_tessellation["source_chart_witness_id"]
    np.testing.assert_array_equal(
        np.asarray(tessellation.vertex_global_ids),
        stored_tessellation["vertex_global_ids"],
    )
    np.testing.assert_array_equal(
        np.asarray(tessellation.triangle_global_ids),
        stored_tessellation["triangle_global_ids"],
    )
    np.testing.assert_array_equal(
        np.asarray(tessellation.face_ids), stored_tessellation["face_ids"]
    )
    assert len(tessellated_box.patches) == 6
    np.testing.assert_array_equal(
        np.unique(np.asarray(tessellation.face_ids)), np.arange(6, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(witness.source_root_cell_ids),
        stored_tessellation["source_root_cell_ids"],
    )
    assert (
        _surface_root_binding_tokens(witness)
        == stored_tessellation["root_mesh_cell_bindings"]
    )
    assert witness.source_piece_ids == tuple(
        tuple(row) for row in stored_tessellation["source_piece_ids"]
    )
    assert witness.geometry_entity_ids == tuple(
        stored_tessellation["geometry_entity_ids"]
    )
    assert witness.occurrence_paths == tuple(
        tuple(path) for path in stored_tessellation["occurrence_paths"]
    )
    chart_tokens = [
        [
            [[int(numerator), int(denominator)] for numerator, denominator in point]
            for point in triangle
        ]
        for triangle in witness.chart_rational_tokens
    ]
    assert chart_tokens == stored_tessellation["chart_rational_tokens"]

    box = load_brep_archive(directory / "box.phx")
    placed = load_brep_archive(directory / "placed.phx")
    assert box.source_revision == expected["box_revision"]
    assert placed.source_revision == expected["placed_revision"]
    assert placed.source_revision != box.source_revision
    bspline_surface = load_brep_archive(directory / "bspline-surface.phx")
    surface_patch = bspline_surface.patches[0]
    assert isinstance(surface_patch, BSplineSurfacePatch)
    assert surface_patch.control_point_ids == ((101, 102), (103, 104))
    assert surface_patch.exact_control_points == (
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
        ),
        (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(1), Fraction(0)),
        ),
    )
    assert surface_patch.exact_weights == (
        (Fraction(1), Fraction(1)),
        (Fraction(1), Fraction(1)),
    )
    assert surface_patch.exact_u_knots == (
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(1),
    )
    assert surface_patch.exact_v_knots == (
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(1),
    )
    offset_ruled = load_brep_archive(directory / "offset-ruled.phx")
    ruled_surface = offset_ruled.patches[0]
    assert isinstance(ruled_surface, RuledSurface)
    assert isinstance(ruled_surface.first, OffsetCurve)
    assert ruled_surface.first.distance_exact == Fraction(1, 3)
    np.testing.assert_array_equal(
        np.asarray(ruled_surface.first.distance), np.asarray(float(Fraction(1, 3)))
    )
    np.testing.assert_array_equal(
        np.asarray(ruled_surface.first.direction), np.asarray((0.0, 0.0, 1.0))
    )
    np.testing.assert_allclose(
        np.asarray(ruled_surface.evaluate(jnp.asarray((0.5, 0.5), dtype=jnp.float64))),
        (0.5, float(Fraction(1, 6)), 0.0),
        rtol=0.0,
        atol=1e-15,
    )
    mapped = load_brep_archive(directory / "mapped.phx")
    assert mapped.source_revision == expected["mapped_revision"]
    mapped_curve = mapped.geometry.curves[0]
    assert isinstance(mapped_curve, SurfaceMappedCurve)
    assert isinstance(mapped_curve.surface, PlanePatch)
    assert isinstance(mapped_curve.pcurve, LineCurve)
    assert array_tree_fingerprint(mapped_curve.surface) == array_tree_fingerprint(
        mapped.patches[0]
    )
    assert array_tree_fingerprint(mapped_curve.pcurve) == array_tree_fingerprint(
        mapped.geometry.pcurves[0]
    )
    mapped_point = np.asarray(mapped_curve.evaluate(jnp.asarray(0.5, dtype=jnp.float64)))
    np.testing.assert_array_equal(mapped_point, (0.5, 0.0, 0.0))
    boolean_carriers = load_brep_archive(directory / "boolean-carriers.phx")
    assert boolean_carriers.source_revision == expected["boolean_carrier_revision"]
    exact_curve = boolean_carriers.geometry.curves[0]
    assert isinstance(exact_curve, BSplineCurve)
    assert exact_curve.exact_control_points == (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
    )
    assert exact_curve.exact_weights == (Fraction(1), Fraction(1))
    assert exact_curve.exact_knots == (Fraction(0), Fraction(0), Fraction(1), Fraction(1))
    affine_pcurve, periodic_pcurve = boolean_carriers.geometry.pcurves[:2]
    assert isinstance(affine_pcurve, AffinePCurve)
    assert isinstance(periodic_pcurve, PeriodicPCurve)
    assert affine_pcurve.matrix == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    assert affine_pcurve.offset == (Fraction(0), Fraction(0))
    assert periodic_pcurve.period_shifts == (0, 0)
    test_parameters = jnp.asarray((0.25, 0.75), dtype=jnp.float64)
    np.testing.assert_array_equal(
        np.asarray(affine_pcurve.evaluate(test_parameters)),
        np.asarray(affine_pcurve.curve.evaluate(test_parameters)),
    )
    np.testing.assert_array_equal(
        np.asarray(periodic_pcurve.evaluate(test_parameters)),
        np.asarray(periodic_pcurve.source_curve.evaluate(test_parameters)),
    )
    conic_pcurve = boolean_carriers.geometry.pcurves[3]
    assert isinstance(conic_pcurve, ConicRootLinePCurve)
    assert conic_pcurve.phase_shift_exact == Fraction(0)
    assert isinstance(conic_pcurve.root, ConicParameterRoot)
    assert isinstance(conic_pcurve.root.conic, CircleCurve)
    assert conic_pcurve.root.cosine_exact == Fraction(1)
    assert conic_pcurve.root.sine_exact == Fraction(0)
    assert conic_pcurve.root.exact_point() == (Fraction(1), Fraction(0), Fraction(0))
    assert conic_pcurve.root.conic.center_exact == (Fraction(0), Fraction(0), Fraction(0))
    assert conic_pcurve.root.conic.first_axis_exact == (
        Fraction(1),
        Fraction(0),
        Fraction(0),
    )
    assert conic_pcurve.root.conic.second_axis_exact == (
        Fraction(0),
        Fraction(1),
        Fraction(0),
    )
    assert conic_pcurve.root.conic.radius_exact == Fraction(1)
    assert conic_pcurve.height_origin_exact == Fraction(1)
    assert conic_pcurve.height_direction_exact == Fraction(-1)
    conic_vertex = boolean_carriers.geometry.vertex_roots[1]
    assert isinstance(conic_vertex, BRepVertexRoot)
    assert isinstance(conic_vertex.primary, BRepConicVertex)
    assert conic_vertex.primary.offset_exact == (Fraction(0), Fraction(0), Fraction(0))
    assert conic_vertex.primary.point_exact == (Fraction(1), Fraction(0), Fraction(0))
    edge_endpoint = boolean_carriers.geometry.edge_endpoint_roots[0][1]
    assert isinstance(edge_endpoint, ConicParameterEndpoint)
    assert edge_endpoint.root.exact_point() == conic_vertex.primary.point_exact
    assert edge_endpoint.parameter_exact == Fraction(1)
    coedge_endpoint = boolean_carriers.geometry.coedge_endpoint_roots[1][0]
    assert isinstance(coedge_endpoint, ConicParameterEndpoint)
    assert coedge_endpoint.root.exact_point() == conic_vertex.primary.point_exact
    assert coedge_endpoint.parameter_exact == Fraction(0)
    conic_parameter = jnp.asarray(0.75, dtype=jnp.float64)
    compiled_conic_point = jax.jit(lambda parameter: conic_pcurve.evaluate(parameter))(
        conic_parameter
    )
    compiled_conic_gradient = jax.jit(
        jax.jacfwd(lambda parameter: conic_pcurve.evaluate(parameter))
    )(conic_parameter)
    np.testing.assert_array_equal(np.asarray(compiled_conic_point), (0.0, 0.25))
    np.testing.assert_array_equal(np.asarray(compiled_conic_gradient), (0.0, -1.0))
    trim_domain = boolean_carriers.trim_domains[0]
    assert isinstance(trim_domain, TrimDomain)
    assert trim_domain.outer is None
    assert trim_domain.holes == ()
    assert len(trim_domain.curve_loops) == 1
    assert len(trim_domain.curve_loops[0]) == 4
    assert trim_domain.identity_payload()["kind"] == "source-curve-trim-domain"
    placed_surface = placed.patches[0]
    placed_curve = placed.geometry.curves[0]
    assert isinstance(placed_surface, PlacedSurface)
    assert isinstance(placed_curve, PlacedCurve)
    exact_affine = (
        (Fraction(1), Fraction(0), Fraction(0), Fraction(1, 3)),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    )
    assert placed_surface.affine.matrix == exact_affine
    assert placed_curve.affine.matrix == exact_affine
    assert placed.geometry.vertex_exact_points == tuple(
        None if point is None else (point[0] + Fraction(1, 3), point[1], point[2])
        for point in box.geometry.vertex_exact_points
    )
    placed_parameters = jnp.asarray(
        np.mean(np.asarray(placed.parameter_bounds[0]), axis=0)[None, :],
        dtype=jnp.float64,
    )
    np.testing.assert_array_equal(
        np.asarray(placed_surface.evaluate(placed_parameters)),
        np.asarray(placed_surface.definition.evaluate(placed_parameters))
        + np.asarray((float(Fraction(1, 3)), 0.0, 0.0)),
    )
    occurrences = tuple(placed.geometry.assembly_graph.iter_occurrences())
    assert tuple(item.path for item in occurrences) == (
        ("pair", "left"),
        ("pair", "right"),
    )
    assert occurrences[1].placement.apply((0, 0, 0)) == (
        Fraction(3, 2),
        Fraction(0),
        Fraction(0),
    )
    assert tuple(item.path for item in placed.geometry.iter_occurrences()) == (
        ("pair", "left"),
        ("pair", "right"),
    )
    containers = tuple(placed.geometry.iter_assembly_containers())
    assert tuple(item.path for item in containers) == (("pair",),)
    assert containers[0].member_paths == (("pair", "left"), ("pair", "right"))
    incidence = placed.geometry.qualified_entity_incidence(
        placed.source_id, placed.source_revision
    )
    entity_ids = tuple(
        entity for record in incidence for entity in (record.container, record.member)
    )
    assert all(entity.source_revision == placed.source_revision for entity in entity_ids)
    assert all(entity.source_id == placed.source_id for entity in entity_ids)
    assert {entity.occurrence_path for entity in entity_ids} == {
        ("pair", "left"),
        ("pair", "right"),
    }

    assert sphere.model_id == expected["offset_source_model_id"]
    offset = load_brep_archive(directory / "offset.phx")
    assert offset.source_revision == expected["offset_revision"]
    offset_patch = offset.patches[0]
    if not isinstance(offset_patch, OffsetSurface) or not isinstance(
        offset_patch.base, RevolutionSurface
    ):
        raise TypeError("Cold sphere offset lost its RevolutionSurface source proof.")
    assert offset_patch.distance_exact == Fraction(1.0 / 3.0)
    assert offset_patch.frame_sign == expected["offset_frame_sign"]
    if _sphere_offset_root_state(sphere, offset) != expected["offset_root_state"]:
        raise ValueError(
            "Cold sphere offset changed exact source or target pole-root ancestry."
        )
    source_witness = offset_patch.base.exact_sphere_revolution_witness()
    if source_witness is None:
        raise ValueError("Cold sphere offset lost its recomputed source sphere witness.")
    center, radius, _, _ = source_witness
    offset_parameters = jnp.asarray(
        np.mean(np.asarray(offset.parameter_bounds[0]), axis=0)[None, :],
        dtype=jnp.float64,
    )
    offset_point = np.asarray(offset_patch.evaluate(offset_parameters))[0]
    target_radius = float(radius + offset_patch.frame_sign * offset_patch.distance_exact)
    np.testing.assert_allclose(
        np.linalg.norm(offset_point - np.asarray([float(value) for value in center])),
        target_radius,
        rtol=0.0,
        atol=3e-15,
    )
    refined = load_brep_archive(directory / "refined.phx")
    assert refined.source_revision == expected["refined_revision"]
    assert isinstance(refined.geometry.curves[0], ExactLineCurve)
    assert refined.geometry.vertex_exact_points[2] == (
        Fraction(1),
        Fraction(1),
        Fraction(0),
    )
    _read_surface_chart_witness(directory, expected, source_model=refined)
    patch = refined.patches[0]
    assert isinstance(patch, TSplineSurfacePatch)
    tiny = Fraction(1, 3 * 2**100)
    assert patch.exact_source_control_points[0][2] == tiny
    assert patch.exact_source_weights[0] == Fraction(1, 3)
    assert len(patch.transfer_plans) == 1
    assert (
        _tspline_mesh_archive_record(patch.source_mesh) == expected["tspline_source_mesh"]
    )
    assert _tspline_mesh_archive_record(patch.mesh) == expected["tspline_target_mesh"]
    assert (
        _fraction_rows_for_archive(patch.exact_source_control_points)
        == expected["tspline_exact_source_control_points"]
    )
    assert [
        _fraction_pair_for_archive(weight) for weight in patch.exact_source_weights
    ] == expected["tspline_exact_source_weights"]
    assert (
        _fraction_rows_for_archive(patch.exact_target_homogeneous)
        == expected["tspline_exact_target_homogeneous"]
    )
    assert patch.source_mesh.mesh_id == expected["tspline_source_mesh_id"]
    assert patch.mesh.mesh_id == expected["tspline_target_mesh_id"]
    assert [cell.cell_id for cell in patch.source_mesh.cells] == expected[
        "tspline_source_cell_ids"
    ]
    assert [cell.cell_id for cell in patch.mesh.cells] == expected[
        "tspline_target_cell_ids"
    ]
    assert [
        [cell_id, list(address)] for cell_id, address in patch.source_mesh.cell_lineage
    ] == expected["tspline_source_cell_lineage"]
    assert [
        [cell_id, list(address)] for cell_id, address in patch.mesh.cell_lineage
    ] == expected["tspline_target_cell_lineage"]
    assert (
        len(patch.source_mesh.t_junction_extensions)
        == expected["tspline_source_extension_count"]
    )
    assert (
        len(patch.mesh.t_junction_extensions)
        == expected["tspline_target_extension_count"]
    )
    patch.validate_exact_transfer_chain()
    parameters = jnp.asarray(
        _expected_array(expected["tspline_parameters"]), dtype=jnp.float64
    )
    compiled_point = jax.jit(lambda value: patch.evaluate(value))(parameters)
    compiled_gradient = jax.jit(jax.jacfwd(lambda value: patch.evaluate(value)))(
        parameters
    )
    np.testing.assert_allclose(
        np.asarray(compiled_point),
        np.asarray(_expected_array(expected["tspline_point"]), dtype=np.float64),
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(compiled_gradient),
        np.asarray(_expected_array(expected["tspline_gradient"]), dtype=np.float64),
        rtol=0.0,
        atol=1e-14,
    )
    assert any(
        Fraction(0) < abs(row[2]) < Fraction(1, 10**20)
        for row in patch.exact_target_homogeneous
    )
    np.testing.assert_array_equal(
        np.asarray(patch.weights),
        np.full((patch.source_mesh.coefficient_count,), float(Fraction(1, 3))),
    )
    assert all(row[-1] == Fraction(1, 3) for row in patch.exact_target_homogeneous)
    np.testing.assert_array_equal(
        np.asarray(patch.control_points),
        np.asarray(
            [
                [
                    float(sum(anchor.local_knots.u[1:4], Fraction(0)) / 3),
                    float(sum(anchor.local_knots.v[1:4], Fraction(0)) / 3),
                    float(tiny) if index == 0 else 0.0,
                ]
                for index, anchor in enumerate(patch.source_mesh.anchors)
            ]
        ),
    )
    continuation_expected = _expected_object_mapping(expected["continuation"])
    _read_continuation(directory / "continuation.phx", continuation_expected)
    print("cold-restored")


def _polyhedral_periodic_archive_case() -> tuple[
    CellMesh, PeriodicMeshTopology, CellMesh
]:
    from phydrax.discretization._cell_complex import PolyhedralConnectivity

    cube_points = (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 0.5, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (2.0, 0.0, 0.5),
        (2.0, 0.5, 0.5),
        (0.0, 0.5, 0.5),
    )
    tetra_points = (
        (0.5, 1.0, 1.0),
        (0.7, 1.0, 1.0),
        (0.5, 1.2, 1.0),
        (0.5, 1.0, 1.2),
    )
    coordinates = np.asarray((*cube_points, *tetra_points), dtype=np.float64)
    cube_faces = tuple(
        np.asarray(face, dtype=np.int32)
        for face in (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (3, 7, 6, 2),
            (0, 4, 7, 3),
            (1, 2, 6, 5),
        )
    )
    tetra_faces = tuple(
        np.asarray(face, dtype=np.int32)
        for face in (
            (8, 10, 9),
            (8, 9, 11),
            (8, 11, 10),
            (9, 10, 11),
        )
    )
    lifted = CellMesh.from_polyhedra(
        coordinates,
        (cube_faces, tetra_faces),
        block_name="periodic-polyhedral",
        vertex_global_ids=np.arange(1000, 1012, dtype=np.int64),
        cell_global_ids=np.asarray((901, 402), dtype=np.int64),
        numeric_version="periodic-polyhedral-v1",
    )
    connectivity = lifted.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral fixture lost its canonical connectivity owner.")
    cell = PeriodicCell(np.diag((2.0, 2.0, 2.0)))
    representatives = np.arange(coordinates.shape[0], dtype=np.int64)
    representatives[[1, 2, 5, 6]] = (0, 3, 4, 7)
    shifts = np.zeros((coordinates.shape[0], 3), dtype=np.int64)
    shifts[[1, 2, 5, 6], 0] = 1
    periodic = PeriodicMeshTopology(lifted, cell, representatives, shifts)
    mesh = CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        polyhedral_connectivity=connectivity,
        numeric_version=lifted.numeric_version,
        periodic_topology=periodic,
    )
    return lifted, periodic, mesh


def _rebuild_polyhedral_periodic_composition(value: object, /) -> Composition:
    if not isinstance(value, Composition) or set(value.entry_ids) != {
        "poly/lifted-mesh",
        "poly/periodicity",
        "poly/periodic-mesh",
    }:
        raise TypeError(
            "Periodic polyhedral archive requires its three Composition owners."
        )
    entries = {entry.entry_id: entry for entry in value.entries}
    lifted = _rebuild_continuation_value(entries["poly/lifted-mesh"].value)
    if not isinstance(lifted, CellMesh):
        raise TypeError("Periodic polyhedral archive lost its lifted mesh owner.")
    periodic = _rebuild_periodic_topology(entries["poly/periodicity"].value, lifted)
    mesh = _rebuild_continuation_value(entries["poly/periodic-mesh"].value)
    from phydrax.discretization._cell_complex import PolyhedralConnectivity

    if (
        not isinstance(mesh, CellMesh)
        or type(lifted.connectivity) is not PolyhedralConnectivity
        or type(mesh.connectivity) is not PolyhedralConnectivity
        or mesh.periodic_topology is None
        or mesh.periodic_topology.periodic_topology_id != periodic.periodic_topology_id
        or mesh.topology_id != entries["poly/periodic-mesh"].structure_id
        or mesh.mesh_id != entries["poly/periodic-mesh"].revision_id
        or lifted.topology_id != entries["poly/lifted-mesh"].structure_id
        or lifted.mesh_id != entries["poly/lifted-mesh"].revision_id
    ):
        raise ValueError("Periodic polyhedral mesh and quotient owners do not rebind.")
    lifted_entry = entries["poly/lifted-mesh"]
    periodic_entry = entries["poly/periodicity"]
    mesh_entry = entries["poly/periodic-mesh"]
    if (
        lifted_entry.role != "topology"
        or lifted_entry.owner_id != "phydrax.discretization"
        or lifted_entry.semantics_id != "polyhedral-lifted-mesh"
        or periodic_entry.role != "topology"
        or periodic_entry.owner_id != "phydrax.discretization"
        or periodic_entry.structure_id != lifted.topology_id
        or periodic_entry.revision_id != periodic.periodic_topology_id
        or periodic_entry.semantics_id != "periodic-polyhedral-quotient"
        or mesh_entry.role != "topology"
        or mesh_entry.owner_id != "phydrax.discretization"
        or mesh_entry.semantics_id != "periodic-polyhedral-cell-mesh"
    ):
        raise ValueError("Periodic polyhedral Composition entry identities changed.")
    dependency_keys = lambda entry: {
        (item.entry_id, item.facet, item.bound_id) for item in entry.dependencies
    }
    if dependency_keys(periodic_entry) != {
        ("poly/lifted-mesh", "structure", lifted_entry.structure_id),
        ("poly/lifted-mesh", "revision", lifted_entry.revision_id),
    } or dependency_keys(mesh_entry) != {
        ("poly/lifted-mesh", "structure", lifted_entry.structure_id),
        ("poly/lifted-mesh", "revision", lifted_entry.revision_id),
        ("poly/periodicity", "revision", periodic_entry.revision_id),
    }:
        raise ValueError("Periodic polyhedral Composition dependencies changed.")
    values = {
        "poly/lifted-mesh": lifted,
        "poly/periodicity": periodic,
        "poly/periodic-mesh": mesh,
    }
    rebuilt_entries = tuple(
        CompositionEntry(
            values[entry.entry_id],
            entry_id=entry.entry_id,
            role=entry.role,
            owner_id=entry.owner_id,
            structure_id=entry.structure_id,
            revision_id=entry.revision_id,
            semantics_id=entry.semantics_id,
            dependencies=tuple(
                CompositionDependency(
                    dependency.entry_id, dependency.facet, dependency.bound_id
                )
                for dependency in entry.dependencies
            ),
        )
        for entry in value.entries
    )
    composition = Composition(rebuilt_entries, boundary_id=value.boundary_id)
    if composition.composition_id != value.composition_id:
        raise ValueError(
            "Periodic polyhedral Composition identity changed on reconstruction."
        )
    return composition


def _write_polyhedral_periodic_archive(directory: Path, /) -> None:
    from phydrax.discretization._cell_complex import PolyhedralConnectivity
    from phydrax.meshing._periodic import periodic_entity_orbit_map

    lifted, periodic, mesh = _polyhedral_periodic_archive_case()
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral periodic archive requires polyhedral connectivity.")
    periodic_cell = periodic.cell
    if not isinstance(periodic_cell, PeriodicCell):
        raise TypeError("Polyhedral periodic archive requires a periodic lattice cell.")
    orbit = periodic_entity_orbit_map(mesh, 3)
    lifted_entry = CompositionEntry(
        lifted,
        entry_id="poly/lifted-mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=lifted.topology_id,
        revision_id=lifted.mesh_id,
        semantics_id="polyhedral-lifted-mesh",
    )
    periodic_entry = CompositionEntry(
        periodic,
        entry_id="poly/periodicity",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=lifted.topology_id,
        revision_id=periodic.periodic_topology_id,
        semantics_id="periodic-polyhedral-quotient",
        dependencies=(
            lifted_entry.binding("structure"),
            lifted_entry.binding("revision"),
        ),
    )
    mesh_entry = CompositionEntry(
        mesh,
        entry_id="poly/periodic-mesh",
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=mesh.topology_id,
        revision_id=mesh.mesh_id,
        semantics_id="periodic-polyhedral-cell-mesh",
        dependencies=(
            lifted_entry.binding("structure"),
            lifted_entry.binding("revision"),
            periodic_entry.binding("revision"),
        ),
    )
    composition = Composition(
        (lifted_entry, periodic_entry, mesh_entry),
        boundary_id="periodic-polyhedral:archive",
    )
    write_model_artifact(directory / "periodic-polyhedral.phx", composition, limits=None)
    wrapped, image_shifts = periodic_cell.wrap(
        np.asarray((2.25, -0.25, 1.5), dtype=np.float64)
    )
    print(
        json.dumps(
            {
                "composition_id": composition.composition_id,
                "lifted_mesh_id": lifted.mesh_id,
                "lifted_topology_id": lifted.topology_id,
                "periodic_topology_id": periodic.periodic_topology_id,
                "periodic_mesh_id": mesh.mesh_id,
                "periodic_mesh_topology_id": mesh.topology_id,
                "block_arities": [block.arity for block in mesh.blocks],
                "face_arities": np.diff(
                    np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
                ).tolist(),
                "cell_vertex_arities": np.diff(
                    np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
                ).tolist(),
                "orbit_corner_offsets": np.asarray(
                    orbit.corner_permutation_offsets, dtype=np.int64
                ).tolist(),
                "orbit_corner_permutations": np.asarray(
                    orbit.corner_permutations, dtype=np.int32
                ).tolist(),
                "wrapped_point": np.asarray(wrapped).tolist(),
                "image_shifts": np.asarray(image_shifts).tolist(),
            }
        )
    )


def _read_polyhedral_periodic_archive(path: Path, expected: dict[str, object], /) -> None:
    from phydrax.discretization._cell_complex import PolyhedralConnectivity
    from phydrax.meshing._periodic import periodic_entity_orbit_map

    composition = read_model_artifact(
        path,
        expected_type=Composition,
        restore=_rebuild_polyhedral_periodic_composition,
        limits=None,
    )
    lifted = composition.value("poly/lifted-mesh")
    periodic = composition.value("poly/periodicity")
    mesh = composition.value("poly/periodic-mesh")
    if (
        not isinstance(lifted, CellMesh)
        or not isinstance(mesh, CellMesh)
        or not isinstance(periodic, PeriodicMeshTopology)
    ):
        raise TypeError("Cold polyhedral periodic archive lost its typed mesh owners.")
    lifted_connectivity = lifted.connectivity
    mesh_connectivity = mesh.connectivity
    if not isinstance(lifted_connectivity, PolyhedralConnectivity) or not isinstance(
        mesh_connectivity, PolyhedralConnectivity
    ):
        raise TypeError(
            "Cold polyhedral periodic archive lost its polyhedral connectivity."
        )
    mesh_periodic = mesh.periodic_topology
    if mesh_periodic is None:
        raise TypeError("Cold polyhedral periodic mesh lost its quotient binding.")
    periodic_cell = periodic.cell
    if not isinstance(periodic_cell, PeriodicCell):
        raise TypeError(
            "Cold polyhedral periodic archive requires a periodic lattice cell."
        )
    orbit = periodic_entity_orbit_map(mesh, 3)
    wrapped, image_shifts = periodic_cell.wrap(
        np.asarray((2.25, -0.25, 1.5), dtype=np.float64)
    )
    if (
        composition.composition_id != expected["composition_id"]
        or lifted.mesh_id != expected["lifted_mesh_id"]
        or lifted.topology_id != expected["lifted_topology_id"]
        or periodic.periodic_topology_id != expected["periodic_topology_id"]
        or mesh.mesh_id != expected["periodic_mesh_id"]
        or mesh.topology_id != expected["periodic_mesh_topology_id"]
        or mesh_periodic.periodic_topology_id != periodic.periodic_topology_id
        or [block.arity for block in mesh.blocks] != expected["block_arities"]
        or np.diff(
            np.asarray(mesh_connectivity.face_vertex_offsets, dtype=np.int64)
        ).tolist()
        != expected["face_arities"]
        or np.diff(
            np.asarray(mesh_connectivity.cell_vertex_offsets, dtype=np.int64)
        ).tolist()
        != expected["cell_vertex_arities"]
        or np.asarray(orbit.corner_permutation_offsets, dtype=np.int64).tolist()
        != expected["orbit_corner_offsets"]
        or np.asarray(orbit.corner_permutations, dtype=np.int32).tolist()
        != expected["orbit_corner_permutations"]
        or not np.array_equal(
            np.asarray(wrapped), _expected_array(expected["wrapped_point"])
        )
        or not np.array_equal(
            np.asarray(image_shifts), _expected_array(expected["image_shifts"])
        )
    ):
        raise ValueError(
            "Cold polyhedral periodic topology or derived orbit maps changed."
        )
    print("polyhedral-periodic-cold-restored")


if __name__ == "__main__":
    operation = sys.argv[1]
    archive_directory = Path(sys.argv[2])
    if operation == "write":
        _write(archive_directory)
    elif operation == "read":
        _read(archive_directory, json.loads(sys.argv[3]))
    elif operation == "read-continuation":
        _read_continuation(archive_directory, json.loads(sys.argv[3]))
    elif operation == "read-implicit":
        _read_adaptive_implicit(
            archive_directory / "implicit-domain.phx", json.loads(sys.argv[3])
        )
    elif operation == "write-implicit":
        _write_adaptive_implicit(archive_directory)
    elif operation == "read-reconstructed":
        _read_reconstructed_source(
            archive_directory / "reconstructed-source.phx", json.loads(sys.argv[3])
        )
    elif operation == "write-reconstructed":
        _write_reconstructed_source(archive_directory)
    elif operation == "read-pwl-boundary-query":
        # The per-entity oracle exceeds the process argument limit; it is a file.
        _read_piecewise_boundary_query(
            archive_directory / "pwl-boundary-query.phx",
            json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")),
        )
    elif operation == "write-pwl-boundary-query":
        _write_piecewise_boundary_query(archive_directory)
    elif operation == "read-curve-query":
        _read_parametric_curve_query(
            archive_directory / "parametric-curve-query.phx", json.loads(sys.argv[3])
        )
    elif operation == "write-curve-query":
        _write_parametric_curve_query(archive_directory)
    elif operation == "write-mapped-reference-domain":
        _write_mapped_reference_domain_artifact(archive_directory)
    elif operation == "read-mapped-reference-domain":
        _read_mapped_reference_domain_artifact(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-piecewise-region-occurrences":
        _write_piecewise_region_occurrence_archive(archive_directory)
    elif operation == "read-piecewise-region-occurrences":
        _read_piecewise_region_occurrence_archive(
            archive_directory, json.loads(sys.argv[3])
        )
    elif operation == "write-mapped-reference-boundary-query":
        _write_mapped_reference_boundary_query(archive_directory)
    elif operation == "read-mapped-reference-boundary-query":
        _read_mapped_reference_boundary_query(
            archive_directory / "mapped-reference-boundary-query.phx",
            json.loads(sys.argv[3]),
        )
    elif operation == "write-coordinate-maps":
        _write_coordinate_map_artifact(archive_directory)
    elif operation == "read-coordinate-maps":
        _read_coordinate_map_artifact(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-periodic-power-map":
        _write_periodic_power_map_archive(archive_directory)
    elif operation == "read-periodic-power-map":
        _read_periodic_power_map_archive(archive_directory, json.loads(sys.argv[3]))
    elif operation == "read-periodic-power-map-without-mesh":
        _read_periodic_power_map_without_mesh(archive_directory)
    elif operation == "write-owner-local-remap":
        _write_owner_local_remap_archive(archive_directory)
    elif operation == "read-owner-local-remap":
        _read_owner_local_remap_archive(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-owner-local-periodic-adaptive":
        _write_owner_local_periodic_adaptive_archive(archive_directory)
    elif operation == "read-owner-local-periodic-adaptive":
        _read_owner_local_periodic_adaptive_archive(
            archive_directory, json.loads(sys.argv[3])
        )
    elif operation == "write-meshing-specs":
        _write_meshing_spec_archive(archive_directory)
    elif operation == "read-meshing-specs":
        _read_meshing_spec_archive(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-analytic-profile":
        _write_analytic_profile_artifact(archive_directory)
    elif operation == "read-analytic-profile":
        _read_analytic_profile_artifact(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-certification":
        _write_certification_artifact(archive_directory)
    elif operation == "read-certification":
        _read_certification_artifact(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-source-coverage":
        _write_source_coverage_artifact(archive_directory)
    elif operation == "read-source-coverage":
        _read_source_coverage_artifact(archive_directory, json.loads(sys.argv[3]))
    elif operation == "read-cad-chart-witness":
        _read_cad_chart_witness(archive_directory, sys.argv[3])
    elif operation == "write-region-boundary-evidence":
        _write_region_boundary_archive(archive_directory)
    elif operation == "read-region-boundary-evidence":
        _read_region_boundary_archive(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-source-chart-rational":
        _write_source_chart_rational_artifact(archive_directory)
    elif operation == "read-source-chart-rational":
        _read_source_chart_rational_artifact(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-brep-entity":
        _write_brep_entity_composition(archive_directory)
    elif operation == "read-brep-entity":
        _read_brep_entity_composition(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-brep-assembly-placement-evidence":
        _write_brep_assembly_placement_evidence_archive(archive_directory)
    elif operation == "read-brep-assembly-placement-evidence":
        _read_brep_assembly_placement_evidence_archive(
            archive_directory, json.loads(sys.argv[3])
        )
    elif operation == "write-brep-projection":
        _write_brep_projection_composition(archive_directory)
    elif operation == "read-brep-projection":
        _read_brep_projection_composition(
            archive_directory / "brep-projection.phx", json.loads(sys.argv[3])
        )
    elif operation == "write-reference-chart-arguments":
        _write_reference_chart_arguments(archive_directory)
    elif operation == "read-reference-chart-arguments":
        _read_reference_chart_arguments(
            archive_directory / "reference-chart-arguments.phx", json.loads(sys.argv[3])
        )
    elif operation == "write-tspline-cold-archive":
        _write_tspline_cold_archive(archive_directory)
    elif operation == "read-tspline-cold-archive":
        _read_tspline_cold_archive(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-plane-cylinder-section-pcurve":
        _write_plane_cylinder_section_pcurve(archive_directory)
    elif operation == "read-plane-cylinder-section-pcurve":
        _read_plane_cylinder_section_pcurve(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-brep-curve-surface-lift":
        _write_brep_curve_surface_lift(archive_directory)
    elif operation == "read-brep-curve-surface-lift":
        _read_brep_curve_surface_lift(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-sphere-chamfer-conic-root":
        _write_sphere_chamfer_conic_root_archive(archive_directory)
    elif operation == "read-sphere-chamfer-conic-root":
        _read_sphere_chamfer_conic_root_archive(
            archive_directory, json.loads(sys.argv[3])
        )
    elif operation == "write-parallel-cylinder-chamfer":
        _write_parallel_cylinder_chamfer_archive(archive_directory)
    elif operation == "read-parallel-cylinder-chamfer":
        _read_parallel_cylinder_chamfer_archive(
            archive_directory, json.loads(sys.argv[3])
        )
    elif operation == "write-orthogonal-quadric-boolean":
        _write_orthogonal_quadric_boolean_archive(archive_directory)
    elif operation == "read-orthogonal-quadric-boolean":
        _read_orthogonal_quadric_boolean_archive(
            archive_directory, json.loads(sys.argv[3])
        )
    elif operation == "write-torus-exact-source":
        _write_torus_exact_source_archive(archive_directory)
    elif operation == "read-torus-exact-source":
        _read_torus_exact_source_archive(archive_directory, json.loads(sys.argv[3]))
    elif operation == "read-surface-witness":
        _read_surface_chart_witness(archive_directory, json.loads(sys.argv[3]))
        print("surface-chart-witness-cold-restored")
    elif operation == "write-surface-chart-query":
        _write_surface_chart_boundary_query(archive_directory)
    elif operation == "read-surface-chart-query":
        _read_surface_chart_boundary_query(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-layer-curving-evidence":
        _write_layer_curving_evidence(archive_directory)
    elif operation == "write-mapped-layer-curving-evidence":
        _write_mapped_layer_curving_evidence(archive_directory)
    elif operation == "read-layer-curving-evidence":
        _read_layer_curving_evidence(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-restricted-layer-curving-evidence":
        _write_restricted_layer_curving_evidence(archive_directory)
    elif operation == "read-restricted-layer-curving-evidence":
        _read_restricted_layer_curving_evidence(
            archive_directory, json.loads(sys.argv[3])
        )
    elif operation == "write-implicit-layer-curving-evidence":
        _write_implicit_layer_curving_evidence(archive_directory)
    elif operation == "read-implicit-layer-curving-evidence":
        _read_layer_curving_evidence(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-implicit-chart-restriction":
        _write_implicit_chart_restriction(archive_directory)
    elif operation == "read-implicit-chart-restriction":
        _read_implicit_chart_restriction(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-exact-tensor-quad-fidelity":
        _write_exact_tensor_quad_fidelity(archive_directory)
    elif operation == "read-exact-tensor-quad-fidelity":
        _read_exact_tensor_quad_fidelity(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-material-reference-seed-transition":
        _write_material_reference_seed_transition(archive_directory)
    elif operation == "read-material-reference-seed-transition":
        _read_material_reference_seed_transition(
            archive_directory, json.loads(sys.argv[3])
        )
    elif operation == "write-polyhedral-periodic":
        _write_polyhedral_periodic_archive(archive_directory)
    elif operation == "read-polyhedral-periodic":
        _read_polyhedral_periodic_archive(archive_directory, json.loads(sys.argv[3]))
    elif operation == "write-finite-element-transfer":
        _write_finite_element_transfer_archive(archive_directory)
    elif operation == "read-finite-element-transfer":
        _read_finite_element_transfer_archive(
            archive_directory / "finite-element-transfer.phx",
            json.loads(sys.argv[3]),
        )
    elif operation == "write-vertex-stencil":
        _write_vertex_interpolation_stencil_archive(archive_directory)
    elif operation == "read-vertex-stencil":
        _read_vertex_interpolation_stencil_archive(
            archive_directory / "vertex-interpolation-stencil.phx",
            json.loads(sys.argv[3]),
        )
    elif operation == "write-periodic-distributed-fe-transfer":
        _write_periodic_distributed_fe_transfer_archive(archive_directory)
    elif operation == "read-periodic-distributed-fe-transfer":
        _read_periodic_distributed_fe_transfer_archive(
            archive_directory, json.loads(sys.argv[3])
        )
    elif operation == "write-revolution-offset":
        _write_revolution_offset_archive(archive_directory)
    elif operation == "read-revolution-offset":
        _read_revolution_offset_archive(archive_directory, json.loads(sys.argv[3]))
    elif operation == "read-brep":
        load_brep_archive(archive_directory)
    else:
        raise ValueError(
            "operation must be write, read, read-continuation, read-implicit, "
            "write-implicit, read-reconstructed, write-reconstructed, read-curve-query, "
            "write-curve-query, write-certification, read-certification, "
            "write-coordinate-maps, read-coordinate-maps, "
            "write-periodic-power-map, read-periodic-power-map, "
            "read-periodic-power-map-without-mesh, "
            "write-owner-local-remap, read-owner-local-remap, "
            "write-owner-local-periodic-adaptive, read-owner-local-periodic-adaptive, "
            "write-meshing-specs, read-meshing-specs, "
            "write-mapped-reference-domain, read-mapped-reference-domain, "
            "write-piecewise-region-occurrences, read-piecewise-region-occurrences, "
            "write-mapped-reference-boundary-query, read-mapped-reference-boundary-query, "
            "write-brep-curve-surface-lift, read-brep-curve-surface-lift, "
            "write-sphere-chamfer-conic-root, read-sphere-chamfer-conic-root, "
            "write-parallel-cylinder-chamfer, read-parallel-cylinder-chamfer, "
            "write-orthogonal-quadric-boolean, read-orthogonal-quadric-boolean, "
            "write-torus-exact-source, read-torus-exact-source, "
            "write-analytic-profile, read-analytic-profile, "
            "write-source-coverage, read-source-coverage, "
            "write-region-boundary-evidence, read-region-boundary-evidence, "
            "write-source-chart-rational, read-source-chart-rational, "
            "write-brep-entity, read-brep-entity, "
            "write-brep-assembly-placement-evidence, "
            "read-brep-assembly-placement-evidence, "
            "write-reference-chart-arguments, read-reference-chart-arguments, "
            "write-brep-projection, read-brep-projection, "
            "write-tspline-cold-archive, read-tspline-cold-archive, "
            "read-surface-witness, write-surface-chart-query, "
            "read-surface-chart-query, write-layer-curving-evidence, "
            "write-mapped-layer-curving-evidence, "
            "read-layer-curving-evidence, write-restricted-layer-curving-evidence, "
            "read-restricted-layer-curving-evidence, write-implicit-layer-curving-evidence, "
            "read-implicit-layer-curving-evidence, write-material-reference-seed-transition, "
            "read-material-reference-seed-transition, write-polyhedral-periodic, "
            "read-polyhedral-periodic, write-finite-element-transfer, "
            "write-vertex-stencil, read-vertex-stencil, "
            "write-periodic-distributed-fe-transfer, "
            "read-periodic-distributed-fe-transfer, read-revolution-offset, "
            "write-revolution-offset, read-finite-element-transfer, or read-brep."
        )
