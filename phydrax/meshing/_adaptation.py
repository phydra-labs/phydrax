#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed mesh-adaptation requests, explicit routes, and certified transitions.

`prepare_mesh_adaptation` binds one request to one certified source result under
one explicit `MeshAdaptationRoute` and resolves protection and organization into
route constraints. `execute_mesh_adaptation` runs exactly that route (there is no
automatic provider fallback) and returns the certified target with its complete
lineage, available nodal transfer or geometric common refinement, route evidence,
and compliance.
"""

from __future__ import annotations

import dataclasses
import time
from enum import StrEnum
from fractions import Fraction
from typing import Any, assert_never, final, NamedTuple, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike, NDArray

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import PredicateMode, resolve_host_predicate_mode
from .._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionBudget
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._adaptive_simplex import AdaptiveSimplexPolicy
from ..discretization._cell_geometry import (
    _require_scalar_coordinate_element,
    CellGeometryElement,
)
from ..discretization._cell_geometry_transfer import (
    cell_geometry_vertex_measures,
    CellGeometryTransition,
    CellGeometryTransitionPolicy,
    is_affine_cell_geometry,
    nested_geometry_degree,
    NestedReferenceWitnesses,
    SourceGeometryRealization,
    transition_nested_cell_geometry,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
)
from ..discretization._coordinate_enclosure import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
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
from ..discretization._surface_chart_deformation import SurfaceChartResourceError
from ..discretization.fem import (
    balanced_hp_refinement_ids,
    coarsen_tensor_hp_cells,
    finite_element_hp_interface_plan,
    FiniteElementHPEpoch,
    FiniteElementHPLineage,
    FiniteElementHPRefinementResult,
    FiniteElementTopologyTransfer,
    hp_active_cell_mesh,
    refine_tensor_hp_cells,
)
from ..geometry._mesh_certificates import (
    GlobalEmbeddingCertificate,
    MeshCertificateBinding,
    MeshCertificateLimits,
    SourceFidelityCertificate,
)
from ..geometry._supermesh import CommonRefinementPolicy, PreparedCommonRefinement
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ..typing import checked
from ._assembly import MeshPart
from ._association import (
    BRepAssociationTransfer,
    GeometryAssociation,
    MappedReferenceAssociationTransfer,
    PlcAssociationTransfer,
    SurfaceAssociationTransfer,
)
from ._association_composition import ComposedAssociationTransfer
from ._audit import _required_audit_policy, CellMeshAuditPolicy
from ._bisection import (
    BisectionCompatibility,
    BisectionEvidence,
    BisectionHierarchy,
    BisectionUniformRefinement,
    execute_bisection,
)
from ._canonical import canonicalize_cell_mesh, certify_cell_mesh
from ._certification_inputs import MeshCertificationInputs
from ._contracts import MeshingFailure, MeshingFailureCategory, MeshingLimits
from ._curving import (
    _straight_geometry,
    CurvedGeometryEvidence,
    HighOrderCurvingResult,
    HighOrderCurvingStatus,
)
from ._device_metric import DeviceMetricEvidence
from ._distribution import (
    MeshDistribution,
    MeshDistributionTransition,
    MeshPartitionPolicy,
    prepare_distribution_transition,
)
from ._implicit_association_transfer import ImplicitAssociationTransfer
from ._level_set import LevelSetEvidence, LevelSetMeshAdaptation
from ._lineage import (
    CellMeshTransition,
    EntityLineage,
    EntityLineageKind,
    inherit_mesh_organization,
    inherit_scope,
    MeshLineage,
    MeshTransitionKind,
    VertexInterpolationStencil,
)
from ._local_metric import execute_local_metric_adaptation, LocalMetricEvidence
from ._measurements import NativeExecutionRecord
from ._metric import MeshMetricField
from ._mixed_adaptation import (
    adapt_mixed_mesh,
    MixedAdaptationEvidence,
    MixedAdaptationHierarchy,
    MixedAdaptationOutcome,
    MixedLayerColumns,
)
from ._organization import (
    MeshAttribute,
    MeshLabel,
    MeshPatch,
    MeshZone,
    RegionBoundaryEvidence,
)
from ._periodic import PeriodicRefinement
from ._polyhedral_adaptation import (
    adapt_polyhedral_mesh,
    PolyhedralAdaptationEvidence,
    PolyhedralAdaptationOperation,
    PolyhedralMeshAdaptation,
    regenerated_polyhedral_associations,
    regenerated_polyhedral_relations,
)
from ._polyhedral_generation import PolyhedralConstruction
from ._result import CellMeshingResult, CollectiveMeshEvidence, MeshingComplianceReport
from ._scope import MeshingEntityKind, MeshingScope, resolve_mesh_scope
from ._surface_metric import SphereMetricOutcome, SurfaceMetricOutcome
from ._tetra_metric import (
    execute_tetra_metric_adaptation,
    MetricRemeshingCriterion,
    MetricRemeshingEvidence,
    MetricRemeshingStatus,
)
from ._topology_edit import (
    assemble_topology_edit,
    CellTopologyEdit,
    entity_keys,
    key_rows,
)
from ._trace import MeshingStageKind
from .providers._mmg import MmgAdaptationResult, MmgOptions, MmgProvider
from .providers._native_periodic import PeriodicAssociationTransfer
from .providers._omega_h import OmegaHAdaptationResult, OmegaHOptions, OmegaHProvider


if TYPE_CHECKING:
    from ..discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        PolynomialComposedCellGeometryElement,
    )
    from ..discretization._coordinate_enclosure import CoordinateSourceBank
    from ..discretization._surface_chart_deformation import SurfaceChartWitness
    from ..discretization.fem._reference import FiniteElementSpec
    from ._surface_association_transfer import PreparedSurfaceCurveWitness


type _FullP1Element = (
    FiniteElementSpec
    | BarycentricCellGeometryElement
    | PolynomialComposedCellGeometryElement
)
type _FullP1OriginalCell = tuple[
    str | None,
    _FullP1Element,
    NDArray[np.int32],
    NDArray[np.int64],
    int,
]
type _FullP1SourceCell = tuple[
    _FullP1Element,
    NDArray[np.int64],
    NDArray[np.int32],
    int,
    NDArray[np.int64],
    str | None,
    int,
    _FullP1Element,
]
type _UniformRationalActions = dict[int, tuple[int, tuple[tuple[Fraction, ...], ...]]]


class MeshAdaptationStatus(StrEnum):
    """Outcome of one executed adaptation.

    ``COMPLETE``: every requested operation was realized (for metric routes the
    unit-mesh criterion holds). ``PARTIAL``: some requested marks were rejected
    (protected entities, ineligible coarsening families); the evidence lists them.
    ``PASS_LIMIT``: a metric route stopped at ``maximum_passes`` before the
    unit-mesh criterion. ``STALLED``: no admissible operation remains while the
    criterion is unmet. ``RESOURCE_LIMIT``: native execution exhausted a hard
    resource budget. ``UNCHANGED``: nothing was applied; the target is the
    source result itself.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    PASS_LIMIT = "pass_limit"
    STALLED = "stalled"
    RESOURCE_LIMIT = "resource_limit"
    UNCHANGED = "unchanged"

    @property
    def converged(self) -> bool:
        """Whether the request is fully realized (COMPLETE or UNCHANGED)."""
        return self in (MeshAdaptationStatus.COMPLETE, MeshAdaptationStatus.UNCHANGED)


class MeshAdaptationRoute(StrEnum):
    """Explicit execution route; each request kind admits only its own routes.

    ``NATIVE_BISECTION``: marked Maubach/newest-vertex bisection and coarsening of
    simplex meshes. ``NATIVE_METRIC_2D``: planar triangle metric adaptation (or
    relocation-only r-adaptation). ``NATIVE_METRIC_3D``: affine tetrahedral metric
    remeshing. ``NATIVE_MIXED``: family-preserving template refinement and sibling
    coarsening. ``NATIVE_LEVEL_SET``: existing-tetrahedron P1 zero-set insertion.
    ``NATIVE_POLYHEDRAL``: planar packed-cell subdivision, connected agglomeration,
    or explicit site regeneration with certified geometric common refinement.
    ``DEVICE_BISECTION`` and ``DEVICE_METRIC_2D``:
    the same requests executed as compiled fixed-capacity device passes (see
    `prepare_adaptive_simplex`); device bisection commits the meshes of the host
    bisection route byte for byte. ``MMG`` and ``OMEGA_H``: metric adaptation by
    the configured provider. ``HP``: marked tensor-product h-refinement or
    coarsening of one prepared `FiniteElementHPEpoch`.
    """

    NATIVE_BISECTION = "native_bisection"
    NATIVE_METRIC_2D = "native_metric_2d"
    NATIVE_METRIC_3D = "native_metric_3d"
    NATIVE_SURFACE_METRIC = "native_surface_metric"
    NATIVE_LEVEL_SET = "native_level_set"
    NATIVE_MIXED = "native_mixed"
    NATIVE_POLYHEDRAL = "native_polyhedral"
    NATIVE_GEOMETRY_REALIZATION = "native_geometry_realization"
    DEVICE_BISECTION = "device_bisection"
    DEVICE_METRIC_2D = "device_metric_2d"
    MMG = "mmg"
    OMEGA_H = "omega_h"
    HP = "hp"


_DEVICE_ROUTES = frozenset(
    (MeshAdaptationRoute.DEVICE_BISECTION, MeshAdaptationRoute.DEVICE_METRIC_2D)
)

type MeshAdaptationHierarchy = (
    BisectionHierarchy
    | FiniteElementHPEpoch
    | MixedAdaptationHierarchy
    | PeriodicRefinement
    | None
)


def _identifier_vector(values: ArrayLike, name: str, /) -> np.ndarray:
    identifiers = np.asarray(values)
    if identifiers.size == 0:
        return np.empty((0,), dtype=np.int64)
    if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
        raise TypeError(f"{name} must be one integer vector.")
    identifiers = identifiers.astype(np.int64, copy=False)
    if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
        raise ValueError(f"{name} must be unique and non-negative.")
    return np.sort(identifiers)


def _flag(value: bool, name: str, /) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool.")
    return bool(value)


def _count(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


@final
class MarkedMeshAdaptation(StrictModule, NonTrainableState):
    """Source cells marked for refinement and for coarsening, by global ID.

    ``hierarchy`` carries the state of earlier adaptations: the
    `BisectionHierarchy` returned by a previous NATIVE_BISECTION result (required
    for coarsening, and to continue its refinement labels), the source
    `FiniteElementHPEpoch` of the HP route, or `MixedAdaptationHierarchy` for
    NATIVE_MIXED template siblings. ``layer_columns`` carries explicit layer
    ancestry and axial-refinement permissions only for NATIVE_MIXED.
    """

    refine_cell_ids: Array
    coarsen_cell_ids: Array
    hierarchy: MeshAdaptationHierarchy
    layer_columns: MixedLayerColumns | None
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        refine_cell_ids: ArrayLike = (),
        coarsen_cell_ids: ArrayLike = (),
        /,
        *,
        hierarchy: MeshAdaptationHierarchy = None,
        layer_columns: MixedLayerColumns | None = None,
    ) -> None:
        refine = _identifier_vector(refine_cell_ids, "refine_cell_ids")
        coarsen = _identifier_vector(coarsen_cell_ids, "coarsen_cell_ids")
        if np.intersect1d(refine, coarsen).size:
            raise ValueError("A cell cannot be marked for refinement and coarsening.")
        if hierarchy is not None and not isinstance(
            hierarchy,
            (
                BisectionHierarchy,
                FiniteElementHPEpoch,
                MixedAdaptationHierarchy,
                PeriodicRefinement,
            ),
        ):
            raise TypeError(
                "hierarchy must be BisectionHierarchy, FiniteElementHPEpoch, "
                "MixedAdaptationHierarchy, PeriodicRefinement, or None."
            )
        if layer_columns is not None and not isinstance(layer_columns, MixedLayerColumns):
            raise TypeError("layer_columns must be MixedLayerColumns or None.")
        self.refine_cell_ids = jnp.asarray(refine)
        self.coarsen_cell_ids = jnp.asarray(coarsen)
        self.hierarchy = hierarchy
        self.layer_columns = layer_columns
        self.request_id = canonical_fingerprint(
            {
                "kind": "marked-mesh-adaptation",
                "refine": array_tree_fingerprint(refine),
                "coarsen": array_tree_fingerprint(coarsen),
                "hierarchy": _hierarchy_id(hierarchy),
                "layer_columns": None
                if layer_columns is None
                else layer_columns.columns_id,
            }
        )


@final
class MetricMeshAdaptation(StrictModule, NonTrainableState):
    """Adapt toward a unit mesh of one vertex metric bound to the source vertices."""

    metric: MeshMetricField
    coordinate_contract: SpatialCoordinateContract | None
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        metric: MeshMetricField,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract | None = None,
    ) -> None:
        if coordinate_contract is not None and not isinstance(
            coordinate_contract, SpatialCoordinateContract
        ):
            raise TypeError(
                "coordinate_contract must be SpatialCoordinateContract or None."
            )
        self.metric = metric
        self.coordinate_contract = coordinate_contract
        self.request_id = canonical_fingerprint(
            {
                "kind": "metric-mesh-adaptation",
                "metric": metric.metric_id,
                "coordinate_contract": None
                if coordinate_contract is None
                else coordinate_contract.spatial_id,
            }
        )


@final
class RelocationMeshAdaptation(StrictModule, NonTrainableState):
    """Move vertices toward one vertex metric without changing the topology."""

    metric: MeshMetricField
    coordinate_contract: SpatialCoordinateContract | None
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        metric: MeshMetricField,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract | None = None,
    ) -> None:
        if coordinate_contract is not None and not isinstance(
            coordinate_contract, SpatialCoordinateContract
        ):
            raise TypeError(
                "coordinate_contract must be SpatialCoordinateContract or None."
            )
        self.metric = metric
        self.coordinate_contract = coordinate_contract
        self.request_id = canonical_fingerprint(
            {
                "kind": "relocation-mesh-adaptation",
                "metric": metric.metric_id,
                "coordinate_contract": None
                if coordinate_contract is None
                else coordinate_contract.spatial_id,
            }
        )


@final
class GeometryRealizationMeshAdaptation(StrictModule, NonTrainableState):
    """Trusted source-realized coordinate-order candidate on a declared material complex.

    Retained topology does not imply retained physical support. The actual
    curving certificates and quantitative old/new measure correspondence are
    required separately from the prior source-fidelity certificate.
    """

    source: CellMeshingResult
    realization: HighOrderCurvingResult
    source_fidelity: SourceFidelityCertificate
    geometry_realization: SourceGeometryRealization
    certificate_limits: MeshCertificateLimits
    source_result_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: CellMeshingResult,
        realization: HighOrderCurvingResult,
        source_fidelity: SourceFidelityCertificate,
        geometry_realization: SourceGeometryRealization,
        certificate_limits: MeshCertificateLimits,
        /,
    ) -> None:
        geometry_realization.require_current()
        if (
            realization.status is not HighOrderCurvingStatus.CURVED
            or not realization.evidence.accepted
        ):
            raise ValueError(
                "Geometry realization requires an actually accepted curving product."
            )
        target_fidelity = realization.evidence.fidelity
        if (
            target_fidelity is None
            or target_fidelity.status != "certified"
            or source_fidelity.status != "certified"
        ):
            raise ValueError(
                "Both coordinate epochs require continuous source-fidelity certificates."
            )
        source_fidelity.binding.require(source.mesh, source.geometry)
        target_fidelity.binding.require(source.mesh, realization.geometry)
        if (
            source_fidelity.binding.source_id,
            source_fidelity.binding.source_revision,
        ) != (target_fidelity.binding.source_id, target_fidelity.binding.source_revision):
            raise ValueError(
                "Coordinate-order realization must retain the original scientific source."
            )
        for certificate in (
            source_fidelity,
            target_fidelity,
            geometry_realization.source_embedding,
            geometry_realization.target_embedding,
        ):
            if certificate.binding.limits_id != certificate_limits.limits_id:
                raise ValueError(
                    "Realization must retain the actual budgets of every source certificate."
                )
        geometry_realization.source_embedding.binding.require(
            source.mesh, source.geometry
        )
        geometry_realization.target_embedding.binding.require(
            source.mesh, realization.geometry
        )
        transition = geometry_realization.transition
        if (
            transition.evidence.kind != "source_realization"
            or transition.source_topology_id != source.mesh.topology_id
            or transition.target_topology_id != source.mesh.topology_id
            or transition.source_geometry_id != cell_geometry_id(source.geometry)
            or transition.target_geometry_id != cell_geometry_id(realization.geometry)
        ):
            raise ValueError(
                "The material correspondence must bind the actual old and realized maps."
            )
        old_elements = source.geometry.resolve(source.mesh)[0]
        new_elements = realization.geometry.resolve(source.mesh)[0]
        old_orders = tuple(
            _require_scalar_coordinate_element(value, "Source realization").degree
            for value in old_elements
        )
        new_orders = tuple(
            _require_scalar_coordinate_element(value, "Source realization").degree
            for value in new_elements
        )
        if any(
            new < old for old, new in zip(old_orders, new_orders, strict=True)
        ) or not any(new > old for old, new in zip(old_orders, new_orders, strict=True)):
            raise ValueError(
                "Geometry-order realization must actually increase coordinate order."
            )
        self.source = source
        self.realization = realization
        self.source_fidelity = source_fidelity
        self.geometry_realization = geometry_realization
        self.certificate_limits = certificate_limits
        self.source_result_id = source.result_id
        self.request_id = canonical_fingerprint(
            {
                "kind": "geometry-realization-mesh-adaptation",
                "source": source.result_id,
                "realization": realization.result_id,
                "old_fidelity": source_fidelity.certificate_id,
                "correspondence": geometry_realization.realization_id,
            }
        )


MeshAdaptationRequest: TypeAlias = (
    MarkedMeshAdaptation
    | MetricMeshAdaptation
    | RelocationMeshAdaptation
    | LevelSetMeshAdaptation
    | PolyhedralMeshAdaptation
    | GeometryRealizationMeshAdaptation
)


def _hierarchy_id(
    hierarchy: MeshAdaptationHierarchy,
    /,
) -> str | None:
    match hierarchy:
        case None:
            return None
        case BisectionHierarchy() | MixedAdaptationHierarchy():
            return hierarchy.hierarchy_id
        case FiniteElementHPEpoch():
            return hierarchy.epoch_id
        case PeriodicRefinement():
            return hierarchy.refinement_id
        case invalid:
            assert_never(invalid)


def _provider_options_record(options: MmgOptions | OmegaHOptions | None, /) -> Any:
    match options:
        case None:
            return None
        case MmgOptions():
            return {"mmg": dataclasses.asdict(options)}
        case OmegaHOptions():
            return {"omega_h": options.worker_record()}
        case _:
            raise TypeError("provider_options must be MmgOptions or OmegaHOptions.")


def _check_provider(
    route: MeshAdaptationRoute,
    provider: MmgProvider | OmegaHProvider | None,
    options: MmgOptions | OmegaHOptions | None,
    protected: tuple[MeshingScope, ...],
    distribution: MeshDistribution | None,
    /,
) -> None:
    match route:
        case MeshAdaptationRoute.MMG:
            if not isinstance(provider, MmgProvider) or not isinstance(
                options, (MmgOptions, type(None))
            ):
                raise TypeError("The MMG route requires an MmgProvider and MmgOptions.")
        case MeshAdaptationRoute.OMEGA_H:
            if not isinstance(provider, OmegaHProvider) or not isinstance(
                options, (OmegaHOptions, type(None))
            ):
                raise TypeError(
                    "The OMEGA_H route requires an OmegaHProvider and OmegaHOptions."
                )
            if protected:
                raise ValueError("Omega_h adaptation cannot pin protected scopes.")
        case (
            MeshAdaptationRoute.NATIVE_BISECTION
            | MeshAdaptationRoute.NATIVE_METRIC_2D
            | MeshAdaptationRoute.NATIVE_METRIC_3D
            | MeshAdaptationRoute.NATIVE_SURFACE_METRIC
            | MeshAdaptationRoute.NATIVE_GEOMETRY_REALIZATION
            | MeshAdaptationRoute.NATIVE_LEVEL_SET
            | MeshAdaptationRoute.NATIVE_MIXED
            | MeshAdaptationRoute.DEVICE_BISECTION
            | MeshAdaptationRoute.NATIVE_POLYHEDRAL
            | MeshAdaptationRoute.DEVICE_METRIC_2D
            | MeshAdaptationRoute.HP
        ):
            if provider is not None or options is not None:
                raise ValueError("Native routes take no provider or provider options.")
        case invalid:
            assert_never(invalid)
    if distribution is not None and route in (
        MeshAdaptationRoute.MMG,
        MeshAdaptationRoute.OMEGA_H,
    ):
        raise ValueError(
            "Provider remeshing has unknown lineage; it cannot carry a distribution."
        )


def _resolved_predicate_mode(
    route: MeshAdaptationRoute, mode: PredicateMode | None, /
) -> PredicateMode:
    """EXACT for host routes and FILTERED_DEVICE for device routes unless given."""

    device = route in _DEVICE_ROUTES
    if mode is None:
        return PredicateMode.FILTERED_DEVICE if device else PredicateMode.EXACT
    if not isinstance(mode, PredicateMode):
        raise TypeError("predicate_mode must be PredicateMode or None.")
    if route is MeshAdaptationRoute.NATIVE_METRIC_3D and mode is not PredicateMode.EXACT:
        raise ValueError("NATIVE_METRIC_3D implements exact native predicates only.")
    if device != (mode is PredicateMode.FILTERED_DEVICE):
        raise ValueError(
            "Device routes evaluate FILTERED_DEVICE predicates; host routes require "
            "FILTERED or EXACT predicates."
        )
    return mode


@final
class MeshAdaptationPolicy(StrictModule, NonTrainableState):
    """Route, protection, and execution controls of one adaptation.

    ``route`` is explicit and never replaced by another route. ``protected_scopes``
    (bound like result organization: source mesh ID and numeric version) are never
    split, removed, flipped, or moved; declare periodic boundaries here as well.
    ``compatibility`` decides incompatible initial bisection labels (REJECT, an
    explicit UNIFORM_REFINEMENT, or the host triangle CONFORMING_CLOSURE, which
    device and periodic orbit bisection refuse). ``predicate_mode`` certifies
    geometric decisions of the native metric route (EXACT uses meshcore when
    available, otherwise the filtered host predicates whose unresolved signs
    reject the operation); device routes evaluate FILTERED_DEVICE predicates and
    escalate unresolved signs. ``maximum_passes`` bounds metric passes and
    ``maximum_closure_iterations`` the bisection closure, whose batches are also
    admitted against the cell, vertex, and work-unit ``limits`` before they are
    bisected. ``relocation`` enables vertex relocation in metric passes.
    ``device_policy`` (required by, and only by, device routes) fixes the capacity
    bucket of the device state. ``distribution`` with ``partition_policy`` carries
    cell ownership through the lineage. ``association_transfer`` carries the
    source's B-Rep associations: B-Rep corner vertices are fixed, ambiguously
    classified edges are protected, CAD classes separate cell, facet, and edge
    classes, and the target associations are propagated through the native
    lineage or re-derived after provider remeshing. Nonconforming HP active meshes
    require an ``audit_policy`` that records rather than rejects nonmanifold
    (hanging) vertices. ``geometry_transition`` governs how nested native routes
    carry a non-affine source coordinate map: refinement restricts it exactly, and
    coarsening follows its declared approximation policy or refuses.
    """

    route: MeshAdaptationRoute = eqx.field(static=True)
    protected_scopes: tuple[MeshingScope, ...]
    compatibility: BisectionCompatibility = eqx.field(static=True)
    predicate_mode: PredicateMode = eqx.field(static=True)
    maximum_passes: int = eqx.field(static=True)
    maximum_closure_iterations: int = eqx.field(static=True)
    relocation: bool = eqx.field(static=True)
    limits: MeshingLimits
    audit_policy: CellMeshAuditPolicy
    device_policy: AdaptiveSimplexPolicy | None
    provider: MmgProvider | OmegaHProvider | None = eqx.field(static=True)
    provider_options: MmgOptions | OmegaHOptions | None
    distribution: MeshDistribution | None
    partition_policy: MeshPartitionPolicy | None
    association_transfer: (
        BRepAssociationTransfer
        | PlcAssociationTransfer
        | ComposedAssociationTransfer
        | SurfaceAssociationTransfer
        | MappedReferenceAssociationTransfer
        | ImplicitAssociationTransfer
        | PeriodicAssociationTransfer
        | None
    )
    geometry_transition: CellGeometryTransitionPolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        route: MeshAdaptationRoute,
        /,
        *,
        protected_scopes: tuple[MeshingScope, ...] = (),
        compatibility: BisectionCompatibility = BisectionCompatibility.REJECT,
        predicate_mode: PredicateMode | None = None,
        maximum_passes: int = 16,
        maximum_closure_iterations: int = 256,
        relocation: bool = True,
        limits: MeshingLimits | None = None,
        audit_policy: CellMeshAuditPolicy | None = None,
        device_policy: AdaptiveSimplexPolicy | None = None,
        provider: MmgProvider | OmegaHProvider | None = None,
        provider_options: MmgOptions | OmegaHOptions | None = None,
        distribution: MeshDistribution | None = None,
        partition_policy: MeshPartitionPolicy | None = None,
        association_transfer: BRepAssociationTransfer
        | PlcAssociationTransfer
        | ComposedAssociationTransfer
        | SurfaceAssociationTransfer
        | MappedReferenceAssociationTransfer
        | ImplicitAssociationTransfer
        | PeriodicAssociationTransfer
        | None = None,
        geometry_transition: CellGeometryTransitionPolicy | None = None,
    ) -> None:
        if not isinstance(route, MeshAdaptationRoute):
            raise TypeError("route must be MeshAdaptationRoute.")
        protected = tuple(protected_scopes)
        if not all(isinstance(scope, MeshingScope) for scope in protected):
            raise TypeError("protected_scopes must contain MeshingScope values.")
        if not isinstance(compatibility, BisectionCompatibility):
            raise TypeError("compatibility must be BisectionCompatibility.")
        mode = _resolved_predicate_mode(route, predicate_mode)
        if (route in _DEVICE_ROUTES) != (device_policy is not None):
            raise ValueError("device_policy is required by, and only by, device routes.")
        if (
            route is MeshAdaptationRoute.DEVICE_BISECTION
            and compatibility is BisectionCompatibility.CONFORMING_CLOSURE
        ):
            raise ValueError(
                "Device bisection closes matching-condition labels only; "
                "CONFORMING_CLOSURE is executed by NATIVE_BISECTION."
            )
        if device_policy is not None and not isinstance(
            device_policy, AdaptiveSimplexPolicy
        ):
            raise TypeError("device_policy must be AdaptiveSimplexPolicy or None.")
        passes = _count(maximum_passes, "maximum_passes")
        closure = _count(maximum_closure_iterations, "maximum_closure_iterations")
        relocate = _flag(relocation, "relocation")
        limits_ = MeshingLimits() if limits is None else limits
        audit = CellMeshAuditPolicy() if audit_policy is None else audit_policy
        if not isinstance(limits_, MeshingLimits):
            raise TypeError("limits must be MeshingLimits or None.")
        if not isinstance(audit, CellMeshAuditPolicy):
            raise TypeError("audit_policy must be CellMeshAuditPolicy or None.")
        if (distribution is None) != (partition_policy is None):
            raise ValueError("distribution and partition_policy are supplied together.")
        if distribution is not None and (
            not isinstance(distribution, MeshDistribution)
            or not isinstance(partition_policy, MeshPartitionPolicy)
        ):
            raise TypeError(
                "distribution must be MeshDistribution with a MeshPartitionPolicy."
            )
        if association_transfer is not None:
            if not isinstance(
                association_transfer,
                (
                    BRepAssociationTransfer,
                    PlcAssociationTransfer,
                    ComposedAssociationTransfer,
                    SurfaceAssociationTransfer,
                    MappedReferenceAssociationTransfer,
                    ImplicitAssociationTransfer,
                    PeriodicAssociationTransfer,
                ),
            ):
                raise TypeError(
                    "association_transfer must be BRepAssociationTransfer, "
                    "PlcAssociationTransfer, ComposedAssociationTransfer, SurfaceAssociationTransfer, MappedReferenceAssociationTransfer, ImplicitAssociationTransfer, PeriodicAssociationTransfer, or None."
                )
            if route is MeshAdaptationRoute.HP:
                raise ValueError("HP adaptation cannot carry geometry associations.")
            if isinstance(association_transfer, MappedReferenceAssociationTransfer) and (
                route is not MeshAdaptationRoute.NATIVE_MIXED or distribution is not None
            ):
                raise ValueError(
                    "Exact mapped-root association transport requires the host nested mixed-family route."
                )
            if isinstance(association_transfer, ComposedAssociationTransfer) and (
                route is not MeshAdaptationRoute.NATIVE_MIXED or distribution is not None
            ):
                raise ValueError(
                    "Exact composed-source association transport requires the host nested mixed-family route."
                )
            if isinstance(association_transfer, ImplicitAssociationTransfer) and (
                route
                not in (
                    MeshAdaptationRoute.NATIVE_BISECTION,
                    MeshAdaptationRoute.DEVICE_BISECTION,
                )
                or distribution is not None
            ):
                raise ValueError(
                    "Implicit surface source transport requires its native nested surface route."
                )
            if isinstance(association_transfer, PeriodicAssociationTransfer) and (
                route is not MeshAdaptationRoute.NATIVE_BISECTION
            ):
                raise ValueError(
                    "Periodic original-source renewal requires the host nested quotient route."
                )
        transition = (
            CellGeometryTransitionPolicy()
            if geometry_transition is None
            else geometry_transition
        )
        if not isinstance(transition, CellGeometryTransitionPolicy):
            raise TypeError(
                "geometry_transition must be CellGeometryTransitionPolicy or None."
            )
        _check_provider(route, provider, provider_options, protected, distribution)
        self.route = route
        self.protected_scopes = protected
        self.compatibility = compatibility
        self.predicate_mode = mode
        self.maximum_passes = passes
        self.maximum_closure_iterations = closure
        self.relocation = relocate
        self.limits = limits_
        self.audit_policy = audit
        self.device_policy = device_policy
        self.provider = provider
        self.provider_options = provider_options
        self.distribution = distribution
        self.partition_policy = partition_policy
        self.association_transfer = association_transfer
        self.geometry_transition = transition
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mesh-adaptation-policy",
                "route": route.value,
                "protected": [scope.scope_id for scope in protected],
                "compatibility": compatibility.value,
                "predicate_mode": mode.value,
                "maximum_passes": passes,
                "maximum_closure_iterations": closure,
                "relocation": relocate,
                "limits": limits_.limits_id,
                "audit": audit.policy_id,
                "device_policy": None
                if device_policy is None
                else device_policy.policy_id,
                "provider": None if provider is None else type(provider).__name__,
                "provider_options": _provider_options_record(provider_options),
                "distribution": None
                if distribution is None
                else distribution.distribution_id,
                "partition_policy": None
                if partition_policy is None
                else partition_policy.policy_id,
                "association_transfer": None
                if association_transfer is None
                else association_transfer.transfer_id,
                "geometry_transition": transition.policy_id,
            }
        )


class _AdaptationConstraints(NamedTuple):
    """Host constraint arrays resolved once from protection and organization."""

    protected_edge_keys: np.ndarray
    protected_edge_mask: np.ndarray
    protected_vertex_ids: np.ndarray
    fixed_vertex_mask: np.ndarray
    cell_classes: np.ndarray
    facet_classes: np.ndarray
    edge_classes: np.ndarray
    metric_values: np.ndarray


def _selected_rows(mesh: CellMesh, scope: MeshingScope, /) -> np.ndarray:
    return np.flatnonzero(np.asarray(resolve_mesh_scope(mesh, scope).mask))


def _closure_rows(
    mesh: CellMesh, dimension: int, rows: np.ndarray, target: int, /
) -> np.ndarray:
    """Rows of the dimension-``target`` entities in the closure of selected rows."""

    for degree in range(dimension, target, -1):
        relation = mesh.topology.incidences[degree - 1].relation
        valid = np.asarray(relation.valid) & np.isin(
            np.asarray(relation.target_indices), rows
        )
        rows = np.unique(np.asarray(relation.source_indices)[valid])
    return rows


def _organization_scopes(source: CellMeshingResult, /) -> tuple[MeshingScope, ...]:
    return tuple(
        value.scope for value in (*source.patches, *source.zones, *source.labels)
    )


def _membership(
    mesh: CellMesh, scopes: tuple[MeshingScope, ...], dimension: int, /
) -> np.ndarray:
    columns = [
        np.asarray(resolve_mesh_scope(mesh, scope).mask, dtype=np.bool_)
        for scope in scopes
        if scope.entity_dimension == dimension
    ]
    count = mesh.entity_set(dimension).count
    if not columns:
        return np.zeros((count, 0), dtype=np.bool_)
    return np.stack(columns, axis=1)


def _row_classes(rows: np.ndarray, /) -> np.ndarray:
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    return inverse.reshape((-1,)).astype(np.int64)


def _cell_classes(
    mesh: CellMesh,
    scopes: tuple[MeshingScope, ...],
    /,
    *,
    family_identity: bool = False,
    scientific_block_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Organization classes; mixed templates retain family, not coordinate-block identity."""

    dimension = mesh.topological_dimension
    families = {
        kind: index
        for index, kind in enumerate(sorted({b.cell_kind for b in mesh.blocks}))
    }
    block_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    block_rows = (
        scientific_block_rows
        if scientific_block_rows is not None
        else np.concatenate(
            [
                np.full(
                    (block.cell_count,),
                    families[block.cell_kind] if family_identity else index,
                    dtype=np.int64,
                )
                for index, block in enumerate(mesh.blocks)
            ]
        )
    )
    cells = key_rows(block_ids[:, None], entity_keys(mesh, dimension))
    return _row_classes(
        np.column_stack(
            (block_rows[cells], _membership(mesh, scopes, dimension).astype(np.int64))
        )
    )


def _edge_classes(
    mesh: CellMesh, scopes: tuple[MeshingScope, ...], cell_classes: np.ndarray, /
) -> np.ndarray:
    """Planar feature classes: 0 for free interior edges, positive otherwise.

    Boundary edges, region interfaces, and edges of any edge organization are
    features; distinct incident region pairs or memberships are distinct classes.
    """

    count = mesh.entity_set(1).count
    relation = mesh.topology.incidences[1].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    edges = np.asarray(relation.source_indices)[valid]
    cells = cell_classes[np.asarray(relation.target_indices)[valid]]
    incident = np.bincount(edges, minlength=count)
    low = np.full((count,), np.iinfo(np.int64).max, dtype=np.int64)
    high = np.full((count,), -1, dtype=np.int64)
    np.minimum.at(low, edges, cells)
    np.maximum.at(high, edges, cells)
    member = _membership(mesh, scopes, 1)
    feature = (incident != 2) | (low != high) | np.any(member, axis=1)
    classes = np.zeros((count,), dtype=np.int64)
    if np.any(feature):
        keys = np.column_stack(
            (incident == 1, low, high, member.astype(np.int64))
        ).astype(np.int64)
        classes[feature] = 1 + _row_classes(keys[feature])
    return classes


def _metric_rows(mesh: CellMesh, metric: MeshMetricField, /) -> np.ndarray:
    scope = metric.scope
    vertices = mesh.entity_set(0)
    if (
        scope.entity_kind is not MeshingEntityKind.MESH
        or scope.entity_dimension != 0
        or scope.source_id != mesh.mesh_id
        or scope.source_revision != mesh.numeric_version
        or scope.entity_set_id != vertices.entity_set_id
    ):
        raise ValueError("The adaptation metric must be bound to the source vertices.")
    identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
    rows = key_rows(
        identifiers[:, None],
        np.asarray(mesh.vertex_global_ids, dtype=np.int64)[:, None],
    )
    if identifiers.size != rows.size or np.any(rows < 0):
        raise ValueError("The adaptation metric must cover every source vertex.")
    values = np.asarray(metric.values, dtype=np.float64)[rows]
    if values.shape[1:] != (mesh.ambient_dimension, mesh.ambient_dimension):
        raise ValueError("The adaptation metric must match the ambient dimension.")
    return values


def _uniform_scientific_block_rows(
    source: CellMeshingResult,
    hierarchy: BisectionHierarchy,
    limits: MeshingLimits,
    /,
) -> np.ndarray:
    from .._meshcore import current_native_execution_budget
    from ._bisection import _uniform_charge

    lineage = hierarchy.uniform_refinement
    if lineage is None:
        raise ValueError(
            "Uniform scientific block lookup requires its actual original source owner."
        )
    original = lineage.source
    origin = source.geometry.restriction_source
    if (
        origin is None
        or origin.source_geometry_id != cell_geometry_id(original.geometry)
        or origin.source_topology_id != original.mesh.topology_id
    ):
        raise ValueError(
            "Uniform scientific classes lost their original source geometry owner."
        )
    count = sum(block.cell_count for block in source.mesh.blocks)
    original_count = sum(block.cell_count for block in original.mesh.blocks)
    storage = (count + original_count) * 256
    if storage > limits.maximum_scratch_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Scientific block preparation exceeds the original host storage allowance.",
            stage="scientific-class-preparation",
        )
    if current_native_execution_budget() is not None:
        _uniform_charge(count + original_count, storage)
    root_blocks = {
        int(identifier): block_index
        for block_index, block in enumerate(original.mesh.blocks)
        for identifier in block.global_ids
    }
    rows = []
    for block in source.mesh.blocks:
        parents = np.asarray(origin.block_parent_cell_ids[block.name], dtype=np.int64)
        if parents.shape != (block.cell_count,) or any(
            int(parent) not in root_blocks for parent in parents
        ):
            raise ValueError(
                "A presentation block has no explicitly owned original scientific cell route."
            )
        rows.append(
            np.asarray([root_blocks[int(parent)] for parent in parents], dtype=np.int64)
        )
    return np.concatenate(rows)


def _uniform_class_declarations_match(
    original: CellMeshingResult,
    current: CellMeshingResult,
    degree: int,
    /,
) -> bool:
    from ._organization import _organization_definition_id

    if len(original.zones) != len(current.zones):
        return False
    zone_ids = {
        before.zone_id: after.zone_id
        for before, after in zip(original.zones, current.zones, strict=True)
    }
    for previous, present in (
        (original.patches, current.patches),
        (original.zones, current.zones),
        (original.labels, current.labels),
    ):
        expected = tuple(
            value for value in previous if value.scope.entity_dimension == degree
        )
        actual = tuple(
            value for value in present if value.scope.entity_dimension == degree
        )
        if len(expected) != len(actual):
            return False
        for before, after in zip(expected, actual, strict=True):
            match before:
                case MeshPatch():
                    if not isinstance(after, MeshPatch) or any(
                        value not in zone_ids for value in before.adjacent_zone_ids
                    ):
                        return False
                    witness = MeshPatch(
                        before.name,
                        after.scope,
                        connected=before.connected,
                        adjacent_zone_ids=tuple(
                            zone_ids[value] for value in before.adjacent_zone_ids
                        ),
                        source_adjacent_region_ids=before.source_adjacent_region_ids,
                    )
                case MeshZone():
                    if not isinstance(after, MeshZone):
                        return False
                    witness = MeshZone(
                        before.name,
                        before.role,
                        after.scope,
                        material_id=before.material_id,
                        region_role=before.region_role,
                    )
                case MeshLabel():
                    if not isinstance(after, MeshLabel):
                        return False
                    witness = MeshLabel(before.name, after.scope)
                case _:
                    raise TypeError(
                        "Uniform scientific classes require canonical organization declarations."
                    )
            if _organization_definition_id(witness) != _organization_definition_id(after):
                return False
    return True


def _uniform_cell_scientific_classes(
    source: CellMeshingResult,
    uniform: BisectionUniformRefinement,
    current_classes: np.ndarray,
    /,
) -> np.ndarray:
    """Keep captured class identities on unchanged, explicitly rooted occurrences."""
    if source.geometry.restriction_source is None:
        raise ValueError(
            "Scientific cell classes require their actual original source routes."
        )
    original = uniform.source
    parent_classes = dict(
        zip(
            np.asarray(uniform.parent_ids).tolist(),
            np.asarray(uniform.parent_classes).tolist(),
            strict=True,
        )
    )
    origin = source.geometry.restriction_source
    root_by_cell = {
        int(identifier): int(parent)
        for block in source.mesh.blocks
        for identifier, parent in zip(
            block.global_ids,
            np.asarray(origin.block_parent_cell_ids[block.name]),
            strict=True,
        )
    }
    cell_ids = np.asarray(
        source.mesh.entity_set(source.mesh.topological_dimension).entity_ids
    )
    roots = np.asarray(
        [root_by_cell[int(identifier)] for identifier in cell_ids], dtype=np.int64
    )
    original_ids = np.asarray(
        original.mesh.entity_set(original.mesh.topological_dimension).entity_ids
    )
    original_rows = key_rows(original_ids[:, None], roots[:, None])
    if np.any(original_rows < 0) or any(
        int(root) not in parent_classes for root in roots
    ):
        raise ValueError("A scientific class occurrence has an unowned original parent.")
    before = _membership(
        original.mesh, _organization_scopes(original), original.mesh.topological_dimension
    )
    after = _membership(
        source.mesh, _organization_scopes(source), source.mesh.topological_dimension
    )
    changed = (
        np.ones(roots.shape, dtype=np.bool_)
        if before.shape[1] != after.shape[1]
        else np.any(after != before[original_rows], axis=1)
    )
    if not _uniform_class_declarations_match(
        original, source, source.mesh.topological_dimension
    ):
        changed[:] = True
    for root in np.unique(roots):
        selected = roots == root
        if np.unique(current_classes[selected]).size != 1:
            changed[selected] = True
    captured = np.asarray([parent_classes[int(root)] for root in roots], dtype=np.int64)
    return np.where(
        changed,
        current_classes + int(np.max(np.asarray(uniform.parent_classes))) + 1,
        captured,
    )


def _resolve_constraints(
    source: CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
) -> _AdaptationConstraints:
    mesh = source.mesh
    dimension = mesh.topological_dimension
    edge_count = mesh.entity_set(1).count
    protected_edges = np.zeros((edge_count,), dtype=np.bool_)
    fixed = np.zeros((mesh.coordinates.shape[0],), dtype=np.bool_)
    for scope in policy.protected_scopes:
        if (
            policy.route is MeshAdaptationRoute.NATIVE_SURFACE_METRIC
            and scope.entity_dimension == dimension
        ):
            # A protected surface owns continuous fidelity, not frozen in-surface
            # topology. Its declared curves/corners remain protected below.
            continue
        rows = _selected_rows(mesh, scope)
        if scope.entity_dimension >= 1:
            protected_edges[_closure_rows(mesh, scope.entity_dimension, rows, 1)] = True
        fixed[_closure_rows(mesh, scope.entity_dimension, rows, 0)] = True
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    organization = _organization_scopes(source)
    uniform_hierarchy = (
        request.hierarchy
        if policy.route
        in (MeshAdaptationRoute.NATIVE_BISECTION, MeshAdaptationRoute.DEVICE_BISECTION)
        and isinstance(request, MarkedMeshAdaptation)
        and isinstance(request.hierarchy, BisectionHierarchy)
        and request.hierarchy.uniform_refinement is not None
        else None
    )
    scientific_blocks = (
        None
        if uniform_hierarchy is None
        else _uniform_scientific_block_rows(source, uniform_hierarchy, policy.limits)
    )
    if (
        uniform_hierarchy is None
        and policy.route
        in (MeshAdaptationRoute.NATIVE_BISECTION, MeshAdaptationRoute.DEVICE_BISECTION)
        and isinstance(request, MarkedMeshAdaptation)
        and isinstance(request.hierarchy, BisectionHierarchy)
    ):
        authority_ids = np.asarray(request.hierarchy.scientific_cell_ids, dtype=np.int64)
        active_ids = _cell_ids(mesh)
        rows = np.searchsorted(authority_ids, active_ids)
        if np.any(rows >= authority_ids.size) or not np.array_equal(
            authority_ids[rows], active_ids
        ):
            raise ValueError(
                "The bisection hierarchy omits active scientific block authority."
            )
        scientific_blocks = np.asarray(
            request.hierarchy.scientific_block_ids, dtype=np.int32
        )[rows]
    if (
        policy.route
        in (MeshAdaptationRoute.NATIVE_BISECTION, MeshAdaptationRoute.DEVICE_BISECTION)
        and isinstance(request, MarkedMeshAdaptation)
        and isinstance(request.hierarchy, PeriodicRefinement)
        and source.geometry.restriction_source is not None
    ):
        from ._periodic import _periodic_scientific_block_rows

        scientific_blocks = _periodic_scientific_block_rows(
            source,
            request.hierarchy,
            limits=policy.limits,
        )
    cell_classes = _cell_classes(
        mesh,
        organization,
        family_identity=policy.route
        in (
            MeshAdaptationRoute.NATIVE_MIXED,
            MeshAdaptationRoute.NATIVE_POLYHEDRAL,
            MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        ),
        scientific_block_rows=scientific_blocks,
    )
    if source.attributes:
        from ._layer_core import layer_interval_classes

        block_rows = key_rows(_cell_ids(mesh)[:, None], entity_keys(mesh, dimension))
        cell_classes = _row_classes(
            np.column_stack((cell_classes, layer_interval_classes(source)[block_rows]))
        )
    facet_classes = _row_classes(
        np.column_stack(
            (
                np.zeros((mesh.entity_set(dimension - 1).count,), dtype=np.int64),
                _membership(mesh, organization, dimension - 1).astype(np.int64),
            )
        )
    )
    edge_codes = np.zeros((edge_count,), dtype=np.int64)
    transfer = policy.association_transfer
    midpoint_required = policy.route in (
        MeshAdaptationRoute.NATIVE_BISECTION,
        MeshAdaptationRoute.DEVICE_BISECTION,
        MeshAdaptationRoute.NATIVE_MIXED,
    )
    if policy.route is MeshAdaptationRoute.NATIVE_MIXED:
        if not isinstance(request, MarkedMeshAdaptation):
            raise TypeError(
                "Mixed source constraints require marked template adaptation."
            )
        # Inverse complete-family restoration uses existing parent vertices;
        # a future split midpoint is not a hard constraint on discarded children.
        midpoint_required = request.refine_cell_ids.size > 0
    if transfer is not None and source.associations:
        match transfer:
            case BRepAssociationTransfer():
                classes = transfer.classes(source)
                codes = tuple(
                    np.where(
                        level.dimensions >= 0,
                        transfer.projection.entity_codes(
                            np.maximum(level.dimensions, 0), np.maximum(level.indices, 0)
                        ),
                        -1,
                    )
                    for level in classes
                )
            case PlcAssociationTransfer() | ComposedAssociationTransfer():
                classes = transfer.classes(source)
                codes = tuple(level.codes for level in classes)
                protected_edges |= transfer.protected_edges(
                    source,
                    midpoint_required=midpoint_required,
                )
            case (
                MappedReferenceAssociationTransfer()
                | ImplicitAssociationTransfer()
                | PeriodicAssociationTransfer()
            ):
                classes = transfer.classes(source)
                codes = tuple(level.codes for level in classes)
            case SurfaceAssociationTransfer():
                classes = transfer.classes(source)
                codes = tuple(level.codes for level in classes)
                protected_edges |= transfer.protected_edges(
                    source,
                    midpoint_required=midpoint_required,
                )
            case invalid:
                assert_never(invalid)
        fixed |= classes[0].dimensions == 0
        protected_edges |= ~classes[1].resolved
        cell_classes = _row_classes(np.column_stack((cell_classes, codes[dimension])))
        facet_classes = _row_classes(
            np.column_stack((facet_classes, codes[dimension - 1]))
        )
        edge_codes = np.where(classes[1].dimensions < dimension, 1 + codes[1], 0)
    protected_vertex_ids = vertex_ids[fixed]
    fixed |= np.any(_membership(mesh, organization, 0), axis=1)
    surface = mesh.topological_dimension == 2
    metric = (
        _metric_rows(mesh, request.metric)
        if isinstance(request, (MetricMeshAdaptation, RelocationMeshAdaptation))
        else np.zeros(
            (0, mesh.ambient_dimension, mesh.ambient_dimension), dtype=np.float64
        )
    )
    if uniform_hierarchy is not None:
        local_uniform = uniform_hierarchy.uniform_refinement
        if local_uniform is None:
            raise RuntimeError(
                "Scientific class preparation lost its admitted original source owner."
            )
        uniform = local_uniform
        collective = source.collective_evidence
        if (
            isinstance(collective, CollectiveMeshEvidence)
            and collective.uniform_refinement is not None
        ):
            collective_uniform = collective.uniform_refinement
            if (
                collective_uniform.dimension != local_uniform.dimension
                or collective_uniform.source.result_id != local_uniform.source.result_id
            ):
                raise ValueError(
                    "Collective scientific classes require the same genuine original "
                    "uniform source."
                )
            uniform = collective_uniform
        cell_classes = _uniform_cell_scientific_classes(source, uniform, cell_classes)
        if not _uniform_class_declarations_match(uniform.source, source, dimension - 1):
            facet_classes += int(np.max(np.asarray(uniform.parent_facet_classes))) + 1
    return _AdaptationConstraints(
        protected_edge_keys=entity_keys(mesh, 1)[protected_edges],
        protected_edge_mask=protected_edges,
        protected_vertex_ids=protected_vertex_ids,
        fixed_vertex_mask=fixed,
        cell_classes=cell_classes,
        facet_classes=facet_classes,
        edge_classes=(
            _feature_classes(_edge_classes(mesh, organization, cell_classes), edge_codes)
            if surface
            else _feature_classes(
                np.where(
                    np.any(_membership(mesh, organization, 1), axis=1),
                    1 + _row_classes(_membership(mesh, organization, 1)),
                    0,
                ),
                edge_codes,
            )
        ),
        metric_values=metric,
    )


def _feature_classes(classes: np.ndarray, cad_codes: np.ndarray, /) -> np.ndarray:
    """Split feature edge classes by B-Rep edge (``cad_codes`` 0 off B-Rep edges)."""
    feature = (classes > 0) | (cad_codes > 0)
    result = np.zeros(classes.shape, dtype=np.int64)
    if np.any(feature):
        result[feature] = 1 + _row_classes(np.column_stack((classes, cad_codes))[feature])
    return result


def _cell_ids(mesh: CellMesh, /) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )


def _require_affine_geometry(source: CellMeshingResult, /) -> None:
    """Refuse sources whose coordinate map is not the mesh's affine carrier.

    Routes without a geometry transition publish affine successors; a curved or
    displaced source map would collapse to its corners, so it is refused before
    any work.
    """

    if not is_affine_cell_geometry(source.mesh, source.geometry):
        raise ValueError(
            "This adaptation route has no geometry transition for a non-affine "
            "source CellGeometrySpec, so its coordinate map would be lost; nested "
            "simplex and mixed routes carry their supported coordinate maps."
        )


def _require_native_source(
    source: CellMeshingResult,
    transfer: BRepAssociationTransfer
    | PlcAssociationTransfer
    | ComposedAssociationTransfer
    | SurfaceAssociationTransfer
    | MappedReferenceAssociationTransfer
    | ImplicitAssociationTransfer
    | PeriodicAssociationTransfer
    | None,
    /,
    *,
    nested: bool,
    plc_ancestry: bool = False,
) -> None:
    """Admit a native source; nested routes carry their supported coordinate maps."""

    if nested:
        nested_geometry_degree(source.mesh, source.geometry)
    else:
        match source.geometry.exact_source:
            case (
                ExactPowerCellGeometrySource()
                | ExactPowerCellGeometryRestrictionSource()
                | ExactPowerCellGeometryLinearActionSource() as power
            ):
                power.prepare(source.geometry.coordinates)
            case ExactPlcCellGeometrySource() | ExactPlcCellGeometryConvexSource() as plc:
                if not plc_ancestry:
                    raise ValueError(
                        "This adaptation route has no transition carrying exact PLC source ancestry."
                    )
                plc.prepare(source.geometry.coordinates)
            case None:
                _require_affine_geometry(source)
            case invalid:
                assert_never(invalid)
    if source.boundary is not None and not isinstance(
        transfer, ImplicitAssociationTransfer
    ):
        raise ValueError("Native adaptation requires an explicit boundary-model remap.")
    if source.attributes:
        from ._layer_core import validate_layer_index_attributes

        validate_layer_index_attributes(source)
    if source.associations or isinstance(transfer, ImplicitAssociationTransfer):
        if transfer is None:
            raise ValueError(
                "Geometry associations require the policy's association_transfer."
            )
        transfer.source_associations(source)
    if not source.coordinate_contract.is_orthonormal_cartesian:
        raise ValueError("Native adaptation requires an orthonormal Cartesian frame.")


def _check_marked_source(
    source: CellMeshingResult, request: MarkedMeshAdaptation, /
) -> None:
    identifiers = _cell_ids(source.mesh)
    marked = np.concatenate(
        (np.asarray(request.refine_cell_ids), np.asarray(request.coarsen_cell_ids))
    )
    if not np.all(np.isin(marked, identifiers)):
        raise ValueError("Marked cell IDs are not cells of the source mesh.")
    if request.layer_columns is not None and not np.all(
        np.isin(request.layer_columns.cell_ids, identifiers)
    ):
        raise ValueError("Layer column cell IDs must belong to the current source mesh.")


def _check_route_request(
    source: CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
) -> None:
    mesh = source.mesh
    kinds = {block.cell_kind for block in mesh.blocks}
    if (
        isinstance(request, MarkedMeshAdaptation)
        and request.layer_columns is not None
        and policy.route is not MeshAdaptationRoute.NATIVE_MIXED
    ):
        raise ValueError("Layer columns are admitted only by NATIVE_MIXED.")
    match policy.route:
        case MeshAdaptationRoute.NATIVE_BISECTION | MeshAdaptationRoute.DEVICE_BISECTION:
            if not isinstance(request, MarkedMeshAdaptation):
                raise TypeError(
                    f"{policy.route.name} requires a marked bisection request."
                )
            if kinds != {"triangle"} and kinds != {"tetrahedron"}:
                raise ValueError("Bisection requires triangle or tetrahedron blocks.")
            if mesh.periodic_topology is None:
                if not isinstance(request.hierarchy, (BisectionHierarchy, type(None))):
                    raise TypeError(
                        "Nonperiodic bisection requires its bisection hierarchy."
                    )
            elif policy.route is MeshAdaptationRoute.DEVICE_BISECTION:
                raise ValueError(
                    "Periodic orbit transactions are implemented on the host route."
                )
            elif policy.compatibility is BisectionCompatibility.CONFORMING_CLOSURE:
                raise ValueError(
                    "Periodic orbit bisection owns its quotient closure and consumes no "
                    "Maubach labels; CONFORMING_CLOSURE applies to nonperiodic sources."
                )
            _require_native_source(source, policy.association_transfer, nested=True)
            _check_marked_source(source, request)
        case MeshAdaptationRoute.NATIVE_MIXED:
            if not isinstance(request, MarkedMeshAdaptation) or not isinstance(
                request.hierarchy, (MixedAdaptationHierarchy, type(None))
            ):
                raise TypeError("NATIVE_MIXED requires marked mixed-template adaptation.")
            match (mesh.topological_dimension, mesh.ambient_dimension):
                case (2, 2):
                    if kinds != {"quadrilateral"}:
                        raise ValueError(
                            "NATIVE_MIXED in 2D requires quadrilateral blocks."
                        )
                    if request.layer_columns is not None:
                        raise ValueError(
                            "Planar quad adaptation has no axial layer columns."
                        )
                case (3, 3):
                    if not kinds <= {"tetrahedron", "hexahedron", "prism", "pyramid"}:
                        raise ValueError(
                            "NATIVE_MIXED in 3D requires tetrahedron, hexahedron, prism, or pyramid blocks."
                        )
                case _:
                    raise ValueError(
                        "NATIVE_MIXED requires certified planar quads or supported 3D cells."
                    )
            _require_native_source(source, policy.association_transfer, nested=True)
            _check_marked_source(source, request)
        case MeshAdaptationRoute.NATIVE_GEOMETRY_REALIZATION:
            if not isinstance(request, GeometryRealizationMeshAdaptation):
                raise TypeError(
                    "NATIVE_GEOMETRY_REALIZATION requires GeometryRealizationMeshAdaptation."
                )
            from ._geometry_realization import validate_geometry_realization_adaptation

            validate_geometry_realization_adaptation(source, request, policy)
        case MeshAdaptationRoute.NATIVE_POLYHEDRAL:
            if not isinstance(request, PolyhedralMeshAdaptation):
                raise TypeError("NATIVE_POLYHEDRAL requires PolyhedralMeshAdaptation.")
            if kinds != {"polyhedron"} or mesh.ambient_dimension != 3:
                raise ValueError("NATIVE_POLYHEDRAL requires packed 3D polyhedral cells.")
            if mesh.periodic_topology is not None:
                raise ValueError(
                    "NATIVE_POLYHEDRAL requires implemented periodic face/site closure."
                )
            regenerated = request.operation is PolyhedralAdaptationOperation.REGENERATE
            if (
                regenerated
                and any(
                    scope.entity_dimension < 3 for scope in _organization_scopes(source)
                )
                and not source.associations
            ):
                raise ValueError(
                    "Organized regeneration requires authoritative source-stratum associations."
                )
            if source.attributes or policy.distribution is not None:
                raise ValueError(
                    "Polyhedral attributes and distribution require an explicit "
                    "common-refinement transition."
                )
            if regenerated and (
                source.associations or source.geometry.exact_source is not None
            ):
                if source.geometry.exact_source is None:
                    _require_affine_geometry(source)
                else:
                    source.geometry.exact_source.prepare(source.geometry.coordinates)
                if (
                    source.boundary is not None
                    or not source.coordinate_contract.is_orthonormal_cartesian
                ):
                    raise ValueError(
                        "Polyhedral regeneration requires its represented Cartesian PLC source."
                    )
                if policy.association_transfer is not None:
                    raise ValueError(
                        "Polyhedral regeneration derives associations from its source carrier, not a projection transfer."
                    )
            else:
                _require_native_source(source, policy.association_transfer, nested=False)
        case MeshAdaptationRoute.NATIVE_METRIC_2D | MeshAdaptationRoute.DEVICE_METRIC_2D:
            if not isinstance(request, (MetricMeshAdaptation, RelocationMeshAdaptation)):
                raise TypeError(
                    f"{policy.route.name} requires a metric or relocation request."
                )
            if kinds != {"triangle"} or mesh.ambient_dimension != 2:
                raise ValueError(f"{policy.route.name} requires planar triangle blocks.")
            _require_native_source(source, policy.association_transfer, nested=False)
        case MeshAdaptationRoute.NATIVE_METRIC_3D:
            if not isinstance(request, (MetricMeshAdaptation, RelocationMeshAdaptation)):
                raise TypeError(
                    "NATIVE_METRIC_3D requires a metric or relocation request."
                )
            if kinds != {"tetrahedron"} or mesh.ambient_dimension != 3:
                raise ValueError("NATIVE_METRIC_3D requires tetrahedron blocks in 3D.")
            if mesh.periodic_topology is not None:
                raise ValueError(
                    "NATIVE_METRIC_3D requires an orbit-synchronized transaction "
                    "for periodic source topology."
                )
            _require_native_source(
                source, policy.association_transfer, nested=False, plc_ancestry=True
            )
        case MeshAdaptationRoute.NATIVE_SURFACE_METRIC:
            from ._surface_association_transfer import source_associations

            if not isinstance(request, (MetricMeshAdaptation, RelocationMeshAdaptation)):
                raise TypeError(
                    "NATIVE_SURFACE_METRIC requires a metric or relocation request."
                )
            if kinds != {"triangle"} or mesh.ambient_dimension != 3:
                raise ValueError(
                    "NATIVE_SURFACE_METRIC requires embedded triangle blocks."
                )
            if (
                request.coordinate_contract is None
                or request.coordinate_contract.spatial_id
                != source.coordinate_contract.spatial_id
            ):
                raise ValueError(
                    "The surface metric must declare the exact source physical coordinate frame."
                )
            if not isinstance(policy.association_transfer, SurfaceAssociationTransfer):
                raise ValueError(
                    "NATIVE_SURFACE_METRIC requires original parametric source support."
                )
            if policy.geometry_transition.reconstruction != "bounded_chart_deformation":
                raise ValueError(
                    "NATIVE_SURFACE_METRIC requires bounded_chart_deformation geometry policy."
                )
            if (
                source.certification is None
                or not source.certification.passed
                or source.certification.request.fidelity_tolerance is None
            ):
                raise ValueError(
                    "NATIVE_SURFACE_METRIC requires accepted original continuous source fidelity."
                )
            _require_native_source(source, policy.association_transfer, nested=True)
            source_associations(policy.association_transfer.support, source)
        case MeshAdaptationRoute.NATIVE_LEVEL_SET:
            if not isinstance(request, LevelSetMeshAdaptation):
                raise TypeError("NATIVE_LEVEL_SET requires LevelSetMeshAdaptation.")
            if kinds != {"tetrahedron"} or mesh.ambient_dimension != 3:
                raise ValueError("NATIVE_LEVEL_SET requires tetrahedron blocks in 3D.")
            _require_native_source(source, policy.association_transfer, nested=False)
        case MeshAdaptationRoute.MMG | MeshAdaptationRoute.OMEGA_H:
            if not isinstance(request, MetricMeshAdaptation):
                raise TypeError("Provider routes require a metric request.")
            _require_affine_geometry(source)
            if source.region_evidence is not None:
                raise ValueError(
                    "Provider remeshing has unknown lineage and cannot carry region evidence."
                )
            if isinstance(
                policy.association_transfer,
                (PlcAssociationTransfer, ComposedAssociationTransfer),
            ):
                raise ValueError(
                    "PLC source transfer requires actual topology lineage; "
                    "provider rederivation is unsupported."
                )
            if isinstance(policy.association_transfer, SurfaceAssociationTransfer):
                raise ValueError(
                    "Parametric surface source transfer requires actual topology lineage; "
                    "provider rederivation is unsupported."
                )
        case MeshAdaptationRoute.HP:
            if not isinstance(request, MarkedMeshAdaptation) or not isinstance(
                request.hierarchy, FiniteElementHPEpoch
            ):
                raise TypeError("HP requires a marked request with its hp epoch.")
            if request.hierarchy.mesh.mesh_id != mesh.mesh_id:
                raise ValueError(
                    "The hp epoch must describe the exact source mesh revision."
                )
            if request.refine_cell_ids.size and request.coarsen_cell_ids.size:
                raise ValueError("One HP adaptation either refines or coarsens.")
            if source.patches or source.zones or source.labels:
                raise ValueError("HP adaptation cannot inherit mesh organization.")
            _require_native_source(source, None, nested=False)
            _check_marked_source(source, request)
        case invalid:
            assert_never(invalid)


@final
class PreparedMeshAdaptation(StrictModule, NonTrainableState):
    """One request bound to one exact source revision under one policy."""

    source: CellMeshingResult
    request: MeshAdaptationRequest
    policy: MeshAdaptationPolicy
    constraints: _AdaptationConstraints
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CellMeshingResult,
        request: MeshAdaptationRequest,
        policy: MeshAdaptationPolicy,
        constraints: _AdaptationConstraints,
        /,
    ) -> None:
        self.source = source
        self.request = request
        self.policy = policy
        self.constraints = constraints
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mesh-adaptation",
                "source": source.result_id,
                "request": request.request_id,
                "policy": policy.policy_id,
            }
        )


def prepare_mesh_adaptation(
    source: CellMeshingResult,
    request: MeshAdaptationRequest,
    /,
    *,
    policy: MeshAdaptationPolicy,
) -> PreparedMeshAdaptation:
    """Validate the route/request pairing and resolve protection and organization."""

    if not isinstance(source, CellMeshingResult):
        raise TypeError("source must be CellMeshingResult.")
    if not isinstance(
        request,
        (
            MarkedMeshAdaptation,
            MetricMeshAdaptation,
            RelocationMeshAdaptation,
            LevelSetMeshAdaptation,
            PolyhedralMeshAdaptation,
            GeometryRealizationMeshAdaptation,
        ),
    ):
        raise TypeError("request must be a typed mesh-adaptation request.")
    if not isinstance(policy, MeshAdaptationPolicy):
        raise TypeError("policy must be MeshAdaptationPolicy.")
    source.audit.require_passed()
    _check_route_request(source, request, policy)
    if policy.distribution is not None:
        policy.distribution.require_current(
            MeshPart(policy.distribution.part.name, source)
        )
    match policy.route:
        case (
            MeshAdaptationRoute.NATIVE_BISECTION
            | MeshAdaptationRoute.NATIVE_METRIC_2D
            | MeshAdaptationRoute.NATIVE_METRIC_3D
            | MeshAdaptationRoute.NATIVE_SURFACE_METRIC
            | MeshAdaptationRoute.NATIVE_LEVEL_SET
            | MeshAdaptationRoute.NATIVE_MIXED
            | MeshAdaptationRoute.NATIVE_POLYHEDRAL
            | MeshAdaptationRoute.DEVICE_BISECTION
            | MeshAdaptationRoute.DEVICE_METRIC_2D
        ):
            constraints = _resolve_constraints(source, request, policy)
        case (
            MeshAdaptationRoute.MMG
            | MeshAdaptationRoute.OMEGA_H
            | MeshAdaptationRoute.HP
            | MeshAdaptationRoute.NATIVE_GEOMETRY_REALIZATION
        ):
            for scope in policy.protected_scopes:
                resolve_mesh_scope(source.mesh, scope)
            constraints = _empty_constraints(source.mesh)
        case invalid:
            assert_never(invalid)
    prepared = PreparedMeshAdaptation(source, request, policy, constraints)
    if policy.route is MeshAdaptationRoute.NATIVE_LEVEL_SET:
        from ._level_set import validate_level_set_adaptation

        validate_level_set_adaptation(prepared)
    if (
        policy.route is MeshAdaptationRoute.NATIVE_BISECTION
        and source.mesh.periodic_topology is not None
    ):
        from ._periodic import validate_periodic_bisection

        validate_periodic_bisection(prepared)
    return prepared


def _empty_constraints(mesh: CellMesh, /) -> _AdaptationConstraints:
    empty = np.zeros((0,), dtype=np.int64)
    return _AdaptationConstraints(
        protected_edge_keys=np.zeros((0, 2), dtype=np.int64),
        protected_edge_mask=np.zeros((0,), dtype=np.bool_),
        protected_vertex_ids=empty,
        fixed_vertex_mask=np.zeros((0,), dtype=np.bool_),
        cell_classes=empty,
        facet_classes=empty,
        edge_classes=empty,
        metric_values=np.zeros((0, mesh.ambient_dimension, mesh.ambient_dimension)),
    )


MeshAdaptationEvidence: TypeAlias = (
    BisectionEvidence
    | LocalMetricEvidence
    | DeviceMetricEvidence
    | MetricRemeshingEvidence
    | LevelSetEvidence
    | MixedAdaptationEvidence
    | PolyhedralAdaptationEvidence
    | CurvedGeometryEvidence
    | MmgAdaptationResult
    | OmegaHAdaptationResult
    | FiniteElementHPRefinementResult
)


def _evidence_id(evidence: MeshAdaptationEvidence | None, /) -> str | None:
    match evidence:
        case None:
            return None
        case (
            BisectionEvidence()
            | LocalMetricEvidence()
            | DeviceMetricEvidence()
            | MetricRemeshingEvidence()
            | LevelSetEvidence()
            | MixedAdaptationEvidence()
            | PolyhedralAdaptationEvidence()
            | CurvedGeometryEvidence()
        ):
            return evidence.evidence_id
        case (
            MmgAdaptationResult()
            | OmegaHAdaptationResult()
            | FiniteElementHPRefinementResult()
        ):
            return evidence.result_id
        case invalid:
            assert_never(invalid)


@final
class MeshAdaptationResult(StrictModule, NonTrainableState):
    """Certified target of one executed adaptation with lineage and evidence.

    ``transition``, ``lineage``, ``stencil``, and ``transfer`` are absent when
    the certified target remains the exact source. That source-preserving result
    is ``UNCHANGED`` only when no requested work was refused and no convergence
    criterion remains unmet; it may instead be ``PARTIAL``, ``STALLED``, or
    ``PASS_LIMIT`` or ``RESOURCE_LIMIT``. Provider routes and nonnested polyhedral
    vertices without interpolation ancestry have no nodal stencil or transfer.
    ``transfer`` maps source vertex values to target vertex values when that
    compatibility map exists; ``common_refinement`` carries the independently
    certified geometric transfer of polyhedral cells. ``request`` is the exact
    executed request, not inferred from route or target shape. ``metric`` is
    target-bound, ``hierarchy`` continues marked adaptation, and elapsed wall time
    is evidence outside ``result_id``.
    """

    status: MeshAdaptationStatus = eqx.field(static=True)
    route: MeshAdaptationRoute = eqx.field(static=True)
    request: MeshAdaptationRequest
    policy: MeshAdaptationPolicy
    source: CellMeshingResult
    target: CellMeshingResult
    transition: CellMeshTransition | None
    lineage: MeshLineage | None
    stencil: VertexInterpolationStencil | None
    transfer: FiniteElementTopologyTransfer | None
    metric: MeshMetricField | None
    compliance: MeshingComplianceReport
    evidence: MeshAdaptationEvidence | None
    common_refinement: PreparedCommonRefinement | None
    hierarchy: MeshAdaptationHierarchy
    distribution: MeshDistributionTransition | None
    elapsed_seconds: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedMeshAdaptation,
        outcome: _RouteOutcome,
        compliance: MeshingComplianceReport,
        distribution: MeshDistributionTransition | None,
        elapsed_seconds: float,
        /,
    ) -> None:
        source_preserved = outcome.target.result_id == prepared.source.result_id
        absent = (
            outcome.transition is None
            and outcome.lineage is None
            and outcome.stencil is None
            and outcome.transfer is None
        )
        if source_preserved != absent:
            raise ValueError(
                "A source-preserving adaptation has no transition or transfer evidence."
            )
        if outcome.status is MeshAdaptationStatus.UNCHANGED and not source_preserved:
            raise ValueError("UNCHANGED must preserve the exact source result.")
        if outcome.transition is not None:
            if outcome.lineage is None:
                raise ValueError("An adaptation transition requires its exact lineage.")
            if (
                outcome.transition.target.result_id != outcome.target.result_id
                or outcome.transition.lineage.lineage_id != outcome.lineage.lineage_id
            ):
                raise ValueError("The transition must describe the certified target.")
        common = outcome.common_refinement
        if common is not None and (
            common.source_mesh_id != prepared.source.mesh.mesh_id
            or common.target_mesh_id != outcome.target.mesh.mesh_id
            or common.source_topology_id != prepared.source.mesh.topology_id
            or common.target_topology_id != outcome.target.mesh.topology_id
        ):
            raise ValueError(
                "Common refinement must bind the exact source and accepted target."
            )
        self.status = outcome.status
        self.route = prepared.policy.route
        self.request = prepared.request
        self.policy = prepared.policy
        self.source = prepared.source
        self.target = outcome.target
        self.transition = outcome.transition
        self.lineage = outcome.lineage
        self.stencil = outcome.stencil
        self.transfer = outcome.transfer
        self.metric = outcome.metric
        self.compliance = compliance
        self.evidence = outcome.evidence
        self.common_refinement = outcome.common_refinement
        self.hierarchy = outcome.hierarchy
        self.distribution = distribution
        self.elapsed_seconds = float(elapsed_seconds)
        self.prepared_id = prepared.prepared_id
        self.result_id = canonical_fingerprint(
            {
                "kind": "mesh-adaptation-result",
                "prepared": prepared.prepared_id,
                "status": outcome.status.value,
                "target": outcome.target.result_id,
                "transition": None
                if outcome.transition is None
                else outcome.transition.transition_id,
                "transfer": None
                if outcome.transfer is None
                else outcome.transfer.transfer_id,
                "metric": None if outcome.metric is None else outcome.metric.metric_id,
                "compliance": compliance.report_id,
                "evidence": _evidence_id(outcome.evidence),
                "common_refinement": None
                if outcome.common_refinement is None
                else outcome.common_refinement.refinement_id,
                "hierarchy": _hierarchy_id(outcome.hierarchy),
                "distribution": None
                if distribution is None
                else distribution.transition_id,
            }
        )

    @property
    def geometry_transition(self) -> CellGeometryTransition | None:
        """How a non-affine source coordinate map reached the target, if any."""

        return None if self.transition is None else self.transition.geometry_transition

    def _parent_ids(self) -> np.ndarray:
        transition = self.transition
        if transition is None or transition.parent_cell_ids is None:
            raise ValueError("This adaptation has no nested parent witnesses.")
        parents = np.asarray(transition.parent_cell_ids, dtype=np.int64)
        if np.any(parents < 0):
            raise ValueError(
                "Coarsened target cells span several source cells and have no parent."
            )
        return parents

    @property
    def parent_cells(self) -> np.ndarray:
        """Source cell row (concatenated block order) containing each target cell.

        Target cells follow concatenated block order; this is the ``parent_cells``
        witness of `prepare_nested_field_transfer`. Refused without a nested
        refinement lineage or when a target cell was coarsened.
        """

        parents = self._parent_ids()
        ids = np.concatenate(
            [np.asarray(block.global_ids, np.int64) for block in self.source.mesh.blocks]
        )
        order = np.argsort(ids, kind="stable")
        return order[np.searchsorted(ids[order], parents)]

    @property
    def parent_reference_vertices(self) -> np.ndarray:
        """Target reference corners in their source parents, with family-dependent extent."""

        self._parent_ids()
        transition = self.transition
        if transition is None or transition.parent_reference_vertices is None:
            raise ValueError("This adaptation has no nested parent witnesses.")
        return np.asarray(transition.parent_reference_vertices, dtype=np.float64)

    @property
    def coarsening_witnesses(self) -> NestedReferenceWitnesses | None:
        """Owning fine-to-coarse charts, never inferred from geometric coincidence."""
        transition = self.transition
        if transition is None:
            return None
        if (
            self.route is MeshAdaptationRoute.NATIVE_BISECTION
            and self.source.mesh.periodic_topology is not None
            and transition.transition_kind is MeshTransitionKind.COARSEN
        ):
            from ._periodic import periodic_coarsening_witnesses

            return periodic_coarsening_witnesses(self)
        if transition.coarsened_cell_ids is not None:
            if (
                transition.coarsened_into_ids is None
                or transition.coarsened_reference_vertices is None
            ):
                raise RuntimeError("Coarsening transition lost its complete witness.")
            return NestedReferenceWitnesses(
                np.asarray(transition.coarsened_cell_ids, dtype=np.int64),
                np.asarray(transition.coarsened_into_ids, dtype=np.int64),
                np.asarray(transition.coarsened_reference_vertices, dtype=np.float64),
            )
        geometry = transition.geometry_transition
        if geometry is None or geometry.coarsened_cell_ids.shape[0] == 0:
            return None
        return NestedReferenceWitnesses(
            np.asarray(geometry.coarsened_cell_ids, dtype=np.int64),
            np.asarray(geometry.coarsened_into_ids, dtype=np.int64),
            np.asarray(geometry.coarsened_reference_vertices, dtype=np.float64),
        )


class _RouteOutcome(NamedTuple):
    status: MeshAdaptationStatus
    target: CellMeshingResult
    transition: CellMeshTransition | None
    lineage: MeshLineage | None
    stencil: VertexInterpolationStencil | None
    transfer: FiniteElementTopologyTransfer | None
    metric: MeshMetricField | None
    evidence: MeshAdaptationEvidence | None
    hierarchy: MeshAdaptationHierarchy
    common_refinement: PreparedCommonRefinement | None = None


def _unchanged(
    prepared: PreparedMeshAdaptation,
    evidence: MeshAdaptationEvidence | None,
    hierarchy: MeshAdaptationHierarchy,
    /,
    *,
    status: MeshAdaptationStatus = MeshAdaptationStatus.UNCHANGED,
) -> _RouteOutcome:
    return _RouteOutcome(
        status, prepared.source, None, None, None, None, None, evidence, hierarchy
    )


def _p1_vertex_measures(mesh: CellMesh, /) -> np.ndarray:
    """Integral of every P1 hat function: each simplex measure over its vertices."""

    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    measures = np.zeros((coordinates.shape[0],), dtype=np.float64)
    dimension = mesh.topological_dimension
    plan = SmallLinearSolvePlan(dimension)
    factorial = float(np.prod(np.arange(1, dimension + 1)))
    for block in mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int64)
        edges = coordinates[cells[:, 1:]] - coordinates[cells[:, :1]]
        if dimension == mesh.ambient_dimension:
            volume = np.abs(np.asarray(determinant_small_linear(plan, edges))) / factorial
        else:
            gram = edges @ np.swapaxes(edges, 1, 2)
            volume = (
                np.sqrt(np.maximum(np.asarray(determinant_small_linear(plan, gram)), 0.0))
                / factorial
            )
        np.add.at(
            measures,
            cells.reshape((-1,)),
            np.repeat(volume / cells.shape[1], cells.shape[1]),
        )
    return measures


class _NativeTarget(NamedTuple):
    target: CellMeshingResult
    transition: CellMeshTransition
    lineage: MeshLineage
    stencil: VertexInterpolationStencil | None
    transfer: FiniteElementTopologyTransfer | None


def _geometry_transition(
    prepared: PreparedMeshAdaptation, edit: CellTopologyEdit, mesh: CellMesh, /
) -> CellGeometryTransition | None:
    """Target coordinate map of a non-affine source through a nested edit.

    Simplex corner carriers keep publishing their affine successor. Mixed corner
    maps are restricted exactly too; degree-one prism/pyramid/hex maps need not be
    affine in reference coordinates. Nonnested curved routes require their owner.
    """

    source = prepared.source
    mixed = prepared.policy.route is MeshAdaptationRoute.NATIVE_MIXED
    if (
        not mixed
        and source.geometry.exact_source is None
        and source.geometry.periodic_source is None
        and source.geometry.restriction_source is None
        and source.geometry.storage is None
        and source.geometry.geometry_layout_id
        == CellGeometrySpec.affine(source.mesh).geometry_layout_id
        and np.array_equal(
            np.asarray(source.geometry.coordinates), np.asarray(source.mesh.coordinates)
        )
    ):
        return None
    match edit.operation:
        case "nested_refinement" | "nested_coarsening" | "nested_adaptation":
            pass
        case "local_reconnection" | "relocation":
            raise ValueError(
                "A non-nested edit cannot carry a non-affine source coordinate map."
            )
        case operation:
            assert_never(operation)
    degree = nested_geometry_degree(source.mesh, source.geometry)
    layout = (
        CellGeometrySpec.affine(mesh)
        if mixed or degree == 1
        else _straight_geometry(mesh, degree)
    )
    return transition_nested_cell_geometry(
        source.mesh,
        source.geometry,
        mesh,
        layout,
        refinement=edit.refinement,
        coarsening=edit.coarsening,
        policy=prepared.policy.geometry_transition,
    )


def _region_boundary_transition(
    source: CellMeshingResult,
    mesh: CellMesh,
    lineage: MeshLineage,
    patches: tuple[MeshPatch, ...],
    zones: tuple[MeshZone, ...],
    labels: tuple[MeshLabel, ...],
    /,
) -> tuple[RegionBoundaryEvidence, ...]:
    """Rebind source-volume controls by canonical declaration and lineage IDs."""
    if not source.region_boundary_evidence:
        return ()
    for evidence in source.region_boundary_evidence:
        evidence.require_current(source.mesh, source.labels, source.patches)
    target_zone_ids = {zone.zone_id for zone in zones}
    zone_ids: dict[str, str] = {}
    for zone in source.zones:
        expected_zone = MeshZone(
            zone.name,
            zone.role,
            inherit_scope(zone.scope, lineage, mesh, zone.name),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        if expected_zone.zone_id not in target_zone_ids:
            raise ValueError("A source region-boundary zone has no canonical successor.")
        zone_ids[zone.zone_id] = expected_zone.zone_id
    target_labels = {label.label_id: label for label in labels}
    inherited_labels: dict[str, MeshLabel] = {}
    for evidence in source.region_boundary_evidence:
        original_label = evidence.boundary_label
        if original_label.label_id in inherited_labels:
            continue
        expected_label = MeshLabel(
            original_label.name,
            inherit_scope(original_label.scope, lineage, mesh, original_label.name),
        )
        if expected_label.label_id not in target_labels:
            raise ValueError("A source region-boundary label has no canonical successor.")
        inherited_labels[original_label.label_id] = target_labels[expected_label.label_id]
    source_patches = {patch.patch_id: patch for patch in source.patches}
    target_patch_ids = {patch.patch_id for patch in patches}
    patch_ids: dict[str, str] = {}
    for evidence in source.region_boundary_evidence:
        for identifier, _ in evidence.patch_sides:
            if identifier in patch_ids:
                continue
            original_patch = source_patches[identifier]
            expected_patch = MeshPatch(
                original_patch.name,
                inherit_scope(original_patch.scope, lineage, mesh, original_patch.name),
                connected=original_patch.connected,
                adjacent_zone_ids=tuple(
                    zone_ids[value] for value in original_patch.adjacent_zone_ids
                ),
                source_adjacent_region_ids=original_patch.source_adjacent_region_ids,
            )
            if expected_patch.patch_id not in target_patch_ids:
                raise ValueError(
                    "A source region-boundary patch has no canonical successor."
                )
            patch_ids[identifier] = expected_patch.patch_id
    renewed = tuple(
        RegionBoundaryEvidence(
            mesh,
            evidence.source_scope,
            evidence.source_region_id,
            evidence.control_id,
            evidence.material_id,
            evidence.role,
            inherited_labels[evidence.boundary_label.label_id],
            tuple(
                (patch_ids[identifier], side) for identifier, side in evidence.patch_sides
            ),
        )
        for evidence in source.region_boundary_evidence
    )
    for evidence in renewed:
        evidence.require_current(mesh, labels, patches)
    return renewed


def _binary_parent_geometry_element(
    element: CellGeometryElement,
    corners: np.ndarray,
    source_id: int,
    target_id: int,
    hierarchy: BisectionHierarchy,
    /,
) -> CellGeometryElement:
    """Recover an actual retained parent action through authored binary halves."""
    from ..discretization._cell_geometry import BarycentricCellGeometryElement

    records = {
        int(parent): index
        for index, parent in enumerate(np.asarray(hierarchy.record_parent_ids))
    }
    predecessors = {
        int(child): int(parent)
        for parent, children in zip(
            hierarchy.record_parent_ids, hierarchy.record_child_ids, strict=True
        )
        for child in children
    }
    visited: set[int] = set()
    while source_id != target_id:
        if (
            source_id in visited
            or source_id not in predecessors
            or not isinstance(element, BarycentricCellGeometryElement)
        ):
            raise ValueError(
                "A coarsened P1 cell lost its actual retained binary parent action."
            )
        visited.add(source_id)
        parent = predecessors[source_id]
        record = records[parent]
        parent_corners = np.asarray(hierarchy.record_parent_rows)[record]
        ordered = np.asarray(hierarchy.record_parent_vertices)[record]
        tag = int(np.asarray(hierarchy.record_parent_tags)[record])
        midpoint = int(np.asarray(hierarchy.record_vertex_ids)[record])
        expected = np.zeros((corners.size, corners.size), dtype=np.float64)
        for row, vertex in enumerate(corners):
            support = (ordered[0], ordered[tag]) if vertex == midpoint else (vertex,)
            for identifier in support:
                positions = np.flatnonzero(parent_corners == identifier)
                if positions.size != 1:
                    raise ValueError(
                        "A binary coefficient action has foreign scientific parent support."
                    )
                expected[row, positions[0]] = 0.5 if vertex == midpoint else 1.0
        if not np.array_equal(np.asarray(element.barycentric_weights), expected):
            raise ValueError(
                "A coarsened P1 cell changed its authored binary coefficient action."
            )
        element, corners, source_id = element.source_element, parent_corners, parent
    return element


def _full_p1_source_cells(
    source: CellMeshingResult,
    authority_mesh: CellMesh,
    authority_geometry: CellGeometrySpec,
    elements: tuple[CellGeometryElement, ...],
    routes: tuple[Array, ...],
    /,
    *,
    direct_authority: bool,
    restored_authority: bool = False,
) -> tuple[
    dict[int, _FullP1SourceCell],
    dict[int, _FullP1OriginalCell],
    Array,
    CoordinateSourceBank,
]:
    """Authenticate coefficient columns and original scientific cell owners.

    A root's presentation owner is its explicitly retained authored source
    block: the authority block itself for an authored/direct authority, or the
    authority restriction record's ``block_source_blocks`` entry. ``None`` marks
    a root whose authored owner block was never retained.
    """
    from ..discretization._cell_geometry import (
        _require_full_p1_source,
        BarycentricCellGeometryElement,
        PolynomialComposedCellGeometryElement,
    )
    from ..discretization._coordinate_enclosure import prepared_coordinate_source_bank

    origin = source.geometry.restriction_source
    original_elements, original_routes, bank = authority_geometry.resolve(authority_mesh)
    original_bank = prepared_coordinate_source_bank(authority_geometry)
    current_bank = (
        original_bank
        if source.geometry.coordinates is authority_geometry.coordinates
        and source.geometry.exact_source is authority_geometry.exact_source
        else prepared_coordinate_source_bank(source.geometry)
    )
    original_cells: dict[int, _FullP1OriginalCell] = {}
    restriction = authority_geometry.restriction_source
    retained_owners = None if restriction is None else restriction.block_source_blocks
    for original_block, original_element, original_route in zip(
        authority_mesh.blocks,
        original_elements,
        original_routes,
        strict=True,
    ):
        root = _require_full_p1_source(original_element)
        root_name: str | None = (
            original_block.name if direct_authority or origin is None else None
        )
        if retained_owners is not None:
            root_name = retained_owners[original_block.name]
            while isinstance(
                root,
                (
                    BarycentricCellGeometryElement,
                    PolynomialComposedCellGeometryElement,
                ),
            ):
                root = root.source_element
            root = _require_full_p1_source(root)
        for original_row, identifier in enumerate(original_block.global_ids):
            root_identifier = (
                int(identifier)
                if direct_authority or origin is None
                else int(
                    np.asarray(origin.block_parent_cell_ids[original_block.name])[
                        original_row
                    ]
                )
            )
            root_corners = (
                np.asarray(authority_mesh.vertex_global_ids)[
                    np.asarray(original_block.vertices)[original_row]
                ]
                if direct_authority or origin is None
                else np.asarray(origin.block_parent_vertex_ids[original_block.name])[
                    original_row
                ]
            )
            original_cells[root_identifier] = (
                root_name,
                root,
                np.asarray(np.asarray(original_route)[original_row], dtype=np.int32),
                np.asarray(root_corners, dtype=np.int64),
                original_row,
            )
    if (
        direct_authority
        and not restored_authority
        and origin is not None
        and (
            origin.source_geometry_id != cell_geometry_id(authority_geometry)
            or origin.source_topology_id != authority_mesh.topology_id
        )
    ):
        raise ValueError(
            "Full P1 actions lost their explicitly retained scientific source owner."
        )
    source_cells: dict[int, _FullP1SourceCell] = {}
    for block, element, route in zip(source.mesh.blocks, elements, routes, strict=True):
        if isinstance(element, PolynomialComposedCellGeometryElement):
            root: CellGeometryElement = element
            while isinstance(root, PolynomialComposedCellGeometryElement):
                root = root.source_element
            _require_full_p1_source(root)
            if element.degree != 1 or element.cell_kind not in (
                "triangle",
                "tetrahedron",
            ):
                raise ValueError("Exact full P1 charts must remain affine simplex maps.")
        else:
            element = _require_full_p1_source(element)
        for cell_index, (cell, vertices, coefficients) in enumerate(
            zip(block.global_ids, block.vertices, route, strict=True)
        ):
            ancestor = (
                int(cell)
                if origin is None
                else int(np.asarray(origin.block_parent_cell_ids[block.name])[cell_index])
            )
            corners = (
                np.asarray(source.mesh.vertex_global_ids)[vertices]
                if origin is None
                else np.asarray(origin.block_parent_vertex_ids[block.name])[cell_index]
            )
            if ancestor not in original_cells:
                raise ValueError(
                    "Full P1 action has an undeclared original scientific cell."
                )
            root_name, original_root, original_route, original_corners, original_row = (
                original_cells[ancestor]
            )
            if not np.array_equal(corners, original_corners):
                raise ValueError(
                    "Full P1 action changed its original scientific corner columns."
                )
            if tuple(current_bank[int(index)] for index in coefficients) != tuple(
                original_bank[int(index)] for index in original_route
            ):
                raise ValueError(
                    "Full P1 action changed its original scientific coefficient bank or route columns."
                )
            source_cells[int(cell)] = (
                element,
                np.asarray(source.mesh.vertex_global_ids)[vertices],
                original_route,
                ancestor,
                corners,
                root_name,
                original_row,
                original_root,
            )
    return source_cells, original_cells, bank, original_bank


def _uniform_rational_p1_chart(
    source_element: _FullP1Element,
    barycentric_weights: tuple[tuple[Fraction, ...], ...],
    /,
) -> _FullP1Element:
    """Encode one uniform-plus-binary action as an exact rational chart."""
    from ..discretization._cell_geometry import (
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )

    width = source_element.local_dof_count
    dimension = source_element.topological_dimension
    if (
        len(barycentric_weights) != width
        or any(len(row) != width for row in barycentric_weights)
        or width != dimension + 1
        or any(value < 0 for row in barycentric_weights for value in row)
        or any(sum(row, Fraction(0)) != 1 for row in barycentric_weights)
    ):
        raise ValueError(
            "A uniform rational chart requires complete simplex barycentric rows."
        )
    numerators = np.asarray(
        [
            [row[source].numerator for source in range(1, width)]
            for row in barycentric_weights
        ],
        dtype=object,
    )
    denominators = np.asarray(
        [
            [row[source].denominator for source in range(1, width)]
            for row in barycentric_weights
        ],
        dtype=object,
    )
    return PolynomialComposedCellGeometryElement(
        source_element,
        coordinate_lagrange_element(source_element.cell_kind, 1),
        numerators,
        denominators,
    )


def _apply_full_p1_action(
    source_element: _FullP1Element, action: ArrayLike, /
) -> _FullP1Element:
    """Compose an exact affine simplex action with either supported P1 owner."""
    from ..discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        PolynomialComposedCellGeometryElement,
    )

    if isinstance(source_element, PolynomialComposedCellGeometryElement):
        values = np.asarray(action)
        exact = tuple(
            tuple(
                value if isinstance(value, Fraction) else Fraction(float(value))
                for value in row
            )
            for row in values
        )
        return _uniform_rational_p1_chart(source_element, exact)
    return BarycentricCellGeometryElement(source_element, np.asarray(action))


def _uniform_exact_target_actions(
    uniform: BisectionUniformRefinement,
    hierarchy: BisectionHierarchy,
    edit: CellTopologyEdit,
    parents: dict[int, int],
    /,
) -> _UniformRationalActions:
    """Compose canonical subset rows and binary midpoint records exactly once."""
    from ._topology_edit import TopologyEditBlock

    arrays = uniform.host_arrays()
    width = uniform.dimension + 1
    vertex_actions: dict[tuple[int, int], tuple[Fraction, ...]] = {}
    cell_roots: dict[int, int] = {}

    for parent, children, vertices, actions in zip(
        np.asarray(arrays["parent_ids"], dtype=np.int64).tolist(),
        np.asarray(arrays["child_ids"], dtype=np.int64).tolist(),
        np.asarray(arrays["child_vertices"], dtype=np.int64).tolist(),
        np.asarray(arrays["barycentric_weights"], dtype=np.float64).tolist(),
        strict=True,
    ):
        root = int(parent)
        cell_roots[root] = root
        for child, child_vertices, child_actions in zip(
            children,
            vertices,
            actions,
            strict=True,
        ):
            cell_roots[int(child)] = root
            for vertex, raw in zip(child_vertices, child_actions, strict=True):
                raw_ = np.asarray(raw, dtype=np.float64)
                support = np.flatnonzero(raw_)
                if support.size == 0:
                    raise ValueError(
                        "A uniform source vertex has no barycentric support."
                    )
                expected = 1.0 / support.size
                if not np.all(raw_[support] == expected) or np.any(
                    np.delete(raw_, support) != 0.0
                ):
                    raise ValueError(
                        "Uniform source vertices changed their canonical subset action."
                    )
                exact = tuple(
                    Fraction(1, int(support.size)) if source in support else Fraction(0)
                    for source in range(width)
                )
                key = root, int(vertex)
                previous = vertex_actions.setdefault(key, exact)
                if previous != exact:
                    raise ValueError(
                        "Uniform siblings disagree on an exact source vertex action."
                    )

    records = sorted(
        zip(
            np.asarray(hierarchy.record_parent_ids, dtype=np.int64).tolist(),
            np.asarray(hierarchy.record_parent_vertices, dtype=np.int64).tolist(),
            np.asarray(hierarchy.record_parent_tags, dtype=np.int64).tolist(),
            np.asarray(hierarchy.record_child_ids, dtype=np.int64).tolist(),
            np.asarray(hierarchy.record_vertex_ids, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    for parent, ordered, tag, children, midpoint in records:
        root = cell_roots.get(parent)
        if root is None:
            raise ValueError("A binary refinement record has no uniform scientific root.")
        first = vertex_actions.get((root, int(ordered[0])))
        second = vertex_actions.get((root, int(ordered[tag])))
        if first is None or second is None:
            raise ValueError(
                "A binary refinement midpoint lost its exact endpoint actions."
            )
        exact = tuple(
            (left + right) / 2 for left, right in zip(first, second, strict=True)
        )
        key = root, midpoint
        previous = vertex_actions.setdefault(key, exact)
        if previous != exact:
            raise ValueError(
                "Binary refinement records disagree on an exact midpoint action."
            )
        for child in children:
            known = cell_roots.setdefault(int(child), root)
            if known != root:
                raise ValueError(
                    "Binary refinement descendants disagree on their uniform root."
                )

    result: _UniformRationalActions = {}
    for block in edit.blocks:
        if not isinstance(block, TopologyEditBlock):
            raise ValueError(
                "Uniform rational actions require fixed-family simplex blocks."
            )
        for identifier, row in zip(block.cell_ids, block.cells, strict=True):
            target = int(identifier)
            predecessor = parents.get(target, target)
            root = cell_roots.get(target, cell_roots.get(predecessor, predecessor))
            known = cell_roots.get(target, root)
            if known != root:
                raise ValueError("A target cell changed its uniform scientific parent.")
            vertices = edit.vertex_global_ids[np.asarray(row, dtype=np.int64)]
            action = tuple(vertex_actions[(root, int(vertex))] for vertex in vertices)
            result[target] = root, action
    return result


def _full_p1_target_element(
    source_cell: _FullP1SourceCell,
    original_cells: dict[int, _FullP1OriginalCell],
    edit: CellTopologyEdit,
    identifier: int,
    row: NDArray[np.int32],
    parent: dict[int, int],
    direct_authority: bool,
    uniform: BisectionUniformRefinement | None,
    hierarchy: BisectionHierarchy | None,
    uniform_actions: _UniformRationalActions | None,
    /,
) -> CellGeometryElement:
    """Recover retained parent actions or author one actual incoming action."""

    current_element, current_corners = source_cell[:2]
    if edit.coarsening is not None and identifier in np.asarray(
        edit.coarsening.coarse_cell_ids
    ):
        if direct_authority and identifier in original_cells:
            return original_cells[identifier][1]
        if uniform_actions is not None and identifier in uniform_actions:
            root, action = uniform_actions[identifier]
            if source_cell[3] != root or root not in original_cells:
                raise ValueError(
                    "A uniform rational chart changed its scientific parent cell."
                )
            return _uniform_rational_p1_chart(original_cells[root][1], action)
        if uniform is not None and hierarchy is not None:
            from ._bisection import _uniform_binary_chart

            root_index, child_index, steps = _uniform_binary_chart(identifier, hierarchy)
            root_identifier = int(np.asarray(uniform.parent_ids)[root_index])
            element = original_cells[root_identifier][1]
            if child_index is not None:
                element = _apply_full_p1_action(
                    element, uniform.barycentric_weights[root_index, child_index]
                )
            for step in steps:
                element = _apply_full_p1_action(element, step)
            return element
        if hierarchy is not None:
            return _binary_parent_geometry_element(
                current_element,
                current_corners,
                parent[identifier],
                identifier,
                hierarchy,
            )
        raise ValueError(
            "A coarsened coefficient action lacks its actual scientific parent chart."
        )
    if uniform_actions is not None and identifier in uniform_actions:
        root, action = uniform_actions[identifier]
        if source_cell[3] != root or root not in original_cells:
            raise ValueError(
                "A uniform rational chart changed its scientific parent cell."
            )
        return _uniform_rational_p1_chart(original_cells[root][1], action)
    action = np.zeros(
        (current_element.local_dof_count, current_element.local_dof_count),
        dtype=np.float64,
    )
    for corner, vertex in enumerate(row):
        for support, weight, valid in zip(
            edit.stencil_sources[vertex],
            edit.stencil_weights[vertex],
            edit.stencil_valid[vertex],
            strict=True,
        ):
            if valid:
                matches = np.flatnonzero(current_corners == support)
                if matches.size != 1:
                    raise ValueError(
                        "A full P1 action has foreign scientific parent corner support."
                    )
                action[corner, matches[0]] += weight
    return (
        current_element
        if np.array_equal(
            action, np.eye(current_element.local_dof_count, dtype=np.float64)
        )
        else _apply_full_p1_action(current_element, action)
    )


def _whole_source_inverse_edit(
    original: CellMesh,
    edit: CellTopologyEdit,
    /,
) -> CellTopologyEdit | None:
    """Restore only complete original scientific parent incidence."""
    from ._topology_edit import TopologyEditBlock

    original_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in original.blocks)
    )
    target_ids = np.concatenate(tuple(block.cell_ids for block in edit.blocks))
    if not np.array_equal(np.sort(target_ids), np.sort(original_ids)):
        return None
    actual_cells = {}
    for block in edit.blocks:
        if not isinstance(block, TopologyEditBlock):
            raise ValueError("A whole source inverse requires original simplex cells.")
        for identifier, row in zip(block.cell_ids, block.cells, strict=True):
            actual_cells[int(identifier)] = (
                block.cell_kind,
                tuple(edit.vertex_global_ids[row].tolist()),
            )
    original_vertices = np.asarray(original.vertex_global_ids)
    expected_cells = {
        int(identifier): (block.cell_kind, tuple(original_vertices[row].tolist()))
        for block in original.blocks
        for identifier, row in zip(
            block.global_ids, np.asarray(block.vertices), strict=True
        )
    }
    if actual_cells != expected_cells:
        raise ValueError(
            "A whole source inverse changed original scientific parent incidence."
        )
    positions = {
        int(identifier): index for index, identifier in enumerate(edit.vertex_global_ids)
    }
    blocks = tuple(
        TopologyEditBlock(
            block.name,
            block.cell_kind,
            None,
            np.asarray(
                [
                    [positions[int(identifier)] for identifier in row]
                    for row in original_vertices[block.vertices]
                ],
                dtype=np.int32,
            ),
            np.asarray(block.global_ids, dtype=np.int64),
        )
        for block in original.blocks
    )
    return edit._replace(blocks=blocks)


def _full_p1_edit_geometry(
    prepared: PreparedMeshAdaptation,
    edit: CellTopologyEdit,
    /,
    *,
    bisection_hierarchy: BisectionHierarchy | None = None,
) -> tuple[CellTopologyEdit, CellGeometrySpec | None]:
    """Group actual full coefficient actions before topology/lineage binding."""
    from ..discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        CellGeometryRestrictionSource,
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )
    from ..discretization._cell_geometry_validity import cell_geometry_id
    from ..discretization._coordinate_enclosure import (
        coordinate_corner_images,
        rounded_point,
    )
    from ._topology_edit import TopologyEditBlock

    if prepared.policy.route not in (
        MeshAdaptationRoute.NATIVE_BISECTION,
        MeshAdaptationRoute.DEVICE_BISECTION,
    ):
        return edit, None
    source = prepared.source
    request = prepared.request
    request_hierarchy = (
        request.hierarchy if isinstance(request, MarkedMeshAdaptation) else None
    )
    hierarchy = (
        bisection_hierarchy
        if bisection_hierarchy is not None
        and (
            bisection_hierarchy.uniform_refinement is not None
            or request_hierarchy is None
        )
        else request_hierarchy
    )
    uniform = (
        hierarchy.uniform_refinement
        if isinstance(hierarchy, BisectionHierarchy)
        else None
    )
    periodic = hierarchy if isinstance(hierarchy, PeriodicRefinement) else None
    elements, routes, bank = source.geometry.resolve(source.mesh)
    origin = source.geometry.restriction_source
    from ..discretization._coordinate_enclosure import coordinate_source_signature
    from ..discretization.fem._reference import FiniteElementSpec

    if not all(
        isinstance(
            element,
            (
                BarycentricCellGeometryElement,
                PolynomialComposedCellGeometryElement,
            ),
        )
        or isinstance(element, FiniteElementSpec)
        and element.cell_kind in ("triangle", "tetrahedron")
        and coordinate_source_signature(element)
        == coordinate_source_signature(coordinate_lagrange_element(element.cell_kind, 1))
        for element in elements
    ):
        return edit, None
    authority_mesh = (
        uniform.source.mesh
        if uniform is not None
        else periodic.source
        if periodic is not None
        else source.mesh
    )
    authority_geometry = (
        uniform.source.geometry
        if uniform is not None
        else periodic.source_geometry
        if periodic is not None
        else source.geometry
    )
    restored_authority = (
        origin is not None
        and origin.source_topology_id == source.mesh.topology_id
        and all(isinstance(element, FiniteElementSpec) for element in elements)
    )
    direct_authority = uniform is not None or periodic is not None or restored_authority
    source_cells, original_cells, bank, original_bank = _full_p1_source_cells(
        source,
        authority_mesh,
        authority_geometry,
        elements,
        routes,
        direct_authority=direct_authority,
        restored_authority=restored_authority,
    )
    if direct_authority:
        restored = _whole_source_inverse_edit(authority_mesh, edit)
        if restored is not None:
            return restored, authority_geometry
    parent = (
        {}
        if edit.refinement is None
        else dict(
            zip(
                np.asarray(edit.refinement.fine_cell_ids).tolist(),
                np.asarray(edit.refinement.coarse_cell_ids).tolist(),
                strict=True,
            )
        )
    )
    if edit.coarsening is not None:
        for child, target in zip(
            edit.coarsening.fine_cell_ids, edit.coarsening.coarse_cell_ids, strict=True
        ):
            parent.setdefault(int(target), int(child))
    uniform_actions = (
        _uniform_exact_target_actions(uniform, hierarchy, edit, parent)
        if uniform is not None and isinstance(hierarchy, BisectionHierarchy)
        else None
    )
    groups = {}
    geometry_elements = {}
    geometry_routes = {}
    source_parent_ids = {}
    source_parent_vertices = {}
    owner_blocks: dict[str, str | None] = {}
    for block in edit.blocks:
        if not isinstance(block, TopologyEditBlock):
            raise ValueError("Full P1 actions require actual fixed-family simplex edits.")
        for identifier, row in zip(block.cell_ids, block.cells, strict=True):
            source_cell = source_cells[parent.get(int(identifier), int(identifier))]
            _, _, coefficients, ancestor, corners, root_name, _, original_root = (
                source_cell
            )
            element = _full_p1_target_element(
                source_cell,
                original_cells,
                edit,
                int(identifier),
                np.asarray(row, dtype=np.int32),
                parent,
                direct_authority,
                uniform,
                hierarchy if isinstance(hierarchy, BisectionHierarchy) else None,
                uniform_actions,
            )
            presentation_root = (
                f"source-cell/{ancestor}" if root_name is None else root_name
            )
            name = (
                presentation_root
                if element.element_id == original_root.element_id
                else f"{presentation_root}/coefficient-action/{element.element_id}"
            )
            owner_blocks[name] = root_name
            groups.setdefault(name, (block, [], []))[1].append(row)
            groups[name][2].append(identifier)
            geometry_elements[name] = element
            geometry_routes.setdefault(name, []).append(coefficients)
            source_parent_ids.setdefault(name, []).append(ancestor)
            source_parent_vertices.setdefault(name, []).append(corners)
    for name, (original, rows, ids) in groups.items():
        order = np.argsort(np.asarray(ids, dtype=np.int64), kind="stable")
        groups[name] = (
            original,
            [rows[index] for index in order],
            [ids[index] for index in order],
        )
        geometry_routes[name] = [geometry_routes[name][index] for index in order]
        source_parent_ids[name] = [source_parent_ids[name][index] for index in order]
        source_parent_vertices[name] = [
            source_parent_vertices[name][index] for index in order
        ]
    blocks = tuple(
        TopologyEditBlock(
            name,
            original.cell_kind,
            None,
            np.asarray(rows, dtype=np.int32),
            np.asarray(ids, dtype=np.int64),
        )
        for name, (original, rows, ids) in sorted(groups.items())
    )
    geometry = CellGeometrySpec(
        geometry_elements,
        {
            name: np.asarray(values, dtype=np.int32)
            for name, values in geometry_routes.items()
        },
        bank,
        restriction_source=CellGeometryRestrictionSource(
            cell_geometry_id(source.geometry)
            if origin is None
            else origin.source_geometry_id,
            source.mesh.topology_id if origin is None else origin.source_topology_id,
            {
                name: np.asarray(values, dtype=np.int64)
                for name, values in source_parent_ids.items()
            },
            {
                name: np.asarray(values, dtype=np.int64)
                for name, values in source_parent_vertices.items()
            },
            block_source_blocks=None
            if any(owner is None for owner in owner_blocks.values())
            else {name: owner for name, owner in owner_blocks.items() if owner},
        ),
        exact_source=authority_geometry.exact_source,
        periodic_source=authority_geometry.periodic_source,
    )
    coordinates = np.array(edit.coordinates, dtype=np.float64, copy=True)
    images = {}
    for name, (_, rows, _) in groups.items():
        for row, route in zip(rows, geometry_routes[name], strict=True):
            corners = coordinate_corner_images(
                geometry_elements[name],
                tuple(original_bank[int(index)] for index in route),
            )
            if corners is None:
                raise ValueError(
                    "The full P1 coefficient action lost its exact physical corner images."
                )
            for vertex, image in zip(row, corners, strict=True):
                if int(vertex) in images and images[int(vertex)] != image:
                    raise ValueError(
                        "Incident full P1 coefficient actions disagree on an exact shared vertex."
                    )
                images[int(vertex)] = image
                coordinates[vertex] = rounded_point(image)
    return edit._replace(blocks=blocks, coordinates=coordinates), geometry


def _complete_coarsening_witnesses(
    source: CellMesh,
    target: CellMesh,
    lineage: MeshLineage,
    witnesses: NestedReferenceWitnesses | None,
    /,
) -> NestedReferenceWitnesses | None:
    """Complete a pure coarsening chart bank with exact preserved-cell identities."""

    if witnesses is None or np.asarray(witnesses.fine_cell_ids).size == 0:
        return None
    dimension = target.topological_dimension
    cells = lineage.entity_lineage(dimension)
    relation_sources = np.asarray(cells.source_global_ids, dtype=np.int64)
    relation_targets = np.asarray(cells.target_global_ids, dtype=np.int64)
    relation_kinds = np.asarray(cells.relation_kinds, dtype=np.int32)
    preserved = relation_kinds == int(EntityLineageKind.PRESERVED)
    coarsened = relation_kinds == int(EntityLineageKind.COARSENED_INTO)
    if (
        not np.all(preserved | coarsened)
        or np.asarray(cells.created_target_ids).size
        or np.asarray(cells.deleted_source_ids).size
    ):
        return None
    source_cells = {
        int(identifier): (
            block.cell_kind,
            tuple(np.asarray(source.vertex_global_ids)[np.asarray(vertices)].tolist()),
        )
        for block in source.blocks
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    }
    target_cells = {
        int(identifier): (
            block.cell_kind,
            tuple(np.asarray(target.vertex_global_ids)[np.asarray(vertices)].tolist()),
        )
        for block in target.blocks
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    }
    from ..discretization._reference_cell import reference_cell_topology

    identity_sources: list[int] = []
    identity_targets: list[int] = []
    identity_vertices: list[np.ndarray] = []
    for source_id, target_id in zip(
        relation_sources[preserved], relation_targets[preserved], strict=True
    ):
        source_cell = source_cells.get(int(source_id))
        target_cell = target_cells.get(int(target_id))
        if source_cell is None or source_cell != target_cell:
            return None
        identity_sources.append(int(source_id))
        identity_targets.append(int(target_id))
        identity_vertices.append(
            np.asarray(reference_cell_topology(source_cell[0]).vertices, dtype=np.float64)
        )
    fine = np.asarray(witnesses.fine_cell_ids, dtype=np.int64)
    coarse = np.asarray(witnesses.coarse_cell_ids, dtype=np.int64)
    reference = np.asarray(witnesses.fine_reference_vertices, dtype=np.float64)
    width = max(
        reference.shape[1],
        max((vertices.shape[0] for vertices in identity_vertices), default=0),
    )
    if reference.shape[1] < width:
        reference = np.pad(reference, ((0, 0), (0, width - reference.shape[1]), (0, 0)))
    if identity_vertices:
        identities = np.zeros(
            (len(identity_vertices), width, dimension), dtype=np.float64
        )
        for row, vertices in enumerate(identity_vertices):
            identities[row, : vertices.shape[0]] = vertices
        fine = np.concatenate((fine, np.asarray(identity_sources, dtype=np.int64)))
        coarse = np.concatenate((coarse, np.asarray(identity_targets, dtype=np.int64)))
        reference = np.concatenate((reference, identities))
    if set(fine.tolist()) != set(source_cells) or set(coarse.tolist()) != set(
        target_cells
    ):
        return None
    return NestedReferenceWitnesses(fine, coarse, reference)


def _adaptation_certification_inputs(
    source: CellMeshingResult, limits: MeshingLimits, /
) -> MeshCertificationInputs | None:
    """Rebind target certification work/storage to the adaptation policy ledger."""

    if source.certification is None:
        return None

    retained = source.certification.request
    original = retained.limits
    certificate_limits = MeshCertificateLimits(
        maximum_candidate_pairs=original.maximum_candidate_pairs,
        maximum_ray_tests=original.maximum_ray_tests,
        maximum_source_samples=original.maximum_source_samples,
        maximum_distance_evaluations=original.maximum_distance_evaluations,
        maximum_subdivision_depth=original.maximum_subdivision_depth,
        maximum_subdivision_pieces=original.maximum_subdivision_pieces,
        maximum_bernstein_nodes=original.maximum_bernstein_nodes,
        maximum_periodic_images=original.maximum_periodic_images,
        maximum_work_units=limits.maximum_work_units,
        maximum_scratch_bytes=limits.maximum_scratch_bytes,
    )
    return MeshCertificationInputs(
        source.mesh,
        source.geometry,
        retained.schedule,
        domain=retained.domain,
        cell_regions=retained.cell_regions,
        source=retained.source,
        fidelity_tolerance=retained.fidelity_tolerance,
        fidelity_sample_order=retained.fidelity_sample_order,
        limits=certificate_limits,
        junction_vertices=retained.junction_vertices,
        scoped_fidelity=retained.scoped_fidelity,
    )


def _finalize_native(
    prepared: PreparedMeshAdaptation,
    edit: CellTopologyEdit,
    kind: MeshTransitionKind,
    /,
    *,
    conservative: bool,
    common_refinement: PreparedCommonRefinement | None = None,
    polyhedral_construction: PolyhedralConstruction | None = None,
    polyhedral_geometry: CellGeometrySpec | None = None,
    surface_metric_outcome: SurfaceMetricOutcome | SphereMetricOutcome | None = None,
    plc_source: ExactPlcCellGeometrySource
    | ExactPlcCellGeometryConvexSource
    | None = None,
    surface_curve_witness: PreparedSurfaceCurveWitness | None = None,
    bisection_hierarchy: BisectionHierarchy | None = None,
) -> _NativeTarget:
    """Assemble, transition geometry, inherit organization, certify, bind transfer.

    A non-affine source map is carried by the accepted geometry transition: its
    target corners become the mesh coordinates and the complete successor
    `CellGeometrySpec` is certified. A nodal compatibility map retains constants;
    supported conservative simplex transfers use actual mapped hat measures.
    Nonnested polyhedral cell transfer is certified geometric common refinement,
    not an invented nodal parent map.

    This host commit boundary keeps source restoration, organization renewal,
    certification, and transfer binding in their required failure order. Numerical
    execution remains with each owning substrate; no intermediate target escapes.
    """

    source = prepared.source
    periodic_surface = (
        surface_metric_outcome.periodic_stage
        if isinstance(surface_metric_outcome, SurfaceMetricOutcome)
        else None
    )
    numeric_version = (
        surface_metric_outcome.target_mesh.numeric_version
        if isinstance(surface_metric_outcome, SphereMetricOutcome)
        else periodic_surface.target_mesh.numeric_version
        if periodic_surface is not None
        else f"adaptation:{prepared.prepared_id}"
    )
    edit, coefficient_geometry = _full_p1_edit_geometry(
        prepared,
        edit,
        bisection_hierarchy=bisection_hierarchy,
    )
    mesh, lineage, stencil = assemble_topology_edit(
        source.mesh, edit, numeric_version=numeric_version
    )
    audit_prepared_validity: CellValidityCertificate | None = None
    metric_prepared_embedding: GlobalEmbeddingCertificate | None = None
    metric_prepared_fidelity: SourceFidelityCertificate | None = None
    if surface_metric_outcome is not None:
        from ..geometry._surface_source_support import (
            SurfaceNativeRestrictionBoundarySource,
            SurfaceSourceRootAtlas,
        )
        from ._periodic import metric_target_lineage
        from ._surface_metric import (
            _surface_metric_source_cover,
            prepare_surface_metric_geometry_transition,
            prepare_surface_metric_source,
        )

        surface_transfer = prepared.policy.association_transfer
        if not isinstance(surface_transfer, SurfaceAssociationTransfer):
            raise TypeError(
                "Surface reconstruction requires its original source transfer."
            )
        if isinstance(surface_metric_outcome, SphereMetricOutcome):
            from ..discretization._cell_geometry_transfer import (
                transition_chart_deformed_cell_geometry,
            )

            actual = surface_metric_outcome
            mesh = actual.target_mesh
            lineage = metric_target_lineage(source.mesh, edit, mesh)
            geometry_transition = transition_chart_deformed_cell_geometry(
                source.mesh,
                source.geometry,
                mesh,
                actual.reconstruction,
                actual.deformation,
                policy=prepared.policy.geometry_transition,
            )
            audit_prepared_validity = actual.reconstruction.target_validity
        elif periodic_surface is not None:
            mesh = periodic_surface.target_mesh
            lineage = metric_target_lineage(source.mesh, edit, mesh)
            geometry_transition = periodic_surface.geometry_stage.geometry_transition
        else:
            witness = prepare_surface_metric_source(source, surface_transfer)
            _, material_root_witness = _surface_material_epoch(source, witness)
            if surface_curve_witness is None:
                raise TypeError(
                    "UV reconstruction requires its actual source curve witness."
                )
            prepared_source_fidelity = None
            if source.certification is not None and isinstance(
                source.certification.request.source,
                SurfaceNativeRestrictionBoundarySource,
            ):
                retained_source = source.certification.request.source
                retained_source.require_current()
                restriction = source.geometry.restriction_source
                if (
                    restriction is None
                    or retained_source.mesh.mesh_id != source.mesh.mesh_id
                    or cell_geometry_id(retained_source.geometry)
                    != cell_geometry_id(source.geometry)
                    or retained_source.support.root_geometry_id
                    != restriction.source_geometry_id
                    or retained_source.support.root_topology_id
                    != restriction.source_topology_id
                ):
                    raise ValueError(
                        "Prepared exact source fidelity has stale restriction authority."
                    )
                prepared_source_fidelity = source.certification.fidelity
                if prepared_source_fidelity is None:
                    raise ValueError(
                        "Prepared exact source restriction lacks fidelity evidence."
                    )
            prepared_source_certificates = None
            if (
                source.certification is not None
                and source.certification.embedding is not None
            ):
                prepared_source_certificates = (
                    source.audit.validity,
                    source.certification.embedding,
                )
            chart_transition = prepare_surface_metric_geometry_transition(
                source.mesh,
                source.geometry,
                mesh,
                surface_metric_outcome,
                surface_transfer.support.domain,
                witness,
                maximum_fidelity=_surface_fidelity_limit(source),
                policy=prepared.policy.geometry_transition,
                maximum_candidate_pairs=prepared.policy.limits.maximum_geometry_queries,
                maximum_memory_bytes=prepared.policy.limits.maximum_scratch_bytes,
                certificate_limits=source.certification.request.limits
                if source.certification is not None
                else None,
                validity_policy=prepared.policy.audit_policy.validity_policy,
                source_chart_cover=_surface_metric_source_cover(source),
                source_atlas=surface_transfer.support.original
                if isinstance(surface_transfer.support.original, SurfaceSourceRootAtlas)
                else None,
                curve_witness=surface_curve_witness,
                material_root_witness=material_root_witness,
                prepared_source_fidelity=prepared_source_fidelity,
                prepared_source_certificates=prepared_source_certificates,
            )
            audit_prepared_validity = chart_transition.deformation.target_validity
            metric_prepared_embedding = chart_transition.deformation.target_embedding
            geometry_transition = chart_transition.transition
            mesh = chart_transition.target_mesh
            target_fidelity_bound = float(
                np.max(
                    np.asarray(chart_transition.deformation.target_fidelity_bounds),
                    initial=0.0,
                )
            )
            if (
                prepared_source_fidelity is not None
                and chart_transition.deformation.target_domain_coverage == "certified"
            ):
                if target_fidelity_bound > prepared_source_fidelity.tolerance:
                    raise ValueError(
                        "Prepared target source fidelity exceeds its original tolerance."
                    )
                if source.certification is None:
                    raise RuntimeError(
                        "Exact surface fidelity reuse lost source certification."
                    )
                fidelity_binding = MeshCertificateBinding(
                    mesh,
                    geometry_transition.geometry,
                    prepared_source_fidelity.binding.coordinate_scope,
                    source.certification.request.limits,
                    source_id=surface_transfer.support.domain.source_id,
                    source_revision=surface_transfer.support.domain.source_revision,
                )
                metric_prepared_fidelity = SourceFidelityCertificate(
                    fidelity_binding,
                    (),
                    tolerance=prepared_source_fidelity.tolerance,
                    semantics=("certified", "certified"),
                    mesh_to_source=(target_fidelity_bound, 0.0),
                    source_to_mesh=(target_fidelity_bound, 0.0),
                    sample_order=prepared_source_fidelity.sample_order,
                    sample_counts=(
                        mesh.entity_set(mesh.topological_dimension).count,
                        surface_transfer.support.root_cell_ids.shape[0],
                    ),
                )
            lineage = metric_target_lineage(source.mesh, edit, mesh)
        geometry = geometry_transition.geometry
    elif plc_source is not None:
        geometry_transition = None
        geometry = CellGeometrySpec.plc(mesh, plc_source)
    elif polyhedral_geometry is not None:
        geometry_transition = None
        match polyhedral_geometry.exact_source:
            case None:
                geometry = CellGeometrySpec.affine(mesh)
            case (
                ExactPowerCellGeometrySource()
                | ExactPowerCellGeometryRestrictionSource()
                | ExactPowerCellGeometryLinearActionSource() as power_source
            ):
                geometry = CellGeometrySpec.power(mesh, power_source)
            case ExactPlcCellGeometrySource() | ExactPlcCellGeometryConvexSource():
                raise ValueError(
                    "Polyhedral adaptation cannot republish an exact PLC source on polyhedra."
                )
            case invalid:
                assert_never(invalid)
    else:
        geometry_transition = (
            None
            if coefficient_geometry is not None
            else _geometry_transition(prepared, edit, mesh)
        )
        geometry = (
            coefficient_geometry
            if coefficient_geometry is not None
            else CellGeometrySpec.affine(mesh)
            if geometry_transition is None
            else geometry_transition.geometry
        )
    if geometry_transition is not None:
        mesh = mesh.with_coordinates(
            geometry_transition.vertex_coordinates, numeric_version=numeric_version
        )
        geometry = geometry_transition.geometry
    patches, zones, labels = inherit_mesh_organization(source, mesh, lineage)
    if prepared.policy.route is MeshAdaptationRoute.NATIVE_LEVEL_SET:
        from ._level_set import level_set_organization

        patches, zones, labels = level_set_organization(
            prepared, mesh, patches, zones, labels, edit
        )
    region_boundary_evidence = _region_boundary_transition(
        source, mesh, lineage, patches, zones, labels
    )
    region_evidence = None
    if source.region_evidence is not None:
        from ._compartments import revalidate_region_evidence

        renewal = revalidate_region_evidence(
            source,
            mesh,
            geometry,
            zones,
            patches,
            lineage=lineage,
            common_refinement=common_refinement,
            limits=prepared.policy.limits,
        )
        zones, patches, region_evidence = (
            renewal.zones,
            renewal.patches,
            renewal.region_evidence,
        )
    certification_inputs = _adaptation_certification_inputs(
        source, prepared.policy.limits
    )
    certification_cell_regions: np.ndarray | None = None
    if (
        certification_inputs is not None
        and certification_inputs.cell_regions is not None
        and region_evidence is not None
    ):
        domain = certification_inputs.domain
        if domain is None or domain.domain_id != region_evidence.domain.domain_id:
            raise ValueError(
                "Region renewal and certification must bind the same authoritative domain."
            )
        region_rows = {region: row for row, region in enumerate(domain.region_ids)}
        certification_cell_regions = np.asarray(
            [region_rows[region] for region in region_evidence.cell_region_ids],
            dtype=np.int64,
        )
    if polyhedral_construction is not None:
        certification_cell_regions = polyhedral_construction.cell_regions
    association_transfer = prepared.policy.association_transfer
    associations: tuple[GeometryAssociation, ...] = ()
    certification_prepared = None
    certification_prepared_embedding = metric_prepared_embedding
    if polyhedral_construction is not None and source.associations:
        if not isinstance(prepared.request, PolyhedralMeshAdaptation):
            raise TypeError(
                "Construction-derived associations require a polyhedral request."
            )
        associations = regenerated_polyhedral_associations(
            source,
            prepared.request,
            polyhedral_construction,
            mesh,
        )
    elif source.associations:
        match association_transfer:
            case BRepAssociationTransfer():
                associations = association_transfer.propagate(source, lineage, mesh)
            case PlcAssociationTransfer() | ComposedAssociationTransfer():
                from ._certification import MeshCertificationPreparedEvidence

                target_evidence: list[MeshCertificationPreparedEvidence] | None = None
                target_request = None
                if (
                    isinstance(association_transfer, ComposedAssociationTransfer)
                    or association_transfer.domain.ambient_dimension == 3
                ) and certification_inputs is not None:
                    target_request = certification_inputs._transition_request(
                        mesh,
                        geometry,
                        lineage=lineage,
                        cell_regions=certification_cell_regions,
                    )
                    target_evidence = []
                associations = association_transfer.propagate(
                    source,
                    lineage,
                    mesh,
                    geometry=geometry,
                    _certification_request=target_request,
                    _prepared_target=target_evidence,
                )
                if target_evidence is not None:
                    if len(target_evidence) != 1:
                        raise ValueError(
                            "PLC propagation must publish one coherent target proof."
                        )
                    certification_prepared = target_evidence[0]
            case SurfaceAssociationTransfer():
                associations = association_transfer.propagate(
                    source,
                    lineage,
                    mesh,
                    geometry=geometry,
                    deformation=None
                    if geometry_transition is None
                    else geometry_transition.chart_deformation,
                )
            case (
                MappedReferenceAssociationTransfer()
                | ImplicitAssociationTransfer()
                | PeriodicAssociationTransfer()
            ):
                associations = association_transfer.propagate(
                    source, lineage, mesh, geometry=geometry
                )
            case None:
                raise ValueError("Associated native source requires an owning transfer.")
            case invalid:
                assert_never(invalid)
    attributes: tuple[MeshAttribute, ...] = ()
    if source.attributes:
        from ._layer_core import remap_layer_index_attribute

        attributes = remap_layer_index_attribute(source, lineage, mesh)
    if (
        isinstance(association_transfer, SurfaceAssociationTransfer)
        and certification_inputs is not None
    ):
        from ._surface_association_transfer import successor_certification_inputs

        certification_inputs = successor_certification_inputs(
            association_transfer.support,
            source,
            certification_inputs,
            lineage,
            mesh,
            geometry,
            deformation=None
            if geometry_transition is None
            else geometry_transition.chart_deformation,
        )
    boundary = None
    if source.boundary is not None:
        if not isinstance(association_transfer, ImplicitAssociationTransfer):
            raise ValueError(
                "Native boundary publication lacks its owning source-model remap."
            )
        boundary = association_transfer.remap_boundary(
            source, lineage, mesh, geometry=geometry
        )
    # Supplying the geometry also refuses a successor canonicalization would reorder.
    target = certify_cell_mesh(
        mesh,
        source.coordinate_contract,
        geometry=geometry,
        audit_policy=(
            prepared.policy.audit_policy
            if certification_inputs is None
            else _required_audit_policy(
                prepared.policy.audit_policy,
                certification_inputs.schedule.required_audit_checks,
            )
        ),
        boundary=boundary,
        patches=patches,
        zones=zones,
        labels=labels,
        attributes=attributes,
        associations=associations,
        region_evidence=region_evidence,
        region_boundary_evidence=region_boundary_evidence,
        certification_inputs=certification_inputs,
        lineage=None if certification_inputs is None else lineage,
        certification_cell_regions=certification_cell_regions,
        certification_prepared=certification_prepared,
        certification_prepared_embedding=certification_prepared_embedding,
        certification_prepared_fidelity=metric_prepared_fidelity,
        audit_prepared_validity=audit_prepared_validity,
        surface_source=source.surface_source,
    )
    transition = CellMeshTransition(
        source.mesh.mesh_id,
        source.mesh.topology_id,
        target,
        lineage,
        kind,
        vertex_stencil=stencil,
        geometry_transition=geometry_transition,
        parents=edit.refinement,
        coarsening=_complete_coarsening_witnesses(
            source.mesh, target.mesh, lineage, edit.coarsening
        ),
    )
    affine = geometry_transition is None
    source_measures = None
    target_measures = None
    if conservative:
        if stencil is None:
            raise ValueError(
                "Conservative vertex transfer requires an exact vertex stencil."
            )
        if affine:
            source_measures = _p1_vertex_measures(source.mesh)
            target_measures = _p1_vertex_measures(target.mesh)
        else:
            source_measures = cell_geometry_vertex_measures(source.mesh, source.geometry)
            target_measures = cell_geometry_vertex_measures(target.mesh, target.geometry)
    transfer = (
        None
        if stencil is None
        else stencil.as_transfer(
            source.mesh.vertex_global_ids,
            source_topology_id=source.mesh.topology_id,
            target_topology_id=target.mesh.topology_id,
            preserves_linear=affine,
            conservative=conservative,
            source_coordinates=source.mesh.coordinates,
            target_coordinates=target.mesh.coordinates,
            source_measures=source_measures,
            target_measures=target_measures,
        )
    )
    return _NativeTarget(target, transition, lineage, stencil, transfer)


def _execute_bisection_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    from .._meshcore import current_native_execution_budget, NativeExecutionBudget
    from ._measurements import NativeExecutionRecord

    if current_native_execution_budget() is not None:
        return _run_bisection_route(prepared)
    limits = prepared.policy.limits
    with NativeExecutionBudget(
        max_work=limits.maximum_work_units,
        max_geometry_queries=limits.maximum_geometry_queries,
        max_cavity_cells=limits.maximum_cavity_cells,
        max_scratch_bytes=limits.maximum_scratch_bytes,
        max_wall_seconds=limits.maximum_wall_seconds,
    ) as budget:
        with budget.host_workspace():
            outcome = _run_bisection_route(prepared)
    if budget.evidence is None:
        raise RuntimeError(
            "Native bisection publication requires its actual ended execution scope."
        )
    target = outcome.target.with_execution_evidence(
        NativeExecutionRecord(budget.evidence, owner_id=prepared.prepared_id),
    )
    return outcome._replace(target=target)


def _run_bisection_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    if prepared.source.mesh.periodic_topology is not None:
        from ._periodic import _execute_periodic_bisection_route

        return _execute_periodic_bisection_route(prepared)
    request = prepared.request
    if not isinstance(request, MarkedMeshAdaptation) or not isinstance(
        request.hierarchy, (BisectionHierarchy, type(None))
    ):
        raise TypeError("A bisection route requires a marked bisection request.")
    policy = prepared.policy
    constraints = prepared.constraints
    outcome = execute_bisection(
        prepared.source.mesh,
        np.asarray(request.refine_cell_ids, dtype=np.int64),
        np.asarray(request.coarsen_cell_ids, dtype=np.int64),
        source_result=prepared.source,
        hierarchy=request.hierarchy,
        compatibility=policy.compatibility,
        protected_edges=constraints.protected_edge_keys,
        protected_vertices=constraints.protected_vertex_ids,
        cell_classes=constraints.cell_classes,
        facet_classes=constraints.facet_classes,
        maximum_closure_iterations=policy.maximum_closure_iterations,
        maximum_cavity_cells=policy.limits.maximum_cavity_cells,
        maximum_cells=policy.limits.maximum_cells,
        maximum_vertices=policy.limits.maximum_vertices,
        maximum_work_units=policy.limits.maximum_work_units,
        maximum_scratch_bytes=policy.limits.maximum_scratch_bytes,
        maximum_wall_seconds=policy.limits.maximum_wall_seconds,
        maximum_geometry_queries=policy.limits.maximum_geometry_queries,
    )
    evidence = outcome.evidence
    refined = evidence.bisections > 0 or evidence.uniform_refinement_applied
    coarsened = evidence.coarsened_vertices > 0
    partial = (
        np.asarray(evidence.rejected_refinement_ids).size > 0
        or np.asarray(evidence.rejected_coarsening_ids).size > 0
    )
    if not refined and not coarsened:
        return _unchanged(
            prepared,
            evidence,
            outcome.hierarchy,
            status=MeshAdaptationStatus.PARTIAL
            if partial
            else MeshAdaptationStatus.UNCHANGED,
        )
    match (refined, coarsened):
        case (True, False):
            kind = MeshTransitionKind.REFINE
        case (False, True):
            kind = MeshTransitionKind.COARSEN
        case _:
            kind = MeshTransitionKind.REMESH
    native = _finalize_native(
        prepared,
        outcome.edit,
        kind,
        conservative=not coarsened,
        bisection_hierarchy=outcome.hierarchy,
    )
    from ._bisection import _rebind_bisection_presentation_blocks

    hierarchy = _rebind_bisection_presentation_blocks(outcome.hierarchy, native.target)
    return _RouteOutcome(
        MeshAdaptationStatus.PARTIAL if partial else MeshAdaptationStatus.COMPLETE,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        None,
        evidence,
        hierarchy,
    )


def _target_metric(
    metric: MeshMetricField, target: CellMesh, values: np.ndarray, /
) -> MeshMetricField:
    vertices = target.entity_set(0)
    identifiers = np.asarray(target.vertex_global_ids, dtype=np.int64)
    order = np.argsort(identifiers, kind="stable")
    return MeshMetricField(
        MeshingScope(
            target.mesh_id,
            target.numeric_version,
            MeshingEntityKind.MESH,
            0,
            vertices.entity_set_id,
            identifiers,
        ),
        np.asarray(values, dtype=np.float64)[order],
        minimum_size=metric.minimum_size,
        maximum_size=metric.maximum_size,
        maximum_anisotropy=metric.maximum_anisotropy,
    )


def _metric_status(
    evidence: LocalMetricEvidence | DeviceMetricEvidence, topology: bool, /
) -> MeshAdaptationStatus:
    if not topology:
        # Relocation reaches its fixed point when a pass moves nothing.
        return (
            MeshAdaptationStatus.COMPLETE
            if evidence.converged or evidence.stalled
            else MeshAdaptationStatus.PASS_LIMIT
        )
    if evidence.converged:
        return MeshAdaptationStatus.COMPLETE
    if evidence.stalled:
        return MeshAdaptationStatus.STALLED
    return MeshAdaptationStatus.PASS_LIMIT


def _execute_metric_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    request = prepared.request
    if not isinstance(request, (MetricMeshAdaptation, RelocationMeshAdaptation)):
        raise TypeError("A native metric route requires a metric or relocation request.")
    policy = prepared.policy
    constraints = prepared.constraints
    topology = isinstance(request, MetricMeshAdaptation)
    outcome = execute_local_metric_adaptation(
        prepared.source.mesh,
        constraints.metric_values,
        cell_classes=constraints.cell_classes,
        edge_classes=constraints.edge_classes,
        protected_edges=constraints.protected_edge_mask,
        fixed_vertices=constraints.fixed_vertex_mask,
        predicate_mode=resolve_host_predicate_mode(policy.predicate_mode),
        maximum_passes=policy.maximum_passes,
        topology_operations=topology,
        relocation=policy.relocation,
    )
    evidence = outcome.evidence
    status = _metric_status(evidence, topology)
    applied = evidence.splits + evidence.collapses + evidence.flips + evidence.relocations
    if applied == 0:
        # Preserve the exact source while retaining whether the request was
        # already satisfied, stalled, or exhausted its pass budget.
        return _unchanged(prepared, evidence, None, status=status)
    native = _finalize_native(
        prepared, outcome.edit, MeshTransitionKind.REMESH, conservative=False
    )
    return _RouteOutcome(
        status,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        _target_metric(request.metric, native.target.mesh, outcome.metric),
        evidence,
        None,
    )


def _execute_tetra_metric_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    from .._meshcore import current_native_execution_budget
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET
    from ._volume_generation import native_volume_execution_budget

    if current_native_execution_budget() is None:
        with native_volume_execution_budget(
            prepared.policy.limits,
            stage=MeshingStageKind.OPTIMIZATION,
        ) as budget:
            outcome = _execute_tetra_metric_route(prepared)
        if budget.evidence is None or not isinstance(
            outcome.evidence, MetricRemeshingEvidence
        ):
            raise RuntimeError(
                "A tetrahedral metric epoch lacks its actual ended execution evidence."
            )
        record = NativeExecutionRecord(budget.evidence, owner_id=prepared.prepared_id)
        evidence = eqx.tree_at(
            lambda value: value.execution_evidence,
            outcome.evidence,
            record,
            is_leaf=lambda value: value is None,
        )
        return outcome._replace(evidence=evidence)
    if _COORDINATE_BUDGET.get() is None:
        limits = prepared.policy.limits
        ledger = CoordinateEnclosureBudget(
            limits.maximum_work_units, limits.maximum_scratch_bytes
        )
        try:
            with ledger.activate():
                outcome = _execute_tetra_metric_route(prepared)
                ledger.charge_native_work(
                    ledger.work_units - ledger.native_charged_work_units
                )
                return outcome
        except CoordinateEnclosureResourceError as error:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Exact tetrahedral source preparation exhausted the original operation allowance.",
                stage=MeshingStageKind.OPTIMIZATION.value,
                requested=(
                    ("maximum_work_units", limits.maximum_work_units),
                    ("maximum_scratch_bytes", limits.maximum_scratch_bytes),
                ),
                achieved=(
                    ("coordinate_work_units", ledger.work_units),
                    ("coordinate_peak_bytes_upper", ledger.peak_bytes_upper),
                ),
            ) from error
    request = prepared.request
    if not isinstance(request, (MetricMeshAdaptation, RelocationMeshAdaptation)):
        raise TypeError("A native metric route requires a metric or relocation request.")
    policy = prepared.policy
    constraints = prepared.constraints
    outcome = execute_tetra_metric_adaptation(
        prepared.source.mesh,
        constraints.metric_values,
        source_geometry=prepared.source.geometry,
        cell_classes=constraints.cell_classes,
        facet_classes=constraints.facet_classes,
        edge_classes=constraints.edge_classes,
        protected_edges=constraints.protected_edge_mask,
        fixed_vertices=constraints.fixed_vertex_mask,
        maximum_passes=policy.maximum_passes,
        topology_operations=isinstance(request, MetricMeshAdaptation),
        relocation=policy.relocation,
        maximum_vertices=policy.limits.maximum_vertices,
        maximum_cells=policy.limits.maximum_cells,
        maximum_operations=policy.limits.maximum_work_units,
        maximum_work_units=policy.limits.maximum_work_units,
        maximum_location_pairs=policy.limits.maximum_geometry_queries,
        maximum_cavity_cells=policy.limits.maximum_cavity_cells,
        maximum_scratch_bytes=policy.limits.maximum_scratch_bytes,
    )
    evidence = outcome.evidence
    match evidence.status:
        case MetricRemeshingStatus.COMPLETE:
            status = MeshAdaptationStatus.COMPLETE
        case MetricRemeshingStatus.STALLED:
            status = MeshAdaptationStatus.STALLED
        case MetricRemeshingStatus.PASS_LIMIT:
            status = MeshAdaptationStatus.PASS_LIMIT
        case MetricRemeshingStatus.RESOURCE_LIMIT:
            status = MeshAdaptationStatus.RESOURCE_LIMIT
        case invalid:
            assert_never(invalid)
    applied = evidence.splits + evidence.collapses + evidence.flips + evidence.relocations
    if applied == 0:
        return _unchanged(prepared, evidence, None, status=status)
    native = _finalize_native(
        prepared,
        outcome.edit,
        MeshTransitionKind.REMESH,
        conservative=False,
        plc_source=outcome.exact_source,
    )
    return _RouteOutcome(
        status,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        _target_metric(request.metric, native.target.mesh, outcome.metric),
        evidence,
        None,
    )


def _surface_fidelity_limit(source: CellMeshingResult, /) -> float:
    certificate = source.certification
    if certificate is None or certificate.request.fidelity_tolerance is None:
        raise ValueError(
            "Surface adaptation must retain its original continuous fidelity bound."
        )
    return certificate.request.fidelity_tolerance


def _surface_material_epoch(
    source: CellMeshingResult,
    witness: SurfaceChartWitness,
    /,
) -> tuple[NDArray[np.object_] | None, SurfaceChartWitness]:
    """Retain exact material anchors and the original admitted chart authority."""
    from ..discretization._surface_chart_deformation import (
        PreparedSurfaceChartDeformation,
    )
    from ._surface_association_transfer import SurfaceChartBoundarySource

    retained = (
        source.certification.request.source if source.certification is not None else None
    )
    if isinstance(retained, SurfaceChartBoundarySource) and isinstance(
        retained.deformation, PreparedSurfaceChartDeformation
    ):
        retained.require_current()
        retained.deformation.require_bound(
            retained.deformation.source_mesh,
            retained.deformation.source_geometry,
            source.mesh,
            source.geometry,
        )
        return (
            retained.deformation.material_charts(target=True),
            retained.deformation.material_root_witness,
        )
    return None, witness


def _execute_surface_metric_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    from ..discretization._sphere_chart_deformation import PreparedSphereChartDeformation
    from ..geometry._sphere_material_atlas import sphere_material_atlas_from_result
    from ..geometry.brep._patches import sphere_source_radius_terms
    from ..geometry.brep._placed import PlacedSurface
    from ._surface_association_transfer import (
        prepare_surface_curve_witness,
        SurfaceChartBoundarySource,
    )
    from ._surface_metric import (
        execute_sphere_metric_adaptation,
        execute_surface_metric_adaptation,
        prepare_surface_metric_source,
    )
    from ._surface_metric_controller import _SurfaceMetricController

    request, policy = prepared.request, prepared.policy
    transfer = policy.association_transfer
    if not isinstance(
        request, (MetricMeshAdaptation, RelocationMeshAdaptation)
    ) or not isinstance(transfer, SurfaceAssociationTransfer):
        raise TypeError(
            "The surface metric route requires its admitted request and source support."
        )
    source = prepared.source
    domain = transfer.support.domain
    patches = np.unique(transfer.classes(source)[2].indices)
    sphere = True
    for patch in patches:
        surface = domain.patches[int(patch)].surface
        operation = surface.definition if isinstance(surface, PlacedSurface) else surface
        if sphere_source_radius_terms(operation) is None:
            sphere = False
            break
    constraints = prepared.constraints
    coordinate_budget = CoordinateEnclosureBudget(
        policy.limits.maximum_work_units, policy.limits.maximum_scratch_bytes
    )
    refusal: (
        MeshcoreError
        | CoordinateEnclosureResourceError
        | SurfaceChartResourceError
        | None
    ) = None
    outcome: _RouteOutcome | None = None
    curve_witness: PreparedSurfaceCurveWitness | None = None
    with coordinate_budget.activate():
        try:
            if sphere:
                retained = (
                    source.certification.request.source
                    if source.certification is not None
                    else None
                )
                if isinstance(retained, SurfaceChartBoundarySource) and isinstance(
                    retained.deformation, PreparedSphereChartDeformation
                ):
                    retained.require_current()
                    atlas = retained.deformation.target_atlas
                else:
                    atlas = sphere_material_atlas_from_result(
                        domain,
                        source,
                        maximum_fidelity=_surface_fidelity_limit(source),
                        maximum_cells=policy.limits.maximum_cells,
                        maximum_work_units=policy.limits.maximum_work_units,
                        maximum_coordinate_memory_bytes=policy.limits.maximum_scratch_bytes,
                        coordinate_budget=coordinate_budget,
                    )
                produced = execute_sphere_metric_adaptation(
                    source,
                    constraints.metric_values,
                    domain,
                    atlas,
                    transfer,
                    policy=policy,
                    coordinate_contract=source.coordinate_contract,
                    maximum_fidelity=_surface_fidelity_limit(source),
                    topology_operations=isinstance(request, MetricMeshAdaptation),
                    relocation=policy.relocation,
                    coordinate_budget=coordinate_budget,
                )
            else:
                witness = prepare_surface_metric_source(source, transfer)
                material_charts, material_root_witness = _surface_material_epoch(
                    source, witness
                )
                curve_witness = prepare_surface_curve_witness(transfer.support, source)
                controller = (
                    _SurfaceMetricController(
                        source,
                        transfer,
                        policy,
                        coordinate_budget,
                        witness,
                        _surface_fidelity_limit(source),
                        material_root_witness,
                    )
                    if source.mesh.periodic_topology is not None
                    else None
                )
                produced = execute_surface_metric_adaptation(
                    source.mesh,
                    constraints.metric_values,
                    domain,
                    source_id=domain.source_id,
                    source_revision=domain.source_revision,
                    cell_patches=witness.patches,
                    cell_charts=witness.charts,
                    cell_classes=constraints.cell_classes,
                    edge_classes=constraints.edge_classes,
                    protected_edges=constraints.protected_edge_mask,
                    fixed_vertices=constraints.fixed_vertex_mask,
                    maximum_fidelity=_surface_fidelity_limit(source),
                    maximum_passes=policy.maximum_passes,
                    topology_operations=isinstance(request, MetricMeshAdaptation),
                    relocation=policy.relocation,
                    maximum_vertices=policy.limits.maximum_vertices,
                    maximum_cells=policy.limits.maximum_cells,
                    maximum_operations=policy.limits.maximum_work_units,
                    maximum_arc_work_units=policy.limits.maximum_work_units,
                    maximum_location_pairs=policy.limits.maximum_geometry_queries,
                    controller=controller,
                    source_material_charts=material_charts,
                    curve_witness=curve_witness,
                )
            evidence = produced.evidence
            status = MeshAdaptationStatus(evidence.status.value)
            if (
                evidence.status is MetricRemeshingStatus.RESOURCE_LIMIT
                or evidence.splits
                + evidence.collapses
                + evidence.flips
                + evidence.relocations
                == 0
            ):
                outcome = _unchanged(prepared, evidence, None, status=status)
            else:
                native = _finalize_native(
                    prepared,
                    produced.edit,
                    MeshTransitionKind.REMESH,
                    conservative=False,
                    surface_metric_outcome=produced,
                    surface_curve_witness=curve_witness,
                )
                outcome = _RouteOutcome(
                    status,
                    native.target,
                    native.transition,
                    native.lineage,
                    native.stencil,
                    native.transfer,
                    _target_metric(request.metric, native.target.mesh, produced.metric),
                    evidence,
                    None,
                )
        except (
            MeshcoreError,
            CoordinateEnclosureResourceError,
            SurfaceChartResourceError,
        ) as error:
            if (
                isinstance(error, MeshcoreError)
                and error.status not in _NATIVE_RESOURCE_STATUSES
            ):
                _charge_remaining_after_failure(coordinate_budget, error)
                raise
            refusal = error
        except Exception as error:
            _charge_remaining_after_failure(coordinate_budget, error)
            raise
        # Actual host visits are charged once to the original native scope. An
        # overrun at this final debit refuses even a staged successor.
        try:
            coordinate_budget.charge_native_work(
                coordinate_budget.work_units - coordinate_budget.native_charged_work_units
            )
        except MeshcoreError as error:
            if error.status not in _NATIVE_RESOURCE_STATUSES:
                raise
            refusal = error if refusal is None else refusal
    if refusal is not None:
        return _unchanged(
            prepared,
            _surface_resource_refusal(prepared, coordinate_budget, refusal),
            None,
            status=MeshAdaptationStatus.RESOURCE_LIMIT,
        )
    if outcome is None:
        raise RuntimeError(
            "The surface metric route ended without an outcome or an actual refusal."
        )
    return outcome


_NATIVE_RESOURCE_STATUSES = (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT)


def _charge_remaining_after_failure(
    budget: CoordinateEnclosureBudget, error: Exception, /
) -> None:
    """Debit actual visits before a genuine failure propagates; never mask it."""
    try:
        budget.charge_native_work(budget.work_units - budget.native_charged_work_units)
    except MeshcoreError as charge_error:
        error.add_note(f"Remaining coordinate work was not charged: {charge_error}")


def _surface_resource_refusal(
    prepared: PreparedMeshAdaptation,
    budget: CoordinateEnclosureBudget,
    error: MeshcoreError | CoordinateEnclosureResourceError | SurfaceChartResourceError,
    /,
) -> MetricRemeshingEvidence:
    """Bounded RESOURCE_LIMIT evidence; the accepted source remains the target."""
    certification = prepared.source.certification
    if certification is None or certification.fidelity is None:
        raise ValueError(
            "A refused surface epoch must retain its accepted source fidelity."
        )
    limits = prepared.policy.limits
    requested = (
        ("maximum_work_units", float(limits.maximum_work_units)),
        ("maximum_scratch_bytes", float(limits.maximum_scratch_bytes)),
    )
    achieved = (
        ("coordinate_work_units", float(budget.work_units)),
        ("native_charged_coordinate_work_units", float(budget.native_charged_work_units)),
        ("coordinate_peak_bytes_upper", float(budget.peak_bytes_upper)),
    )
    if isinstance(error, SurfaceChartResourceError):
        requested += tuple(
            (name, float(limit)) for name, _, limit in error.resource_counts
        )
        achieved += tuple(
            (name, float(completed)) for name, completed, _ in error.resource_counts
        )
    return MetricRemeshingEvidence(
        MetricRemeshingStatus.RESOURCE_LIMIT,
        passes=0,
        counts=(0, 0, 0, 0),
        rejected_operations=0,
        work_units=budget.work_units,
        lengths=np.zeros((0,), dtype=np.float64),
        quality=np.zeros((0,), dtype=np.float64),
        maximum_fidelity_bound=certification.fidelity.mesh_to_source_upper,
        criterion=(
            MetricRemeshingCriterion.UNIT_MESH
            if isinstance(prepared.request, MetricMeshAdaptation)
            else MetricRemeshingCriterion.RELOCATION_FIXED_POINT
        ),
        native_work_units=0,
        resource_message=str(error),
        resource_requested=requested,
        resource_achieved=achieved,
    )


def _prepare_mixed_route(prepared: PreparedMeshAdaptation, /) -> MixedAdaptationOutcome:
    request = prepared.request
    if not isinstance(request, MarkedMeshAdaptation) or not isinstance(
        request.hierarchy, (MixedAdaptationHierarchy, type(None))
    ):
        raise TypeError("NATIVE_MIXED requires marked mixed-template adaptation.")
    source = prepared.source.mesh
    rows = key_rows(
        entity_keys(source, source.topological_dimension), _cell_ids(source)[:, None]
    )
    cell_ids = np.asarray(
        source.entity_set(source.topological_dimension).entity_ids, dtype=np.int64
    )
    protected_cells = [
        cell_ids[_selected_rows(source, scope)]
        for scope in prepared.policy.protected_scopes
        if scope.entity_dimension == source.topological_dimension
    ]
    protected_cell_ids = (
        np.unique(np.concatenate(protected_cells))
        if protected_cells
        else np.zeros((0,), dtype=np.int64)
    )
    protected_face_keys = np.zeros((0, 0), dtype=np.int64)
    if source.topological_dimension == 3:
        face_keys = entity_keys(source, 2)
        protected_faces = np.zeros((face_keys.shape[0],), dtype=np.bool_)
        for scope in prepared.policy.protected_scopes:
            if scope.entity_dimension >= 2:
                face_rows = _closure_rows(
                    source, scope.entity_dimension, _selected_rows(source, scope), 2
                )
                protected_faces[face_rows] = True
        protected_face_keys = face_keys[protected_faces]
    outcome = adapt_mixed_mesh(
        source,
        refine_cell_ids=request.refine_cell_ids,
        coarsen_cell_ids=request.coarsen_cell_ids,
        hierarchy=request.hierarchy,
        cell_classes=prepared.constraints.cell_classes[rows],
        protected_vertex_ids=prepared.constraints.protected_vertex_ids,
        protected_cell_ids=protected_cell_ids,
        protected_edge_keys=prepared.constraints.protected_edge_keys,
        protected_face_keys=protected_face_keys,
        layer_columns=request.layer_columns,
        maximum_cells=prepared.policy.limits.maximum_cells,
        maximum_closure_steps=prepared.policy.maximum_closure_iterations,
        maximum_work_units=prepared.policy.limits.maximum_work_units,
        maximum_scratch_bytes=prepared.policy.limits.maximum_scratch_bytes,
    )
    return outcome


def _finalize_mixed_route(
    prepared: PreparedMeshAdaptation, outcome: MixedAdaptationOutcome, /
) -> _RouteOutcome:
    evidence = outcome.evidence
    partial = evidence.rejected_coarsening_ids.size > 0
    refined = evidence.refined_cell_ids.size > 0
    coarsened = evidence.coarsened_cell_ids.size > 0
    if not refined and not coarsened:
        return _unchanged(
            prepared,
            evidence,
            outcome.hierarchy,
            status=MeshAdaptationStatus.PARTIAL
            if partial
            else MeshAdaptationStatus.UNCHANGED,
        )
    match (refined, coarsened):
        case (True, False):
            kind = MeshTransitionKind.REFINE
        case (False, True):
            kind = MeshTransitionKind.COARSEN
        case _:
            kind = MeshTransitionKind.REMESH
    native = _finalize_native(prepared, outcome.edit, kind, conservative=False)
    return _RouteOutcome(
        MeshAdaptationStatus.PARTIAL if partial else MeshAdaptationStatus.COMPLETE,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        None,
        evidence,
        outcome.hierarchy,
    )


def _polyhedral_touched_cells(
    source: CellMeshingResult,
    request: PolyhedralMeshAdaptation,
    /,
) -> frozenset[int]:
    """Actual original-cell workset of a host split/agglomeration proposal."""
    from fractions import Fraction

    if request.operation is PolyhedralAdaptationOperation.AGGLOMERATE:
        return frozenset(
            identifier for group in request.agglomerations for identifier in group
        )
    if request.operation is not PolyhedralAdaptationOperation.PLANE_SPLIT:
        return frozenset()
    coordinates = source.geometry.source_coordinates()
    planes = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(request.planes)
    )
    touched = set()
    for block in source.mesh.blocks:
        vertices, valid = np.asarray(block.vertices), np.asarray(block.vertex_valid)
        identifiers = np.asarray(block.global_ids).tolist()
        for cell in range(block.cell_count):
            row = vertices[cell : cell + 1].reshape((-1,))
            mask = valid[cell : cell + 1].reshape((-1,))
            for plane in planes:
                sides = tuple(
                    sum(
                        (
                            plane[axis] * coordinates[int(vertex)][axis]
                            for axis in range(3)
                        ),
                        -plane[3],
                    )
                    for vertex in row[mask]
                )
                if min(sides) < 0 < max(sides):
                    touched.add(identifiers[cell])
                    break
    return frozenset(touched)


def _execute_polyhedral_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    request = prepared.request
    if not isinstance(request, PolyhedralMeshAdaptation):
        raise TypeError("NATIVE_POLYHEDRAL requires PolyhedralMeshAdaptation.")
    from .._meshcore import current_native_execution_budget
    from ._volume_generation import native_volume_execution_budget

    execution = current_native_execution_budget()
    if execution is None:
        with native_volume_execution_budget(prepared.policy.limits):
            return _execute_polyhedral_route(prepared)
    touched = _polyhedral_touched_cells(prepared.source, request)
    execution.admit_cavity(len(touched))
    source = prepared.source.mesh
    rows = key_rows(entity_keys(source, 3), _cell_ids(source)[:, None])
    face_keys = entity_keys(source, 2)
    protected_faces = np.any(
        np.isin(face_keys, prepared.constraints.protected_vertex_ids), axis=1
    )
    if request.operation is PolyhedralAdaptationOperation.AGGLOMERATE:
        protected_faces |= np.any(
            _membership(source, _organization_scopes(prepared.source), 2), axis=1
        )
    limits = prepared.policy.limits
    outcome = adapt_polyhedral_mesh(
        source,
        request,
        cell_classes=prepared.constraints.cell_classes[rows],
        source_geometry=prepared.source.geometry,
        protected_face_ids=np.asarray(source.entity_set(2).entity_ids)[protected_faces],
        maximum_cells=limits.maximum_cells,
        maximum_vertices=limits.maximum_vertices,
        maximum_work_units=limits.maximum_work_units,
        maximum_scratch_bytes=limits.maximum_scratch_bytes,
        source_id=(
            prepared.source.associations[0].source_id
            if prepared.source.associations
            else "native-polyhedral"
        ),
        numeric_version=f"adaptation:{prepared.prepared_id}",
        common_refinement_policy=CommonRefinementPolicy(
            predicate_mode=resolve_host_predicate_mode(prepared.policy.predicate_mode),
            maximum_candidate_pairs=limits.maximum_geometry_queries,
            maximum_accepted_pairs=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
            maximum_exact_work=limits.maximum_work_units,
            overlap_simplices=True,
        ),
    )
    edit = outcome.edit
    if outcome.construction is not None and prepared.source.associations:
        candidate, _, _ = assemble_topology_edit(
            source,
            edit,
            numeric_version=f"adaptation:{prepared.prepared_id}",
        )
        associations = regenerated_polyhedral_associations(
            prepared.source,
            request,
            outcome.construction,
            candidate,
        )
        edit = regenerated_polyhedral_relations(
            prepared.source,
            candidate,
            edit,
            associations,
            limits.maximum_work_units,
            maximum_geometry_queries=limits.maximum_geometry_queries,
            target_geometry=outcome.geometry,
        )
    match request.operation:
        case PolyhedralAdaptationOperation.PLANE_SPLIT:
            kind = MeshTransitionKind.REFINE
        case PolyhedralAdaptationOperation.AGGLOMERATE:
            kind = MeshTransitionKind.COARSEN
        case PolyhedralAdaptationOperation.REGENERATE:
            kind = MeshTransitionKind.REMESH
        case invalid:
            assert_never(invalid)
    native = _finalize_native(
        prepared,
        edit,
        kind,
        conservative=False,
        common_refinement=outcome.common_refinement,
        polyhedral_construction=outcome.construction,
        polyhedral_geometry=outcome.geometry,
    )
    return _RouteOutcome(
        MeshAdaptationStatus.COMPLETE,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        None,
        outcome.evidence,
        None,
        outcome.common_refinement,
    )


def _unknown_lineage(source: CellMesh, target: CellMesh, /) -> MeshLineage:
    """Provider remeshing: every target entity created, every source entity deleted."""

    records = []
    for dimension in range(source.topological_dimension + 1):
        source_entities = source.entity_set(dimension)
        target_entities = target.entity_set(dimension)
        empty = np.zeros((0,), dtype=np.int64)
        records.append(
            EntityLineage(
                dimension,
                source_entities.entity_set_id,
                target_entities.entity_set_id,
                empty,
                empty,
                np.zeros((0,), dtype=np.int32),
                created_target_ids=np.sort(np.asarray(target_entities.entity_ids)),
                deleted_source_ids=np.sort(np.asarray(source_entities.entity_ids)),
            )
        )
    return MeshLineage(source.topology_id, target.topology_id, tuple(records))


def _provider_outcome(
    prepared: PreparedMeshAdaptation,
    target: CellMeshingResult,
    metric: MeshMetricField | None,
    evidence: MmgAdaptationResult | OmegaHAdaptationResult,
    /,
) -> _RouteOutcome:
    source = prepared.source
    lineage = _unknown_lineage(source.mesh, target.mesh)
    transition = CellMeshTransition(
        source.mesh.mesh_id,
        source.mesh.topology_id,
        target,
        lineage,
        MeshTransitionKind.REMESH,
    )
    return _RouteOutcome(
        MeshAdaptationStatus.COMPLETE,
        target,
        transition,
        lineage,
        None,
        None,
        metric,
        evidence,
        None,
    )


def _execute_mmg_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    policy = prepared.policy
    transfer = policy.association_transfer
    if isinstance(transfer, (PlcAssociationTransfer, ComposedAssociationTransfer)):
        raise ValueError("Provider routes cannot rederive represented PLC associations.")
    if isinstance(transfer, SurfaceAssociationTransfer):
        raise ValueError(
            "Provider routes cannot rederive parametric surface associations."
        )
    if isinstance(transfer, MappedReferenceAssociationTransfer):
        raise ValueError(
            "Provider routes cannot rederive original mapped-root associations."
        )
    if isinstance(transfer, ImplicitAssociationTransfer):
        raise ValueError(
            "Provider routes cannot replace original implicit-source authority."
        )
    if isinstance(transfer, PeriodicAssociationTransfer):
        raise ValueError(
            "Provider routes cannot replace original periodic-source authority."
        )
    provider = policy.provider
    # ty: ignore[unresolved-attribute]
    plan = provider.plan(
        prepared.source,
        # ty: ignore[unresolved-attribute]
        metric=prepared.request.metric,
        required=policy.protected_scopes,
        # ty: ignore[invalid-argument-type]
        options=policy.provider_options,
        limits=policy.limits,
        audit_policy=policy.audit_policy,
        association_transfer=transfer,
    )
    # ty: ignore[invalid-argument-type, missing-argument, unresolved-attribute]
    result = provider.execute(plan)
    # ty: ignore[unresolved-attribute]
    return _provider_outcome(prepared, result.mesh, result.metric, result)


def _execute_omega_h_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    policy = prepared.policy
    transfer = policy.association_transfer
    if isinstance(transfer, (PlcAssociationTransfer, ComposedAssociationTransfer)):
        raise ValueError("Provider routes cannot rederive represented PLC associations.")
    if isinstance(transfer, SurfaceAssociationTransfer):
        raise ValueError(
            "Provider routes cannot rederive parametric surface associations."
        )
    if isinstance(transfer, MappedReferenceAssociationTransfer):
        raise ValueError(
            "Provider routes cannot rederive original mapped-root associations."
        )
    if isinstance(transfer, ImplicitAssociationTransfer):
        raise ValueError(
            "Provider routes cannot replace original implicit-source authority."
        )
    if isinstance(transfer, PeriodicAssociationTransfer):
        raise ValueError(
            "Provider routes cannot replace original periodic-source authority."
        )
    # ty: ignore[unresolved-attribute]
    result = policy.provider.execute(
        # ty: ignore[invalid-argument-type]
        prepared.source,
        # ty: ignore[too-many-positional-arguments, unresolved-attribute]
        prepared.request.metric,
        # ty: ignore[invalid-argument-type, unknown-argument]
        options=policy.provider_options,
        # ty: ignore[unknown-argument]
        limits=policy.limits,
        # ty: ignore[unknown-argument]
        association_transfer=transfer,
    )
    # ty: ignore[unresolved-attribute]
    if result.target is None:
        raise ValueError("A serial Omega_h adaptation must return its target.")
    # ty: ignore[unresolved-attribute]
    return _provider_outcome(prepared, result.target, result.metric, result)


def project_hp_lineage(
    source: FiniteElementHPEpoch,
    target: FiniteElementHPEpoch,
    lineage: FiniteElementHPLineage,
    /,
) -> MeshLineage:
    """Project fixed-capacity hp slot lineage onto the epochs' active cell meshes."""

    if not isinstance(source, FiniteElementHPEpoch) or not isinstance(
        target, FiniteElementHPEpoch
    ):
        raise TypeError("source and target must be FiniteElementHPEpoch.")
    if not isinstance(lineage, FiniteElementHPLineage):
        raise TypeError("lineage must be FiniteElementHPLineage.")
    if (
        lineage.source_topology_id != source.topology.topology_id
        or lineage.target_topology_id != target.topology.topology_id
    ):
        raise ValueError("hp lineage endpoints do not match the supplied epochs.")
    valid = np.asarray(lineage.valid, dtype=np.bool_)
    source_slots = np.asarray(lineage.source_slots, dtype=np.int32)[valid]
    target_slots = np.asarray(lineage.target_slots, dtype=np.int32)[valid]
    source_ids = np.asarray(source.topology.cell_global_ids, dtype=np.int64)[source_slots]
    target_ids = np.asarray(target.topology.cell_global_ids, dtype=np.int64)[target_slots]
    kinds = np.full(source_ids.shape, int(EntityLineageKind.PRESERVED), dtype=np.int32)
    refinement = np.asarray(lineage.relation_mask("refinement"), dtype=np.bool_)[valid]
    coarsening = np.asarray(lineage.relation_mask("coarsening"), dtype=np.bool_)[valid]
    kinds[refinement] = int(EntityLineageKind.REFINED_FROM)
    kinds[coarsening] = int(EntityLineageKind.COARSENED_INTO)
    dimension = source.mesh.topological_dimension
    entities = EntityLineage(
        dimension,
        source.mesh.entity_set(dimension).entity_set_id,
        target.mesh.entity_set(dimension).entity_set_id,
        source_ids,
        target_ids,
        kinds,
    )
    return MeshLineage(source.mesh.topology_id, target.mesh.topology_id, (entities,))


def _complete_hp_parents(epoch: FiniteElementHPEpoch, marked: np.ndarray, /) -> Any:
    """Parents whose complete active leaf family is marked for coarsening."""

    topology = epoch.topology
    identifiers = np.asarray(topology.cell_global_ids, dtype=np.int64)
    active = np.asarray(topology.active, dtype=np.bool_)
    children = np.asarray(topology.child_slots, dtype=np.int64)
    child_valid = np.asarray(topology.child_valid, dtype=np.bool_)
    safe = np.where(child_valid, children, 0)
    leaf = ~np.any(child_valid[safe], axis=2)
    family = (
        np.asarray(topology.allocated, dtype=np.bool_)
        & ~active
        & (np.count_nonzero(child_valid, axis=1) == topology.child_capacity)
        & np.all(~child_valid | (active[safe] & leaf), axis=1)
    )
    chosen = family & np.all(~child_valid | np.isin(identifiers[safe], marked), axis=1)
    return identifiers[chosen]


def _execute_hp_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    request = prepared.request
    # ty: ignore[unresolved-attribute]
    epoch = request.hierarchy
    source = prepared.source
    # ty: ignore[unresolved-attribute]
    refine = np.asarray(request.refine_cell_ids, dtype=np.int64)
    # ty: ignore[unresolved-attribute]
    coarsen = np.asarray(request.coarsen_cell_ids, dtype=np.int64)
    if refine.size:
        # ty: ignore[unresolved-attribute]
        closed, _ = balanced_hp_refinement_ids(epoch.topology, epoch.interfaces, refine)
        result = refine_tensor_hp_cells(
            # ty: ignore[unresolved-attribute]
            epoch.topology,
            # ty: ignore[unresolved-attribute]
            epoch.geometry,
            np.asarray(closed, dtype=np.int64),
        )
        kind = MeshTransitionKind.REFINE
        partial = False
    else:
        # ty: ignore[invalid-argument-type]
        parents = _complete_hp_parents(epoch, coarsen)
        if parents.size == 0:
            return _unchanged(prepared, None, epoch)
        # ty: ignore[unresolved-attribute]
        result = coarsen_tensor_hp_cells(epoch.topology, epoch.geometry, parents)
        kind = MeshTransitionKind.COARSEN
        covered = np.asarray(result.lineage.source_slots)[
            np.asarray(result.lineage.relation_mask("coarsening"), dtype=np.bool_)
        ]
        partial = covered.size != coarsen.size
    mesh = canonicalize_cell_mesh(
        hp_active_cell_mesh(
            result.topology,
            result.geometry,
            numeric_version=f"adaptation:{prepared.prepared_id}",
        )[0]
    )
    target = certify_cell_mesh(
        mesh,
        source.coordinate_contract,
        geometry=CellGeometrySpec.affine(mesh),
        audit_policy=prepared.policy.audit_policy,
    )
    target_epoch = FiniteElementHPEpoch(
        target.mesh,
        result.topology,
        result.geometry,
        finite_element_hp_interface_plan(result.topology, result.geometry),
    )
    # ty: ignore[invalid-argument-type]
    lineage = project_hp_lineage(epoch, target_epoch, result.lineage)
    transition = CellMeshTransition(
        source.mesh.mesh_id, source.mesh.topology_id, target, lineage, kind
    )
    return _RouteOutcome(
        MeshAdaptationStatus.PARTIAL if partial else MeshAdaptationStatus.COMPLETE,
        target,
        transition,
        lineage,
        None,
        None,
        None,
        result,
        target_epoch,
    )


def _compliance(
    prepared: PreparedMeshAdaptation, outcome: _RouteOutcome, elapsed: float, /
) -> MeshingComplianceReport:
    limits = prepared.policy.limits
    audit = outcome.target.audit
    observations = (
        ("vertices", audit.vertex_count, limits.maximum_vertices),
        ("cells", audit.entity_counts[-1], limits.maximum_cells),
        (
            "connectivity_entries",
            audit.connectivity_entries,
            limits.maximum_connectivity_entries,
        ),
    )
    issues = tuple(
        f"maximum_{name}" for name, actual, maximum in observations if actual > maximum
    )
    if elapsed > limits.maximum_wall_seconds:
        issues += ("maximum_wall_seconds",)
    requested = tuple(
        (f"maximum_{name}", float(maximum)) for name, _, maximum in observations
    )
    achieved = tuple((name, float(actual)) for name, actual, _ in observations)
    evidence = outcome.evidence
    if isinstance(evidence, MetricRemeshingEvidence):
        requested += (("minimum_metric_quality", 0.05),)
        achieved += (
            ("minimum_metric_quality", evidence.minimum_metric_quality),
            ("out_of_range_edges", float(evidence.out_of_range_edges)),
            ("work_units", float(evidence.work_units)),
        )
        if evidence.execution_evidence is not None:
            record = evidence.execution_evidence
            work, queries = (
                float(np.asarray(record.work)[0]),
                float(np.asarray(record.externally_charged_geometry_queries)),
            )
            requested += (
                ("maximum_work_units", float(limits.maximum_work_units)),
                ("maximum_geometry_queries", float(limits.maximum_geometry_queries)),
                ("maximum_scratch_bytes", float(limits.maximum_scratch_bytes)),
            )
            achieved += (
                ("native_scope_work_units", work),
                ("source_geometry_queries", queries),
                (
                    "native_primitive_queries",
                    float(np.asarray(record.native_primitive_queries)),
                ),
            )
            if (
                work > limits.maximum_work_units
                or queries > limits.maximum_geometry_queries
            ):
                issues += ("native_surface_cumulative_resource_limit",)
        if evidence.metric_arc_bounds is not None and isinstance(
            prepared.request, MetricMeshAdaptation
        ):
            arcs = np.asarray(evidence.metric_arc_bounds)
            requested += (("certified_metric_arcs_out_of_range", 0.0),)
            arc_failures = float(
                np.count_nonzero(
                    (arcs[:, 0] < evidence.lower_metric_length)
                    | (arcs[:, 1] > evidence.upper_metric_length)
                )
            )
            achieved += (("certified_metric_arcs_out_of_range", arc_failures),)
            if arc_failures:
                issues += ("source_metric_arc_goal",)
        if evidence.minimum_metric_quality < 0.05:
            issues += ("minimum_metric_quality",)
        if isinstance(prepared.request, MetricMeshAdaptation):
            requested += (("out_of_range_edges", 0.0),)
            if evidence.out_of_range_edges:
                issues += ("metric_unit_mesh",)
        if evidence.status is MetricRemeshingStatus.RESOURCE_LIMIT:
            issues += ("native_metric_resource_limit",)
    return MeshingComplianceReport(
        f"mesh-adaptation:{prepared.prepared_id}",
        issues=issues,
        requested=requested,
        achieved=achieved,
    )


def _adaptation_result(
    prepared: PreparedMeshAdaptation, outcome: _RouteOutcome, started: float, /
) -> MeshAdaptationResult:
    """Carry the distribution through the lineage, measure compliance, and bind."""

    policy = prepared.policy
    distribution = None
    if policy.distribution is not None and outcome.transition is not None:
        distribution = prepare_distribution_transition(
            policy.distribution,
            MeshPart(policy.distribution.part.name, outcome.target),
            # ty: ignore[invalid-argument-type]
            outcome.lineage,
            # ty: ignore[invalid-argument-type]
            policy=policy.partition_policy,
        )
    elapsed = time.monotonic() - started
    return MeshAdaptationResult(
        prepared,
        outcome,
        _compliance(prepared, outcome, elapsed),
        distribution,
        elapsed,
    )


def execute_mesh_adaptation(prepared: PreparedMeshAdaptation, /) -> MeshAdaptationResult:
    """Execute exactly the prepared route and certify its target."""

    if not isinstance(prepared, PreparedMeshAdaptation):
        raise TypeError("prepared must be PreparedMeshAdaptation.")
    started = time.monotonic()
    match prepared.policy.route:
        case MeshAdaptationRoute.NATIVE_BISECTION:
            outcome = _execute_bisection_route(prepared)
        case MeshAdaptationRoute.NATIVE_METRIC_2D:
            outcome = _execute_metric_route(prepared)
        case MeshAdaptationRoute.NATIVE_METRIC_3D:
            outcome = _execute_tetra_metric_route(prepared)
        case MeshAdaptationRoute.NATIVE_SURFACE_METRIC:
            limits = prepared.policy.limits
            budget = NativeExecutionBudget(
                max_work=limits.maximum_work_units,
                max_geometry_queries=limits.maximum_geometry_queries,
                max_cavity_cells=limits.maximum_cavity_cells,
                max_scratch_bytes=limits.maximum_scratch_bytes,
                max_wall_seconds=limits.maximum_wall_seconds,
            )
            surface_outcome: _RouteOutcome | None = None
            try:
                with budget:
                    surface_outcome = _execute_surface_metric_route(prepared)
            except MeshcoreError as error:
                if (
                    error.status not in _NATIVE_RESOURCE_STATUSES
                    or surface_outcome is None
                    or surface_outcome.status is not MeshAdaptationStatus.RESOURCE_LIMIT
                ):
                    raise
                # The route already rolled back the actual refusing stage.
                # Keep the sticky ended status and counters in its record;
                # do not turn the scope-end refusal into another exception.
            if surface_outcome is None:
                raise RuntimeError(
                    "A surface metric epoch ended without its actual outcome."
                )
            outcome = surface_outcome
            if budget.evidence is None or not isinstance(
                outcome.evidence, MetricRemeshingEvidence
            ):
                raise RuntimeError(
                    "A surface metric epoch lacks its actual ended execution evidence."
                )
            record = NativeExecutionRecord(budget.evidence, owner_id=prepared.prepared_id)
            evidence = eqx.tree_at(
                lambda value: value.execution_evidence,
                outcome.evidence,
                record,
                is_leaf=lambda value: value is None,
            )
            outcome = outcome._replace(evidence=evidence)
        case MeshAdaptationRoute.NATIVE_LEVEL_SET:
            from ._level_set import execute_level_set_route

            outcome = execute_level_set_route(prepared)
        case MeshAdaptationRoute.NATIVE_MIXED:
            limits = prepared.policy.limits
            with NativeExecutionBudget(
                max_work=limits.maximum_work_units,
                max_geometry_queries=limits.maximum_geometry_queries,
                max_cavity_cells=limits.maximum_cavity_cells,
                max_scratch_bytes=limits.maximum_scratch_bytes,
                max_wall_seconds=limits.maximum_wall_seconds,
            ) as budget:
                with budget.host_workspace():
                    produced = _prepare_mixed_route(prepared)
            if budget.evidence is None:
                raise RuntimeError(
                    "A mixed epoch lacks its actual ended original execution evidence."
                )
            produced = produced._replace(
                evidence=produced.evidence._replace(
                    execution_evidence=NativeExecutionRecord(
                        budget.evidence,
                        owner_id=prepared.prepared_id,
                    )
                )
            )
            outcome = _finalize_mixed_route(prepared, produced)
        case MeshAdaptationRoute.NATIVE_POLYHEDRAL:
            outcome = _execute_polyhedral_route(prepared)
        case MeshAdaptationRoute.NATIVE_GEOMETRY_REALIZATION:
            from ._geometry_realization import execute_geometry_realization_route

            outcome = execute_geometry_realization_route(prepared)
        case MeshAdaptationRoute.DEVICE_BISECTION:
            # Lazy: the device epoch module builds on this module's result types.
            from ._device_adaptation import (
                _execute_device_bisection_route,
                execute_partitioned_mesh_adaptation,
                prepare_partitioned_mesh_adaptation,
            )

            storage = prepared.source.mesh.storage
            if storage is not None:
                results = execute_partitioned_mesh_adaptation(
                    prepare_partitioned_mesh_adaptation(prepared),
                )
                for result in results:
                    target_storage = result.target.mesh.storage
                    if target_storage is not None and (
                        target_storage.partition_index == storage.partition_index
                    ):
                        return result
                raise RuntimeError(
                    "Distributed adaptation did not publish the source's addressable partition."
                )
            outcome = _execute_device_bisection_route(prepared)
        case MeshAdaptationRoute.DEVICE_METRIC_2D:
            from ._device_metric import _execute_device_metric_route

            outcome = _execute_device_metric_route(prepared)
        case MeshAdaptationRoute.MMG:
            outcome = _execute_mmg_route(prepared)
        case MeshAdaptationRoute.OMEGA_H:
            outcome = _execute_omega_h_route(prepared)
        case MeshAdaptationRoute.HP:
            outcome = _execute_hp_route(prepared)
        case invalid:
            assert_never(invalid)
    return _adaptation_result(prepared, outcome, started)


__all__ = [
    "GeometryRealizationMeshAdaptation",
    "LevelSetMeshAdaptation",
    "MarkedMeshAdaptation",
    "MeshAdaptationPolicy",
    "MeshAdaptationResult",
    "MeshAdaptationRoute",
    "MeshAdaptationStatus",
    "MetricMeshAdaptation",
    "PolyhedralMeshAdaptation",
    "PreparedMeshAdaptation",
    "RelocationMeshAdaptation",
    "execute_mesh_adaptation",
    "prepare_mesh_adaptation",
    "project_hp_lineage",
]
