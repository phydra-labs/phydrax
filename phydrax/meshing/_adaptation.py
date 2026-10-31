#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed mesh-adaptation requests, explicit routes, and certified transitions.

`prepare_mesh_adaptation` binds one request to one certified source result under
one explicit `MeshAdaptationRoute` and resolves protection and organization into
route constraints. `execute_mesh_adaptation` runs exactly that route (there is no
automatic provider fallback) and returns the certified target with its complete
lineage, sparse vertex stencil and FE transfer, route evidence, and compliance.
"""

from __future__ import annotations

import dataclasses
import time
from enum import StrEnum
from fractions import Fraction
from typing import Any, final, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import PredicateMode, resolve_host_predicate_mode
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh, PolyhedralConnectivity
from ..discretization._adaptive_simplex import (
    AdaptiveSimplexPolicy,
    bisection_label_tables,
)
from ..discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_transition_types import (
    CellGeometryTransition,
    CellGeometryTransitionEvidence,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_enclosure import (
    compose_expression,
    coordinate_map_expressions,
    coordinate_reference_chain,
    coordinate_source_basis,
    evaluate_expression,
    expression_equal,
    ExpressionInput,
    linear_combinations,
)
from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from ..discretization._nested_reference import (
    coarsen_reference_cell_maps,
    compose_reference_cell_maps,
    nested_cell_reference_maps,
)
from ..discretization._nested_reference_types import (
    evaluate_reference_chart_arguments,
    exact_reference_cell_vertices,
    exact_reference_vertices,
    freeze_reference_chart_arguments,
    NestedReferenceWitnesses,
    ReferenceCellMapWitnesses,
    validate_nested_reference_witnesses,
)
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._surface_chart_records import (
    _exact_chart_from_tokens,
    rational_chart_tokens as surface_chart_tokens,
    SurfaceChartWitness,
    SurfaceRootMeshCellBinding,
)
from ..discretization.fem import (
    balanced_hp_refinement_ids,
    coarsen_tensor_hp_cells,
    finite_element_hp_interface_plan,
    FiniteElementHPEpoch,
    FiniteElementHPLineage,
    FiniteElementHPRefinementResult,
    FiniteElementTopologyTransfer,
    hp_active_cell_mesh,
    lagrange_element,
    refine_tensor_hp_cells,
)
from ..discretization.finite_volume._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
)
from ..geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
    SourceBoundaryDistanceWitness,
    SourceBoundaryPointWitness,
    SurfaceEnvelopeDistanceWitness,
)
from ..geometry._source_fidelity_certificate import SourceFidelityWitness
from ..geometry._volume_material_deformation import (
    MaterialReferenceDeclaration,
    VolumeMaterialDeformation,
)
from ..geometry._volume_material_map import VolumeMaterialChain, VolumeMaterialMap
from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ..geometry.implicit._restricted_delaunay_coverage import (
    RestrictedDelaunayBoundaryWitness,
    RestrictedDelaunayCoverageTransportWitness,
)
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ..typing import checked
from ._assembly import MeshPart
from ._association import (
    AssociationTransfer,
    BRepAssociationTransfer,
    GeometrySourceEntityRole,
)
from ._association_composition import ComposedAssociationTransfer
from ._association_transfer import propagate_material_association_rows
from ._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from ._bisection import (
    _SUBDIVISION,
    BisectionCompatibility,
    BisectionEvidence,
    BisectionHierarchy,
    execute_bisection,
)
from ._canonical import certify_cell_mesh
from ._certification_inputs import MeshCertificationInputs
from ._device_metric import DeviceMetricEvidence
from ._device_tetra_metric import DeviceTetraMetricEvidence
from ._distribution import (
    MeshDistribution,
    MeshDistributionTransition,
    MeshPartitionPolicy,
    prepare_distribution_transition,
)
from ._implicit_association_transfer import ImplicitAssociationTransfer
from ._lineage import (
    CellMeshTransition,
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    MeshTransitionKind,
    VertexInterpolationStencil,
)
from ._local_metric import execute_local_metric_adaptation, LocalMetricEvidence
from ._mapped_reference_association import MappedReferenceAssociationTransfer
from ._metric import MeshMetricField
from ._mixed_adaptation import (
    _mixed_coarsening_reference_transforms,
    adapt_mixed_mesh,
    MixedAdaptationEvidence,
    MixedAdaptationHierarchy,
    MixedAdaptationOutcome,
    MixedCellContentTransfer,
    MixedCoarseningMap,
    MixedLayerColumns,
)
from ._organization import MeshLabel, MeshPatch, MeshZone, RegionBoundaryEvidence
from ._periodic import (
    periodic_quotient_facet_cell_classes,
    periodic_topology_after_edit,
    PeriodicRefinementEdgeLock,
    stage_periodic_refinement_candidate,
    stage_periodic_topology_edit,
)
from ._polyhedral_adaptation import (
    adapt_polyhedral_mesh,
    PolyhedralAdaptationEvidence,
    PolyhedralAdaptationHierarchy,
    PolyhedralAdaptationOutcome,
    PolyhedralAdaptationRequest,
    PolyhedralCoverageRenewal,
)
from ._result import CellMeshingResult, MeshingComplianceReport
from ._scope import MeshingEntityKind, MeshingScope, resolve_mesh_scope
from ._surface_association_transfer import SurfaceAssociationTransfer
from ._surface_metric import (
    SurfaceMetricOutcome,
    SurfaceMetricStatus,
    target_surface_source,
)
from ._surface_metric_controller import SurfaceMetricController
from ._tetra_metric import (
    execute_tetra_metric_adaptation,
    TetraMetricEvidence,
    TetraMetricStatus,
)
from ._topology_edit import (
    assemble_topology_edit,
    bind_periodic_topology_edit,
    CellTopologyEdit,
    entity_keys,
    key_rows,
    nested_reference_witnesses_from_edit,
    reference_cell_maps_from_edit,
)
from .providers._mmg import MmgAdaptationResult, MmgOptions, MmgProvider
from .providers._omega_h import OmegaHAdaptationResult, OmegaHOptions, OmegaHProvider


class MeshAdaptationStatus(StrEnum):
    """Outcome of one executed adaptation.

    ``COMPLETE``: every requested operation was realized (for metric routes the
    unit-mesh criterion holds). ``PARTIAL``: some requested marks were rejected
    (protected entities, ineligible coarsening families); the evidence lists them.
    ``NUMERICAL_STOP``: an unresolved numerical decision prevents completion.
    ``STALLED``: no admissible operation remains while the criterion is unmet.
    ``UNCHANGED``: nothing was applied; the target is the source result itself.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    NUMERICAL_STOP = "numerical_stop"
    STALLED = "stalled"
    UNCHANGED = "unchanged"

    @property
    def converged(self) -> bool:
        """Whether the request is fully realized (COMPLETE or UNCHANGED)."""
        return self in (MeshAdaptationStatus.COMPLETE, MeshAdaptationStatus.UNCHANGED)


class MeshAdaptationRoute(StrEnum):
    """Explicit execution route; each request kind admits only its own routes.

    ``NATIVE_BISECTION``: marked Maubach/newest-vertex bisection and coarsening of
    simplex meshes. ``NATIVE_METRIC_2D``: planar triangle metric adaptation (or
    relocation-only r-adaptation). ``DEVICE_BISECTION`` and ``DEVICE_METRIC_2D``:
    the same requests executed as compiled fixed-capacity device passes (see
    `prepare_adaptive_simplex`); device bisection commits the meshes of the host
    bisection route byte for byte. ``MMG`` and ``OMEGA_H``: metric adaptation by
    the configured provider. ``HP``: marked tensor-product h-refinement or
    coarsening of one prepared `FiniteElementHPEpoch`.
    ``NATIVE_METRIC_3D`` and ``DEVICE_METRIC_3D``: host and compiled 3D tetrahedral
    metric adaptation. ``NATIVE_METRIC_SURFACE``: source-chart surface metric
    adaptation. ``NATIVE_MIXED``: face-signature-closed mixed-family adaptation.
    ``NATIVE_POLYHEDRAL``: same-region polyhedral edits and site regeneration.
    """

    NATIVE_BISECTION = "native_bisection"
    NATIVE_METRIC_2D = "native_metric_2d"
    NATIVE_METRIC_3D = "native_metric_3d"
    NATIVE_METRIC_SURFACE = "native_metric_surface"
    NATIVE_MIXED = "native_mixed"
    NATIVE_POLYHEDRAL = "native_polyhedral"
    DEVICE_BISECTION = "device_bisection"
    DEVICE_METRIC_2D = "device_metric_2d"
    DEVICE_METRIC_3D = "device_metric_3d"
    MMG = "mmg"
    OMEGA_H = "omega_h"
    HP = "hp"


_DEVICE_ROUTES = frozenset(
    (
        MeshAdaptationRoute.DEVICE_BISECTION,
        MeshAdaptationRoute.DEVICE_METRIC_2D,
        MeshAdaptationRoute.DEVICE_METRIC_3D,
    )
)


# Relations through which organization membership (patches/zones/labels) is inherited.
_INHERITING_KINDS = np.asarray(
    [
        EntityLineageKind.PRESERVED,
        EntityLineageKind.REFINED_FROM,
        EntityLineageKind.COARSENED_INTO,
        EntityLineageKind.MERGED_INTO,
        EntityLineageKind.COLLAPSED_INTO,
        EntityLineageKind.SWAPPED_FROM,
        EntityLineageKind.RELOCATED,
    ],
    dtype=np.int32,
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
    """Source cells marked for refinement and coarsening by global ID.

    ``hierarchy`` carries state from bisection, mixed-family, or HP adaptation.
    ``layer_columns`` carries explicit first-layer and axial-schedule constraints
    for the mixed-family route.
    """

    refine_cell_ids: Array
    coarsen_cell_ids: Array
    hierarchy: BisectionHierarchy | MixedAdaptationHierarchy | FiniteElementHPEpoch | None
    periodic_edge_lock: PeriodicRefinementEdgeLock | None
    layer_columns: MixedLayerColumns | None
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        refine_cell_ids: ArrayLike = (),
        coarsen_cell_ids: ArrayLike = (),
        /,
        *,
        hierarchy: BisectionHierarchy
        | MixedAdaptationHierarchy
        | FiniteElementHPEpoch
        | None = None,
        periodic_edge_lock: PeriodicRefinementEdgeLock | None = None,
        layer_columns: MixedLayerColumns | None = None,
    ) -> None:
        refine = _identifier_vector(refine_cell_ids, "refine_cell_ids")
        coarsen = _identifier_vector(coarsen_cell_ids, "coarsen_cell_ids")
        if np.intersect1d(refine, coarsen).size:
            raise ValueError("A cell cannot be marked for refinement and coarsening.")
        if hierarchy is not None and not isinstance(
            hierarchy,
            (BisectionHierarchy, MixedAdaptationHierarchy, FiniteElementHPEpoch),
        ):
            raise TypeError("hierarchy must be a supported adaptation hierarchy or None.")
        if periodic_edge_lock is not None and not isinstance(
            periodic_edge_lock, PeriodicRefinementEdgeLock
        ):
            raise TypeError(
                "periodic_edge_lock must be PeriodicRefinementEdgeLock or None."
            )
        if periodic_edge_lock is not None and not refine.size:
            raise ValueError("A periodic edge lock requires marked refinement cells.")
        if layer_columns is not None and not isinstance(layer_columns, MixedLayerColumns):
            raise TypeError("layer_columns must be MixedLayerColumns or None.")
        self.refine_cell_ids = jnp.asarray(refine)
        self.coarsen_cell_ids = jnp.asarray(coarsen)
        self.hierarchy = hierarchy
        self.periodic_edge_lock = periodic_edge_lock
        self.layer_columns = layer_columns
        identity = {
            "kind": "marked-mesh-adaptation",
            "refine": array_tree_fingerprint(refine),
            "coarsen": array_tree_fingerprint(coarsen),
            "hierarchy": _hierarchy_id(hierarchy),
            "periodic_edge_lock": None
            if periodic_edge_lock is None
            else periodic_edge_lock.lock_id,
        }
        if layer_columns is not None:
            identity["layer_columns"] = layer_columns.columns_id
        self.request_id = canonical_fingerprint(identity)


@final
class MetricMeshAdaptation(StrictModule, NonTrainableState):
    """Adapt toward a unit mesh of one vertex metric bound to the source vertices."""

    metric: MeshMetricField
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(self, metric: MeshMetricField, /) -> None:
        self.metric = metric
        self.request_id = canonical_fingerprint(
            {"kind": "metric-mesh-adaptation", "metric": metric.metric_id}
        )


@final
class RelocationMeshAdaptation(StrictModule, NonTrainableState):
    """Move vertices toward one vertex metric without changing the topology."""

    metric: MeshMetricField
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(self, metric: MeshMetricField, /) -> None:
        self.metric = metric
        self.request_id = canonical_fingerprint(
            {"kind": "relocation-mesh-adaptation", "metric": metric.metric_id}
        )


MeshAdaptationRequest: TypeAlias = (
    MarkedMeshAdaptation
    | PolyhedralAdaptationRequest
    | MetricMeshAdaptation
    | RelocationMeshAdaptation
)


def _hierarchy_id(
    hierarchy: (
        BisectionHierarchy
        | MixedAdaptationHierarchy
        | PolyhedralAdaptationHierarchy
        | FiniteElementHPEpoch
        | None
    ),
    /,
) -> Any:
    match hierarchy:
        case None:
            return None
        case (
            BisectionHierarchy()
            | MixedAdaptationHierarchy()
            | PolyhedralAdaptationHierarchy()
        ):
            return hierarchy.hierarchy_id
        case FiniteElementHPEpoch():
            return hierarchy.epoch_id
        case _:
            raise TypeError("Unsupported adaptation hierarchy.")


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
            | MeshAdaptationRoute.NATIVE_METRIC_SURFACE
            | MeshAdaptationRoute.NATIVE_MIXED
            | MeshAdaptationRoute.NATIVE_POLYHEDRAL
            | MeshAdaptationRoute.DEVICE_BISECTION
            | MeshAdaptationRoute.DEVICE_METRIC_2D
            | MeshAdaptationRoute.DEVICE_METRIC_3D
            | MeshAdaptationRoute.HP
        ):
            if provider is not None or options is not None:
                raise ValueError("Native routes take no provider or provider options.")
        case _:
            raise TypeError("route must be MeshAdaptationRoute.")
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
    ``compatibility`` decides incompatible initial bisection labels (REJECT, or an
    explicit UNIFORM_REFINEMENT). ``predicate_mode`` certifies geometric decisions
    of the native metric route (EXACT uses meshcore when available, otherwise the
    filtered host predicates whose unresolved signs reject the operation); device
    routes evaluate FILTERED_DEVICE predicates and escalate unresolved signs.
    ``relocation`` enables vertex relocation in metric passes. Native operations
    continue to their mathematical criterion or an explicit unresolved/stalled
    outcome. ``device_policy`` selects an initial execution bucket, not a
    scientific problem-size ceiling. ``distribution`` with ``partition_policy`` carries
    ``association_transfer`` validates source identity, provides entity-class
    evidence for native closure, and propagates target associations through
    lineage. Provider routes accept only the B-Rep adapter and rederive its
    classes on output. Nonconforming HP active meshes require an ``audit_policy``
    that records rather than rejects nonmanifold (hanging) vertices.
    """

    route: MeshAdaptationRoute = eqx.field(static=True)
    protected_scopes: tuple[MeshingScope, ...]
    compatibility: BisectionCompatibility = eqx.field(static=True)
    predicate_mode: PredicateMode = eqx.field(static=True)
    relocation: bool = eqx.field(static=True)
    audit_policy: CellMeshAuditPolicy
    device_policy: AdaptiveSimplexPolicy | None
    provider: MmgProvider | OmegaHProvider | None = eqx.field(static=True)
    provider_options: MmgOptions | OmegaHOptions | None
    distribution: MeshDistribution | None
    partition_policy: MeshPartitionPolicy | None
    association_transfer: AssociationTransfer | None
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        route: MeshAdaptationRoute,
        /,
        *,
        protected_scopes: tuple[MeshingScope, ...] = (),
        compatibility: BisectionCompatibility = BisectionCompatibility.REJECT,
        predicate_mode: PredicateMode | None = None,
        relocation: bool = True,
        audit_policy: CellMeshAuditPolicy | None = None,
        device_policy: AdaptiveSimplexPolicy | None = None,
        provider: MmgProvider | OmegaHProvider | None = None,
        provider_options: MmgOptions | OmegaHOptions | None = None,
        distribution: MeshDistribution | None = None,
        partition_policy: MeshPartitionPolicy | None = None,
        association_transfer: AssociationTransfer | None = None,
    ) -> None:
        if not isinstance(route, MeshAdaptationRoute):
            raise TypeError("route must be MeshAdaptationRoute.")
        protected = tuple(protected_scopes)
        if not all(isinstance(scope, MeshingScope) for scope in protected):
            raise TypeError("protected_scopes must contain MeshingScope values.")
        if not isinstance(compatibility, BisectionCompatibility):
            raise TypeError("compatibility must be BisectionCompatibility.")
        mode = _resolved_predicate_mode(route, predicate_mode)
        if route not in _DEVICE_ROUTES and device_policy is not None:
            raise ValueError("device_policy applies only to device routes.")
        if route in _DEVICE_ROUTES and device_policy is None:
            device_policy = AdaptiveSimplexPolicy()
        if device_policy is not None and not isinstance(
            device_policy, AdaptiveSimplexPolicy
        ):
            raise TypeError("device_policy must be AdaptiveSimplexPolicy or None.")
        relocate = _flag(relocation, "relocation")
        audit = CellMeshAuditPolicy() if audit_policy is None else audit_policy
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
                    ImplicitAssociationTransfer,
                    SurfaceAssociationTransfer,
                    MappedReferenceAssociationTransfer,
                    ComposedAssociationTransfer,
                ),
            ):
                raise TypeError(
                    "association_transfer must be a supported "
                    "AssociationTransfer adapter or None."
                )
            if route is MeshAdaptationRoute.HP:
                raise ValueError("HP adaptation cannot carry source associations.")
            if route in (
                MeshAdaptationRoute.MMG,
                MeshAdaptationRoute.OMEGA_H,
            ) and not isinstance(association_transfer, BRepAssociationTransfer):
                raise ValueError(
                    "MMG and OMEGA_H routes require BRepAssociationTransfer."
                )
        _check_provider(route, provider, provider_options, protected, distribution)
        self.route = route
        self.protected_scopes = protected
        self.compatibility = compatibility
        self.predicate_mode = mode
        self.relocation = relocate
        self.audit_policy = audit
        self.device_policy = device_policy
        self.provider = provider
        self.provider_options = provider_options
        self.distribution = distribution
        self.partition_policy = partition_policy
        self.association_transfer = association_transfer
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mesh-adaptation-policy",
                "route": route.value,
                "protected": [scope.scope_id for scope in protected],
                "compatibility": compatibility.value,
                "predicate_mode": mode.value,
                "relocation": relocate,
                "audit": audit.policy_id,
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


def _protected_scope_rows(
    mesh: CellMesh, scopes: tuple[MeshingScope, ...], dimension: int, /
) -> np.ndarray:
    selected = tuple(
        _selected_rows(mesh, scope)
        for scope in scopes
        if scope.entity_dimension == dimension
    )
    if not selected:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.concatenate(selected))


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


def _cell_classes(mesh: CellMesh, scopes: tuple[MeshingScope, ...], /) -> np.ndarray:
    """One class per distinct (block, cell-organization membership) combination."""

    dimension = mesh.topological_dimension
    block_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    block_rows = np.concatenate(
        [
            np.full((block.cell_count,), index, dtype=np.int64)
            for index, block in enumerate(mesh.blocks)
        ]
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
    Periodic meshes decide boundary and interface status on the quotient: the
    outer boundary of a lifted cover is a representation artifact.
    """

    count = mesh.entity_set(1).count
    if mesh.periodic_topology is not None:
        incident, low, high = periodic_quotient_facet_cell_classes(mesh, cell_classes)
    else:
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
        rows = _selected_rows(mesh, scope)
        if scope.entity_dimension >= 1:
            protected_edges[_closure_rows(mesh, scope.entity_dimension, rows, 1)] = True
        fixed[_closure_rows(mesh, scope.entity_dimension, rows, 0)] = True
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    organization = _organization_scopes(source)
    cell_classes = _cell_classes(mesh, organization)
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
    if transfer is not None and source.associations:
        # Associated strata define the source classes used for closure and target propagation.
        classes = transfer.classes(source)
        codes = tuple(np.asarray(level.codes, dtype=np.int64) for level in classes)
        fixed |= classes[0].dimensions == 0
        protected_edges |= ~classes[1].resolved
        cell_classes = _row_classes(np.column_stack((cell_classes, codes[dimension])))
        facet_classes = _row_classes(
            np.column_stack((facet_classes, codes[dimension - 1]))
        )
        edge_codes = np.where(classes[1].dimensions < dimension, 1 + codes[1], 0)
    protected_vertex_ids = vertex_ids[fixed]
    fixed |= np.any(_membership(mesh, organization, 0), axis=1)
    metric_route = policy.route in (
        MeshAdaptationRoute.NATIVE_METRIC_2D,
        MeshAdaptationRoute.DEVICE_METRIC_2D,
        MeshAdaptationRoute.NATIVE_METRIC_3D,
        MeshAdaptationRoute.DEVICE_METRIC_3D,
        MeshAdaptationRoute.NATIVE_METRIC_SURFACE,
    )
    edge_feature_route = policy.route in (
        MeshAdaptationRoute.NATIVE_METRIC_2D,
        MeshAdaptationRoute.DEVICE_METRIC_2D,
        MeshAdaptationRoute.NATIVE_METRIC_SURFACE,
    )
    return _AdaptationConstraints(
        protected_edge_keys=entity_keys(mesh, 1)[protected_edges],
        protected_edge_mask=protected_edges,
        protected_vertex_ids=protected_vertex_ids,
        fixed_vertex_mask=fixed,
        cell_classes=cell_classes,
        facet_classes=facet_classes,
        edge_classes=(
            _feature_classes(_edge_classes(mesh, organization, cell_classes), edge_codes)
            if edge_feature_route
            else np.zeros((edge_count,), dtype=np.int64)
        ),
        metric_values=(
            # ty: ignore[unresolved-attribute]
            _metric_rows(mesh, request.metric)
            if metric_route
            else np.zeros((0, mesh.ambient_dimension, mesh.ambient_dimension))
        ),
    )


def _feature_classes(classes: np.ndarray, cad_codes: np.ndarray, /) -> np.ndarray:
    """Split feature edge classes by source entity (0 off associated edges)."""
    feature = (classes > 0) | (cad_codes > 0)
    result = np.zeros(classes.shape, dtype=np.int64)
    if np.any(feature):
        result[feature] = 1 + _row_classes(np.column_stack((classes, cad_codes))[feature])
    return result


def _cell_ids(mesh: CellMesh, /) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )


def _acceptance_audit_policy(
    policy: CellMeshAuditPolicy, inputs: MeshCertificationInputs | None, /
) -> CellMeshAuditPolicy:
    """Keep schedule-required audit checks evaluated under the route's policy."""
    if (
        inputs is None
        or "open_boundary" not in inputs.schedule.required_audit_checks
        or policy.watertight_boundary is not CellMeshAuditDisposition.SKIP
    ):
        return policy
    settings = {
        field.name: getattr(policy, field.name)
        for field in dataclasses.fields(policy)
        if field.name != "policy_id"
    }
    settings["watertight_boundary"] = CellMeshAuditDisposition.REJECT
    return CellMeshAuditPolicy(**settings)


def _require_native_source(
    source: CellMeshingResult, transfer: AssociationTransfer | None, /
) -> None:
    if source.attributes:
        raise ValueError("Native adaptation needs an explicit attribute remap.")
    if source.boundary is not None and not isinstance(
        transfer, (ImplicitAssociationTransfer, SurfaceAssociationTransfer)
    ):
        raise ValueError(
            "Native adaptation of a boundary model requires an implicit or surface association transfer."
        )
    if source.associations or source.boundary is not None:
        if transfer is None:
            raise ValueError(
                "Geometry associations require the policy's association_transfer."
            )
        transfer.validate_source(source)
    if source.coordinate_contract.coordinate_system != "cartesian":
        raise ValueError("Native adaptation requires Cartesian coordinates.")


def _check_marked_source(
    source: CellMeshingResult, request: MarkedMeshAdaptation, /
) -> None:
    identifiers = _cell_ids(source.mesh)
    marked = np.concatenate(
        (np.asarray(request.refine_cell_ids), np.asarray(request.coarsen_cell_ids))
    )
    if not np.all(np.isin(marked, identifiers)):
        raise ValueError("Marked cell IDs are not cells of the source mesh.")


def _require_implicit_surface_metric_source(
    source: CellMeshingResult,
    transfer: AssociationTransfer | None,
    /,
) -> None:
    certification = source.certification
    if certification is None or not isinstance(transfer, ImplicitAssociationTransfer):
        raise ValueError(
            "NATIVE_METRIC_SURFACE without source charts requires implicit source certification and ImplicitAssociationTransfer."
        )
    inputs = certification.inputs
    witness = inputs.fidelity_witness
    profile = transfer.profile
    source_query = inputs.source
    if (
        not isinstance(witness, SourceBoundaryChartWitness)
        or not isinstance(profile, AnalyticImplicitProfile)
        or source_query is None
        or witness.implicit_cell_keys.ndim != 2
        or witness.implicit_cell_keys.shape[1:] != (4,)
        or witness.implicit_root_edge_keys.ndim != 4
        or witness.implicit_root_edge_keys.shape[1:] != (3, 2, 3)
        or witness.implicit_cell_keys.shape[0] == 0
        or witness.implicit_root_edge_keys.shape[0] != witness.implicit_cell_keys.shape[0]
        or witness.source_id != profile.source_id
        or witness.source_revision != profile.source_revision
    ):
        raise ValueError(
            "NATIVE_METRIC_SURFACE without source charts requires an exact implicit root-chart witness and analytic profile."
        )
    witness.require_current(source.mesh, source.geometry, source_query)
    transfer.validate_source(source)


def _has_retained_source_claims(source: CellMeshingResult, /) -> bool:
    if any(
        (
            source.boundary is not None,
            bool(source.patches),
            bool(source.zones),
            bool(source.labels),
            bool(source.attributes),
            bool(source.associations),
            bool(source.region_boundary_evidence),
            source.surface_source is not None,
            source.planar_embedding is not None,
            source.layer_curving_evidence is not None,
        )
    ):
        return True
    if source.certification is None:
        return False
    inputs = source.certification.inputs
    return any(
        (
            inputs.domain is not None,
            inputs.source is not None,
            inputs.cell_regions is not None,
            inputs.fidelity_witness is not None,
            inputs.source_coverage_witness is not None,
            inputs.junction_vertices is not None,
            bool(inputs.protected_features),
            bool(inputs.scoped_fidelity),
            inputs.feature_source is not None,
        )
    )


def _check_route_request(
    source: CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
) -> None:
    mesh = source.mesh
    kinds = {block.cell_kind for block in mesh.blocks}
    # Host routes treat native 3-simplex blocks as T4; device layouts are T4 only.
    tetrahedra = (
        {"tetrahedron"}
        if policy.route
        in (MeshAdaptationRoute.DEVICE_BISECTION, MeshAdaptationRoute.DEVICE_METRIC_3D)
        else {"tetrahedron", "simplex:3"}
    )
    match policy.route:
        case MeshAdaptationRoute.NATIVE_BISECTION | MeshAdaptationRoute.DEVICE_BISECTION:
            if not isinstance(request, MarkedMeshAdaptation) or not isinstance(
                request.hierarchy, (BisectionHierarchy, type(None))
            ):
                raise TypeError(
                    f"{policy.route.name} requires a marked bisection request."
                )
            if request.layer_columns is not None:
                raise ValueError("layer_columns applies only to NATIVE_MIXED.")
            if kinds != {"triangle"} and not kinds <= tetrahedra:
                raise ValueError(
                    f"{policy.route.name} requires triangle or tetrahedron blocks."
                )
            _require_native_source(source, policy.association_transfer)
            _check_marked_source(source, request)
        case MeshAdaptationRoute.NATIVE_MIXED:
            if not isinstance(request, MarkedMeshAdaptation) or not isinstance(
                request.hierarchy, (MixedAdaptationHierarchy, type(None))
            ):
                raise TypeError(
                    "NATIVE_MIXED requires marked cells and a mixed-family hierarchy."
                )
            if (
                not kinds.issubset({"tetrahedron", "prism", "pyramid", "hexahedron"})
                or mesh.topological_dimension != 3
                or mesh.ambient_dimension != 3
            ):
                raise ValueError(
                    "NATIVE_MIXED requires tetrahedron, prism, pyramid, or hexahedron volume cells."
                )
            if (
                mesh.periodic_topology is not None
                or request.periodic_edge_lock is not None
            ):
                raise ValueError(
                    "NATIVE_MIXED requires an explicit periodic closure route."
                )
            _require_native_source(source, policy.association_transfer)
            _check_marked_source(source, request)
        case MeshAdaptationRoute.NATIVE_METRIC_2D | MeshAdaptationRoute.DEVICE_METRIC_2D:
            if not isinstance(request, (MetricMeshAdaptation, RelocationMeshAdaptation)):
                raise TypeError(
                    f"{policy.route.name} requires a metric or relocation request."
                )
            if kinds != {"triangle"} or mesh.ambient_dimension != 2:
                raise ValueError(f"{policy.route.name} requires planar triangle blocks.")
            _require_native_source(source, policy.association_transfer)
        case MeshAdaptationRoute.NATIVE_METRIC_SURFACE:
            if not isinstance(request, (MetricMeshAdaptation, RelocationMeshAdaptation)):
                raise TypeError(
                    "NATIVE_METRIC_SURFACE requires a metric or relocation request."
                )
            if (
                kinds != {"triangle"}
                or mesh.topological_dimension != 2
                or mesh.ambient_dimension != 3
            ):
                raise ValueError(
                    "NATIVE_METRIC_SURFACE requires embedded triangle surface blocks."
                )
            if source.surface_source is None:
                _require_implicit_surface_metric_source(
                    source, policy.association_transfer
                )
            _require_native_source(source, policy.association_transfer)
        case MeshAdaptationRoute.NATIVE_METRIC_3D | MeshAdaptationRoute.DEVICE_METRIC_3D:
            if not isinstance(request, MetricMeshAdaptation):
                raise TypeError(f"{policy.route.name} requires a metric request.")
            if (
                not kinds <= tetrahedra
                or mesh.ambient_dimension != 3
                or mesh.topological_dimension != 3
            ):
                raise ValueError(f"{policy.route.name} requires 3D tetrahedron blocks.")
            _require_native_source(source, policy.association_transfer)
        case MeshAdaptationRoute.NATIVE_POLYHEDRAL:
            if not isinstance(request, PolyhedralAdaptationRequest):
                raise TypeError("NATIVE_POLYHEDRAL requires PolyhedralAdaptationRequest.")
            if (
                mesh.topological_dimension != 3
                or mesh.ambient_dimension != 3
                or not isinstance(mesh.connectivity, PolyhedralConnectivity)
            ):
                raise ValueError(
                    "NATIVE_POLYHEDRAL requires a three-dimensional polyhedral mesh."
                )
            _require_native_source(source, policy.association_transfer)
        case MeshAdaptationRoute.MMG | MeshAdaptationRoute.OMEGA_H:
            if not isinstance(request, MetricMeshAdaptation):
                raise TypeError("Provider routes require a metric request.")
        case MeshAdaptationRoute.HP:
            if not isinstance(request, MarkedMeshAdaptation) or not isinstance(
                request.hierarchy, FiniteElementHPEpoch
            ):
                raise TypeError("HP requires a marked request with its hp epoch.")
            if request.layer_columns is not None:
                raise ValueError("layer_columns applies only to NATIVE_MIXED.")
            if request.hierarchy.mesh.topology_id != mesh.topology_id:
                raise ValueError("The hp epoch does not describe the source mesh.")
            if request.refine_cell_ids.size and request.coarsen_cell_ids.size:
                raise ValueError("One HP adaptation either refines or coarsens.")
            if source.patches or source.zones or source.labels:
                raise ValueError("HP adaptation cannot inherit mesh organization.")
            _require_native_source(source, None)
            _check_marked_source(source, request)
        case _:
            raise TypeError("route must be MeshAdaptationRoute.")


@final
class PreparedMeshAdaptation(StrictModule, NonTrainableState):
    """One request bound to one exact source revision under one policy."""

    source: CellMeshingResult
    request: MeshAdaptationRequest
    policy: MeshAdaptationPolicy
    constraints: _AdaptationConstraints
    prior_transition: CellMeshTransition | None
    material_reference: MaterialReferenceDeclaration | None
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CellMeshingResult,
        request: MeshAdaptationRequest,
        policy: MeshAdaptationPolicy,
        constraints: _AdaptationConstraints,
        prior_transition: CellMeshTransition | None = None,
        material_reference: MaterialReferenceDeclaration | None = None,
        /,
    ) -> None:
        self.source = source
        self.request = request
        self.policy = policy
        self.constraints = constraints
        self.prior_transition = prior_transition
        if prior_transition is not None and not isinstance(
            prior_transition, CellMeshTransition
        ):
            raise TypeError("prior_transition must be CellMeshTransition or None.")
        if material_reference is not None:
            if not isinstance(material_reference, MaterialReferenceDeclaration):
                raise TypeError(
                    "material_reference must be MaterialReferenceDeclaration or None."
                )
            if prior_transition is not None and (
                prior_transition.volume_material_chain is not None
                or prior_transition.material_reference is not None
            ):
                raise ValueError(
                    "An explicit material reference seed is mutually exclusive with a prior material chain."
                )
            material_reference = _validate_material_reference_seed(
                source, material_reference
            )
        self.material_reference = material_reference
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mesh-adaptation",
                "source": source.result_id,
                "request": request.request_id,
                "policy": policy.policy_id,
                "prior_transition": None
                if prior_transition is None
                else prior_transition.transition_id,
                "material_reference": (
                    None
                    if material_reference is None
                    else material_reference.declaration_id
                ),
            }
        )


def _validate_material_reference_seed(
    source: CellMeshingResult,
    reference: MaterialReferenceDeclaration,
    /,
) -> MaterialReferenceDeclaration:
    if not isinstance(reference, MaterialReferenceDeclaration):
        raise TypeError("material_reference must be MaterialReferenceDeclaration.")
    certification = source.certification
    if certification is None or certification.coverage is None:
        raise ValueError("Material reference seeding requires certified source coverage.")
    certification.require_passed()
    coverage = certification.coverage
    rebuilt = MaterialReferenceDeclaration.revalidate(
        reference,
        reference.root_mesh,
        reference.root_geometry,
        reference.source_map,
        reference.root_coverage,
    )
    if (
        rebuilt.root_mesh.mesh_id != source.mesh.mesh_id
        or rebuilt.root_mesh.topology_id != source.mesh.topology_id
        or cell_geometry_id(rebuilt.root_geometry) != cell_geometry_id(source.geometry)
        or rebuilt.root_coverage.certificate_id != coverage.certificate_id
        or rebuilt.frame_id != source.coordinate_contract.spatial_id
        or rebuilt.source_map.source_id != coverage.source_id
        or rebuilt.source_map.source_revision != coverage.source_revision
        or rebuilt.source_map.domain_id != coverage.domain_id
    ):
        raise ValueError(
            "A material reference seed must bind this exact source, frame and coverage."
        )
    inputs = certification.inputs
    if (inputs.source_id, inputs.source_revision) != (
        rebuilt.source_map.source_id,
        rebuilt.source_map.source_revision,
    ):
        raise ValueError("Material reference source identity differs from certification.")
    if source.geometry.source_map is not None and (
        source.geometry.source_map.source_id != rebuilt.source_map.source_id
        or source.geometry.source_map.source_revision
        != rebuilt.source_map.source_revision
    ):
        raise ValueError(
            "Material reference differs from the represented coordinate source."
        )
    cell_ids = tuple(
        int(identifier)
        for block in source.mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    )
    region_indices = np.asarray(coverage.cell_regions, dtype=np.int64)
    if region_indices.shape != (len(cell_ids),) or np.any(
        (region_indices < 0) | (region_indices >= len(coverage.region_ids))
    ):
        raise ValueError("Certified material region rows do not align with source cells.")
    regions = tuple(coverage.region_ids[int(index)] for index in region_indices)
    if (
        len(rebuilt.cell_entity_ids) != len(cell_ids)
        or len(rebuilt.region_ids) != len(cell_ids)
        or rebuilt.region_ids != regions
    ):
        raise ValueError(
            "Material reference rows do not match certified source cells/regions."
        )
    cell_entity_set_id = source.mesh.entity_set(3).entity_set_id
    candidate_associations = tuple(
        association
        for association in source.associations
        if association.target_entity_set_id == cell_entity_set_id
        and association.source_id == rebuilt.source_map.source_id
        and association.source_revision == rebuilt.source_map.source_revision
    )
    if candidate_associations:
        matched = False
        for association in candidate_associations:
            association_ids = tuple(
                int(value)
                for value in np.asarray(association.target_global_ids, dtype=np.int64)
            )
            if set(association_ids) != set(cell_ids):
                continue
            row_by_id = {
                identifier: row for row, identifier in enumerate(association_ids)
            }
            ordered_rows = tuple(row_by_id[identifier] for identifier in cell_ids)
            if (
                not np.all(
                    np.asarray(association.resolved, dtype=np.bool_)[
                        np.asarray(ordered_rows, dtype=np.int64)
                    ]
                )
                or np.any(
                    np.asarray(association.ambiguous, dtype=np.bool_)[
                        np.asarray(ordered_rows, dtype=np.int64)
                    ]
                )
                or any(
                    association.source_roles[row] is not GeometrySourceEntityRole.REGION
                    for row in ordered_rows
                )
            ):
                continue
            entities = tuple(association.source_entity_ids[row] for row in ordered_rows)
            paths = tuple(
                association.source_occurrence_paths[row] for row in ordered_rows
            )
            if entities == rebuilt.cell_entity_ids and paths == rebuilt.occurrence_paths:
                matched = True
                break
        if not matched:
            raise ValueError(
                "Material reference cells disagree with accepted source-region associations."
            )
    return rebuilt


def prepare_mesh_adaptation(
    source: CellMeshingResult,
    request: MeshAdaptationRequest,
    /,
    *,
    policy: MeshAdaptationPolicy,
    prior_transition: CellMeshTransition | None = None,
    material_reference: MaterialReferenceDeclaration | None = None,
) -> PreparedMeshAdaptation:
    """Validate the route/request pairing and resolve protection and organization."""

    if not isinstance(source, CellMeshingResult):
        raise TypeError("source must be CellMeshingResult.")
    if not isinstance(
        request,
        (
            MarkedMeshAdaptation,
            PolyhedralAdaptationRequest,
            MetricMeshAdaptation,
            RelocationMeshAdaptation,
        ),
    ):
        raise TypeError("request must be a typed mesh-adaptation request.")
    if not isinstance(policy, MeshAdaptationPolicy):
        raise TypeError("policy must be MeshAdaptationPolicy.")
    if prior_transition is not None and not isinstance(
        prior_transition, CellMeshTransition
    ):
        raise TypeError("prior_transition must be CellMeshTransition or None.")
    if material_reference is not None and not isinstance(
        material_reference, MaterialReferenceDeclaration
    ):
        raise TypeError(
            "material_reference must be MaterialReferenceDeclaration or None."
        )
    if (
        material_reference is not None
        and prior_transition is not None
        and (
            prior_transition.volume_material_chain is not None
            or prior_transition.material_reference is not None
        )
    ):
        raise ValueError(
            "An explicit material reference seed is mutually exclusive with a prior material chain."
        )
    if prior_transition is not None:
        prior_target = prior_transition.target
        if (
            prior_target.result_id != source.result_id
            or prior_target.mesh.mesh_id != source.mesh.mesh_id
            or prior_target.mesh.topology_id != source.mesh.topology_id
            or cell_geometry_id(prior_target.geometry)
            != cell_geometry_id(source.geometry)
        ):
            raise ValueError(
                "prior_transition target must be the exact accepted mesh/geometry result."
            )
        chain = prior_transition.volume_material_chain
        if chain is not None:
            # A chain carries its reference; the seed check applies only to
            # transitions that retain a reference without a chain.
            if (
                chain.target_mesh_id != source.mesh.mesh_id
                or chain.target_geometry_id != cell_geometry_id(source.geometry)
            ):
                raise ValueError(
                    "prior_transition material chain is stale for the source result."
                )
        elif prior_transition.material_reference is not None:
            reference = MaterialReferenceDeclaration.revalidate(
                prior_transition.material_reference,
                prior_transition.material_reference.root_mesh,
                prior_transition.material_reference.root_geometry,
                prior_transition.material_reference.source_map,
                prior_transition.material_reference.root_coverage,
            )
            if reference.root_mesh.mesh_id != source.mesh.mesh_id or cell_geometry_id(
                reference.root_geometry
            ) != cell_geometry_id(source.geometry):
                raise ValueError(
                    "A prior material-reference seed must bind the exact current source configuration."
                )
    if policy.route in (MeshAdaptationRoute.MMG, MeshAdaptationRoute.OMEGA_H) and (
        _has_retained_source_claims(source)
        or material_reference is not None
        or (
            prior_transition is not None
            and (
                prior_transition.volume_material_chain is not None
                or prior_transition.material_reference is not None
            )
        )
    ):
        raise ValueError(
            "Provider remeshing has unknown lineage and cannot renew retained source claims."
        )
    source.audit.require_passed()
    _check_route_request(source, request, policy)
    if policy.distribution is not None:
        expected_part = MeshPart(policy.distribution.part.name, source)
        if policy.distribution.part.part_id != expected_part.part_id:
            raise ValueError("Mesh distribution is stale for the accepted source result.")
    match policy.route:
        case (
            MeshAdaptationRoute.NATIVE_BISECTION
            | MeshAdaptationRoute.NATIVE_MIXED
            | MeshAdaptationRoute.NATIVE_METRIC_2D
            | MeshAdaptationRoute.NATIVE_METRIC_3D
            | MeshAdaptationRoute.NATIVE_METRIC_SURFACE
            | MeshAdaptationRoute.DEVICE_BISECTION
            | MeshAdaptationRoute.DEVICE_METRIC_2D
            | MeshAdaptationRoute.DEVICE_METRIC_3D
            | MeshAdaptationRoute.NATIVE_POLYHEDRAL
        ):
            constraints = _resolve_constraints(source, request, policy)
        case (
            MeshAdaptationRoute.MMG | MeshAdaptationRoute.OMEGA_H | MeshAdaptationRoute.HP
        ):
            for scope in policy.protected_scopes:
                resolve_mesh_scope(source.mesh, scope)
            constraints = _empty_constraints(source.mesh)
        case _:
            raise TypeError("route must be MeshAdaptationRoute.")
    return PreparedMeshAdaptation(
        source, request, policy, constraints, prior_transition, material_reference
    )


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
    | TetraMetricEvidence
    | DeviceTetraMetricEvidence
    | MixedAdaptationEvidence
    | PolyhedralAdaptationEvidence
    | SurfaceMetricOutcome
    | MmgAdaptationResult
    | OmegaHAdaptationResult
    | FiniteElementHPRefinementResult
)


def _evidence_id(evidence: MeshAdaptationEvidence | None, /) -> Any:
    match evidence:
        case None:
            return None
        case SurfaceMetricOutcome():
            return canonical_fingerprint(
                {
                    "kind": "surface-metric-route-evidence",
                    "chart": evidence.geometry.target_chart_witness.witness_id,
                    "evidence": evidence.evidence.evidence_id,
                    "status": evidence.status.value,
                    "passes": evidence.passes,
                    "splits": evidence.splits,
                    "collapses": evidence.collapses,
                    "flips": evidence.flips,
                    "relocations": evidence.relocations,
                }
            )
        case (
            BisectionEvidence()
            | LocalMetricEvidence()
            | DeviceMetricEvidence()
            | TetraMetricEvidence()
            | DeviceTetraMetricEvidence()
            | MixedAdaptationEvidence()
            | PolyhedralAdaptationEvidence()
        ):
            return evidence.evidence_id
        case (
            MmgAdaptationResult()
            | OmegaHAdaptationResult()
            | FiniteElementHPRefinementResult()
        ):
            return evidence.result_id
        case _:
            raise TypeError("Unsupported adaptation evidence.")


@final
class MeshAdaptationResult(StrictModule, NonTrainableState):
    """Certified target of one executed adaptation with lineage and evidence.

    ``transition``, ``lineage``, ``stencil``, and ``transfer`` are absent when
    the certified target remains the exact source. That source-preserving result
    is ``UNCHANGED`` only when no requested work was refused and no convergence
    criterion remains unmet; it may instead be ``PARTIAL``, ``STALLED``, or
    ``NUMERICAL_STOP``. Provider routes have unknown vertex lineage and therefore no
    stencil or transfer. ``transfer`` maps P1 vertex values only when the vertex
    stencil preserves constants and every target cell has explicit source lineage;
    otherwise field consumers must use their declared transfer policy or retain
    the accepted state. ``metric`` is the target-bound metric of metric routes,
    ``hierarchy`` the state to pass to the next marked request, and
    ``elapsed_seconds`` is wall-time evidence outside ``result_id``.
    """

    status: MeshAdaptationStatus = eqx.field(static=True)
    route: MeshAdaptationRoute = eqx.field(static=True)
    source: CellMeshingResult
    target: CellMeshingResult
    transition: CellMeshTransition | None
    lineage: MeshLineage | None
    stencil: VertexInterpolationStencil | None
    transfer: FiniteElementTopologyTransfer | None
    cell_content_transfer: MixedCellContentTransfer | None
    conservative_remap: UnstructuredConservativeRemapPlan | None
    surface_chart_witness: SourceFidelityWitness | None
    metric: MeshMetricField | None
    compliance: MeshingComplianceReport
    evidence: MeshAdaptationEvidence | None
    hierarchy: (
        BisectionHierarchy
        | MixedAdaptationHierarchy
        | PolyhedralAdaptationHierarchy
        | FiniteElementHPEpoch
        | None
    )
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
            and outcome.cell_content_transfer is None
            and outcome.conservative_remap is None
            and outcome.surface_chart_witness is None
        )
        if source_preserved != absent:
            raise ValueError(
                "A source-preserving adaptation has no transition or transfer evidence."
            )
        if outcome.status is MeshAdaptationStatus.UNCHANGED and not source_preserved:
            raise ValueError("UNCHANGED must preserve the exact source result.")
        if outcome.transition is not None and (
            outcome.transition.target.result_id != outcome.target.result_id
            # ty: ignore[unresolved-attribute]
            or outcome.transition.lineage.lineage_id != outcome.lineage.lineage_id
        ):
            raise ValueError("The transition must describe the certified target.")
        self.status = outcome.status
        self.route = prepared.policy.route
        self.source = prepared.source
        self.target = outcome.target
        self.transition = outcome.transition
        self.lineage = outcome.lineage
        self.stencil = outcome.stencil
        self.transfer = outcome.transfer
        self.cell_content_transfer = outcome.cell_content_transfer
        self.conservative_remap = outcome.conservative_remap
        self.surface_chart_witness = outcome.surface_chart_witness
        self.metric = outcome.metric
        self.compliance = compliance
        self.evidence = outcome.evidence
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
                "cell_content_transfer": (
                    None
                    if outcome.cell_content_transfer is None
                    else outcome.cell_content_transfer.transfer_id
                ),
                "conservative_remap": (
                    None
                    if outcome.conservative_remap is None
                    else outcome.conservative_remap.plan_id
                ),
                "surface_chart_witness": (
                    None
                    if outcome.surface_chart_witness is None
                    else outcome.surface_chart_witness.witness_id
                ),
                "metric": None if outcome.metric is None else outcome.metric.metric_id,
                "compliance": compliance.report_id,
                "evidence": _evidence_id(outcome.evidence),
                "hierarchy": _hierarchy_id(outcome.hierarchy),
                "distribution": None
                if distribution is None
                else distribution.transition_id,
            }
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
    hierarchy: (
        BisectionHierarchy
        | MixedAdaptationHierarchy
        | PolyhedralAdaptationHierarchy
        | FiniteElementHPEpoch
        | None
    )
    cell_content_transfer: MixedCellContentTransfer | None = None
    conservative_remap: UnstructuredConservativeRemapPlan | None = None
    surface_chart_witness: SourceFidelityWitness | None = None


def _unchanged(
    prepared: PreparedMeshAdaptation,
    evidence: MeshAdaptationEvidence | None,
    hierarchy: (
        BisectionHierarchy
        | MixedAdaptationHierarchy
        | PolyhedralAdaptationHierarchy
        | FiniteElementHPEpoch
        | None
    ),
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


def _inherited_ids(scope: MeshingScope, record: EntityLineage, /) -> np.ndarray:
    """Target IDs inheriting one organization scope through one lineage record.

    COLLAPSED_INTO sources are non-dominant: a target with any other inheriting
    relation takes its membership from those; otherwise the collapsed sources
    decide. Mixed membership among deciding sources is a route error.
    """

    kinds = np.asarray(record.relation_kinds, dtype=np.int32)
    inheriting = np.isin(kinds, _INHERITING_KINDS)
    targets = np.asarray(record.target_global_ids, dtype=np.int64)[inheriting]
    members = np.isin(
        np.asarray(record.source_global_ids, dtype=np.int64)[inheriting],
        np.asarray(scope.entity_ids, dtype=np.int64),
    )
    collapsed = kinds[inheriting] == EntityLineageKind.COLLAPSED_INTO
    unique, inverse = np.unique(targets, return_inverse=True)
    dominant = np.bincount(inverse, weights=~collapsed, minlength=unique.size) > 0
    deciding = ~collapsed | ~dominant[inverse]
    total = np.bincount(inverse, weights=deciding, minlength=unique.size)
    member = np.bincount(inverse, weights=deciding & members, minlength=unique.size)
    if np.any((member > 0) & (member < total)):
        raise ValueError(
            "Adaptation merged entities of different organization membership."
        )
    return unique[(member > 0) & (member == total)]


def _inherited_scope(
    scope: MeshingScope, lineage: MeshLineage, target: CellMesh, name: str, /
) -> MeshingScope:
    dimension = scope.entity_dimension
    identifiers = _inherited_ids(scope, lineage.entity_lineage(dimension))
    if identifiers.size == 0:
        raise ValueError(f"Adaptation removed every entity of organization {name!r}.")
    return MeshingScope(
        target.mesh_id,
        target.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        target.entity_set(dimension).entity_set_id,
        identifiers,
    )


def _inherited_organization(
    source: CellMeshingResult, target: CellMesh, lineage: MeshLineage, /
) -> tuple[
    tuple[MeshPatch, ...],
    tuple[MeshZone, ...],
    tuple[MeshLabel, ...],
    tuple[RegionBoundaryEvidence, ...],
]:
    zone_ids = {}
    zones = []
    for zone in source.zones:
        inherited = MeshZone(
            zone.name,
            zone.role,
            _inherited_scope(zone.scope, lineage, target, zone.name),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        zone_ids[zone.zone_id] = inherited.zone_id
        zones.append(inherited)
    patches = tuple(
        MeshPatch(
            patch.name,
            _inherited_scope(patch.scope, lineage, target, patch.name),
            connected=patch.connected,
            adjacent_zone_ids=tuple(zone_ids[value] for value in patch.adjacent_zone_ids),
            source_adjacent_region_ids=patch.source_adjacent_region_ids,
            source_adjacent_region_occurrence_paths=(
                patch.source_adjacent_region_occurrence_paths
            ),
        )
        for patch in source.patches
    )
    labels = tuple(
        MeshLabel(label.name, _inherited_scope(label.scope, lineage, target, label.name))
        for label in source.labels
    )
    patch_ids = {
        original.patch_id: inherited.patch_id
        for original, inherited in zip(source.patches, patches, strict=True)
    }
    label_by_source = {
        original.label_id: inherited
        for original, inherited in zip(source.labels, labels, strict=True)
    }
    boundaries = tuple(
        RegionBoundaryEvidence(
            target,
            evidence.source_scope,
            evidence.source_region_id,
            evidence.control_id,
            evidence.material_id,
            evidence.role,
            label_by_source[evidence.boundary_label.label_id],
            tuple(
                (patch_ids[identifier], side) for identifier, side in evidence.patch_sides
            ),
            source_occurrence_path=evidence.source_occurrence_path,
        )
        for evidence in source.region_boundary_evidence
    )
    return patches, tuple(zones), labels, boundaries


def _identity_lineage(source: CellMesh, target: CellMesh, /) -> MeshLineage:
    """Bind unchanged entity IDs across one coordinate-only mesh epoch."""
    if source.topology_id != target.topology_id:
        raise ValueError("Identity lineage requires unchanged mesh topology.")
    records = []
    for dimension in range(source.topological_dimension + 1):
        source_set, target_set = (
            source.entity_set(dimension),
            target.entity_set(dimension),
        )
        source_ids = np.asarray(source_set.entity_ids, dtype=np.int64)
        target_ids = np.asarray(target_set.entity_ids, dtype=np.int64)
        if not np.array_equal(source_ids, target_ids):
            raise ValueError("Coordinate-only transitions must preserve every entity ID.")
        records.append(
            EntityLineage(
                dimension,
                source_set.entity_set_id,
                target_set.entity_set_id,
                source_ids,
                target_ids,
                np.full(
                    source_ids.shape, int(EntityLineageKind.PRESERVED), dtype=np.int32
                ),
            )
        )
    return MeshLineage(source.topology_id, target.topology_id, tuple(records))


def _transition_regions(
    source_inputs: MeshCertificationInputs,
    target: CellMesh,
    lineage: MeshLineage,
    /,
) -> np.ndarray | None:
    """Reassign every target cell from complete, same-material source lineage."""
    if source_inputs.domain is None:
        return None
    if source_inputs.cell_regions is None:
        raise ValueError("A retained source domain has no material assignments.")
    source_regions = dict(
        zip(
            np.asarray(source_inputs.cell_global_ids, dtype=np.int64).tolist(),
            np.asarray(source_inputs.cell_regions, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target.blocks]
    )
    contributions: dict[int, list[int]] = {}
    relation = lineage.entity_lineage(target.topological_dimension)
    for source_id, target_id, kind in zip(
        np.asarray(relation.source_global_ids, dtype=np.int64),
        np.asarray(relation.target_global_ids, dtype=np.int64),
        np.asarray(relation.relation_kinds, dtype=np.int32),
        strict=True,
    ):
        if int(kind) not in _INHERITING_KINDS:
            continue
        try:
            region = source_regions[int(source_id)]
        except KeyError as error:
            raise ValueError(
                "Cell lineage names a cell outside retained source regions."
            ) from error
        contributions.setdefault(int(target_id), []).append(region)
    assignments = np.empty(target_ids.shape, dtype=np.int64)
    for row, identifier in enumerate(target_ids.tolist()):
        values = contributions.get(int(identifier), ())
        if not values:
            raise ValueError(
                f"Target cell {identifier} has no source region ancestry; source coverage cannot be retained."
            )
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Target cell {identifier} merges different source regions.")
        assignments[row] = values[0]
    return assignments


def _transition_junctions(
    source_inputs: MeshCertificationInputs,
    target: CellMesh,
    lineage: MeshLineage,
    /,
) -> np.ndarray | None:
    if source_inputs.junction_vertices is None:
        return None
    wanted = set(int(value) for value in np.asarray(source_inputs.junction_vertices))
    relation = lineage.entity_lineage(0)
    targets = {
        int(target_id)
        for source_id, target_id, kind in zip(
            np.asarray(relation.source_global_ids, dtype=np.int64),
            np.asarray(relation.target_global_ids, dtype=np.int64),
            np.asarray(relation.relation_kinds, dtype=np.int32),
            strict=True,
        )
        if int(source_id) in wanted and int(kind) in _INHERITING_KINDS
    }
    if len(targets) != len(wanted):
        raise ValueError("A declared source junction lost its target vertex ancestry.")
    target_ids = set(int(value) for value in np.asarray(target.vertex_global_ids))
    if not targets <= target_ids:
        raise ValueError("A declared source junction maps outside the target mesh.")
    return np.asarray(sorted(targets), dtype=np.int64)


def _piecewise_linear_geometry(mesh: CellMesh, geometry: CellGeometrySpec, /) -> bool:
    """Recognize an actual P1 map by exact represented coordinate expressions."""
    try:
        elements, routes, coefficients = geometry.resolve(mesh)
        points = np.asarray(coefficients, dtype=np.float64)
        for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
            if not isinstance(element, ScalarCoordinateElement):
                return False
            linear = coordinate_source_basis(lagrange_element(block.cell_kind, 1))
            vertices = np.asarray(block.vertices, dtype=np.int64)
            for row, route_row in enumerate(np.asarray(route, dtype=np.int64)):
                actual = coordinate_map_expressions(
                    element, points[route_row], cell_row=row
                )
                corner_points = np.asarray(mesh.coordinates, dtype=np.float64)[
                    vertices[row]
                ]
                expected = linear_combinations(
                    linear,
                    tuple(
                        tuple(Fraction(float(value)) for value in corner_points[:, axis])
                        for axis in range(mesh.ambient_dimension)
                    ),
                )
                if any(
                    not expression_equal(left, right)
                    for left, right in zip(actual, expected, strict=True)
                ):
                    return False
    except (TypeError, ValueError, NotImplementedError):
        return False
    return True


def _same_volume_boundary_geometry(source: CellMesh, target: CellMesh, /) -> bool:
    if (
        source.topological_dimension != 3
        or target.topological_dimension != 3
        or source.topology_id != target.topology_id
    ):
        return False
    source_mask = np.asarray(
        source.topology.entities(0).subset("boundary").mask, dtype=np.bool_
    )
    target_mask = np.asarray(
        target.topology.entities(0).subset("boundary").mask, dtype=np.bool_
    )
    source_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    target_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    source_boundary = source_ids[source_mask]
    target_boundary = target_ids[target_mask]
    if not np.array_equal(np.sort(source_boundary), np.sort(target_boundary)):
        return False
    source_rows = {int(identifier): row for row, identifier in enumerate(source_ids)}
    target_rows = {int(identifier): row for row, identifier in enumerate(target_ids)}
    source_coordinates = np.asarray(source.coordinates, dtype=np.float64)
    target_coordinates = np.asarray(target.coordinates, dtype=np.float64)
    return all(
        np.array_equal(
            source_coordinates[source_rows[int(identifier)]],
            target_coordinates[target_rows[int(identifier)]],
        )
        for identifier in source_boundary
    )


def _reference_map_rows(
    reference_maps: tuple[ReferenceCellMapWitnesses, ...], /
) -> dict[int, tuple[ReferenceCellMapWitnesses, int, tuple[tuple[Fraction, ...], ...]]]:
    result = {}
    for witness in reference_maps:
        exact = exact_reference_cell_vertices(witness)
        for row, identifier in enumerate(
            np.asarray(witness.target_cell_ids, dtype=np.int64)
        ):
            if int(identifier) in result:
                raise ValueError("Reference maps assign one target cell more than once.")
            result[int(identifier)] = (witness, row, exact[row])
    return result


def _reference_chart_expressions(
    cell_kind: str,
    points: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[Any, ...]:
    basis = coordinate_source_basis(lagrange_element(cell_kind, 1))
    if len(basis) != len(points):
        raise ValueError(
            "Exact reference map corners do not match the target cell basis."
        )
    dimension = len(points[0])
    return linear_combinations(
        basis,
        tuple(tuple(point[axis] for point in points) for axis in range(dimension)),
    )


def _available_reference_maps(
    source: CellMesh,
    target: CellMesh,
    edit: CellTopologyEdit,
    /,
) -> tuple[ReferenceCellMapWitnesses, ...]:
    try:
        return reference_cell_maps_from_edit(source, target, edit)
    except ValueError:
        pass
    dimension = source.topological_dimension
    relation = next(
        (value for value in edit.relations if value.dimension == dimension), None
    )
    if relation is None:
        raise ValueError(
            "Topology edit has no top-cell lineage for exact reference maps."
        )
    source_rows = {
        int(identifier): (block, row)
        for block in source.blocks
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    target_kinds = {
        int(identifier): block.cell_kind
        for block in target.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    source_kinds = {
        int(identifier): block.cell_kind
        for block in source.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    refinement = nested_reference_witnesses_from_edit(source, target, edit)
    refined = {}
    if refinement is not None:
        exact_rows = exact_reference_vertices(refinement)
        refined = {
            int(target_id): (int(source_id), exact)
            for target_id, source_id, exact in zip(
                np.asarray(refinement.fine_cell_ids, dtype=np.int64),
                np.asarray(refinement.coarse_cell_ids, dtype=np.int64),
                exact_rows,
                strict=True,
            )
        }
    target_vertex_ids = np.asarray(edit.vertex_global_ids, dtype=np.int64)
    stencil_sources = np.asarray(edit.stencil_sources, dtype=np.int64)
    stencil_weights = np.asarray(edit.stencil_weights, dtype=np.float64)
    stencil_valid = np.asarray(edit.stencil_valid, dtype=np.bool_)
    vertex_rows = {
        int(identifier): row for row, identifier in enumerate(target_vertex_ids)
    }
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    grouped: dict[
        tuple[str, str], list[tuple[int, int, tuple[tuple[Fraction, ...], ...]]]
    ] = {}
    source_keys = np.asarray(relation.source_keys, dtype=np.int64)
    target_keys = np.asarray(relation.target_keys, dtype=np.int64)
    kinds = np.asarray(relation.kinds, dtype=np.int32)
    target_rows = {
        int(identifier): (block, row)
        for block in target.blocks
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    for source_id_raw, target_id_raw, kind in zip(
        source_keys[:, 0], target_keys[:, 0], kinds, strict=True
    ):
        source_id, target_id = int(source_id_raw), int(target_id_raw)
        if int(kind) == int(EntityLineageKind.REFINED_FROM):
            mapped = refined.get(target_id)
            if mapped is None or mapped[0] != source_id:
                raise ValueError(
                    "Refined cell lineage has no exact nested reference map."
                )
            exact = mapped[1]
        elif int(kind) in (
            int(EntityLineageKind.PRESERVED),
            int(EntityLineageKind.RELOCATED),
        ):
            source_owner = source_rows.get(source_id)
            target_owner = target_rows.get(target_id)
            if source_owner is None or target_owner is None:
                raise ValueError("Preserved cell lineage names an absent endpoint cell.")
            source_block, source_row = source_owner
            target_block, target_row = target_owner
            if target_block.cell_kind != source_block.cell_kind:
                raise ValueError(
                    "Preserved cell lineage changes its reference-cell family."
                )
            source_topology = reference_cell_topology(source_block.cell_kind)
            source_global_vertices = source_vertex_ids[
                np.asarray(source_block.vertices[source_row], dtype=np.int64)
            ]
            source_slots = {
                int(identifier): row
                for row, identifier in enumerate(source_global_vertices)
            }
            target_global_vertices = np.asarray(target.vertex_global_ids, dtype=np.int64)[
                np.asarray(target_block.vertices[target_row], dtype=np.int64)
            ]
            exact_points = []
            for vertex_id in target_global_vertices.tolist():
                stencil_row = vertex_rows.get(int(vertex_id))
                if stencil_row is None:
                    raise ValueError(
                        "Preserved cell corner lacks its exact vertex stencil."
                    )
                active = stencil_valid[stencil_row]
                active_ids = stencil_sources[stencil_row, active]
                active_weights = stencil_weights[stencil_row, active]
                accumulated = [Fraction(0) for _ in source_global_vertices]
                for active_id, weight in zip(active_ids, active_weights, strict=True):
                    slot = source_slots.get(int(active_id))
                    if slot is None:
                        raise ValueError(
                            "Preserved vertex ancestry escapes its source cell."
                        )
                    accumulated[slot] += Fraction(float(weight))
                if sum(accumulated, Fraction(0)) != 1:
                    raise ValueError("Preserved vertex stencils must sum exactly to one.")
                exact_points.append(
                    tuple(
                        sum(
                            (
                                accumulated[corner]
                                * Fraction(float(source_topology.vertices[corner][axis]))
                                for corner in range(len(source_global_vertices))
                            ),
                            Fraction(0),
                        )
                        for axis in range(dimension)
                    )
                )
            exact = tuple(exact_points)
        else:
            continue
        target_kind = target_kinds.get(target_id)
        source_kind = source_kinds.get(source_id)
        if target_kind is None or source_kind is None:
            raise ValueError("Exact reference maps name cells outside an endpoint mesh.")
        grouped.setdefault((target_kind, source_kind), []).append(
            (target_id, source_id, exact)
        )
    # Cells outside the edit keep their stable ID and corners: identity charts.
    related = {int(value) for value in target_keys[:, 0]}
    target_vertex_table = np.asarray(target.vertex_global_ids, dtype=np.int64)
    for target_block in target.blocks:
        target_cells = np.asarray(target_block.vertices, dtype=np.int64)
        for row, identifier in enumerate(
            np.asarray(target_block.global_ids, dtype=np.int64).tolist()
        ):
            source_owner = source_rows.get(int(identifier))
            if int(identifier) in related or source_owner is None:
                continue
            source_block, source_row = source_owner
            if source_block.cell_kind != target_block.cell_kind or not np.array_equal(
                source_vertex_ids[np.asarray(source_block.vertices[source_row])],
                target_vertex_table[target_cells[row]],
            ):
                continue
            grouped.setdefault(
                (target_block.cell_kind, source_block.cell_kind), []
            ).append(
                (
                    int(identifier),
                    int(identifier),
                    _fraction_reference_corners(target_block.cell_kind),
                )
            )
    result = []
    for (target_kind, source_kind), rows in sorted(grouped.items()):
        rows.sort(key=lambda value: value[0])
        exact = tuple(value[2] for value in rows)
        execution = np.asarray(
            [[[float(value) for value in point] for point in cell] for cell in exact],
            dtype=np.float64,
        )
        witness = ReferenceCellMapWitnesses(
            np.asarray([value[0] for value in rows], dtype=np.int64),
            np.asarray([value[1] for value in rows], dtype=np.int64),
            target_kind,
            source_kind,
            execution,
            exact,
        )
        exact_reference_cell_vertices(witness)
        result.append(witness)
    return tuple(result)


def _bisection_coarsening_witness(
    source: CellMesh,
    target: CellMesh,
    edit: CellTopologyEdit,
    hierarchy: BisectionHierarchy | None,
    /,
) -> NestedReferenceWitnesses | None:
    """Compose exact bisection midpoint ancestry from active leaves to restored parents."""
    if hierarchy is None:
        return None
    if (
        hierarchy.dimension != source.topological_dimension
        or source.topological_dimension not in (2, 3)
    ):
        raise ValueError(
            "Bisection hierarchy dimension differs from its coarsening mesh."
        )
    relation = next(
        (
            value
            for value in edit.relations
            if value.dimension == source.topological_dimension
        ),
        None,
    )
    if relation is None:
        raise ValueError("Coarsening topology edit lacks top-cell lineage.")
    source_ids = np.asarray(relation.source_keys, dtype=np.int64)[:, 0]
    target_ids = np.asarray(relation.target_keys, dtype=np.int64)[:, 0]
    kinds = np.asarray(relation.kinds, dtype=np.int32)
    selected = kinds == int(EntityLineageKind.COARSENED_INTO)
    if not np.any(selected):
        return None
    fine_to_coarse = {
        int(fine): int(coarse)
        for fine, coarse in zip(source_ids[selected], target_ids[selected], strict=True)
    }
    if len(fine_to_coarse) != int(np.count_nonzero(selected)):
        raise ValueError("A coarsened source cell maps to more than one target parent.")
    records = {
        int(parent): row
        for row, parent in enumerate(
            np.asarray(hierarchy.record_parent_ids, dtype=np.int64)
        )
    }
    if len(records) != np.asarray(hierarchy.record_parent_ids).size:
        raise ValueError("Bisection hierarchy repeats an exact parent record.")
    record_parent_vertices = np.asarray(hierarchy.record_parent_vertices, dtype=np.int64)
    record_parent_tags = np.asarray(hierarchy.record_parent_tags, dtype=np.int64)
    record_children = np.asarray(hierarchy.record_child_ids, dtype=np.int64)
    record_midpoints = np.asarray(hierarchy.record_vertex_ids, dtype=np.int64)
    if (
        record_parent_vertices.shape != (len(records), source.topological_dimension + 1)
        or record_parent_tags.shape != (len(records),)
        or record_children.shape != (len(records), 2)
        or record_midpoints.shape != (len(records),)
    ):
        raise ValueError("Bisection hierarchy exact parent/child banks are misaligned.")
    subdivisions = {
        int(parent): row
        for row, parent in enumerate(
            np.asarray(hierarchy.subdivision_parent_ids, dtype=np.int64)
        )
    }
    subdivision_rows = np.sort(
        np.asarray(hierarchy.subdivision_parent_rows, dtype=np.int64), axis=1
    )
    subdivision_children = np.asarray(hierarchy.subdivision_child_ids, dtype=np.int64)
    subdivision_vertices = np.asarray(hierarchy.subdivision_vertex_ids, dtype=np.int64)
    subsets = _SUBDIVISION[source.topological_dimension][0]
    target_cells = {
        int(identifier): (block, row)
        for block in target.blocks
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    source_cells = {
        int(identifier): (block, row)
        for block in source.blocks
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    fine_reference: dict[int, tuple[tuple[Fraction, ...], ...]] = {}
    for coarse_id in sorted(set(fine_to_coarse.values())):
        root = target_cells.get(coarse_id)
        record_row = records.get(coarse_id, subdivisions.get(coarse_id))
        if root is None or record_row is None:
            raise ValueError("Coarsening target is not a retained refinement parent.")
        root_block, root_row = root
        if root_block.cell_kind not in ("triangle", "tetrahedron", "simplex:3"):
            raise ValueError("Exact bisection coarsening maps require simplex parents.")
        root_vertices = source_vertex_ids[
            np.asarray(root_block.vertices[root_row], dtype=np.int64)
        ]
        topology = reference_cell_topology(root_block.cell_kind)
        reference_by_vertex = {
            int(vertex): tuple(Fraction(float(value)) for value in reference)
            for vertex, reference in zip(root_vertices, topology.vertices, strict=True)
        }
        parent_queue = [coarse_id]
        visited: set[int] = set()
        while parent_queue:
            parent_id = parent_queue.pop()
            if parent_id in visited:
                continue
            visited.add(parent_id)
            row = records.get(parent_id)
            if row is None:
                row = subdivisions.get(parent_id)
                if row is None:
                    continue
                parent_corners: list[tuple[Fraction, ...]] = []
                for vertex in subdivision_rows[row]:
                    corner = reference_by_vertex.get(int(vertex))
                    if corner is None:
                        raise ValueError(
                            "Subdivision corner lacks exact parent coordinates."
                        )
                    parent_corners.append(corner)
                for subset, created in zip(
                    subsets, subdivision_vertices[row], strict=True
                ):
                    share = Fraction(1, len(subset))
                    created_reference = tuple(
                        share * sum(values, Fraction(0))
                        for values in zip(
                            *(parent_corners[position] for position in subset),
                            strict=True,
                        )
                    )
                    previous = reference_by_vertex.get(int(created))
                    if previous is not None and previous != created_reference:
                        raise ValueError(
                            "Bisection hierarchy assigns one vertex two exact parent "
                            "locations."
                        )
                    reference_by_vertex[int(created)] = created_reference
                parent_queue.extend(int(value) for value in subdivision_children[row])
                continue
            tag = int(record_parent_tags[row])
            vertices = record_parent_vertices[row]
            positions = bisection_label_tables(source.topological_dimension).position
            if not 1 <= tag < positions.size:
                raise ValueError("Bisection parent tag is outside the label tables.")
            first = int(vertices[0])
            second = int(vertices[positions[tag]])
            first_reference = reference_by_vertex.get(first)
            second_reference = reference_by_vertex.get(second)
            if first_reference is None or second_reference is None:
                raise ValueError(
                    "Bisection midpoint endpoint lacks exact parent coordinates."
                )
            midpoint = int(record_midpoints[row])
            midpoint_reference = tuple(
                (left + right) / 2
                for left, right in zip(first_reference, second_reference, strict=True)
            )
            previous = reference_by_vertex.get(midpoint)
            if previous is not None and previous != midpoint_reference:
                raise ValueError(
                    "Bisection hierarchy assigns one vertex two exact parent locations."
                )
            reference_by_vertex[midpoint] = midpoint_reference
            parent_queue.extend(int(value) for value in record_children[row])
        for fine_id, parent_id in fine_to_coarse.items():
            if parent_id != coarse_id:
                continue
            fine = source_cells.get(fine_id)
            if fine is None:
                raise ValueError("Coarsening lineage names an absent active source cell.")
            fine_block, fine_row = fine
            if fine_block.cell_kind != root_block.cell_kind:
                raise ValueError("Bisection coarsening changed its simplex family.")
            corners = source_vertex_ids[
                np.asarray(fine_block.vertices[fine_row], dtype=np.int64)
            ]
            try:
                fine_reference[fine_id] = tuple(
                    reference_by_vertex[int(vertex)] for vertex in corners
                )
            except KeyError as error:
                raise ValueError(
                    "A coarsened fine corner lacks exact hierarchy ancestry."
                ) from error
    fine_ids = np.asarray(sorted(fine_reference), dtype=np.int64)
    coarse_ids = np.asarray(
        [fine_to_coarse[int(value)] for value in fine_ids], dtype=np.int64
    )
    exact = tuple(fine_reference[int(value)] for value in fine_ids)
    execution = np.asarray(
        [[[float(value) for value in point] for point in cell] for cell in exact],
        dtype=np.float64,
    )
    witness = NestedReferenceWitnesses(
        fine_ids,
        coarse_ids,
        execution,
        fine_reference_exact_vertices=exact,
    )
    validate_nested_reference_witnesses(witness, dimension=source.topological_dimension)
    return witness


def _identity_reference_maps_for_transition(
    source: CellMesh, target: CellMesh, /
) -> tuple[ReferenceCellMapWitnesses, ...]:
    """Bind exact same-cell local reference charts across a fixed topology."""
    if source.topology_id != target.topology_id:
        raise ValueError("Identity reference maps require identical cell topology.")
    source_blocks = {block.name: block for block in source.blocks}
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    result = []
    for target_block in target.blocks:
        source_block = source_blocks.get(target_block.name)
        if source_block is None or source_block.cell_kind != target_block.cell_kind:
            raise ValueError(
                "Identity reference maps require matching endpoint cell blocks."
            )
        source_ids = np.asarray(source_block.global_ids, dtype=np.int64)
        target_ids = np.asarray(target_block.global_ids, dtype=np.int64)
        source_corners = source_vertex_ids[
            np.asarray(source_block.vertices, dtype=np.int64)
        ]
        target_corners = target_vertex_ids[
            np.asarray(target_block.vertices, dtype=np.int64)
        ]
        if not np.array_equal(source_ids, target_ids) or not np.array_equal(
            source_corners, target_corners
        ):
            raise ValueError(
                "Identity reference maps require unchanged stable cell-corner identity."
            )
        topology = reference_cell_topology(target_block.cell_kind)
        exact = tuple(
            tuple(Fraction(float(value)) for value in point)
            for point in topology.vertices
        )
        exact_bank = tuple(exact for _ in range(target_block.cell_count))
        result.append(
            ReferenceCellMapWitnesses(
                target_ids,
                source_ids,
                target_block.cell_kind,
                source_block.cell_kind,
                np.asarray(
                    [
                        [[float(value) for value in point] for point in cell]
                        for cell in exact_bank
                    ],
                    dtype=np.float64,
                ),
                exact_bank,
            )
        )
    return tuple(result)


class _RootParent(NamedTuple):
    """Root cell of one source cell under a retained root restriction."""

    cell_id: int
    block_name: str | None
    vertex_ids: np.ndarray


class _CoarsenedChart(NamedTuple):
    """Exact restored-parent→root chart of one coarsened target cell."""

    root: ScalarCoordinateElement
    route: np.ndarray
    parent: _RootParent
    root_kind: str
    arguments: tuple[Any, ...]
    exact_points: tuple[tuple[Fraction, ...], ...]


def _fraction_reference_corners(cell_kind: str, /) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology(cell_kind).vertices
    )


def _source_root_parents(source: CellMeshingResult, /) -> dict[int, _RootParent] | None:
    """Root parent of every source cell when the source map is itself a restriction."""
    restriction = source.geometry.restriction_source
    if restriction is None:
        return None
    parent_ids = restriction.block_parent_cell_ids
    parent_vertices = restriction.block_parent_vertex_ids
    root_blocks = restriction.block_source_blocks
    result: dict[int, _RootParent] = {}
    for block in source.mesh.blocks:
        if block.name not in parent_ids:
            raise ValueError("Source restriction ancestry must name every source block.")
        root_block = None if root_blocks is None else root_blocks[block.name]
        vertices = np.asarray(parent_vertices[block.name], dtype=np.int64)
        for row, (global_id, parent_id) in enumerate(
            zip(
                np.asarray(block.global_ids, dtype=np.int64).tolist(),
                np.asarray(parent_ids[block.name], dtype=np.int64).tolist(),
                strict=True,
            )
        ):
            result[int(global_id)] = _RootParent(
                int(parent_id), root_block, vertices[row][vertices[row] >= 0]
            )
    return result


def _coarsened_root_charts(
    source: CellMeshingResult,
    target: CellMesh,
    coarsening: NestedReferenceWitnesses,
    roots: dict[int, _RootParent] | None,
    /,
) -> dict[int, _CoarsenedChart]:
    """Restore each coarsened parent's exact root chart from its children's charts.

    Every child carries one exact chart into its root cell. Composing it with the
    inverse child→parent bisection map must give one and the same parent chart and
    root cell for all children of a family; otherwise no exact parent map exists.
    """
    if roots is None:
        raise ValueError(
            "Coarsening a non-P1 map needs the retained root restriction of its children."
        )
    elements, routes, _ = source.geometry.resolve(source.mesh)
    fine_to_root: dict[str, ReferenceCellMapWitnesses] = {}
    child_roots: dict[int, tuple[ScalarCoordinateElement, np.ndarray]] = {}
    for block, element, route in zip(source.mesh.blocks, elements, routes, strict=True):
        if not isinstance(element, ScalarCoordinateElement):
            raise ValueError("Source coordinate map has no exact scalar ancestry.")
        corners = _fraction_reference_corners(block.cell_kind)
        route_rows = np.asarray(route, dtype=np.int64)
        charts = []
        exact = []
        root_ids = []
        root_kinds = set()
        for row, identifier in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            root, chart = coordinate_reference_chain(element, cell_row=row)
            charts.append(chart)
            exact.append(
                tuple(
                    evaluate_reference_chart_arguments(block.cell_kind, chart, point)
                    for point in corners
                )
            )
            root_ids.append(roots[int(identifier)].cell_id)
            root_kinds.add(root.cell_kind)
            child_roots[int(identifier)] = (root, route_rows[row])
        if len(root_kinds) != 1:
            raise ValueError("One source block cannot mix root reference-cell kinds.")
        if any(
            not isinstance(value, Fraction)
            for cell in exact
            for point in cell
            for value in point
        ):
            raise ValueError("Source root-chart corners must be exact rationals.")
        exact_bank = tuple(
            tuple(
                tuple(value for value in point if isinstance(value, Fraction))
                for point in cell
            )
            for cell in exact
        )
        fine_to_root[block.name] = ReferenceCellMapWitnesses(
            np.arange(block.cell_count, dtype=np.int64),
            np.asarray(root_ids, dtype=np.int64),
            block.cell_kind,
            next(iter(root_kinds)),
            np.asarray(
                [
                    [[float(value) for value in point] for point in cell]
                    for cell in exact_bank
                ],
                dtype=np.float64,
            ),
            exact_bank,
            freeze_reference_chart_arguments(charts),
        )
    coarsened = coarsen_reference_cell_maps(coarsening, fine_to_root, source.mesh, target)
    children: dict[int, list[int]] = {}
    for fine_id, coarse_id in zip(
        np.asarray(coarsening.fine_cell_ids, dtype=np.int64).tolist(),
        np.asarray(coarsening.coarse_cell_ids, dtype=np.int64).tolist(),
        strict=True,
    ):
        children.setdefault(int(coarse_id), []).append(int(fine_id))
    result: dict[int, _CoarsenedChart] = {}
    for block in target.blocks:
        witness = coarsened.get(block.name)
        if witness is None:
            continue
        arguments = witness.source_reference_chart_arguments
        if arguments is None:
            raise ValueError("Coarsened root charts must retain exact chart arguments.")
        exact_bank = exact_reference_cell_vertices(witness)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        for row, (local, root_id) in enumerate(
            zip(
                np.asarray(witness.target_cell_ids, dtype=np.int64).tolist(),
                np.asarray(witness.source_cell_ids, dtype=np.int64).tolist(),
                strict=True,
            )
        ):
            parent_id = int(identifiers[int(local)])
            family = children[parent_id]
            root, route = child_roots[family[0]]
            if any(
                child_roots[child][0].element_id != root.element_id
                or not np.array_equal(child_roots[child][1], route)
                for child in family[1:]
            ):
                raise ValueError(
                    f"Coarsened cell {parent_id} children bind different root coordinate dofs."
                )
            parent = roots[family[0]]
            if parent.cell_id != int(root_id):
                raise ValueError(
                    "Coarsened root chart disagrees with child root ancestry."
                )
            result[parent_id] = _CoarsenedChart(
                root,
                route,
                parent,
                witness.source_cell_kind,
                tuple(arguments[row]),
                exact_bank[row],
            )
    return result


def _coarsened_corner_coordinates(
    source: CellMeshingResult,
    target: CellMesh,
    coarse: dict[int, _CoarsenedChart],
    coarsening: NestedReferenceWitnesses,
    /,
) -> dict[int, tuple[tuple[Any, ...], ...]]:
    """Exact physical corners of restored parents, read at their children's corners."""
    parents = set(coarse)
    child_ids = [
        int(fine_id)
        for fine_id, coarse_id in zip(
            np.asarray(coarsening.fine_cell_ids, dtype=np.int64).tolist(),
            np.asarray(coarsening.coarse_cell_ids, dtype=np.int64).tolist(),
            strict=True,
        )
        if int(coarse_id) in parents
    ]
    source_vertex_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    child_rows: dict[int, tuple[str, np.ndarray]] = {}
    for block in source.mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            child_rows[int(identifier)] = (
                block.cell_kind,
                source_vertex_ids[vertices[row]],
            )
    images = source.geometry.evaluate_exact(
        source.mesh,
        np.asarray(child_ids, dtype=np.int64),
        tuple(_fraction_reference_corners(child_rows[child][0]) for child in child_ids),
    )
    vertex_coordinates: dict[int, tuple[Any, ...]] = {}
    for child, corners in zip(child_ids, images, strict=True):
        for vertex_id, coordinate in zip(
            child_rows[child][1].tolist(), corners, strict=True
        ):
            vertex_coordinates[int(vertex_id)] = tuple(coordinate)
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    result: dict[int, tuple[tuple[Any, ...], ...]] = {}
    for block in target.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            if int(identifier) not in parents:
                continue
            corner_ids = target_vertex_ids[vertices[row]].tolist()
            if any(int(vertex_id) not in vertex_coordinates for vertex_id in corner_ids):
                raise ValueError(
                    "A restored parent corner is not a corner of its children."
                )
            result[int(identifier)] = tuple(
                vertex_coordinates[int(vertex_id)] for vertex_id in corner_ids
            )
    return result


def _quotient_chart_bank(
    source: CellMeshingResult,
    target: CellMesh,
    exact_rows: tuple[tuple[Any, ...], ...] | None,
    /,
) -> ExactPlcCellGeometrySource | None:
    """Exact vertex bank of a periodic P1 target whose vertices are not binary64.

    Every exact row is a quotient-chart point (exact root position translated
    exactly by its winding); the target carries the canonical roundings. Without
    such a row the binary64 carrier is itself exact and needs no bank.
    """
    if exact_rows is None or all(
        Fraction(float(value)) == value for row in exact_rows for value in row
    ):
        return None
    vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    return ExactPlcCellGeometrySource(
        "periodic-quotient-chart",
        source.mesh.mesh_id,
        vertex_ids,
        exact_rows,
        vertex_source_entity_ids=(None,) * vertex_ids.size,
        vertex_occurrence_paths=((),) * vertex_ids.size,
    )


def _quotient_chart_target_geometry(
    target: CellMesh, bank: ExactPlcCellGeometrySource | None, /
) -> CellGeometrySpec:
    """Publish a periodic P1 target on its lifted vertex-indexed carrier."""
    affine = CellGeometrySpec.affine(target)
    if bank is None:
        return affine
    return CellGeometrySpec(
        dict(zip(affine.block_names, affine.elements, strict=True)),
        {
            name: np.asarray(route)
            for name, route in zip(affine.block_names, affine.geometry_dofs, strict=True)
        },
        target.coordinates,
        source_map=bank,
        source_mesh=target,
    )


def _restricted_target_geometry(
    source: CellMeshingResult,
    target: CellMesh,
    edit: CellTopologyEdit,
    /,
    *,
    coarsening: NestedReferenceWitnesses | None = None,
) -> tuple[
    CellMesh,
    CellGeometrySpec,
    tuple[ReferenceCellMapWitnesses, ...],
    tuple[tuple[Any, ...], ...] | None,
]:
    """Restrict an exact source coordinate map through target→source cell maps.

    The last item is the exact target vertex bank (target vertex order) of a
    periodic P1 source, whose target is published on its lifted vertex-indexed
    carrier once the caller binds the quotient; it is ``None`` otherwise.
    """
    if (
        coarsening is None
        and source.mesh.topology_id == target.topology_id
        and _piecewise_linear_geometry(source.mesh, source.geometry)
    ):
        target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
        edit_vertex_ids = np.asarray(edit.vertex_global_ids, dtype=np.int64)
        stencil_valid = np.asarray(edit.stencil_valid, dtype=np.bool_)
        if (
            edit_vertex_ids.shape != target_vertex_ids.shape
            or not np.array_equal(np.sort(edit_vertex_ids), np.sort(target_vertex_ids))
            or stencil_valid.ndim != 2
            or stencil_valid.shape[0] != edit_vertex_ids.size
        ):
            raise ValueError(
                "Fixed-topology vertex ancestry does not cover the target vertices."
            )
        stencil_rows = {
            int(identifier): row for row, identifier in enumerate(edit_vertex_ids)
        }
        if any(
            not np.any(stencil_valid[stencil_rows[int(identifier)]])
            for identifier in target_vertex_ids
        ):
            return target, CellGeometrySpec.affine(target), (), None
        reference_maps = _identity_reference_maps_for_transition(source.mesh, target)
        return target, CellGeometrySpec.affine(target), reference_maps, None
    try:
        reference_maps = reference_cell_maps_from_edit(source.mesh, target, edit)
    except ValueError:
        try:
            reference_maps = _available_reference_maps(source.mesh, target, edit)
        except ValueError:
            if _piecewise_linear_geometry(source.mesh, source.geometry):
                return target, CellGeometrySpec.affine(target), (), None
            raise ValueError(
                "A non-P1 source map needs exact reference ancestry for every target cell."
            ) from None
        if (
            not _piecewise_linear_geometry(source.mesh, source.geometry)
            and coarsening is None
        ):
            raise ValueError(
                "A non-P1 source map needs exact reference ancestry for every target cell."
            ) from None
    if coarsening is not None and _piecewise_linear_geometry(
        source.mesh, source.geometry
    ):
        # A restored P1 parent is affine on its own exact corners; its children are
        # distinct roots, so no single root chart exists or is needed.
        return target, CellGeometrySpec.affine(target), reference_maps, None
    roots = _source_root_parents(source)
    coarse = (
        {}
        if coarsening is None
        else _coarsened_root_charts(source, target, coarsening, roots)
    )
    coarse_corners = (
        {}
        if coarsening is None or not coarse
        else _coarsened_corner_coordinates(source, target, coarse, coarsening)
    )
    records = _reference_map_rows(reference_maps)
    if set(records) & set(coarse):
        raise ValueError("Reference maps assign one target cell more than once.")
    target_ids = {
        int(identifier)
        for block in target.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }
    if set(records) | set(coarse) != target_ids:
        if _piecewise_linear_geometry(source.mesh, source.geometry):
            return target, CellGeometrySpec.affine(target), reference_maps, None
        raise ValueError("A non-P1 source map needs an exact map for every target cell.")
    source_cell_rows = {
        int(identifier): (block_index, row)
        for block_index, block in enumerate(source.mesh.blocks)
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    source_elements, source_routes, _ = source.geometry.resolve(source.mesh)
    source_vertex_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    target_elements: dict[str, Any] = {}
    target_routes: dict[str, np.ndarray] = {}
    target_parents: dict[str, np.ndarray] = {}
    target_parent_vertices: dict[str, np.ndarray] = {}
    target_parent_maps: dict[str, ReferenceCellMapWitnesses] = {}
    target_source_blocks: dict[str, str | None] = {}
    exact_vertex_coordinates: dict[int, tuple[Any, ...]] = {}
    for target_block in target.blocks:
        identifiers = np.asarray(target_block.global_ids, dtype=np.int64)
        target_cells = np.asarray(target_block.vertices, dtype=np.int64)
        target_corners = _fraction_reference_corners(target_block.cell_kind)
        block_roots: list[ScalarCoordinateElement] = []
        charts: list[tuple[ExpressionInput, ...]] = []
        parent_ids: list[int] = []
        parent_vertices: list[np.ndarray] = []
        parent_blocks: list[str | None] = []
        routes = []
        exact_cells = []
        parent_kinds = set()
        # Exact source images of target corners, evaluated for the whole block
        # at once: each evaluation resolves the source map over every cell.
        corner_images: dict[int, tuple[tuple[Any, ...], ...]] = {}
        pending_rows: list[int] = []
        pending_sources: list[int] = []
        pending_points: list[tuple[tuple[Fraction, ...], ...]] = []
        for row, target_id in enumerate(identifiers.tolist()):
            restored = coarse.get(int(target_id))
            if restored is not None:
                block_roots.append(restored.root)
                charts.append(restored.arguments)
                routes.append(restored.route)
                parent_ids.append(restored.parent.cell_id)
                parent_vertices.append(restored.parent.vertex_ids)
                parent_blocks.append(restored.parent.block_name)
                parent_kinds.add(restored.root_kind)
                exact_cells.append(restored.exact_points)
                corner_images[row] = coarse_corners[int(target_id)]
            else:
                witness, map_row, exact_points = records[int(target_id)]
                source_id = int(
                    np.asarray(witness.source_cell_ids, dtype=np.int64)[map_row]
                )
                owner = source_cell_rows.get(source_id)
                if owner is None:
                    raise ValueError("A reference map names an absent source cell.")
                source_block_index, source_row = owner
                source_block = source.mesh.blocks[source_block_index]
                source_element = source_elements[source_block_index]
                if not isinstance(source_element, ScalarCoordinateElement):
                    raise ValueError(
                        "Source coordinate map has no exact scalar ancestry."
                    )
                root, source_chart = coordinate_reference_chain(
                    source_element, cell_row=source_row
                )
                chart = _reference_chart_expressions(target_block.cell_kind, exact_points)
                composed = tuple(
                    compose_expression(value, chart) for value in source_chart
                )
                block_roots.append(root)
                charts.append(composed)
                routes.append(np.asarray(source_routes[source_block_index])[source_row])
                if roots is None:
                    # The source is the root: ancestry is the target→source map itself.
                    parent_ids.append(source_id)
                    parent_vertices.append(
                        source_vertex_ids[
                            np.asarray(source_block.vertices[source_row], dtype=np.int64)
                        ]
                    )
                    parent_blocks.append(source_block.name)
                    parent_kinds.add(witness.source_cell_kind)
                    exact_cells.append(exact_points)
                else:
                    # A restricted source composes its root chart, so ancestry stays
                    # bound to the retained root instead of the intermediate mesh.
                    parent = roots[source_id]
                    parent_ids.append(parent.cell_id)
                    parent_vertices.append(parent.vertex_ids)
                    parent_blocks.append(parent.block_name)
                    parent_kinds.add(root.cell_kind)
                    root_points = tuple(
                        evaluate_reference_chart_arguments(
                            target_block.cell_kind, composed, point
                        )
                        for point in target_corners
                    )
                    if any(
                        not isinstance(value, Fraction)
                        for point in root_points
                        for value in point
                    ):
                        raise ValueError(
                            "Composed root-chart corners must be exact rationals."
                        )
                    exact_cells.append(
                        tuple(
                            tuple(value for value in point if isinstance(value, Fraction))
                            for point in root_points
                        )
                    )
                pending_rows.append(row)
                pending_sources.append(source_id)
                pending_points.append(exact_points)
        if pending_rows:
            corner_images.update(
                zip(
                    pending_rows,
                    source.geometry.evaluate_exact(
                        source.mesh,
                        np.asarray(pending_sources, dtype=np.int64),
                        tuple(pending_points),
                    ),
                    strict=True,
                )
            )
        for row in range(identifiers.size):
            for vertex_id, coordinate in zip(
                target_vertex_ids[target_cells[row]].tolist(),
                corner_images[row],
                strict=True,
            ):
                previous = exact_vertex_coordinates.get(int(vertex_id))
                if previous is not None and previous != tuple(coordinate):
                    raise ValueError(
                        "Adjacent exact source restrictions disagree at a shared target vertex."
                    )
                exact_vertex_coordinates[int(vertex_id)] = tuple(coordinate)
        if len({root.element_id for root in block_roots}) != 1:
            raise ValueError(
                "One target block cannot mix unrelated source coordinate roots."
            )
        if len(parent_kinds) != 1:
            raise ValueError("One target block cannot mix source reference-cell kinds.")
        root = block_roots[0]
        route_array = np.asarray(routes, dtype=np.int32)
        if route_array.shape[1] != root.local_dof_count:
            raise ValueError(
                "Exact source restriction changed the coordinate basis width."
            )
        target_elements[target_block.name] = PerCellRestrictedCellGeometryElement(
            root,
            target_block.cell_kind,
            identifiers,
            parent_ids,
            tuple(charts),
        )
        target_routes[target_block.name] = route_array
        target_parents[target_block.name] = np.asarray(parent_ids, dtype=np.int64)
        parent_width = max(len(value) for value in parent_vertices)
        padded = np.full((len(parent_vertices), parent_width), -1, dtype=np.int64)
        for row, values in enumerate(parent_vertices):
            padded[row, : len(values)] = values
        target_parent_vertices[target_block.name] = padded
        target_source_blocks[target_block.name] = parent_blocks[0]
        exact_values = tuple(exact_cells)
        target_parent_maps[target_block.name] = ReferenceCellMapWitnesses(
            np.arange(identifiers.size, dtype=np.int64),
            np.asarray(parent_ids, dtype=np.int64),
            target_block.cell_kind,
            next(iter(parent_kinds)),
            np.asarray(
                [
                    [[float(value) for value in point] for point in cell]
                    for cell in exact_values
                ],
                dtype=np.float64,
            ),
            exact_values,
            None if roots is None else freeze_reference_chart_arguments(charts),
        )
    if set(exact_vertex_coordinates) != set(int(value) for value in target_vertex_ids):
        raise ValueError("Exact target maps do not cover every target coordinate vertex.")
    target_coordinates = np.asarray(
        [
            [float(value) for value in exact_vertex_coordinates[int(identifier)]]
            for identifier in target_vertex_ids
        ],
        dtype=np.float64,
    )
    if not np.array_equal(target_coordinates, np.asarray(target.coordinates)):
        target = CellMesh(
            target_coordinates,
            target.blocks,
            vertex_global_ids=target.vertex_global_ids,
            entity_global_ids={
                dimension: np.asarray(
                    target.entity_set(dimension).entity_ids, dtype=np.int64
                )
                for dimension in range(target.topological_dimension + 1)
            },
            numeric_version=target.numeric_version,
            periodic_topology=target.periodic_topology,
        )
    if source.mesh.periodic_topology is not None and _piecewise_linear_geometry(
        source.mesh, source.geometry
    ):
        exact_rows = tuple(
            exact_vertex_coordinates[int(identifier)] for identifier in target_vertex_ids
        )
        return target, CellGeometrySpec.affine(target), reference_maps, exact_rows
    named_source_blocks = (
        None
        if any(name is None for name in target_source_blocks.values())
        else {name: str(value) for name, value in target_source_blocks.items()}
    )
    ancestry = source.geometry.restriction_source
    if ancestry is None:
        restriction = CellGeometryRestrictionSource(
            cell_geometry_id(source.geometry),
            source.mesh.topology_id,
            target_parents,
            target_parent_vertices,
            block_source_blocks=named_source_blocks,
            block_parent_reference_maps=target_parent_maps,
        )
    else:
        restriction = CellGeometryRestrictionSource(
            ancestry.source_geometry_id,
            ancestry.source_topology_id,
            target_parents,
            target_parent_vertices,
            block_source_blocks=named_source_blocks,
            carrier_publication=ancestry.carrier_publication,
            block_parent_reference_maps=target_parent_maps,
            carrier_geometry=ancestry.carrier_geometry,
        )
    geometry = CellGeometrySpec(
        target_elements,
        target_routes,
        source.geometry.coordinates,
        restriction_source=restriction,
    )
    return target, geometry, reference_maps, None


def _transition_geometry_record(
    source: CellMeshingResult,
    target: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    reference_maps: tuple[ReferenceCellMapWitnesses, ...] = (),
    refinement: Any = None,
    coarsening: Any = None,
) -> CellGeometryTransition:
    """Bind only coordinate-map equality witnesses, not relocation charts."""
    retained_maps = reference_maps
    if (
        retained_maps
        and refinement is None
        and coarsening is None
        and source.mesh.topology_id == target.topology_id
        and _piecewise_linear_geometry(source.mesh, source.geometry)
        and _piecewise_linear_geometry(target, geometry)
        and not np.array_equal(
            np.asarray(source.mesh.coordinates, dtype=np.float64),
            np.asarray(target.coordinates, dtype=np.float64),
        )
    ):
        retained_maps = ()
    return CellGeometryTransition(
        source.mesh,
        source.geometry,
        target,
        geometry,
        evidence=CellGeometryTransitionEvidence(
            "nested_restriction"
            if retained_maps or refinement or coarsening
            else "vertex_displacement",
            reference_map_count=sum(
                np.asarray(value.target_cell_ids).size for value in retained_maps
            ),
            refinement_map_count=0
            if refinement is None
            else len(refinement.fine_cell_ids),
            coarsening_map_count=0
            if coarsening is None
            else len(coarsening.fine_cell_ids),
        ),
        reference_map=retained_maps or None,
        refinement=refinement,
        coarsening=coarsening,
    )


def _retarget_surface_chart_witness(
    source: CellMeshingResult,
    target: CellMesh,
    geometry: CellGeometrySpec,
    lineage: MeshLineage,
    witness: SurfaceChartWitness,
    /,
    *,
    reference_maps: tuple[ReferenceCellMapWitnesses, ...],
    coarsening: NestedReferenceWitnesses | None = None,
) -> SurfaceChartWitness:
    """Compose exact source material charts through cell lineage/reference maps."""
    if any(
        block.cell_kind != "triangle" for block in (*source.mesh.blocks, *target.blocks)
    ):
        raise ValueError(
            "Surface chart transitions require triangular source and target cells."
        )
    source_rows = {
        int(identifier): (block_index, row)
        for block_index, block in enumerate(source.mesh.blocks)
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    witness_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(witness.cell_global_ids, dtype=np.int64)
        )
    }
    maps = _reference_map_rows(reference_maps)
    source_patches = np.asarray(witness.patches, dtype=np.int64)
    source_roots = np.asarray(witness.source_root_cell_ids, dtype=np.int64)
    relation = lineage.entity_lineage(target.topological_dimension)
    coarsened: dict[int, list[int]] = {}
    for source_id, target_id, kind in zip(
        np.asarray(relation.source_global_ids, dtype=np.int64),
        np.asarray(relation.target_global_ids, dtype=np.int64),
        np.asarray(relation.relation_kinds, dtype=np.int32),
        strict=True,
    ):
        if int(kind) == int(EntityLineageKind.COARSENED_INTO):
            coarsened.setdefault(int(target_id), []).append(int(source_id))
    exact_coarsening: dict[int, tuple[int, tuple[tuple[Fraction, ...], ...]]] = {}
    if coarsening is not None:
        validate_nested_reference_witnesses(
            coarsening, dimension=target.topological_dimension
        )
        for fine_id, coarse_id, exact_points in zip(
            np.asarray(coarsening.fine_cell_ids, dtype=np.int64),
            np.asarray(coarsening.coarse_cell_ids, dtype=np.int64),
            exact_reference_vertices(coarsening),
            strict=True,
        ):
            fine, coarse = int(fine_id), int(coarse_id)
            if fine in exact_coarsening:
                raise ValueError("Exact chart coarsening repeats a source cell.")
            exact_coarsening[fine] = (coarse, exact_points)
    preserved_cells = {
        int(target_id)
        for source_id, target_id, kind in zip(
            np.asarray(relation.source_global_ids, dtype=np.int64),
            np.asarray(relation.target_global_ids, dtype=np.int64),
            np.asarray(relation.relation_kinds, dtype=np.int32),
            strict=True,
        )
        if int(kind)
        in (int(EntityLineageKind.PRESERVED), int(EntityLineageKind.RELOCATED))
    }

    target_ids: list[int] = []
    patches: list[int] = []
    charts: list[tuple[tuple[Fraction, ...], ...]] = []
    roots: list[int] = []
    root_mesh_bindings: list[SurfaceRootMeshCellBinding | None] = []
    pieces: list[tuple[int, ...]] = []
    entities: list[str] = []
    occurrences: list[tuple[str, ...]] = []
    source_vertex_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    # One fingerprint of the whole target geometry binds every cell.
    geometry_id = cell_geometry_id(geometry)
    for block in target.blocks:
        cell_vertices = np.asarray(block.vertices, dtype=np.int64)
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            cell_id = int(identifier)
            map_record = maps.get(cell_id)
            if map_record is not None:
                map_witness, map_row, exact_points = map_record
                source_id = int(
                    np.asarray(map_witness.source_cell_ids, dtype=np.int64)[map_row]
                )
                parent = witness_rows.get(source_id)
                if parent is None or source_id not in source_rows:
                    raise ValueError(
                        "A target surface cell lacks witnessed source chart ancestry."
                    )
                source_block_index, source_row = source_rows[source_id]
                source_kind = source.mesh.blocks[source_block_index].cell_kind
                basis = coordinate_source_basis(lagrange_element(source_kind, 1))
                source_chart = _exact_chart_from_tokens(
                    witness.chart_rational_tokens, parent
                )
                chart_expressions = linear_combinations(
                    basis,
                    tuple(
                        tuple(point[axis] for point in source_chart) for axis in range(2)
                    ),
                )
                corner_values = tuple(
                    tuple(
                        evaluate_expression(value, point) for value in chart_expressions
                    )
                    for point in exact_points
                )
                chart = tuple(
                    tuple(value for value in point if isinstance(value, Fraction))
                    for point in corner_values
                )
                if chart != corner_values:
                    raise ValueError(
                        "A refined surface chart left the rational source chart."
                    )
                patch = int(source_patches[parent])
                root = int(source_roots[parent])
                root_mesh_binding = witness.root_mesh_cell_bindings[parent]
                piece = witness.source_piece_ids[parent]
                entity = witness.geometry_entity_ids[parent]
                occurrence = witness.occurrence_paths[parent]
            elif cell_id in witness_rows and cell_id in preserved_cells:
                parent = witness_rows[cell_id]
                if cell_id not in source_rows:
                    raise ValueError(
                        "Preserved chart cell ID is absent from its source mesh."
                    )
                chart = _exact_chart_from_tokens(witness.chart_rational_tokens, parent)
                patch = int(source_patches[parent])
                root = int(source_roots[parent])
                root_mesh_binding = witness.root_mesh_cell_bindings[parent]
                piece = witness.source_piece_ids[parent]
                entity = witness.geometry_entity_ids[parent]
                occurrence = witness.occurrence_paths[parent]
            else:
                parents = coarsened.get(cell_id, ())
                if not parents:
                    raise ValueError(
                        f"Target surface cell {cell_id} has unknown chart lineage."
                    )
                nested_parents = {
                    fine
                    for fine, (coarse, _) in exact_coarsening.items()
                    if coarse == cell_id
                }
                if len(set(parents)) != len(parents) or set(parents) != nested_parents:
                    raise ValueError(
                        "Coarsened surface chart lacks complete exact reference ancestry."
                    )
                rows = [witness_rows.get(parent) for parent in parents]
                if any(value is None for value in rows):
                    raise ValueError(
                        "A coarsened surface cell lost source chart ancestry."
                    )
                parent_rows = [value for value in rows if value is not None]
                # Root-mesh bindings name each child cell itself; only their
                # presence is family identity, and the parent rebinds below.
                identities = {
                    (
                        int(source_patches[parent]),
                        int(source_roots[parent]),
                        witness.source_piece_ids[parent],
                        witness.geometry_entity_ids[parent],
                        witness.occurrence_paths[parent],
                        witness.root_mesh_cell_bindings[parent] is None,
                    )
                    for parent in parent_rows
                }
                if len(identities) != 1:
                    raise ValueError(
                        "Coarsening crosses distinct source chart pieces or occurrences."
                    )
                patch, root, piece, entity, occurrence, unbound = next(iter(identities))
                root_mesh_binding = (
                    None if unbound else witness.root_mesh_cell_bindings[parent_rows[0]]
                )
                vertex_chart: dict[int, tuple[Fraction, ...]] = {}
                for parent, parent_row in zip(parents, parent_rows, strict=True):
                    source_block_index, source_row = source_rows[parent]
                    source_block = source.mesh.blocks[source_block_index]
                    source_vertices = source_vertex_ids[
                        np.asarray(source_block.vertices[source_row], dtype=np.int64)
                    ]
                    for vertex_id, value in zip(
                        source_vertices.tolist(),
                        _exact_chart_from_tokens(
                            witness.chart_rational_tokens, parent_row
                        ),
                        strict=True,
                    ):
                        previous = vertex_chart.get(int(vertex_id))
                        if previous is not None and previous != value:
                            raise ValueError(
                                "Neighboring source chart cells disagree at a shared vertex."
                            )
                        vertex_chart[int(vertex_id)] = value
                target_vertices = target_vertex_ids[cell_vertices[row]]
                try:
                    chart = tuple(vertex_chart[int(value)] for value in target_vertices)
                except KeyError as error:
                    raise ValueError(
                        "A coarsened chart corner has no exact source vertex ancestry."
                    ) from error
                parent_chart = _reference_chart_expressions(block.cell_kind, chart)
                for parent in parents:
                    _, child_to_parent = exact_coarsening[parent]
                    child_map = _reference_chart_expressions(
                        block.cell_kind, child_to_parent
                    )
                    child_chart = tuple(
                        compose_expression(value, child_map) for value in parent_chart
                    )
                    parent_row = witness_rows[parent]
                    for local_point, observed in zip(
                        reference_cell_topology(block.cell_kind).vertices,
                        _exact_chart_from_tokens(
                            witness.chart_rational_tokens, parent_row
                        ),
                        strict=True,
                    ):
                        exact_local_point = tuple(
                            Fraction(float(value)) for value in local_point
                        )
                        predicted = tuple(
                            evaluate_expression(value, exact_local_point)
                            for value in child_chart
                        )
                        if predicted != observed:
                            raise ValueError(
                                "Coarsened surface chart is not an exact restriction "
                                "of its source children."
                            )
            target_ids.append(cell_id)
            patches.append(patch)
            charts.append(chart)
            roots.append(root)
            # Root-mesh ancestry rebinds to the actual target cell, as for the root.
            root_mesh_bindings.append(
                None
                if root_mesh_binding is None
                else SurfaceRootMeshCellBinding(geometry_id, target.topology_id, cell_id)
            )
            pieces.append(tuple(piece))
            entities.append(entity)
            occurrences.append(tuple(occurrence))
    return SurfaceChartWitness(
        geometry_id=geometry_id,
        topology_id=target.topology_id,
        domain_id=witness.domain_id,
        cell_global_ids=np.asarray(target_ids, dtype=np.int64),
        patches=np.asarray(patches, dtype=np.int32),
        charts=np.asarray(
            [[[float(value) for value in point] for point in chart] for chart in charts],
            dtype=np.float64,
        ),
        chart_rational_tokens=surface_chart_tokens(tuple(charts)),
        source_root_cell_ids=np.asarray(roots, dtype=np.int64),
        source_piece_ids=tuple(pieces),
        geometry_entity_ids=tuple(entities),
        occurrence_paths=tuple(occurrences),
        root_mesh_cell_bindings=tuple(root_mesh_bindings),
    )


def _simplex_chart_weights(
    point: tuple[Fraction, ...],
    vertices: tuple[tuple[Fraction, ...], ...],
    dimension: int,
    /,
) -> tuple[Fraction, ...] | None:
    if len(vertices) != dimension + 1 or any(
        len(value) != len(point) for value in vertices
    ):
        return None
    origin = vertices[0]
    if dimension == 1:
        direction = tuple(vertices[1][axis] - origin[axis] for axis in range(len(point)))
        axis = next((index for index, value in enumerate(direction) if value), None)
        if axis is None:
            return None
        weight = (point[axis] - origin[axis]) / direction[axis]
        if any(
            origin[index] + weight * direction[index] != point[index]
            for index in range(len(point))
        ):
            return None
        weights = (Fraction(1) - weight, weight)
        return weights if min(weights) >= 0 else None
    if dimension == 3:
        if len(point) != 3:
            return None
        first = tuple(vertices[1][axis] - origin[axis] for axis in range(3))
        second = tuple(vertices[2][axis] - origin[axis] for axis in range(3))
        third = tuple(vertices[3][axis] - origin[axis] for axis in range(3))
        delta = tuple(point[axis] - origin[axis] for axis in range(3))

        def determinant(
            a: tuple[Fraction, ...], b: tuple[Fraction, ...], c: tuple[Fraction, ...]
        ) -> Fraction:
            return (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - b[0] * (a[1] * c[2] - a[2] * c[1])
                + c[0] * (a[1] * b[2] - a[2] * b[1])
            )

        denominator = determinant(first, second, third)
        if not denominator:
            return None
        first_weight = determinant(delta, second, third) / denominator
        second_weight = determinant(first, delta, third) / denominator
        third_weight = determinant(first, second, delta) / denominator
        if any(
            origin[axis]
            + first_weight * first[axis]
            + second_weight * second[axis]
            + third_weight * third[axis]
            != point[axis]
            for axis in range(3)
        ):
            return None
        weights = (
            Fraction(1) - first_weight - second_weight - third_weight,
            first_weight,
            second_weight,
            third_weight,
        )
        return weights if min(weights) >= 0 else None
    if dimension != 2:
        return None
    first = tuple(vertices[1][axis] - origin[axis] for axis in range(len(point)))
    second = tuple(vertices[2][axis] - origin[axis] for axis in range(len(point)))
    delta = tuple(point[axis] - origin[axis] for axis in range(len(point)))
    for first_axis in range(len(point)):
        for second_axis in range(first_axis + 1, len(point)):
            determinant = (
                first[first_axis] * second[second_axis]
                - first[second_axis] * second[first_axis]
            )
            if not determinant:
                continue
            first_weight = (
                delta[first_axis] * second[second_axis]
                - delta[second_axis] * second[first_axis]
            ) / determinant
            second_weight = (
                first[first_axis] * delta[second_axis]
                - first[second_axis] * delta[first_axis]
            ) / determinant
            if any(
                origin[axis] + first_weight * first[axis] + second_weight * second[axis]
                != point[axis]
                for axis in range(len(point))
            ):
                continue
            weights = (
                Fraction(1) - first_weight - second_weight,
                first_weight,
                second_weight,
            )
            return weights if min(weights) >= 0 else None
    return None


def _chart_orientation(parameters: Any, dimension: int, /) -> int:
    values = tuple(
        tuple(
            value if isinstance(value, Fraction) else Fraction(float(value))
            for value in row
        )
        for row in parameters
    )
    if dimension == 1:
        delta = values[1][0] - values[0][0]
    else:
        delta = (values[1][0] - values[0][0]) * (values[2][1] - values[0][1]) - (
            values[1][1] - values[0][1]
        ) * (values[2][0] - values[0][0])
    return 1 if delta > 0 else -1 if delta < 0 else 0


def _retarget_polyhedral_chart_by_identity(
    source: CellMeshingResult,
    target: CellMesh,
    geometry: CellGeometrySpec,
    lineage: MeshLineage,
    witness: SourceBoundaryChartWitness,
    /,
) -> SourceBoundaryChartWitness:
    if (
        source.mesh.topology_id != lineage.source_topology_id
        or target.topology_id != lineage.target_topology_id
    ):
        raise ValueError("Boundary chart transition lineage endpoints are stale.")
    if (
        source.mesh.topological_dimension != 3
        or target.topological_dimension != 3
        or any(block.cell_kind != "polyhedron" for block in source.mesh.blocks)
        or any(block.cell_kind != "polyhedron" for block in target.blocks)
        or witness.chart_dimension != 2
        or witness.reference_dimension != 2
    ):
        raise ValueError(
            "Polyhedral chart identity transfer requires 3D polyhedral endpoints."
        )
    face_vertices = np.asarray(witness.target_face_vertex_global_ids, dtype=np.int64)
    if face_vertices.shape != (len(witness.source_chart_ids), 3):
        raise ValueError(
            "Polyhedral chart transfer requires actual source face-triangle IDs."
        )
    if source.mesh.topology_id != target.topology_id or not np.array_equal(
        np.asarray(source.mesh.vertex_global_ids, dtype=np.int64),
        np.asarray(target.vertex_global_ids, dtype=np.int64),
    ):
        raise ValueError(
            "Polyhedral chart transition lacks exact changed-vertex ancestry."
        )

    def retained_ids(dimension: int) -> set[int]:
        relation = lineage.entity_lineage(dimension)
        return {
            int(target_id)
            for source_id, target_id, kind in zip(
                np.asarray(relation.source_global_ids, dtype=np.int64),
                np.asarray(relation.target_global_ids, dtype=np.int64),
                np.asarray(relation.relation_kinds, dtype=np.int32),
                strict=True,
            )
            if int(source_id) == int(target_id)
            and int(kind)
            in (int(EntityLineageKind.PRESERVED), int(EntityLineageKind.RELOCATED))
        }

    witnessed_cells = set(np.asarray(witness.cell_global_ids, dtype=np.int64).tolist())
    witnessed_faces = set(np.asarray(witness.entity_global_ids, dtype=np.int64).tolist())
    witnessed_vertices = set(face_vertices.reshape((-1,)).tolist())
    if not witnessed_cells <= retained_ids(3):
        raise ValueError("Polyhedral chart owner cells lack exact identity lineage.")
    if not witnessed_faces <= retained_ids(2):
        raise ValueError("Polyhedral chart faces lack exact identity lineage.")
    if not witnessed_vertices <= retained_ids(0):
        raise ValueError("Polyhedral chart vertices lack exact identity lineage.")
    return SourceBoundaryChartWitness(
        mesh_id=target.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=target.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        chart_dimension=witness.chart_dimension,
        reference_dimension=witness.reference_dimension,
        cell_global_ids=np.asarray(witness.cell_global_ids),
        entity_global_ids=np.asarray(witness.entity_global_ids),
        local_entity_indices=np.asarray(witness.local_entity_indices),
        target_reference_vertices=np.asarray(witness.target_reference_vertices),
        target_reference_rational_tokens=witness.target_reference_rational_tokens,
        target_face_vertex_global_ids=face_vertices,
        source_chart_ids=witness.source_chart_ids,
        source_entity_ids=witness.source_entity_ids,
        occurrence_paths=witness.occurrence_paths,
        source_parameters=np.asarray(witness.source_parameters),
        source_parameter_rational_tokens=witness.source_parameter_rational_tokens,
        orientations=np.asarray(witness.orientations),
        implicit_cell_keys=np.asarray(witness.implicit_cell_keys),
        implicit_root_edge_keys=np.asarray(witness.implicit_root_edge_keys),
    )


def _retarget_source_boundary_chart_witness(
    source: CellMeshingResult,
    target: CellMesh,
    geometry: CellGeometrySpec,
    lineage: MeshLineage,
    witness: SourceBoundaryChartWitness,
    /,
    *,
    reference_maps: tuple[ReferenceCellMapWitnesses, ...],
    coarsening: NestedReferenceWitnesses | None = None,
    mixed_coarsening_maps: tuple[MixedCoarseningMap, ...] = (),
) -> SourceBoundaryChartWitness:
    if any(
        block.cell_kind == "polyhedron" for block in (*source.mesh.blocks, *target.blocks)
    ):
        return _retarget_polyhedral_chart_by_identity(
            source, target, geometry, lineage, witness
        )
    chart_dimension = witness.chart_dimension
    top_dimension = target.topological_dimension
    if (
        source.mesh.topology_id != lineage.source_topology_id
        or target.topology_id != lineage.target_topology_id
    ):
        raise ValueError("Boundary chart transition lineage endpoints are stale.")
    maps = _reference_map_rows(reference_maps)
    coarsened_by_target: dict[int, list[int]] = {}
    coarsening_reference: dict[int, tuple[int, tuple[tuple[Fraction, ...], ...]]] = {}
    mixed_reference_transforms: dict[
        int,
        tuple[
            tuple[tuple[Fraction, ...], ...],
            tuple[Fraction, ...],
        ],
    ] = {}
    mixed_affine_maps = _mixed_coarsening_reference_transforms(mixed_coarsening_maps)
    if coarsening is not None:
        validate_nested_reference_witnesses(coarsening, dimension=top_dimension)
        for fine_id, coarse_id, exact_vertices in zip(
            np.asarray(coarsening.fine_cell_ids, dtype=np.int64),
            np.asarray(coarsening.coarse_cell_ids, dtype=np.int64),
            exact_reference_vertices(coarsening),
            strict=True,
        ):
            fine, coarse = int(fine_id), int(coarse_id)
            if fine in coarsening_reference:
                raise ValueError("Coarsening ancestry repeats a fine source cell.")
            coarsening_reference[fine] = (coarse, exact_vertices)
            coarsened_by_target.setdefault(coarse, []).append(fine)
    for mixed_map in mixed_coarsening_maps:
        child_ids = np.asarray(mixed_map.child_cell_ids, dtype=np.int64)
        if child_ids.size != len(mixed_map.child_reference_vertices):
            raise ValueError("Mixed coarsening charts do not align with child IDs.")
        for child_id_raw, exact_vertices in zip(
            child_ids, mixed_map.child_reference_vertices, strict=True
        ):
            child_id = int(child_id_raw)
            if child_id in coarsening_reference:
                raise ValueError("Coarsening ancestry repeats a fine source cell.")
            coarsening_reference[child_id] = (
                mixed_map.parent_cell_id,
                exact_vertices,
            )
            coarsened_by_target.setdefault(mixed_map.parent_cell_id, []).append(child_id)
            mixed_reference_transforms[child_id] = mixed_affine_maps[
                (mixed_map.parent_cell_id, child_id)
            ]
    source_cells = {
        int(identifier): (block_index, row, block)
        for block_index, block in enumerate(source.mesh.blocks)
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    parent_by_target = {}
    relation = lineage.entity_lineage(top_dimension)
    for source_id, target_id, kind in zip(
        np.asarray(relation.source_global_ids, dtype=np.int64),
        np.asarray(relation.target_global_ids, dtype=np.int64),
        np.asarray(relation.relation_kinds, dtype=np.int32),
        strict=True,
    ):
        if int(kind) in (
            int(EntityLineageKind.PRESERVED),
            int(EntityLineageKind.REFINED_FROM),
            int(EntityLineageKind.RELOCATED),
        ):
            parent_by_target[int(target_id)] = int(source_id)
    if coarsening_reference:
        lineage_coarsening_pairs = {
            (int(source_id), int(target_id))
            for source_id, target_id, kind in zip(
                np.asarray(relation.source_global_ids, dtype=np.int64),
                np.asarray(relation.target_global_ids, dtype=np.int64),
                np.asarray(relation.relation_kinds, dtype=np.int32),
                strict=True,
            )
            if int(kind) == int(EntityLineageKind.COARSENED_INTO)
        }
        nested_coarsening_pairs = {
            (fine_id, coarse_id)
            for fine_id, (coarse_id, _) in coarsening_reference.items()
        }
        if not nested_coarsening_pairs <= lineage_coarsening_pairs:
            raise ValueError("Nested chart coarsening maps disagree with cell lineage.")
    target_to_source_entity = {}
    coarsened_source_to_target_entity = {}
    source_cells_by_entity: dict[tuple[int, int], list[int]] = {}
    source_cell_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    source_entity_ids = np.asarray(witness.entity_global_ids, dtype=np.int64)
    witnessed_source_entities = set(int(value) for value in source_entity_ids)
    witnessed_source_cells = set(int(value) for value in source_cell_ids)
    old_reference_exact = tuple(
        witness.target_reference_chart(row)
        for row in range(len(witness.source_chart_ids))
    )
    old_parameters_exact = tuple(
        witness.source_parameter_chart(row)
        for row in range(len(witness.source_chart_ids))
    )
    parameter_points = {point for row in old_parameters_exact for point in row}
    parameter_vertex_ids = {
        point: row for row, point in enumerate(sorted(parameter_points))
    }

    def _source_parameter_vertices(
        parameters: tuple[tuple[Fraction, ...], ...], /
    ) -> tuple[int, ...]:
        result = []
        for exact_parameter in parameters:
            identifier = parameter_vertex_ids.get(exact_parameter)
            if identifier is None:
                identifier = len(parameter_vertex_ids)
                parameter_vertex_ids[exact_parameter] = identifier
            result.append(identifier)
        return tuple(result)

    source_rows_by_cell: dict[int, list[int]] = {}
    for row, (cell_id, entity_id) in enumerate(
        zip(source_cell_ids, source_entity_ids, strict=True)
    ):
        source_cells_by_entity.setdefault((int(cell_id), int(entity_id)), []).append(row)
        source_rows_by_cell.setdefault(int(cell_id), []).append(row)
    coarse_cell_relation = np.asarray(relation.relation_kinds, dtype=np.int32)
    coarse_cell_source_ids = np.asarray(relation.source_global_ids, dtype=np.int64)
    nested_coarsened_cells = set(
        int(value)
        for value in coarse_cell_source_ids[
            coarse_cell_relation == int(EntityLineageKind.COARSENED_INTO)
        ]
    )
    missing_nested_cells = (witnessed_source_cells & nested_coarsened_cells) - set(
        coarsening_reference
    )
    if missing_nested_cells:
        raise ValueError(
            "Source-boundary chart coarsening lacks an exact nested child-to-parent map."
        )
    if chart_dimension < top_dimension:
        facet_lineage = lineage.entity_lineage(chart_dimension)
        source_facet_ids = np.asarray(facet_lineage.source_global_ids, dtype=np.int64)
        target_facet_ids = np.asarray(facet_lineage.target_global_ids, dtype=np.int64)
        facet_kinds = np.asarray(facet_lineage.relation_kinds, dtype=np.int32)
        for source_id, target_id, kind in zip(
            source_facet_ids, target_facet_ids, facet_kinds, strict=True
        ):
            source_entity, target_entity, relation_kind = (
                int(source_id),
                int(target_id),
                int(kind),
            )
            if relation_kind in (
                int(EntityLineageKind.PRESERVED),
                int(EntityLineageKind.REFINED_FROM),
                int(EntityLineageKind.RELOCATED),
            ):
                target_to_source_entity[target_entity] = source_entity
            if relation_kind in (
                int(EntityLineageKind.PRESERVED),
                int(EntityLineageKind.COARSENED_INTO),
                int(EntityLineageKind.RELOCATED),
            ):
                previous = coarsened_source_to_target_entity.get(source_entity)
                if previous is not None and previous != target_entity:
                    raise ValueError(
                        "One coarsened chart entity has multiple target ancestors."
                    )
                coarsened_source_to_target_entity[source_entity] = target_entity
        missing = witnessed_source_entities - (
            set(target_to_source_entity.values()) | set(coarsened_source_to_target_entity)
        )
        if missing:
            raise ValueError("A source chart entity has no exact target lineage.")
    elif not witnessed_source_cells <= (
        set(parent_by_target.values()) | set(coarsening_reference)
    ):
        raise ValueError("A source chart cell has no exact target lineage.")
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    entity_lookup: dict[tuple[int, ...], int] = {}
    if chart_dimension < top_dimension:
        keys = entity_keys(target, chart_dimension)
        identifiers = np.asarray(
            target.entity_set(chart_dimension).entity_ids, dtype=np.int64
        )
        entity_lookup = {
            tuple(int(value) for value in key): int(identifier)
            for key, identifier in zip(keys, identifiers, strict=True)
        }
    candidates: dict[
        tuple[int, str, str, tuple[str, ...]], dict[tuple[int, int], list[dict[str, Any]]]
    ] = {}
    for block in target.blocks:
        topology = reference_cell_topology(block.cell_kind)
        cell_vertices = np.asarray(block.vertices, dtype=np.int64)
        target_cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        for cell_row, target_cell_id_raw in enumerate(target_cell_ids):
            target_cell_id = int(target_cell_id_raw)
            coarsened_sources = coarsened_by_target.get(target_cell_id)
            if coarsened_sources is not None:
                for source_cell_id in coarsened_sources:
                    coarse_id, exact_cell = coarsening_reference[source_cell_id]
                    if coarse_id != target_cell_id:
                        raise ValueError(
                            "Coarsening chart map targets a different parent cell."
                        )
                    source_owner = source_cells.get(source_cell_id)
                    if source_owner is None:
                        raise ValueError(
                            "Coarsening chart map names an absent source cell."
                        )
                    source_block = source_owner[2]
                    if source_block.cell_kind != block.cell_kind:
                        raise ValueError(
                            "Coarsening chart ancestry changes its simplex family."
                        )
                    source_topology = reference_cell_topology(source_block.cell_kind)
                    source_reference_vertices = tuple(
                        tuple(Fraction(float(value)) for value in point)
                        for point in np.asarray(
                            source_topology.vertices, dtype=np.float64
                        )
                    )
                    for old_row in source_rows_by_cell.get(source_cell_id, ()):
                        source_entity_id = int(source_entity_ids[old_row])
                        target_entity_id = target_cell_id
                        local_entity = -1
                        if chart_dimension < top_dimension:
                            target_entity_id = coarsened_source_to_target_entity.get(
                                source_entity_id, -1
                            )
                            if target_entity_id < 0:
                                raise ValueError(
                                    "A coarsened source chart facet has no exact target entity."
                                )
                            matching_local_entities = []
                            for local, local_vertices in enumerate(
                                topology.entities[chart_dimension]
                            ):
                                global_vertices = sorted(
                                    int(
                                        target_vertex_ids[
                                            int(cell_vertices[cell_row, index])
                                        ]
                                    )
                                    for index in local_vertices
                                )
                                key = tuple(
                                    (
                                        *global_vertices,
                                        *([-1] * (keys.shape[1] - len(global_vertices))),
                                    )
                                )
                                if entity_lookup.get(key) == target_entity_id:
                                    matching_local_entities.append(local)
                            if len(matching_local_entities) != 1:
                                raise ValueError(
                                    "A coarsened source chart facet has no unique parent local entity."
                                )
                            local_entity = matching_local_entities[0]
                        reference_points = []
                        mixed_transform = mixed_reference_transforms.get(source_cell_id)
                        for exact_point in old_reference_exact[old_row]:
                            if mixed_transform is not None:
                                matrix, offset = mixed_transform
                                reference_points.append(
                                    tuple(
                                        offset[axis]
                                        + sum(
                                            (
                                                matrix[axis][dimension]
                                                * exact_point[dimension]
                                                for dimension in range(top_dimension)
                                            ),
                                            Fraction(0),
                                        )
                                        for axis in range(top_dimension)
                                    )
                                )
                                continue
                            weights = _simplex_chart_weights(
                                exact_point,
                                source_reference_vertices,
                                top_dimension,
                            )
                            if weights is None:
                                raise ValueError(
                                    "A source chart piece lies outside its exact coarsening cell map."
                                )
                            reference_points.append(
                                tuple(
                                    sum(
                                        (
                                            weights[corner] * exact_cell[corner][axis]
                                            for corner in range(len(weights))
                                        ),
                                        Fraction(0),
                                    )
                                    for axis in range(top_dimension)
                                )
                            )
                        target_reference = tuple(reference_points)
                        parameters_exact = old_parameters_exact[old_row]
                        parameters = np.asarray(
                            [
                                [float(value) for value in point]
                                for point in parameters_exact
                            ],
                            dtype=np.float64,
                        )
                        if _chart_orientation(parameters_exact, chart_dimension) == 0:
                            raise ValueError(
                                "A coarsened source chart piece is degenerate."
                            )
                        if chart_dimension == 1:
                            target_orientation = 1
                        elif top_dimension == 2:
                            target_orientation = _chart_orientation(target_reference, 2)
                        else:
                            from ..geometry._source_fidelity import _face_reference_chart

                            target_chart, _ = _face_reference_chart(
                                block.cell_kind,
                                local_entity,
                                target_reference,
                            )
                            target_orientation = _chart_orientation(
                                np.asarray(target_chart[0], dtype=np.float64), 2
                            )
                        if target_orientation == 0:
                            raise ValueError(
                                "A coarsened source chart piece has degenerate target orientation."
                            )
                        chart_key = (
                            target_entity_id,
                            witness.source_chart_ids[old_row],
                            witness.source_entity_ids[old_row],
                            witness.occurrence_paths[old_row],
                        )
                        parameter_vertices = _source_parameter_vertices(parameters_exact)
                        side = (target_cell_id, local_entity)
                        candidates.setdefault(chart_key, {}).setdefault(side, []).append(
                            {
                                "cell_id": target_cell_id,
                                "entity_id": target_entity_id,
                                "local_entity": local_entity,
                                "reference": target_reference,
                                "parameters": parameters,
                                "parameters_exact": parameters_exact,
                                "orientation": _chart_orientation(
                                    parameters_exact, chart_dimension
                                )
                                * target_orientation,
                                "vertices": parameter_vertices,
                                "source_row": old_row,
                                "parent_corners": tuple(
                                    tuple(
                                        Fraction(float(value))
                                        for value in topology.vertices[int(local)]
                                    )
                                    for local in (
                                        range(len(topology.vertices))
                                        if local_entity < 0
                                        else topology.entities[chart_dimension][
                                            local_entity
                                        ]
                                    )
                                ),
                                "cell_kind": block.cell_kind,
                            }
                        )
                continue
            map_record = maps.get(target_cell_id)
            if map_record is not None:
                map_witness, map_row, exact_cell = map_record
                source_cell_id = int(
                    np.asarray(map_witness.source_cell_ids, dtype=np.int64)[map_row]
                )
            else:
                source_cell_id = parent_by_target.get(target_cell_id)
                if source_cell_id is None:
                    if target_cell_id in source_cells:
                        raise ValueError(
                            "A changed source chart cell lacks exact reference ancestry."
                        )
                    continue
                if source_cell_id != target_cell_id:
                    raise ValueError(
                        "A boundary chart child has no exact cell reference map."
                    )
                exact_cell = tuple(
                    tuple(Fraction(float(value)) for value in point)
                    for point in np.asarray(topology.vertices, dtype=np.float64)
                )
            source_owner = source_cells.get(source_cell_id)
            if source_owner is None:
                raise ValueError("A boundary chart map names an absent source cell.")
            source_block = source.mesh.blocks[source_owner[0]]
            if (
                chart_dimension == top_dimension
                and source_cell_id not in witnessed_source_cells
            ):
                continue
            if chart_dimension == top_dimension:
                local_entities = ((-1, tuple(range(len(topology.vertices)))),)
            else:
                local_entities = tuple(
                    (local, tuple(int(value) for value in vertices))
                    for local, vertices in enumerate(topology.entities[chart_dimension])
                )
            for local_entity, local_vertices in local_entities:
                target_entity_id = target_cell_id
                source_entity_id = source_cell_id
                if chart_dimension < top_dimension:
                    global_vertices = sorted(
                        int(target_vertex_ids[int(cell_vertices[cell_row, local])])
                        for local in local_vertices
                    )
                    key = tuple(
                        (
                            *global_vertices,
                            *([-1] * (keys.shape[1] - len(global_vertices))),
                        )
                    )
                    target_entity_id = entity_lookup.get(key, -1)
                    source_entity_id = target_to_source_entity.get(target_entity_id, -1)
                if (
                    target_entity_id < 0
                    or source_entity_id < 0
                    or source_entity_id not in witnessed_source_entities
                ):
                    continue
                old_rows = source_cells_by_entity.get(
                    (source_cell_id, source_entity_id), ()
                )
                if not old_rows:
                    continue
                if chart_dimension == 1:
                    subfaces = (local_vertices,)
                elif chart_dimension == 2:
                    if len(local_vertices) == 3:
                        subfaces = (local_vertices,)
                    elif len(local_vertices) == 4:
                        subfaces = (
                            (local_vertices[0], local_vertices[1], local_vertices[2]),
                            (local_vertices[0], local_vertices[2], local_vertices[3]),
                        )
                    else:
                        raise ValueError(
                            "Source chart faces require triangular or quadrilateral cells."
                        )
                else:
                    raise ValueError(
                        "Only curve and surface source charts are supported."
                    )
                for subface in subfaces:
                    target_reference = tuple(
                        tuple(
                            Fraction(float(value))
                            for value in np.asarray(topology.vertices)[local]
                        )
                        for local in subface
                    )
                    source_reference = tuple(exact_cell[local] for local in subface)
                    old_rows = source_cells_by_entity.get(
                        (source_cell_id, source_entity_id), ()
                    )
                    found_rows = []
                    for old_row in old_rows:
                        old_points = old_reference_exact[old_row]
                        weights = tuple(
                            _simplex_chart_weights(point, old_points, chart_dimension)
                            for point in source_reference
                        )
                        valid_weights = tuple(
                            value for value in weights if value is not None
                        )
                        if len(valid_weights) != len(weights):
                            continue
                        parameters_exact = tuple(
                            tuple(
                                sum(
                                    (
                                        valid_weights[vertex][corner]
                                        * old_parameters_exact[old_row][corner][axis]
                                        for corner in range(chart_dimension + 1)
                                    ),
                                    Fraction(0),
                                )
                                for axis in range(chart_dimension)
                            )
                            for vertex in range(chart_dimension + 1)
                        )
                        parameters = np.asarray(
                            [
                                [float(value) for value in point]
                                for point in parameters_exact
                            ],
                            dtype=np.float64,
                        )
                        if _chart_orientation(parameters_exact, chart_dimension) == 0:
                            continue
                        if chart_dimension == 1:
                            target_orientation = 1
                        elif top_dimension == 2:
                            target_orientation = _chart_orientation(target_reference, 2)
                        else:
                            from ..geometry._source_fidelity import _face_reference_chart

                            target_chart, _ = _face_reference_chart(
                                block.cell_kind,
                                local_entity,
                                target_reference,
                            )
                            target_orientation = _chart_orientation(
                                np.asarray(target_chart[0], dtype=np.float64), 2
                            )
                        if target_orientation == 0:
                            continue
                        found_rows.append((old_row, parameters_exact, target_orientation))
                    if not found_rows:
                        raise ValueError(
                            "A target chart subface has no exact source-piece ancestor."
                        )
                    if len(found_rows) > 1:
                        first = found_rows[0]
                        if any(
                            witness.source_chart_ids[value[0]]
                            != witness.source_chart_ids[first[0]]
                            or witness.source_entity_ids[value[0]]
                            != witness.source_entity_ids[first[0]]
                            or witness.occurrence_paths[value[0]]
                            != witness.occurrence_paths[first[0]]
                            or value[1] != first[1]
                            for value in found_rows[1:]
                        ):
                            raise ValueError(
                                "A target chart subface crosses source chart ancestry."
                            )
                        found_rows = [first]
                    old_row, parameters_exact, target_orientation = found_rows[0]
                    parameters = np.asarray(
                        [[float(value) for value in point] for point in parameters_exact],
                        dtype=np.float64,
                    )
                    chart_key = (
                        target_entity_id,
                        witness.source_chart_ids[old_row],
                        witness.source_entity_ids[old_row],
                        witness.occurrence_paths[old_row],
                    )
                    side = (target_cell_id, local_entity)
                    parameter_vertices = _source_parameter_vertices(parameters_exact)
                    candidates.setdefault(chart_key, {}).setdefault(side, []).append(
                        {
                            "cell_id": target_cell_id,
                            "entity_id": target_entity_id,
                            "local_entity": local_entity,
                            "reference": target_reference,
                            "parameters": parameters,
                            "parameters_exact": parameters_exact,
                            "orientation": _chart_orientation(
                                parameters_exact, chart_dimension
                            )
                            * target_orientation,
                            "vertices": parameter_vertices,
                            "source_row": old_row,
                        }
                    )

    # An implicit root chart is the target face itself lifted into the zero set
    # of its root cell: every row keeps its ancestor's root cell and edges with
    # the standard root parameters, oriented like its target reference face.
    implicit = np.asarray(witness.implicit_cell_keys).size > 0
    standard_chart = tuple(
        tuple(Fraction(value) for value in point) for point in ((0, 0), (1, 0), (0, 1))
    )

    def merged_pieces(rows: list[dict[str, Any]], /) -> list[dict[str, Any]]:
        """Merge the child pieces of one restored parent entity into its chart."""
        if len(rows) < 2 or any("parent_corners" not in value for value in rows):
            return rows
        corners = rows[0]["parent_corners"]
        local_entity = rows[0]["local_entity"]
        if chart_dimension == 1:
            target_orientation = 1
        elif top_dimension == 2:
            target_orientation = _chart_orientation(corners, 2)
        else:
            from ..geometry._source_fidelity import _face_reference_chart

            target_chart, _ = _face_reference_chart(
                rows[0]["cell_kind"], local_entity, corners
            )
            target_orientation = _chart_orientation(
                np.asarray(target_chart[0], dtype=np.float64), 2
            )
        if implicit:
            covered = {tuple(point) for value in rows for point in value["reference"]}
            if any(corner not in covered for corner in corners):
                raise ValueError(
                    "A restored chart entity corner has no source chart piece."
                )
            corner_parameters = standard_chart
            orientation = target_orientation
        else:
            point_parameters: dict[tuple[Fraction, ...], tuple[Fraction, ...]] = {}
            for value in rows:
                for point, parameter in zip(
                    value["reference"], value["parameters_exact"], strict=True
                ):
                    previous = point_parameters.setdefault(tuple(point), tuple(parameter))
                    if previous != tuple(parameter):
                        raise ValueError(
                            "Coarsened source chart pieces disagree at a shared point."
                        )
            if any(corner not in point_parameters for corner in corners):
                raise ValueError(
                    "A restored chart entity corner has no source chart piece."
                )
            corner_parameters = tuple(point_parameters[corner] for corner in corners)
            for point, parameter in point_parameters.items():
                weights = _simplex_chart_weights(point, corners, chart_dimension)
                if weights is None or parameter != tuple(
                    sum(
                        (
                            weights[corner] * corner_parameters[corner][axis]
                            for corner in range(chart_dimension + 1)
                        ),
                        Fraction(0),
                    )
                    for axis in range(chart_dimension)
                ):
                    raise ValueError(
                        "Coarsened source chart pieces are not one exact affine parent chart."
                    )
            orientation = (
                _chart_orientation(corner_parameters, chart_dimension)
                * target_orientation
            )
            if orientation == 0 or any(
                value["orientation"] != orientation for value in rows
            ):
                raise ValueError("Coarsened source chart pieces disagree on orientation.")
        merged = dict(rows[0])
        merged.update(
            reference=corners,
            parameters_exact=corner_parameters,
            parameters=np.asarray(
                [[float(value) for value in point] for point in corner_parameters],
                dtype=np.float64,
            ),
            orientation=orientation,
            vertices=_source_parameter_vertices(corner_parameters),
        )
        return [merged]

    rows_to_publish: list[dict[str, Any]] = []
    for chart_key, sides in candidates.items():
        side_values = []
        for side, side_rows in sides.items():
            rows = merged_pieces(side_rows)
            vertex_parameters: dict[int, tuple[Fraction, ...]] = {}
            for value in rows:
                for vertex_id, parameter in zip(
                    value["vertices"], value["parameters_exact"], strict=True
                ):
                    previous = vertex_parameters.get(int(vertex_id))
                    if previous is not None and previous != parameter:
                        raise ValueError(
                            "One target source edge has inconsistent chart values at a shared vertex."
                        )
                    vertex_parameters[int(vertex_id)] = parameter
            side_values.append((side, rows, vertex_parameters))
        if len(side_values) > 1:
            reference_values = side_values[0][2]
            if any(
                set(values) != set(reference_values)
                or any(values[key] != reference_values[key] for key in values)
                for _, _, values in side_values[1:]
            ):
                raise ValueError(
                    "Incident target cells disagree on source chart ancestry."
                )
        selected = min(side_values, key=lambda value: value[0])
        rows_to_publish.extend(selected[1])
    by_entity: dict[int, set[tuple[str, str, tuple[str, ...]]]] = {}
    for key in candidates:
        by_entity.setdefault(key[0], set()).add((key[1], key[2], key[3]))
    if any(len(values) > 1 for values in by_entity.values()):
        raise ValueError(
            "One target chart entity has ambiguous source-chart occurrence ancestry."
        )
    if not rows_to_publish:
        raise ValueError("Transition removed every witnessed source chart stratum.")
    rows_to_publish.sort(
        key=lambda value: (
            value["entity_id"],
            value["cell_id"],
            value["local_entity"],
            value["vertices"],
        )
    )
    if implicit:
        rows_to_publish = [
            {
                **value,
                "parameters_exact": standard_chart,
                "parameters": np.asarray(standard_chart, dtype=np.float64),
                # The stored sign is chart orientation times target orientation.
                "orientation": value["orientation"]
                * _chart_orientation(value["parameters_exact"], chart_dimension),
            }
            for value in rows_to_publish
        ]
    source_rows = [int(value["source_row"]) for value in rows_to_publish]
    implicit_cells = np.asarray(witness.implicit_cell_keys, dtype=np.int64)
    implicit_edges = np.asarray(witness.implicit_root_edge_keys, dtype=np.int64)
    implicit_cells_out = None if implicit_cells.size == 0 else implicit_cells[source_rows]
    implicit_edges_out = None if implicit_edges.size == 0 else implicit_edges[source_rows]
    return SourceBoundaryChartWitness(
        mesh_id=target.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=target.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        chart_dimension=chart_dimension,
        reference_dimension=target.topological_dimension,
        cell_global_ids=np.asarray(
            [value["cell_id"] for value in rows_to_publish], dtype=np.int64
        ),
        entity_global_ids=np.asarray(
            [value["entity_id"] for value in rows_to_publish], dtype=np.int64
        ),
        local_entity_indices=np.asarray(
            [value["local_entity"] for value in rows_to_publish], dtype=np.int32
        ),
        target_reference_vertices=np.asarray(
            [
                [[float(value) for value in point] for point in row["reference"]]
                for row in rows_to_publish
            ],
            dtype=np.float64,
        ),
        target_reference_rational_tokens=rational_chart_tokens(
            tuple(value["reference"] for value in rows_to_publish)
        ),
        source_chart_ids=tuple(witness.source_chart_ids[row] for row in source_rows),
        source_entity_ids=tuple(witness.source_entity_ids[row] for row in source_rows),
        occurrence_paths=tuple(witness.occurrence_paths[row] for row in source_rows),
        source_parameters=np.asarray(
            [value["parameters"] for value in rows_to_publish], dtype=np.float64
        ),
        source_parameter_rational_tokens=rational_chart_tokens(
            tuple(value["parameters_exact"] for value in rows_to_publish)
        ),
        orientations=np.asarray(
            [value["orientation"] for value in rows_to_publish], dtype=np.int8
        ),
        implicit_cell_keys=implicit_cells_out,
        implicit_root_edge_keys=implicit_edges_out,
    )


def _retarget_fidelity_witness(
    source: CellMeshingResult,
    target: CellMesh,
    geometry: CellGeometrySpec,
    lineage: MeshLineage,
    witness: SourceFidelityWitness,
    /,
    *,
    reference_maps: tuple[ReferenceCellMapWitnesses, ...] = (),
    coarsening: NestedReferenceWitnesses | None = None,
    mixed_coarsening_maps: tuple[MixedCoarseningMap, ...] = (),
) -> SourceFidelityWitness:
    if isinstance(witness, SurfaceChartWitness):
        if mixed_coarsening_maps:
            raise ValueError(
                "Triangle surface charts have no mixed volume coarsening maps."
            )
        return _retarget_surface_chart_witness(
            source,
            target,
            geometry,
            lineage,
            witness,
            reference_maps=reference_maps,
            coarsening=coarsening,
        )
    geometry_id = cell_geometry_id(geometry)
    if isinstance(witness, SourceBoundaryPointWitness):
        target_ids = set(int(value) for value in np.asarray(target.vertex_global_ids))
        if (
            not set(int(value) for value in np.asarray(witness.vertex_global_ids))
            <= target_ids
        ):
            raise ValueError(
                "A source-bound point correspondence lost its target vertex."
            )
        return SourceBoundaryPointWitness(
            mesh_id=target.mesh_id,
            geometry_id=geometry_id,
            topology_id=target.topology_id,
            source_id=witness.source_id,
            source_revision=witness.source_revision,
            source_query_id=witness.source_query_id,
            vertex_global_ids=np.asarray(witness.vertex_global_ids),
            source_entity_ids=witness.source_entity_ids,
            occurrence_paths=witness.occurrence_paths,
        )
    if isinstance(witness, SourceBoundaryChartWitness):
        return _retarget_source_boundary_chart_witness(
            source,
            target,
            geometry,
            lineage,
            witness,
            reference_maps=reference_maps,
            coarsening=coarsening,
            mixed_coarsening_maps=mixed_coarsening_maps,
        )
    if isinstance(witness, SourceBoundaryDistanceWitness):
        same_coordinates = (
            np.array_equal(
                np.asarray(source.mesh.coordinates), np.asarray(target.coordinates)
            )
            and cell_geometry_id(source.geometry) == geometry_id
        )
        if source.mesh.topology_id != target.topology_id or not (
            same_coordinates or _same_volume_boundary_geometry(source.mesh, target)
        ):
            raise ValueError(
                "Distance-only source evidence requires unchanged target boundary geometry."
            )
        return SourceBoundaryDistanceWitness(
            mesh_id=target.mesh_id,
            geometry_id=geometry_id,
            topology_id=target.topology_id,
            source_id=witness.source_id,
            source_revision=witness.source_revision,
            source_query_id=witness.source_query_id,
            sample_count=witness.sample_count,
            cover_radius=witness.cover_radius,
            target_facet_global_ids=np.asarray(witness.target_facet_global_ids),
        )
    if isinstance(witness, SurfaceEnvelopeDistanceWitness):
        if (
            source.mesh.topology_id != target.topology_id
            or cell_geometry_id(source.geometry) != geometry_id
            or not np.array_equal(
                np.asarray(source.mesh.coordinates), np.asarray(target.coordinates)
            )
        ):
            raise ValueError(
                "Surface-envelope distance evidence requires unchanged target coordinates."
            )
        return SurfaceEnvelopeDistanceWitness(
            mesh_id=target.mesh_id,
            geometry_id=geometry_id,
            topology_id=target.topology_id,
            source_id=witness.source_id,
            source_revision=witness.source_revision,
            source_query_id=witness.source_query_id,
            offset=witness.offset,
            spacing=witness.spacing,
            certificate_spacing=witness.certificate_spacing,
            bounds=np.asarray(witness.bounds),
            envelope_vertices=np.asarray(witness.envelope_vertices),
            envelope_triangles=np.asarray(witness.envelope_triangles),
            target_vertex_global_ids=np.asarray(witness.target_vertex_global_ids),
            target_facet_global_ids=np.asarray(witness.target_facet_global_ids),
            reproduction=witness.reproduction,
        )
    raise TypeError("No transition ancestry rule accepts this source fidelity witness.")


def _retarget_coverage_witness(
    original: MeshCertificationInputs,
    target_regions: np.ndarray | None,
    target_distance_witness: SourceFidelityWitness | None,
    transition: CellGeometryTransition,
    /,
) -> RestrictedDelaunayCoverageTransportWitness:
    if original.cell_regions is None or target_regions is None:
        raise ValueError(
            "Restricted coverage transport requires source and target region rows."
        )
    if not isinstance(target_distance_witness, SourceBoundaryDistanceWitness):
        raise ValueError(
            "Restricted coverage transport requires a renewed source-distance witness."
        )
    source_witness = original.source_coverage_witness
    if isinstance(source_witness, RestrictedDelaunayBoundaryWitness):
        if not isinstance(original.fidelity_witness, SourceBoundaryDistanceWitness):
            raise ValueError(
                "Direct restricted coverage lacks its source-distance witness."
            )
        root_witness = source_witness
        root_distance = original.fidelity_witness
        transitions = (transition,)
        stages = (np.asarray(original.cell_regions), np.asarray(target_regions))
    elif isinstance(source_witness, RestrictedDelaunayCoverageTransportWitness):
        if source_witness.target_distance_witness_id != getattr(
            original.fidelity_witness, "witness_id", None
        ):
            raise ValueError(
                "Prior restricted coverage and current fidelity witness disagree."
            )
        if not np.array_equal(
            source_witness.stage_cell_regions[-1], np.asarray(original.cell_regions)
        ):
            raise ValueError("Prior restricted coverage region endpoint is stale.")
        root_witness = source_witness.root_witness
        root_distance = source_witness.root_distance_witness
        transitions = (*source_witness.transitions, transition)
        stages = (*source_witness.stage_cell_regions, np.asarray(target_regions))
    else:
        raise TypeError(
            "No source coverage transport accepts this retained witness type."
        )
    return RestrictedDelaunayCoverageTransportWitness(
        root_witness=root_witness,
        root_distance_witness=root_distance,
        transitions=transitions,
        stage_cell_regions=stages,
        target_distance_witness_id=target_distance_witness.witness_id,
    )


def _renew_transition_certification_inputs(
    source: CellMeshingResult,
    target: CellMesh,
    geometry: CellGeometrySpec,
    lineage: MeshLineage,
    /,
    *,
    reference_maps: tuple[ReferenceCellMapWitnesses, ...] = (),
    transition: CellGeometryTransition | None = None,
    coarsening: NestedReferenceWitnesses | None = None,
    mixed_coarsening_maps: tuple[MixedCoarseningMap, ...] = (),
    coverage_renewal: PolyhedralCoverageRenewal | None = None,
    fidelity_witness_override: SourceFidelityWitness | None = None,
) -> MeshCertificationInputs | None:
    """Rebuild target-bound acceptance inputs without reducing retained claims."""
    report = source.certification
    if report is None:
        if coverage_renewal is not None:
            raise ValueError("Coverage renewal requires retained source certification.")
        return None
    report.require_passed()
    report.revalidate(source.mesh, source.geometry)
    original = report.inputs
    regions = _transition_regions(original, target, lineage)
    if coverage_renewal is not None and fidelity_witness_override is not None:
        raise ValueError(
            "Polyhedral coverage and surface fidelity renewals are mutually exclusive."
        )
    if coverage_renewal is not None:
        if not isinstance(coverage_renewal, PolyhedralCoverageRenewal):
            raise TypeError("coverage_renewal must be PolyhedralCoverageRenewal or None.")
        if (
            original.source_coverage_witness is None
            or not isinstance(original.fidelity_witness, SourceBoundaryDistanceWitness)
            or coverage_renewal.source_witness_id
            != original.source_coverage_witness.witness_id
            or coverage_renewal.source_distance_witness_id
            != original.fidelity_witness.witness_id
        ):
            raise ValueError(
                "Polyhedral coverage renewal does not bind the retained source witnesses."
            )
        fidelity: SourceFidelityWitness | None = coverage_renewal.target_distance_witness
    elif fidelity_witness_override is not None:
        if original.fidelity_witness is None:
            raise ValueError("A route cannot add an undeclared source-fidelity claim.")
        fidelity = fidelity_witness_override
    else:
        fidelity = (
            None
            if original.fidelity_witness is None
            else _retarget_fidelity_witness(
                source,
                target,
                geometry,
                lineage,
                original.fidelity_witness,
                reference_maps=reference_maps,
                coarsening=coarsening,
                mixed_coarsening_maps=mixed_coarsening_maps,
            )
        )
    coverage = None
    if original.source_coverage_witness is not None:
        if coverage_renewal is not None:
            coverage = coverage_renewal.target_witness
        else:
            if transition is None:
                raise ValueError(
                    "Source coverage renewal requires a retained exact transition record."
                )
            coverage = _retarget_coverage_witness(original, regions, fidelity, transition)
    elif coverage_renewal is not None:
        raise ValueError(
            "Polyhedral coverage renewal cannot add an undeclared source claim."
        )
    scoped = tuple(
        (
            query,
            _retarget_fidelity_witness(
                source,
                target,
                geometry,
                lineage,
                witness,
                reference_maps=reference_maps,
                coarsening=coarsening,
            ),
            tolerance,
        )
        for query, witness, tolerance in original.scoped_fidelity
    )
    return MeshCertificationInputs(
        target,
        geometry,
        original.schedule,
        source_id=original.source_id,
        source_revision=original.source_revision,
        domain=original.domain,
        cell_regions=regions,
        source=original.source,
        fidelity_tolerance=original.fidelity_tolerance,
        maximum_normal_deviation=original.maximum_normal_deviation,
        junction_vertices=_transition_junctions(original, target, lineage),
        protected_features=original.protected_features,
        scoped_fidelity=scoped,
        fidelity_witness=fidelity,
        feature_source=original.feature_source,
        source_coverage_witness=coverage,
    )


def _compose_material_reference_maps(
    source: CellMeshingResult,
    target: CellMeshingResult,
    source_reference_maps: dict[str, ReferenceCellMapWitnesses],
    target_to_source_maps: tuple[ReferenceCellMapWitnesses, ...],
    /,
    *,
    coarsening: NestedReferenceWitnesses | None = None,
) -> dict[str, ReferenceCellMapWitnesses]:
    source_blocks = {block.name: block for block in source.mesh.blocks}
    if set(source_reference_maps) != set(source_blocks):
        raise ValueError(
            "Material reference charts do not cover every current source block."
        )
    source_cell_ids: list[int] = []
    reference_cell_ids: list[int] = []
    reference_corners: list[tuple[tuple[Fraction, ...], ...]] = []
    reference_chart_rows: list[tuple[Any, ...]] = []
    source_kind: str | None = None
    reference_kind: str | None = None
    for block in source.mesh.blocks:
        witness = source_reference_maps[block.name]
        if witness.target_cell_kind != block.cell_kind or not np.array_equal(
            np.asarray(witness.target_cell_ids, dtype=np.int64),
            np.arange(block.cell_count),
        ):
            raise ValueError(
                "Material reference charts must use block-local current-source rows."
            )
        if source_kind is None:
            source_kind = witness.target_cell_kind
            reference_kind = witness.source_cell_kind
        if (source_kind, reference_kind) != (
            witness.target_cell_kind,
            witness.source_cell_kind,
        ):
            raise ValueError(
                "Material reference continuation requires one tetrahedral cell family."
            )
        source_cell_ids.extend(
            int(value) for value in np.asarray(block.global_ids, dtype=np.int64)
        )
        reference_cell_ids.extend(
            int(value) for value in np.asarray(witness.source_cell_ids, dtype=np.int64)
        )
        for mapping in nested_cell_reference_maps(witness):
            if mapping.arguments is None:
                raise ValueError(
                    "Material source reference maps lack exact chart expressions."
                )
            reference_chart_rows.append(mapping.arguments)
        reference_corners.extend(exact_reference_cell_vertices(witness))
    if source_kind is None or reference_kind is None:
        raise ValueError("A material transition requires nonempty volume cell endpoints.")
    current_to_reference = ReferenceCellMapWitnesses(
        np.arange(len(source_cell_ids), dtype=np.int64),
        np.asarray(reference_cell_ids, dtype=np.int64),
        source_kind,
        reference_kind,
        np.asarray(
            [
                [[float(value) for value in point] for point in cell]
                for cell in reference_corners
            ],
            dtype=np.float64,
        ),
        tuple(reference_corners),
        freeze_reference_chart_arguments(tuple(reference_chart_rows)),
    )
    target_blocks_by_name = {block.name: block for block in target.mesh.blocks}
    target_cell_rows = {
        int(identifier): (block.name, row)
        for block in target.mesh.blocks
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    rows_by_block: dict[
        str,
        dict[
            int,
            tuple[
                int,
                tuple[tuple[Fraction, ...], ...],
                tuple[Any, ...],
            ],
        ],
    ] = {block.name: {} for block in target.mesh.blocks}
    for target_to_source in target_to_source_maps:
        composed = compose_reference_cell_maps(
            target_to_source,
            current_to_reference,
            intermediate_cell_global_ids=source_cell_ids,
        )
        exact = exact_reference_cell_vertices(composed)
        arguments = composed.source_reference_chart_arguments
        if arguments is None:
            raise ValueError(
                "Composed material charts lost their exact chart expressions."
            )
        reference_ids = np.asarray(composed.source_cell_ids, dtype=np.int64)
        for row, target_id in enumerate(
            np.asarray(composed.target_cell_ids, dtype=np.int64)
        ):
            location = target_cell_rows.get(int(target_id))
            if location is None:
                raise ValueError("Material reference chart names an absent target cell.")
            block_name, block_row = location
            if composed.target_cell_kind != target_blocks_by_name[block_name].cell_kind:
                raise ValueError(
                    "Material reference chart names a different target cell family."
                )
            value = (int(reference_ids[row]), exact[row], arguments[row])
            previous = rows_by_block[block_name].get(block_row)
            if previous is not None and previous != value:
                raise ValueError("Target cell has conflicting material-reference charts.")
            rows_by_block[block_name][block_row] = value
    if coarsening is not None:
        # Restored parents have no chart into one source cell; their exact
        # reference chart is rebuilt from the charts of all their children.
        for block_name, witness in coarsen_reference_cell_maps(
            coarsening, source_reference_maps, source.mesh, target.mesh
        ).items():
            arguments = witness.source_reference_chart_arguments
            if arguments is None:
                raise ValueError(
                    "Coarsened material charts lost their exact chart expressions."
                )
            exact = exact_reference_cell_vertices(witness)
            reference_ids = np.asarray(witness.source_cell_ids, dtype=np.int64)
            for row, block_row in enumerate(
                np.asarray(witness.target_cell_ids, dtype=np.int64).tolist()
            ):
                value = (int(reference_ids[row]), exact[row], arguments[row])
                previous = rows_by_block[block_name].get(int(block_row))
                if previous is not None and previous != value:
                    raise ValueError(
                        "Target cell has conflicting material-reference charts."
                    )
                rows_by_block[block_name][int(block_row)] = value
    result: dict[str, ReferenceCellMapWitnesses] = {}
    for block in target.mesh.blocks:
        rows = rows_by_block[block.name]
        if set(rows) != set(range(block.cell_count)):
            raise ValueError(
                "Exact material-reference charts do not cover every target cell."
            )
        exact = tuple(rows[row][1] for row in range(block.cell_count))
        chart_arguments = freeze_reference_chart_arguments(
            tuple(rows[row][2] for row in range(block.cell_count))
        )
        result[block.name] = ReferenceCellMapWitnesses(
            np.arange(block.cell_count, dtype=np.int64),
            np.asarray([rows[row][0] for row in range(block.cell_count)], dtype=np.int64),
            block.cell_kind,
            reference_kind,
            np.asarray(
                [[[float(value) for value in point] for point in cell] for cell in exact],
                dtype=np.float64,
            ),
            exact,
            chart_arguments,
        )
    return result


def _extend_prior_volume_material_chain(
    prior_transition: CellMeshTransition | None,
    source: CellMeshingResult,
    target: CellMeshingResult,
    lineage: MeshLineage,
    /,
    *,
    material_reference: MaterialReferenceDeclaration | None = None,
    reference_maps: tuple[ReferenceCellMapWitnesses, ...] = (),
    coarsening: NestedReferenceWitnesses | None = None,
    coordinate_transition: CellGeometryTransition | None = None,
) -> VolumeMaterialChain | None:
    if prior_transition is None and material_reference is None:
        return None
    chain = None if prior_transition is None else prior_transition.volume_material_chain
    prior_reference = (
        None if prior_transition is None else prior_transition.material_reference
    )
    if (
        material_reference is not None
        and prior_transition is not None
        and (chain is not None or prior_reference is not None)
    ):
        raise ValueError(
            "An explicit material reference seed is mutually exclusive with a prior material chain."
        )
    reference = material_reference if material_reference is not None else prior_reference
    if chain is None and reference is None:
        return None
    if source.certification is None or source.certification.coverage is None:
        raise ValueError("Material-reference renewal requires certified source coverage.")
    if prior_transition is not None and (
        prior_transition.target.result_id != source.result_id
        or prior_transition.target.mesh.mesh_id != source.mesh.mesh_id
        or cell_geometry_id(prior_transition.target.geometry)
        != cell_geometry_id(source.geometry)
    ):
        raise ValueError(
            "Prior material transition does not bind the current source result."
        )
    if chain is not None:
        if (
            reference is None
            or reference.declaration_id != chain.reference.declaration_id
        ):
            raise ValueError(
                "Prior material chain lacks its explicit material reference declaration."
            )
        previous = chain.maps[-1]
        if (
            previous.target_mesh.mesh_id != source.mesh.mesh_id
            or cell_geometry_id(previous.target_geometry)
            != cell_geometry_id(source.geometry)
            or chain.target_mesh_id != source.mesh.mesh_id
            or chain.target_geometry_id != cell_geometry_id(source.geometry)
        ):
            raise ValueError(
                "Retained material chain does not bind the current source geometry."
            )
        source_reference_maps = dict(previous.deformation.target_reference_maps)
        source_entities = previous.target_cell_entity_ids
        source_regions = previous.target_region_ids
        source_occurrences = previous.target_occurrence_paths
        map_policy = previous.policy
        integration_policy = previous.integration_policy
        if (
            previous.target_coverage.certificate_id
            != source.certification.coverage.certificate_id
        ):
            raise ValueError(
                "Retained material coverage is stale for the current source."
            )
    else:
        if reference is None:
            raise ValueError("Material reference declaration is absent.")
        if reference.root_mesh.mesh_id != source.mesh.mesh_id or cell_geometry_id(
            reference.root_geometry
        ) != cell_geometry_id(source.geometry):
            raise ValueError(
                "A seed material reference must be the exact accepted current source configuration."
            )
        source_reference_maps = dict(reference.identity_reference_maps)
        source_entities = reference.cell_entity_ids
        source_regions = reference.region_ids
        source_occurrences = reference.occurrence_paths
        map_policy = None
        integration_policy = None
        if (
            reference.root_coverage.certificate_id
            != source.certification.coverage.certificate_id
        ):
            raise ValueError(
                "Material reference coverage is stale for the current source."
            )
    reference = MaterialReferenceDeclaration.revalidate(
        reference,
        reference.root_mesh,
        reference.root_geometry,
        reference.source_map,
        reference.root_coverage,
    )
    if (
        source.certification is None
        or target.certification is None
        or source.certification.coverage is None
        or target.certification.coverage is None
    ):
        raise ValueError(
            "Material-reference renewal requires certified source and target coverage."
        )
    source.certification.require_passed()
    target.certification.require_passed()
    source_coverage = source.certification.coverage
    target_coverage = target.certification.coverage
    source_inputs = source.certification.inputs
    if (
        reference.source_map.source_id,
        reference.source_map.source_revision,
    ) != (source_inputs.source_id, source_inputs.source_revision):
        raise ValueError(
            "Material reference source revision disagrees with source certification."
        )
    source_cell_ids = np.asarray(source_inputs.cell_global_ids, dtype=np.int64)
    mesh_source_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in source.mesh.blocks
        )
    )
    if not np.array_equal(source_cell_ids, mesh_source_ids):
        raise ValueError("Source material rows do not match canonical source cell order.")
    if not (
        len(source_entities) == source_cell_ids.size
        and len(source_regions) == source_cell_ids.size
        and len(source_occurrences) == source_cell_ids.size
    ):
        raise ValueError(
            "Material-reference identity rows do not cover the source endpoint."
        )
    if tuple(
        source_coverage.region_ids[int(value)]
        for value in np.asarray(source_coverage.cell_regions, dtype=np.int64)
    ) != tuple(source_regions):
        raise ValueError(
            "Material-reference regions differ from certified source assignments."
        )
    target_region_ids = tuple(
        target_coverage.region_ids[int(value)]
        for value in np.asarray(target_coverage.cell_regions, dtype=np.int64)
    )
    if target_region_ids != tuple(
        target_coverage.region_ids[int(value)]
        for value in np.asarray(target.certification.inputs.cell_regions, dtype=np.int64)
    ):
        raise ValueError(
            "Target material regions disagree with certified target assignments."
        )
    material_rows = propagate_material_association_rows(
        source.mesh,
        target.mesh,
        lineage,
        source_entities,
        source_regions,
        source_occurrences,
        target_region_ids,
    )
    target_cell_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in target.mesh.blocks
        )
    )
    if material_rows.target_cell_global_ids != tuple(target_cell_ids.tolist()):
        raise ValueError("Target material rows differ from canonical target cell order.")
    target_reference_maps = _compose_material_reference_maps(
        source, target, source_reference_maps, reference_maps, coarsening=coarsening
    )
    deformation = VolumeMaterialDeformation(
        reference,
        source.mesh,
        source.geometry,
        target.mesh,
        target.geometry,
        source_reference_maps=source_reference_maps,
        target_reference_maps=target_reference_maps,
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        source_frame_id=source.coordinate_contract.spatial_id,
        target_frame_id=target.coordinate_contract.spatial_id,
        coordinate_transition=coordinate_transition,
    )
    material_map = VolumeMaterialMap(
        source.mesh,
        source.geometry,
        target.mesh,
        target.geometry,
        deformation,
        source_entities,
        material_rows.cell_entity_ids,
        source_regions,
        material_rows.region_ids,
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        policy=map_policy,
        integration_policy=integration_policy,
        source_occurrence_paths=source_occurrences,
        target_occurrence_paths=material_rows.occurrence_paths,
    )
    return VolumeMaterialChain(
        (material_map,) if chain is None else (*chain.maps, material_map)
    )


class _NativeTarget(NamedTuple):
    target: CellMeshingResult
    transition: CellMeshTransition
    lineage: MeshLineage
    stencil: VertexInterpolationStencil
    transfer: FiniteElementTopologyTransfer | None
    surface_chart_witness: SurfaceChartWitness | SourceBoundaryChartWitness | None = None


def _mixed_geometry_for_target(
    source: CellMeshingResult,
    target: CellMesh,
    lineage: MeshLineage,
    outcome: MixedAdaptationOutcome,
    /,
) -> tuple[CellGeometrySpec, tuple[ReferenceCellMapWitnesses, ...]]:
    geometry = outcome.target_geometry
    if geometry is None:
        raise ValueError("Mixed adaptation did not retain target coordinate geometry.")
    direct_maps = _reference_map_rows(outcome.reference_maps)
    source_ids = set(int(value) for value in _cell_ids(source.mesh))
    target_ids = set(int(value) for value in _cell_ids(target))
    mapped_source_ids = {
        int(np.asarray(witness.source_cell_ids, dtype=np.int64)[row])
        for witness, row, _ in direct_maps.values()
    }
    if not mapped_source_ids <= source_ids:
        raise ValueError("Mixed reference maps name absent source cells.")
    coarsened_targets: set[int] = set()
    coarsening_pairs: set[tuple[int, int]] = set()
    for record in outcome.coarsening_maps:
        parent_id = record.parent_cell_id
        child_ids = np.asarray(record.child_cell_ids, dtype=np.int64)
        if (
            parent_id in coarsened_targets
            or parent_id not in target_ids
            or child_ids.size != len(record.child_reference_vertices)
            or not set(int(value) for value in child_ids) <= source_ids
        ):
            raise ValueError("Mixed coarsening maps do not bind source and target cells.")
        coarsened_targets.add(parent_id)
        coarsening_pairs.update((int(value), parent_id) for value in child_ids)
    if (
        direct_maps.keys() & coarsened_targets
        or (set(direct_maps) | coarsened_targets) != target_ids
    ):
        raise ValueError("Mixed reference and coarsening maps do not cover target cells.")
    cell_lineage = lineage.entity_lineage(target.topological_dimension)
    relation_sources = np.asarray(cell_lineage.source_global_ids, dtype=np.int64)
    relation_targets = np.asarray(cell_lineage.target_global_ids, dtype=np.int64)
    relation_kinds = np.asarray(cell_lineage.relation_kinds, dtype=np.int32)
    actual_coarsening_pairs = {
        (int(source_id), int(target_id))
        for source_id, target_id, relation_kind in zip(
            relation_sources, relation_targets, relation_kinds, strict=True
        )
        if int(relation_kind) == int(EntityLineageKind.COARSENED_INTO)
    }
    if actual_coarsening_pairs != coarsening_pairs:
        raise ValueError("Mixed coarsening maps disagree with finalized cell lineage.")
    elements = dict(zip(geometry.block_names, geometry.elements, strict=True))
    routes = dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True))
    if geometry.source_map is None:
        rebound = CellGeometrySpec(
            elements,
            routes,
            geometry.coordinates,
            restriction_source=geometry.restriction_source,
        )
    else:
        rebound = CellGeometrySpec(
            elements,
            routes,
            geometry.coordinates,
            source_map=geometry.source_map,
            source_mesh=target,
            restriction_source=geometry.restriction_source,
        )
    rebound.resolve(target)
    return rebound, outcome.reference_maps


def _surface_metric_geometry_transition(
    source: CellMeshingResult,
    target: CellMesh,
    outcome: SurfaceMetricOutcome,
    /,
) -> CellGeometryTransition:
    reference_maps = outcome.geometry.reference_maps
    if not reference_maps:
        raise ValueError("Surface metric adaptation lacks exact source-reference maps.")
    fidelity_tolerance = (
        None
        if source.certification is None
        else source.certification.inputs.fidelity_tolerance
    )
    reconstruction_bound = 0.0 if fidelity_tolerance is None else fidelity_tolerance
    source_elements, _, _ = source.geometry.resolve(source.mesh)
    coordinate_degree = 1
    for element in source_elements:
        if not isinstance(element, ScalarCoordinateElement):
            raise ValueError("Surface reconstruction requires scalar H1 source maps.")
        coordinate_degree = max(coordinate_degree, element.degree)
    return CellGeometryTransition.reconstruct_surface(
        source.mesh,
        source.geometry,
        target,
        reference_map=reference_maps,
        maximum_displacement=reconstruction_bound,
        maximum_continuity_residual=reconstruction_bound,
        coordinate_degree=coordinate_degree,
    )


def _rebind_surface_metric_witness(
    witness: SurfaceChartWitness | SourceBoundaryChartWitness,
    target: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> SurfaceChartWitness | SourceBoundaryChartWitness:
    geometry_id = cell_geometry_id(geometry)
    target_ids = set(int(value) for value in _cell_ids(target))
    witness_ids = set(
        int(value) for value in np.asarray(witness.cell_global_ids, dtype=np.int64)
    )
    if witness.topology_id != target.topology_id or not witness_ids <= target_ids:
        raise ValueError(
            "Surface metric chart witness does not bind its reconstructed topology."
        )
    if isinstance(witness, SurfaceChartWitness):
        if witness_ids != target_ids:
            raise ValueError(
                "Surface chart witness must cover every reconstructed target cell."
            )
        return SurfaceChartWitness(
            geometry_id=geometry_id,
            topology_id=target.topology_id,
            domain_id=witness.domain_id,
            cell_global_ids=witness.cell_global_ids,
            patches=witness.patches,
            charts=witness.charts,
            source_root_cell_ids=witness.source_root_cell_ids,
            source_piece_ids=witness.source_piece_ids,
            geometry_entity_ids=witness.geometry_entity_ids,
            occurrence_paths=witness.occurrence_paths,
            chart_rational_tokens=witness.chart_rational_tokens,
            root_mesh_cell_bindings=tuple(
                None
                if binding is None
                else SurfaceRootMeshCellBinding(
                    geometry_id, target.topology_id, binding.root_cell_global_id
                )
                for binding in witness.root_mesh_cell_bindings
            ),
        )
    if isinstance(witness, SourceBoundaryChartWitness):
        return SourceBoundaryChartWitness(
            mesh_id=target.mesh_id,
            geometry_id=geometry_id,
            topology_id=target.topology_id,
            source_id=witness.source_id,
            source_revision=witness.source_revision,
            source_query_id=witness.source_query_id,
            chart_dimension=witness.chart_dimension,
            reference_dimension=witness.reference_dimension,
            cell_global_ids=witness.cell_global_ids,
            entity_global_ids=witness.entity_global_ids,
            local_entity_indices=witness.local_entity_indices,
            target_reference_vertices=witness.target_reference_vertices,
            target_reference_rational_tokens=(witness.target_reference_rational_tokens),
            target_face_vertex_global_ids=witness.target_face_vertex_global_ids,
            source_chart_ids=witness.source_chart_ids,
            source_entity_ids=witness.source_entity_ids,
            occurrence_paths=witness.occurrence_paths,
            source_parameters=witness.source_parameters,
            source_parameter_rational_tokens=witness.source_parameter_rational_tokens,
            orientations=witness.orientations,
            implicit_cell_keys=(
                witness.implicit_cell_keys
                if witness.implicit_cell_keys.shape[0]
                else None
            ),
            implicit_root_edge_keys=(
                witness.implicit_root_edge_keys
                if witness.implicit_root_edge_keys.shape[0]
                else None
            ),
        )
    raise TypeError("Unsupported surface metric chart witness.")


def _finalize_native(
    prepared: PreparedMeshAdaptation,
    edit: CellTopologyEdit,
    kind: MeshTransitionKind,
    /,
    *,
    conservative: bool,
    periodic_edge_lock: PeriodicRefinementEdgeLock | None = None,
    refinement_parent_cell_ids: ArrayLike | None = None,
    refinement_source_edge_ids: ArrayLike | None = None,
    refinement_edge_vertex_ids: ArrayLike | None = None,
    mixed_outcome: MixedAdaptationOutcome | None = None,
    polyhedral_outcome: PolyhedralAdaptationOutcome | None = None,
    surface_metric_outcome: SurfaceMetricOutcome | None = None,
    coverage_renewal: PolyhedralCoverageRenewal | None = None,
) -> _NativeTarget:
    """Assemble, inherit organization and associations, certify, and bind transfer."""

    source = prepared.source
    periodic_source = source.mesh.periodic_topology is not None
    if polyhedral_outcome is None:
        mesh, lineage, stencil = assemble_topology_edit(
            source.mesh,
            edit,
            numeric_version=f"adaptation:{prepared.prepared_id}",
            bind_periodic=not periodic_source,
        )
    else:
        mesh = polyhedral_outcome.target_mesh
        lineage = polyhedral_outcome.lineage
        stencil = polyhedral_outcome.vertex_stencil
        if (
            lineage.source_topology_id != source.mesh.topology_id
            or lineage.target_topology_id != mesh.topology_id
            or stencil.source_entity_set_id != source.mesh.entity_set(0).entity_set_id
            or stencil.target_entity_set_id != mesh.entity_set(0).entity_set_id
        ):
            raise ValueError(
                "Polyhedral adaptation outcome endpoints do not match the source."
            )
    coarsening = edit.coarsening
    exact_bisection_route = prepared.policy.route in (
        MeshAdaptationRoute.NATIVE_BISECTION,
        MeshAdaptationRoute.DEVICE_BISECTION,
    )
    if (
        coarsening is None
        and exact_bisection_route
        and isinstance(prepared.request, MarkedMeshAdaptation)
    ):
        hierarchy = prepared.request.hierarchy
        if hierarchy is not None and not isinstance(hierarchy, BisectionHierarchy):
            raise TypeError(
                "Native bisection hierarchy must be BisectionHierarchy or None."
            )
        if hierarchy is not None:
            coarsening = _bisection_coarsening_witness(source.mesh, mesh, edit, hierarchy)
    quotient_rows = None
    if polyhedral_outcome is not None:
        geometry = polyhedral_outcome.target_geometry
        geometry.resolve(mesh)
        reference_maps = ()
    elif mixed_outcome is not None:
        geometry, reference_maps = _mixed_geometry_for_target(
            source, mesh, lineage, mixed_outcome
        )
    elif surface_metric_outcome is not None:
        geometry = CellGeometrySpec.affine(mesh)
        reference_maps = surface_metric_outcome.geometry.reference_maps
    else:
        mesh, geometry, reference_maps, quotient_rows = _restricted_target_geometry(
            source, mesh, edit, coarsening=coarsening
        )
    if periodic_source:
        edit = edit._replace(coordinates=np.asarray(mesh.coordinates, dtype=np.float64))
        quotient_bank = _quotient_chart_bank(source, mesh, quotient_rows)
        if periodic_edge_lock is not None:
            if (
                refinement_parent_cell_ids is None
                or refinement_source_edge_ids is None
                or refinement_edge_vertex_ids is None
            ):
                raise ValueError(
                    "Periodic edge-locked adaptation lacks exact split-operation evidence."
                )
            candidate = stage_periodic_refinement_candidate(
                source.mesh,
                mesh,
                stencil,
                periodic_edge_lock,
                np.asarray(refinement_parent_cell_ids, dtype=np.int64),
                np.asarray(refinement_source_edge_ids, dtype=np.int64),
                np.asarray(refinement_edge_vertex_ids, dtype=np.int64),
                exact_coordinate_source=quotient_bank,
            )
            mesh, lineage, stencil = bind_periodic_topology_edit(
                source.mesh,
                mesh,
                stencil,
                edit,
                candidate.target.periodic_topology,
                exact_coordinate_source=quotient_bank,
            )
            if mesh.mesh_id != candidate.target.mesh_id:
                raise ValueError(
                    "Periodic candidate rebinding changed exact target mesh content."
                )
        else:
            if source.mesh.topological_dimension == 2 and all(
                block.cell_kind == "triangle" for block in mesh.blocks
            ):
                periodic_candidate = stage_periodic_topology_edit(
                    source.mesh, edit, exact_coordinate_source=quotient_bank
                )
                periodic_topology = (
                    None
                    if periodic_candidate is None
                    else periodic_candidate.periodic_topology
                )
            else:
                periodic_topology = periodic_topology_after_edit(
                    source.mesh, mesh, stencil, exact_coordinate_source=quotient_bank
                )
            if periodic_topology is None:
                raise ValueError(
                    "Periodic topology transition did not retain its quotient descriptor."
                )
            mesh, lineage, stencil = bind_periodic_topology_edit(
                source.mesh,
                mesh,
                stencil,
                edit,
                periodic_topology,
                exact_coordinate_source=quotient_bank,
            )
        if quotient_rows is not None:
            geometry = _quotient_chart_target_geometry(mesh, quotient_bank)
    transition_geometry = None
    surface_chart_witness = None
    if surface_metric_outcome is not None:
        transition_geometry = _surface_metric_geometry_transition(
            source, mesh, surface_metric_outcome
        )
        mesh = transition_geometry.target_mesh
        geometry = transition_geometry.target_geometry
        reference_maps = surface_metric_outcome.geometry.reference_maps
        edit = edit._replace(coordinates=np.asarray(mesh.coordinates, dtype=np.float64))
        surface_chart_witness = _rebind_surface_metric_witness(
            surface_metric_outcome.geometry.target_chart_witness, mesh, geometry
        )
        surface_metric_outcome = surface_metric_outcome._replace(
            geometry=surface_metric_outcome.geometry._replace(
                target_chart_witness=surface_chart_witness
            )
        )
    retained_inputs = (
        None if source.certification is None else source.certification.inputs
    )
    requires_source_transition = (
        retained_inputs is not None
        and retained_inputs.source_coverage_witness is not None
        and coverage_renewal is None
    )
    has_exact_reference_maps = (
        bool(reference_maps) or edit.refinement is not None or coarsening is not None
    )
    preserves_topology = source.mesh.topology_id == mesh.topology_id

    if surface_metric_outcome is None and (
        exact_bisection_route
        or requires_source_transition
        or has_exact_reference_maps
        or preserves_topology
    ):
        try:
            transition_geometry = _transition_geometry_record(
                source,
                mesh,
                geometry,
                reference_maps=reference_maps,
                refinement=edit.refinement,
                coarsening=coarsening,
            )
        except ValueError as error:
            material_state_retained = prepared.prior_transition is not None and (
                prepared.prior_transition.volume_material_chain is not None
                or prepared.prior_transition.material_reference is not None
            )
            if (
                str(error)
                == "Exact reference maps do not prove source/target coordinate-map equality."
                and not exact_bisection_route
                and not _has_retained_source_claims(source)
                and not material_state_retained
            ):
                transition_geometry = None
                reference_maps = ()
            else:
                raise
    patches, zones, labels, boundaries = _inherited_organization(source, mesh, lineage)
    association_transfer = prepared.policy.association_transfer
    associations = (
        association_transfer.propagate(source, lineage, mesh, vertex_stencil=stencil)
        if association_transfer is not None and source.associations
        else ()
    )
    target_boundary = None
    if source.boundary is not None:
        if not isinstance(
            association_transfer,
            (ImplicitAssociationTransfer, SurfaceAssociationTransfer),
        ):
            raise TypeError(
                "Boundary remapping requires an implicit or surface association transfer."
            )
        target_boundary = association_transfer.remap_boundary(source, lineage, mesh)
        if target_boundary is None:
            raise ValueError(
                "Association transfer did not produce a target boundary model."
            )
    surface_source = source.surface_source
    if surface_metric_outcome is not None and surface_source is not None:
        surface_source = target_surface_source(
            surface_source, mesh, geometry, associations, surface_metric_outcome
        )
    certification_inputs = _renew_transition_certification_inputs(
        source,
        mesh,
        geometry,
        lineage,
        reference_maps=reference_maps,
        transition=transition_geometry,
        coarsening=coarsening,
        mixed_coarsening_maps=(
            () if mixed_outcome is None else mixed_outcome.coarsening_maps
        ),
        coverage_renewal=coverage_renewal,
        fidelity_witness_override=(
            surface_metric_outcome.geometry.target_chart_witness
            if surface_metric_outcome is not None
            and source.certification is not None
            and source.certification.inputs.fidelity_witness is not None
            else None
        ),
    )
    layer_curving_evidence = None
    if source.layer_curving_evidence is not None:
        target_chart_witness = _retarget_source_boundary_chart_witness(
            source,
            mesh,
            geometry,
            lineage,
            source.layer_curving_evidence.chart_witness,
            reference_maps=reference_maps,
            coarsening=coarsening,
            mixed_coarsening_maps=(
                () if mixed_outcome is None else mixed_outcome.coarsening_maps
            ),
        )
        from ._layer_curving import renew_layer_curving_source_evidence

        layer_curving_evidence = renew_layer_curving_source_evidence(
            source.layer_curving_evidence,
            source.mesh,
            source.geometry,
            mesh,
            geometry,
            target_chart_witness,
            lineage,
            reference_maps,
            coarsening=coarsening,
            mixed_coarsening_maps=(
                () if mixed_outcome is None else mixed_outcome.coarsening_maps
            ),
        )
    target = certify_cell_mesh(
        mesh,
        source.coordinate_contract,
        geometry=geometry,
        boundary=target_boundary,
        audit_policy=_acceptance_audit_policy(
            prepared.policy.audit_policy, certification_inputs
        ),
        certification_inputs=certification_inputs,
        patches=patches,
        zones=zones,
        labels=labels,
        associations=associations,
        surface_source=surface_source,
        region_boundary_evidence=boundaries,
        planar_embedding=source.planar_embedding,
        layer_curving_evidence=layer_curving_evidence,
    )
    if target.mesh.topology_id != mesh.topology_id:
        raise ValueError("Certification reordered the adapted topology.")
    material_chain = _extend_prior_volume_material_chain(
        prepared.prior_transition,
        source,
        target,
        lineage,
        material_reference=prepared.material_reference,
        reference_maps=reference_maps,
        coarsening=coarsening,
        coordinate_transition=transition_geometry,
    )
    transition = CellMeshTransition(
        source.mesh.mesh_id,
        source.mesh.topology_id,
        target,
        lineage,
        kind,
        vertex_stencil=stencil,
        geometry_transition=transition_geometry,
        volume_material_chain=material_chain,
    )
    stencil_valid = np.asarray(stencil.valid, dtype=np.bool_)
    cell_lineage = lineage.entity_lineage(target.mesh.topological_dimension)
    target_cell_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in target.mesh.blocks
        )
    )
    related_target_cells = np.asarray(cell_lineage.target_global_ids, dtype=np.int64)
    created_target_cells = np.asarray(cell_lineage.created_target_ids, dtype=np.int64)
    has_cell_ancestry = (
        np.all(np.isin(target_cell_ids, related_target_cells))
        and not np.intersect1d(target_cell_ids, created_target_cells).size
    )
    transfer = None
    if (
        has_cell_ancestry
        and stencil.preserves_constants
        and np.all(np.any(stencil_valid, axis=1))
    ):
        try:
            transfer = stencil.as_transfer(
                source.mesh.vertex_global_ids,
                source_topology_id=source.mesh.topology_id,
                target_topology_id=target.mesh.topology_id,
                preserves_linear=True,
                conservative=conservative,
                source_coordinates=source.mesh.coordinates,
                target_coordinates=target.mesh.coordinates,
                source_measures=(
                    _p1_vertex_measures(source.mesh) if conservative else None
                ),
                target_measures=(
                    _p1_vertex_measures(target.mesh) if conservative else None
                ),
            )
        except ValueError as error:
            if str(error) not in (
                "Transfer claims linear preservation but moves coordinates.",
                "Transfer claims conservation but changes DOF measures.",
            ):
                raise
            transfer = None
    return _NativeTarget(
        target, transition, lineage, stencil, transfer, surface_chart_witness
    )


def _execute_bisection_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    request = prepared.request
    if not isinstance(request, MarkedMeshAdaptation):
        raise TypeError("Native bisection requires MarkedMeshAdaptation.")
    hierarchy = request.hierarchy
    if hierarchy is not None and not isinstance(hierarchy, BisectionHierarchy):
        raise TypeError("Native bisection hierarchy must be BisectionHierarchy or None.")
    policy = prepared.policy
    constraints = prepared.constraints
    outcome = execute_bisection(
        prepared.source.mesh,
        np.asarray(request.refine_cell_ids, dtype=np.int64),
        np.asarray(request.coarsen_cell_ids, dtype=np.int64),
        hierarchy=hierarchy,
        compatibility=policy.compatibility,
        protected_edges=constraints.protected_edge_keys,
        protected_vertices=constraints.protected_vertex_ids,
        cell_classes=constraints.cell_classes,
        facet_classes=constraints.facet_classes,
        periodic_edge_lock=request.periodic_edge_lock,
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
        periodic_edge_lock=request.periodic_edge_lock,
        refinement_parent_cell_ids=outcome.refinement_parent_cell_global_ids,
        refinement_source_edge_ids=outcome.refinement_source_edge_global_ids,
        refinement_edge_vertex_ids=outcome.refinement_edge_vertex_ids,
    )
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


def _execute_mixed_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    request = prepared.request
    if not isinstance(request, MarkedMeshAdaptation):
        raise TypeError("NATIVE_MIXED requires MarkedMeshAdaptation.")
    hierarchy = request.hierarchy
    if hierarchy is not None and not isinstance(hierarchy, MixedAdaptationHierarchy):
        raise TypeError(
            "NATIVE_MIXED hierarchy must be MixedAdaptationHierarchy or None."
        )
    mesh = prepared.source.mesh
    constraints = prepared.constraints
    protected_cells = _protected_scope_rows(mesh, prepared.policy.protected_scopes, 3)
    protected_faces = _protected_scope_rows(mesh, prepared.policy.protected_scopes, 2)
    outcome = adapt_mixed_mesh(
        mesh,
        refine_cell_ids=np.asarray(request.refine_cell_ids, dtype=np.int64),
        coarsen_cell_ids=np.asarray(request.coarsen_cell_ids, dtype=np.int64),
        hierarchy=hierarchy,
        cell_classes=constraints.cell_classes,
        protected_vertex_ids=constraints.protected_vertex_ids,
        protected_cell_ids=np.asarray(mesh.entity_set(3).entity_ids)[protected_cells],
        protected_edge_keys=constraints.protected_edge_keys,
        protected_face_keys=entity_keys(mesh, 2)[protected_faces],
        layer_columns=request.layer_columns,
        source_geometry=prepared.source.geometry,
    )
    evidence = outcome.evidence
    refinement_rejected = not np.all(
        np.isin(evidence.requested_refine_cell_ids, evidence.refined_cell_ids)
    )
    partial = bool(evidence.rejected_coarsening_ids.size) or refinement_rejected
    refined = bool(evidence.refined_cell_ids.size) or bool(evidence.closure_cell_ids.size)
    coarsened = bool(evidence.coarsened_cell_ids.size)
    schedule_changed = bool(evidence.schedule_changed_cell_ids.size)
    if not refined and not coarsened and not schedule_changed:
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
        conservative=False,
        mixed_outcome=outcome,
    )
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
        outcome.cell_content_transfer,
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
    if evidence.converged:
        return MeshAdaptationStatus.COMPLETE
    if isinstance(evidence, LocalMetricEvidence) and evidence.rejected_uncertain:
        return MeshAdaptationStatus.NUMERICAL_STOP
    if evidence.stalled:
        return MeshAdaptationStatus.STALLED if topology else MeshAdaptationStatus.COMPLETE
    return MeshAdaptationStatus.NUMERICAL_STOP


def _execute_metric_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    request = prepared.request
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
        topology_operations=topology,
        relocation=policy.relocation,
    )
    evidence = outcome.evidence
    status = _metric_status(evidence, topology)
    applied = evidence.splits + evidence.collapses + evidence.flips + evidence.relocations
    if applied == 0:
        # Retain whether the unchanged source satisfied the request or a
        # numerical/geometry obstruction prevented progress.
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
        # ty: ignore[unresolved-attribute]
        _target_metric(request.metric, native.target.mesh, outcome.metric),
        evidence,
        None,
    )


def _execute_tetra_metric_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    request = prepared.request
    if not isinstance(request, MetricMeshAdaptation):
        raise TypeError("The 3D tetra metric route requires MetricMeshAdaptation.")
    policy = prepared.policy
    constraints = prepared.constraints
    outcome = execute_tetra_metric_adaptation(
        prepared.source,
        constraints.metric_values,
        cell_classes=constraints.cell_classes,
        facet_classes=constraints.facet_classes,
        edge_classes=constraints.edge_classes,
        protected_edges=constraints.protected_edge_mask,
        fixed_vertices=constraints.fixed_vertex_mask,
        predicate_mode=resolve_host_predicate_mode(policy.predicate_mode),
        topology_operations=True,
        relocation=policy.relocation,
    )
    evidence = outcome.evidence
    match evidence.status:
        case TetraMetricStatus.COMPLETE:
            status = MeshAdaptationStatus.COMPLETE
        case TetraMetricStatus.STALLED:
            status = MeshAdaptationStatus.STALLED
        case TetraMetricStatus.NUMERICAL_STOP:
            status = MeshAdaptationStatus.NUMERICAL_STOP
    applied = (
        evidence.splits
        + evidence.collapses
        + evidence.flips23
        + evidence.flips32
        + evidence.ring_reconnections
        + evidence.relocations
    )
    if applied == 0:
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


def _execute_surface_metric_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    request = prepared.request
    if not isinstance(request, (MetricMeshAdaptation, RelocationMeshAdaptation)):
        raise TypeError("NATIVE_METRIC_SURFACE requires a metric or relocation request.")
    policy = prepared.policy
    constraints = prepared.constraints
    implicit_profile = None
    if prepared.source.surface_source is None:
        transfer = policy.association_transfer
        if not isinstance(transfer, ImplicitAssociationTransfer):
            raise TypeError(
                "Implicit surface metric adaptation requires ImplicitAssociationTransfer."
            )
        implicit_profile = transfer.profile
    controller = SurfaceMetricController(
        prepared.source,
        constraints.metric_values,
        resolve_host_predicate_mode(policy.predicate_mode),
        topology_operations=isinstance(request, MetricMeshAdaptation),
        relocation=policy.relocation,
        cell_classes=constraints.cell_classes,
        edge_classes=constraints.edge_classes,
        protected_edges=constraints.protected_edge_mask,
        fixed_vertices=constraints.fixed_vertex_mask,
        implicit_profile=implicit_profile,
    )
    outcome = controller.stage()
    match outcome.status:
        case SurfaceMetricStatus.COMPLETE:
            status = MeshAdaptationStatus.COMPLETE
        case SurfaceMetricStatus.STALLED:
            status = MeshAdaptationStatus.STALLED
        case SurfaceMetricStatus.NUMERICAL_STOP:
            status = MeshAdaptationStatus.NUMERICAL_STOP
    evidence = outcome
    applied = outcome.splits + outcome.collapses + outcome.flips + outcome.relocations
    if applied == 0:
        return _unchanged(prepared, evidence, None, status=status)
    native = _finalize_native(
        prepared,
        outcome.edit,
        MeshTransitionKind.REMESH,
        conservative=False,
        surface_metric_outcome=outcome,
    )
    surface_chart_witness = native.surface_chart_witness
    if surface_chart_witness is None:
        raise ValueError("Surface metric finalization lost its target chart witness.")
    outcome = outcome._replace(
        geometry=outcome.geometry._replace(target_chart_witness=surface_chart_witness)
    )
    return _RouteOutcome(
        status,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        _target_metric(request.metric, native.target.mesh, outcome.metric),
        outcome,
        None,
        surface_chart_witness=surface_chart_witness,
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


def _provider_association_transfer(
    transfer: AssociationTransfer | None, /
) -> BRepAssociationTransfer | None:
    if transfer is None or isinstance(transfer, BRepAssociationTransfer):
        return transfer
    raise TypeError("Provider routes require BRepAssociationTransfer.")


def _execute_polyhedral_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    request = prepared.request
    if not isinstance(request, PolyhedralAdaptationRequest):
        raise TypeError("NATIVE_POLYHEDRAL requires PolyhedralAdaptationRequest.")
    source = prepared.source
    mesh = source.mesh
    protected_faces = _protected_scope_rows(mesh, prepared.policy.protected_scopes, 2)
    certification_inputs = (
        None if source.certification is None else source.certification.inputs
    )
    source_coverage_witness = (
        None
        if certification_inputs is None
        else certification_inputs.source_coverage_witness
    )
    source_distance_witness = None
    source_coverage_profile = None
    if source_coverage_witness is not None:
        if not isinstance(
            source_coverage_witness,
            (
                RestrictedDelaunayBoundaryWitness,
                RestrictedDelaunayCoverageTransportWitness,
            ),
        ):
            raise ValueError(
                "Polyhedral coverage renewal transports only restricted-Delaunay coverage."
            )
        if certification_inputs is None or not isinstance(
            certification_inputs.fidelity_witness, SourceBoundaryDistanceWitness
        ):
            raise ValueError(
                "Polyhedral coverage renewal requires a retained source distance witness."
            )
        if not isinstance(certification_inputs.domain, AnalyticImplicitProfile):
            raise ValueError(
                "Polyhedral coverage renewal requires its retained analytic profile."
            )
        source_distance_witness = certification_inputs.fidelity_witness
        source_coverage_profile = certification_inputs.domain
    outcome = adapt_polyhedral_mesh(
        mesh,
        request,
        source_geometry=source.geometry,
        cell_classes=prepared.constraints.cell_classes,
        protected_face_ids=np.asarray(mesh.entity_set(2).entity_ids)[protected_faces],
        source_coverage_witness=source_coverage_witness,
        source_distance_witness=source_distance_witness,
        source_coverage_profile=source_coverage_profile,
    )
    evidence = outcome.evidence
    if not evidence.changed_cell_ids.size:
        return _unchanged(
            prepared, evidence, outcome.hierarchy, status=MeshAdaptationStatus.UNCHANGED
        )
    native = _finalize_native(
        prepared,
        outcome.edit,
        MeshTransitionKind.REMESH,
        conservative=False,
        polyhedral_outcome=outcome,
        coverage_renewal=outcome.coverage_renewal,
    )
    return _RouteOutcome(
        MeshAdaptationStatus.COMPLETE,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        None,
        evidence,
        outcome.hierarchy,
        None,
        outcome.conservative_remap,
    )


def _execute_mmg_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    policy = prepared.policy
    provider = policy.provider
    # ty: ignore[unresolved-attribute]
    plan = provider.plan(
        prepared.source,
        # ty: ignore[unresolved-attribute]
        metric=prepared.request.metric,
        required=policy.protected_scopes,
        # ty: ignore[invalid-argument-type]
        options=policy.provider_options,
        audit_policy=policy.audit_policy,
        association_transfer=_provider_association_transfer(policy.association_transfer),
    )
    # ty: ignore[invalid-argument-type, missing-argument, unresolved-attribute]
    result = provider.execute(plan)
    # ty: ignore[unresolved-attribute]
    return _provider_outcome(prepared, result.mesh, result.metric, result)


def _execute_omega_h_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    policy = prepared.policy
    # ty: ignore[unresolved-attribute]
    result = policy.provider.execute(
        # ty: ignore[invalid-argument-type]
        prepared.source,
        # ty: ignore[too-many-positional-arguments, unresolved-attribute]
        prepared.request.metric,
        # ty: ignore[invalid-argument-type, unknown-argument]
        options=policy.provider_options,
        # ty: ignore[unknown-argument]
        association_transfer=_provider_association_transfer(policy.association_transfer),
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
    mesh, _, _ = hp_active_cell_mesh(
        result.topology,
        result.geometry,
        numeric_version=f"adaptation:{prepared.prepared_id}",
    )
    if source.certification is not None and not _piecewise_linear_geometry(
        source.mesh, source.geometry
    ):
        raise ValueError(
            "The hp route does not retain a certified non-P1 source coordinate map."
        )
    if source.attributes:
        raise ValueError(
            "The hp route has no exact attribute transfer for retained mesh scopes."
        )
    target_geometry = CellGeometrySpec.affine(mesh)
    target_epoch = FiniteElementHPEpoch(
        mesh,
        result.topology,
        result.geometry,
        finite_element_hp_interface_plan(result.topology, result.geometry),
    )
    # ty: ignore[invalid-argument-type]
    lineage = project_hp_lineage(epoch, target_epoch, result.lineage)
    material_requested = prepared.material_reference is not None or (
        prepared.prior_transition is not None
        and (
            prepared.prior_transition.volume_material_chain is not None
            or prepared.prior_transition.material_reference is not None
        )
    )
    hp_reference_maps = (
        _identity_reference_maps_for_transition(source.mesh, mesh)
        if material_requested or source.layer_curving_evidence is not None
        else ()
    )
    hp_geometry_transition = (
        _transition_geometry_record(
            source, mesh, target_geometry, reference_maps=hp_reference_maps
        )
        if hp_reference_maps
        else None
    )
    layer_curving_evidence = None
    if source.layer_curving_evidence is not None:
        target_chart_witness = _retarget_source_boundary_chart_witness(
            source,
            mesh,
            target_geometry,
            lineage,
            source.layer_curving_evidence.chart_witness,
            reference_maps=hp_reference_maps,
        )
        from ._layer_curving import renew_layer_curving_source_evidence

        layer_curving_evidence = renew_layer_curving_source_evidence(
            source.layer_curving_evidence,
            source.mesh,
            source.geometry,
            mesh,
            target_geometry,
            target_chart_witness,
            lineage,
            hp_reference_maps,
        )
    patches, zones, labels, boundaries = _inherited_organization(source, mesh, lineage)
    association_transfer = prepared.policy.association_transfer
    if source.associations and association_transfer is None:
        raise ValueError(
            "The hp route needs an exact transfer for retained source associations."
        )
    associations = (
        association_transfer.propagate(source, lineage, mesh, vertex_stencil=None)
        if association_transfer is not None and source.associations
        else ()
    )
    certification_inputs = _renew_transition_certification_inputs(
        source,
        mesh,
        target_geometry,
        lineage,
        reference_maps=hp_reference_maps,
        transition=hp_geometry_transition,
    )
    target = certify_cell_mesh(
        mesh,
        source.coordinate_contract,
        geometry=target_geometry,
        audit_policy=_acceptance_audit_policy(
            prepared.policy.audit_policy, certification_inputs
        ),
        certification_inputs=certification_inputs,
        patches=patches,
        zones=zones,
        labels=labels,
        associations=associations,
        surface_source=source.surface_source,
        region_boundary_evidence=boundaries,
        planar_embedding=source.planar_embedding,
        layer_curving_evidence=layer_curving_evidence,
    )
    if target.mesh.topology_id != mesh.topology_id:
        raise ValueError("Certification reordered the hp target topology.")
    material_chain = _extend_prior_volume_material_chain(
        prepared.prior_transition,
        source,
        target,
        lineage,
        material_reference=prepared.material_reference,
        reference_maps=hp_reference_maps,
        coordinate_transition=hp_geometry_transition,
    )
    transition = CellMeshTransition(
        source.mesh.mesh_id,
        source.mesh.topology_id,
        target,
        lineage,
        kind,
        geometry_transition=hp_geometry_transition,
        volume_material_chain=material_chain,
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
    prepared: PreparedMeshAdaptation, target: CellMeshingResult, /
) -> MeshingComplianceReport:
    """Bind the route's scientific findings to the adaptation request."""
    return MeshingComplianceReport(
        f"mesh-adaptation:{prepared.prepared_id}",
        issues=target.compliance.issues,
        requested=target.compliance.requested,
        achieved=target.compliance.achieved,
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
        _compliance(prepared, outcome.target),
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
        case MeshAdaptationRoute.NATIVE_MIXED:
            outcome = _execute_mixed_route(prepared)
        case MeshAdaptationRoute.NATIVE_POLYHEDRAL:
            outcome = _execute_polyhedral_route(prepared)
        case MeshAdaptationRoute.NATIVE_METRIC_2D:
            outcome = _execute_metric_route(prepared)
        case MeshAdaptationRoute.NATIVE_METRIC_SURFACE:
            outcome = _execute_surface_metric_route(prepared)
        case MeshAdaptationRoute.NATIVE_METRIC_3D:
            outcome = _execute_tetra_metric_route(prepared)
        case MeshAdaptationRoute.DEVICE_BISECTION:
            # Lazy: the device epoch module builds on this module's result types.
            from ._device_adaptation import _execute_device_bisection_route

            outcome = _execute_device_bisection_route(prepared)
        case MeshAdaptationRoute.DEVICE_METRIC_2D:
            from ._device_metric import _execute_device_metric_route

            outcome = _execute_device_metric_route(prepared)
        case MeshAdaptationRoute.DEVICE_METRIC_3D:
            from ._device_tetra_metric import _execute_device_tetra_metric_route

            outcome = _execute_device_tetra_metric_route(prepared)
        case MeshAdaptationRoute.MMG:
            outcome = _execute_mmg_route(prepared)
        case MeshAdaptationRoute.OMEGA_H:
            outcome = _execute_omega_h_route(prepared)
        case MeshAdaptationRoute.HP:
            outcome = _execute_hp_route(prepared)
        case _:
            raise TypeError("route must be MeshAdaptationRoute.")
    return _adaptation_result(prepared, outcome, started)


__all__ = [
    "MarkedMeshAdaptation",
    "MeshAdaptationPolicy",
    "MeshAdaptationResult",
    "MeshAdaptationRoute",
    "MeshAdaptationStatus",
    "MetricMeshAdaptation",
    "PreparedMeshAdaptation",
    "RelocationMeshAdaptation",
    "execute_mesh_adaptation",
    "prepare_mesh_adaptation",
    "project_hp_lineage",
    "_extend_prior_volume_material_chain",
    "_identity_lineage",
    "_identity_reference_maps_for_transition",
    "_inherited_organization",
    "_inherited_scope",
    "_piecewise_linear_geometry",
    "_renew_transition_certification_inputs",
    "_retarget_source_boundary_chart_witness",
    "_transition_geometry_record",
]
