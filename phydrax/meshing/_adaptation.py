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
from typing import final, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from numpy.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._adaptive_simplex import AdaptiveSimplexPolicy
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
from ..geometry._predicates import PredicateMode, resolve_host_predicate_mode
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ._assembly import MeshPart
from ._association import BRepAssociationTransfer
from ._audit import CellMeshAuditPolicy
from ._bisection import (
    BisectionCompatibility,
    BisectionEvidence,
    BisectionHierarchy,
    execute_bisection,
)
from ._canonical import certify_cell_mesh
from ._contracts import MeshingLimits
from ._device_metric import DeviceMetricEvidence
from ._distribution import (
    MeshDistribution,
    MeshDistributionTransition,
    MeshPartitionPolicy,
    prepare_distribution_transition,
)
from ._lineage import (
    CellMeshTransition,
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    MeshTransitionKind,
    VertexInterpolationStencil,
)
from ._local_metric import execute_local_metric_adaptation, LocalMetricEvidence
from ._metric import MeshMetricField
from ._organization import MeshLabel, MeshPatch, MeshZone
from ._result import CellMeshingResult, MeshingComplianceReport
from ._scope import MeshingEntityKind, MeshingScope, resolve_mesh_scope
from ._topology_edit import (
    assemble_topology_edit,
    entity_keys,
    key_rows,
    SimplexTopologyEdit,
)
from .providers._mmg import MmgAdaptationResult, MmgOptions, MmgProvider
from .providers._omega_h import OmegaHAdaptationResult, OmegaHOptions, OmegaHProvider


class MeshAdaptationStatus(StrEnum):
    """Outcome of one executed adaptation.

    ``COMPLETE``: every requested operation was realized (for metric routes the
    unit-mesh criterion holds). ``PARTIAL``: some requested marks were rejected
    (protected entities, ineligible coarsening families); the evidence lists them.
    ``PASS_LIMIT``: a metric route stopped at ``maximum_passes`` before the
    unit-mesh criterion. ``STALLED``: no admissible operation remains while the
    criterion is unmet. ``UNCHANGED``: nothing was applied; the target is the
    source result itself.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    PASS_LIMIT = "pass_limit"
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
    """

    NATIVE_BISECTION = "native_bisection"
    NATIVE_METRIC_2D = "native_metric_2d"
    DEVICE_BISECTION = "device_bisection"
    DEVICE_METRIC_2D = "device_metric_2d"
    MMG = "mmg"
    OMEGA_H = "omega_h"
    HP = "hp"


_DEVICE_ROUTES = frozenset(
    (MeshAdaptationRoute.DEVICE_BISECTION, MeshAdaptationRoute.DEVICE_METRIC_2D)
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
    """Source cells marked for refinement and for coarsening, by global ID.

    ``hierarchy`` carries the state of earlier adaptations: the
    `BisectionHierarchy` returned by a previous NATIVE_BISECTION result (required
    for coarsening, and to continue its refinement labels), or the source
    `FiniteElementHPEpoch` of the HP route.
    """

    refine_cell_ids: Array
    coarsen_cell_ids: Array
    hierarchy: BisectionHierarchy | FiniteElementHPEpoch | None
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        refine_cell_ids: ArrayLike = (),
        coarsen_cell_ids: ArrayLike = (),
        /,
        *,
        hierarchy: BisectionHierarchy | FiniteElementHPEpoch | None = None,
    ):
        refine = _identifier_vector(refine_cell_ids, "refine_cell_ids")
        coarsen = _identifier_vector(coarsen_cell_ids, "coarsen_cell_ids")
        if np.intersect1d(refine, coarsen).size:
            raise ValueError("A cell cannot be marked for refinement and coarsening.")
        if hierarchy is not None and not isinstance(
            hierarchy, (BisectionHierarchy, FiniteElementHPEpoch)
        ):
            raise TypeError(
                "hierarchy must be BisectionHierarchy, FiniteElementHPEpoch, or None."
            )
        self.refine_cell_ids = jnp.asarray(refine)
        self.coarsen_cell_ids = jnp.asarray(coarsen)
        self.hierarchy = hierarchy
        self.request_id = canonical_fingerprint(
            {
                "kind": "marked-mesh-adaptation",
                "refine": array_tree_fingerprint(refine),
                "coarsen": array_tree_fingerprint(coarsen),
                "hierarchy": _hierarchy_id(hierarchy),
            }
        )


@final
class MetricMeshAdaptation(StrictModule, NonTrainableState):
    """Adapt toward a unit mesh of one vertex metric bound to the source vertices."""

    metric: MeshMetricField
    request_id: str = eqx.field(static=True)

    def __init__(self, metric: MeshMetricField, /):
        if not isinstance(metric, MeshMetricField):
            raise TypeError("metric must be MeshMetricField.")
        self.metric = metric
        self.request_id = canonical_fingerprint(
            {"kind": "metric-mesh-adaptation", "metric": metric.metric_id}
        )


@final
class RelocationMeshAdaptation(StrictModule, NonTrainableState):
    """Move vertices toward one vertex metric without changing the topology."""

    metric: MeshMetricField
    request_id: str = eqx.field(static=True)

    def __init__(self, metric: MeshMetricField, /):
        if not isinstance(metric, MeshMetricField):
            raise TypeError("metric must be MeshMetricField.")
        self.metric = metric
        self.request_id = canonical_fingerprint(
            {"kind": "relocation-mesh-adaptation", "metric": metric.metric_id}
        )


MeshAdaptationRequest: TypeAlias = (
    MarkedMeshAdaptation | MetricMeshAdaptation | RelocationMeshAdaptation
)


def _hierarchy_id(hierarchy: BisectionHierarchy | FiniteElementHPEpoch | None, /):
    match hierarchy:
        case None:
            return None
        case BisectionHierarchy():
            return hierarchy.hierarchy_id
        case FiniteElementHPEpoch():
            return hierarchy.epoch_id
        case _:
            raise TypeError("Unsupported adaptation hierarchy.")


def _provider_options_record(options: MmgOptions | OmegaHOptions | None, /):
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
            | MeshAdaptationRoute.DEVICE_BISECTION
            | MeshAdaptationRoute.DEVICE_METRIC_2D
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
    ``maximum_passes`` bounds metric passes and ``maximum_closure_iterations`` the
    bisection closure. ``relocation`` enables vertex relocation in metric passes.
    ``device_policy`` (required by, and only by, device routes) fixes the capacity
    bucket of the device state. ``distribution`` with ``partition_policy`` carries
    cell ownership through the lineage. ``association_transfer`` carries the
    source's B-Rep associations: B-Rep corner vertices are fixed, ambiguously
    classified edges are protected, CAD classes separate cell, facet, and edge
    classes, and the target associations are propagated through the native
    lineage or re-derived after provider remeshing. Nonconforming HP active meshes
    require an ``audit_policy`` that records rather than rejects nonmanifold
    (hanging) vertices.
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
    association_transfer: BRepAssociationTransfer | None
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
        association_transfer: BRepAssociationTransfer | None = None,
    ):
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
            if not isinstance(association_transfer, BRepAssociationTransfer):
                raise TypeError(
                    "association_transfer must be BRepAssociationTransfer or None."
                )
            if route is MeshAdaptationRoute.HP:
                raise ValueError("HP adaptation cannot carry B-Rep associations.")
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
        # B-Rep corners are fixed, ambiguous edges protected (ambiguity rejects the
        # operation), and CAD classes separate regions, facets, and feature edges.
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
        fixed |= classes[0].dimensions == 0
        protected_edges |= ~classes[1].resolved
        cell_classes = _row_classes(np.column_stack((cell_classes, codes[dimension])))
        facet_classes = _row_classes(
            np.column_stack((facet_classes, codes[dimension - 1]))
        )
        edge_codes = np.where(classes[1].dimensions < dimension, 1 + codes[1], 0)
    protected_vertex_ids = vertex_ids[fixed]
    fixed |= np.any(_membership(mesh, organization, 0), axis=1)
    planar = policy.route in (
        MeshAdaptationRoute.NATIVE_METRIC_2D,
        MeshAdaptationRoute.DEVICE_METRIC_2D,
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
            if planar
            else np.zeros((edge_count,), dtype=np.int64)
        ),
        metric_values=(
            _metric_rows(mesh, request.metric)
            if planar
            else np.zeros((0, mesh.ambient_dimension, mesh.ambient_dimension))
        ),
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


def _require_native_source(
    source: CellMeshingResult, transfer: BRepAssociationTransfer | None, /
) -> None:
    if source.boundary is not None or source.attributes:
        raise ValueError(
            "Native adaptation carries patches, zones, labels, and transferred B-Rep "
            "associations only; boundary models and attributes need an explicit remap."
        )
    if source.associations:
        if transfer is None:
            raise ValueError(
                "Geometry associations require the policy's association_transfer."
            )
        transfer.source_associations(source)
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


def _check_route_request(
    source: CellMeshingResult,
    request: MeshAdaptationRequest,
    policy: MeshAdaptationPolicy,
    /,
) -> None:
    mesh = source.mesh
    kinds = {block.cell_kind for block in mesh.blocks}
    match policy.route:
        case MeshAdaptationRoute.NATIVE_BISECTION | MeshAdaptationRoute.DEVICE_BISECTION:
            if not isinstance(request, MarkedMeshAdaptation) or isinstance(
                request.hierarchy, FiniteElementHPEpoch
            ):
                raise TypeError(
                    f"{policy.route.name} requires a marked bisection request."
                )
            if kinds != {"triangle"} and kinds != {"tetrahedron"}:
                raise ValueError("Bisection requires triangle or tetrahedron blocks.")
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
        case MeshAdaptationRoute.MMG | MeshAdaptationRoute.OMEGA_H:
            if not isinstance(request, MetricMeshAdaptation):
                raise TypeError("Provider routes require a metric request.")
        case MeshAdaptationRoute.HP:
            if not isinstance(request, MarkedMeshAdaptation) or not isinstance(
                request.hierarchy, FiniteElementHPEpoch
            ):
                raise TypeError("HP requires a marked request with its hp epoch.")
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
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CellMeshingResult,
        request: MeshAdaptationRequest,
        policy: MeshAdaptationPolicy,
        constraints: _AdaptationConstraints,
        /,
    ):
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
        request, (MarkedMeshAdaptation, MetricMeshAdaptation, RelocationMeshAdaptation)
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
            | MeshAdaptationRoute.DEVICE_BISECTION
            | MeshAdaptationRoute.DEVICE_METRIC_2D
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
    return PreparedMeshAdaptation(source, request, policy, constraints)


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
    | MmgAdaptationResult
    | OmegaHAdaptationResult
    | FiniteElementHPRefinementResult
)


def _evidence_id(evidence: MeshAdaptationEvidence | None, /):
    match evidence:
        case None:
            return None
        case BisectionEvidence() | LocalMetricEvidence() | DeviceMetricEvidence():
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

    ``transition``, ``lineage``, ``stencil``, and ``transfer`` are ``None`` exactly
    when the status is UNCHANGED; provider routes have unknown vertex lineage and
    therefore no stencil or transfer. ``transfer`` maps P1 vertex values from the
    source vertex row order to the target vertex row order. ``metric`` is the
    target-bound metric of metric routes, ``hierarchy`` the state to pass to the
    next marked request, and ``elapsed_seconds`` is wall-time evidence outside
    ``result_id``.
    """

    status: MeshAdaptationStatus = eqx.field(static=True)
    route: MeshAdaptationRoute = eqx.field(static=True)
    source: CellMeshingResult
    target: CellMeshingResult
    transition: CellMeshTransition | None
    lineage: MeshLineage | None
    stencil: VertexInterpolationStencil | None
    transfer: FiniteElementTopologyTransfer | None
    metric: MeshMetricField | None
    compliance: MeshingComplianceReport
    evidence: MeshAdaptationEvidence | None
    hierarchy: BisectionHierarchy | FiniteElementHPEpoch | None
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
    ):
        unchanged = outcome.status is MeshAdaptationStatus.UNCHANGED
        if unchanged != (outcome.transition is None) or (
            unchanged and outcome.target.result_id != prepared.source.result_id
        ):
            raise ValueError("Exactly the UNCHANGED status keeps the source result.")
        if outcome.transition is not None and (
            outcome.transition.target.result_id != outcome.target.result_id
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
    hierarchy: BisectionHierarchy | FiniteElementHPEpoch | None


def _unchanged(
    prepared: PreparedMeshAdaptation,
    evidence: MeshAdaptationEvidence | None,
    hierarchy: BisectionHierarchy | FiniteElementHPEpoch | None,
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
) -> tuple[tuple[MeshPatch, ...], tuple[MeshZone, ...], tuple[MeshLabel, ...]]:
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
        )
        for patch in source.patches
    )
    labels = tuple(
        MeshLabel(label.name, _inherited_scope(label.scope, lineage, target, label.name))
        for label in source.labels
    )
    return patches, tuple(zones), labels


class _NativeTarget(NamedTuple):
    target: CellMeshingResult
    transition: CellMeshTransition
    lineage: MeshLineage
    stencil: VertexInterpolationStencil
    transfer: FiniteElementTopologyTransfer


def _finalize_native(
    prepared: PreparedMeshAdaptation,
    edit: SimplexTopologyEdit,
    kind: MeshTransitionKind,
    /,
    *,
    conservative: bool,
) -> _NativeTarget:
    """Assemble, inherit organization and associations, certify, and bind transfer."""

    source = prepared.source
    mesh, lineage, stencil = assemble_topology_edit(
        source.mesh, edit, numeric_version=f"adaptation:{prepared.prepared_id}"
    )
    patches, zones, labels = _inherited_organization(source, mesh, lineage)
    association_transfer = prepared.policy.association_transfer
    associations = (
        association_transfer.propagate(source, lineage, mesh)
        if association_transfer is not None and source.associations
        else ()
    )
    target = certify_cell_mesh(
        mesh,
        source.coordinate_contract,
        audit_policy=prepared.policy.audit_policy,
        patches=patches,
        zones=zones,
        labels=labels,
        associations=associations,
    )
    if target.mesh.topology_id != mesh.topology_id:
        raise ValueError("Certification reordered the adapted topology.")
    transition = CellMeshTransition(
        source.mesh.mesh_id,
        source.mesh.topology_id,
        target,
        lineage,
        kind,
        vertex_stencil=stencil,
    )
    transfer = stencil.as_transfer(
        source.mesh.vertex_global_ids,
        source_topology_id=source.mesh.topology_id,
        target_topology_id=target.mesh.topology_id,
        preserves_linear=True,
        conservative=conservative,
        source_coordinates=source.mesh.coordinates,
        target_coordinates=target.mesh.coordinates,
        source_measures=_p1_vertex_measures(source.mesh) if conservative else None,
        target_measures=_p1_vertex_measures(target.mesh) if conservative else None,
    )
    return _NativeTarget(target, transition, lineage, stencil, transfer)


def _execute_bisection_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    request = prepared.request
    policy = prepared.policy
    constraints = prepared.constraints
    outcome = execute_bisection(
        prepared.source.mesh,
        np.asarray(request.refine_cell_ids, dtype=np.int64),
        np.asarray(request.coarsen_cell_ids, dtype=np.int64),
        hierarchy=request.hierarchy,
        compatibility=policy.compatibility,
        protected_edges=constraints.protected_edge_keys,
        protected_vertices=constraints.protected_vertex_ids,
        cell_classes=constraints.cell_classes,
        facet_classes=constraints.facet_classes,
        maximum_closure_iterations=policy.maximum_closure_iterations,
    )
    evidence = outcome.evidence
    refined = evidence.bisections > 0 or evidence.uniform_refinement_applied
    coarsened = evidence.coarsened_vertices > 0
    partial = (
        np.asarray(evidence.rejected_refinement_ids).size > 0
        or np.asarray(evidence.rejected_coarsening_ids).size > 0
    )
    if not refined and not coarsened:
        return _unchanged(prepared, evidence, outcome.hierarchy)
    match (refined, coarsened):
        case (True, False):
            kind = MeshTransitionKind.REFINE
        case (False, True):
            kind = MeshTransitionKind.COARSEN
        case _:
            kind = MeshTransitionKind.REMESH
    native = _finalize_native(prepared, outcome.edit, kind, conservative=not coarsened)
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
        maximum_gradation=metric.maximum_gradation,
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
        # Nothing applied: the target is the source (UNCHANGED); whether the
        # criterion holds or the run stalled is in the evidence.
        return _unchanged(prepared, evidence, None)
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
    provider = policy.provider
    plan = provider.plan(
        prepared.source,
        metric=prepared.request.metric,
        required=policy.protected_scopes,
        options=policy.provider_options,
        limits=policy.limits,
        audit_policy=policy.audit_policy,
        association_transfer=policy.association_transfer,
    )
    result = provider.execute(plan)
    return _provider_outcome(prepared, result.mesh, result.metric, result)


def _execute_omega_h_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    policy = prepared.policy
    result = policy.provider.execute(
        prepared.source,
        prepared.request.metric,
        options=policy.provider_options,
        limits=policy.limits,
        association_transfer=policy.association_transfer,
    )
    if result.target is None:
        raise ValueError("A serial Omega_h adaptation must return its target.")
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


def _complete_hp_parents(epoch: FiniteElementHPEpoch, marked: np.ndarray, /):
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
    epoch = request.hierarchy
    source = prepared.source
    refine = np.asarray(request.refine_cell_ids, dtype=np.int64)
    coarsen = np.asarray(request.coarsen_cell_ids, dtype=np.int64)
    if refine.size:
        closed, _ = balanced_hp_refinement_ids(epoch.topology, epoch.interfaces, refine)
        result = refine_tensor_hp_cells(
            epoch.topology, epoch.geometry, np.asarray(closed, dtype=np.int64)
        )
        kind = MeshTransitionKind.REFINE
        partial = False
    else:
        parents = _complete_hp_parents(epoch, coarsen)
        if parents.size == 0:
            return _unchanged(prepared, None, epoch)
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
    target = certify_cell_mesh(
        mesh, source.coordinate_contract, audit_policy=prepared.policy.audit_policy
    )
    target_epoch = FiniteElementHPEpoch(
        target.mesh,
        result.topology,
        result.geometry,
        finite_element_hp_interface_plan(result.topology, result.geometry),
    )
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
    prepared: PreparedMeshAdaptation, target: CellMeshingResult, elapsed: float, /
) -> MeshingComplianceReport:
    limits = prepared.policy.limits
    audit = target.audit
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
    return MeshingComplianceReport(
        f"mesh-adaptation:{prepared.prepared_id}",
        issues=issues,
        requested=tuple(
            (f"maximum_{name}", float(maximum)) for name, _, maximum in observations
        ),
        achieved=tuple((name, float(actual)) for name, actual, _ in observations),
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
            outcome.lineage,
            policy=policy.partition_policy,
        )
    elapsed = time.monotonic() - started
    return MeshAdaptationResult(
        prepared,
        outcome,
        _compliance(prepared, outcome.target, elapsed),
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
        case MeshAdaptationRoute.DEVICE_BISECTION:
            # Lazy: the device epoch module builds on this module's result types.
            from ._device_adaptation import _execute_device_bisection_route

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
]
