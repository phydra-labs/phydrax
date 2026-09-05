#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import heapq
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh, PolygonalConnectivity
from ..discretization.fem import FiniteElementTransferBundle
from ._adaptation import refine_triangle_mesh
from ._audit import audit_cell_mesh, CellMeshAuditPolicy, CellMeshAuditReport
from ._contracts import MeshingLimits
from ._lineage import CellMeshTransition
from ._optimization import (
    MeshOptimizationResult,
    optimize_cell_mesh,
    TargetMatrixOptimizationPlan,
)
from ._quality import evaluate_cell_quality
from ._result import CellMeshingResult, MeshingComplianceReport
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import (
    MeshMetricField,
    normalize_mesh_metric,
    ResolvedSizeField,
    SizeFieldDomain,
)


def mesh_proposal_scope(
    source: CellMeshingResult,
    dimension: int,
    entity_ids: ArrayLike | None = None,
    /,
) -> MeshingScope:
    """Bind proposal entities to the exact certified mesh and result revision.

    Proposal values always follow the scope's sorted global-ID order, not the
    mesh's storage order. Omitting IDs selects all entities of the given degree.
    """
    if not isinstance(source, CellMeshingResult):
        raise TypeError("source must be CellMeshingResult.")
    entities = source.mesh.entity_set(dimension)
    scope = MeshingScope(
        source.mesh.mesh_id,
        source.result_id,
        MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        entities.entity_ids if entity_ids is None else entity_ids,
    )
    _scope_rows(source, scope)
    return scope


def _scope_rows(source: CellMeshingResult, scope: MeshingScope) -> np.ndarray:
    if not isinstance(scope, MeshingScope):
        raise TypeError("scope must be MeshingScope.")
    mesh = source.mesh
    if (
        scope.entity_kind is not MeshingEntityKind.MESH
        or scope.source_id != mesh.mesh_id
        or scope.source_revision != source.result_id
        or scope.entity_dimension > mesh.topological_dimension
    ):
        raise ValueError(
            "Proposal scope has a stale or incompatible mesh revision binding."
        )
    entities = mesh.entity_set(scope.entity_dimension)
    if scope.entity_set_id != entities.entity_set_id:
        raise ValueError("Proposal scope has a stale entity-set binding.")
    identifiers = np.asarray(entities.entity_ids)
    requested = np.asarray(scope.entity_ids)
    order = np.argsort(identifiers)
    positions = np.searchsorted(identifiers[order], requested)
    if np.any(positions >= identifiers.size) or not np.array_equal(
        identifiers[order[np.minimum(positions, identifiers.size - 1)]], requested
    ):
        raise ValueError("Proposal scope contains unknown entity IDs.")
    rows = order[positions]
    if not np.all(np.asarray(entities.active_mask)[rows]):
        raise ValueError("Proposal scope contains inactive entities.")
    return rows


class _AbstractMeshProposal(StrictModule, NonTrainableState):
    scope: MeshingScope
    values: Array
    coordinate_contract_id: str = eqx.field(static=True)
    proposer_id: str = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        values: ArrayLike,
        *,
        dimension: int,
        value_shape: tuple[int, ...],
        proposer_id: str,
    ):
        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        _scope_rows(source, scope)
        if scope.entity_dimension != dimension:
            raise ValueError("Proposal scope has the wrong entity dimension.")
        data = np.asarray(values, dtype=float)
        if data.shape != (scope.entity_ids.size, *value_shape) or not np.all(
            np.isfinite(data)
        ):
            raise ValueError(
                "Proposal values must be finite and aligned with the sorted scope IDs."
            )
        proposer = str(proposer_id).strip()
        if not proposer:
            raise ValueError("proposer_id must be non-empty.")
        self.scope = scope
        self.values = jnp.asarray(data)
        self.coordinate_contract_id = source.coordinate_contract.spatial_id
        self.proposer_id = proposer
        self.proposal_id = canonical_fingerprint(
            {
                "kind": type(self).__name__,
                "scope": scope.scope_id,
                "values": array_tree_fingerprint(self.values),
                "coordinate_contract": self.coordinate_contract_id,
                "proposer": proposer,
            }
        )


class MeshMarkingProposal(_AbstractMeshProposal):
    """Untrusted cell priorities; positive scores request one refinement step."""

    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        scores: ArrayLike,
        /,
        *,
        proposer_id: str,
    ):
        super().__init__(
            source,
            scope,
            scores,
            dimension=source.mesh.topological_dimension,
            value_shape=(),
            proposer_id=proposer_id,
        )


class MeshSizeProposal(_AbstractMeshProposal):
    """Untrusted vertex sizes, expressed in the source coordinate length unit."""

    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        sizes: ArrayLike,
        /,
        *,
        proposer_id: str,
    ):
        super().__init__(
            source, scope, sizes, dimension=0, value_shape=(), proposer_id=proposer_id
        )


class MeshMetricProposal(_AbstractMeshProposal):
    """Untrusted vertex tensors; projection repairs symmetry and definiteness."""

    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        metrics: ArrayLike,
        /,
        *,
        proposer_id: str,
    ):
        dimension = source.mesh.ambient_dimension
        super().__init__(
            source,
            scope,
            metrics,
            dimension=0,
            value_shape=(dimension, dimension),
            proposer_id=proposer_id,
        )


class MeshCoordinateProposal(_AbstractMeshProposal):
    """Untrusted vertex targets in one explicit, unchanged coordinate contract."""

    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        coordinates: ArrayLike,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        proposer_id: str,
    ):
        if (
            not isinstance(coordinate_contract, SpatialCoordinateContract)
            or coordinate_contract.spatial_id != source.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Coordinate proposals require the exact source coordinate contract."
            )
        super().__init__(
            source,
            scope,
            coordinates,
            dimension=0,
            value_shape=(source.mesh.ambient_dimension,),
            proposer_id=proposer_id,
        )


MeshProposal = (
    MeshMarkingProposal | MeshSizeProposal | MeshMetricProposal | MeshCoordinateProposal
)


class MeshProposalSafetyPolicy(StrictModule, NonTrainableState):
    """Trusted, revision-bound constraints, independent of proposer evidence.

    Protected scopes preserve their entities and fix their incident vertices.
    Coordinate bounds and displacement are in source coordinate units. Limits
    are admission limits on candidate payloads, not a process memory quota or
    a preemptive timeout. Wall-time overruns cannot be committed.
    """

    source_result_id: str = eqx.field(static=True)
    protected_scopes: tuple[MeshingScope, ...]
    limits: MeshingLimits
    audit_policy: CellMeshAuditPolicy
    coordinate_bounds: Array | None
    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    maximum_anisotropy: float = eqx.field(static=True)
    maximum_gradation: float = eqx.field(static=True)
    maximum_displacement: float = eqx.field(static=True)
    maximum_marked_cells: int = eqx.field(static=True)
    maximum_optimization_iterations: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CellMeshingResult,
        /,
        *,
        minimum_size: float,
        maximum_size: float,
        maximum_displacement: float,
        protected_scopes: tuple[MeshingScope, ...] = (),
        limits: MeshingLimits | None = None,
        audit_policy: CellMeshAuditPolicy | None = None,
        coordinate_bounds: ArrayLike | None = None,
        maximum_anisotropy: float = 100.0,
        maximum_gradation: float = 1.3,
        maximum_marked_cells: int = 100_000,
        maximum_optimization_iterations: int = 50,
    ):
        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        minimum, maximum = float(minimum_size), float(maximum_size)
        anisotropy, gradation = float(maximum_anisotropy), float(maximum_gradation)
        displacement = float(maximum_displacement)
        if not np.all(
            np.isfinite((minimum, maximum, anisotropy, gradation, displacement))
        ):
            raise ValueError("Safety bounds must be finite.")
        if (
            minimum <= 0
            or maximum < minimum
            or anisotropy < 1
            or gradation < 1
            or displacement < 0
        ):
            raise ValueError(
                "Invalid proposal size, anisotropy, gradation, or displacement bounds."
            )
        normal_floor = np.sqrt(np.finfo(float).tiny)
        if (
            minimum < normal_floor
            or maximum > 1.0 / normal_floor
            or anisotropy > 1.0 / normal_floor
        ):
            raise ValueError("Safety bounds must admit representable metric eigenvalues.")
        marked, iterations = (
            int(maximum_marked_cells),
            int(maximum_optimization_iterations),
        )
        if marked < 0 or iterations <= 0:
            raise ValueError(
                "Mark capacity must be non-negative and optimization iterations positive."
            )
        protected = tuple(protected_scopes)
        for scope in protected:
            _scope_rows(source, scope)
        limits_ = MeshingLimits() if limits is None else limits
        audit_ = CellMeshAuditPolicy() if audit_policy is None else audit_policy
        if not isinstance(limits_, MeshingLimits) or not isinstance(
            audit_, CellMeshAuditPolicy
        ):
            raise TypeError(
                "Safety policy requires MeshingLimits and CellMeshAuditPolicy."
            )
        bounds = (
            None
            if coordinate_bounds is None
            else np.asarray(coordinate_bounds, dtype=float)
        )
        if bounds is not None:
            if (
                bounds.shape != (2, source.mesh.ambient_dimension)
                or not np.all(np.isfinite(bounds))
                or np.any(bounds[0] > bounds[1])
            ):
                raise ValueError(
                    "coordinate_bounds must contain finite ordered lower/upper vectors."
                )
            if np.any(source.mesh.coordinates < bounds[0]) or np.any(
                source.mesh.coordinates > bounds[1]
            ):
                raise ValueError("Source coordinates lie outside the safety bounds.")
        self.source_result_id = source.result_id
        self.protected_scopes = protected
        self.limits = limits_
        self.audit_policy = audit_
        self.coordinate_bounds = None if bounds is None else jnp.asarray(bounds)
        self.minimum_size, self.maximum_size = minimum, maximum
        self.maximum_anisotropy, self.maximum_gradation = anisotropy, gradation
        self.maximum_displacement = displacement
        self.maximum_marked_cells = marked
        self.maximum_optimization_iterations = iterations
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mesh-proposal-safety-policy",
                "source": source.result_id,
                "protected": tuple(scope.scope_id for scope in protected),
                "limits": limits_.limits_id,
                "audit": audit_.policy_id,
                "bounds": None
                if bounds is None
                else array_tree_fingerprint(self.coordinate_bounds),
                "minimum_size": minimum,
                "maximum_size": maximum,
                "anisotropy": anisotropy,
                "gradation": gradation,
                "displacement": displacement,
                "marks": marked,
                "iterations": iterations,
            }
        )


def _entity_closure(
    mesh: CellMesh, dimension: int, rows: np.ndarray, target: int
) -> np.ndarray:
    for degree in range(dimension, target, -1):
        relation = mesh.topology.incidences[degree - 1].relation
        valid = np.asarray(relation.valid) & np.isin(
            np.asarray(relation.target_indices), rows
        )
        rows = np.unique(np.asarray(relation.source_indices)[valid])
    return rows


def _protected_vertices(
    source: CellMeshingResult, policy: MeshProposalSafetyPolicy
) -> np.ndarray:
    fixed = np.zeros(source.mesh.coordinates.shape[0], dtype=bool)
    for scope in policy.protected_scopes:
        rows = _entity_closure(
            source.mesh, scope.entity_dimension, _scope_rows(source, scope), 0
        )
        fixed[rows] = True
    return fixed


def _payload_bytes(value) -> int:
    return sum(
        leaf.nbytes
        for leaf in jax.tree_util.tree_leaves(value)
        if isinstance(leaf, (jax.Array, np.ndarray))
    )


def _limit_issues(result: CellMeshingResult, limits: MeshingLimits) -> tuple[str, ...]:
    counts = result.audit.entity_counts
    observations = (
        ("vertices", result.audit.vertex_count, limits.maximum_vertices),
        ("edges", counts[1] if len(counts) > 1 else 0, limits.maximum_edges),
        ("faces", counts[2] if len(counts) > 2 else 0, limits.maximum_faces),
        ("cells", counts[-1], limits.maximum_cells),
        (
            "connectivity_entries",
            result.audit.connectivity_entries,
            limits.maximum_connectivity_entries,
        ),
        ("data_bytes", _payload_bytes(result), limits.maximum_data_bytes),
    )
    return tuple(
        f"maximum_{name}" for name, actual, maximum in observations if actual > maximum
    )


def _check_binding(source, proposal, policy):
    if not isinstance(source, CellMeshingResult):
        raise TypeError("source must be CellMeshingResult.")
    if not isinstance(
        proposal,
        (
            MeshMarkingProposal,
            MeshSizeProposal,
            MeshMetricProposal,
            MeshCoordinateProposal,
        ),
    ):
        raise TypeError("proposal must be a typed mesh proposal.")
    if not isinstance(policy, MeshProposalSafetyPolicy):
        raise TypeError("policy must be MeshProposalSafetyPolicy.")
    _scope_rows(source, proposal.scope)
    if policy.source_result_id != source.result_id:
        raise ValueError("Safety policy has a stale result revision binding.")
    if proposal.coordinate_contract_id != source.coordinate_contract.spatial_id:
        raise ValueError("Proposal coordinate contract does not match the source.")
    if source.coordinate_contract.coordinate_system != "cartesian":
        raise ValueError("Native proposal routes require Cartesian coordinates.")
    source.audit.require_passed()
    if not source.compliance.passed or _limit_issues(source, policy.limits):
        raise ValueError(
            "Source does not satisfy proposal compliance and capacity limits."
        )
    for scope in policy.protected_scopes:
        _scope_rows(source, scope)


def _grade_sizes(values: np.ndarray, edges: np.ndarray, growth: float) -> np.ndarray:
    """Exact graph envelope; no iteration cap can leave a distant edge unsafe."""
    sizes = np.log(values)
    adjacency = [[] for _ in sizes]
    for first, second in edges:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    queue = [(float(value), row) for row, value in enumerate(sizes)]
    heapq.heapify(queue)
    step = np.log(growth)
    while queue:
        value, row = heapq.heappop(queue)
        if value != sizes[row]:
            continue
        for neighbor in adjacency[row]:
            candidate = value + step
            if candidate < sizes[neighbor]:
                sizes[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return np.exp(sizes)


def _project_metric(scope, raw, edges, policy) -> MeshMetricField:
    symmetric = 0.5 * raw + 0.5 * np.swapaxes(raw, -1, -2)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    lower, upper = 1.0 / policy.maximum_size**2, 1.0 / policy.minimum_size**2
    eigenvalues = np.clip(eigenvalues, lower, upper)
    repaired = (eigenvectors * eigenvalues[:, None, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
    repaired = 0.5 * repaired + 0.5 * np.swapaxes(repaired, -1, -2)
    metric = normalize_mesh_metric(
        MeshMetricField(
            scope,
            repaired,
            minimum_size=policy.minimum_size,
            maximum_size=policy.maximum_size,
            maximum_anisotropy=policy.maximum_anisotropy,
            maximum_gradation=policy.maximum_gradation,
        )
    )
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(metric.values))
    logarithms = np.log(eigenvalues)
    sizes = np.exp(-0.5 * logarithms.mean(axis=1))
    graded = _grade_sizes(sizes, edges, policy.maximum_gradation)
    # Uniform log-eigenvalue shifts, saturated at the upper bound, realize the
    # scalar graph envelope without increasing anisotropy or violating size bounds.
    target = -2.0 * np.log(graded)
    low = np.zeros(sizes.size)
    high = np.maximum(0.0, np.log(upper) - logarithms.min(axis=1))
    for _ in range(64):
        middle = 0.5 * (low + high)
        mean = np.minimum(logarithms + middle[:, None], np.log(upper)).mean(axis=1)
        low = np.where(mean < target, middle, low)
        high = np.where(mean < target, high, middle)
    eigenvalues = np.exp(np.minimum(logarithms + high[:, None], np.log(upper)))
    projected = (eigenvectors * eigenvalues[:, None, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
    projected = 0.5 * projected + 0.5 * np.swapaxes(projected, -1, -2)
    return MeshMetricField(
        scope,
        projected,
        minimum_size=policy.minimum_size,
        maximum_size=policy.maximum_size,
        maximum_anisotropy=policy.maximum_anisotropy,
        maximum_gradation=policy.maximum_gradation,
    )


def _coordinate_projector(source, proposal, policy):
    points = jnp.asarray(source.mesh.coordinates)
    movable = np.zeros(points.shape[0], dtype=bool)
    movable[_scope_rows(source, proposal.scope)] = True
    fixed = jnp.asarray(~movable | _protected_vertices(source, policy))
    bounds = policy.coordinate_bounds

    def project(values):
        if bounds is not None:
            values = jnp.clip(values, bounds[0], bounds[1])
        delta = values - points
        lengths = jnp.linalg.norm(delta, axis=1, keepdims=True)
        factor = jnp.minimum(
            1.0,
            policy.maximum_displacement
            / jnp.maximum(lengths, jnp.finfo(points.dtype).tiny),
        )
        return jnp.where(fixed[:, None], points, points + delta * factor)

    return fixed, project


def _safe_marks(source, scores, policy):
    mesh = source.mesh
    cells = np.asarray(mesh.blocks[0].global_ids)
    cell_edges = np.asarray(mesh.connectivity.cell_edges)[:, :3]
    forbidden_edges = np.zeros(mesh.entity_set(1).count, dtype=bool)
    for scope in policy.protected_scopes:
        if scope.entity_dimension > 0:
            rows = _entity_closure(
                mesh, scope.entity_dimension, _scope_rows(source, scope), 1
            )
            forbidden_edges[rows] = True
    # Excluding all incident cells is conservative for longest-edge bisection:
    # no requested split can enter a protected cell through conformity closure.
    allowed = ~np.any(forbidden_edges[cell_edges], axis=1)
    order = np.lexsort((cells, -scores))
    counts = source.audit.entity_counts
    # Native FE refinement retains both a dense primal and its dual pullback.
    # Budget that known payload before the trusted refinement allocates it.
    vertices = source.mesh.coordinates.shape[0]
    transfer_row_bytes = 2 * vertices * source.mesh.coordinates.dtype.itemsize
    transfer_capacity = max(
        0,
        (policy.limits.maximum_data_bytes - _payload_bytes(source)) // transfer_row_bytes
        - vertices,
    )
    capacity = min(
        policy.maximum_marked_cells,
        transfer_capacity,
        max(0, policy.limits.maximum_vertices - counts[0]),
        max(0, (policy.limits.maximum_cells - counts[2]) // 2),
        max(0, (policy.limits.maximum_faces - counts[2]) // 2),
        max(0, (policy.limits.maximum_edges - counts[1]) // 3),
        max(
            0,
            (
                policy.limits.maximum_connectivity_entries
                - source.audit.connectivity_entries
            )
            // 18,
        ),
    )
    selected = order[(scores[order] > 0) & allowed[order]][:capacity]
    return np.sort(cells[selected]).astype(np.int64, copy=False)


class MeshProposalProjection(StrictModule, NonTrainableState):
    """Projected proposal evidence; this is not a mesh or a certification."""

    proposal: MeshProposal
    policy: MeshProposalSafetyPolicy
    marked_cell_ids: Array
    size_field: ResolvedSizeField | None
    metric: MeshMetricField | None
    target_coordinates: Array | None
    projection_id: str = eqx.field(static=True)

    def __init__(
        self, proposal, policy, marked_cell_ids, size_field, metric, target_coordinates
    ):
        self.proposal, self.policy = proposal, policy
        self.marked_cell_ids = jnp.asarray(marked_cell_ids, dtype=jnp.int64)
        self.size_field, self.metric = size_field, metric
        self.target_coordinates = target_coordinates
        self.projection_id = canonical_fingerprint(
            {
                "kind": "mesh-proposal-projection",
                "proposal": proposal.proposal_id,
                "policy": policy.policy_id,
                "marks": array_tree_fingerprint(self.marked_cell_ids),
                "size": None if size_field is None else size_field.field_id,
                "metric": None if metric is None else metric.metric_id,
                "coordinates": None
                if target_coordinates is None
                else array_tree_fingerprint(target_coordinates),
            }
        )


def project_mesh_proposal(
    source: CellMeshingResult,
    proposal: MeshProposal,
    policy: MeshProposalSafetyPolicy,
    /,
) -> MeshProposalProjection:
    """Deterministically project untrusted values without generating a mesh.

    Marking, size and metric proposals currently request one native T3
    longest-edge refinement step. A metric controls directional edge marking,
    not an unsupported anisotropic remesher or an achieved-size guarantee.
    """
    _check_binding(source, proposal, policy)
    mesh = source.mesh
    rows = _scope_rows(source, proposal.scope)
    sizes, metric, target = None, None, None
    marks = np.empty(0, dtype=np.int64)
    if isinstance(proposal, MeshCoordinateProposal):
        _, project = _coordinate_projector(source, proposal, policy)
        target = project(jnp.asarray(mesh.coordinates).at[rows].set(proposal.values))
        # A safe target must itself be a valid optimization reference. Backtrack
        # toward the certified source, rather than accepting an inverted target.
        for _ in range(64):
            if bool(jnp.all(evaluate_cell_quality(mesh, target).valid)):
                break
            target = project(0.5 * (target + mesh.coordinates))
        else:
            raise ValueError("Coordinate proposal has no certifiable projected target.")
    else:
        if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind != "triangle":
            raise ValueError(
                "Native marking, size and metric proposals require one T3 block."
            )
        connectivity = mesh.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise TypeError("Native T3 proposals require polygonal connectivity.")
        scores = np.zeros(mesh.blocks[0].global_ids.size)
        if isinstance(proposal, MeshMarkingProposal):
            scores[rows] = np.asarray(proposal.values)
        else:
            if rows.size != mesh.coordinates.shape[0]:
                raise ValueError(
                    "Size and metric proposals must cover every source vertex."
                )
            edges = np.asarray(connectivity.edges)
            # Field arrays are scope ordered; connectivity and points are mesh ordered.
            inverse = np.empty(rows.size, dtype=np.int64)
            inverse[rows] = np.arange(rows.size)
            field_edges = inverse[edges]
            if isinstance(proposal, MeshSizeProposal):
                values = _grade_sizes(
                    np.clip(
                        np.asarray(proposal.values),
                        policy.minimum_size,
                        policy.maximum_size,
                    ),
                    field_edges,
                    policy.maximum_gradation,
                )
                sizes = ResolvedSizeField(
                    SizeFieldDomain.SAMPLE_CLOUD,
                    np.asarray(mesh.coordinates)[rows],
                    values,
                    source_control_ids=(proposal.proposal_id, policy.policy_id),
                )
                values = values[inverse]
                lengths = np.linalg.norm(
                    np.asarray(mesh.coordinates)[edges[:, 1]]
                    - np.asarray(mesh.coordinates)[edges[:, 0]],
                    axis=1,
                )
                edge_scores = (
                    lengths / np.minimum(values[edges[:, 0]], values[edges[:, 1]]) - 1.0
                )
            else:
                metric = _project_metric(
                    proposal.scope, np.asarray(proposal.values), field_edges, policy
                )
                values = np.asarray(metric.values)[inverse]
                delta = (
                    np.asarray(mesh.coordinates)[edges[:, 1]]
                    - np.asarray(mesh.coordinates)[edges[:, 0]]
                )
                average = 0.5 * (values[edges[:, 0]] + values[edges[:, 1]])
                lengths = np.sqrt(
                    np.sum(delta * (average @ delta[..., None])[..., 0], axis=1)
                )
                edge_scores = lengths - 1.0
            scores = np.max(
                edge_scores[np.asarray(connectivity.cell_edges)[:, :3]], axis=1
            )
        marks = _safe_marks(source, scores, policy)
    return MeshProposalProjection(proposal, policy, marks, sizes, metric, target)


def _entity_vertex_signatures(mesh: CellMesh, dimension: int):
    vertices = [{int(identifier)} for identifier in np.asarray(mesh.vertex_global_ids)]
    for degree in range(1, dimension + 1):
        upper = [set() for _ in range(mesh.entity_set(degree).count)]
        relation = mesh.topology.incidences[degree - 1].relation
        valid = np.asarray(relation.valid)
        for lower_row, upper_row in zip(
            np.asarray(relation.source_indices)[valid],
            np.asarray(relation.target_indices)[valid],
            strict=True,
        ):
            upper[int(upper_row)].update(vertices[int(lower_row)])
        vertices = upper
    return tuple(tuple(sorted(values)) for values in vertices)


def _preservation_issues(source, candidate, projection):
    policy = projection.policy
    issues = []
    if candidate.coordinate_contract.spatial_id != source.coordinate_contract.spatial_id:
        issues.append("coordinate_contract")
    fixed = _protected_vertices(source, policy)
    source_ids = np.asarray(source.mesh.vertex_global_ids)
    target_ids = np.asarray(candidate.mesh.vertex_global_ids)
    lookup = {int(identifier): row for row, identifier in enumerate(target_ids)}
    for row in np.flatnonzero(fixed):
        target_row = lookup.get(int(source_ids[row]))
        if target_row is None or not np.array_equal(
            source.mesh.coordinates[row], candidate.mesh.coordinates[target_row]
        ):
            issues.append("protected_vertices")
            break
    signatures = {}
    for scope in policy.protected_scopes:
        dimension = scope.entity_dimension
        if dimension == 0 or source.mesh.topology_id == candidate.mesh.topology_id:
            continue
        # IDs alone do not establish preservation: intermediate entity IDs may
        # be regenerated by refinement. Compare the incident vertex identities.
        source_rows = _scope_rows(source, scope)
        if dimension not in signatures:
            signatures[dimension] = (
                _entity_vertex_signatures(source.mesh, dimension),
                _entity_vertex_signatures(candidate.mesh, dimension),
            )
        source_signatures, target_signatures = signatures[dimension]
        target_entities = candidate.mesh.entity_set(scope.entity_dimension)
        target_lookup = {
            int(identifier): row
            for row, identifier in enumerate(np.asarray(target_entities.entity_ids))
        }
        for wanted, row in zip(np.asarray(scope.entity_ids), source_rows, strict=True):
            target_row = target_lookup.get(int(wanted))
            if target_row is None:
                issues.append("protected_entities")
                break
            if source_signatures[int(row)] != target_signatures[target_row]:
                issues.append("protected_entities")
                break
    bounds = policy.coordinate_bounds
    if bounds is not None and (
        np.any(candidate.mesh.coordinates < bounds[0])
        or np.any(candidate.mesh.coordinates > bounds[1])
    ):
        issues.append("coordinate_bounds")
    if isinstance(projection.proposal, MeshCoordinateProposal):
        if candidate.mesh.topology_id != source.mesh.topology_id:
            issues.append("fixed_topology")
        else:
            delta = np.asarray(candidate.mesh.coordinates) - np.asarray(
                source.mesh.coordinates
            )
            tolerance = (
                16 * np.finfo(delta.dtype).eps * max(1.0, policy.maximum_displacement)
            )
            if np.any(
                np.linalg.norm(delta, axis=1) > policy.maximum_displacement + tolerance
            ):
                issues.append("maximum_displacement")
            movable = np.zeros(delta.shape[0], dtype=bool)
            movable[_scope_rows(source, projection.proposal.scope)] = True
            if np.any(delta[~movable] != 0):
                issues.append("coordinate_scope")
    return tuple(dict.fromkeys(issues))


class MeshProposalTransaction(StrictModule, NonTrainableState):
    """Prepared trusted result and separate safety evidence; source stays intact.

    A refinement exposes the native transition and transfer bundle for the
    solver's FiniteElementTopologyTransaction; commit here promotes only a mesh,
    never solution fields. Explicit rejection returns the identical source.
    """

    source: CellMeshingResult
    projection: MeshProposalProjection
    trusted_result: CellMeshingResult
    safety_audit: CellMeshAuditReport
    compliance: MeshingComplianceReport
    transition: CellMeshTransition | None
    transfer: FiniteElementTransferBundle | None
    optimization: MeshOptimizationResult | None
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        source,
        projection,
        trusted_result,
        safety_audit,
        compliance,
        transition,
        transfer,
        optimization,
    ):
        _check_binding(source, projection.proposal, projection.policy)
        if (
            safety_audit.mesh_id != trusted_result.mesh.mesh_id
            or safety_audit.policy_id != projection.policy.audit_policy.policy_id
        ):
            raise ValueError("Safety audit must match the candidate and safety policy.")
        if compliance.specification_id != projection.projection_id:
            raise ValueError("Compliance must be bound to the exact projected proposal.")
        if transition is not None and (
            transition.source_mesh_id != source.mesh.mesh_id
            or transition.target.result_id != trusted_result.result_id
            or transfer is None
        ):
            raise ValueError(
                "Proposal transition must match the source, candidate, and transfer."
            )
        if isinstance(projection.proposal, MeshCoordinateProposal):
            if (
                optimization is None
                or optimization.result.result_id != trusted_result.result_id
                or transition is not None
            ):
                raise ValueError(
                    "Coordinate proposal candidate must match its trusted optimization."
                )
        elif projection.marked_cell_ids.size:
            if transition is None or optimization is not None:
                raise ValueError(
                    "Marked proposal candidate requires trusted refinement evidence."
                )
        elif trusted_result.result_id != source.result_id:
            raise ValueError(
                "An empty projected marking must preserve the exact source result."
            )
        if transfer is not None and transfer.primal.shape != (
            trusted_result.mesh.coordinates.shape[0],
            source.mesh.coordinates.shape[0],
        ):
            raise ValueError(
                "Proposal transfer must match the source and candidate vertex counts."
            )
        self.source, self.projection, self.trusted_result = (
            source,
            projection,
            trusted_result,
        )
        self.safety_audit, self.compliance = safety_audit, compliance
        self.transition, self.transfer, self.optimization = (
            transition,
            transfer,
            optimization,
        )
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "mesh-proposal-transaction",
                "source": source.result_id,
                "projection": projection.projection_id,
                "result": trusted_result.result_id,
                "transfer": None if transfer is None else transfer.transfer_id,
                "audit": safety_audit.report_id,
                "compliance": compliance.report_id,
                "transition": None if transition is None else transition.transition_id,
                "optimization": None
                if optimization is None
                else optimization.optimization_id,
            }
        )

    @property
    def admissible(self) -> bool:
        return self.safety_audit.passed and self.compliance.passed

    def commit(
        self, current: CellMeshingResult, /, *, accept: bool = True
    ) -> CellMeshingResult:
        if not isinstance(accept, (bool, np.bool_)):
            raise TypeError("Proposal commit requires an explicit host boolean decision.")
        if (
            not isinstance(current, CellMeshingResult)
            or current.result_id != self.source.result_id
        ):
            raise ValueError("Cannot commit a proposal against a stale current revision.")
        _check_binding(current, self.projection.proposal, self.projection.policy)
        if not accept:
            return current
        issues = self.compliance.issues + _limit_issues(
            self.trusted_result, self.projection.policy.limits
        )
        issues += _preservation_issues(current, self.trusted_result, self.projection)
        if not self.admissible or issues:
            raise ValueError(
                "Proposal is not admissible: "
                + "; ".join((*self.safety_audit.issues, *issues))
            )
        self.trusted_result.audit.require_passed()
        return self.trusted_result


def prepare_mesh_proposal(
    source: CellMeshingResult,
    proposal: MeshProposal,
    policy: MeshProposalSafetyPolicy,
    /,
) -> MeshProposalTransaction:
    """Project, execute a native trusted path, audit and prepare atomic promotion.

    No caller-supplied mesh or audit is accepted as proposal evidence. Native
    refinement and optimization currently operate on affine, unassociated
    meshes: metadata requiring remapping is rejected rather than discarded.
    An empty projected marking is an explicit unchanged-source transaction.
    """
    started = time.monotonic()
    projection = project_mesh_proposal(source, proposal, policy)
    affine = CellGeometrySpec.affine(source.mesh)
    if (
        source.geometry.geometry_layout_id != affine.geometry_layout_id
        or not np.array_equal(source.geometry.coordinates, affine.coordinates)
    ):
        raise ValueError("Native proposal execution requires affine mesh geometry.")
    if source.boundary is not None or any(
        (
            source.patches,
            source.zones,
            source.labels,
            source.attributes,
            source.associations,
        )
    ):
        raise ValueError(
            "Native proposal execution cannot discard revision-bound mesh metadata."
        )
    transition, transfer, optimization = None, None, None
    candidate = source
    if isinstance(proposal, MeshCoordinateProposal):
        fixed, project = _coordinate_projector(source, proposal, policy)
        plan = TargetMatrixOptimizationPlan(
            source.mesh,
            target_coordinates=projection.target_coordinates,
            fixed_vertices=fixed,
            maximum_iterations=policy.maximum_optimization_iterations,
        )
        optimization = optimize_cell_mesh(
            plan,
            source.coordinate_contract,
            project=project,
            numeric_version=f"proposal:{projection.projection_id}",
        )
        candidate = optimization.result
    elif projection.marked_cell_ids.size:
        transition, transfer = refine_triangle_mesh(
            source.mesh,
            projection.marked_cell_ids,
            source.coordinate_contract,
            numeric_version=f"proposal:{projection.projection_id}",
        )
        candidate = transition.target
    quality = evaluate_cell_quality(candidate.mesh, candidate.geometry.coordinates)
    audit = audit_cell_mesh(
        candidate.mesh, candidate.geometry, quality, policy=policy.audit_policy
    )
    issues = _limit_issues(candidate, policy.limits) + _preservation_issues(
        source, candidate, projection
    )
    if not audit.passed:
        issues += ("safety_audit",)
    if (
        _payload_bytes(candidate) + _payload_bytes(transfer)
        > policy.limits.maximum_data_bytes
    ):
        issues += ("maximum_data_bytes",)
    elapsed = time.monotonic() - started
    if elapsed > policy.limits.maximum_wall_seconds:
        issues += ("maximum_wall_seconds",)
    compliance = MeshingComplianceReport(
        projection.projection_id,
        issues=tuple(dict.fromkeys(issues)),
        requested=(
            ("maximum_cells", policy.limits.maximum_cells),
            ("maximum_vertices", policy.limits.maximum_vertices),
        ),
        achieved=(
            ("cells", candidate.audit.entity_counts[-1]),
            ("vertices", candidate.audit.vertex_count),
        ),
    )
    return MeshProposalTransaction(
        source,
        projection,
        candidate,
        audit,
        compliance,
        transition,
        transfer,
        optimization,
    )


__all__ = [
    "MeshCoordinateProposal",
    "MeshMarkingProposal",
    "MeshMetricProposal",
    "MeshProposal",
    "MeshProposalProjection",
    "MeshProposalSafetyPolicy",
    "MeshProposalTransaction",
    "MeshSizeProposal",
    "mesh_proposal_scope",
    "prepare_mesh_proposal",
    "project_mesh_proposal",
]
