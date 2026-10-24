#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rank audits of the scaled rigid joint Jacobian.

The audited operator is the unweighted Jacobian of the characteristic-length
scaled joint residuals with respect to the mobile translation/rotation
coordinates (the constraint operator of the velocity projection), not the
mass-weighted Schur operator. A step requires full numerical row rank under the
strict relative rule ``sigma > rank_tolerance * sigma_max``.

``general-dense-svd`` (the default) densifies that operator and consumes the
native exact SVD rank evidence. ``certified-forest`` is opt-in for acyclic joint
graphs with at most one fixed body per connected component: it extracts the
endpoint blocks from the same operator and certifies full row rank with the
verified right-inverse bound of `phydrax.linalg._forest_bounds`. A failed
certificate is inconclusive, never a rank-deficiency claim; it either refuses the
step or, when admitted before tracing, runs the dense SVD as an explicit fallback.
"""

from __future__ import annotations

from collections import deque
from enum import IntEnum
from typing import assert_never, final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractLinearOperator, RankPolicy, svd as svd_linalg
from ...linalg._forest_bounds import (
    BlockForestRankCertificate,
    BlockForestTopology,
    certify_block_forest_rank,
    prepare_block_forest,
    PreparedBlockForest,
    refresh_block_forest,
)
from ...typing import Bool, checked, Float, Int32, parse, Scalar
from ._rigid_joints import _RigidMobileIncrement, PreparedRigidJointGraph


RigidConstraintRankAuditKind: TypeAlias = Literal[
    "general-dense-svd", "certified-forest"
]


class RigidConstraintRankRoute(IntEnum):
    """Executed rank route of one evaluation."""

    NONE = 0
    DENSE_SVD = 1
    FOREST_CERTIFICATE = 2
    FOREST_DENSE_FALLBACK = 3


class RigidConstraintRankStatus(IntEnum):
    """Rank outcome; only dense measurements may report deficiency."""

    FULL_ROW_RANK_CERTIFIED = 0
    FULL_ROW_RANK_MEASURED = 1
    RANK_DEFICIENT_MEASURED = 2
    DENSE_SVD_FAILED = 3
    INCONCLUSIVE = 4
    EMPTY = 5


_DEFAULT_FOREST_WORKSPACE = 512 * 1024 * 1024


@final
class RigidConstraintRankAudit(StrictModule):
    """Rank route of the rigid joint Jacobian.

    ``dense_fallback`` (``certified-forest`` only) admits the dense SVD after an
    inconclusive certificate under the given SVD resource budget; preparation
    plans it and refuses when the budget cannot hold it. Without a fallback an
    inconclusive certificate rejects the step. ``workspace_bytes`` bounds the
    logical certificate workspace (``certified-forest`` only).
    """

    kind: RigidConstraintRankAuditKind = eqx.field(static=True)
    dense_fallback: svd_linalg.SVDResourcePolicy | None
    workspace_bytes: int | None = eqx.field(static=True)
    audit_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: RigidConstraintRankAuditKind = "general-dense-svd",
        /,
        *,
        dense_fallback: svd_linalg.SVDResourcePolicy | None = None,
        workspace_bytes: int | None = None,
    ) -> None:
        kind_ = parse(kind, RigidConstraintRankAuditKind, "kind")
        if dense_fallback is not None and not isinstance(
            dense_fallback, svd_linalg.SVDResourcePolicy
        ):
            raise TypeError("dense_fallback must be an SVDResourcePolicy or None.")
        match kind_:
            case "general-dense-svd":
                if dense_fallback is not None or workspace_bytes is not None:
                    raise ValueError(
                        "dense_fallback and workspace_bytes apply only to the "
                        "certified-forest rank audit."
                    )
                workspace = None
            case "certified-forest":
                workspace = (
                    _DEFAULT_FOREST_WORKSPACE
                    if workspace_bytes is None
                    else workspace_bytes
                )
                if (
                    isinstance(workspace, bool)
                    or not isinstance(workspace, int)
                    or workspace < 0
                ):
                    raise ValueError("workspace_bytes must be a nonnegative integer.")
            case unknown:
                assert_never(unknown)
        self.kind = kind_
        self.dense_fallback = dense_fallback
        self.workspace_bytes = workspace
        self.audit_id = canonical_fingerprint(
            {
                "kind": "rigid-constraint-rank-audit",
                "route": kind_,
                "workspace_bytes": workspace,
                "dense_fallback": None
                if dense_fallback is None
                else {
                    "preparation_bytes": dense_fallback.preparation_bytes,
                    "workspace_bytes": dense_fallback.workspace_bytes,
                    "operator_matvecs": dense_fallback.operator_matvecs,
                },
            }
        )


@final
class RigidConstraintRankEvidence(StrictModule):
    """Route-aware rank evidence of one evaluated joint Jacobian.

    ``rank_lower_bound``/``rank_upper_bound`` coincide for a dense measurement or
    a certificate; an inconclusive certificate or a failed dense SVD (unavailable
    or inexact rank evidence, e.g. a nonfinite Jacobian) keeps ``0`` and
    ``row_count``.
    ``measured_condition`` is the dense ``sigma_max / sigma_min`` estimate of an
    executed SVD (``1`` for an empty graph, NaN when no SVD ran). ``certificate``
    holds the forest bounds, including the condition upper bound, separation and
    numeric binding; it is ``None`` on the general route.
    """

    __strict_contract__ = True
    status: Int32[Scalar]
    executed_route: Int32[Scalar]
    full_row_rank: Bool[Scalar]
    rank_lower_bound: Int32[Scalar]
    rank_upper_bound: Int32[Scalar]
    measured_condition: Float[Scalar]
    certificate: BlockForestRankCertificate | None
    audit: RigidConstraintRankAuditKind = eqx.field(static=True)
    row_count: int = eqx.field(static=True)


def empty_rigid_constraint_rank_evidence(
    audit: RigidConstraintRankAudit, dtype: DTypeLike, /
) -> RigidConstraintRankEvidence:
    """The zero-row convention: rank zero, condition one, nothing executed."""
    zero = jnp.asarray(0, dtype=jnp.int32)
    return RigidConstraintRankEvidence(
        jnp.asarray(int(RigidConstraintRankStatus.EMPTY), dtype=jnp.int32),
        jnp.asarray(int(RigidConstraintRankRoute.NONE), dtype=jnp.int32),
        jnp.asarray(True),
        zero,
        zero,
        jnp.asarray(1, dtype=dtype),
        None,
        audit.kind,
        0,
    )


class _DenseRank(NamedTuple):
    """Dense SVD outcome; ``rank`` is meaningful only when ``exact`` holds."""

    exact: Array
    successful: Array
    rank: Array
    condition: Array


def _dense_rank(
    constraint: AbstractLinearOperator,
    policy: svd_linalg.SVDSolvePolicy,
    problem_id: str,
    dtype: DTypeLike,
    /,
) -> _DenseRank:
    """The general audit: native dense SVD rank, condition estimate and status.

    The exact rank is consumed only under the predicate that
    `require_exact_svd_rank` enforces. Unavailable or inexact evidence (a
    nonfinite operator or failed decomposition) becomes a failed audit instead of
    a runtime error, so the step can reject transactionally and batched
    evaluations never raise from an unselected branch.
    """
    result = svd_linalg.svd(
        svd_linalg.SVDProblem(constraint, problem_id=problem_id), policy=policy
    )
    evidence = result.rank_evidence
    exact = (
        evidence.available
        & (evidence.lower_bound == evidence.upper_bound)
        & evidence.deterministic_exact
        & evidence.full_spectrum
    )
    values = result.singular_values
    smallest = jnp.min(values)
    condition = jnp.max(values) / jnp.maximum(smallest, jnp.finfo(values.dtype).tiny)
    return _DenseRank(
        jnp.asarray(exact, dtype=jnp.bool_),
        result.successful,
        jnp.where(exact, evidence.lower_bound, 0).astype(jnp.int32),
        condition.astype(dtype),
    )


def _dense_outcome(dense: _DenseRank, rows: int, /) -> tuple[Array, Array, Array]:
    """``(status, rank_lower_bound, rank_upper_bound)`` of one dense audit.

    Failed audits keep the uninformative bounds ``[0, rows]``.
    """
    full = (
        dense.exact
        & dense.successful
        & (dense.rank == rows)
        & jnp.isfinite(dense.condition)
    )
    deficient = dense.exact & (dense.rank < rows)
    status = jnp.where(
        full,
        int(RigidConstraintRankStatus.FULL_ROW_RANK_MEASURED),
        jnp.where(
            deficient,
            int(RigidConstraintRankStatus.RANK_DEFICIENT_MEASURED),
            int(RigidConstraintRankStatus.DENSE_SVD_FAILED),
        ),
    ).astype(jnp.int32)
    measured = full | deficient
    return (
        status,
        jnp.where(measured, dense.rank, 0).astype(jnp.int32),
        jnp.where(measured, dense.rank, rows).astype(jnp.int32),
    )


class _ForestOrientation(NamedTuple):
    """Host orientation of an admitted forest; vertices are mobile slots."""

    parent_vertices: np.ndarray
    row_counts: np.ndarray
    vertex_joints: np.ndarray
    child_is_left: np.ndarray
    body_colors: np.ndarray


def _breadth_first(
    start: int, adjacency: list[list[tuple[int, int]]], /
) -> dict[int, tuple[int, int, int]]:
    """``body -> (distance, parent body, joint slot)`` over one tree component."""
    visited = {start: (0, -1, -1)}
    queue = deque((start,))
    while queue:
        body = queue.popleft()
        distance = visited[body][0]
        for neighbor, slot in adjacency[body]:
            if neighbor not in visited:
                visited[neighbor] = (distance + 1, body, slot)
                queue.append(neighbor)
    return visited


def _component_root(
    component: list[int],
    adjacency: list[list[tuple[int, int]]],
    fixed: np.ndarray,
    ids: np.ndarray,
    /,
) -> int:
    """The unique fixed body, else the tree center with the smallest stable ID.

    Rooting a free tree at its center minimizes the recurrence depth. The center
    is a property of the tree, so the choice is independent of joint order.
    """
    anchors = [body for body in component if fixed[body]]
    if len(anchors) > 1:
        raise ValueError(
            "Certified-forest rank audit admits at most one fixed body per connected "
            "joint component."
        )
    if anchors:
        return anchors[0]
    start = min(component, key=lambda body: ids[body])

    def farthest(distances: dict[int, tuple[int, int, int]]) -> int:
        return min(distances, key=lambda body: (-distances[body][0], ids[body]))

    first = farthest(_breadth_first(start, adjacency))
    from_first = _breadth_first(first, adjacency)
    from_second = _breadth_first(farthest(from_first), adjacency)
    return min(
        component,
        key=lambda body: (
            max(from_first[body][0], from_second[body][0]),
            ids[body],
        ),
    )


def _union_root(parents: np.ndarray, body: int, /) -> int:
    while parents[body] != body:
        parents[body] = parents[parents[body]]
        body = int(parents[body])
    return body


def _forest_orientation(joints: PreparedRigidJointGraph, /) -> _ForestOrientation:
    """Admit and orient the joint graph as a rooted forest of mobile vertices."""
    bodies = joints.bodies
    capacity = bodies.capacity
    ids = np.asarray(bodies.particles.particle_ids)
    fixed = np.asarray(bodies.fixed_mask, dtype=np.bool_)
    mobile = np.asarray(joints.mobile_indices).astype(np.int64)
    vertex_of = np.full(capacity, -1, dtype=np.int64)
    vertex_of[mobile] = np.arange(mobile.size)
    left, right = (np.asarray(value).astype(np.int64) for value in joints.layout_endpoints())
    layout = joints.row_layout
    joint_ids = np.asarray(layout.joint_ids)
    rows_per_joint = np.bincount(
        np.asarray(layout.row_joint_slots), minlength=layout.joint_count
    )
    union = np.arange(capacity)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(capacity)]
    for slot in np.argsort(joint_ids, kind="stable").tolist():
        first, second = int(left[slot]), int(right[slot])
        first_root, second_root = _union_root(union, first), _union_root(union, second)
        if first_root == second_root:
            raise ValueError(
                "Certified-forest rank audit requires an acyclic joint graph; joint "
                f"{int(joint_ids[slot])} closes a cycle or duplicates an edge."
            )
        union[first_root] = second_root
        adjacency[first].append((second, slot))
        adjacency[second].append((first, slot))

    parent_vertices = np.full(mobile.size, -1, dtype=np.int64)
    row_counts = np.zeros(mobile.size, dtype=np.int64)
    vertex_joints = np.full(mobile.size, -1, dtype=np.int64)
    child_is_left = np.zeros(mobile.size, dtype=np.bool_)
    colors = np.full(capacity, -1, dtype=np.int32)
    members: dict[int, list[int]] = {}
    for body in sorted(
        {int(value) for value in np.concatenate((left, right, mobile))},
        key=lambda value: ids[value],
    ):
        members.setdefault(_union_root(union, body), []).append(body)
    for component in members.values():
        root = _component_root(component, adjacency, fixed, ids)
        for body, (distance, parent, slot) in _breadth_first(root, adjacency).items():
            if vertex_of[body] >= 0:
                colors[body] = distance % 2
            if parent < 0:
                continue
            vertex = int(vertex_of[body])
            parent_vertices[vertex] = vertex_of[parent]
            row_counts[vertex] = rows_per_joint[slot]
            vertex_joints[vertex] = slot
            child_is_left[vertex] = left[slot] == body
    return _ForestOrientation(
        parent_vertices, row_counts, vertex_joints, child_is_left, colors
    )


def _vertex_blocks(
    left: Array,
    right: Array,
    vertex_joints: Array,
    child_is_left: Array,
    parent_is_vertex: Array,
    /,
) -> tuple[Array, Array]:
    """Child ``A_w`` and parent ``B_w`` blocks of each vertex's incoming joint."""
    has_edge = (vertex_joints >= 0)[:, None, None]
    slot = jnp.maximum(vertex_joints, 0)
    orient = child_is_left[:, None, None]
    child = jnp.where(orient, left[slot], right[slot])
    parent = jnp.where(orient, right[slot], left[slot])
    return (
        jnp.where(has_edge, child, 0),
        jnp.where(has_edge & parent_is_vertex[:, None, None], parent, 0),
    )


@final
class _RigidForestRank(StrictModule, NonTrainableState):
    """Admitted forest route: block template, coloring and vertex/joint maps."""

    template: PreparedBlockForest
    body_colors: Array
    vertex_joints: Array
    child_is_left: Array
    parent_is_vertex: Array

    def vertex_blocks(self, left: Array, right: Array, /) -> tuple[Array, Array]:
        return _vertex_blocks(
            left, right, self.vertex_joints, self.child_is_left, self.parent_is_vertex
        )


def _qualify_block_action(
    joints: PreparedRigidJointGraph,
    constraint: AbstractLinearOperator,
    forest: _RigidForestRank,
    orientation: _ForestOrientation,
    /,
) -> None:
    """Admission: assembled blocks reproduce the audited operator on a probe.

    Exact by construction (each joint row reads only its endpoints); the probe
    rejects a coloring, orientation or packing defect before any tracing.
    """
    child = np.asarray(forest.template.child_blocks)
    parent = np.asarray(forest.template.parent_blocks)
    count, _, width = child.shape
    dimension = joints.bodies.ambient_dimension
    probe = np.cos(0.7 * np.arange(count * width, dtype=child.dtype) + 0.3).reshape(
        count, width
    )
    increment = _RigidMobileIncrement(
        jnp.asarray(probe[:, :dimension]), jnp.asarray(probe[:, dimension:])
    )
    direct = np.asarray(joints.pack_residuals(constraint.mv(increment)))
    parents = orientation.parent_vertices
    parent_probe = np.where((parents >= 0)[:, None], probe[np.maximum(parents, 0)], 0)
    block_rows = contract("nmc,nc->nm", child, probe) + contract(
        "nmc,nc->nm", parent, parent_probe
    )
    scale = contract("nmc,nc->nm", np.abs(child), np.abs(probe)) + contract(
        "nmc,nc->nm", np.abs(parent), np.abs(parent_probe)
    )
    layout = joints.row_layout
    row_index = np.full((layout.joint_count, width), -1, dtype=np.int64)
    row_index[np.asarray(layout.row_joint_slots), np.asarray(layout.row_local_indices)] = (
        np.arange(layout.row_count)
    )
    assembled = np.zeros_like(direct)
    magnitude = np.zeros_like(direct)
    for vertex in np.flatnonzero(orientation.row_counts > 0).tolist():
        rows = int(orientation.row_counts[vertex])
        target = row_index[int(orientation.vertex_joints[vertex]), :rows]
        assembled[target] = block_rows[vertex, :rows]
        magnitude[target] = scale[vertex, :rows]
    info = np.finfo(child.dtype)
    tolerance = 64 * width * info.eps * (magnitude + np.abs(direct)) + 64 * info.tiny
    if not np.all(np.abs(direct - assembled) <= tolerance):
        raise ValueError(
            "Certified-forest blocks do not reproduce the audited joint Jacobian "
            "action at the reference configuration."
        )


@final
class PreparedRigidConstraintRank(StrictModule, NonTrainableState):
    """Prepared rank route bound to one joint graph and rank tolerance.

    ``dense_policy`` is the dense SVD policy of the general route or of the
    admitted forest fallback; it is ``None`` for an empty graph or a forest route
    without fallback.
    """

    audit: RigidConstraintRankAudit
    dense_policy: svd_linalg.SVDSolvePolicy | None
    forest: _RigidForestRank | None
    row_count: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def evidence(
        self,
        joints: PreparedRigidJointGraph,
        constraint: AbstractLinearOperator,
        dtype: DTypeLike,
        /,
    ) -> RigidConstraintRankEvidence:
        """Rank evidence of the audited operator at its current numeric state."""
        if self.row_count == 0:
            return empty_rigid_constraint_rank_evidence(self.audit, dtype)
        match self.audit.kind:
            case "general-dense-svd":
                return self._general(constraint, dtype)
            case "certified-forest":
                return self._certified(joints, constraint, dtype)
            case unknown:
                assert_never(unknown)

    def _require_dense_policy(self) -> svd_linalg.SVDSolvePolicy:
        if self.dense_policy is None:
            raise ValueError("The prepared rank route admits no dense SVD.")
        return self.dense_policy

    def _general(
        self, constraint: AbstractLinearOperator, dtype: DTypeLike, /
    ) -> RigidConstraintRankEvidence:
        dense = _dense_rank(
            constraint, self._require_dense_policy(), self.problem_id, dtype
        )
        status, lower, upper = _dense_outcome(dense, self.row_count)
        return RigidConstraintRankEvidence(
            status,
            jnp.asarray(int(RigidConstraintRankRoute.DENSE_SVD), dtype=jnp.int32),
            status == int(RigidConstraintRankStatus.FULL_ROW_RANK_MEASURED),
            lower,
            upper,
            dense.condition,
            None,
            self.audit.kind,
            self.row_count,
        )

    def _certified(
        self,
        joints: PreparedRigidJointGraph,
        constraint: AbstractLinearOperator,
        dtype: DTypeLike,
        /,
    ) -> RigidConstraintRankEvidence:
        if self.forest is None:
            raise ValueError("The prepared rank route has no admitted forest.")
        forest = self.forest
        left, right = joints.endpoint_jacobian_blocks(
            constraint.mv, forest.body_colors, dtype
        )
        child, parent = forest.vertex_blocks(left, right)
        bound = refresh_block_forest(forest.template, child, parent)
        certificate = certify_block_forest_rank(
            bound, relative_cutoff=self.rank_tolerance
        )
        rows = self.row_count
        certified = certificate.certified
        if self.dense_policy is None:
            return RigidConstraintRankEvidence(
                jnp.where(
                    certified,
                    int(RigidConstraintRankStatus.FULL_ROW_RANK_CERTIFIED),
                    int(RigidConstraintRankStatus.INCONCLUSIVE),
                ).astype(jnp.int32),
                jnp.asarray(
                    int(RigidConstraintRankRoute.FOREST_CERTIFICATE), dtype=jnp.int32
                ),
                certified,
                certificate.rank_lower_bound,
                jnp.asarray(rows, dtype=jnp.int32),
                jnp.asarray(jnp.nan, dtype=dtype),
                certificate,
                self.audit.kind,
                rows,
            )
        policy = self.dense_policy

        def measured(_: None) -> _DenseRank:
            return _dense_rank(constraint, policy, self.problem_id, dtype)

        def skipped(_: None) -> _DenseRank:
            return _DenseRank(
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(jnp.nan, dtype=dtype),
            )

        # The fallback was planned within its admitted budget before tracing. A
        # batched (vmapped) evaluation executes both branches; neither raises.
        dense = jax.lax.cond(certified, skipped, measured, None)
        fallback, fallback_lower, fallback_upper = _dense_outcome(dense, rows)
        status = jnp.where(
            certified, int(RigidConstraintRankStatus.FULL_ROW_RANK_CERTIFIED), fallback
        ).astype(jnp.int32)
        return RigidConstraintRankEvidence(
            status,
            jnp.where(
                certified,
                int(RigidConstraintRankRoute.FOREST_CERTIFICATE),
                int(RigidConstraintRankRoute.FOREST_DENSE_FALLBACK),
            ).astype(jnp.int32),
            certified
            | (fallback == int(RigidConstraintRankStatus.FULL_ROW_RANK_MEASURED)),
            jnp.where(certified, rows, fallback_lower).astype(jnp.int32),
            jnp.where(certified, rows, fallback_upper).astype(jnp.int32),
            jnp.where(certified, jnp.nan, dense.condition).astype(dtype),
            certificate,
            self.audit.kind,
            rows,
        )


def _dense_policy(
    rows: int,
    rank_tolerance: float,
    resources: svd_linalg.SVDResourcePolicy | None,
    /,
) -> svd_linalg.SVDSolvePolicy:
    return svd_linalg.SVDSolvePolicy(
        count=rows,
        rank=RankPolicy(relative_cutoff=rank_tolerance, require_full_rank=True),
        resources=resources,
    )


def _prepare_forest(
    audit: RigidConstraintRankAudit,
    joints: PreparedRigidJointGraph,
    constraint: AbstractLinearOperator,
    dtype: DTypeLike,
    /,
) -> _RigidForestRank:
    orientation = _forest_orientation(joints)
    width = joints.bodies.ambient_dimension + joints.bodies.angular_dimension
    topology = BlockForestTopology(
        orientation.parent_vertices, orientation.row_counts, column_width=width
    )
    body_colors = jnp.asarray(orientation.body_colors)
    vertex_joints = jnp.asarray(orientation.vertex_joints.astype(np.int32))
    child_is_left = jnp.asarray(orientation.child_is_left)
    parent_is_vertex = jnp.asarray(orientation.parent_vertices >= 0)
    left, right = joints.endpoint_jacobian_blocks(constraint.mv, body_colors, dtype)
    child, parent = _vertex_blocks(
        left, right, vertex_joints, child_is_left, parent_is_vertex
    )
    workspace = audit.workspace_bytes
    if workspace is None:
        raise ValueError("A certified-forest audit declares a workspace budget.")
    forest = _RigidForestRank(
        prepare_block_forest(topology, child, parent, workspace_bytes=workspace),
        body_colors,
        vertex_joints,
        child_is_left,
        parent_is_vertex,
    )
    _qualify_block_action(joints, constraint, forest, orientation)
    return forest


@checked
def prepare_rigid_constraint_rank(
    audit: RigidConstraintRankAudit,
    joints: PreparedRigidJointGraph,
    constraint: AbstractLinearOperator | None,
    /,
    *,
    rank_tolerance: float,
    problem_id: str,
    dtype: DTypeLike,
) -> PreparedRigidConstraintRank:
    """Admit the rank route on host, using ``constraint`` at the reference state.

    ``constraint`` is ``None`` exactly for a graph without rows. ``dtype`` is the
    increment dtype of ``constraint``. Forest admission, block qualification and
    fallback planning all happen here; an unsupported topology or an
    unaffordable fallback raises before tracing.
    """
    rows = joints.constraint_count
    if (constraint is None) != (rows == 0):
        raise ValueError("A rank audit needs the constraint operator of a nonempty graph.")
    dense_policy = None
    forest = None
    if constraint is not None:
        match audit.kind:
            case "general-dense-svd":
                dense_policy = _dense_policy(rows, rank_tolerance, None)
            case "certified-forest":
                forest = _prepare_forest(audit, joints, constraint, dtype)
                if audit.dense_fallback is not None:
                    dense_policy = _dense_policy(
                        rows, rank_tolerance, audit.dense_fallback
                    )
                    svd_linalg.plan_svd(
                        svd_linalg.SVDProblem(constraint, problem_id=problem_id),
                        dense_policy,
                    )
            case unknown:
                assert_never(unknown)
    return PreparedRigidConstraintRank(
        audit, dense_policy, forest, rows, rank_tolerance, problem_id
    )


__all__ = [
    "empty_rigid_constraint_rank_evidence",
    "prepare_rigid_constraint_rank",
    "PreparedRigidConstraintRank",
    "RigidConstraintRankAudit",
    "RigidConstraintRankAuditKind",
    "RigidConstraintRankEvidence",
    "RigidConstraintRankRoute",
    "RigidConstraintRankStatus",
]
