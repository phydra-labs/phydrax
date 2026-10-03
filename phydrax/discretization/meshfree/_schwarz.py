# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Overlapping meshfree patches for native subspace-correction preconditioning.

This module owns only patch geometry, stable ordering, overlap layers, and the
partition of unity. Local and coarse solves, the additive or multiplicative
composition, Krylov iteration, and refresh stay with ``phydrax.linalg``.
"""

from __future__ import annotations

import time
from numbers import Integral
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    adjoint,
    ArraySpace,
    assemble_sparse,
    BlockJacobiPreconditionerBuilder,
    EuclideanPairing,
    LinearCapabilityError,
    PreconditionerSource,
    SparseFactorizationPreconditionerBuilder,
    SubspaceCorrectionTerm,
)
from ...sparse import EdgeRelation, RowRelation, SparseCoordinateOperator
from ...typing import Dim, Float, Int32, Integer, parse
from ..spatial._morton import canonical_morton_order, MortonAddressPlan
from ._multilevel import PreparedMeshfreeHierarchy


MeshfreePartitionOfUnity: TypeAlias = Literal["ownership", "layer_decay"]
MeshfreeSchwarzProlongation: TypeAlias = Literal["adjoint", "partition_of_unity"]

# Stacked patch blocks are padded to a multiple of this many coordinates so
# clouds whose largest patches differ slightly share one compiled block shape.
_PATCH_BUCKET = 32


class _SchwarzPointDim(Dim):
    """Global meshfree points covered by the patch decomposition."""


class _SchwarzAxisDim(Dim):
    """Spatial coordinates used for Morton core ordering."""


class _SchwarzOffsetDim(Dim):
    """Patch offsets into the concatenated patch membership."""


class _SchwarzEntryDim(Dim):
    """Concatenated patch memberships, one per local patch coordinate."""


def _host_capacity(value: int, name: str, /, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a host integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


@final
class MeshfreeSchwarzPolicy(StrictModule, NonTrainableState):
    """Patch construction controls and hard host-setup capacities.

    Cores are contiguous runs of ``core_points`` in the stable-ID Morton order;
    the last runs absorb the remainder evenly. ``overlap_layers`` adds whole hops
    of the symmetrized adjacency graph. ``ownership`` weights each point fully
    to its core patch (restricted Schwarz); ``layer_decay`` weights a member
    ``(L + 1 - layer)`` before normalization. Exceeding ``maximum_patches``,
    ``maximum_patch_points`` or ``maximum_transfer_entries`` (summed patch
    memberships) refuses preparation; the overlap expansion workspace is at
    most ``maximum_transfer_entries`` times the largest adjacency degree.
    """

    core_points: int = eqx.field(static=True)
    overlap_layers: int = eqx.field(static=True)
    partition: MeshfreePartitionOfUnity = eqx.field(static=True)
    maximum_patches: int = eqx.field(static=True)
    maximum_patch_points: int = eqx.field(static=True)
    maximum_transfer_entries: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        core_points: int = 128,
        overlap_layers: int = 1,
        partition: MeshfreePartitionOfUnity = "ownership",
        maximum_patches: int = 65_536,
        maximum_patch_points: int = 4_096,
        maximum_transfer_entries: int = 16_000_000,
    ) -> None:
        core = _host_capacity(core_points, "core_points", minimum=1)
        layers = _host_capacity(overlap_layers, "overlap_layers", minimum=0)
        partition_ = parse(partition, MeshfreePartitionOfUnity, "partition")
        patches = _host_capacity(maximum_patches, "maximum_patches", minimum=1)
        patch_points = _host_capacity(
            maximum_patch_points, "maximum_patch_points", minimum=1
        )
        entries = _host_capacity(
            maximum_transfer_entries, "maximum_transfer_entries", minimum=1
        )
        self.core_points = core
        self.overlap_layers = layers
        self.partition = partition_
        self.maximum_patches = patches
        self.maximum_patch_points = patch_points
        self.maximum_transfer_entries = entries


@final
class MeshfreeSchwarzEvidence(StrictModule, NonTrainableState):
    """Observed decomposition, overlap, partition-of-unity and setup cost.

    ``partition_residual`` is ``max |Σ_i W_i R_i 1 - 1|`` over the retained
    membership weights that the weighted prolongations scatter.
    ``maximum_multiplicity`` is the largest number of patches sharing a point.
    ``transfer_bytes`` counts retained membership, layer, weight and offset
    arrays. ``setup_seconds`` is observed host wall time; it never enters an
    identity.
    """

    patch_sizes: tuple[int, ...] = eqx.field(static=True)
    core_sizes: tuple[int, ...] = eqx.field(static=True)
    overlap_layers: int = eqx.field(static=True)
    partition: MeshfreePartitionOfUnity = eqx.field(static=True)
    transfer_entries: int = eqx.field(static=True)
    overlap_entries: int = eqx.field(static=True)
    maximum_multiplicity: int = eqx.field(static=True)
    adjacency_edges: int = eqx.field(static=True)
    partition_residual: float = eqx.field(static=True)
    transfer_bytes: int = eqx.field(static=True)
    setup_seconds: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        patch_sizes: tuple[int, ...],
        core_sizes: tuple[int, ...],
        overlap_layers: int,
        partition: MeshfreePartitionOfUnity,
        maximum_multiplicity: int,
        adjacency_edges: int,
        partition_residual: float,
        transfer_bytes: int,
        setup_seconds: float,
    ) -> None:
        if not patch_sizes or len(patch_sizes) != len(core_sizes):
            raise ValueError("Evidence requires one patch and one core size per patch.")
        if any(
            core < 1 or patch < core
            for patch, core in zip(patch_sizes, core_sizes, strict=True)
        ):
            raise ValueError("Every patch must contain its nonempty core.")
        if not np.isfinite(partition_residual) or partition_residual < 0:
            raise ValueError("partition_residual must be finite and nonnegative.")
        self.patch_sizes = patch_sizes
        self.core_sizes = core_sizes
        self.overlap_layers = overlap_layers
        self.partition = parse(partition, MeshfreePartitionOfUnity, "partition")
        self.transfer_entries = sum(patch_sizes)
        self.overlap_entries = sum(patch_sizes) - sum(core_sizes)
        self.maximum_multiplicity = maximum_multiplicity
        self.adjacency_edges = adjacency_edges
        self.partition_residual = float(partition_residual)
        self.transfer_bytes = transfer_bytes
        self.setup_seconds = float(setup_seconds)


@final
class MeshfreeSchwarzPlan(StrictModule, NonTrainableState):
    """Stable-ID overlapping patch plan over one meshfree point graph.

    ``adjacency`` is a native point-to-point relation, normally the prepared
    stencil relation of the discretization whose operator is preconditioned.
    It is symmetrized for overlap: a point joins a patch if it is within
    ``overlap_layers`` hops of the core in either direction. Core assignment
    depends only on coordinates and stable IDs, never on storage order.
    """

    __strict_contract__ = True
    points: Float[_SchwarzPointDim, _SchwarzAxisDim]
    stable_ids: Integer[_SchwarzPointDim]
    adjacency: EdgeRelation
    policy: MeshfreeSchwarzPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        adjacency: EdgeRelation | RowRelation,
        /,
        *,
        stable_ids: ArrayLike | None = None,
        policy: MeshfreeSchwarzPolicy | None = None,
    ) -> None:
        points_ = jnp.asarray(points)
        if points_.ndim != 2 or points_.shape[0] < 1 or points_.shape[1] not in (1, 2, 3):
            raise ValueError(
                "points must have shape (point_count, spatial_dimension) with dimension 1, 2 or 3."
            )
        if not jnp.issubdtype(points_.dtype, jnp.floating):
            raise TypeError("points must have a real floating dtype.")
        count = points_.shape[0]
        ids = (
            jnp.arange(count, dtype=jnp.int32)
            if stable_ids is None
            else jnp.asarray(stable_ids)
        )
        if ids.shape != (count,) or not jnp.issubdtype(ids.dtype, jnp.integer):
            raise ValueError(
                "stable_ids must be an integer vector with one entry per point."
            )
        if isinstance(adjacency, RowRelation):
            relation = adjacency.as_edge_relation()
        elif isinstance(adjacency, EdgeRelation):
            relation = adjacency
        else:
            raise TypeError("adjacency must be an EdgeRelation or RowRelation.")
        if relation.source_size != count or relation.target_size != count:
            raise ValueError("adjacency must relate the plan's points to themselves.")
        policy_ = MeshfreeSchwarzPolicy() if policy is None else policy
        if not isinstance(policy_, MeshfreeSchwarzPolicy):
            raise TypeError("policy must be MeshfreeSchwarzPolicy or None.")
        host_points, host_ids = jax.device_get((points_, ids))
        if not np.all(np.isfinite(host_points)) or np.unique(host_ids).size != count:
            raise ValueError("Points must be finite and stable IDs must be unique.")
        self.points = points_
        self.stable_ids = ids
        self.adjacency = relation
        self.policy = policy_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "meshfree-schwarz-plan",
                "geometry": array_tree_fingerprint(
                    (
                        points_,
                        ids,
                        relation.source_indices,
                        relation.target_indices,
                        relation.valid,
                    )
                ),
                "core_points": policy_.core_points,
                "overlap_layers": policy_.overlap_layers,
                "partition": policy_.partition,
                "maximum_patches": policy_.maximum_patches,
                "maximum_patch_points": policy_.maximum_patch_points,
                "maximum_transfer_entries": policy_.maximum_transfer_entries,
            }
        )

    def prepare(self, space: ArraySpace, /) -> PreparedMeshfreeSchwarz:
        """Prepare restrictions/prolongations for coordinates in ``space``.

        ``space`` must be the Euclidean scalar point space of the operator to be
        preconditioned, so that each restriction is the transpose of its
        unweighted extension.
        """
        if not isinstance(space, ArraySpace) or space.shape != (self.points.shape[0],):
            raise ValueError(
                "space must be a scalar ArraySpace matching the point count."
            )
        if not isinstance(space.pairing, EuclideanPairing):
            raise ValueError(
                "Schwarz restrictions require Euclidean coordinates; a mass pairing would change the transpose."
            )
        if not np.issubdtype(space.dtype, np.floating):
            raise TypeError("Schwarz patches require a real floating coordinate space.")
        return _prepare_schwarz(self, space)


@final
class PreparedMeshfreeSchwarz(StrictModule, NonTrainableState):
    """Retained patches, partition weights and native transfer operators.

    Patch ``i`` occupies ``patch_points[patch_offsets[i]:patch_offsets[i + 1]]``
    in local order (global Morton rank). ``patch_layers`` is the graph distance
    from the core (zero inside it) and ``partition_weights`` the normalized
    partition-of-unity weight of each membership.
    """

    __strict_contract__ = True
    plan: MeshfreeSchwarzPlan
    space: ArraySpace
    patch_offsets: Int32[_SchwarzOffsetDim]
    patch_points: Int32[_SchwarzEntryDim]
    patch_layers: Int32[_SchwarzEntryDim]
    partition_weights: Float[_SchwarzEntryDim]
    local_spaces: tuple[ArraySpace, ...]
    extensions: tuple[SparseCoordinateOperator, ...]
    partition_prolongations: tuple[SparseCoordinateOperator, ...]
    evidence: MeshfreeSchwarzEvidence
    schwarz_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MeshfreeSchwarzPlan,
        space: ArraySpace,
        patch_offsets: ArrayLike,
        patch_points: ArrayLike,
        patch_layers: ArrayLike,
        partition_weights: ArrayLike,
        local_spaces: tuple[ArraySpace, ...],
        extensions: tuple[SparseCoordinateOperator, ...],
        partition_prolongations: tuple[SparseCoordinateOperator, ...],
        evidence: MeshfreeSchwarzEvidence,
        schwarz_id: str,
    ) -> None:
        if not isinstance(plan, MeshfreeSchwarzPlan) or not isinstance(
            evidence, MeshfreeSchwarzEvidence
        ):
            raise TypeError("Prepared Schwarz requires its plan and evidence.")
        count = len(evidence.patch_sizes)
        if not (
            len(local_spaces) == len(extensions) == len(partition_prolongations) == count
        ):
            raise ValueError("Every patch requires one local space and transfer pair.")
        for local, extension, weighted in zip(
            local_spaces, extensions, partition_prolongations, strict=True
        ):
            for operator in (extension, weighted):
                if not operator.source.compatible(
                    local
                ) or not operator.target.compatible(space):
                    raise ValueError(
                        "Patch transfers must map their local space into the global space."
                    )
        self.plan = plan
        self.space = space
        self.patch_offsets = jnp.asarray(patch_offsets, dtype=jnp.int32)
        self.patch_points = jnp.asarray(patch_points, dtype=jnp.int32)
        self.patch_layers = jnp.asarray(patch_layers, dtype=jnp.int32)
        self.partition_weights = jnp.asarray(partition_weights, dtype=space.dtype)
        self.local_spaces = local_spaces
        self.extensions = extensions
        self.partition_prolongations = partition_prolongations
        self.evidence = evidence
        self.schwarz_id = schwarz_id

    def terms(
        self,
        local_solver: PreconditionerSource | None = None,
        /,
        *,
        prolongation: MeshfreeSchwarzProlongation = "partition_of_unity",
    ) -> tuple[SubspaceCorrectionTerm, ...]:
        """Return one native correction term per patch, in Morton core order.

        Every local block is the unweighted Galerkin block ``R_i A R_iᵀ``.
        ``adjoint`` corrects with ``R_iᵀ`` (classical additive Schwarz,
        symmetric when local solves are). ``partition_of_unity`` corrects with
        ``R_iᵀ D_i`` (restricted Schwarz for ``ownership``), which is generally
        nonsymmetric and therefore needs GMRES/FGMRES. The default local solve
        is a complete native sparse factorization under its own capacities.
        """
        solver: PreconditionerSource = (
            SparseFactorizationPreconditionerBuilder()
            if local_solver is None
            else local_solver
        )
        prolongation_ = parse(prolongation, MeshfreeSchwarzProlongation, "prolongation")
        terms = []
        for extension, weighted in zip(
            self.extensions, self.partition_prolongations, strict=True
        ):
            restriction = adjoint(extension)
            match prolongation_:
                case "adjoint":
                    term = SubspaceCorrectionTerm(restriction, extension, solver)
                case "partition_of_unity":
                    term = SubspaceCorrectionTerm(
                        restriction, weighted, solver, local_extension=extension
                    )
                case _:
                    assert_never(prolongation_)
            terms.append(term)
        return tuple(terms)

    def block_term(
        self,
        /,
        *,
        prolongation: MeshfreeSchwarzProlongation = "partition_of_unity",
    ) -> SubspaceCorrectionTerm:
        """Return one additive term solving every patch in one batched factorization.

        Patches are stacked into ``P`` homogeneous blocks of ``S`` coordinates,
        ``S`` the largest patch size rounded up to a multiple of
        ``_PATCH_BUCKET``; coordinates beyond a patch's size are declared
        padding. The diagonal blocks of the stacked Galerkin operator are
        exactly ``R_i A R_iᵀ`` (padding rows and columns are structurally zero
        and factored as identity), so this term equals the additive
        composition of :meth:`terms` with exact local solves, while all local
        factors share one compiled batched native local-block factorization
        instead of one sparse factorization per patch. Multiplicative sweeps
        need the ordered per-patch :meth:`terms`.
        """
        prolongation_ = parse(prolongation, MeshfreeSchwarzProlongation, "prolongation")
        offsets = np.asarray(self.patch_offsets)
        points = np.asarray(self.patch_points)
        sizes = np.diff(offsets)
        block = -(-int(np.max(sizes)) // _PATCH_BUCKET) * _PATCH_BUCKET
        total = sizes.size * block
        stacked = np.repeat(np.arange(sizes.size), sizes) * block + (
            np.arange(points.size) - np.repeat(offsets[:-1], sizes)
        )
        padding = np.ones(total, dtype=np.bool_)
        padding[stacked] = False
        local = ArraySpace(
            (total,),
            dtype=self.space.dtype,
            space_id=f"{self.schwarz_id}:stacked:{block}",
        )
        relation = EdgeRelation(
            stacked.astype(np.int32),
            points.astype(np.int32),
            source_size=total,
            target_size=self.space.shape[0],
            valid=np.ones(points.size, dtype=np.bool_),
        )
        extension = SparseCoordinateOperator(
            relation,
            jnp.ones(points.size, dtype=self.space.dtype),
            source=local,
            target=self.space,
            operator_id=f"{self.schwarz_id}:stacked-extension:{block}",
        )
        solver = BlockJacobiPreconditionerBuilder(block, padding=padding)
        match prolongation_:
            case "adjoint":
                return SubspaceCorrectionTerm(adjoint(extension), extension, solver)
            case "partition_of_unity":
                weighted = SparseCoordinateOperator(
                    relation,
                    self.partition_weights,
                    source=local,
                    target=self.space,
                    operator_id=f"{self.schwarz_id}:stacked-partition-prolongation:{block}",
                )
                return SubspaceCorrectionTerm(
                    adjoint(extension), weighted, solver, local_extension=extension
                )
            case _:
                assert_never(prolongation_)


def meshfree_coarse_correction_term(
    hierarchy: PreparedMeshfreeHierarchy,
    /,
    *,
    level: int = 1,
    solver: PreconditionerSource | None = None,
) -> SubspaceCorrectionTerm:
    """Two-level coarse correction through a prepared meshfree hierarchy.

    The fine-to-``level`` prolongation is the exact native sparse product of
    the hierarchy prolongations; restriction is its Euclidean transpose, which
    is the hierarchy's stiffness-coordinate restriction. The coarse Galerkin
    block ``Pᵀ A P`` is solved by ``solver`` (default: complete native sparse
    factorization). Deep levels of a boundary-retaining hierarchy can consist
    almost entirely of retained boundary rows and then carry little interior
    correction; the level is therefore an explicit choice, not "coarsest".
    """
    if not isinstance(hierarchy, PreparedMeshfreeHierarchy):
        raise TypeError("hierarchy must be PreparedMeshfreeHierarchy.")
    depth = len(hierarchy.spaces) - 1
    if depth < 1:
        raise ValueError("A coarse correction requires a hierarchy with a coarse level.")
    if isinstance(level, bool) or not isinstance(level, Integral):
        raise TypeError("level must be a host integer.")
    selected = int(level)
    if selected < 1 or selected > depth:
        raise ValueError(f"level must lie in [1, {depth}].")
    if not all(
        isinstance(space.pairing, EuclideanPairing)
        for space in hierarchy.spaces[: selected + 1]
    ):
        raise ValueError(
            "Coarse restriction as a transpose requires Euclidean level coordinates."
        )
    prolongation: AbstractLinearOperator = hierarchy.transfers[0][1]
    for transition in range(1, selected):
        prolongation = prolongation @ hierarchy.transfers[transition][1]
    if selected > 1:
        prolongation = assemble_sparse(prolongation)
    coarse_solver: PreconditionerSource = (
        SparseFactorizationPreconditionerBuilder() if solver is None else solver
    )
    return SubspaceCorrectionTerm(adjoint(prolongation), prolongation, coarse_solver)


def _morton_order(points: np.ndarray, stable_ids: np.ndarray, /) -> np.ndarray:
    lower = points.min(axis=0)
    extent = points.max(axis=0) - lower
    # Strictly enlarge the box so the maximal point is inside the half-open cell
    # range; flat axes receive a unit extent.
    span = np.where(extent > 0.0, extent, 1.0) * (1.0 + 2.0**-20)
    dimension = points.shape[1]
    address = MortonAddressPlan(
        tuple(lower.tolist()),
        tuple((lower + span).tolist()),
        63 // dimension,
    )
    encoding = address.encode(jnp.asarray(points))
    order = canonical_morton_order(
        encoding.codes, jnp.asarray(stable_ids), encoding.in_domain
    )
    host_order, in_domain = jax.device_get((order, encoding.in_domain))
    if not np.all(in_domain):
        raise ValueError(
            "Morton patch ordering requires every point in its bounding box."
        )
    return np.asarray(host_order, dtype=np.int64)


def _symmetric_graph(relation: EdgeRelation, count: int, /) -> sp.csr_matrix:
    targets, sources, valid = jax.device_get(
        (relation.target_indices, relation.source_indices, relation.valid)
    )
    keep = np.asarray(valid, dtype=bool) & (np.asarray(targets) != np.asarray(sources))
    rows = np.asarray(targets, dtype=np.int64)[keep]
    columns = np.asarray(sources, dtype=np.int64)[keep]
    ones = np.ones(rows.size, dtype=np.int32)
    graph = sp.csr_matrix((ones, (rows, columns)), shape=(count, count))
    graph = (graph + graph.T).tocsr()
    graph.data[:] = 1
    graph.sort_indices()
    return graph


def _overlap_layers(
    cores: list[np.ndarray],
    graph: sp.csr_matrix,
    policy: MeshfreeSchwarzPolicy,
    /,
) -> sp.csr_matrix:
    """Return patch-by-point memberships storing ``layer + 1``."""
    count = graph.shape[0]
    rows = np.repeat(np.arange(len(cores), dtype=np.int64), [core.size for core in cores])
    columns = np.concatenate(cores)
    reached = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.int32), (rows, columns)),
        shape=(len(cores), count),
    )
    labels = reached.copy()
    for layer in range(1, policy.overlap_layers + 1):
        expanded = (reached @ graph).tocsr()
        expanded.data[:] = 1
        new = (expanded - expanded.multiply(reached)).tocsr()
        new.eliminate_zeros()
        if not new.nnz:
            break
        if reached.nnz + new.nnz > policy.maximum_transfer_entries:
            raise LinearCapabilityError(
                f"Schwarz overlap layer {layer} needs {reached.nnz + new.nnz} patch "
                f"memberships, exceeding maximum_transfer_entries="
                f"{policy.maximum_transfer_entries}."
            )
        labels = (labels + new * (layer + 1)).tocsr()
        reached = (reached + new).tocsr()
    labels.sort_indices()
    return labels


def _partition_weights(
    patch_points: np.ndarray,
    patch_layers: np.ndarray,
    count: int,
    policy: MeshfreeSchwarzPolicy,
    dtype: np.dtype,
    /,
) -> np.ndarray:
    match policy.partition:
        case "ownership":
            return (patch_layers == 0).astype(dtype)
        case "layer_decay":
            raw = (policy.overlap_layers + 1 - patch_layers).astype(dtype) / dtype.type(
                policy.overlap_layers + 1
            )
            total = np.zeros(count, dtype=dtype)
            np.add.at(total, patch_points, raw)
            return raw / total[patch_points]
        case _:
            assert_never(policy.partition)


def _patch_operators(
    space: ArraySpace,
    schwarz_id: str,
    offsets: np.ndarray,
    patch_points: np.ndarray,
    weights: np.ndarray,
    /,
) -> tuple[
    tuple[ArraySpace, ...],
    tuple[SparseCoordinateOperator, ...],
    tuple[SparseCoordinateOperator, ...],
]:
    count = space.shape[0]
    local_spaces = []
    extensions = []
    weighted = []
    for patch in range(offsets.size - 1):
        start, stop = int(offsets[patch]), int(offsets[patch + 1])
        size = stop - start
        local = ArraySpace(
            (size,), dtype=space.dtype, space_id=f"{schwarz_id}:patch:{patch}"
        )
        relation = EdgeRelation(
            np.arange(size, dtype=np.int32),
            patch_points[start:stop].astype(np.int32),
            source_size=size,
            target_size=count,
            valid=np.ones(size, dtype=np.bool_),
        )
        extensions.append(
            SparseCoordinateOperator(
                relation,
                np.ones(size, dtype=space.dtype),
                source=local,
                target=space,
                operator_id=f"{schwarz_id}:extension:{patch}",
            )
        )
        weighted.append(
            SparseCoordinateOperator(
                relation,
                weights[start:stop],
                source=local,
                target=space,
                operator_id=f"{schwarz_id}:partition-prolongation:{patch}",
            )
        )
        local_spaces.append(local)
    return tuple(local_spaces), tuple(extensions), tuple(weighted)


def _partition_residual(
    patch_points: np.ndarray, weights: np.ndarray, count: int, /
) -> float:
    # Σ_i W_i R_i 1 at each point is the sum of that point's membership weights,
    # exactly the coefficients the weighted prolongations scatter.
    total = np.zeros(count, dtype=weights.dtype)
    np.add.at(total, patch_points, weights)
    return float(np.max(np.abs(total - 1.0)))


def _patch_cores(order: np.ndarray, policy: MeshfreeSchwarzPolicy, /) -> list[np.ndarray]:
    patch_count = -(-order.size // policy.core_points)
    if patch_count > policy.maximum_patches:
        raise LinearCapabilityError(
            f"Schwarz decomposition needs {patch_count} patches, exceeding "
            f"maximum_patches={policy.maximum_patches}."
        )
    return [
        np.asarray(core, dtype=np.int64) for core in np.array_split(order, patch_count)
    ]


def _prepare_schwarz(
    plan: MeshfreeSchwarzPlan, space: ArraySpace, /
) -> PreparedMeshfreeSchwarz:
    started = time.perf_counter()
    policy = plan.policy
    points, stable_ids = jax.device_get((plan.points, plan.stable_ids))
    points = np.asarray(points)
    count = points.shape[0]
    order = _morton_order(points, np.asarray(stable_ids))
    rank = np.empty(count, dtype=np.int64)
    rank[order] = np.arange(count, dtype=np.int64)
    cores = _patch_cores(order, policy)
    graph = _symmetric_graph(plan.adjacency, count)
    labels = _overlap_layers(cores, graph, policy)
    offsets = np.zeros(len(cores) + 1, dtype=np.int64)
    members = []
    layers = []
    for patch in range(len(cores)):
        start, stop = labels.indptr[patch], labels.indptr[patch + 1]
        if stop - start > policy.maximum_patch_points:
            raise LinearCapabilityError(
                f"Schwarz patch {patch} has {stop - start} points, exceeding "
                f"maximum_patch_points={policy.maximum_patch_points}."
            )
        columns = labels.indices[start:stop].astype(np.int64)
        # Local coordinates follow the global Morton rank: deterministic under
        # input permutation and spatially local for natural-order factors.
        local_order = np.argsort(rank[columns], kind="stable")
        members.append(columns[local_order])
        layers.append(labels.data[start:stop][local_order].astype(np.int64) - 1)
        offsets[patch + 1] = offsets[patch] + (stop - start)
    patch_points = np.concatenate(members)
    patch_layers = np.concatenate(layers)
    weights = _partition_weights(patch_points, patch_layers, count, policy, space.dtype)
    schwarz_id = canonical_fingerprint(
        {
            "kind": "prepared-meshfree-schwarz",
            "plan": plan.plan_id,
            "space": space.space_id,
        }
    )
    local_spaces, extensions, weighted = _patch_operators(
        space, schwarz_id, offsets, patch_points, weights
    )
    multiplicity = np.bincount(patch_points, minlength=count)
    evidence = MeshfreeSchwarzEvidence(
        patch_sizes=tuple(int(value) for value in np.diff(offsets)),
        core_sizes=tuple(core.size for core in cores),
        overlap_layers=policy.overlap_layers,
        partition=policy.partition,
        maximum_multiplicity=int(multiplicity.max()),
        adjacency_edges=graph.nnz // 2,
        partition_residual=_partition_residual(patch_points, weights, count),
        transfer_bytes=offsets.size * np.dtype(np.int32).itemsize
        + patch_points.size * (2 * np.dtype(np.int32).itemsize + space.dtype.itemsize),
        setup_seconds=time.perf_counter() - started,
    )
    return PreparedMeshfreeSchwarz(
        plan,
        space,
        offsets,
        patch_points,
        patch_layers,
        weights,
        local_spaces,
        extensions,
        weighted,
        evidence,
        schwarz_id,
    )


__all__ = [
    "meshfree_coarse_correction_term",
    "MeshfreePartitionOfUnity",
    "MeshfreeSchwarzEvidence",
    "MeshfreeSchwarzPlan",
    "MeshfreeSchwarzPolicy",
    "MeshfreeSchwarzProlongation",
    "PreparedMeshfreeSchwarz",
]
