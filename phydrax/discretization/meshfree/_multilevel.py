#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    AbstractPreconditionerBuilder,
    ArraySpace,
    EuclideanPairing,
    GalerkinHierarchyBuilder,
    GaussSeidelPreconditionerBuilder,
    LinearCapabilityError,
    LinearSubspace,
    NullspacePolicy,
    ProjectedPseudoinversePreconditionerBuilder,
    SparseFactorizationPreconditionerBuilder,
)
from ...sparse import RowRelation, SparseCoordinateOperator
from ._neighbors import MeshfreeNeighborhoodPlan
from ._stencils import LocalStencilPolicy, MeshfreeFunctional, prepare_local_stencils


@final
class MeshfreeCoarseningPolicy(StrictModule, NonTrainableState):
    """Finite host-setup capacities and deterministic subset stopping controls.

    ``coarsening_neighbors`` bounds each greedy exclusion neighborhood, including
    its center. No global distance matrix is formed. Boundaries are retained,
    but a level with no reduction terminates instead of repeating indefinitely.
    ``maximum_transfer_entries`` bounds summed prolongation route capacity;
    restriction reuses these numeric coefficients on the transposed relation.
    ``maximum_candidates`` is a per-target search capacity, capped by the source
    count at each level, and ``chunk_rows`` bounds query and stencil setup rows.
    """

    maximum_levels: int = eqx.field(static=True)
    minimum_coarse_points: int = eqx.field(static=True)
    coarsening_neighbors: int = eqx.field(static=True)
    interpolation_neighbors: int = eqx.field(static=True)
    maximum_points: int = eqx.field(static=True)
    maximum_transfer_entries: int = eqx.field(static=True)
    maximum_candidates: int = eqx.field(static=True)
    chunk_rows: int = eqx.field(static=True)
    reproduction_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_levels: int = 8,
        minimum_coarse_points: int = 16,
        coarsening_neighbors: int = 4,
        interpolation_neighbors: int = 12,
        maximum_points: int = 1_000_000,
        maximum_transfer_entries: int = 16_000_000,
        maximum_candidates: int = 4096,
        chunk_rows: int = 256,
        reproduction_tolerance: float = 1e-8,
    ) -> None:
        incoming = (
            maximum_levels,
            minimum_coarse_points,
            coarsening_neighbors,
            interpolation_neighbors,
            maximum_points,
            maximum_transfer_entries,
            maximum_candidates,
            chunk_rows,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in incoming
        ):
            raise TypeError("Hierarchy capacities must be host integers.")
        values = tuple(int(value) for value in incoming)
        if any(value < 1 for value in values) or values[2] < 2:
            raise ValueError(
                "Hierarchy capacities must be positive; exclusion width must exceed one."
            )
        tolerance = float(reproduction_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("reproduction_tolerance must be finite and positive.")
        (
            self.maximum_levels,
            self.minimum_coarse_points,
            self.coarsening_neighbors,
            self.interpolation_neighbors,
            self.maximum_points,
            self.maximum_transfer_entries,
            self.maximum_candidates,
            self.chunk_rows,
        ) = values
        self.reproduction_tolerance = tolerance


@final
class MeshfreeHierarchyEvidence(StrictModule, NonTrainableState):
    """Per-transition affine reproduction and summed prolongation route capacity.

    ``transfer_entries`` does not double-count the same numeric coefficients
    reused by stiffness-transpose restriction. Resident object-graph byte
    accounting must include both relations and their index arrays.
    """

    level_sizes: tuple[int, ...] = eqx.field(static=True)
    constant_residuals: tuple[float, ...] = eqx.field(static=True)
    linear_residuals: tuple[float, ...] = eqx.field(static=True)
    transfer_entries: int = eqx.field(static=True)
    stopping_reason: str = eqx.field(static=True)
    restriction_coordinates: str = eqx.field(static=True)

    def __init__(
        self,
        level_sizes: tuple[int, ...],
        constant_residuals: tuple[float, ...],
        linear_residuals: tuple[float, ...],
        transfer_entries: int,
        stopping_reason: str,
    ) -> None:
        if not level_sizes or any(size < 1 for size in level_sizes):
            raise ValueError("Evidence requires positive level sizes.")
        if any(
            coarse >= fine
            for fine, coarse in zip(level_sizes[:-1], level_sizes[1:], strict=True)
        ):
            raise ValueError(
                "Every hierarchy transition must genuinely reduce the point count."
            )
        if len(constant_residuals) != len(level_sizes) - 1 or len(
            linear_residuals
        ) != len(constant_residuals):
            raise ValueError("Reproduction evidence must have one entry per transition.")
        if any(
            not np.isfinite(error) or error < 0
            for error in (*constant_residuals, *linear_residuals)
        ):
            raise ValueError("Reproduction residuals must be finite and nonnegative.")
        if transfer_entries < 0 or not stopping_reason:
            raise ValueError(
                "Transfer storage must be nonnegative and stopping_reason nonempty."
            )
        self.level_sizes = level_sizes
        self.constant_residuals = constant_residuals
        self.linear_residuals = linear_residuals
        self.transfer_entries = transfer_entries
        self.stopping_reason = stopping_reason
        self.restriction_coordinates = "stiffness-coordinate-transpose"


@final
class MeshfreeHierarchyPlan(StrictModule, NonTrainableState):
    """Host-prepared stable-ID subset hierarchy for scalar stiffness coordinates.

    Fine-space coordinates must be Euclidean. For a mass-paired strong-form
    operator, first express its equation in stiffness coordinates; silently
    substituting a Hilbert mass adjoint would represent a different equation.

    Retained coarse nodes use exact nodal inclusion, preserving arbitrary
    coarse values even when neighboring retained boundary nodes are collinear.
    Nonretained target nodes use strictly accepted degree-one PHS reconstruction.
    """

    points: Array
    boundary: Array
    stable_ids: Array
    policy: MeshfreeCoarseningPolicy

    def __init__(
        self,
        points: ArrayLike,
        /,
        *,
        boundary: ArrayLike | None = None,
        stable_ids: ArrayLike | None = None,
        policy: MeshfreeCoarseningPolicy | None = None,
    ) -> None:
        points_ = jnp.asarray(points)
        policy_ = MeshfreeCoarseningPolicy() if policy is None else policy
        if not isinstance(policy_, MeshfreeCoarseningPolicy):
            raise TypeError("policy must be MeshfreeCoarseningPolicy.")
        if points_.ndim != 2 or points_.shape[0] < 1 or points_.shape[1] < 1:
            raise ValueError("points must have shape (point_count, spatial_dimension).")
        if not jnp.issubdtype(points_.dtype, jnp.floating):
            raise TypeError("points must have a real floating dtype.")
        count = points_.shape[0]
        if count > policy_.maximum_points:
            raise LinearCapabilityError("Meshfree hierarchy exceeds maximum_points.")
        boundary_ = (
            jnp.zeros(count, dtype=jnp.bool_)
            if boundary is None
            else jnp.asarray(boundary)
        )
        ids = (
            jnp.arange(count, dtype=jnp.int32)
            if stable_ids is None
            else jnp.asarray(stable_ids)
        )
        if boundary_.shape != (count,) or boundary_.dtype != jnp.bool_:
            raise ValueError(
                "boundary must be a Boolean vector with one entry per point."
            )
        if ids.shape != (count,) or not jnp.issubdtype(ids.dtype, jnp.integer):
            raise ValueError(
                "stable_ids must be an integer vector with one entry per point."
            )
        host_points, host_ids = jax.device_get((points_, ids))
        if not np.all(np.isfinite(host_points)) or np.unique(host_ids).size != count:
            raise ValueError("Points must be finite and stable IDs must be unique.")
        self.points = points_
        self.boundary = boundary_
        self.stable_ids = ids
        self.policy = policy_

    def prepare(self, fine_space: ArraySpace, /) -> PreparedMeshfreeHierarchy:
        if not isinstance(fine_space, ArraySpace) or fine_space.shape != (
            self.points.shape[0],
        ):
            raise ValueError(
                "fine_space must be a scalar ArraySpace matching the point count."
            )
        if not isinstance(fine_space.pairing, EuclideanPairing):
            raise ValueError(
                "Meshfree Galerkin transfers require Euclidean stiffness coordinates, not a mass pairing."
            )
        if fine_space.dtype != np.dtype(self.points.dtype):
            raise TypeError("fine_space dtype must match points.")
        return _prepare_hierarchy(self, fine_space)


@final
class PreparedMeshfreeHierarchy(StrictModule, NonTrainableState):
    plan: MeshfreeHierarchyPlan
    level_points: tuple[Array, ...]
    level_ids: tuple[Array, ...]
    spaces: tuple[ArraySpace, ...]
    transfers: tuple[tuple[AbstractLinearOperator, AbstractLinearOperator], ...]
    evidence: MeshfreeHierarchyEvidence
    hierarchy_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MeshfreeHierarchyPlan,
        level_points: tuple[Array, ...],
        level_ids: tuple[Array, ...],
        spaces: tuple[ArraySpace, ...],
        transfers: tuple[tuple[AbstractLinearOperator, AbstractLinearOperator], ...],
        evidence: MeshfreeHierarchyEvidence,
    ) -> None:
        if not isinstance(plan, MeshfreeHierarchyPlan) or not isinstance(
            evidence, MeshfreeHierarchyEvidence
        ):
            raise TypeError(
                "Prepared hierarchy requires a meshfree plan and hierarchy evidence."
            )
        if (
            not level_points
            or len(level_points) != len(level_ids)
            or len(spaces) != len(level_points)
        ):
            raise ValueError(
                "Every level requires geometry, stable IDs and a typed space."
            )
        if len(transfers) != len(level_points) - 1:
            raise ValueError("Hierarchy requires one transfer pair per transition.")
        if tuple(points.shape[0] for points in level_points) != evidence.level_sizes:
            raise ValueError(
                "Hierarchy geometry must match observed evidence capacities."
            )
        for points, ids, space in zip(level_points, level_ids, spaces, strict=True):
            if (
                points.ndim != 2
                or points.shape[1] != plan.points.shape[1]
                or ids.shape != (points.shape[0],)
            ):
                raise ValueError(
                    "Each level must retain the spatial dimension and one stable ID per point."
                )
            if not isinstance(space, ArraySpace) or space.shape != (points.shape[0],):
                raise ValueError(
                    "Each level space must match its scalar geometry coordinates."
                )
        for level, (restriction, prolongation) in enumerate(transfers):
            if not isinstance(restriction, AbstractLinearOperator) or not isinstance(
                prolongation, AbstractLinearOperator
            ):
                raise TypeError(
                    "Hierarchy transfers must be native typed linear operators."
                )
            if not (
                prolongation.source.compatible(spaces[level + 1])
                and prolongation.target.compatible(spaces[level])
                and restriction.source.compatible(spaces[level])
                and restriction.target.compatible(spaces[level + 1])
            ):
                raise ValueError(
                    "Hierarchy transfer source/target identities do not match level spaces."
                )
        identifier = canonical_fingerprint(
            {
                "kind": "meshfree-hierarchy",
                "fine_space": spaces[0].space_id,
                "geometry": array_tree_fingerprint(
                    (plan.points, plan.boundary, plan.stable_ids)
                ),
                "transfers": tuple((r.operator_id, p.operator_id) for r, p in transfers),
            }
        )
        self.plan = plan
        self.level_points = level_points
        self.level_ids = level_ids
        self.spaces = spaces
        self.transfers = transfers
        self.evidence = evidence
        self.hierarchy_id = identifier


def _subset_indices(
    points: Array, boundary: Array, stable_ids: Array, policy: MeshfreeCoarseningPolicy
) -> np.ndarray:
    # Canonicalize before neighborhood selection so geometric ties do not depend
    # on the caller's point ordering. Only bounded k-neighbor routes are stored.
    order = np.argsort(np.asarray(jax.device_get(stable_ids)), kind="stable")
    cloud = points[jnp.asarray(order)]
    neighborhood = MeshfreeNeighborhoodPlan(
        cloud,
        min(policy.coarsening_neighbors, cloud.shape[0]),
        maximum_candidates=min(policy.maximum_candidates, cloud.shape[0]),
        target_chunk_size=policy.chunk_rows,
    ).prepare()
    routes, valid, keep = jax.device_get(
        (
            neighborhood.relation.source_indices,
            neighborhood.relation.valid,
            boundary[jnp.asarray(order)],
        )
    )
    selected = np.array(keep, dtype=np.bool_, copy=True)
    excluded = np.zeros(order.size, dtype=np.bool_)
    for row in np.flatnonzero(selected):
        excluded[routes[row][valid[row]]] = True
    for row in range(order.size):
        if not excluded[row] and not selected[row]:
            selected[row] = True
            excluded[routes[row][valid[row]]] = True
    return order[np.flatnonzero(selected)]


def _prepare_hierarchy(
    plan: MeshfreeHierarchyPlan, fine_space: ArraySpace
) -> PreparedMeshfreeHierarchy:
    policy = plan.policy
    points, ids, boundaries = [plan.points], [plan.stable_ids], [plan.boundary]
    spaces = [fine_space]
    transfers: list[tuple[AbstractLinearOperator, AbstractLinearOperator]] = []
    constant_errors: list[float] = []
    linear_errors: list[float] = []
    entries = 0
    reason = "maximum-levels"
    dimension = plan.points.shape[1]
    if policy.interpolation_neighbors < dimension + 1:
        raise ValueError("interpolation_neighbors cannot reproduce affine polynomials.")
    functional = MeshfreeFunctional(((0,) * dimension,), (1.0,), name="interpolation")
    stencil_policy = LocalStencilPolicy(
        approximation="phs-rbf-fd",
        polynomial_degree=1,
        phs_power=3,
        acceptance="refuse",
        chunk_rows=policy.chunk_rows,
    )
    for level in range(policy.maximum_levels - 1):
        fine = points[-1]
        if fine.shape[0] <= max(policy.minimum_coarse_points, dimension + 1):
            reason = "minimum-coarse-points"
            break
        subset = _subset_indices(fine, boundaries[-1], ids[-1], policy)
        if subset.size >= fine.shape[0]:
            reason = "no-progress-boundary-retention"
            break
        if subset.size < dimension + 1:
            reason = "affine-space-minimum"
            break
        coarse = fine[jnp.asarray(subset)]
        coarse_ids = ids[-1][jnp.asarray(subset)]
        width = min(policy.interpolation_neighbors, subset.size)
        next_entries = fine.shape[0] * width
        if entries + next_entries > policy.maximum_transfer_entries:
            raise LinearCapabilityError(
                "Meshfree hierarchy exceeds maximum_transfer_entries."
            )
        # Retained subset nodes have an exact inclusion map, regardless of
        # whether their nearest coarse neighbors span the ambient polynomial
        # space. Only genuinely cross-target rows require a PHS reconstruction.
        cross_target = np.ones(fine.shape[0], dtype=np.bool_)
        cross_target[subset] = False
        target_indices = np.flatnonzero(cross_target)
        targets = fine[jnp.asarray(target_indices)]
        neighborhood = MeshfreeNeighborhoodPlan(
            coarse,
            width,
            targets=targets,
            maximum_candidates=min(policy.maximum_candidates, subset.size),
            target_chunk_size=policy.chunk_rows,
        ).prepare()
        stencils = prepare_local_stencils(
            neighborhood, coarse, targets, (functional,), stencil_policy
        )
        indices = np.zeros((fine.shape[0], width), dtype=np.int32)
        valid = np.zeros((fine.shape[0], width), dtype=np.bool_)
        coefficients = np.zeros((fine.shape[0], width), dtype=np.dtype(fine.dtype))
        indices[target_indices] = np.asarray(
            jax.device_get(neighborhood.relation.source_indices)
        )
        valid[target_indices] = np.asarray(jax.device_get(neighborhood.relation.valid))
        coefficients[target_indices] = np.asarray(jax.device_get(stencils.weights[0]))
        indices[subset, 0] = np.arange(subset.size, dtype=np.int32)
        valid[subset, 0] = True
        coefficients[subset, 0] = 1.0
        relation = RowRelation(indices, source_size=subset.size, valid=valid)
        weights = jnp.asarray(coefficients)
        coarse_space = ArraySpace(
            (subset.size,),
            dtype=fine_space.dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "meshfree-level-space",
                    "fine": fine_space.space_id,
                    "level": level + 1,
                    "ids": array_tree_fingerprint(coarse_ids),
                    "geometry": array_tree_fingerprint(coarse),
                }
            ),
        )
        prolongation = SparseCoordinateOperator(
            relation, weights, source=coarse_space, target=spaces[-1]
        )
        edge = relation.as_edge_relation()
        restriction = SparseCoordinateOperator(
            edge.transpose(),
            weights.reshape((-1,)),
            source=spaces[-1],
            target=coarse_space,
        )
        # Observe affine reproduction on the actual prepared sparse action. Host
        # synchronization is confined to this explicit immutable setup boundary.
        constant = np.asarray(
            jax.device_get(prolongation.mv(jnp.ones(subset.size, dtype=fine.dtype)))
        )
        reproduced = jnp.stack(
            tuple(prolongation.mv(coarse[:, axis]) for axis in range(dimension)), axis=1
        )
        linear = np.asarray(jax.device_get(reproduced - fine))
        constant_error = float(np.max(np.abs(constant - 1)))
        scale = max(1.0, float(np.max(np.abs(np.asarray(jax.device_get(fine))))))
        linear_error = float(np.max(np.abs(linear))) / scale
        if max(constant_error, linear_error) > policy.reproduction_tolerance:
            raise LinearCapabilityError(
                "Meshfree transfer failed its declared affine reproduction tolerance."
            )
        constant_errors.append(constant_error)
        linear_errors.append(linear_error)
        transfers.append((restriction, prolongation))
        points.append(coarse)
        ids.append(coarse_ids)
        boundaries.append(boundaries[-1][jnp.asarray(subset)])
        spaces.append(coarse_space)
        entries += next_entries
    evidence = MeshfreeHierarchyEvidence(
        tuple(point.shape[0] for point in points),
        tuple(constant_errors),
        tuple(linear_errors),
        entries,
        reason,
    )
    return PreparedMeshfreeHierarchy(
        plan, tuple(points), tuple(ids), tuple(spaces), tuple(transfers), evidence
    )


def meshfree_multigrid_builder(
    hierarchy: PreparedMeshfreeHierarchy,
    /,
    *,
    smoothers: tuple[AbstractPreconditioner | AbstractPreconditionerBuilder, ...]
    | None = None,
    coarse_solver: AbstractPreconditioner | AbstractPreconditionerBuilder | None = None,
    nullspace_policy: NullspacePolicy | None = None,
    pre_smoothing: int = 1,
    post_smoothing: int = 1,
) -> AbstractPreconditionerBuilder:
    """Use native Galerkin setup and cycles, reusing immutable numeric transfers.

    Native materialization/assembly policies still bound coarse setup. A cloud
    too small (or boundary-only) for a genuine transition uses the coarse source
    directly; no fictitious duplicate level or alternate V-cycle is introduced.

    Forward sparse Gauss--Seidel is the nonsymmetric default. Damped Jacobi
    requires spectral/diagonal-dominance assumptions that collocated meshfree
    operators do not generally satisfy. Neither this smoother nor this cycle
    is claimed to contract every cloud; the outer native Krylov solve checks
    the original fine-system residual. Callers may supply SPD-specific smoothers.
    """
    if not isinstance(hierarchy, PreparedMeshfreeHierarchy):
        raise TypeError("hierarchy must be PreparedMeshfreeHierarchy.")
    coarse = (
        SparseFactorizationPreconditionerBuilder()
        if coarse_solver is None
        else coarse_solver
    )
    if nullspace_policy is not None:
        if not isinstance(nullspace_policy, NullspacePolicy):
            raise TypeError("nullspace_policy must be native NullspacePolicy.")
        if coarse_solver is not None:
            raise ValueError(
                "Choose either the declared constant-nullspace coarse solve or a supplied coarse_solver."
            )
        if not hierarchy.transfers:
            raise LinearCapabilityError(
                "A projected coarse solve requires genuine coarsening; dense fine fallback is forbidden."
            )
        fine_space = hierarchy.spaces[0]
        for kernel in (nullspace_policy.right, nullspace_policy.left):
            if (
                kernel is None
                or not kernel.space.compatible(fine_space)
                or kernel.batch_shape
            ):
                raise ValueError(
                    "Constant kernels must be declared in the fine scalar space."
                )
            ones = jnp.ones(fine_space.size, dtype=fine_space.dtype)
            dimension, mismatch = jax.device_get(
                (
                    kernel.dimension,
                    jnp.max(jnp.abs(kernel.project_coordinates(ones) - ones)),
                )
            )
            if (
                int(dimension) != 1
                or float(mismatch) > hierarchy.plan.policy.reproduction_tolerance
            ):
                raise ValueError(
                    "Geometric meshfree nullspace transfer supports only declared constant kernels."
                )
        coarse_space = hierarchy.spaces[-1]
        constant_kernel = LinearSubspace(
            coarse_space,
            jnp.ones((coarse_space.size, 1), dtype=coarse_space.dtype),
        )
        coarse_policy = NullspacePolicy(
            right=constant_kernel,
            left=constant_kernel,
            compatibility=nullspace_policy.compatibility,
            gauge=nullspace_policy.gauge,
        )
        coarse = ProjectedPseudoinversePreconditionerBuilder(
            coarse_policy,
            tolerance=hierarchy.plan.policy.reproduction_tolerance,
        )
    if not hierarchy.transfers:
        if not isinstance(coarse, AbstractPreconditionerBuilder):
            raise TypeError("A one-level hierarchy requires a coarse-solver builder.")
        return coarse
    smoothing = (
        tuple(
            GaussSeidelPreconditionerBuilder(direction="forward")
            for _ in hierarchy.transfers
        )
        if smoothers is None
        else tuple(smoothers)
    )
    return GalerkinHierarchyBuilder(
        hierarchy.transfers,
        smoothing,
        coarse,
        refresh_mode="reuse-symbolic-sparse-products",
        pre_smoothing=pre_smoothing,
        post_smoothing=post_smoothing,
    )


__all__ = [
    "MeshfreeCoarseningPolicy",
    "MeshfreeHierarchyEvidence",
    "MeshfreeHierarchyPlan",
    "PreparedMeshfreeHierarchy",
    "meshfree_multigrid_builder",
]
