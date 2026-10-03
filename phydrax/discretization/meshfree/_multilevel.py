#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import comb
from numbers import Integral
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._total_degree import TotalDegreePolynomialFeatures
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
    ILUPreconditionerBuilder,
    LinearCapabilityError,
    LinearSubspace,
    MultigridCycleKind,
    MultigridCyclePolicy,
    MultigridRefreshMode,
    NullspacePolicy,
    ProjectedPseudoinversePreconditionerBuilder,
    RankPolicy,
    SparseAssemblyPolicy,
    SparseFactorizationPreconditionerBuilder,
)
from ...sparse import (
    EdgeRelation,
    gather_routes,
    route_reduce,
    RowRelation,
    SparseCoordinateOperator,
)
from ...typing import parse
from ._capacity import bucketed_storage_capacity
from ._neighbors import MeshfreeNeighborhoodPlan
from ._stencils import LocalStencilPolicy, MeshfreeFunctional, prepare_local_stencils


MeshfreeComponentLayout: TypeAlias = Literal["nodal", "block"]
MeshfreeNearNullspaceKind: TypeAlias = Literal["none", "constant", "rigid-body", "affine"]
MeshfreeHierarchyStop: TypeAlias = Literal[
    "maximum-levels",
    "minimum-coarse-points",
    "no-progress-retention",
    "coarsening-round-limit",
    "unisolvence-minimum",
    "transfer-row-refusal",
    "insufficient-reduction",
    "empty-coarse-coordinate",
]

_UNDECIDED, _SELECTED, _EXCLUDED = 0, 1, 2


@final
class MeshfreeCoarseningPolicy(StrictModule, NonTrainableState):
    """Finite setup capacities, transfer degree, and deterministic stopping controls.

    ``coarsening_neighbors`` is the k-nearest-neighbor width, including the
    center, of the symmetric conflict graph on which the coarse independent
    set is selected. No global distance matrix is formed. Declared ``boundary``
    nodes are forced coarse on the ``boundary_retention_levels`` finest
    transitions (``None``: every transition, ``0``: never); ``features`` are
    forced coarse on every transition. A level whose coarse/fine point ratio
    exceeds ``maximum_coarse_fraction`` stops with ``"insufficient-reduction"``
    (``"no-progress-retention"`` when nothing is removed) instead of adding a
    level that only raises operator complexity. The independent set runs at
    most ``maximum_coarsening_rounds`` device rounds; an undecided node after
    that bound stops coarsening with ``"coarsening-round-limit"``.

    Nonretained fine nodes are reconstructed by polyharmonic-spline RBF-FD
    interpolation with total-degree ``reproduction_degree`` polynomial
    augmentation (PHS power ``2 * degree + 1``); every monomial up to that
    degree must be reproduced within ``reproduction_tolerance`` in coordinates
    normalized to the level bounding box. ``maximum_transfer_entries`` bounds
    summed prolongation route capacity over all components; restriction reuses
    these numeric coefficients on the transposed relation.
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
    maximum_coarsening_rounds: int = eqx.field(static=True)
    reproduction_degree: int = eqx.field(static=True)
    reproduction_tolerance: float = eqx.field(static=True)
    boundary_retention_levels: int | None = eqx.field(static=True)
    maximum_coarse_fraction: float = eqx.field(static=True)

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
        maximum_coarsening_rounds: int = 64,
        reproduction_degree: int = 1,
        reproduction_tolerance: float = 1e-8,
        boundary_retention_levels: int | None = 0,
        maximum_coarse_fraction: float = 0.8,
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
            maximum_coarsening_rounds,
            reproduction_degree,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in incoming
        ):
            raise TypeError("Hierarchy capacities must be host integers.")
        values = tuple(int(value) for value in incoming)
        if any(value < 1 for value in values) or values[2] < 2:
            raise ValueError(
                "Hierarchy capacities and reproduction_degree must be positive; "
                "the conflict-graph width must exceed one."
            )
        tolerance = float(reproduction_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("reproduction_tolerance must be finite and positive.")
        if boundary_retention_levels is not None and (
            isinstance(boundary_retention_levels, bool)
            or not isinstance(boundary_retention_levels, Integral)
        ):
            raise TypeError("boundary_retention_levels must be a host integer or None.")
        if boundary_retention_levels is not None and boundary_retention_levels < 0:
            raise ValueError("boundary_retention_levels must be nonnegative.")
        fraction = float(maximum_coarse_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("maximum_coarse_fraction must lie strictly between 0 and 1.")
        (
            self.maximum_levels,
            self.minimum_coarse_points,
            self.coarsening_neighbors,
            self.interpolation_neighbors,
            self.maximum_points,
            self.maximum_transfer_entries,
            self.maximum_candidates,
            self.chunk_rows,
            self.maximum_coarsening_rounds,
            self.reproduction_degree,
        ) = values
        self.reproduction_tolerance = tolerance
        self.boundary_retention_levels = (
            None if boundary_retention_levels is None else int(boundary_retention_levels)
        )
        self.maximum_coarse_fraction = fraction


@final
class MeshfreeComponentSpace(StrictModule, NonTrainableState):
    """Scalar or coupled per-point component coordinates of a meshfree system.

    ``"nodal"`` coordinates are point-major (``point * components + component``);
    ``"block"`` coordinates are component-major (``component * points + point``).
    Every component shares the scalar point transfer: ``P ⊗ I`` for nodal and
    ``I ⊗ P`` for block layouts, so an affine-reproducing transfer reproduces
    every rigid-body mode of a coupled vector field.
    """

    components: int = eqx.field(static=True)
    layout: MeshfreeComponentLayout = eqx.field(static=True)

    def __init__(
        self, components: int = 1, /, *, layout: MeshfreeComponentLayout = "nodal"
    ) -> None:
        if isinstance(components, bool) or not isinstance(components, Integral):
            raise TypeError("components must be a host integer.")
        if components < 1:
            raise ValueError("components must be positive.")
        self.components = int(components)
        self.layout = parse(layout, MeshfreeComponentLayout, "layout")


@final
class MeshfreeNearNullspace(StrictModule, NonTrainableState):
    """Declared near-nullspace candidates whose transfer defects are reported.

    ``"constant"`` declares one constant field per component, ``"rigid-body"``
    the translations and infinitesimal rotations of a 2-D/3-D vector field
    (components must equal the spatial dimension), and ``"affine"`` every
    component times each affine coordinate. ``vectors`` appends supplied fine
    coordinate candidates, for example indicators of disconnected components.
    Candidates are transferred by exact nodal injection at retained nodes.
    """

    kind: MeshfreeNearNullspaceKind = eqx.field(static=True)
    vectors: Array | None

    def __init__(
        self,
        kind: MeshfreeNearNullspaceKind = "constant",
        /,
        *,
        vectors: ArrayLike | None = None,
    ) -> None:
        kind_ = parse(kind, MeshfreeNearNullspaceKind, "kind")
        supplied = None if vectors is None else jnp.asarray(vectors)
        if supplied is not None:
            if supplied.ndim != 2 or supplied.shape[1] < 1:
                raise ValueError(
                    "vectors must have shape (coordinate_count, candidate_count)."
                )
            if not jnp.issubdtype(supplied.dtype, jnp.floating):
                raise TypeError("vectors must have a real floating dtype.")
            if not np.all(np.isfinite(np.asarray(jax.device_get(supplied)))):
                raise ValueError("vectors must be finite.")
        self.kind = kind_
        self.vectors = supplied


@final
class MeshfreeHierarchyEvidence(StrictModule, NonTrainableState):
    """Per-transition reproduction, near-nullspace defects, rounds and capacity.

    ``reproduction_residuals[level][q]`` is the largest normalized defect of the
    scalar point transfer on monomials of exact total degree ``q``.
    ``near_nullspace_defects[level][j]`` is ``|P b_coarse - b_fine|_inf /
    |b_fine|_inf`` for declared candidate ``j`` on the component transfer.
    ``transfer_entries`` does not double-count the numeric coefficients reused
    by stiffness-transpose restriction. ``grid_complexity`` is the summed level
    coordinate count divided by the fine count; Galerkin operator complexity is
    reported by the native multigrid setup diagnostics.
    """

    level_sizes: tuple[int, ...] = eqx.field(static=True)
    components: int = eqx.field(static=True)
    reproduction_degree: int = eqx.field(static=True)
    reproduction_residuals: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    near_nullspace_defects: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    coarsening_rounds: tuple[int, ...] = eqx.field(static=True)
    transfer_entries: int = eqx.field(static=True)
    stopping_reason: MeshfreeHierarchyStop = eqx.field(static=True)
    grid_complexity: float = eqx.field(static=True)
    restriction_coordinates: str = eqx.field(static=True)

    def __init__(
        self,
        level_sizes: tuple[int, ...],
        *,
        components: int,
        reproduction_degree: int,
        reproduction_residuals: tuple[tuple[float, ...], ...],
        near_nullspace_defects: tuple[tuple[float, ...], ...],
        coarsening_rounds: tuple[int, ...],
        transfer_entries: int,
        stopping_reason: MeshfreeHierarchyStop,
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
        transitions = len(level_sizes) - 1
        if (
            len(reproduction_residuals) != transitions
            or len(near_nullspace_defects) != transitions
            or len(coarsening_rounds) != transitions
        ):
            raise ValueError("Transfer evidence must have one entry per transition.")
        if any(len(level) != reproduction_degree + 1 for level in reproduction_residuals):
            raise ValueError("Reproduction evidence requires one entry per degree.")
        if any(
            not np.isfinite(error) or error < 0
            for level in (*reproduction_residuals, *near_nullspace_defects)
            for error in level
        ):
            raise ValueError("Transfer defects must be finite and nonnegative.")
        if transfer_entries < 0 or any(rounds < 0 for rounds in coarsening_rounds):
            raise ValueError("Transfer storage and round counts must be nonnegative.")
        self.level_sizes = level_sizes
        self.components = components
        self.reproduction_degree = reproduction_degree
        self.reproduction_residuals = reproduction_residuals
        self.near_nullspace_defects = near_nullspace_defects
        self.coarsening_rounds = coarsening_rounds
        self.transfer_entries = transfer_entries
        self.stopping_reason = parse(
            stopping_reason, MeshfreeHierarchyStop, "stopping_reason"
        )
        self.grid_complexity = sum(level_sizes) / level_sizes[0]
        self.restriction_coordinates = "stiffness-coordinate-transpose"


@final
class MeshfreeHierarchyPlan(StrictModule, NonTrainableState):
    """Stable-ID independent-set hierarchy for scalar or coupled stiffness coordinates.

    Fine-space coordinates must be Euclidean. For a mass-paired strong-form
    operator, first express its equation in stiffness coordinates; silently
    substituting a Hilbert mass adjoint would represent a different equation.

    Coarse nodes are a maximal independent set of the symmetric kNN conflict
    graph selected with deterministic priorities hashed from stable IDs, so the
    hierarchy is independent of input ordering. ``boundary`` nodes are forced
    coarse on the policy's retention levels and ``features`` on every level.
    Retained coarse nodes use exact nodal inclusion, preserving arbitrary
    coarse values even when neighboring retained boundary nodes are collinear.
    Nonretained target nodes use strictly accepted PHS reconstruction
    reproducing ``policy.reproduction_degree`` polynomials.

    ``eliminated`` marks fine coordinates whose equations are decoupled
    identity rows of the solve operator (eliminated Dirichlet values, gauges).
    Their corrections are exactly zero, so their prolongation rows are empty
    and their identity equations never enter a Galerkin coarse operator;
    points with every coordinate eliminated are never coarse. Not every
    boundary-heavy cloud coarsens; that outcome is reported.
    """

    points: Array
    boundary: Array
    features: Array
    eliminated: Array
    stable_ids: Array
    components: MeshfreeComponentSpace
    near_nullspace: MeshfreeNearNullspace
    policy: MeshfreeCoarseningPolicy

    def __init__(
        self,
        points: ArrayLike,
        /,
        *,
        boundary: ArrayLike | None = None,
        features: ArrayLike | None = None,
        eliminated: ArrayLike | None = None,
        stable_ids: ArrayLike | None = None,
        components: MeshfreeComponentSpace | None = None,
        near_nullspace: MeshfreeNearNullspace | None = None,
        policy: MeshfreeCoarseningPolicy | None = None,
    ) -> None:
        points_ = jnp.asarray(points)
        policy_ = MeshfreeCoarseningPolicy() if policy is None else policy
        components_ = MeshfreeComponentSpace() if components is None else components
        nullspace_ = MeshfreeNearNullspace() if near_nullspace is None else near_nullspace
        if not isinstance(policy_, MeshfreeCoarseningPolicy):
            raise TypeError("policy must be MeshfreeCoarseningPolicy.")
        if not isinstance(components_, MeshfreeComponentSpace):
            raise TypeError("components must be MeshfreeComponentSpace.")
        if not isinstance(nullspace_, MeshfreeNearNullspace):
            raise TypeError("near_nullspace must be MeshfreeNearNullspace.")
        if points_.ndim != 2 or points_.shape[0] < 1 or points_.shape[1] < 1:
            raise ValueError("points must have shape (point_count, spatial_dimension).")
        if not jnp.issubdtype(points_.dtype, jnp.floating):
            raise TypeError("points must have a real floating dtype.")
        count, dimension = points_.shape
        if count > policy_.maximum_points:
            raise LinearCapabilityError("Meshfree hierarchy exceeds maximum_points.")
        boundary_ = _point_mask(boundary, count, "boundary")
        features_ = _point_mask(features, count, "features")
        eliminated_ = _point_mask(
            eliminated, count * components_.components, "eliminated"
        )
        ids = (
            jnp.arange(count, dtype=jnp.int32)
            if stable_ids is None
            else jnp.asarray(stable_ids)
        )
        if ids.shape != (count,) or not jnp.issubdtype(ids.dtype, jnp.integer):
            raise ValueError(
                "stable_ids must be an integer vector with one entry per point."
            )
        host_points, host_ids = jax.device_get((points_, ids))
        if not np.all(np.isfinite(host_points)) or np.unique(host_ids).size != count:
            raise ValueError("Points must be finite and stable IDs must be unique.")
        if nullspace_.kind == "rigid-body" and (
            dimension not in (2, 3) or components_.components != dimension
        ):
            raise ValueError(
                "Rigid-body candidates require a 2-D or 3-D vector field with one component per axis."
            )
        if nullspace_.vectors is not None and (
            nullspace_.vectors.shape[0] != count * components_.components
            or nullspace_.vectors.dtype != points_.dtype
        ):
            raise ValueError(
                "Supplied near-nullspace vectors must use fine coordinates and the point dtype."
            )
        self.points = points_
        self.boundary = boundary_
        self.features = features_
        self.eliminated = eliminated_
        self.stable_ids = ids
        self.components = components_
        self.near_nullspace = nullspace_
        self.policy = policy_

    def prepare(self, fine_space: ArraySpace, /) -> PreparedMeshfreeHierarchy:
        size = self.points.shape[0] * self.components.components
        if not isinstance(fine_space, ArraySpace) or fine_space.shape != (size,):
            raise ValueError(
                "fine_space must be a flat ArraySpace with points * components coordinates."
            )
        if not isinstance(fine_space.pairing, EuclideanPairing):
            raise ValueError(
                "Meshfree Galerkin transfers require Euclidean stiffness coordinates, not a mass pairing."
            )
        if fine_space.dtype != np.dtype(self.points.dtype):
            raise TypeError("fine_space dtype must match points.")
        unisolvent = comb(
            self.points.shape[1] + self.policy.reproduction_degree,
            self.policy.reproduction_degree,
        )
        if self.policy.interpolation_neighbors < unisolvent:
            raise ValueError(
                "interpolation_neighbors cannot reproduce the declared polynomial degree."
            )
        return _prepare_hierarchy(self, fine_space)


@final
class PreparedMeshfreeHierarchy(StrictModule, NonTrainableState):
    """Prepared levels, typed spaces, transfers and the retained-node maps.

    ``retained_indices[level]`` lists, in coarse coordinate order, the fine
    point index of every coarse point; it defines exact nodal injection.
    """

    plan: MeshfreeHierarchyPlan
    level_points: tuple[Array, ...]
    level_ids: tuple[Array, ...]
    retained_indices: tuple[Array, ...]
    spaces: tuple[ArraySpace, ...]
    transfers: tuple[tuple[AbstractLinearOperator, AbstractLinearOperator], ...]
    evidence: MeshfreeHierarchyEvidence
    hierarchy_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MeshfreeHierarchyPlan,
        level_points: tuple[Array, ...],
        level_ids: tuple[Array, ...],
        retained_indices: tuple[Array, ...],
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
        if len(transfers) != len(level_points) - 1 or len(retained_indices) != len(
            transfers
        ):
            raise ValueError(
                "Hierarchy requires one transfer pair and retained map per transition."
            )
        if tuple(points.shape[0] for points in level_points) != evidence.level_sizes:
            raise ValueError(
                "Hierarchy geometry must match observed evidence capacities."
            )
        components = plan.components.components
        for points, ids, space in zip(level_points, level_ids, spaces, strict=True):
            if (
                points.ndim != 2
                or points.shape[1] != plan.points.shape[1]
                or ids.shape != (points.shape[0],)
            ):
                raise ValueError(
                    "Each level must retain the spatial dimension and one stable ID per point."
                )
            if not isinstance(space, ArraySpace) or space.shape != (
                points.shape[0] * components,
            ):
                raise ValueError(
                    "Each level space must match its point and component coordinates."
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
            if retained_indices[level].shape != (level_points[level + 1].shape[0],):
                raise ValueError("Each retained map must list every coarse point.")
        identifier = canonical_fingerprint(
            {
                "kind": "meshfree-hierarchy",
                "fine_space": spaces[0].space_id,
                "geometry": array_tree_fingerprint(
                    (
                        plan.points,
                        plan.boundary,
                        plan.features,
                        plan.eliminated,
                        plan.stable_ids,
                    )
                ),
                "components": plan.components.components,
                "layout": plan.components.layout,
                "near_nullspace": plan.near_nullspace.kind,
                "near_nullspace_vectors": (
                    None
                    if plan.near_nullspace.vectors is None
                    else array_tree_fingerprint(plan.near_nullspace.vectors)
                ),
                "transfers": tuple((r.operator_id, p.operator_id) for r, p in transfers),
            }
        )
        self.plan = plan
        self.level_points = level_points
        self.level_ids = level_ids
        self.retained_indices = retained_indices
        self.spaces = spaces
        self.transfers = transfers
        self.evidence = evidence
        self.hierarchy_id = identifier


def _point_mask(value: ArrayLike | None, count: int, name: str, /) -> Array:
    mask = jnp.zeros(count, dtype=jnp.bool_) if value is None else jnp.asarray(value)
    if mask.shape != (count,) or mask.dtype != jnp.bool_:
        raise ValueError(f"{name} must be a Boolean vector with one entry per point.")
    return mask


def _coordinate_rows(
    points: ArrayLike, point_count: int, components: MeshfreeComponentSpace, /
) -> np.ndarray:
    """Host coordinate indices of the given points in a level of ``point_count``."""
    selected = np.asarray(jax.device_get(points)).astype(np.int32)
    offsets = np.arange(components.components, dtype=np.int32)
    match components.layout:
        case "nodal":
            rows = selected[:, None] * components.components + offsets[None, :]
        case "block":
            rows = offsets[:, None] * point_count + selected[None, :]
        case unreachable:
            assert_never(unreachable)
    return rows.reshape((-1,))


def _point_coordinates(
    values: np.ndarray, point_count: int, components: MeshfreeComponentSpace, /
) -> np.ndarray:
    """Host coordinate values regrouped as ``(point_count, components)``."""
    match components.layout:
        case "nodal":
            return values.reshape((point_count, components.components))
        case "block":
            return values.reshape((components.components, point_count)).T
        case unreachable:
            assert_never(unreachable)


def _host_take(values: ArrayLike, rows: np.ndarray, /) -> Array:
    """Gather prepared level data on the host and transfer it once.

    Level sizes are data dependent; an eager device gather would compile one
    program per level shape, whereas ``device_put`` is a plain transfer.
    """
    return jax.device_put(np.asarray(jax.device_get(values))[rows])


def _stable_priorities(ids: np.ndarray, /) -> np.ndarray:
    # A SplitMix64 finalizer decorrelates priority from geometry-aligned ID
    # numbering; ties break by the ID itself. The resulting order is a pure
    # function of the stable IDs, so the selected set ignores input ordering,
    # while priorities uncorrelated with the graph bound the expected round
    # count logarithmically instead of by the length of an ordered chain.
    state = ids.astype(np.int64).view(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    state = state ^ (state >> np.uint64(31))
    priority = np.empty(ids.size, dtype=np.int32)
    priority[np.lexsort((ids, state))] = np.arange(1, ids.size + 1, dtype=np.int32)
    return priority


@jax.jit
def _priority_independent_set(
    conflicts: EdgeRelation,
    priority: Array,
    retained: Array,
    active: Array,
    maximum_rounds: Array,
) -> tuple[Array, Array]:
    """Deterministic parallel greedy MIS with forced retained nodes.

    With fixed unique priorities every round admits each undecided node that
    outranks all undecided neighbors and excludes their neighbors; the result
    equals the sequential greedy set in priority order. The highest-ranked
    undecided node always joins, so each round makes progress. Inactive
    storage padding starts excluded and has no conflicts.
    """

    def adjacent_maximum(values: Array, /) -> Array:
        return route_reduce(conflicts, gather_routes(conflicts, values), reduction="max")

    covered = adjacent_maximum(retained.astype(jnp.int32)) > 0
    status = jnp.where(
        ~active,
        _EXCLUDED,
        jnp.where(retained, _SELECTED, jnp.where(covered, _EXCLUDED, _UNDECIDED)),
    ).astype(jnp.int8)

    def unfinished(state: tuple[Array, Array], /) -> Array:
        current, rounds = state
        return jnp.any(current == _UNDECIDED) & (rounds < maximum_rounds)

    def admit(state: tuple[Array, Array], /) -> tuple[Array, Array]:
        current, rounds = state
        undecided = current == _UNDECIDED
        strongest = adjacent_maximum(jnp.where(undecided, priority, 0))
        joined = undecided & (priority > strongest)
        blocked = adjacent_maximum(joined.astype(jnp.int32)) > 0
        updated = jnp.where(
            joined,
            jnp.int8(_SELECTED),
            jnp.where(undecided & blocked, jnp.int8(_EXCLUDED), current),
        )
        return updated, rounds + 1

    return jax.lax.while_loop(
        unfinished, admit, (status, jnp.asarray(0, dtype=jnp.int32))
    )


def _coarse_subset(
    points: Array,
    retained: Array,
    eligible: np.ndarray,
    stable_ids: Array,
    policy: MeshfreeCoarseningPolicy,
) -> tuple[np.ndarray, int, bool]:
    """Return canonical-order coarse indices, device rounds, and completion.

    Ineligible points start excluded: they are never coarse and, like every
    excluded node, never block a neighbor.
    """
    # Canonicalize before neighborhood selection so geometric ties do not depend
    # on the caller's point ordering. Only bounded k-neighbor routes are stored.
    # Level topology is host-prepared: gathers and transfers stay on the host
    # (``device_put`` is a transfer, not a per-shape compiled program).
    host_ids, host_points, host_retained = jax.device_get((stable_ids, points, retained))
    order = np.argsort(host_ids, kind="stable")
    count = order.size
    cloud = host_points[order]
    width = min(policy.coarsening_neighbors, count)
    neighborhood = MeshfreeNeighborhoodPlan(
        cloud,
        width,
        maximum_candidates=policy.maximum_candidates,
        target_chunk_size=policy.chunk_rows,
    ).prepare()
    # The conflict graph is padded to the neighborhood's bucketed storage so
    # the compiled MIS kernel is reused across levels; padding has no edges.
    storage = bucketed_storage_capacity(count)
    neighbors, neighbor_valid = jax.device_get(
        (neighborhood.relation.source_indices, neighborhood.relation.valid)
    )
    host_retained = host_retained[order]
    host_eligible = eligible[order]
    host_retained = host_retained & host_eligible
    rows = np.broadcast_to(
        np.arange(count, dtype=neighbors.dtype)[:, None], neighbors.shape
    )
    padding = ((0, storage - count), (0, 0))
    sources = np.pad(neighbors, padding).reshape((-1,))
    targets = np.pad(rows, padding).reshape((-1,))
    valid = np.pad(neighbor_valid & (neighbors != rows), padding).reshape((-1,))
    conflicts = EdgeRelation(
        np.concatenate((sources, targets)),
        np.concatenate((targets, sources)),
        source_size=storage,
        target_size=storage,
        valid=np.concatenate((valid, valid)),
    )
    status, rounds = jax.device_get(
        _priority_independent_set(
            conflicts,
            jax.device_put(
                np.pad(_stable_priorities(host_ids[order]), (0, storage - count))
            ),
            jax.device_put(np.pad(host_retained, (0, storage - count))),
            jax.device_put(np.pad(host_eligible, (0, storage - count))),
            jax.device_put(np.int32(policy.maximum_coarsening_rounds)),
        )
    )
    status = status[:count]
    complete = not np.any(status == _UNDECIDED)
    return order[np.flatnonzero(status == _SELECTED)], int(rounds), complete


@jax.jit
def _monomial_columns(
    points: Array, exponents: Array, lower: Array, upper: Array
) -> Array:
    center = (upper + lower) / 2
    half = jnp.where(upper > lower, (upper - lower) / 2, 1.0)
    scaled = (points - center) / half
    return jnp.prod(scaled[:, None, :] ** exponents[None, :, :], axis=-1)


def _scalar_routes(
    fine: Array,
    coarse: Array,
    subset: np.ndarray,
    skipped: np.ndarray,
    width: int,
    policy: MeshfreeCoarseningPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Prolongation routes, or ``None`` when any reconstruction row is refused.

    ``skipped`` fine points (every coordinate eliminated) keep empty rows.
    """
    dimension = fine.shape[1]
    indices = np.zeros((fine.shape[0], width), dtype=np.int32)
    valid = np.zeros((fine.shape[0], width), dtype=np.bool_)
    coefficients = np.zeros((fine.shape[0], width), dtype=np.dtype(fine.dtype))
    indices[subset, 0] = np.arange(subset.size, dtype=np.int32)
    valid[subset, 0] = True
    coefficients[subset, 0] = 1.0
    # Retained subset nodes have an exact inclusion map, regardless of whether
    # their nearest coarse neighbors span the ambient polynomial space. Only
    # genuinely cross-target rows require a PHS reconstruction.
    cross_target = ~skipped
    cross_target[subset] = False
    target_indices = np.flatnonzero(cross_target)
    if not target_indices.size:
        return indices, valid, coefficients
    targets = np.asarray(jax.device_get(fine))[target_indices]
    neighborhood = MeshfreeNeighborhoodPlan(
        coarse,
        width,
        targets=targets,
        maximum_candidates=policy.maximum_candidates,
        target_chunk_size=policy.chunk_rows,
    ).prepare()
    degree = policy.reproduction_degree
    stencils = prepare_local_stencils(
        neighborhood,
        coarse,
        targets,
        (MeshfreeFunctional(((0,) * dimension,), (1.0,), name="interpolation"),),
        LocalStencilPolicy(
            approximation="phs-rbf-fd",
            polynomial_degree=degree,
            phs_power=2 * degree + 1,
            acceptance="mask",
            chunk_rows=policy.chunk_rows,
        ),
    )
    if stencils.report.refused_rows:
        # Nearest coarse supports that do not admit the declared polynomial
        # space (for example collinear retained edges) are not repaired by
        # widening or regularizing; this transition is not admitted.
        return None
    routes, route_valid, weights = jax.device_get(
        (
            neighborhood.relation.source_indices,
            neighborhood.relation.valid,
            stencils.weights[0],
        )
    )
    indices[target_indices] = routes
    valid[target_indices] = route_valid
    coefficients[target_indices] = weights
    return indices, valid, coefficients


def _component_routes(
    indices: np.ndarray,
    valid: np.ndarray,
    coefficients: np.ndarray,
    coarse_count: int,
    components: MeshfreeComponentSpace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = components.components
    if count == 1:
        return indices, valid, coefficients
    offsets = np.arange(count, dtype=np.int32)
    match components.layout:
        case "nodal":
            expanded = indices[:, None, :] * count + offsets[None, :, None]
            return (
                expanded.reshape((-1, indices.shape[1])),
                np.repeat(valid, count, axis=0),
                np.repeat(coefficients, count, axis=0),
            )
        case "block":
            expanded = offsets[:, None, None] * coarse_count + indices[None, :, :]
            return (
                expanded.reshape((-1, indices.shape[1])),
                np.tile(valid, (count, 1)),
                np.tile(coefficients, (count, 1)),
            )
        case unreachable:
            assert_never(unreachable)


def _near_nullspace_candidates(
    points: Array,
    space: MeshfreeComponentSpace,
    near_nullspace: MeshfreeNearNullspace,
    /,
) -> Array:
    """Fine-coordinate candidate columns of ``near_nullspace`` on ``points``."""
    count, dimension = points.shape
    components = space.components
    if near_nullspace.kind == "rigid-body" and (
        dimension not in (2, 3) or components != dimension
    ):
        raise ValueError(
            "Rigid-body candidates require a 2-D or 3-D vector field with one component per axis."
        )
    centered = points - jnp.mean(points, axis=0)
    unit = jnp.eye(components, dtype=points.dtype)
    match near_nullspace.kind:
        case "none":
            fields = jnp.zeros((count, components, 0), dtype=points.dtype)
        case "constant":
            fields = jnp.broadcast_to(unit[None], (count, components, components))
        case "affine":
            affine = jnp.concatenate(
                (jnp.ones((count, 1), dtype=points.dtype), centered), axis=1
            )
            fields = (affine[:, None, None, :] * unit[None, :, :, None]).reshape(
                (count, components, components * (dimension + 1))
            )
        case "rigid-body":
            x = centered
            zero = jnp.zeros((count,), dtype=points.dtype)
            if dimension == 2:
                rotations = jnp.stack((-x[:, 1], x[:, 0]), axis=1)[:, :, None]
            else:
                rotations = jnp.stack(
                    (
                        jnp.stack((zero, -x[:, 2], x[:, 1]), axis=1),
                        jnp.stack((x[:, 2], zero, -x[:, 0]), axis=1),
                        jnp.stack((-x[:, 1], x[:, 0], zero), axis=1),
                    ),
                    axis=2,
                )
            fields = jnp.concatenate(
                (
                    jnp.broadcast_to(unit[None], (count, components, components)),
                    rotations,
                ),
                axis=2,
            )
        case unreachable:
            assert_never(unreachable)
    candidate_count = fields.shape[2]
    match space.layout:
        case "nodal":
            candidates = fields.reshape((count * components, candidate_count))
        case "block":
            candidates = jnp.swapaxes(fields, 0, 1).reshape(
                (count * components, candidate_count)
            )
        case unreachable:
            assert_never(unreachable)
    if near_nullspace.vectors is not None:
        candidates = jnp.concatenate((candidates, near_nullspace.vectors), axis=1)
    return candidates


# One fused executable per level shape instead of an eager op-by-op lowering of
# the sparse action and its reductions.
@eqx.filter_jit
def _relative_column_defects(
    prolongation: AbstractLinearOperator, coarse: Array, fine: Array
) -> Array:
    reproduced = jax.vmap(prolongation.mv, in_axes=1, out_axes=1)(coarse)
    scale = jnp.maximum(jnp.max(jnp.abs(fine), axis=0), jnp.finfo(fine.dtype).tiny)
    return jnp.max(jnp.abs(reproduced - fine), axis=0) / scale


def _column_defects(
    prolongation: AbstractLinearOperator, coarse: Array, fine: Array
) -> np.ndarray:
    return np.asarray(
        jax.device_get(_relative_column_defects(prolongation, coarse, fine))
    )


def _prepare_hierarchy(
    plan: MeshfreeHierarchyPlan, fine_space: ArraySpace
) -> PreparedMeshfreeHierarchy:
    policy = plan.policy
    components = plan.components
    boundary, features = jax.device_get((plan.boundary, plan.features))
    # Eliminated identity rows exist only in the fine solve operator; coarse
    # coordinates are genuine Galerkin unknowns.
    eliminated = np.asarray(jax.device_get(plan.eliminated))
    points, ids = [plan.points], [plan.stable_ids]
    spaces = [fine_space]
    subsets: list[Array] = []
    transfers: list[tuple[AbstractLinearOperator, AbstractLinearOperator]] = []
    reproduction: list[tuple[float, ...]] = []
    defects: list[tuple[float, ...]] = []
    rounds_used: list[int] = []
    candidates = _near_nullspace_candidates(
        plan.points, plan.components, plan.near_nullspace
    )
    entries = 0
    reason: MeshfreeHierarchyStop = "maximum-levels"
    dimension = plan.points.shape[1]
    degree = policy.reproduction_degree
    basis = TotalDegreePolynomialFeatures(dimension, degree)
    exponents = jnp.concatenate(
        (jnp.zeros((1, dimension), dtype=jnp.int32), basis.exponents), axis=0
    )
    exponent_degrees = np.asarray(jax.device_get(jnp.sum(exponents, axis=1)))
    unisolvent = basis.feature_count + 1
    for level in range(policy.maximum_levels - 1):
        fine = points[-1]
        count = fine.shape[0]
        if count <= max(policy.minimum_coarse_points, unisolvent):
            reason = "minimum-coarse-points"
            break
        point_eliminated = np.all(
            _point_coordinates(eliminated, count, components), axis=1
        )
        retention = policy.boundary_retention_levels
        retained = features | (
            boundary if retention is None or level < retention else False
        )
        subset, rounds, complete = _coarse_subset(
            fine, retained, ~point_eliminated, ids[-1], policy
        )
        if not complete:
            reason = "coarsening-round-limit"
            break
        if subset.size >= count:
            reason = "no-progress-retention"
            break
        if subset.size > policy.maximum_coarse_fraction * count:
            reason = "insufficient-reduction"
            break
        if subset.size < unisolvent:
            reason = "unisolvence-minimum"
            break
        coarse = _host_take(fine, subset)
        coarse_ids = _host_take(ids[-1], subset)
        width = min(policy.interpolation_neighbors, subset.size)
        next_entries = count * width * components.components
        if entries + next_entries > policy.maximum_transfer_entries:
            raise LinearCapabilityError(
                "Meshfree hierarchy exceeds maximum_transfer_entries."
            )
        routes = _scalar_routes(fine, coarse, subset, point_eliminated, width, policy)
        if routes is None:
            reason = "transfer-row-refusal"
            break
        indices, valid, coefficients = routes
        coarse_space = ArraySpace(
            (subset.size * components.components,),
            dtype=fine_space.dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "meshfree-level-space",
                    "fine": fine_space.space_id,
                    "level": level + 1,
                    "ids": array_tree_fingerprint(coarse_ids),
                    "geometry": array_tree_fingerprint(coarse),
                    "components": components.components,
                    "layout": components.layout,
                }
            ),
        )
        # Observe polynomial reproduction on the actual scalar sparse action.
        # Host synchronization is confined to this immutable setup boundary.
        scalar = SparseCoordinateOperator(
            RowRelation(indices, source_size=subset.size, valid=valid),
            coefficients,
            source=ArraySpace((subset.size,), dtype=fine_space.dtype),
            target=ArraySpace((fine.shape[0],), dtype=fine_space.dtype),
        )
        host_fine = np.asarray(jax.device_get(fine))
        lower, upper = jax.device_put((host_fine.min(axis=0), host_fine.max(axis=0)))
        fine_monomials = np.array(
            jax.device_get(_monomial_columns(fine, exponents, lower, upper)), copy=True
        )
        fine_monomials[point_eliminated] = 0.0
        monomial_defects = _column_defects(
            scalar,
            _monomial_columns(coarse, exponents, lower, upper),
            jax.device_put(fine_monomials),
        )
        by_degree = tuple(
            float(np.max(monomial_defects[exponent_degrees == q]))
            for q in range(degree + 1)
        )
        if max(by_degree) > policy.reproduction_tolerance:
            raise LinearCapabilityError(
                "Meshfree transfer failed its declared polynomial reproduction tolerance."
            )
        routes, route_valid, weights = _component_routes(
            indices, valid, coefficients, subset.size, components
        )
        route_valid = route_valid & ~eliminated[:, None]
        if np.any(np.bincount(routes[route_valid], minlength=coarse_space.size) == 0):
            # A coarse coordinate reaching only eliminated fine rows would make
            # every Galerkin coarse operator singular.
            reason = "empty-coarse-coordinate"
            break
        relation = RowRelation(routes, source_size=coarse_space.size, valid=route_valid)
        values = jax.device_put(weights)
        prolongation = SparseCoordinateOperator(
            relation, values, source=coarse_space, target=spaces[-1]
        )
        restriction = SparseCoordinateOperator(
            relation.as_edge_relation().transpose(),
            jax.device_put(weights.reshape((-1,))),
            source=spaces[-1],
            target=coarse_space,
        )
        coarse_candidates = _host_take(
            candidates, _coordinate_rows(subset, count, components)
        )
        free_candidates = jnp.where(jax.device_put(eliminated)[:, None], 0.0, candidates)
        defects.append(
            tuple(
                float(value)
                for value in _column_defects(
                    prolongation, coarse_candidates, free_candidates
                )
            )
            if candidates.shape[1]
            else ()
        )
        candidates = coarse_candidates
        reproduction.append(by_degree)
        rounds_used.append(rounds)
        transfers.append((restriction, prolongation))
        subsets.append(jax.device_put(subset.astype(np.int32)))
        points.append(coarse)
        ids.append(coarse_ids)
        boundary, features = boundary[subset], features[subset]
        eliminated = np.zeros(subset.size * components.components, dtype=np.bool_)
        spaces.append(coarse_space)
        entries += next_entries
    evidence = MeshfreeHierarchyEvidence(
        tuple(point.shape[0] for point in points),
        components=components.components,
        reproduction_degree=degree,
        reproduction_residuals=tuple(reproduction),
        near_nullspace_defects=tuple(defects),
        coarsening_rounds=tuple(rounds_used),
        transfer_entries=entries,
        stopping_reason=reason,
    )
    return PreparedMeshfreeHierarchy(
        plan,
        tuple(points),
        tuple(ids),
        tuple(subsets),
        tuple(spaces),
        tuple(transfers),
        evidence,
    )


def _level_ordering(hierarchy: PreparedMeshfreeHierarchy, level: int, /) -> Array | None:
    """Stable-ID coordinate sweep order of one level, or ``None`` when natural."""
    ids = np.asarray(jax.device_get(hierarchy.level_ids[level]))
    order = np.argsort(ids, kind="stable")
    if np.array_equal(order, np.arange(order.size)):
        return None
    return jax.device_put(_coordinate_rows(order, order.size, hierarchy.plan.components))


def _coarse_kernel(
    hierarchy: PreparedMeshfreeHierarchy, kernel: LinearSubspace, /
) -> LinearSubspace:
    fine_space = hierarchy.spaces[0]
    if not kernel.space.compatible(fine_space) or kernel.batch_shape:
        raise ValueError("Declared kernels must be unbatched in the fine space.")
    tolerance = hierarchy.plan.policy.reproduction_tolerance
    basis = kernel.basis
    for level, (_, prolongation) in enumerate(hierarchy.transfers):
        coarse = _host_take(
            basis,
            _coordinate_rows(
                hierarchy.retained_indices[level],
                hierarchy.level_points[level].shape[0],
                hierarchy.plan.components,
            ),
        )
        defect = float(np.max(_column_defects(prolongation, coarse, basis)))
        if not np.isfinite(defect) or defect > tolerance:
            raise ValueError(
                f"Declared kernel is not reproduced by meshfree transfer {level} "
                f"(relative defect {defect:.3e}); a projected coarse solve would "
                "represent a different singular equation."
            )
        basis = coarse
    return LinearSubspace(hierarchy.spaces[-1], basis, dimension=kernel.dimension)


def _default_smoothers(
    hierarchy: PreparedMeshfreeHierarchy, /
) -> tuple[AbstractPreconditionerBuilder, ...]:
    coupled = hierarchy.plan.components.components > 1
    return tuple(
        ILUPreconditionerBuilder()
        if coupled and level
        else GaussSeidelPreconditionerBuilder(
            direction="forward", ordering=_level_ordering(hierarchy, level)
        )
        for level in range(len(hierarchy.transfers))
    )


def meshfree_multigrid_builder(
    hierarchy: PreparedMeshfreeHierarchy,
    /,
    *,
    smoothers: tuple[AbstractPreconditioner | AbstractPreconditionerBuilder, ...]
    | None = None,
    coarse_solver: AbstractPreconditioner | AbstractPreconditionerBuilder | None = None,
    nullspace_policy: NullspacePolicy | None = None,
    cycle: MultigridCycleKind = "v",
    pre_smoothing: int = 1,
    post_smoothing: int = 1,
    refresh_mode: MultigridRefreshMode = "reuse-symbolic-sparse-products",
    assembly: SparseAssemblyPolicy | None = None,
) -> AbstractPreconditionerBuilder:
    """Use native Galerkin setup and the selected cycle over meshfree transfers.

    Native materialization policies still bound coarse setup, and ``assembly``
    bounds every exact sparse Galerkin product (Galerkin products grow with the
    fine operator, so large clouds declare limits for their capacity). A cloud
    too small (or boundary-only) for a genuine transition uses the coarse source
    directly; no fictitious duplicate level is introduced.

    The default fine-level smoother is a forward sparse Gauss--Seidel sweep in
    stable-ID coordinate order, so the action does not depend on caller point
    ordering. Scalar hierarchies use the same sweep on coarse levels. Coupled
    component hierarchies use ILU(0) on the Galerkin coarse levels: on a
    collocated 2-D elasticity cloud (N=289) every stationary fine smoother
    diverges (spectral radius of forward Gauss--Seidel 3.41, ILU(0) 55.8,
    point-block Jacobi 1.17), and coarse Gauss--Seidel amplifies further
    (radius 23 on level 1), whereas coarse ILU(0) contracts (0.2, 0.002) and
    the cycle needs 34 GMRES iterations at any depth instead of 59-61.
    Damped Jacobi requires spectral/diagonal-dominance assumptions that
    collocated meshfree operators do not generally satisfy. Callers may supply
    native point-block Jacobi, symmetric Gauss--Seidel, ILU/ILUT, or Chebyshev
    builders; each enforces its own property requirements. Neither smoother
    nor cycle is claimed to contract every cloud; the outer native Krylov
    solve checks the original fine-system residual.

    ``nullspace_policy`` kernels must be exactly reproduced by every transfer
    (constants, rigid-body modes, component indicators); their injected coarse
    bases define the projected coarse pseudoinverse.
    """
    if not isinstance(hierarchy, PreparedMeshfreeHierarchy):
        raise TypeError("hierarchy must be PreparedMeshfreeHierarchy.")
    cycle_policy = MultigridCyclePolicy(cycle)
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
                "Choose either the declared-nullspace coarse solve or a supplied coarse_solver."
            )
        if not hierarchy.transfers:
            raise LinearCapabilityError(
                "A projected coarse solve requires genuine coarsening; dense fine fallback is forbidden."
            )
        if nullspace_policy.right is None or nullspace_policy.left is None:
            raise ValueError("Projected coarse solves require both declared kernels.")
        tolerance = hierarchy.plan.policy.reproduction_tolerance
        # The Galerkin coarse kernel is exact only to the transfer reproduction
        # tolerance, so singular values below that relative level are declared
        # kernel directions; the rank must still equal the declared complement.
        coarse = ProjectedPseudoinversePreconditionerBuilder(
            NullspacePolicy(
                right=_coarse_kernel(hierarchy, nullspace_policy.right),
                left=_coarse_kernel(hierarchy, nullspace_policy.left),
                compatibility=nullspace_policy.compatibility,
                gauge=nullspace_policy.gauge,
            ),
            rank_policy=RankPolicy(relative_cutoff=tolerance),
            tolerance=tolerance,
        )
    if not hierarchy.transfers:
        if not isinstance(coarse, AbstractPreconditionerBuilder):
            raise TypeError("A one-level hierarchy requires a coarse-solver builder.")
        return coarse
    smoothing = _default_smoothers(hierarchy) if smoothers is None else tuple(smoothers)
    return GalerkinHierarchyBuilder(
        hierarchy.transfers,
        smoothing,
        coarse,
        cycle_policy=cycle_policy,
        refresh_mode=refresh_mode,
        pre_smoothing=pre_smoothing,
        post_smoothing=post_smoothing,
        assembly=assembly,
    )


__all__ = [
    "MeshfreeComponentLayout",
    "MeshfreeComponentSpace",
    "MeshfreeCoarseningPolicy",
    "MeshfreeHierarchyEvidence",
    "MeshfreeHierarchyPlan",
    "MeshfreeHierarchyStop",
    "MeshfreeNearNullspace",
    "MeshfreeNearNullspaceKind",
    "PreparedMeshfreeHierarchy",
    "meshfree_multigrid_builder",
]
