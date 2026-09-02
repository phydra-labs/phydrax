#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array

from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ._checkpoint import (
    checkpoint_compatibility,
    CheckpointCorruptionError,
    pack_array_tree,
    read_checkpoint_archive,
    unpack_array_tree,
    write_checkpoint_archive,
)
from ._nested_extensions import (
    EllipsoidalNestedBounds,
    NestedSamplingPlan,
    PeriodicNestedCoordinate,
    PhantomNestedState,
)


_CHECKPOINT_KIND = "nested_sampling"
_PREPARED_PRIOR = SampleAddress("phydrax.uq", "prepared-nested-prior", role="replacement")
_PREPARED_BASE = SampleAddress("phydrax.uq", "prepared-nested-base", role="proposal")
_PREPARED_ELLIPSOID = SampleAddress(
    "phydrax.uq", "prepared-nested-ellipsoid", role="proposal"
)
_PREPARED_FLOW = SampleAddress("phydrax.uq", "prepared-nested-flow", role="proposal")
_PREPARED_GRADIENT = SampleAddress(
    "phydrax.uq", "prepared-nested-gradient", role="proposal"
)
_PREPARED_DISCRETE = SampleAddress(
    "phydrax.uq", "prepared-nested-discrete", role="proposal"
)
_PREPARED_PERIODIC = SampleAddress(
    "phydrax.uq", "prepared-nested-periodic", role="proposal"
)
_PREPARED_FALLBACK = SampleAddress(
    "phydrax.uq", "prepared-nested-fallback", role="proposal"
)


class PreparedNestedProposalState(StrictModule):
    """Frozen proposal geometry prepared at one dead-point epoch boundary."""

    ellipsoid_centers: Array
    ellipsoid_factors: Array
    ellipsoid_active: Array
    ellipsoid_log_volumes: Array
    ellipsoid_condition: Array
    flow_mean: Array
    flow_factor: Array
    flow_log_determinant: Array
    flow_condition: Array
    flow_active: Array
    epoch: Array


class PreparedNestedAdaptationState(StrictModule):
    """Persistent branch evidence for the bounded prepared proposal composition."""

    base_attempts: Array
    base_acceptances: Array
    ellipsoid_attempts: Array
    ellipsoid_acceptances: Array
    flow_attempts: Array
    flow_acceptances: Array
    gradient_attempts: Array
    gradient_acceptances: Array
    discrete_updates: Array
    discrete_moves: Array
    periodic_updates: Array
    periodic_moves: Array
    wrap_crossings: Array
    phantom_creations: Array
    phantom_revalidations: Array
    phantom_reuses: Array
    fallback_draws: Array
    fallback_acceptances: Array
    contour_rejections: Array
    proposal_failures: Array
    dynamic_additions: Array

    @property
    def branch_evidence(self) -> dict[str, dict[str, int]]:
        """Return host-readable attempts and accepted moves for every kernel."""
        return {
            "base": {
                "attempts": int(self.base_attempts),
                "accepted": int(self.base_acceptances),
            },
            "ellipsoid_independence_mh": {
                "attempts": int(self.ellipsoid_attempts),
                "accepted": int(self.ellipsoid_acceptances),
            },
            "frozen_flow_independence_mh": {
                "attempts": int(self.flow_attempts),
                "accepted": int(self.flow_acceptances),
            },
            "gradient_mala": {
                "attempts": int(self.gradient_attempts),
                "accepted": int(self.gradient_acceptances),
            },
            "finite_gibbs": {
                "attempts": int(self.discrete_updates),
                "accepted": int(self.discrete_moves),
            },
            "periodic_slice": {
                "attempts": int(self.periodic_updates),
                "accepted": int(self.periodic_moves),
            },
            "exact_rejection_fallback": {
                "attempts": int(self.fallback_draws),
                "accepted": int(self.fallback_acceptances),
            },
        }


class PreparedNestedState(StrictModule):
    """Fixed-capacity variable-live execution, proposal, and quadrature state."""

    root_key: Array
    initial_log_likelihood: Array
    live_positions: Array
    live_log_prior: Array
    live_log_likelihood: Array
    live_birth_log_likelihood: Array
    live_lineage: Array
    live_mask: Array
    dead_positions: Array
    dead_log_prior: Array
    dead_log_likelihood: Array
    dead_birth_log_likelihood: Array
    dead_log_weights: Array
    dead_log_prior_volume: Array
    dead_live_counts: Array
    dead_batch_indices: Array
    dead_lineage: Array
    insertion_ranks: Array
    inner_accepted: Array
    proposal_attempts: Array
    proposal_shrinkage: Array
    phantom: PhantomNestedState
    proposal: PreparedNestedProposalState
    adaptation: PreparedNestedAdaptationState
    log_prior_volume: Array
    dead_count: Array
    likelihood_evaluations: Array
    dynamic_batches: Array
    step: Array
    status: Array
    finished: Array
    plan_id: str = eqx.field(static=True)


class _CoordinateLayout(NamedTuple):
    smooth: tuple[int, ...]
    finite: tuple[tuple[int, Array, Array], ...]
    periodic: tuple[tuple[int, PeriodicNestedCoordinate], ...]


class _ProposalOutcome(NamedTuple):
    position: Array
    log_prior: Array
    log_likelihood: Array
    lineage: Array
    phantom: PhantomNestedState
    proposal: PreparedNestedProposalState
    adaptation: PreparedNestedAdaptationState
    attempts: int
    likelihood_evaluations: int
    shrinkage: int
    moved: bool
    budget_exhausted: bool
    invalid_likelihood: bool
    failed: bool


class _Evaluator:
    def __init__(self, evaluate_one, *, count: int, limit: int):
        self.evaluate_one = evaluate_one
        self.count = int(count)
        self.limit = int(limit)
        self.exhausted = False
        self.invalid = False

    def one(self, position: Array) -> tuple[Array, Array]:
        if self.count >= self.limit:
            self.exhausted = True
            dtype = position.dtype
            return jnp.asarray(-jnp.inf, dtype=dtype), jnp.asarray(-jnp.inf, dtype=dtype)
        self.count += 1
        log_prior, log_likelihood = self.evaluate_one(position)
        if bool(jnp.isnan(log_prior) | jnp.isposinf(log_prior)) or bool(
            jnp.isnan(log_likelihood) | jnp.isposinf(log_likelihood)
        ):
            self.invalid = True
        return log_prior, log_likelihood


def _coordinate_layout(
    reference_tree: Any, plan: NestedSamplingPlan, /
) -> _CoordinateLayout:
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(reference_tree)
    declared_continuous = set(plan.prior.continuous_paths)
    declared_finite = {path for path, _, _ in plan.prior.finite_supports}
    actual_paths = tuple(
        jax.tree_util.keystr(path) or "<root>" for path, _ in path_leaves
    )
    declared = declared_continuous | declared_finite
    if set(actual_paths) != declared or len(actual_paths) != len(declared):
        raise ValueError(
            "NestedPriorPlan paths must classify every parameter leaf exactly; "
            f"runtime={actual_paths}, declared={tuple(sorted(declared))}."
        )

    periodic_by_path = {coordinate.path: coordinate for coordinate in plan.prior.periodic}
    finite_by_path = {
        path: (jnp.asarray(support), jnp.asarray(masses))
        for path, support, masses in plan.prior.finite_supports
    }
    smooth: list[int] = []
    finite: list[tuple[int, Array, Array]] = []
    periodic: list[tuple[int, PeriodicNestedCoordinate]] = []
    offset = 0
    for (path, leaf), path_name in zip(path_leaves, actual_paths, strict=True):
        size = int(jnp.asarray(leaf).size)
        if path_name in finite_by_path:
            if size != 1:
                raise ValueError("Finite nested-prior paths must name scalar leaves.")
            support, masses = finite_by_path[path_name]
            finite.append((offset, support, masses))
        elif path_name in periodic_by_path:
            if size != 1:
                raise ValueError("Periodic nested-prior paths must name scalar leaves.")
            periodic.append((offset, periodic_by_path[path_name]))
        else:
            smooth.extend(range(offset, offset + size))
        offset += size

    if plan.proposal.ellipsoid and not smooth:
        raise ValueError(
            "Ellipsoid proposals require a nonperiodic continuous coordinate."
        )
    if plan.proposal.learned_flow and not smooth:
        raise ValueError(
            "Frozen-flow proposals require a nonperiodic continuous coordinate."
        )
    if plan.proposal.gradient_guided and not smooth:
        raise ValueError(
            "Gradient-guided proposals require a nonperiodic continuous coordinate."
        )
    return _CoordinateLayout(tuple(smooth), tuple(finite), tuple(periodic))


def _empty_adaptation(dtype: Any) -> PreparedNestedAdaptationState:
    zero = jnp.asarray(0, dtype=jnp.int32)
    return PreparedNestedAdaptationState(
        base_attempts=zero,
        base_acceptances=zero,
        ellipsoid_attempts=zero,
        ellipsoid_acceptances=zero,
        flow_attempts=zero,
        flow_acceptances=zero,
        gradient_attempts=zero,
        gradient_acceptances=zero,
        discrete_updates=zero,
        discrete_moves=zero,
        periodic_updates=zero,
        periodic_moves=zero,
        wrap_crossings=zero,
        phantom_creations=zero,
        phantom_revalidations=zero,
        phantom_reuses=zero,
        fallback_draws=zero,
        fallback_acceptances=zero,
        contour_rejections=zero,
        proposal_failures=zero,
        dynamic_additions=zero,
    )


def _empty_proposal(
    plan: NestedSamplingPlan, smooth_dimension: int, dtype: Any
) -> PreparedNestedProposalState:
    clusters = plan.capacity.max_clusters
    return PreparedNestedProposalState(
        ellipsoid_centers=jnp.zeros((clusters, smooth_dimension), dtype=dtype),
        ellipsoid_factors=jnp.zeros(
            (clusters, smooth_dimension, smooth_dimension), dtype=dtype
        ),
        ellipsoid_active=jnp.zeros((clusters,), dtype=bool),
        ellipsoid_log_volumes=jnp.full((clusters,), -jnp.inf, dtype=dtype),
        ellipsoid_condition=jnp.asarray(jnp.inf, dtype=dtype),
        flow_mean=jnp.zeros((smooth_dimension,), dtype=dtype),
        flow_factor=jnp.zeros((smooth_dimension, smooth_dimension), dtype=dtype),
        flow_log_determinant=jnp.asarray(-jnp.inf, dtype=dtype),
        flow_condition=jnp.asarray(jnp.inf, dtype=dtype),
        flow_active=jnp.asarray(False),
        epoch=jnp.asarray(0, dtype=jnp.int32),
    )


def _strict_covariance_factor(values: Array, /) -> tuple[Array, Array] | None:
    count, dimension = map(int, values.shape)
    if count <= dimension:
        return None
    centered = values - jnp.mean(values, axis=0, keepdims=True)
    covariance = centered.T @ centered / jnp.asarray(count - 1, dtype=values.dtype)
    factor = jnp.linalg.cholesky(covariance)
    eigenvalues = jnp.linalg.eigvalsh(covariance)
    if not bool(jnp.all(jnp.isfinite(factor))) or not bool(jnp.all(eigenvalues > 0.0)):
        return None
    condition = jnp.max(eigenvalues) / jnp.min(eigenvalues)
    return factor, condition


def _prepare_proposal_geometry(
    state: PreparedNestedState,
    plan: NestedSamplingPlan,
    layout: _CoordinateLayout,
    /,
) -> PreparedNestedProposalState:
    smooth = jnp.asarray(layout.smooth, dtype=jnp.int32)
    smooth_dimension = len(layout.smooth)
    proposal = _empty_proposal(plan, smooth_dimension, state.live_positions.dtype)
    if smooth_dimension == 0:
        return eqx.tree_at(lambda value: value.epoch, proposal, state.step)

    active = state.live_positions[state.live_mask][:, smooth]
    cluster_centers = proposal.ellipsoid_centers
    cluster_factors = proposal.ellipsoid_factors
    cluster_active = proposal.ellipsoid_active
    cluster_log_volumes = proposal.ellipsoid_log_volumes
    ellipsoid_condition = proposal.ellipsoid_condition
    if plan.proposal.ellipsoid:
        count = int(active.shape[0])
        maximum_clusters = min(
            plan.capacity.max_clusters,
            max(1, count // (smooth_dimension + 1)),
        )
        built = False
        for cluster_count in range(maximum_clusters, 0, -1):
            order = jnp.argsort(active[:, 0])
            ordered = active[order]
            centers: list[Array] = []
            factors: list[Array] = []
            volumes: list[Array] = []
            conditions: list[Array] = []
            valid = True
            for cluster in range(cluster_count):
                start = cluster * count // cluster_count
                stop = (cluster + 1) * count // cluster_count
                members = ordered[start:stop]
                resolved = _strict_covariance_factor(members)
                if resolved is None:
                    valid = False
                    break
                factor, condition = resolved
                center = jnp.mean(members, axis=0)
                coordinates = jsp.linalg.solve_triangular(
                    factor, (members - center).T, lower=True
                ).T
                radius = jnp.sqrt(jnp.max(jnp.sum(coordinates**2, axis=-1)))
                enlarged = factor * (
                    jnp.maximum(radius, jnp.asarray(1.0, dtype=factor.dtype))
                    * plan.proposal.ellipsoid_enlargement
                )
                unit_log_volume = 0.5 * smooth_dimension * jnp.log(
                    jnp.pi
                ) - jsp.special.gammaln(0.5 * smooth_dimension + 1.0)
                log_volume = unit_log_volume + jnp.sum(jnp.log(jnp.diag(enlarged)))
                centers.append(center)
                factors.append(enlarged)
                volumes.append(log_volume)
                conditions.append(condition)
            if valid:
                for cluster, (center, factor, volume) in enumerate(
                    zip(centers, factors, volumes, strict=True)
                ):
                    cluster_centers = cluster_centers.at[cluster].set(center)
                    cluster_factors = cluster_factors.at[cluster].set(factor)
                    cluster_active = cluster_active.at[cluster].set(True)
                    cluster_log_volumes = cluster_log_volumes.at[cluster].set(volume)
                ellipsoid_condition = jnp.max(jnp.stack(conditions))
                built = True
                break
        if not built:
            ellipsoid_condition = jnp.asarray(jnp.inf, dtype=active.dtype)

    flow_mean = proposal.flow_mean
    flow_factor = proposal.flow_factor
    flow_log_determinant = proposal.flow_log_determinant
    flow_condition = proposal.flow_condition
    flow_active = proposal.flow_active
    if plan.proposal.learned_flow:
        resolved = _strict_covariance_factor(active)
        if resolved is not None:
            flow_factor, flow_condition = resolved
            flow_mean = jnp.mean(active, axis=0)
            flow_log_determinant = jnp.sum(jnp.log(jnp.diag(flow_factor)))
            flow_active = jnp.asarray(True)

    return PreparedNestedProposalState(
        ellipsoid_centers=cluster_centers,
        ellipsoid_factors=cluster_factors,
        ellipsoid_active=cluster_active,
        ellipsoid_log_volumes=cluster_log_volumes,
        ellipsoid_condition=ellipsoid_condition,
        flow_mean=flow_mean,
        flow_factor=flow_factor,
        flow_log_determinant=flow_log_determinant,
        flow_condition=flow_condition,
        flow_active=flow_active,
        epoch=state.step,
    )


def _increment_adaptation(
    adaptation: PreparedNestedAdaptationState,
    counts: dict[str, int],
    /,
) -> PreparedNestedAdaptationState:
    def increment(name: str) -> Array:
        return jnp.asarray(counts.get(name, 0), dtype=jnp.int32)

    return PreparedNestedAdaptationState(
        base_attempts=adaptation.base_attempts + increment("base_attempts"),
        base_acceptances=adaptation.base_acceptances + increment("base_acceptances"),
        ellipsoid_attempts=adaptation.ellipsoid_attempts
        + increment("ellipsoid_attempts"),
        ellipsoid_acceptances=adaptation.ellipsoid_acceptances
        + increment("ellipsoid_acceptances"),
        flow_attempts=adaptation.flow_attempts + increment("flow_attempts"),
        flow_acceptances=adaptation.flow_acceptances + increment("flow_acceptances"),
        gradient_attempts=adaptation.gradient_attempts + increment("gradient_attempts"),
        gradient_acceptances=adaptation.gradient_acceptances
        + increment("gradient_acceptances"),
        discrete_updates=adaptation.discrete_updates + increment("discrete_updates"),
        discrete_moves=adaptation.discrete_moves + increment("discrete_moves"),
        periodic_updates=adaptation.periodic_updates + increment("periodic_updates"),
        periodic_moves=adaptation.periodic_moves + increment("periodic_moves"),
        wrap_crossings=adaptation.wrap_crossings + increment("wrap_crossings"),
        phantom_creations=adaptation.phantom_creations + increment("phantom_creations"),
        phantom_revalidations=adaptation.phantom_revalidations
        + increment("phantom_revalidations"),
        phantom_reuses=adaptation.phantom_reuses + increment("phantom_reuses"),
        fallback_draws=adaptation.fallback_draws + increment("fallback_draws"),
        fallback_acceptances=adaptation.fallback_acceptances
        + increment("fallback_acceptances"),
        contour_rejections=adaptation.contour_rejections
        + increment("contour_rejections"),
        proposal_failures=adaptation.proposal_failures + increment("proposal_failures"),
        dynamic_additions=adaptation.dynamic_additions + increment("dynamic_additions"),
    )


def _flow_log_density(position: Array, proposal: PreparedNestedProposalState, /) -> Array:
    coordinates = jsp.linalg.solve_triangular(
        proposal.flow_factor,
        position - proposal.flow_mean,
        lower=True,
    )
    dimension = int(position.size)
    return -0.5 * (
        dimension * jnp.log(2.0 * jnp.pi)
        + 2.0 * proposal.flow_log_determinant
        + jnp.sum(coordinates**2)
    )


def _write_prepared_checkpoint(
    destination: Path,
    state: PreparedNestedState,
    /,
    *,
    compatibility: dict[str, Any],
) -> None:
    arrays: dict[str, Any] = {}
    specification = pack_array_tree("prepared_state", state, arrays)
    write_checkpoint_archive(
        destination,
        kind=_CHECKPOINT_KIND,
        compatibility=compatibility,
        state={"prepared_state": specification},
        arrays=arrays,
    )


def _read_prepared_checkpoint(
    source: Path,
    template: PreparedNestedState,
    /,
    *,
    compatibility: dict[str, Any],
) -> PreparedNestedState:
    state, arrays = read_checkpoint_archive(
        source,
        kind=_CHECKPOINT_KIND,
        compatibility=compatibility,
    )
    if set(state) != {"prepared_state"}:
        raise CheckpointCorruptionError("Prepared nested checkpoint state is invalid.")
    restored = unpack_array_tree(state["prepared_state"], arrays, template)
    if restored.plan_id != template.plan_id:
        raise CheckpointCorruptionError(
            "Prepared nested checkpoint plan identity changed."
        )
    return restored


def execute_prepared_nested(
    problem,
    plan: NestedSamplingPlan,
    /,
    *,
    key: Array,
    remaining_evidence_tolerance: float,
    prior_position_sampler,
    checkpoint_path: str | Path | None,
    checkpoint_id: str | None,
    checkpoint_every: int,
    resume_from: str | Path | None,
):
    """Execute one exact-correction, fixed-capacity variable-live nested lifecycle."""
    from ._nested import (
        NESTED_SAMPLING_INNER_KERNEL_FAILURE,
        NESTED_SAMPLING_INVALID_LIKELIHOOD,
        NESTED_SAMPLING_LIKELIHOOD_PLATEAU,
        NESTED_SAMPLING_MAX_DEAD_POINTS,
        NESTED_SAMPLING_MAX_LIKELIHOOD_EVALUATIONS,
        NESTED_SAMPLING_NO_FINITE_LIVE_POINT,
        NESTED_SAMPLING_SUCCESS,
        NestedSamplingResult,
    )
    from ._nested_diagnostics import build_nested_diagnostics

    reference, unravel = ravel_pytree(problem.initial_position)
    dimension = int(reference.size)
    dtype = reference.dtype
    capacity = plan.capacity
    layout = _coordinate_layout(problem.initial_position, plan)
    smooth_indices = jnp.asarray(layout.smooth, dtype=jnp.int32)
    root_key = jnp.asarray(jr.key_data(key), dtype=jnp.uint32)
    tolerance = float(remaining_evidence_tolerance)
    interval = int(checkpoint_every)
    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else (Path(resume_from) if resume_from is not None else None)
    )
    if destination is not None and (checkpoint_id is None or not str(checkpoint_id)):
        raise ValueError("checkpoint_id is required for nested checkpointing.")

    sample_treedef = jax.tree_util.tree_structure(problem.initial_position)

    def sample_prior(sample_key: Array, count: int) -> Array:
        tree = (
            problem.parameter_space.sample_prior(sample_key, num_samples=count)
            if prior_position_sampler is None
            else prior_position_sampler(sample_key, count)
        )
        if jax.tree_util.tree_structure(tree) != sample_treedef:
            raise ValueError(
                "Nested prior sampler returned a different parameter PyTree."
            )
        leaves = jax.tree_util.tree_leaves(tree)
        matrix = jnp.concatenate(
            tuple(jnp.asarray(leaf).reshape((count, -1)) for leaf in leaves), axis=1
        )
        if matrix.shape != (count, dimension):
            raise ValueError("Nested prior sampler returned an incompatible event shape.")
        return matrix

    def evaluate_one(vector: Array) -> tuple[Array, Array]:
        position = unravel(vector)
        log_prior = problem.parameter_space.unconstrained_log_prior(position)
        log_likelihood = problem.log_likelihood(
            problem.parameter_space.constrain(position)
        )
        if jnp.shape(log_prior) != () or jnp.shape(log_likelihood) != ():
            raise ValueError("Nested prior and likelihood evaluations must be scalar.")
        return jnp.asarray(log_prior), jnp.asarray(log_likelihood)

    def validate_declared_coordinates(matrix: Array) -> None:
        for index, support, _masses in layout.finite:
            values = matrix[:, index]
            if not bool(jnp.all(jnp.any(values[:, None] == support[None, :], axis=-1))):
                raise ValueError(
                    "Prior samples violate a declared finite counting support."
                )
        for index, coordinate in layout.periodic:
            values = matrix[:, index]
            inside = (values >= coordinate.origin) & (
                values < coordinate.origin + coordinate.period
            )
            if not bool(jnp.all(inside)):
                raise ValueError("Prior samples violate a declared periodic interval.")

    proposal_template = _empty_proposal(plan, len(layout.smooth), dtype)
    phantom_template = PhantomNestedState.initialize(
        capacity.max_phantoms, dimension, dtype=dtype
    )
    false_live = jnp.zeros((capacity.max_live,), dtype=bool)
    template = PreparedNestedState(
        root_key=root_key,
        initial_log_likelihood=jnp.full((capacity.max_live,), -jnp.inf, dtype=dtype),
        live_positions=jnp.zeros((capacity.max_live, dimension), dtype=dtype),
        live_log_prior=jnp.full((capacity.max_live,), -jnp.inf, dtype=dtype),
        live_log_likelihood=jnp.full((capacity.max_live,), -jnp.inf, dtype=dtype),
        live_birth_log_likelihood=jnp.full((capacity.max_live,), -jnp.inf, dtype=dtype),
        live_lineage=jnp.zeros((capacity.max_live,), dtype=jnp.int32),
        live_mask=false_live,
        dead_positions=jnp.zeros((capacity.max_dead_points, dimension), dtype=dtype),
        dead_log_prior=jnp.full((capacity.max_dead_points,), -jnp.inf, dtype=dtype),
        dead_log_likelihood=jnp.full((capacity.max_dead_points,), -jnp.inf, dtype=dtype),
        dead_birth_log_likelihood=jnp.full(
            (capacity.max_dead_points,), -jnp.inf, dtype=dtype
        ),
        dead_log_weights=jnp.full((capacity.max_dead_points,), -jnp.inf, dtype=dtype),
        dead_log_prior_volume=jnp.full(
            (capacity.max_dead_points,), -jnp.inf, dtype=dtype
        ),
        dead_live_counts=jnp.zeros((capacity.max_dead_points,), dtype=jnp.int32),
        dead_batch_indices=jnp.zeros((capacity.max_dead_points,), dtype=jnp.int32),
        dead_lineage=jnp.zeros((capacity.max_dead_points,), dtype=jnp.int32),
        insertion_ranks=jnp.zeros((capacity.max_dead_points,), dtype=jnp.int32),
        inner_accepted=jnp.zeros((capacity.max_dead_points,), dtype=bool),
        proposal_attempts=jnp.zeros((capacity.max_dead_points,), dtype=jnp.int32),
        proposal_shrinkage=jnp.zeros((capacity.max_dead_points,), dtype=jnp.int32),
        phantom=phantom_template,
        proposal=proposal_template,
        adaptation=_empty_adaptation(dtype),
        log_prior_volume=jnp.asarray(0.0, dtype=dtype),
        dead_count=jnp.asarray(0, dtype=jnp.int32),
        likelihood_evaluations=jnp.asarray(0, dtype=jnp.int32),
        dynamic_batches=jnp.asarray(0, dtype=jnp.int32),
        step=jnp.asarray(0, dtype=jnp.int32),
        status=jnp.asarray(NESTED_SAMPLING_MAX_DEAD_POINTS, dtype=jnp.int32),
        finished=jnp.asarray(False),
        plan_id=plan.plan_id,
    )

    settings = {
        "plan_id": plan.plan_id,
        "remaining_evidence_tolerance": tolerance,
        "custom_prior_position_sampler": prior_position_sampler is not None,
        "root_key": [int(value) for value in jr.key_data(root_key).reshape(-1)],
    }
    compatibility = (
        checkpoint_compatibility(
            problem,
            checkpoint_id=str(checkpoint_id),
            settings=settings,
            gradient_probe=plan.proposal.gradient_guided,
        )
        if destination is not None
        else None
    )

    if resume_from is None:
        if capacity.max_likelihood_evaluations <= plan.initial_live:
            raise ValueError(
                "Prepared nested execution needs capacity for the deterministic "
                "likelihood revalidation beyond the initial live set."
            )
        initial = sample_prior(
            derive_key(root_key, _PREPARED_PRIOR, 0), plan.initial_live
        )
        validate_declared_coordinates(initial)
        initial_prior, initial_likelihood = jax.vmap(evaluate_one)(initial)
        probe_prior, probe_likelihood = evaluate_one(initial[0])
        if not bool(
            jnp.array_equal(probe_prior, initial_prior[0], equal_nan=True)
            & jnp.array_equal(probe_likelihood, initial_likelihood[0], equal_nan=True)
        ):
            raise ValueError(
                "Nested sampling requires a deterministic prior and likelihood."
            )
        invalid_initial = ~jnp.isfinite(initial_prior)
        invalid_initial |= jnp.isnan(initial_likelihood) | jnp.isposinf(
            initial_likelihood
        )
        live_mask = false_live.at[: plan.initial_live].set(True)
        state = eqx.tree_at(
            lambda value: (
                value.initial_log_likelihood,
                value.live_positions,
                value.live_log_prior,
                value.live_log_likelihood,
                value.live_lineage,
                value.live_mask,
                value.likelihood_evaluations,
                value.status,
                value.finished,
            ),
            template,
            (
                template.initial_log_likelihood.at[: plan.initial_live].set(
                    initial_likelihood
                ),
                template.live_positions.at[: plan.initial_live].set(initial),
                template.live_log_prior.at[: plan.initial_live].set(initial_prior),
                template.live_log_likelihood.at[: plan.initial_live].set(
                    initial_likelihood
                ),
                template.live_lineage.at[: plan.initial_live].set(
                    jnp.arange(plan.initial_live, dtype=jnp.int32)
                ),
                live_mask,
                jnp.asarray(plan.initial_live + 1, dtype=jnp.int32),
                jnp.asarray(
                    NESTED_SAMPLING_INVALID_LIKELIHOOD
                    if bool(jnp.any(invalid_initial))
                    else NESTED_SAMPLING_MAX_DEAD_POINTS,
                    dtype=jnp.int32,
                ),
                jnp.asarray(bool(jnp.any(invalid_initial))),
            ),
        )
        if not bool(jnp.any(jnp.isfinite(initial_likelihood))) and not bool(
            state.finished
        ):
            state = eqx.tree_at(
                lambda value: (value.status, value.finished),
                state,
                (
                    jnp.asarray(NESTED_SAMPLING_NO_FINITE_LIVE_POINT, dtype=jnp.int32),
                    jnp.asarray(True),
                ),
            )
    else:
        if compatibility is None:
            raise RuntimeError(
                "Prepared nested resume compatibility was not initialized."
            )
        state = _read_prepared_checkpoint(
            Path(resume_from), template, compatibility=compatibility
        )

    def slice_move(
        evaluator: _Evaluator,
        current: Array,
        current_prior: Array,
        current_likelihood: Array,
        threshold: Array,
        direction: Array,
        width: float,
        move_key: Array,
    ) -> tuple[Array, Array, Array, int, int, bool]:
        if not bool(jnp.any(direction != 0.0)):
            return current, current_prior, current_likelihood, 0, 0, False
        slice_key, allocation_key, bracket_key = jr.split(move_key, 3)
        log_slice = current_prior + jnp.log(
            jr.uniform(slice_key, (), minval=jnp.finfo(dtype).tiny, maxval=1.0)
        )
        offset = float(jr.uniform(bracket_key, (), minval=0.0, maxval=width))
        left, right = -offset, width - offset
        left_steps = int(
            jr.randint(
                allocation_key,
                (),
                minval=0,
                maxval=plan.proposal.maximum_attempts + 1,
            )
        )
        right_steps = plan.proposal.maximum_attempts - left_steps
        attempts = 0
        shrinkage = 0
        for _ in range(left_steps):
            candidate = current + left * direction
            candidate_prior, candidate_likelihood = evaluator.one(candidate)
            attempts += 1
            member = bool(
                jnp.isfinite(candidate_prior)
                & (candidate_likelihood > threshold)
                & (candidate_prior >= log_slice)
            )
            if evaluator.exhausted or evaluator.invalid or not member:
                break
            left -= width
        for _ in range(right_steps):
            candidate = current + right * direction
            candidate_prior, candidate_likelihood = evaluator.one(candidate)
            attempts += 1
            member = bool(
                jnp.isfinite(candidate_prior)
                & (candidate_likelihood > threshold)
                & (candidate_prior >= log_slice)
            )
            if evaluator.exhausted or evaluator.invalid or not member:
                break
            right += width
        for attempt in range(plan.proposal.maximum_attempts):
            proposal_key = derive_key(move_key, _PREPARED_BASE, attempt)
            distance = float(jr.uniform(proposal_key, (), minval=left, maxval=right))
            candidate = current + distance * direction
            candidate_prior, candidate_likelihood = evaluator.one(candidate)
            attempts += 1
            if evaluator.exhausted or evaluator.invalid:
                break
            member = bool(
                jnp.isfinite(candidate_prior)
                & (candidate_likelihood > threshold)
                & (candidate_prior >= log_slice)
            )
            if member:
                moved = bool(jnp.any(candidate != current))
                return (
                    candidate,
                    candidate_prior,
                    candidate_likelihood,
                    attempts,
                    shrinkage,
                    moved,
                )
            shrinkage += 1
            if distance < 0.0:
                left = distance
            else:
                right = distance
        return current, current_prior, current_likelihood, attempts, shrinkage, False

    def propose(
        current_state: PreparedNestedState,
        threshold: Array,
        *,
        excluded_slot: int,
        purpose: int,
    ) -> _ProposalOutcome:
        counts: dict[str, int] = {}
        evaluator = _Evaluator(
            evaluate_one,
            count=int(current_state.likelihood_evaluations),
            limit=capacity.max_likelihood_evaluations,
        )
        proposal_state = _prepare_proposal_geometry(current_state, plan, layout)
        active_indices = jnp.flatnonzero(
            current_state.live_mask & (jnp.arange(capacity.max_live) != excluded_slot),
            size=int(jnp.sum(current_state.live_mask)) - int(excluded_slot >= 0),
        )
        anchor_key = derive_key(
            current_state.root_key,
            _PREPARED_BASE,
            int(current_state.step),
            purpose,
            0,
        )
        anchor_local = int(jr.randint(anchor_key, (), 0, int(active_indices.size)))
        anchor = int(active_indices[anchor_local])
        position = current_state.live_positions[anchor]
        log_prior = current_state.live_log_prior[anchor]
        log_likelihood = current_state.live_log_likelihood[anchor]
        lineage = current_state.live_lineage[anchor]
        accepted_states: list[tuple[Array, Array, Array]] = []
        total_attempts = 0
        total_shrinkage = 0
        failed = False

        if layout.smooth:
            active_values = current_state.live_positions[current_state.live_mask][
                :, smooth_indices
            ]
            empirical_scale = float(jnp.mean(jnp.std(active_values, axis=0)))
            width = plan.proposal.slice_scale * max(
                empirical_scale,
                plan.proposal.gradient_step_size,
            )
            if plan.proposal.base == "hit-and-run":
                direction_key = derive_key(
                    current_state.root_key,
                    _PREPARED_BASE,
                    int(current_state.step),
                    purpose,
                    1,
                )
                smooth_direction = jr.normal(
                    direction_key, (len(layout.smooth),), dtype=dtype
                )
                norm = jnp.sqrt(jnp.sum(smooth_direction**2))
                direction = (
                    jnp.zeros((dimension,), dtype=dtype)
                    .at[smooth_indices]
                    .set(smooth_direction / norm)
                )
                (
                    position,
                    log_prior,
                    log_likelihood,
                    attempts,
                    shrinkage,
                    moved,
                ) = slice_move(
                    evaluator,
                    position,
                    log_prior,
                    log_likelihood,
                    threshold,
                    direction,
                    width,
                    derive_key(
                        current_state.root_key,
                        _PREPARED_BASE,
                        int(current_state.step),
                        purpose,
                        2,
                    ),
                )
                counts["base_attempts"] = attempts
                counts["base_acceptances"] = int(moved)
                total_attempts += attempts
                total_shrinkage += shrinkage
                if moved:
                    accepted_states.append((position, log_prior, log_likelihood))
            else:
                for local, coordinate in enumerate(layout.smooth):
                    direction = (
                        jnp.zeros((dimension,), dtype=dtype).at[coordinate].set(1.0)
                    )
                    (
                        position,
                        log_prior,
                        log_likelihood,
                        attempts,
                        shrinkage,
                        moved,
                    ) = slice_move(
                        evaluator,
                        position,
                        log_prior,
                        log_likelihood,
                        threshold,
                        direction,
                        width,
                        derive_key(
                            current_state.root_key,
                            _PREPARED_BASE,
                            int(current_state.step),
                            purpose,
                            10 + local,
                        ),
                    )
                    counts["base_attempts"] = counts.get("base_attempts", 0) + attempts
                    counts["base_acceptances"] = counts.get("base_acceptances", 0) + int(
                        moved
                    )
                    total_attempts += attempts
                    total_shrinkage += shrinkage
                    if moved:
                        accepted_states.append((position, log_prior, log_likelihood))
                    if evaluator.exhausted or evaluator.invalid:
                        break

        if plan.proposal.ellipsoid and not evaluator.exhausted and not evaluator.invalid:
            counts["ellipsoid_attempts"] = 1
            if not bool(jnp.any(proposal_state.ellipsoid_active)):
                failed = True
            else:
                bounds = EllipsoidalNestedBounds(
                    centers=proposal_state.ellipsoid_centers,
                    factors=proposal_state.ellipsoid_factors,
                    active=proposal_state.ellipsoid_active,
                    log_volumes=proposal_state.ellipsoid_log_volumes,
                    enlargement=plan.proposal.ellipsoid_enlargement,
                )
                ellipsoid_key = derive_key(
                    current_state.root_key,
                    _PREPARED_ELLIPSOID,
                    int(current_state.step),
                    purpose,
                )
                proposed_smooth, proposed_log_q = bounds.sample(ellipsoid_key)
                current_log_q = jnp.log(
                    jnp.asarray(
                        bounds.overlap_count(position[smooth_indices]), dtype=dtype
                    )
                ) - jsp.special.logsumexp(
                    jnp.where(bounds.active, bounds.log_volumes, -jnp.inf)
                )
                candidate = position.at[smooth_indices].set(proposed_smooth)
                candidate_prior, candidate_likelihood = evaluator.one(candidate)
                total_attempts += 1
                valid = bool(
                    jnp.isfinite(candidate_prior) & (candidate_likelihood > threshold)
                )
                if not valid:
                    counts["contour_rejections"] = counts.get("contour_rejections", 0) + 1
                if valid:
                    log_acceptance = (
                        candidate_prior - log_prior + current_log_q - proposed_log_q
                    )
                    accept_key = derive_key(ellipsoid_key, _PREPARED_ELLIPSOID, 1)
                    accept = bool(
                        jnp.log(
                            jr.uniform(
                                accept_key,
                                (),
                                minval=jnp.finfo(dtype).tiny,
                                maxval=1.0,
                            )
                        )
                        < jnp.minimum(log_acceptance, 0.0)
                    )
                    if accept:
                        position, log_prior, log_likelihood = (
                            candidate,
                            candidate_prior,
                            candidate_likelihood,
                        )
                        counts["ellipsoid_acceptances"] = 1
                        accepted_states.append((position, log_prior, log_likelihood))

        if (
            plan.proposal.learned_flow
            and not evaluator.exhausted
            and not evaluator.invalid
        ):
            counts["flow_attempts"] = 1
            if not bool(proposal_state.flow_active):
                failed = True
            else:
                flow_key = derive_key(
                    current_state.root_key,
                    _PREPARED_FLOW,
                    int(current_state.step),
                    purpose,
                )
                noise_key, accept_key = jr.split(flow_key)
                proposed_smooth = proposal_state.flow_mean + (
                    proposal_state.flow_factor
                    @ jr.normal(noise_key, (len(layout.smooth),), dtype=dtype)
                )
                proposed_log_q = _flow_log_density(proposed_smooth, proposal_state)
                current_log_q = _flow_log_density(
                    position[smooth_indices], proposal_state
                )
                candidate = position.at[smooth_indices].set(proposed_smooth)
                candidate_prior, candidate_likelihood = evaluator.one(candidate)
                total_attempts += 1
                valid = bool(
                    jnp.isfinite(candidate_prior) & (candidate_likelihood > threshold)
                )
                if not valid:
                    counts["contour_rejections"] = counts.get("contour_rejections", 0) + 1
                if valid:
                    log_acceptance = (
                        candidate_prior - log_prior + current_log_q - proposed_log_q
                    )
                    accept = bool(
                        jnp.log(
                            jr.uniform(
                                accept_key,
                                (),
                                minval=jnp.finfo(dtype).tiny,
                                maxval=1.0,
                            )
                        )
                        < jnp.minimum(log_acceptance, 0.0)
                    )
                    if accept:
                        position, log_prior, log_likelihood = (
                            candidate,
                            candidate_prior,
                            candidate_likelihood,
                        )
                        counts["flow_acceptances"] = 1
                        accepted_states.append((position, log_prior, log_likelihood))

        if (
            plan.proposal.gradient_guided
            and not evaluator.exhausted
            and not evaluator.invalid
        ):
            counts["gradient_attempts"] = 1
            step_size = plan.proposal.gradient_step_size
            barrier_scale = plan.proposal.gradient_barrier_scale

            def guided(value):
                prior_value, likelihood_value = evaluate_one(value)
                barrier = jax.nn.log_sigmoid(
                    (likelihood_value - threshold) / barrier_scale
                )
                return prior_value + barrier

            if evaluator.count + 3 > evaluator.limit:
                evaluator.exhausted = True
            else:
                evaluator.count += 1
                gradient = jax.grad(guided)(position)
                if not bool(jnp.all(jnp.isfinite(gradient[smooth_indices]))):
                    failed = True
                else:
                    gradient_key = derive_key(
                        current_state.root_key,
                        _PREPARED_GRADIENT,
                        int(current_state.step),
                        purpose,
                    )
                    noise_key, accept_key = jr.split(gradient_key)
                    mean = (
                        position[smooth_indices]
                        + 0.5 * step_size**2 * gradient[smooth_indices]
                    )
                    proposed_smooth = mean + step_size * jr.normal(
                        noise_key, (len(layout.smooth),), dtype=dtype
                    )
                    candidate = position.at[smooth_indices].set(proposed_smooth)
                    candidate_prior, candidate_likelihood = evaluator.one(candidate)
                    total_attempts += 1
                    valid = bool(
                        jnp.isfinite(candidate_prior) & (candidate_likelihood > threshold)
                    )
                    if not valid:
                        counts["contour_rejections"] = (
                            counts.get("contour_rejections", 0) + 1
                        )
                    else:
                        evaluator.count += 1
                        reverse_gradient = jax.grad(guided)(candidate)
                        if not bool(
                            jnp.all(jnp.isfinite(reverse_gradient[smooth_indices]))
                        ):
                            failed = True
                        else:
                            reverse_mean = (
                                proposed_smooth
                                + 0.5 * step_size**2 * (reverse_gradient[smooth_indices])
                            )
                            forward_error = (proposed_smooth - mean) / step_size
                            reverse_error = (
                                position[smooth_indices] - reverse_mean
                            ) / step_size
                            log_q_reverse_minus_forward = -0.5 * (
                                jnp.sum(reverse_error**2) - jnp.sum(forward_error**2)
                            )
                            log_acceptance = (
                                candidate_prior - log_prior + log_q_reverse_minus_forward
                            )
                            accept = bool(
                                jnp.log(
                                    jr.uniform(
                                        accept_key,
                                        (),
                                        minval=jnp.finfo(dtype).tiny,
                                        maxval=1.0,
                                    )
                                )
                                < jnp.minimum(log_acceptance, 0.0)
                            )
                            if accept:
                                position, log_prior, log_likelihood = (
                                    candidate,
                                    candidate_prior,
                                    candidate_likelihood,
                                )
                                counts["gradient_acceptances"] = 1
                                accepted_states.append(
                                    (position, log_prior, log_likelihood)
                                )

        if (
            plan.proposal.discrete_gibbs
            and not evaluator.exhausted
            and not evaluator.invalid
        ):
            for local, (coordinate, support, masses) in enumerate(layout.finite):
                counts["discrete_updates"] = counts.get("discrete_updates", 0) + 1
                candidates: list[Array] = []
                candidate_priors: list[Array] = []
                candidate_likelihoods: list[Array] = []
                for value in support:
                    candidate = position.at[coordinate].set(value)
                    candidate_prior, candidate_likelihood = evaluator.one(candidate)
                    total_attempts += 1
                    candidates.append(candidate)
                    candidate_priors.append(candidate_prior)
                    candidate_likelihoods.append(candidate_likelihood)
                    if evaluator.exhausted or evaluator.invalid:
                        break
                if evaluator.exhausted or evaluator.invalid:
                    break
                prior_values = jnp.stack(candidate_priors)
                likelihood_values = jnp.stack(candidate_likelihoods)
                conditional_offsets = prior_values - jnp.log(masses)
                if not bool(
                    jnp.allclose(
                        conditional_offsets,
                        conditional_offsets[0],
                        rtol=1e-5,
                        atol=1e-6,
                    )
                ):
                    failed = True
                    break
                valid = jnp.isfinite(prior_values) & (likelihood_values > threshold)
                if not bool(jnp.any(valid)):
                    failed = True
                    break
                gibbs_key = derive_key(
                    current_state.root_key,
                    _PREPARED_DISCRETE,
                    int(current_state.step),
                    purpose,
                    local,
                )
                selected = int(
                    jr.categorical(
                        gibbs_key,
                        jnp.where(valid, jnp.log(masses), -jnp.inf),
                    )
                )
                candidate = candidates[selected]
                moved = bool(candidate[coordinate] != position[coordinate])
                position = candidate
                log_prior = prior_values[selected]
                log_likelihood = likelihood_values[selected]
                if moved:
                    counts["discrete_moves"] = counts.get("discrete_moves", 0) + 1
                    accepted_states.append((position, log_prior, log_likelihood))

        if (
            plan.proposal.periodic_slice
            and not evaluator.exhausted
            and not evaluator.invalid
        ):
            for local, (coordinate, topology) in enumerate(layout.periodic):
                counts["periodic_updates"] = counts.get("periodic_updates", 0) + 1
                periodic_key = derive_key(
                    current_state.root_key,
                    _PREPARED_PERIODIC,
                    int(current_state.step),
                    purpose,
                    local,
                )
                slice_key = derive_key(periodic_key, _PREPARED_PERIODIC, 0x51)
                log_slice = log_prior + jnp.log(
                    jr.uniform(
                        slice_key,
                        (),
                        minval=jnp.finfo(dtype).tiny,
                        maxval=1.0,
                    )
                )
                left = -0.5 * topology.period
                right = 0.5 * topology.period
                origin_value = position[coordinate]
                moved = False
                for attempt in range(plan.proposal.maximum_attempts):
                    proposal_key = derive_key(periodic_key, _PREPARED_PERIODIC, attempt)
                    displacement = float(
                        jr.uniform(proposal_key, (), minval=left, maxval=right)
                    )
                    raw = origin_value + displacement
                    wrapped = topology.wrap(raw)
                    shortest_displacement = float(
                        topology.displacement(origin_value, wrapped)
                    )
                    counts["wrap_crossings"] = counts.get("wrap_crossings", 0) + int(
                        raw < topology.origin or raw >= topology.origin + topology.period
                    )
                    candidate = position.at[coordinate].set(wrapped)
                    candidate_prior, candidate_likelihood = evaluator.one(candidate)
                    total_attempts += 1
                    if evaluator.exhausted or evaluator.invalid:
                        break
                    valid = bool(
                        jnp.isfinite(candidate_prior)
                        & (candidate_likelihood > threshold)
                        & (candidate_prior >= log_slice)
                    )
                    if valid:
                        moved = bool(wrapped != origin_value)
                        position, log_prior, log_likelihood = (
                            candidate,
                            candidate_prior,
                            candidate_likelihood,
                        )
                        if moved:
                            counts["periodic_moves"] = counts.get("periodic_moves", 0) + 1
                            accepted_states.append((position, log_prior, log_likelihood))
                        break
                    total_shrinkage += 1
                    if shortest_displacement < 0.0:
                        left = shortest_displacement
                    else:
                        right = shortest_displacement

        moved_any = bool(jnp.any(position != current_state.live_positions[anchor]))
        if (
            (failed or not moved_any)
            and plan.proposal.rejection_fallback
            and not evaluator.exhausted
            and not evaluator.invalid
        ):
            fallback_found = False
            for attempt in range(plan.proposal.maximum_attempts):
                if evaluator.count >= evaluator.limit:
                    evaluator.exhausted = True
                    break
                candidate = sample_prior(
                    derive_key(
                        current_state.root_key,
                        _PREPARED_FALLBACK,
                        int(current_state.step),
                        purpose,
                        attempt,
                    ),
                    1,
                )[0]
                validate_declared_coordinates(candidate[None, :])
                candidate_prior, candidate_likelihood = evaluator.one(candidate)
                counts["fallback_draws"] = counts.get("fallback_draws", 0) + 1
                total_attempts += 1
                if evaluator.invalid:
                    break
                if bool(
                    jnp.isfinite(candidate_prior) & (candidate_likelihood > threshold)
                ):
                    position, log_prior, log_likelihood = (
                        candidate,
                        candidate_prior,
                        candidate_likelihood,
                    )
                    accepted_states.append((position, log_prior, log_likelihood))
                    counts["fallback_acceptances"] = (
                        counts.get("fallback_acceptances", 0) + 1
                    )
                    fallback_found = True
                    failed = False
                    moved_any = True
                    break
                counts["contour_rejections"] = counts.get("contour_rejections", 0) + 1
            if not fallback_found:
                failed = True

        phantom = current_state.phantom
        if plan.proposal.phantom_recycling and len(accepted_states) > 1:
            for candidate, _candidate_prior, candidate_likelihood in accepted_states[:-1]:
                phantom = phantom.add(
                    candidate,
                    log_likelihood=candidate_likelihood,
                    birth_log_likelihood=threshold,
                    proposal_epoch=current_state.step,
                    ancestry=lineage,
                )
                counts["phantom_creations"] = counts.get("phantom_creations", 0) + 1
        if failed:
            counts["proposal_failures"] = counts.get("proposal_failures", 0) + 1
        adaptation = _increment_adaptation(current_state.adaptation, counts)
        return _ProposalOutcome(
            position=position,
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            lineage=lineage,
            phantom=phantom,
            proposal=proposal_state,
            adaptation=adaptation,
            attempts=total_attempts,
            likelihood_evaluations=evaluator.count,
            shrinkage=total_shrinkage,
            moved=moved_any,
            budget_exhausted=evaluator.exhausted,
            invalid_likelihood=evaluator.invalid,
            failed=failed,
        )

    while not bool(state.finished):
        dead_count = int(state.dead_count)
        if dead_count >= capacity.max_dead_points:
            state = eqx.tree_at(
                lambda value: (value.status, value.finished),
                state,
                (
                    jnp.asarray(NESTED_SAMPLING_MAX_DEAD_POINTS, dtype=jnp.int32),
                    jnp.asarray(True),
                ),
            )
            break
        if int(state.likelihood_evaluations) >= capacity.max_likelihood_evaluations:
            state = eqx.tree_at(
                lambda value: (value.status, value.finished),
                state,
                (
                    jnp.asarray(
                        NESTED_SAMPLING_MAX_LIKELIHOOD_EVALUATIONS, dtype=jnp.int32
                    ),
                    jnp.asarray(True),
                ),
            )
            break

        active_count = int(jnp.sum(state.live_mask))
        active_likelihood = jnp.where(state.live_mask, state.live_log_likelihood, jnp.inf)
        finite_active = state.live_log_likelihood[state.live_mask]
        if bool(jnp.ptp(finite_active) == 0.0):
            state = eqx.tree_at(
                lambda value: (value.status, value.finished),
                state,
                (
                    jnp.asarray(NESTED_SAMPLING_LIKELIHOOD_PLATEAU, dtype=jnp.int32),
                    jnp.asarray(True),
                ),
            )
            break
        worst = int(jnp.argmin(active_likelihood))
        threshold = state.live_log_likelihood[worst]

        phantom = state.phantom
        replacement: _ProposalOutcome | None = None
        if plan.proposal.phantom_recycling:
            while True:
                eligible = phantom.eligible(threshold)
                if not bool(jnp.any(eligible)):
                    break
                epochs = jnp.where(
                    eligible, phantom.proposal_epoch, jnp.iinfo(jnp.int32).max
                )
                selected = int(jnp.argmin(epochs))
                phantom = eqx.tree_at(
                    lambda value: value.mask,
                    phantom,
                    phantom.mask.at[selected].set(False),
                )
                evaluator = _Evaluator(
                    evaluate_one,
                    count=int(state.likelihood_evaluations),
                    limit=capacity.max_likelihood_evaluations,
                )
                revalidated_prior, revalidated_likelihood = evaluator.one(
                    phantom.positions[selected]
                )
                counts = {"phantom_revalidations": 1}
                likelihood_consistent = bool(
                    jnp.array_equal(
                        revalidated_likelihood,
                        phantom.log_likelihood[selected],
                        equal_nan=True,
                    )
                )
                evaluator.invalid = evaluator.invalid or not likelihood_consistent
                valid = bool(
                    likelihood_consistent
                    & jnp.isfinite(revalidated_prior)
                    & (revalidated_likelihood > threshold)
                    & (phantom.birth_log_likelihood[selected] <= threshold)
                )
                if valid:
                    counts["phantom_reuses"] = 1
                    replacement = _ProposalOutcome(
                        position=phantom.positions[selected],
                        log_prior=revalidated_prior,
                        log_likelihood=revalidated_likelihood,
                        lineage=phantom.ancestry[selected],
                        phantom=phantom,
                        proposal=state.proposal,
                        adaptation=_increment_adaptation(state.adaptation, counts),
                        attempts=1,
                        likelihood_evaluations=evaluator.count,
                        shrinkage=0,
                        moved=True,
                        budget_exhausted=evaluator.exhausted,
                        invalid_likelihood=evaluator.invalid,
                        failed=False,
                    )
                    break
                state = eqx.tree_at(
                    lambda value: (
                        value.phantom,
                        value.adaptation,
                        value.likelihood_evaluations,
                    ),
                    state,
                    (
                        phantom,
                        _increment_adaptation(state.adaptation, counts),
                        jnp.asarray(evaluator.count, dtype=jnp.int32),
                    ),
                )
                if evaluator.invalid or evaluator.exhausted:
                    replacement = _ProposalOutcome(
                        position=state.live_positions[worst],
                        log_prior=state.live_log_prior[worst],
                        log_likelihood=state.live_log_likelihood[worst],
                        lineage=state.live_lineage[worst],
                        phantom=phantom,
                        proposal=state.proposal,
                        adaptation=state.adaptation,
                        attempts=1,
                        likelihood_evaluations=evaluator.count,
                        shrinkage=0,
                        moved=False,
                        budget_exhausted=evaluator.exhausted,
                        invalid_likelihood=evaluator.invalid,
                        failed=True,
                    )
                    break

        if replacement is None:
            replacement = propose(state, threshold, excluded_slot=worst, purpose=0)
        evaluation_count = replacement.likelihood_evaluations
        if replacement.invalid_likelihood:
            state = eqx.tree_at(
                lambda value: (
                    value.phantom,
                    value.proposal,
                    value.adaptation,
                    value.likelihood_evaluations,
                    value.status,
                    value.finished,
                ),
                state,
                (
                    replacement.phantom,
                    replacement.proposal,
                    replacement.adaptation,
                    jnp.asarray(evaluation_count, dtype=jnp.int32),
                    jnp.asarray(NESTED_SAMPLING_INVALID_LIKELIHOOD, dtype=jnp.int32),
                    jnp.asarray(True),
                ),
            )
            break
        if replacement.budget_exhausted:
            state = eqx.tree_at(
                lambda value: (
                    value.phantom,
                    value.proposal,
                    value.adaptation,
                    value.likelihood_evaluations,
                    value.status,
                    value.finished,
                ),
                state,
                (
                    replacement.phantom,
                    replacement.proposal,
                    replacement.adaptation,
                    jnp.asarray(capacity.max_likelihood_evaluations, dtype=jnp.int32),
                    jnp.asarray(
                        NESTED_SAMPLING_MAX_LIKELIHOOD_EVALUATIONS, dtype=jnp.int32
                    ),
                    jnp.asarray(True),
                ),
            )
            break
        if replacement.failed or not bool(replacement.log_likelihood > threshold):
            state = eqx.tree_at(
                lambda value: (
                    value.phantom,
                    value.proposal,
                    value.adaptation,
                    value.likelihood_evaluations,
                    value.status,
                    value.finished,
                ),
                state,
                (
                    replacement.phantom,
                    replacement.proposal,
                    replacement.adaptation,
                    jnp.asarray(evaluation_count, dtype=jnp.int32),
                    jnp.asarray(NESTED_SAMPLING_INNER_KERNEL_FAILURE, dtype=jnp.int32),
                    jnp.asarray(True),
                ),
            )
            break

        next_log_volume = state.log_prior_volume - 1.0 / active_count
        log_shell = state.log_prior_volume + jnp.log1p(
            -jnp.exp(next_log_volume - state.log_prior_volume)
        )
        new_likelihood = state.live_log_likelihood.at[worst].set(
            replacement.log_likelihood
        )
        insertion_rank = jnp.sum(
            jnp.where(
                state.live_mask,
                new_likelihood < replacement.log_likelihood,
                False,
            )
        )
        index = int(state.dead_count)
        state = eqx.tree_at(
            lambda value: (
                value.live_positions,
                value.live_log_prior,
                value.live_log_likelihood,
                value.live_birth_log_likelihood,
                value.live_lineage,
                value.dead_positions,
                value.dead_log_prior,
                value.dead_log_likelihood,
                value.dead_birth_log_likelihood,
                value.dead_log_weights,
                value.dead_log_prior_volume,
                value.dead_live_counts,
                value.dead_batch_indices,
                value.dead_lineage,
                value.insertion_ranks,
                value.inner_accepted,
                value.proposal_attempts,
                value.proposal_shrinkage,
                value.phantom,
                value.proposal,
                value.adaptation,
                value.log_prior_volume,
                value.dead_count,
                value.likelihood_evaluations,
                value.step,
            ),
            state,
            (
                state.live_positions.at[worst].set(replacement.position),
                state.live_log_prior.at[worst].set(replacement.log_prior),
                new_likelihood,
                state.live_birth_log_likelihood.at[worst].set(threshold),
                state.live_lineage.at[worst].set(replacement.lineage),
                state.dead_positions.at[index].set(state.live_positions[worst]),
                state.dead_log_prior.at[index].set(state.live_log_prior[worst]),
                state.dead_log_likelihood.at[index].set(threshold),
                state.dead_birth_log_likelihood.at[index].set(
                    state.live_birth_log_likelihood[worst]
                ),
                state.dead_log_weights.at[index].set(log_shell + threshold),
                state.dead_log_prior_volume.at[index].set(log_shell),
                state.dead_live_counts.at[index].set(active_count),
                state.dead_batch_indices.at[index].set(state.dynamic_batches),
                state.dead_lineage.at[index].set(state.live_lineage[worst]),
                state.insertion_ranks.at[index].set(insertion_rank),
                state.inner_accepted.at[index].set(replacement.moved),
                state.proposal_attempts.at[index].set(replacement.attempts),
                state.proposal_shrinkage.at[index].set(replacement.shrinkage),
                replacement.phantom,
                replacement.proposal,
                replacement.adaptation,
                next_log_volume,
                state.dead_count + 1,
                jnp.asarray(evaluation_count, dtype=jnp.int32),
                state.step + 1,
            ),
        )

        if plan.dynamic is not None and (
            int(state.dead_count) >= plan.dynamic.pilot_dead_points
            and (int(state.dead_count) - plan.dynamic.pilot_dead_points)
            % plan.dynamic.allocation_cadence
            == 0
            and int(state.dynamic_batches) < capacity.max_dynamic_batches
        ):
            free = capacity.max_live - int(jnp.sum(state.live_mask))
            additions = min(plan.dynamic.additional_live_per_batch, free)
            added = 0
            for local in range(additions):
                extra = propose(state, threshold, excluded_slot=-1, purpose=100 + local)
                state = eqx.tree_at(
                    lambda value: (
                        value.phantom,
                        value.proposal,
                        value.adaptation,
                        value.likelihood_evaluations,
                    ),
                    state,
                    (
                        extra.phantom,
                        extra.proposal,
                        extra.adaptation,
                        jnp.asarray(extra.likelihood_evaluations, dtype=jnp.int32),
                    ),
                )
                if extra.invalid_likelihood:
                    state = eqx.tree_at(
                        lambda value: (value.status, value.finished),
                        state,
                        (
                            jnp.asarray(
                                NESTED_SAMPLING_INVALID_LIKELIHOOD, dtype=jnp.int32
                            ),
                            jnp.asarray(True),
                        ),
                    )
                    break
                if extra.budget_exhausted:
                    state = eqx.tree_at(
                        lambda value: (
                            value.likelihood_evaluations,
                            value.status,
                            value.finished,
                        ),
                        state,
                        (
                            jnp.asarray(
                                capacity.max_likelihood_evaluations, dtype=jnp.int32
                            ),
                            jnp.asarray(
                                NESTED_SAMPLING_MAX_LIKELIHOOD_EVALUATIONS,
                                dtype=jnp.int32,
                            ),
                            jnp.asarray(True),
                        ),
                    )
                    break
                if extra.failed or not bool(extra.log_likelihood > threshold):
                    state = eqx.tree_at(
                        lambda value: (value.status, value.finished),
                        state,
                        (
                            jnp.asarray(
                                NESTED_SAMPLING_INNER_KERNEL_FAILURE, dtype=jnp.int32
                            ),
                            jnp.asarray(True),
                        ),
                    )
                    break
                slot = int(jnp.flatnonzero(~state.live_mask, size=1)[0])
                state = eqx.tree_at(
                    lambda value: (
                        value.live_positions,
                        value.live_log_prior,
                        value.live_log_likelihood,
                        value.live_birth_log_likelihood,
                        value.live_lineage,
                        value.live_mask,
                    ),
                    state,
                    (
                        state.live_positions.at[slot].set(extra.position),
                        state.live_log_prior.at[slot].set(extra.log_prior),
                        state.live_log_likelihood.at[slot].set(extra.log_likelihood),
                        state.live_birth_log_likelihood.at[slot].set(threshold),
                        state.live_lineage.at[slot].set(extra.lineage),
                        state.live_mask.at[slot].set(True),
                    ),
                )
                added += 1
            if added > 0:
                adaptation = _increment_adaptation(
                    state.adaptation, {"dynamic_additions": added}
                )
                state = eqx.tree_at(
                    lambda value: (value.dynamic_batches, value.adaptation),
                    state,
                    (state.dynamic_batches + 1, adaptation),
                )

        if bool(state.finished):
            break
        dead_weights = state.dead_log_weights[: int(state.dead_count)]
        maximum_live_likelihood = jnp.max(
            jnp.where(state.live_mask, state.live_log_likelihood, -jnp.inf)
        )
        dead_log_evidence = jsp.special.logsumexp(dead_weights)
        remaining_log_evidence = state.log_prior_volume + maximum_live_likelihood
        remaining_fraction = jax.nn.sigmoid(remaining_log_evidence - dead_log_evidence)
        if float(remaining_fraction) <= tolerance:
            state = eqx.tree_at(
                lambda value: (value.status, value.finished),
                state,
                (
                    jnp.asarray(NESTED_SAMPLING_SUCCESS, dtype=jnp.int32),
                    jnp.asarray(True),
                ),
            )
        elif int(state.likelihood_evaluations) >= capacity.max_likelihood_evaluations:
            state = eqx.tree_at(
                lambda value: (value.status, value.finished),
                state,
                (
                    jnp.asarray(
                        NESTED_SAMPLING_MAX_LIKELIHOOD_EVALUATIONS, dtype=jnp.int32
                    ),
                    jnp.asarray(True),
                ),
            )

        if (
            destination is not None
            and compatibility is not None
            and int(state.dead_count) % interval == 0
        ):
            _write_prepared_checkpoint(destination, state, compatibility=compatibility)

    if destination is not None and compatibility is not None:
        _write_prepared_checkpoint(destination, state, compatibility=compatibility)

    dead_count = int(state.dead_count)
    active_indices = jnp.flatnonzero(state.live_mask, size=int(jnp.sum(state.live_mask)))
    active_indices = active_indices[
        jnp.argsort(state.live_log_likelihood[active_indices])
    ]
    final_positions = state.live_positions[active_indices]
    final_prior = state.live_log_prior[active_indices]
    final_likelihood = state.live_log_likelihood[active_indices]
    final_birth = state.live_birth_log_likelihood[active_indices]
    final_lineage = state.live_lineage[active_indices]
    final_count = int(active_indices.size)
    final_log_weights = state.log_prior_volume - jnp.log(final_count) + final_likelihood
    dead_positions = state.dead_positions[:dead_count]
    dead_prior = state.dead_log_prior[:dead_count]
    dead_likelihood = state.dead_log_likelihood[:dead_count]
    dead_birth = state.dead_birth_log_likelihood[:dead_count]
    all_positions = jnp.concatenate((dead_positions, final_positions), axis=0)
    all_prior = jnp.concatenate((dead_prior, final_prior), axis=0)
    all_likelihood = jnp.concatenate((dead_likelihood, final_likelihood), axis=0)
    all_birth = jnp.concatenate((dead_birth, final_birth), axis=0)
    log_weights = jnp.concatenate(
        (state.dead_log_weights[:dead_count], final_log_weights), axis=0
    )
    log_evidence = jsp.special.logsumexp(log_weights)
    posterior_log_weights = jnp.where(
        jnp.isfinite(log_evidence), log_weights - log_evidence, -jnp.inf
    )
    weights = jnp.where(
        jnp.isfinite(posterior_log_weights), jnp.exp(posterior_log_weights), 0.0
    )
    live_counts = jnp.concatenate(
        (
            state.dead_live_counts[:dead_count],
            jnp.full((final_count,), final_count, dtype=jnp.int32),
        )
    )
    batch_indices = jnp.concatenate(
        (
            state.dead_batch_indices[:dead_count],
            jnp.full((final_count,), state.dynamic_batches, dtype=jnp.int32),
        )
    )
    sample_ids = jnp.concatenate((state.dead_lineage[:dead_count], final_lineage), axis=0)
    samples_tree = jax.vmap(unravel)(all_positions)
    constrained = problem.parameter_space.constrain(samples_tree)
    terminal_valid = (
        jnp.isfinite(log_evidence)
        & (state.status != NESTED_SAMPLING_NO_FINITE_LIVE_POINT)
        & (state.status != NESTED_SAMPLING_INVALID_LIKELIHOOD)
        & (state.status != NESTED_SAMPLING_INNER_KERNEL_FAILURE)
    )
    diagnostics = build_nested_diagnostics(
        dead_log_likelihood=dead_likelihood,
        dead_birth_log_likelihood=dead_birth,
        insertion_ranks=state.insertion_ranks[:dead_count],
        inner_accepted=state.inner_accepted[:dead_count, None],
        num_expansions=state.proposal_attempts[:dead_count, None],
        num_shrink=state.proposal_shrinkage[:dead_count, None],
        max_expansions=plan.proposal.maximum_attempts,
        max_shrinkage=plan.proposal.maximum_attempts,
        initial_log_likelihood=state.initial_log_likelihood[: plan.initial_live],
        sample_ids=sample_ids,
        posterior_log_weights=posterior_log_weights,
        num_live=capacity.max_live,
        quadrature_valid=terminal_valid,
        final_live_positions=jax.vmap(unravel)(final_positions),
    )
    information = jnp.sum(
        jnp.where(weights > 0.0, weights * (all_likelihood - log_evidence), 0.0)
    )
    effective_sample_size = jnp.where(
        jnp.sum(weights**2) > 0.0, 1.0 / jnp.sum(weights**2), 0.0
    )
    remaining_log_evidence = state.log_prior_volume + jnp.max(final_likelihood)
    remaining_fraction = jnp.where(
        jnp.isfinite(log_evidence),
        jax.nn.sigmoid(remaining_log_evidence - log_evidence),
        1.0,
    )
    method_parts = [plan.proposal.base]
    if plan.proposal.ellipsoid:
        method_parts.append("ellipsoid-independence-mh")
    if plan.proposal.learned_flow:
        method_parts.append("frozen-flow-independence-mh")
    if plan.proposal.gradient_guided:
        method_parts.append("gradient-mala")
    if plan.proposal.discrete_gibbs:
        method_parts.append("finite-gibbs")
    if plan.proposal.periodic_slice:
        method_parts.append("periodic-slice")
    if plan.proposal.phantom_recycling:
        method_parts.append("phantom-recycling")
    if plan.proposal.rejection_fallback:
        method_parts.append("exact-rejection-fallback")
    return NestedSamplingResult(
        problem=problem,
        samples=constrained,
        unconstrained_samples=samples_tree,
        log_prior=all_prior,
        log_likelihood=all_likelihood,
        birth_log_likelihood=all_birth,
        posterior_log_weights=posterior_log_weights,
        log_prior_volume=jnp.concatenate(
            (
                state.dead_log_prior_volume[:dead_count],
                jnp.full((final_count,), state.log_prior_volume),
            )
        ),
        live_counts=live_counts,
        sample_ids=sample_ids,
        batch_indices=batch_indices,
        log_evidence=log_evidence,
        log_evidence_replicates=jnp.asarray([log_evidence]),
        log_evidence_shrinkage_std=jnp.asarray(0.0, dtype=dtype),
        information=information,
        posterior_effective_sample_size=effective_sample_size,
        remaining_log_evidence=remaining_log_evidence,
        remaining_evidence_fraction=remaining_fraction,
        final_state=state,
        diagnostics=diagnostics,
        root_key=state.root_key,
        status=state.status,
        valid=terminal_valid & diagnostics.constraints_satisfied,
        num_live=final_count,
        num_dead=dead_count,
        num_likelihood_evaluations=int(state.likelihood_evaluations),
        num_inner_steps=plan.proposal.maximum_attempts,
        num_delete=1,
        method="prepared:" + "+".join(method_parts),
        duration_seconds=0.0,
    )


__all__ = [
    "PreparedNestedAdaptationState",
    "PreparedNestedProposalState",
    "PreparedNestedState",
    "execute_prepared_nested",
]
