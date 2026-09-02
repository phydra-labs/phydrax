#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._core import TensorTrain
from ._local import regularized_least_squares


class TTCrossPlan(StrictModule):
    """Static resource and accuracy policy for deterministic two-site TT cross."""

    mode_sizes: tuple[int, ...] = eqx.field(static=True)
    max_rank: int = eqx.field(static=True)
    sweeps: int = eqx.field(static=True)
    evaluation_budget: int = eqx.field(static=True)
    holdout_count: int = eqx.field(static=True)
    max_local_unknowns: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_sizes: Sequence[int],
        /,
        *,
        max_rank: int,
        sweeps: int,
        evaluation_budget: int,
        holdout_count: int,
        max_local_unknowns: int,
        regularization: float,
        relative_tolerance: float,
    ):
        modes = tuple(int(size) for size in mode_sizes)
        rank = int(max_rank)
        sweep_count = int(sweeps)
        budget = int(evaluation_budget)
        holdout = int(holdout_count)
        local_limit = int(max_local_unknowns)
        ridge = float(regularization)
        tolerance = float(relative_tolerance)
        total = prod(modes) if modes else 0
        if len(modes) < 2 or any(size <= 0 for size in modes):
            raise ValueError("TT cross requires at least two positive modes.")
        if rank <= 0 or sweep_count <= 0 or budget <= 0 or budget > total:
            raise ValueError(
                "TT cross ranks, sweeps, and evaluation budget must be feasible."
            )
        if holdout <= 0 or holdout >= budget:
            raise ValueError(
                "TT cross holdout_count must be between zero and its budget."
            )
        if local_limit <= 0 or ridge <= 0.0 or tolerance < 0.0:
            raise ValueError(
                "TT cross local budget, regularization, and tolerance are invalid."
            )
        self.mode_sizes = modes
        self.max_rank = rank
        self.sweeps = sweep_count
        self.evaluation_budget = budget
        self.holdout_count = holdout
        self.max_local_unknowns = local_limit
        self.regularization = ridge
        self.relative_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-site-tt-cross-plan",
                "modes": modes,
                "max_rank": rank,
                "sweeps": sweep_count,
                "evaluation_budget": budget,
                "holdout_count": holdout,
                "max_local_unknowns": local_limit,
                "regularization": ridge,
                "relative_tolerance": tolerance,
            }
        )


class TTCrossEvidence(StrictModule):
    """Auditable evaluations, residual pivots, and held-out error estimator."""

    evaluation_indices: Array
    evaluation_values: Array
    pivot_indices: Array
    training_relative_errors: Array
    holdout_root_mean_square_error: Array
    holdout_relative_error_estimator: Array
    holdout_maximum_absolute_error: Array
    evaluations_used: int = eqx.field(static=True)
    holdout_count: int = eqx.field(static=True)
    estimator_is_guarantee: bool = eqx.field(static=True)

    def __init__(
        self,
        evaluation_indices: Array,
        evaluation_values: Array,
        pivot_indices: Array,
        training_relative_errors: Array,
        holdout_root_mean_square_error: Array,
        holdout_relative_error_estimator: Array,
        holdout_maximum_absolute_error: Array,
        /,
        *,
        holdout_count: int,
    ):
        indices = jnp.asarray(evaluation_indices, dtype=jnp.int32)
        values = jnp.asarray(evaluation_values)
        pivots = jnp.asarray(pivot_indices, dtype=jnp.int32)
        errors = jnp.asarray(training_relative_errors)
        if indices.ndim != 2 or values.shape != (indices.shape[0],):
            raise ValueError("TT cross evaluation evidence has inconsistent shapes.")
        if pivots.ndim != 2 or pivots.shape[1] != indices.shape[1] or errors.ndim != 1:
            raise ValueError("TT cross pivot or sweep evidence has inconsistent shape.")
        self.evaluation_indices = indices
        self.evaluation_values = values
        self.pivot_indices = pivots
        self.training_relative_errors = errors
        self.holdout_root_mean_square_error = jnp.asarray(holdout_root_mean_square_error)
        self.holdout_relative_error_estimator = jnp.asarray(
            holdout_relative_error_estimator
        )
        self.holdout_maximum_absolute_error = jnp.asarray(holdout_maximum_absolute_error)
        self.evaluations_used = int(indices.shape[0])
        self.holdout_count = int(holdout_count)
        self.estimator_is_guarantee = False


class TTCrossResult(StrictModule):
    tensor: TensorTrain
    evidence: TTCrossEvidence
    converged: bool = eqx.field(static=True)
    status: str = eqx.field(static=True)

    def __init__(
        self,
        tensor: TensorTrain,
        evidence: TTCrossEvidence,
        converged: bool,
        /,
    ):
        self.tensor = tensor
        self.evidence = evidence
        self.converged = bool(converged)
        self.status = "converged" if self.converged else "holdout_tolerance_not_met"


def _permuted_indices(mode_sizes: tuple[int, ...], count: int, /) -> Array:
    """Place a deterministic rank-one cross first, then fill lexicographically."""
    anchor = (0,) * len(mode_sizes)
    selected = [anchor]
    for axis, size in enumerate(mode_sizes):
        for coordinate in range(1, size):
            point = [0] * len(mode_sizes)
            point[axis] = coordinate
            selected.append(tuple(point))
    selected_set = set(selected)
    for point in np.ndindex(mode_sizes):
        if point not in selected_set:
            selected.append(tuple(int(value) for value in point))
        if len(selected) >= count:
            break
    return jnp.asarray(selected[:count], dtype=jnp.int32)


def _rank_one_cross(
    mode_sizes: tuple[int, ...],
    values: Array,
    /,
) -> TensorTrain | None:
    anchor = values[0]
    if not bool(np.asarray(jnp.isfinite(anchor) & (jnp.abs(anchor) > 0.0))):
        return None
    vectors = []
    cursor = 1
    for size in mode_sizes:
        vector = jnp.empty((size,), dtype=values.dtype).at[0].set(anchor)
        if size > 1:
            vector = vector.at[1:].set(values[cursor : cursor + size - 1])
        vectors.append(vector)
        cursor += size - 1
    cores = []
    for axis, vector in enumerate(vectors):
        normalized = vector if axis == len(vectors) - 1 else vector / anchor
        cores.append(normalized[None, :, None])
    return TensorTrain(tuple(cores))


def _initial_train(
    mode_sizes: tuple[int, ...], max_rank: int, mean: Array, dtype, /
) -> TensorTrain:
    total = prod(mode_sizes)
    ranks = [1]
    prefix = 1
    for cut, size in enumerate(mode_sizes[:-1]):
        prefix *= size
        suffix = total // prefix
        ranks.append(min(max_rank, prefix, suffix))
    ranks.append(1)
    cores = []
    for axis, size in enumerate(mode_sizes):
        shape = (ranks[axis], size, ranks[axis + 1])
        phase = jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape)
        core = (1.0e-3 * jnp.sin(phase + axis + 1)).astype(dtype)
        core = core.at[0, :, 0].set(jnp.ones((size,), dtype=dtype))
        cores.append(core)
    cores[-1] = cores[-1].at[0, :, 0].set(jnp.asarray(mean, dtype=dtype))
    return TensorTrain(tuple(cores))


def _left_samples(tensor: TensorTrain, points: Array, stop: int, /) -> Array:
    def one(point):
        value = jnp.ones((1,), dtype=tensor.dtype)
        for axis in range(stop):
            value = ein.contract("a,ab->b", value, tensor.cores[axis][:, point[axis], :])
        return value

    return jax.vmap(one)(points)


def _right_samples(tensor: TensorTrain, points: Array, start: int, /) -> Array:
    def one(point):
        value = jnp.ones((1,), dtype=tensor.dtype)
        for axis in range(tensor.order - 1, start - 1, -1):
            value = ein.contract("ab,b->a", tensor.cores[axis][:, point[axis], :], value)
        return value

    return jax.vmap(one)(points)


def _two_site_design(tensor: TensorTrain, points: Array, axis: int, /) -> Array:
    left = _left_samples(tensor, points, axis)
    right = _right_samples(tensor, points, axis + 2)
    first = jax.nn.one_hot(points[:, axis], tensor.mode_sizes[axis], dtype=tensor.dtype)
    second = jax.nn.one_hot(
        points[:, axis + 1], tensor.mode_sizes[axis + 1], dtype=tensor.dtype
    )
    frame = ein.contract("sa,sp,sq,sb->sapqb", left, first, second, right)
    return frame.reshape((points.shape[0], -1))


def _update_pair(
    tensor: TensorTrain,
    points: Array,
    values: Array,
    axis: int,
    plan: TTCrossPlan,
    /,
) -> TensorTrain:
    design = _two_site_design(tensor, points, axis)
    if design.shape[1] > plan.max_local_unknowns:
        raise ValueError(
            f"TT cross local pair needs {design.shape[1]} unknowns, exceeding "
            f"budget {plan.max_local_unknowns}."
        )
    local = regularized_least_squares(design, values, plan.regularization)
    left_rank = tensor.cores[axis].shape[0]
    right_rank = tensor.cores[axis + 1].shape[2]
    first_size = tensor.mode_sizes[axis]
    second_size = tensor.mode_sizes[axis + 1]
    matrix = local.reshape((left_rank * first_size, second_size * right_rank))
    left, singular_values, right = jnp.linalg.svd(matrix, full_matrices=False)
    rank = min(plan.max_rank, left.shape[1])
    first_core = left[:, :rank].reshape((left_rank, first_size, rank))
    second_core = (singular_values[:rank, None] * right[:rank, :]).reshape(
        (rank, second_size, right_rank)
    )
    cores = list(tensor.cores)
    cores[axis] = first_core
    cores[axis + 1] = second_core
    return TensorTrain(tuple(cores))


def tensor_train_cross(
    evaluator: Callable[[Array], Array],
    plan: TTCrossPlan,
    /,
) -> TTCrossResult:
    """Fit by deterministic bounded two-site sweeps and report held-out estimates."""
    if not callable(evaluator):
        raise TypeError("TT cross evaluator must be callable on batched integer indices.")
    indices = _permuted_indices(plan.mode_sizes, plan.evaluation_budget)
    values = jnp.asarray(evaluator(indices))
    if values.shape != (plan.evaluation_budget,):
        raise ValueError("TT cross evaluator must return one scalar per requested index.")
    training_count = plan.evaluation_budget - plan.holdout_count
    training_indices = indices[:training_count]
    training_values = values[:training_count]
    holdout_indices = indices[training_count:]
    holdout_values = values[training_count:]
    rank_one = _rank_one_cross(plan.mode_sizes, training_values)
    if rank_one is not None:
        rank_one_holdout = rank_one.evaluate(holdout_indices)
        rank_one_residual = rank_one_holdout - holdout_values
        rank_one_scale = jnp.sqrt(jnp.mean(jnp.abs(holdout_values) ** 2))
        rank_one_relative = jnp.sqrt(
            jnp.mean(jnp.abs(rank_one_residual) ** 2)
        ) / jnp.where(rank_one_scale > 0, rank_one_scale, 1)
        if bool(np.asarray(rank_one_relative <= plan.relative_tolerance)):
            training_residual = rank_one.evaluate(training_indices) - training_values
            training_scale = jnp.sqrt(jnp.mean(jnp.abs(training_values) ** 2))
            training_relative = jnp.sqrt(
                jnp.mean(jnp.abs(training_residual) ** 2)
            ) / jnp.where(training_scale > 0, training_scale, 1)
            pivot = training_indices[jnp.argmax(jnp.abs(training_residual))]
            pivot_count = plan.sweeps * 2 * (len(plan.mode_sizes) - 1)
            evidence = TTCrossEvidence(
                indices,
                values,
                jnp.broadcast_to(pivot, (pivot_count, len(plan.mode_sizes))),
                jnp.broadcast_to(training_relative, (plan.sweeps,)),
                jnp.sqrt(jnp.mean(jnp.abs(rank_one_residual) ** 2)),
                rank_one_relative,
                jnp.max(jnp.abs(rank_one_residual)),
                holdout_count=plan.holdout_count,
            )
            return TTCrossResult(rank_one, evidence, True)
    tensor = _initial_train(
        plan.mode_sizes,
        plan.max_rank,
        jnp.mean(training_values),
        training_values.dtype,
    )
    pivots: list[Array] = []
    training_errors: list[Array] = []
    for _ in range(plan.sweeps):
        for axis in range(len(plan.mode_sizes) - 1):
            tensor = _update_pair(tensor, training_indices, training_values, axis, plan)
            residual = training_values - tensor.evaluate(training_indices)
            pivots.append(training_indices[jnp.argmax(jnp.abs(residual))])
        for axis in range(len(plan.mode_sizes) - 2, -1, -1):
            tensor = _update_pair(tensor, training_indices, training_values, axis, plan)
            residual = training_values - tensor.evaluate(training_indices)
            pivots.append(training_indices[jnp.argmax(jnp.abs(residual))])
        prediction = tensor.evaluate(training_indices)
        residual = prediction - training_values
        denominator = jnp.sqrt(jnp.mean(jnp.abs(training_values) ** 2))
        training_errors.append(
            jnp.sqrt(jnp.mean(jnp.abs(residual) ** 2))
            / jnp.where(denominator > 0, denominator, 1)
        )
    holdout_prediction = tensor.evaluate(holdout_indices)
    holdout_residual = holdout_prediction - holdout_values
    holdout_rmse = jnp.sqrt(jnp.mean(jnp.abs(holdout_residual) ** 2))
    holdout_scale = jnp.sqrt(jnp.mean(jnp.abs(holdout_values) ** 2))
    relative_estimator = holdout_rmse / jnp.where(holdout_scale > 0, holdout_scale, 1)
    maximum_error = jnp.max(jnp.abs(holdout_residual))
    evidence = TTCrossEvidence(
        indices,
        values,
        jnp.stack(pivots),
        jnp.stack(training_errors),
        holdout_rmse,
        relative_estimator,
        maximum_error,
        holdout_count=plan.holdout_count,
    )
    converged = bool(np.asarray(relative_estimator <= plan.relative_tolerance))
    return TTCrossResult(tensor, evidence, converged)


__all__ = [
    "TTCrossEvidence",
    "TTCrossPlan",
    "TTCrossResult",
    "tensor_train_cross",
]
