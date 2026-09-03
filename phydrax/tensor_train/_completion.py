#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._core import TensorTrain
from ._local import regularized_least_squares


class TensorCompletionPlan(StrictModule):
    """Static ranks, sweeps, and local resources for weighted TT completion."""

    mode_sizes: tuple[int, ...] = eqx.field(static=True)
    max_rank: int = eqx.field(static=True)
    sweeps: int = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    max_local_unknowns: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_sizes: Sequence[int],
        /,
        *,
        max_rank: int,
        sweeps: int,
        relative_tolerance: float,
        regularization: float,
        max_local_unknowns: int,
    ):
        modes = tuple(int(size) for size in mode_sizes)
        rank = int(max_rank)
        sweep_count = int(sweeps)
        tolerance = float(relative_tolerance)
        ridge = float(regularization)
        local_limit = int(max_local_unknowns)
        if not modes or any(size <= 0 for size in modes):
            raise ValueError("Completion modes must be nonempty and positive.")
        if (
            rank <= 0
            or sweep_count <= 0
            or tolerance < 0.0
            or ridge <= 0.0
            or local_limit <= 0
        ):
            raise ValueError(
                "Completion rank, sweeps, tolerance, ridge, or local budget is invalid."
            )
        self.mode_sizes = modes
        self.max_rank = rank
        self.sweeps = sweep_count
        self.relative_tolerance = tolerance
        self.regularization = ridge
        self.max_local_unknowns = local_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "weighted-tensor-completion-plan",
                "modes": modes,
                "max_rank": rank,
                "sweeps": sweep_count,
                "relative_tolerance": tolerance,
                "regularization": ridge,
                "max_local_unknowns": local_limit,
            }
        )


class TensorCompletionEvidence(StrictModule):
    training_weighted_root_mean_square_errors: Array
    holdout_weighted_root_mean_square_errors: Array
    holdout_relative_error_estimator: Array
    observed_count: int = eqx.field(static=True)
    holdout_count: int = eqx.field(static=True)
    estimator_is_guarantee: bool = eqx.field(static=True)

    def __init__(
        self,
        training_errors: ArrayLike,
        holdout_errors: ArrayLike,
        holdout_relative_error_estimator: ArrayLike,
        /,
        *,
        observed_count: int,
        holdout_count: int,
    ):
        training = jnp.asarray(training_errors)
        holdout = jnp.asarray(holdout_errors)
        if training.ndim != 1 or holdout.shape != training.shape:
            raise ValueError("Completion training and holdout histories must agree.")
        self.training_weighted_root_mean_square_errors = training
        self.holdout_weighted_root_mean_square_errors = holdout
        self.holdout_relative_error_estimator = jnp.asarray(
            holdout_relative_error_estimator
        )
        self.observed_count = int(observed_count)
        self.holdout_count = int(holdout_count)
        self.estimator_is_guarantee = False


class TensorCompletionResult(StrictModule):
    tensor: TensorTrain
    evidence: TensorCompletionEvidence
    converged: bool = eqx.field(static=True)
    status: str = eqx.field(static=True)

    def __init__(
        self,
        tensor: TensorTrain,
        evidence: TensorCompletionEvidence,
        converged: bool,
        /,
    ):
        self.tensor = tensor
        self.evidence = evidence
        self.converged = bool(converged)
        self.status = "converged" if self.converged else "iteration_budget_exhausted"


def _validate_samples(
    indices: ArrayLike,
    values: ArrayLike,
    weights: ArrayLike,
    mode_sizes: tuple[int, ...],
    label: str,
    /,
) -> tuple[Array, Array, Array]:
    points = jnp.asarray(indices, dtype=jnp.int32)
    targets = jnp.asarray(values)
    importance = jnp.asarray(weights)
    if points.ndim != 2 or points.shape[1] != len(mode_sizes) or points.shape[0] == 0:
        raise ValueError(f"{label} indices must be a nonempty multi-index matrix.")
    if targets.shape != (points.shape[0],) or importance.shape != targets.shape:
        raise ValueError(f"{label} values and weights must match its index count.")
    host_points = np.asarray(points)
    if any(
        np.any(host_points[:, axis] < 0) or np.any(host_points[:, axis] >= size)
        for axis, size in enumerate(mode_sizes)
    ):
        raise IndexError(f"{label} contains an index outside the tensor shape.")
    if bool(np.any(np.asarray(importance) <= 0)):
        raise ValueError(f"{label} weights must be strictly positive.")
    return points, targets, importance


def _deterministic_initial(
    mode_sizes: tuple[int, ...], max_rank: int, mean: Array, dtype, /
) -> TensorTrain:
    total = prod(mode_sizes)
    ranks = [1]
    prefix = 1
    for size in mode_sizes[:-1]:
        prefix *= size
        ranks.append(min(max_rank, prefix, total // prefix))
    ranks.append(1)
    cores = []
    for axis, size in enumerate(mode_sizes):
        shape = (ranks[axis], size, ranks[axis + 1])
        phase = jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape)
        core = (1.0e-3 * jnp.cos(phase + 0.5 * axis)).astype(dtype)
        core = core.at[0, :, 0].set(jnp.ones((size,), dtype=dtype))
        cores.append(core)
    cores[-1] = cores[-1].at[0, :, 0].set(jnp.asarray(mean, dtype=dtype))
    return TensorTrain(tuple(cores))


def _sample_core_frame(
    tensor: TensorTrain,
    indices: Array,
    axis: int,
    /,
) -> Array:
    def one(point):
        left = jnp.ones((1,), dtype=tensor.dtype)
        for position in range(axis):
            left = ein.contract(
                "a,ab->b", left, tensor.cores[position][:, point[position], :]
            )
        right = jnp.ones((1,), dtype=tensor.dtype)
        for position in range(tensor.order - 1, axis, -1):
            right = ein.contract(
                "ab,b->a", tensor.cores[position][:, point[position], :], right
            )
        physical = jax.nn.one_hot(
            point[axis], tensor.mode_sizes[axis], dtype=tensor.dtype
        )
        return ein.contract("a,i,b->aib", left, physical, right).reshape((-1,))

    return jax.vmap(one)(indices)


def _weighted_rmse(
    tensor: TensorTrain,
    indices: Array,
    values: Array,
    weights: Array,
    /,
) -> Array:
    residual = tensor.evaluate(indices) - values
    return jnp.sqrt(jnp.sum(weights * jnp.abs(residual) ** 2) / jnp.sum(weights))


def weighted_tensor_completion(
    plan: TensorCompletionPlan,
    observed_indices: ArrayLike,
    observed_values: ArrayLike,
    observed_weights: ArrayLike,
    holdout_indices: ArrayLike,
    holdout_values: ArrayLike,
    holdout_weights: ArrayLike,
    /,
    *,
    initial: TensorTrain | None = None,
) -> TensorCompletionResult:
    """Fit weighted observations by finite alternating local TT regressions."""
    indices, values, weights = _validate_samples(
        observed_indices,
        observed_values,
        observed_weights,
        plan.mode_sizes,
        "observed",
    )
    held_indices, held_values, held_weights = _validate_samples(
        holdout_indices,
        holdout_values,
        holdout_weights,
        plan.mode_sizes,
        "holdout",
    )
    if initial is None:
        tensor = _deterministic_initial(
            plan.mode_sizes,
            plan.max_rank,
            jnp.sum(weights * values) / jnp.sum(weights),
            values.dtype,
        )
    else:
        if initial.mode_sizes != plan.mode_sizes or any(
            rank > plan.max_rank for rank in initial.ranks
        ):
            raise ValueError(
                "Completion initial tensor violates the plan shape or rank cap."
            )
        tensor = initial
    square_root_weights = jnp.sqrt(weights)
    training_errors: list[Array] = []
    holdout_errors: list[Array] = []
    for _ in range(plan.sweeps):
        for axis in range(tensor.order):
            frame = _sample_core_frame(tensor, indices, axis)
            if frame.shape[1] > plan.max_local_unknowns:
                raise ValueError(
                    f"Completion core needs {frame.shape[1]} unknowns, exceeding "
                    f"budget {plan.max_local_unknowns}."
                )
            solution = regularized_least_squares(
                square_root_weights[:, None] * frame,
                square_root_weights * values,
                plan.regularization,
            )
            cores = list(tensor.cores)
            cores[axis] = solution.reshape(cores[axis].shape)
            tensor = TensorTrain(tuple(cores))
        for axis in range(tensor.order - 1, -1, -1):
            frame = _sample_core_frame(tensor, indices, axis)
            solution = regularized_least_squares(
                square_root_weights[:, None] * frame,
                square_root_weights * values,
                plan.regularization,
            )
            cores = list(tensor.cores)
            cores[axis] = solution.reshape(cores[axis].shape)
            tensor = TensorTrain(tuple(cores))
        training_errors.append(_weighted_rmse(tensor, indices, values, weights))
        holdout_errors.append(
            _weighted_rmse(tensor, held_indices, held_values, held_weights)
        )
    training_history = jnp.stack(training_errors)
    holdout_history = jnp.stack(holdout_errors)
    holdout_scale = jnp.sqrt(
        jnp.sum(held_weights * jnp.abs(held_values) ** 2) / jnp.sum(held_weights)
    )
    relative_estimator = holdout_history[-1] / jnp.where(
        holdout_scale > 0, holdout_scale, 1
    )
    evidence = TensorCompletionEvidence(
        training_history,
        holdout_history,
        relative_estimator,
        observed_count=indices.shape[0],
        holdout_count=held_indices.shape[0],
    )
    converged = bool(np.asarray(relative_estimator <= plan.relative_tolerance))
    return TensorCompletionResult(tensor, evidence, converged)


__all__ = [
    "TensorCompletionEvidence",
    "TensorCompletionPlan",
    "TensorCompletionResult",
    "weighted_tensor_completion",
]
