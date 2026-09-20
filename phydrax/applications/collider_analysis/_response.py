#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._analysis import HistogramPlan


class ResponseMatrix(StrictModule, NonTrainableState):
    counts: Array
    sum_squared_weights: Array
    probabilities: Array
    efficiency: Array
    valid: Array
    truth_plan_id: str = eqx.field(static=True)
    reconstructed_plan_id: str = eqx.field(static=True)


def build_response_matrix(
    truth_plan: HistogramPlan,
    reconstructed_plan: HistogramPlan,
    truth_values: ArrayLike,
    reconstructed_values: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    matched: ArrayLike | None = None,
) -> ResponseMatrix:
    if not isinstance(truth_plan, HistogramPlan) or not isinstance(
        reconstructed_plan, HistogramPlan
    ):
        raise TypeError("Response matrices require histogram plans.")
    truth = jnp.asarray(truth_values)
    reconstructed = jnp.asarray(reconstructed_values, dtype=truth.dtype)
    weights_ = jnp.asarray(weights, dtype=truth.dtype)
    if (
        truth.ndim != 1
        or reconstructed.shape != truth.shape
        or weights_.shape != truth.shape
    ):
        raise ValueError("Response inputs must be aligned vectors.")
    matched_ = (
        jnp.ones(truth.shape, dtype=jnp.bool_)
        if matched is None
        else jnp.asarray(matched, dtype=jnp.bool_)
    )
    if matched_.shape != truth.shape:
        raise ValueError("matched must align with response inputs.")
    truth_index = jnp.searchsorted(truth_plan.edges, truth, side="right") - 1
    reco_index = (
        jnp.searchsorted(reconstructed_plan.edges, reconstructed, side="right") - 1
    )
    in_truth = (truth_index >= 0) & (truth_index < truth_plan.bin_count)
    in_reco = (reco_index >= 0) & (reco_index < reconstructed_plan.bin_count)
    finite = jnp.isfinite(truth) & jnp.isfinite(reconstructed) & jnp.isfinite(weights_)
    active = matched_ & in_truth & in_reco & finite
    truth_membership = jax.nn.one_hot(
        jnp.clip(truth_index, 0, truth_plan.bin_count - 1),
        truth_plan.bin_count,
        dtype=weights_.dtype,
    )
    reco_membership = jax.nn.one_hot(
        jnp.clip(reco_index, 0, reconstructed_plan.bin_count - 1),
        reconstructed_plan.bin_count,
        dtype=weights_.dtype,
    )
    counts = ein.contract(
        "n,nr,nt->rt",
        jnp.where(active, weights_, 0.0),
        reco_membership,
        truth_membership,
    )
    squared = ein.contract(
        "n,nr,nt->rt",
        jnp.where(active, weights_ * weights_, 0.0),
        reco_membership,
        truth_membership,
    )
    generated_truth = jnp.sum(
        truth_membership * jnp.where(in_truth & finite, weights_, 0.0)[:, None],
        axis=0,
    )
    probabilities = counts / jnp.maximum(
        generated_truth[None, :], jnp.finfo(weights_.dtype).tiny
    )
    efficiency = jnp.sum(counts, axis=0) / jnp.maximum(
        generated_truth, jnp.finfo(weights_.dtype).tiny
    )
    valid = jnp.all(jnp.isfinite(probabilities)) & jnp.all(generated_truth >= 0.0)
    return ResponseMatrix(
        counts,
        squared,
        probabilities,
        efficiency,
        valid,
        truth_plan.plan_id,
        reconstructed_plan.plan_id,
    )


class UnfoldingResult(StrictModule, NonTrainableState):
    unfolded: Array
    covariance: Array
    residual_norm: Array
    status: Array
    successful: Array
    response_plan_id: str = eqx.field(static=True)


def unfold_tikhonov(
    response: ResponseMatrix,
    observed: ArrayLike,
    variances: ArrayLike,
    /,
    *,
    regularization: float,
) -> UnfoldingResult:
    """Solve a fixed response inverse with explicit Tikhonov regularization."""
    if not isinstance(response, ResponseMatrix):
        raise TypeError("response must be ResponseMatrix.")
    observed_ = jnp.asarray(observed)
    variances_ = jnp.asarray(variances, dtype=observed_.dtype)
    reco_count, truth_count = response.probabilities.shape
    if observed_.shape != (reco_count,) or variances_.shape != (reco_count,):
        raise ValueError(
            "Observed values and variances must align with reconstructed bins."
        )
    regularization_ = float(regularization)
    if not math.isfinite(regularization_) or regularization_ <= 0.0:
        raise ValueError("regularization must be finite and positive.")
    weights = 1.0 / jnp.maximum(variances_, jnp.finfo(observed_.dtype).tiny)
    matrix = response.probabilities
    normal = ein.contract(
        "ri,r,rj->ij", matrix, weights, matrix
    ) + regularization_ * jnp.eye(truth_count)
    rhs = ein.contract("ri,r,r->i", matrix, weights, observed_)
    solved = solve(
        LinearSystem(DenseLinearOperator(normal)),
        rhs,
        policy=LinearSolvePolicy(DenseLU()),
    )
    covariance = solve(
        LinearSystem(DenseLinearOperator(normal)),
        jnp.eye(truth_count, dtype=normal.dtype),
        policy=LinearSolvePolicy(DenseLU()),
    )
    residual = matrix @ solved.value - observed_
    successful = (
        response.valid
        & jnp.all(solved.status == 0)
        & jnp.all(covariance.status == 0)
        & jnp.all(jnp.isfinite(solved.value))
    )
    return UnfoldingResult(
        solved.value,
        covariance.value,
        jnp.linalg.norm(residual),
        jnp.asarray(jnp.where(successful, 0, 1), dtype=jnp.int32),
        successful,
        response.truth_plan_id,
    )


__all__ = [
    "ResponseMatrix",
    "UnfoldingResult",
    "build_response_matrix",
    "unfold_tikhonov",
]
