#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from .._trajectory import TrajectoryData
from ..identification._markov_state import MarkovStateModel
from ..identification._variational_kinetics import (
    _event_mask,
    _lagged_pair_data,
    VAMPResult,
)


class VAMPValidationResult(StrictModule):
    vamp_e_score: Array
    covariance_residual: Array
    effective_samples: Array
    valid: Array
    training_method_id: str = eqx.field(static=True)
    validation_dataset_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class MarkovValidationResult(StrictModule):
    chapman_kolmogorov_residual: Array
    implied_timescale_residual: Array
    stationary_residual: Array
    valid: Array
    short_model_id: str = eqx.field(static=True)
    long_model_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def score_vamp(result: VAMPResult, data: TrajectoryData, /) -> VAMPValidationResult:
    """Evaluate the training canonical functions with the held-out VAMP-E score."""

    if not isinstance(result, VAMPResult) or not isinstance(data, TrajectoryData):
        raise TypeError("score_vamp requires a VAMPResult and TrajectoryData.")
    if result.library.state_layout.layout_id != data.state_layout.layout_id:
        raise ValueError("VAMP result and validation trajectory layouts differ.")
    lag = result.diagnostics.lag
    transitions, weights, evidence = _lagged_pair_data(
        data, lag.lag, lag.weighting, 1.0e-8
    )
    event_rank = len(data.state_layout.shape)
    source_states = jnp.where(
        _event_mask(transitions.valid, event_rank), transitions.source_states, 0.0
    )
    target_states = jnp.where(
        _event_mask(transitions.valid, event_rank), transitions.target_states, 0.0
    )
    source_evaluation = result.library.evaluate(source_states)
    target_evaluation = result.library.evaluate(target_states)
    valid = transitions.valid & source_evaluation.valid & target_evaluation.valid
    weight = jnp.where(valid, weights, 0.0).reshape((-1,))
    source = result.model.transform(
        source_evaluation.values.reshape((-1, result.library.num_features))
    )
    target = result.model.transform_targets(
        target_evaluation.values.reshape((-1, result.library.num_features))
    )
    source = jnp.where(valid.reshape((-1, 1)), source, 0.0)
    target = jnp.where(valid.reshape((-1, 1)), target, 0.0)
    total = jnp.sum(weight)
    denominator = jnp.maximum(total, 1.0)
    c00 = contract("ni,n,nj->ij", source, weight, source) / denominator
    c11 = contract("ni,n,nj->ij", target, weight, target) / denominator
    c01 = contract("ni,n,nj->ij", source, weight, target) / denominator
    singular = jnp.diag(result.model.singular_values)
    score = 2.0 * jnp.trace(singular @ c01) - jnp.trace(singular @ c11 @ singular @ c00)
    identity = jnp.eye(source.shape[-1], dtype=source.dtype)
    covariance_residual = jnp.maximum(
        jnp.max(jnp.abs(c00 - identity)), jnp.max(jnp.abs(c11 - identity))
    )
    finite = jnp.isfinite(score) & jnp.isfinite(covariance_residual)
    successful = finite & (total > 0.0) & evidence.uniform_physical_lag
    result_id = canonical_fingerprint(
        {
            "kind": "vamp-validation",
            "training": result.method_id,
            "validation": data.dataset_id,
        }
    )
    return VAMPValidationResult(
        vamp_e_score=score,
        covariance_residual=covariance_residual,
        effective_samples=evidence.effective_samples,
        valid=successful,
        training_method_id=result.method_id,
        validation_dataset_id=data.dataset_id,
        result_id=result_id,
    )


def validate_markov_models(
    short: MarkovStateModel,
    long: MarkovStateModel,
    multiplier: int,
    /,
) -> MarkovValidationResult:
    """Compare a lag-multiplied transition with an independently fitted long-lag model."""

    if not isinstance(short, MarkovStateModel) or not isinstance(long, MarkovStateModel):
        raise TypeError("validate_markov_models requires MarkovStateModel values.")
    factor = int(multiplier)
    if factor < 1:
        raise ValueError("multiplier must be positive.")
    if short.state_count != long.state_count:
        raise ValueError("Markov models must use the same state count.")
    if long.diagnostics.lag.lag != short.diagnostics.lag.lag * factor:
        raise ValueError("Long-model lag must equal short lag times multiplier.")
    propagated = jnp.eye(short.state_count, dtype=short.transition_matrix.dtype)
    for _ in range(factor):
        propagated = propagated @ short.transition_matrix
    ck = jnp.sqrt(jnp.sum((propagated - long.transition_matrix) ** 2))
    short_times, short_valid = short.implied_timescales()
    long_times, long_valid = long.implied_timescales()
    mode_valid = short_valid & long_valid
    time_scale = jnp.maximum(jnp.abs(long_times), 1.0)
    time_residuals = jnp.where(
        mode_valid, jnp.abs(short_times - long_times) / time_scale, 0.0
    )
    time_residual = jnp.max(time_residuals)
    stationary = jnp.sqrt(
        jnp.sum((short.stationary_probabilities - long.stationary_probabilities) ** 2)
    )
    finite = jnp.all(jnp.isfinite(jnp.asarray([ck, time_residual, stationary])))
    valid = short.valid & long.valid & finite
    result_id = canonical_fingerprint(
        {
            "kind": "markov-kinetic-validation",
            "short": short.model_id,
            "long": long.model_id,
            "multiplier": factor,
        }
    )
    return MarkovValidationResult(
        chapman_kolmogorov_residual=ck,
        implied_timescale_residual=time_residual,
        stationary_residual=stationary,
        valid=valid,
        short_model_id=short.model_id,
        long_model_id=long.model_id,
        result_id=result_id,
    )


__all__ = [
    "MarkovValidationResult",
    "VAMPValidationResult",
    "score_vamp",
    "validate_markov_models",
]
