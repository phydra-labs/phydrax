#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._commercial import MPMDerivativeEvidence, MPMDerivativeKind


class MPMGradientResult(StrictModule):
    primal: Any
    derivative: Any
    candidate_derivatives: Any
    evidence: MPMDerivativeEvidence


class MPMEventLocalizationResult(StrictModule):
    event_time: Array
    event_value: Array
    iterations: Array
    bracket_width: Array
    transversality: Array
    localized: Array


def _tree_vdot(left: Any, right: Any, /) -> Array:
    terms = []
    for first, second in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if eqx.is_inexact_array(first) and eqx.is_inexact_array(second):
            terms.append(jnp.vdot(first, second))
    return jnp.asarray(0.0) if not terms else sum(terms[1:], terms[0])


def _evidence(
    kind: MPMDerivativeKind,
    *,
    valid,
    branch_margin=jnp.nan,
    event_time=jnp.nan,
    transversality_margin=jnp.nan,
    primal_residual=0.0,
    transpose_residual=0.0,
    sample_count=0,
    estimator_variance=jnp.nan,
    reason_code=0,
    journal_digest=0,
    evidence_id: str,
):
    return MPMDerivativeEvidence(
        jnp.asarray(int(kind), dtype=jnp.int32),
        jnp.asarray(valid, dtype=bool),
        jnp.asarray(branch_margin),
        jnp.asarray(event_time),
        jnp.asarray(transversality_margin),
        jnp.asarray(primal_residual),
        jnp.asarray(transpose_residual),
        jnp.asarray(sample_count, dtype=jnp.int32),
        jnp.asarray(estimator_variance),
        jnp.asarray(reason_code, dtype=jnp.int32),
        jnp.asarray(journal_digest, dtype=jnp.int64),
        evidence_id,
    )


def branchwise_gradient(
    objective: Callable[[Any], ArrayLike],
    primal: Any,
    direction: Any,
    /,
    *,
    branch_margin: ArrayLike,
    journal_digest: ArrayLike,
    evidence_id: str,
) -> MPMGradientResult:
    if not callable(objective):
        raise TypeError("objective must be callable.")
    value, directional = jax.jvp(objective, (primal,), (direction,))
    gradient = jax.grad(lambda current: jnp.asarray(objective(current)))(primal)
    reverse = _tree_vdot(gradient, direction)
    residual = jnp.abs(jnp.asarray(directional) - reverse)
    margin = jnp.asarray(branch_margin)
    valid = jnp.isfinite(residual) & jnp.isfinite(margin) & (margin > 0.0)
    evidence = _evidence(
        MPMDerivativeKind.BRANCHWISE,
        valid=valid,
        branch_margin=margin,
        primal_residual=residual,
        transpose_residual=residual,
        journal_digest=journal_digest,
        evidence_id=evidence_id,
    )
    return MPMGradientResult(value, directional, (), evidence)


def smooth_surrogate_gradient(
    objective: Callable[[Any], ArrayLike],
    primal: Any,
    direction: Any,
    /,
    *,
    model_bias_bound: ArrayLike,
    evidence_id: str,
) -> MPMGradientResult:
    value, derivative = jax.jvp(objective, (primal,), (direction,))
    bias = jnp.asarray(model_bias_bound)
    evidence = _evidence(
        MPMDerivativeKind.SURROGATE,
        valid=jnp.isfinite(bias) & (bias >= 0.0),
        primal_residual=bias,
        evidence_id=evidence_id,
    )
    return MPMGradientResult(value, derivative, (), evidence)


def locate_event(
    event: Callable[[Array], ArrayLike],
    lower: ArrayLike,
    upper: ArrayLike,
    /,
    *,
    maximum_steps: int = 64,
    tolerance: float = 1.0e-12,
) -> MPMEventLocalizationResult:
    if not callable(event):
        raise TypeError("event must be callable.")
    steps = int(maximum_steps)
    if steps <= 0 or tolerance <= 0.0:
        raise ValueError("Event localization policy is invalid.")
    lower_ = jnp.asarray(lower)
    upper_ = jnp.asarray(upper, dtype=lower_.dtype)
    if lower_.shape != () or upper_.shape != ():
        raise ValueError("Event brackets must be scalar.")
    lower_value = jnp.asarray(event(lower_))
    upper_value = jnp.asarray(event(upper_))

    def body(_, carry):
        left, right, left_value, right_value = carry
        middle = 0.5 * (left + right)
        middle_value = jnp.asarray(event(middle))
        choose_left = jnp.signbit(left_value) != jnp.signbit(middle_value)
        next_left = jnp.where(choose_left, left, middle)
        next_right = jnp.where(choose_left, middle, right)
        next_left_value = jnp.where(choose_left, left_value, middle_value)
        next_right_value = jnp.where(choose_left, middle_value, right_value)
        return next_left, next_right, next_left_value, next_right_value

    left, right, _, _ = jax.lax.fori_loop(
        0, steps, body, (lower_, upper_, lower_value, upper_value)
    )
    event_time = 0.5 * (left + right)
    value = jnp.asarray(event(event_time))
    transversality = jax.grad(lambda time: jnp.asarray(event(time)))(event_time)
    bracket = right - left
    sign_change = jnp.signbit(lower_value) != jnp.signbit(upper_value)
    localized = (
        sign_change
        & jnp.isfinite(value)
        & (jnp.abs(value) <= tolerance)
        & (jnp.abs(transversality) > tolerance)
    )
    return MPMEventLocalizationResult(
        event_time,
        value,
        jnp.asarray(steps, dtype=jnp.int32),
        bracket,
        transversality,
        localized,
    )


def saltation_action(
    vector_field_before: ArrayLike,
    vector_field_after: ArrayLike,
    event_normal: ArrayLike,
    reset_jacobian: ArrayLike,
    tangent_before: ArrayLike,
    /,
    *,
    minimum_transversality: float = 1.0e-10,
    evidence_id: str,
) -> MPMGradientResult:
    before = jnp.asarray(vector_field_before)
    after = jnp.asarray(vector_field_after)
    normal = jnp.asarray(event_normal)
    reset = jnp.asarray(reset_jacobian)
    tangent = jnp.asarray(tangent_before)
    denominator = jnp.vdot(normal, before)
    safe = jnp.where(jnp.abs(denominator) >= minimum_transversality, denominator, 1.0)
    reset_before = reset @ before
    saltation = reset + jnp.outer(after - reset_before, normal) / safe
    derivative = saltation @ tangent
    valid = (jnp.abs(denominator) >= minimum_transversality) & jnp.all(
        jnp.isfinite(saltation)
    )
    evidence = _evidence(
        MPMDerivativeKind.EVENT_AWARE,
        valid=valid,
        transversality_margin=jnp.abs(denominator),
        evidence_id=evidence_id,
    )
    return MPMGradientResult(after, derivative, saltation, evidence)


def generalized_contact_derivative(
    candidate_derivatives: ArrayLike,
    selected_index: ArrayLike,
    /,
    *,
    complementarity_residual: ArrayLike,
    strict_complementarity_margin: ArrayLike,
    evidence_id: str,
) -> MPMGradientResult:
    candidates = jnp.asarray(candidate_derivatives)
    selected = jnp.asarray(selected_index, dtype=jnp.int32)
    if candidates.ndim < 1:
        raise ValueError("Generalized derivative candidates need leading set axis.")
    derivative = candidates[jnp.clip(selected, 0, candidates.shape[0] - 1)]
    residual = jnp.asarray(complementarity_residual)
    margin = jnp.asarray(strict_complementarity_margin)
    valid = (
        (selected >= 0)
        & (selected < candidates.shape[0])
        & jnp.all(jnp.isfinite(candidates))
        & jnp.isfinite(residual)
        & (residual >= 0.0)
    )
    evidence = _evidence(
        MPMDerivativeKind.GENERALIZED_SET,
        valid=valid,
        branch_margin=margin,
        primal_residual=residual,
        evidence_id=evidence_id,
    )
    return MPMGradientResult(jnp.asarray(jnp.nan), derivative, candidates, evidence)


def stochastic_derivative_estimate(
    samples: ArrayLike,
    /,
    *,
    journal_digest: ArrayLike,
    evidence_id: str,
) -> MPMGradientResult:
    values = jnp.asarray(samples)
    if values.ndim < 1 or values.shape[0] < 2:
        raise ValueError("Stochastic derivative requires at least two samples.")
    mean = jnp.mean(values, axis=0)
    variance = jnp.mean((values - mean) ** 2)
    valid = jnp.all(jnp.isfinite(values)) & jnp.isfinite(variance)
    evidence = _evidence(
        MPMDerivativeKind.STOCHASTIC_ESTIMATOR,
        valid=valid,
        sample_count=values.shape[0],
        estimator_variance=variance,
        journal_digest=journal_digest,
        evidence_id=evidence_id,
    )
    return MPMGradientResult(jnp.asarray(jnp.nan), mean, values, evidence)


def nondifferentiable_result(
    primal: Any,
    /,
    *,
    reason_code: int,
    journal_digest: ArrayLike,
    evidence_id: str,
) -> MPMGradientResult:
    evidence = _evidence(
        MPMDerivativeKind.NONDIFFERENTIABLE,
        valid=False,
        reason_code=reason_code,
        journal_digest=journal_digest,
        evidence_id=evidence_id,
    )
    return MPMGradientResult(primal, None, (), evidence)


__all__ = [
    "MPMEventLocalizationResult",
    "MPMGradientResult",
    "branchwise_gradient",
    "generalized_contact_derivative",
    "locate_event",
    "nondifferentiable_result",
    "saltation_action",
    "smooth_surrogate_gradient",
    "stochastic_derivative_estimate",
]
