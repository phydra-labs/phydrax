#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded sign-free stochastic-lattice candidate evidence contracts."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule


class SignFreeStochasticCandidatePlan(StrictModule):
    chain_count: int = eqx.field(static=True)
    draw_count: int = eqx.field(static=True)
    chain_state_width: int = eqx.field(static=True)
    observable_count: int = eqx.field(static=True)
    maximum_expansion_order: int = eqx.field(static=True)
    maximum_autocorrelation_lag: int = eqx.field(static=True)
    minimum_signed_effective_samples: float = eqx.field(static=True)
    sign_tolerance: float = eqx.field(static=True)
    maximum_raw_bytes: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        chain_count: int,
        draw_count: int,
        chain_state_width: int,
        observable_count: int,
        maximum_expansion_order: int,
        maximum_autocorrelation_lag: int,
        minimum_signed_effective_samples: float,
        sign_tolerance: float,
        maximum_raw_bytes: int,
        method_id: str,
    ):
        chains = int(chain_count)
        draws = int(draw_count)
        state_width = int(chain_state_width)
        observables = int(observable_count)
        maximum_order = int(maximum_expansion_order)
        lag = int(maximum_autocorrelation_lag)
        minimum_ess = float(minimum_signed_effective_samples)
        tolerance = float(sign_tolerance)
        byte_limit = int(maximum_raw_bytes)
        method = str(method_id)
        if (
            min(chains, draws, state_width, observables, byte_limit) < 1
            or maximum_order < 0
            or lag < 0
            or lag >= draws
            or not np.isfinite(minimum_ess)
            or minimum_ess <= 0.0
            or not np.isfinite(tolerance)
            or tolerance < 0.0
            or not method
        ):
            raise ValueError("Sign-free stochastic candidate plan values are invalid.")
        required = (
            chains
            * draws
            * (
                np.dtype(np.complex128).itemsize * (state_width + observables + 1)
                + np.dtype(np.int32).itemsize
                + np.dtype(np.bool_).itemsize
            )
        )
        if required > byte_limit:
            raise ValueError("Raw stochastic evidence exceeds maximum_raw_bytes.")
        self.chain_count = chains
        self.draw_count = draws
        self.chain_state_width = state_width
        self.observable_count = observables
        self.maximum_expansion_order = maximum_order
        self.maximum_autocorrelation_lag = lag
        self.minimum_signed_effective_samples = minimum_ess
        self.sign_tolerance = tolerance
        self.maximum_raw_bytes = byte_limit
        self.method_id = method
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sign-free-stochastic-candidate-plan",
                "chain_count": chains,
                "draw_count": draws,
                "chain_state_width": state_width,
                "observable_count": observables,
                "maximum_expansion_order": maximum_order,
                "maximum_autocorrelation_lag": lag,
                "minimum_signed_effective_samples": minimum_ess,
                "sign_tolerance": tolerance,
                "maximum_raw_bytes": byte_limit,
                "method_id": method,
            }
        )


class SignFreeStochasticCandidateEvidence(StrictModule):
    """Raw chain/order/sign records plus autocorrelation, ESS, and covariance."""

    raw_chain: Array
    raw_orders: Array
    raw_observables: Array
    raw_signs: Array
    accepted: Array
    order_histogram: Array
    mean_phase: Array
    average_sign: Array
    maximum_unit_phase_residual: Array
    sign_autocorrelation: Array
    signed_effective_samples: Array
    reweighted_mean: Array
    reweighted_covariance: Array
    sign_free: Array
    finite: Array
    sufficient_effective_samples: Array
    successful: Array
    method_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def assess_sign_free_stochastic_candidate(
    plan: SignFreeStochasticCandidatePlan,
    raw_chain: ArrayLike,
    raw_orders: ArrayLike,
    raw_observables: ArrayLike,
    raw_signs: ArrayLike,
    accepted: ArrayLike,
    /,
) -> SignFreeStochasticCandidateEvidence:
    """Assess retained raw chains without IID or sign-problem-cure claims."""
    if not isinstance(plan, SignFreeStochasticCandidatePlan):
        raise TypeError("plan must be SignFreeStochasticCandidatePlan.")
    chain = jnp.asarray(raw_chain)
    if not jnp.issubdtype(chain.dtype, jnp.number) or jnp.issubdtype(
        chain.dtype, jnp.bool_
    ):
        raise TypeError("raw_chain must use a numeric non-boolean dtype.")
    order_values = jnp.asarray(raw_orders)
    if not jnp.issubdtype(order_values.dtype, jnp.integer):
        raise TypeError("raw_orders must use an integer dtype.")
    orders = order_values.astype(jnp.int32)
    observables = jnp.asarray(raw_observables, dtype=jnp.complex128)
    signs = jnp.asarray(raw_signs, dtype=jnp.complex128)
    acceptance = jnp.asarray(accepted, dtype=jnp.bool_)
    expected_states = (
        plan.chain_count,
        plan.draw_count,
        plan.chain_state_width,
    )
    expected_observables = (
        plan.chain_count,
        plan.draw_count,
        plan.observable_count,
    )
    expected_chain = (plan.chain_count, plan.draw_count)
    if chain.shape != expected_states:
        raise ValueError(f"raw_chain must have shape {expected_states}.")
    if orders.shape != expected_chain:
        raise ValueError(f"raw_orders must have shape {expected_chain}.")
    if observables.shape != expected_observables:
        raise ValueError(f"raw_observables must have shape {expected_observables}.")
    if signs.shape != expected_chain or acceptance.shape != expected_chain:
        raise ValueError(f"raw_signs and accepted must have shape {expected_chain}.")
    orders = eqx.error_if(
        orders,
        jnp.any((orders < 0) | (orders > plan.maximum_expansion_order)),
        "raw_orders exceed the admitted expansion-order range.",
    )
    finite = (
        jnp.all(jnp.isfinite(chain))
        & jnp.all(jnp.isfinite(observables))
        & jnp.all(jnp.isfinite(signs))
    )
    order_histogram = jnp.bincount(
        orders.reshape((-1,)), length=plan.maximum_expansion_order + 1
    )
    magnitude_residual = jnp.max(jnp.abs(jnp.abs(signs) - 1.0))
    mean_phase = jnp.mean(signs)
    average_sign = jnp.abs(mean_phase)
    centered = signs - jnp.mean(signs, axis=1, keepdims=True)
    variance = jnp.mean(jnp.abs(centered) ** 2, axis=1)
    correlations = []
    for lag in range(plan.maximum_autocorrelation_lag + 1):
        products = centered[:, : plan.draw_count - lag] * jnp.conj(centered[:, lag:])
        covariance = jnp.real(jnp.mean(products, axis=1))
        correlations.append(
            jnp.mean(jnp.where(variance > 0.0, covariance / variance, 0.0))
        )
    autocorrelation = jnp.stack(correlations)
    positive_tail = jnp.maximum(autocorrelation[1:], 0.0)
    integrated_time = 1.0 + 2.0 * jnp.sum(positive_tail)
    sample_count = plan.chain_count * plan.draw_count
    signed_ess = sample_count * average_sign**2 / integrated_time
    denominator = jnp.sum(signs)
    safe_denominator = jnp.where(jnp.abs(denominator) > 0.0, denominator, 1.0 + 0.0j)
    flattened = observables.reshape((-1, plan.observable_count))
    flat_signs = signs.reshape((-1,))
    mean = jnp.sum(flat_signs[:, None] * flattened, axis=0) / safe_denominator
    centered_observable = flattened - mean[None, :]
    covariance = (
        jnp.conj(centered_observable).T
        @ (flat_signs[:, None] * centered_observable)
        / safe_denominator
    )
    sign_free = (
        (magnitude_residual <= plan.sign_tolerance)
        & (jnp.max(jnp.abs(jnp.imag(signs))) <= plan.sign_tolerance)
        & (jnp.min(jnp.real(signs)) >= -plan.sign_tolerance)
    )
    sufficient = signed_ess >= plan.minimum_signed_effective_samples
    successful = finite & sign_free & sufficient & (jnp.abs(denominator) > 0.0)
    return SignFreeStochasticCandidateEvidence(
        raw_chain=chain,
        raw_orders=orders,
        raw_observables=observables,
        raw_signs=signs,
        accepted=acceptance,
        order_histogram=order_histogram,
        mean_phase=mean_phase,
        average_sign=average_sign,
        maximum_unit_phase_residual=magnitude_residual,
        sign_autocorrelation=autocorrelation,
        signed_effective_samples=signed_ess,
        reweighted_mean=mean,
        reweighted_covariance=covariance,
        sign_free=sign_free,
        finite=finite,
        sufficient_effective_samples=sufficient,
        successful=successful,
        method_id=plan.method_id,
        plan_id=plan.plan_id,
        evidence_id=canonical_fingerprint(
            {
                "kind": "sign-free-stochastic-candidate-evidence",
                "plan": plan.plan_id,
                "raw": array_tree_fingerprint(
                    {
                        "chain": np.asarray(chain),
                        "orders": np.asarray(orders),
                        "observables": np.asarray(observables),
                        "signs": np.asarray(signs),
                        "accepted": np.asarray(acceptance),
                    }
                ),
            }
        ),
    )


__all__ = [
    "SignFreeStochasticCandidateEvidence",
    "SignFreeStochasticCandidatePlan",
    "assess_sign_free_stochastic_candidate",
]
