#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtri
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..core import PhysicalLaw


def _loss_probability(
    losses: ArrayLike, probabilities: ArrayLike, /
) -> tuple[Array, Array]:
    loss = jnp.asarray(losses)
    probability = jnp.asarray(probabilities)
    if loss.ndim != 1 or loss.shape[0] == 0:
        raise ValueError("losses must be a non-empty vector.")
    if probability.shape != loss.shape:
        raise ValueError("probabilities must have the same shape as losses.")
    if jnp.issubdtype(loss.dtype, jnp.complexfloating):
        raise TypeError("losses must be real-valued.")
    loss = loss.astype(jnp.result_type(loss.dtype, jnp.float32))
    probability = probability.astype(loss.dtype)
    loss_host, probability_host = np.asarray(loss), np.asarray(probability)
    if not np.all(np.isfinite(loss_host)) or not np.all(np.isfinite(probability_host)):
        raise ValueError("Losses and probabilities must be finite.")
    if np.any(probability_host < 0.0) or not np.any(probability_host > 0.0):
        raise ValueError("Probabilities must be non-negative with positive mass.")
    tolerance = 128.0 * np.finfo(probability_host.dtype).eps
    if abs(float(np.sum(probability_host)) - 1.0) > tolerance:
        raise ValueError("Probabilities must sum to one.")
    return loss, probability


def _confidence(value: float, /) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    return result


class CVaRAtoms(StrictModule):
    """Exact finite-distribution upper-tail atom weights in source order."""

    losses: Array
    probabilities: Array
    tail_weights: Array
    value_at_risk: Array
    expected_shortfall: Array
    confidence: float = eqx.field(static=True)


class TailRiskEstimate(StrictModule):
    """P-law loss-tail estimate; not a pricing or stress-law result."""

    value_at_risk: Array
    expected_shortfall: Array
    confidence: float = eqx.field(static=True)
    law: PhysicalLaw = eqx.field(static=True)
    method: str = eqx.field(static=True)


class EntropicRiskEstimate(StrictModule):
    value: Array
    optimal_temperature: Array
    relative_entropy_radius: float = eqx.field(static=True)
    law: PhysicalLaw = eqx.field(static=True)


class KellyRisk(StrictModule):
    expected_log_growth: Array
    gross_returns: Array
    bankrupt: Array
    bankruptcy_probability: Array
    floor: float = eqx.field(static=True)
    law: PhysicalLaw = eqx.field(static=True)


class DrawdownRisk(StrictModule):
    wealth: Array
    running_peak: Array
    drawdown: Array
    maximum_drawdown: Array
    maximum_duration: Array


def cvar_atoms(
    losses: ArrayLike,
    probabilities: ArrayLike,
    confidence: float,
    /,
) -> CVaRAtoms:
    """Resolve fractional VaR atoms exactly for one finite loss distribution."""

    loss, probability = _loss_probability(losses, probabilities)
    level = _confidence(confidence)
    order = jnp.argsort(loss, stable=True)
    sorted_loss = loss[order]
    sorted_probability = probability[order]
    cumulative = jnp.cumsum(sorted_probability)
    previous = jnp.concatenate((jnp.zeros((1,), dtype=loss.dtype), cumulative[:-1]))
    mass = jnp.maximum(jnp.minimum(cumulative, 1.0) - jnp.maximum(previous, level), 0.0)
    sorted_tail_weight = mass / (1.0 - level)
    tail_weight = jnp.zeros_like(probability).at[order].set(sorted_tail_weight)
    quantile_index = jnp.minimum(
        jnp.searchsorted(cumulative, level, side="left"), loss.shape[0] - 1
    )
    value_at_risk = sorted_loss[quantile_index]
    expected_shortfall = jnp.sum(sorted_tail_weight * sorted_loss)
    return CVaRAtoms(
        losses=loss,
        probabilities=probability,
        tail_weights=tail_weight,
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
        confidence=level,
    )


def value_at_risk(
    losses: ArrayLike,
    probabilities: ArrayLike,
    confidence: float,
    /,
) -> Array:
    """Return the lower finite quantile for an explicit discrete loss law."""

    return cvar_atoms(losses, probabilities, confidence).value_at_risk


def expected_shortfall(
    losses: ArrayLike,
    probabilities: ArrayLike,
    confidence: float,
    /,
) -> Array:
    """Return upper-tail expected loss, including a fractional VaR atom."""

    return cvar_atoms(losses, probabilities, confidence).expected_shortfall


def historical_var_es(
    losses: ArrayLike,
    probabilities: ArrayLike,
    confidence: float,
    law: PhysicalLaw,
    /,
) -> TailRiskEstimate:
    if not isinstance(law, PhysicalLaw):
        raise TypeError("Historical VaR/ES requires a PhysicalLaw.")
    atoms = cvar_atoms(losses, probabilities, confidence)
    return TailRiskEstimate(
        value_at_risk=atoms.value_at_risk,
        expected_shortfall=atoms.expected_shortfall,
        confidence=atoms.confidence,
        law=law,
        method="weighted-historical",
    )


def gaussian_var_es(
    mean_loss: ArrayLike,
    standard_deviation: ArrayLike,
    confidence: float,
    law: PhysicalLaw,
    /,
) -> TailRiskEstimate:
    """Analytic Gaussian loss VaR and ES under an explicit physical law."""

    if not isinstance(law, PhysicalLaw):
        raise TypeError("Gaussian VaR/ES requires a PhysicalLaw.")
    mean = jnp.asarray(mean_loss)
    if jnp.issubdtype(mean.dtype, jnp.complexfloating):
        raise TypeError("Gaussian loss moments must be real-valued.")
    mean = mean.astype(jnp.result_type(mean.dtype, jnp.float32))
    deviation = jnp.asarray(standard_deviation, dtype=mean.dtype)
    if mean.shape != () or deviation.shape != ():
        raise ValueError("mean_loss and standard_deviation must be scalars.")
    if (
        not bool(np.isfinite(float(np.asarray(mean))))
        or not bool(np.isfinite(float(np.asarray(deviation))))
        or float(np.asarray(deviation)) < 0.0
    ):
        raise ValueError(
            "Gaussian loss moments must be finite with non-negative deviation."
        )
    level = _confidence(confidence)
    z = ndtri(jnp.asarray(level, dtype=mean.dtype))
    density = jnp.exp(-0.5 * z * z) / jnp.sqrt(jnp.asarray(2.0 * pi, dtype=mean.dtype))
    return TailRiskEstimate(
        value_at_risk=mean + deviation * z,
        expected_shortfall=mean + deviation * density / (1.0 - level),
        confidence=level,
        law=law,
        method="gaussian",
    )


def entropic_value_at_risk(
    losses: ArrayLike,
    probabilities: ArrayLike,
    confidence: float,
    law: PhysicalLaw,
    /,
) -> EntropicRiskEstimate:
    """Compute finite-law EVaR by monotone KL-temperature bisection."""

    if not isinstance(law, PhysicalLaw):
        raise TypeError("EVaR requires a PhysicalLaw.")
    loss, probability = _loss_probability(losses, probabilities)
    level = _confidence(confidence)
    radius = -float(np.log1p(-level))
    loss_host, probability_host = np.asarray(loss), np.asarray(probability)
    spread = float(np.max(loss_host) - np.min(loss_host))
    if spread == 0.0:
        return EntropicRiskEstimate(
            value=loss[0],
            optimal_temperature=jnp.asarray(jnp.inf, dtype=loss.dtype),
            relative_entropy_radius=radius,
            law=law,
        )
    log_probability = np.full(probability_host.shape, -np.inf, dtype=loss_host.dtype)
    positive_probability = probability_host > 0.0
    log_probability[positive_probability] = np.log(probability_host[positive_probability])
    lower = max(np.finfo(loss_host.dtype).eps * spread, np.finfo(loss_host.dtype).tiny)
    upper = spread * 1e8
    for _ in range(96):
        temperature = np.sqrt(lower * upper)
        logits = loss_host / temperature + log_probability
        maximum = float(np.max(logits))
        tilted = np.exp(logits - maximum)
        tilted /= np.sum(tilted)
        positive = tilted > 0.0
        divergence = float(
            np.sum(
                tilted[positive] * (np.log(tilted[positive]) - log_probability[positive])
            )
        )
        if divergence > radius:
            lower = temperature
        else:
            upper = temperature
    temperature = upper
    logits = loss_host / temperature + log_probability
    maximum = float(np.max(logits))
    log_partition = maximum + float(np.log(np.sum(np.exp(logits - maximum))))
    value = temperature * (radius + log_partition)
    return EntropicRiskEstimate(
        value=jnp.asarray(value, dtype=loss.dtype),
        optimal_temperature=jnp.asarray(temperature, dtype=loss.dtype),
        relative_entropy_radius=radius,
        law=law,
    )


def spectral_risk(
    losses: ArrayLike,
    probabilities: ArrayLike,
    confidences: ArrayLike,
    weights: ArrayLike,
    /,
) -> Array:
    """Evaluate a normalized positive mixture of finite-law ES atoms."""

    loss, probability = _loss_probability(losses, probabilities)
    levels = np.asarray(confidences, dtype=np.dtype(loss.dtype))
    masses = np.asarray(weights, dtype=np.dtype(loss.dtype))
    if levels.ndim != 1 or masses.shape != levels.shape or levels.size == 0:
        raise ValueError("confidences and weights must be same-sized non-empty vectors.")
    if np.any(~np.isfinite(levels)) or np.any((levels <= 0.0) | (levels >= 1.0)):
        raise ValueError("Spectral confidence levels must lie in (0, 1).")
    if np.any(~np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("Spectral weights must be finite and non-negative.")
    tolerance = 128.0 * np.finfo(masses.dtype).eps
    if abs(float(np.sum(masses)) - 1.0) > tolerance:
        raise ValueError("Spectral weights must sum to one.")
    result = jnp.asarray(0.0, dtype=loss.dtype)
    for level, mass in zip(levels, masses, strict=True):
        result = (
            result
            + float(mass) * cvar_atoms(loss, probability, float(level)).expected_shortfall
        )
    return result


def kelly_risk(
    simple_returns: ArrayLike,
    probabilities: ArrayLike,
    law: PhysicalLaw,
    /,
    *,
    floor: float = 0.0,
) -> KellyRisk:
    """Expose bankruptcy explicitly instead of clipping invalid logarithmic wealth."""

    if not isinstance(law, PhysicalLaw):
        raise TypeError("Kelly risk requires a PhysicalLaw.")
    returns, probability = _loss_probability(simple_returns, probabilities)
    floor_ = float(floor)
    if not isfinite(floor_) or floor_ < 0.0:
        raise ValueError("floor must be finite and non-negative.")
    gross = 1.0 + returns
    bankrupt = gross <= floor_
    safe = jnp.where(bankrupt, 1.0, gross)
    expected = jnp.where(
        jnp.any(bankrupt & (probability > 0.0)),
        -jnp.inf,
        jnp.sum(probability * jnp.log(safe)),
    )
    return KellyRisk(
        expected_log_growth=expected,
        gross_returns=gross,
        bankrupt=bankrupt,
        bankruptcy_probability=jnp.sum(jnp.where(bankrupt, probability, 0.0)),
        floor=floor_,
        law=law,
    )


def drawdown_risk(wealth: ArrayLike, /) -> DrawdownRisk:
    """Return pathwise relative drawdown and longest underwater duration."""

    value = jnp.asarray(wealth)
    if value.ndim not in (1, 2) or value.shape[-1] == 0:
        raise ValueError("wealth must be a non-empty path or batch of paths.")
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise TypeError("wealth must be real-valued.")
    value = value.astype(jnp.result_type(value.dtype, jnp.float32))
    if not np.all(np.isfinite(np.asarray(value))) or np.any(np.asarray(value) <= 0.0):
        raise ValueError("wealth must be finite and strictly positive.")
    peak = jnp.maximum.accumulate(value, axis=-1)
    drawdown = (peak - value) / peak
    maximum = jnp.max(drawdown, axis=-1)
    underwater = np.asarray(drawdown) > 0.0
    flat = underwater.reshape((-1, underwater.shape[-1]))
    durations = []
    for path in flat:
        longest = 0
        current = 0
        for active in path:
            current = current + 1 if active else 0
            longest = max(longest, current)
        durations.append(longest)
    duration_shape = underwater.shape[:-1]
    duration = np.asarray(durations, dtype=np.int32).reshape(duration_shape)
    return DrawdownRisk(
        wealth=value,
        running_peak=peak,
        drawdown=drawdown,
        maximum_drawdown=maximum,
        maximum_duration=jnp.asarray(duration),
    )


__all__ = [
    "CVaRAtoms",
    "DrawdownRisk",
    "EntropicRiskEstimate",
    "KellyRisk",
    "TailRiskEstimate",
    "cvar_atoms",
    "expected_shortfall",
    "drawdown_risk",
    "entropic_value_at_risk",
    "gaussian_var_es",
    "historical_var_es",
    "kelly_risk",
    "spectral_risk",
    "value_at_risk",
]
