#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Path-payoff Monte Carlo, randomized QMC and MLMC valuation routes."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...integration._multilevel import MultilevelSampleBatch
from ...integration._plans import QuasiMonteCarloPlan
from ..contracts._base import AbstractPayoff
from ..contracts._options import (
    AsianPayoff,
    AverageType,
    BarrierActivation,
    BarrierDirection,
    BasketPayoff,
    BermudanPayoff,
    CashDigitalPayoff,
    LookbackPayoff,
    OptionType,
    PathBarrierPayoff,
    VanillaPayoff,
    VarianceSwapPayoff,
)
from ..core._currency import Currency
from ..core._evidence import FinanceEvidenceBinding
from ..core._laws import PricingLaw
from ._types import ValuationEvidence, ValuationResult


class MonteCarloPathBatch(StrictModule):
    """Fixed-shape positive asset paths with prefix validity by path."""

    times: Array
    values: Array
    valid: Array
    path_count: int = eqx.field(static=True)
    time_count: int = eqx.field(static=True)
    asset_count: int = eqx.field(static=True)
    path_id: str = eqx.field(static=True)

    def __init__(
        self, times: ArrayLike, values: ArrayLike, valid: ArrayLike, /, *, path_id: str
    ):
        times_ = jnp.asarray(times, dtype=float)
        values_ = jnp.asarray(values, dtype=float)
        valid_ = jnp.asarray(valid, dtype=bool)
        if times_.ndim != 1 or times_.size < 2:
            raise ValueError("times must be a vector with at least two nodes.")
        if (
            values_.ndim != 3
            or values_.shape[1] != times_.size
            or values_.shape[0] < 2
            or values_.shape[2] < 1
        ):
            raise ValueError(
                "values must have shape (path, time, asset) with at least two paths."
            )
        if valid_.shape != values_.shape[:2]:
            raise ValueError("valid must have shape (path, time).")
        if not isinstance(path_id, str) or not path_id:
            raise ValueError("path_id must be a non-empty string.")
        times_ = eqx.error_if(
            times_,
            jnp.any(~jnp.isfinite(times_)) | jnp.any(jnp.diff(times_) <= 0.0),
            "path times must be finite and strictly increasing.",
        )
        counts = jnp.sum(valid_, axis=1, dtype=jnp.int32)
        prefix = jnp.arange(times_.size)[None, :] < counts[:, None]
        valid_ = eqx.error_if(
            valid_,
            jnp.any(valid_ != prefix) | jnp.any(counts < 2),
            "each path validity row must be a prefix with at least two nodes.",
        )
        values_ = eqx.error_if(
            values_,
            jnp.any(valid_[..., None] & (~jnp.isfinite(values_) | (values_ <= 0.0)))
            | jnp.any(~valid_[..., None] & (values_ != 0.0)),
            "active path values must be finite/positive and inactive values zero.",
        )
        self.times, self.values, self.valid = times_, values_, valid_
        self.path_count, self.time_count, self.asset_count = map(int, values_.shape)
        self.path_id = path_id

    @property
    def path_valid(self) -> Array:
        return jnp.all(self.valid, axis=1)


class MonteCarloValuationPlan(StrictModule):
    minimum_paths: int = eqx.field(static=True)
    confidence_level: float = eqx.field(static=True)

    def __init__(self, *, minimum_paths: int = 32, confidence_level: float = 0.95):
        if (
            isinstance(minimum_paths, bool)
            or not isinstance(minimum_paths, int)
            or minimum_paths < 2
        ):
            raise ValueError("minimum_paths must be an integer at least two.")
        confidence = float(confidence_level)
        if not isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0, 1).")
        self.minimum_paths = minimum_paths
        self.confidence_level = confidence


class PreparedMonteCarlo(StrictModule):
    paths: MonteCarloPathBatch
    payoff: AbstractPayoff
    discount_factor: Array
    plan: MonteCarloValuationPlan
    currency: Currency | None = eqx.field(static=True)
    evidence_binding: FinanceEvidenceBinding | None
    pricing_law: PricingLaw | None


class PreparedQMCValuation(StrictModule):
    replicate_paths: tuple[MonteCarloPathBatch, ...]
    payoff: AbstractPayoff
    discount_factor: Array
    plan: QuasiMonteCarloPlan
    currency: Currency | None = eqx.field(static=True)
    evidence_binding: FinanceEvidenceBinding | None
    pricing_law: PricingLaw | None


class MLMCValuationPlan(StrictModule):
    minimum_samples_per_level: int = eqx.field(static=True)

    def __init__(self, *, minimum_samples_per_level: int = 2):
        if (
            isinstance(minimum_samples_per_level, bool)
            or not isinstance(minimum_samples_per_level, int)
            or minimum_samples_per_level < 2
        ):
            raise ValueError("minimum_samples_per_level must be an integer at least two.")
        self.minimum_samples_per_level = minimum_samples_per_level


class MonteCarloDiagnostics(StrictModule):
    sample_count: Array
    failed_count: Array
    sample_variance: Array
    standard_error: Array
    confidence_half_width: Array
    path_id: str = eqx.field(static=True)
    route: str = eqx.field(static=True)


class QMCValuationDiagnostics(StrictModule):
    replicate_means: Array
    replicate_count: int = eqx.field(static=True)
    samples_per_replicate: int = eqx.field(static=True)
    sequence: str = eqx.field(static=True)


class MLMCValuationDiagnostics(StrictModule):
    correction_means: Array
    correction_variances: Array
    sample_counts: Array
    standard_error: Array
    bias_estimate: Array
    level_count: int = eqx.field(static=True)


def _validate_payoff(payoff: AbstractPayoff) -> None:
    if not isinstance(
        payoff,
        (
            VanillaPayoff,
            CashDigitalPayoff,
            PathBarrierPayoff,
            AsianPayoff,
            LookbackPayoff,
            BasketPayoff,
            VarianceSwapPayoff,
            BermudanPayoff,
        ),
    ):
        raise ValueError("unsupported payoff for the stochastic path route.")


def _validate_metadata(
    currency: Currency | None,
    evidence_binding: FinanceEvidenceBinding | None,
    pricing_law: PricingLaw | None,
    /,
) -> None:
    if currency is not None and not isinstance(currency, Currency):
        raise TypeError("currency must be Currency or None.")
    if evidence_binding is not None and not isinstance(
        evidence_binding, FinanceEvidenceBinding
    ):
        raise TypeError("evidence_binding must be FinanceEvidenceBinding or None.")
    if pricing_law is not None and not isinstance(pricing_law, PricingLaw):
        raise TypeError("pricing_law must be PricingLaw or None.")


def evaluate_path_payoff(
    paths: MonteCarloPathBatch, payoff: AbstractPayoff, /
) -> tuple[Array, Array]:
    """Evaluate one supported payoff per path without changing sampling semantics."""

    if not isinstance(paths, MonteCarloPathBatch):
        raise TypeError("paths must be a MonteCarloPathBatch.")
    _validate_payoff(payoff)
    path_valid = paths.path_valid
    terminal = paths.values[:, -1]
    if isinstance(payoff, (VanillaPayoff, BermudanPayoff)):
        sign = 1.0 if payoff.option_type is OptionType.CALL else -1.0
        values = payoff.notional * jnp.maximum(
            sign * (terminal[:, 0] - payoff.strike), 0.0
        )
    elif isinstance(payoff, CashDigitalPayoff):
        sign = 1.0 if payoff.option_type is OptionType.CALL else -1.0
        values = payoff.cash_amount * (sign * (terminal[:, 0] - payoff.strike) > 0.0)
    elif isinstance(payoff, PathBarrierPayoff):
        monitored = paths.values[:, :, 0]
        touched = (
            jnp.any(monitored >= payoff.barrier, axis=1)
            if payoff.direction is BarrierDirection.UP
            else jnp.any(monitored <= payoff.barrier, axis=1)
        )
        active = touched if payoff.activation is BarrierActivation.KNOCK_IN else ~touched
        sign = 1.0 if payoff.option_type is OptionType.CALL else -1.0
        vanilla = payoff.notional * jnp.maximum(
            sign * (terminal[:, 0] - payoff.strike), 0.0
        )
        values = jnp.where(active, vanilla, payoff.rebate)
    elif isinstance(payoff, AsianPayoff):
        monitored = paths.values[:, :, 0]
        if payoff.average_type is AverageType.ARITHMETIC:
            average = jnp.mean(monitored, axis=1)
        else:
            average = jnp.exp(jnp.mean(jnp.log(monitored), axis=1))
        sign = 1.0 if payoff.option_type is OptionType.CALL else -1.0
        values = payoff.notional * jnp.maximum(sign * (average - payoff.strike), 0.0)
    elif isinstance(payoff, LookbackPayoff):
        monitored = paths.values[:, :, 0]
        extremum = (
            jnp.max(monitored, axis=1)
            if payoff.option_type is OptionType.CALL
            else jnp.min(monitored, axis=1)
        )
        sign = 1.0 if payoff.option_type is OptionType.CALL else -1.0
        values = payoff.notional * jnp.maximum(sign * (extremum - payoff.strike), 0.0)
    elif isinstance(payoff, BasketPayoff):
        if paths.asset_count != payoff.asset_count:
            raise ValueError("basket payoff weights do not match path asset count.")
        basket = terminal @ payoff.weights
        sign = 1.0 if payoff.option_type is OptionType.CALL else -1.0
        values = payoff.notional * jnp.maximum(sign * (basket - payoff.strike), 0.0)
    else:
        if paths.asset_count != 1:
            raise ValueError("variance-swap path payoff requires one asset.")
        log_returns = jnp.diff(jnp.log(paths.values[:, :, 0]), axis=1)
        horizon = paths.times[-1] - paths.times[0]
        realized_variance = jnp.sum(log_returns**2, axis=1) / horizon
        values = payoff.variance_notional * (realized_variance - payoff.variance_strike)
    return jnp.where(path_valid, values, 0.0), path_valid & jnp.isfinite(values)


def prepare_monte_carlo(
    paths: MonteCarloPathBatch,
    payoff: AbstractPayoff,
    discount_factor: ArrayLike,
    plan: MonteCarloValuationPlan,
    /,
    *,
    currency: Currency | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
    pricing_law: PricingLaw | None = None,
) -> PreparedMonteCarlo:
    if not isinstance(paths, MonteCarloPathBatch) or not isinstance(
        plan, MonteCarloValuationPlan
    ):
        raise TypeError("paths and plan must be Monte Carlo records.")
    _validate_payoff(payoff)
    _validate_metadata(currency, evidence_binding, pricing_law)
    discount = jnp.asarray(discount_factor, dtype=float)
    if discount.shape != ():
        raise ValueError("discount_factor must be scalar.")
    discount = eqx.error_if(
        discount,
        ~jnp.isfinite(discount) | (discount <= 0.0) | (discount > 1.0),
        "discount_factor must lie in (0, 1].",
    )
    return PreparedMonteCarlo(
        paths, payoff, discount, plan, currency, evidence_binding, pricing_law
    )


def _mean_evidence(
    values, valid, discount, minimum_paths, confidence_level, route, path_id, binding, law
):
    count = jnp.sum(valid, dtype=jnp.int32)
    safe_count = jnp.maximum(count, 1)
    mean = discount * jnp.sum(jnp.where(valid, values, 0.0)) / safe_count
    centered = jnp.where(valid, discount * values - mean, 0.0)
    variance = jnp.sum(centered**2) / jnp.maximum(count - 1, 1)
    standard_error = jnp.sqrt(variance / safe_count)
    enough = count >= minimum_paths
    mean = eqx.error_if(
        mean,
        ~enough | ~jnp.isfinite(mean) | ~jnp.isfinite(standard_error),
        "stochastic valuation has too few valid finite paths.",
    )
    critical = jsp.special.ndtri(jnp.asarray(0.5 + 0.5 * confidence_level))
    evidence = ValuationEvidence(
        route=route,
        finite=jnp.isfinite(mean),
        converged=enough,
        error_estimate=standard_error,
        binding=binding,
        pricing_law=law,
    )
    diagnostics = MonteCarloDiagnostics(
        count,
        values.size - count,
        variance,
        standard_error,
        critical * standard_error,
        path_id,
        route,
    )
    return ValuationResult(
        mean, evidence, standard_error=standard_error, diagnostics=diagnostics
    )


def evaluate_monte_carlo(prepared: PreparedMonteCarlo, /) -> ValuationResult:
    if not isinstance(prepared, PreparedMonteCarlo):
        raise TypeError("evaluate_monte_carlo requires PreparedMonteCarlo.")
    values, valid = evaluate_path_payoff(prepared.paths, prepared.payoff)
    result = _mean_evidence(
        values,
        valid,
        prepared.discount_factor,
        prepared.plan.minimum_paths,
        prepared.plan.confidence_level,
        "monte-carlo",
        prepared.paths.path_id,
        prepared.evidence_binding,
        prepared.pricing_law,
    )
    return ValuationResult(
        result.value,
        result.evidence,
        standard_error=result.standard_error,
        currency=prepared.currency,
        diagnostics=result.diagnostics,
    )


def prepare_qmc_monte_carlo(
    replicate_paths: Sequence[MonteCarloPathBatch],
    payoff: AbstractPayoff,
    discount_factor: ArrayLike,
    plan: QuasiMonteCarloPlan,
    /,
    *,
    currency: Currency | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
    pricing_law: PricingLaw | None = None,
) -> PreparedQMCValuation:
    paths = tuple(replicate_paths)
    if not isinstance(plan, QuasiMonteCarloPlan):
        raise TypeError("plan must be a QuasiMonteCarloPlan.")
    if not paths or any(not isinstance(value, MonteCarloPathBatch) for value in paths):
        raise TypeError("replicate_paths must contain MonteCarloPathBatch values.")
    expected_replicates = plan.design.num_replicates
    if len(paths) != expected_replicates:
        raise ValueError(
            "replicate path count must match randomized-QMC plan replicates."
        )
    if any(value.path_count != plan.num_samples for value in paths):
        raise ValueError("each QMC replicate must contain plan.num_samples paths.")
    _validate_payoff(payoff)
    _validate_metadata(currency, evidence_binding, pricing_law)
    discount = jnp.asarray(discount_factor, dtype=float)
    if discount.shape != ():
        raise ValueError("discount_factor must be scalar.")
    discount = eqx.error_if(
        discount,
        ~jnp.isfinite(discount) | (discount <= 0.0) | (discount > 1.0),
        "discount_factor must lie in (0, 1].",
    )
    return PreparedQMCValuation(
        paths, payoff, discount, plan, currency, evidence_binding, pricing_law
    )


def evaluate_qmc_monte_carlo(prepared: PreparedQMCValuation, /) -> ValuationResult:
    if not isinstance(prepared, PreparedQMCValuation):
        raise TypeError("evaluate_qmc_monte_carlo requires PreparedQMCValuation.")
    means = []
    all_valid = []
    for paths in prepared.replicate_paths:
        values, valid = evaluate_path_payoff(paths, prepared.payoff)
        count = jnp.sum(valid)
        enough = count >= 2
        mean = (
            prepared.discount_factor
            * jnp.sum(jnp.where(valid, values, 0.0))
            / jnp.maximum(count, 1)
        )
        means.append(mean)
        all_valid.append(enough & jnp.isfinite(mean))
    replicate_means = jnp.stack(tuple(means))
    valid = jnp.stack(tuple(all_valid))
    value = jnp.mean(replicate_means)
    standard_error = (
        jnp.std(replicate_means, ddof=1) / jnp.sqrt(replicate_means.size)
        if len(means) > 1
        else jnp.asarray(0.0)
    )
    value = eqx.error_if(
        value,
        ~jnp.all(valid) | ~jnp.isfinite(value) | ~jnp.isfinite(standard_error),
        "randomized-QMC replicate valuation is invalid.",
    )
    evidence = ValuationEvidence(
        route="randomized-qmc-monte-carlo",
        finite=jnp.isfinite(value),
        converged=jnp.all(valid),
        error_estimate=standard_error if len(means) > 1 else None,
        binding=prepared.evidence_binding,
        pricing_law=prepared.pricing_law,
    )
    diagnostics = QMCValuationDiagnostics(
        replicate_means,
        len(means),
        prepared.plan.num_samples,
        prepared.plan.design.sequence,
    )
    return ValuationResult(
        value,
        evidence,
        standard_error=standard_error if len(means) > 1 else None,
        currency=prepared.currency,
        diagnostics=diagnostics,
    )


def evaluate_mlmc(
    level_batches: Sequence[MultilevelSampleBatch],
    payoff: AbstractPayoff,
    discount_factor: ArrayLike,
    plan: MLMCValuationPlan,
    /,
    *,
    currency: Currency | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
    pricing_law: PricingLaw | None = None,
) -> ValuationResult:
    """Value coupled path levels carried by the generic integration MLMC substrate."""

    batches = tuple(level_batches)
    if not isinstance(plan, MLMCValuationPlan):
        raise TypeError("plan must be an MLMCValuationPlan.")
    if not batches or any(
        not isinstance(value, MultilevelSampleBatch) for value in batches
    ):
        raise TypeError("level_batches must contain MultilevelSampleBatch values.")
    if tuple(value.level_index for value in batches) != tuple(range(len(batches))):
        raise ValueError(
            "MLMC level batches must be ordered from level zero without gaps."
        )
    _validate_payoff(payoff)
    _validate_metadata(currency, evidence_binding, pricing_law)
    discount = jnp.asarray(discount_factor, dtype=float)
    if discount.shape != ():
        raise ValueError("discount_factor must be scalar.")
    discount = eqx.error_if(
        discount,
        ~jnp.isfinite(discount) | (discount <= 0.0) | (discount > 1.0),
        "discount_factor must lie in (0, 1].",
    )
    correction_means, correction_variances, counts = [], [], []
    for batch in batches:
        if not isinstance(batch.fine_samples, MonteCarloPathBatch):
            raise TypeError("MLMC fine_samples must be MonteCarloPathBatch values.")
        if batch.fine_samples.path_count != batch.num_samples:
            raise ValueError("MLMC fine path count must match batch sample count.")
        fine, fine_valid = evaluate_path_payoff(batch.fine_samples, payoff)
        valid = fine_valid & batch.fine_valid
        if batch.level_index == 0:
            correction = fine
        else:
            if (
                not isinstance(batch.coarse_samples, MonteCarloPathBatch)
                or batch.coarse_samples.path_count != batch.num_samples
            ):
                raise TypeError(
                    "positive MLMC levels require aligned coarse path batches."
                )
            coarse, coarse_valid = evaluate_path_payoff(batch.coarse_samples, payoff)
            valid = valid & coarse_valid & batch.coarse_valid
            correction = fine - coarse
        count = jnp.sum(valid, dtype=jnp.int32)
        mean = jnp.sum(jnp.where(valid, correction, 0.0)) / jnp.maximum(count, 1)
        variance = jnp.sum(jnp.where(valid, (correction - mean) ** 2, 0.0)) / jnp.maximum(
            count - 1, 1
        )
        correction_means.append(mean)
        correction_variances.append(variance)
        counts.append(count)
    means = jnp.stack(tuple(correction_means))
    variances = jnp.stack(tuple(correction_variances))
    sample_counts = jnp.stack(tuple(counts))
    enough = jnp.all(sample_counts >= plan.minimum_samples_per_level)
    standard_error = discount * jnp.sqrt(jnp.sum(variances / sample_counts))
    bias = discount * jnp.abs(means[-1])
    value = discount * jnp.sum(means)
    value = eqx.error_if(
        value,
        ~enough | ~jnp.isfinite(value) | ~jnp.isfinite(standard_error),
        "MLMC valuation has insufficient or non-finite coupled samples.",
    )
    evidence = ValuationEvidence(
        route="multilevel-monte-carlo",
        finite=jnp.isfinite(value),
        converged=enough,
        error_estimate=jnp.sqrt(standard_error**2 + bias**2),
        binding=evidence_binding,
        pricing_law=pricing_law,
    )
    diagnostics = MLMCValuationDiagnostics(
        discount * means,
        discount**2 * variances,
        sample_counts,
        standard_error,
        bias,
        len(batches),
    )
    return ValuationResult(
        value,
        evidence,
        standard_error=standard_error,
        currency=currency,
        diagnostics=diagnostics,
    )


__all__ = [
    "MLMCValuationDiagnostics",
    "MLMCValuationPlan",
    "MonteCarloDiagnostics",
    "MonteCarloPathBatch",
    "MonteCarloValuationPlan",
    "PreparedMonteCarlo",
    "PreparedQMCValuation",
    "QMCValuationDiagnostics",
    "evaluate_mlmc",
    "evaluate_monte_carlo",
    "evaluate_path_payoff",
    "evaluate_qmc_monte_carlo",
    "prepare_monte_carlo",
    "prepare_qmc_monte_carlo",
]
