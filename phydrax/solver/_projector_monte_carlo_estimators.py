#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical projector estimators composed with the shared correlated-ratio owner."""

from __future__ import annotations

import math
from enum import IntFlag
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._strict import StrictModule
from ..typing import Bool, Dim, Float64, Int32, Int64, Scalar
from ..units import UnitDefinition
from ..uq._correlated_ratio import (
    correlated_ratio_of_means,
    CorrelatedRatioPolicy,
    CorrelatedRatioResult,
)
from ._projector_monte_carlo_contracts import (
    HistoryDim,
    PreparedProjectorMonteCarlo,
    ProjectorMonteCarloHistory,
    ProjectorMonteCarloResult,
)


class WeightStreamDim(Dim):
    """Aligned streams in a weight diagnostic (replicas or ordered pairs)."""


class EstimateDim(Dim):
    """Physical Hamiltonian followed by requested observables."""


class ProjectorWeightStatus(IntFlag):
    SUCCESS = 0
    INSUFFICIENT_HISTORY = 1 << 0
    NONFINITE = 1 << 1
    NUMERICAL_RANGE = 1 << 2
    CONCENTRATED = 1 << 3


@final
class ProjectorEstimatorPolicy(StrictModule):
    """Analysis-only selection, denominator/correlation policy, and history depths.

    ``burn_in`` counts discarded accepted transitions; cadence is anchored at
    the first retained state. ``deterministic_records`` is an explicit source
    declaration, never inferred from observed zero variation.
    """

    ratio_policy: CorrelatedRatioPolicy
    burn_in: int = eqx.field(static=True)
    cadence: int = eqx.field(static=True)
    history_depths: tuple[int, ...] = eqx.field(static=True)
    reference_energy: float = eqx.field(static=True)
    minimum_weight_ess: float = eqx.field(static=True)
    minimum_weight_fraction: float = eqx.field(static=True)
    deterministic_records: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        ratio_policy: CorrelatedRatioPolicy | None = None,
        burn_in: int = 0,
        cadence: int = 1,
        history_depths: tuple[int, ...] = (0,),
        reference_energy: float = 0.0,
        minimum_weight_ess: float = 2.0,
        minimum_weight_fraction: float = 0.01,
        deterministic_records: bool = False,
    ) -> None:
        if not isinstance(burn_in, int) or isinstance(burn_in, bool) or burn_in < 0:
            raise ValueError("burn_in must be a nonnegative integer.")
        if not isinstance(cadence, int) or isinstance(cadence, bool) or cadence < 1:
            raise ValueError("cadence must be a positive integer.")
        if not isinstance(history_depths, tuple) or any(
            not isinstance(h, int) or isinstance(h, bool) or h < 0 for h in history_depths
        ):
            raise ValueError("history_depths must be a tuple of nonnegative integers.")
        if len(set(history_depths)) != len(history_depths):
            raise ValueError("history_depths must be distinct.")
        if not math.isfinite(reference_energy):
            raise ValueError("reference_energy must be finite.")
        if not math.isfinite(minimum_weight_ess) or minimum_weight_ess < 1:
            raise ValueError("minimum_weight_ess must be finite and at least one.")
        if (
            not math.isfinite(minimum_weight_fraction)
            or not 0 <= minimum_weight_fraction <= 1
        ):
            raise ValueError("minimum_weight_fraction must lie in [0,1].")
        if not isinstance(deterministic_records, bool):
            raise TypeError("deterministic_records must be boolean.")
        if ratio_policy is not None and not isinstance(
            ratio_policy, CorrelatedRatioPolicy
        ):
            raise TypeError("ratio_policy must be a CorrelatedRatioPolicy.")
        self.ratio_policy = (
            CorrelatedRatioPolicy(max_lag=64) if ratio_policy is None else ratio_policy
        )
        self.burn_in = burn_in
        self.cadence = cadence
        self.history_depths = history_depths
        self.reference_energy = reference_energy
        self.minimum_weight_ess = minimum_weight_ess
        self.minimum_weight_fraction = minimum_weight_fraction
        self.deterministic_records = deterministic_records


@final
class ProjectorWeightDiagnostics(StrictModule):
    """Weight ESS measures concentration, not temporal independence."""

    __strict_contract__ = True

    log_normalization: Float64[Scalar]
    effective_samples: Float64[WeightStreamDim]
    effective_fraction: Float64[WeightStreamDim]
    status: Int32[Scalar]
    successful: Bool[Scalar]
    retained_states: Int64[Scalar]
    excluded_incomplete_windows: int = eqx.field(static=True)


@final
class ProjectorReweightedEstimate(StrictModule):
    __strict_contract__ = True

    history_depth: int = eqx.field(static=True)
    history_horizon: float = eqx.field(static=True)
    measured_state_numbers: Int64[HistoryDim]
    projected: CorrelatedRatioResult
    replicas: tuple[CorrelatedRatioResult, ...]
    projected_weights: ProjectorWeightDiagnostics
    pair_weights: ProjectorWeightDiagnostics
    projected_statistically_valid: Bool[Scalar]
    replica_statistically_valid: Bool[EstimateDim]


@final
class ProjectorSystematicRecord(StrictModule):
    """Scientific assumptions kept distinct from statistical status/covariance."""

    scientific_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    guide_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)
    finite_history_claim: str = eqx.field(static=True)
    asymptotic_claim: str = eqx.field(static=True)


@final
class ProjectorMonteCarloAnalysis(StrictModule):
    __strict_contract__ = True

    shifts: CorrelatedRatioResult
    projected: CorrelatedRatioResult
    replicas: tuple[CorrelatedRatioResult, ...]
    reweighted: tuple[ProjectorReweightedEstimate, ...]
    measured_state_numbers: Int64[HistoryDim]
    propagation_status: Int32[Scalar]
    systematic: ProjectorSystematicRecord
    observable_ids: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[UnitDefinition, ...] = eqx.field(static=True)


def _validated_history(
    prepared: PreparedProjectorMonteCarlo,
    records: ProjectorMonteCarloResult | ProjectorMonteCarloHistory,
) -> tuple[ProjectorMonteCarloHistory, Array, int]:
    match records:
        case ProjectorMonteCarloResult():
            history = records.state.history
            propagation_status = records.status
        case ProjectorMonteCarloHistory():
            history = records
            propagation_status = jnp.asarray(-1, dtype=jnp.int32)
        case _:
            raise TypeError("records must be a projector result or typed raw history.")
    expected = (
        prepared.scientific_id,
        prepared.original_operator.domain.domain_id,
        prepared.original_operator.operator_id,
        "unguided" if prepared.guide is None else prepared.guide.guide_id,
        "euclidean" if prepared.guide is None else prepared.guide.metric_id,
    )
    actual = (
        history.scientific_id,
        history.domain_id,
        history.operator_id,
        history.guide_id,
        history.metric_id,
    )
    if actual != expected:
        raise ValueError(
            "Projector history scientific/domain/operator/guide/metric identity mismatch."
        )
    count = int(np.asarray(history.count))
    if not 0 <= count <= history.valid.shape[0]:
        raise ValueError("Projector history count lies outside its retained capacity.")
    if not np.all(np.asarray(history.valid[:count])):
        raise ValueError(
            "Committed projector history must retain every applied shift without gaps."
        )
    if history.applied_shifts.shape[0] != prepared.plan.replicas:
        raise ValueError("Projector history replica count does not match preparation.")
    if history.pair_numerators.shape[0] != len(prepared.pair_ids):
        raise ValueError(
            "Projector history ordered pair count does not match preparation."
        )
    if history.pair_numerators.shape[-1] != 1 + len(prepared.observables):
        raise ValueError("Projector history observable count does not match preparation.")
    return history, propagation_status, count


def _ratio(
    numerator: Array,
    denominator: Array,
    prepared: PreparedProjectorMonteCarlo,
    policy: ProjectorEstimatorPolicy,
    *,
    paired: bool,
    mean: bool = False,
) -> CorrelatedRatioResult:
    ids = tuple(f"replica-{r}" for r in range(numerator.shape[0]))
    if paired:
        ids = ("ordered-pair-aggregate",)
    ratio_policy = policy.ratio_policy
    if mean:
        # A mean's denominator is exactly one, not a physical overlap subject
        # to the user's overlap magnitude/sign admission floor.
        ratio_policy = CorrelatedRatioPolicy(
            max_lag=ratio_policy.max_lag,
            minimum_draws=ratio_policy.minimum_draws,
            minimum_blocks=ratio_policy.minimum_blocks,
            block_length=ratio_policy.block_length,
            confidence_multiplier=ratio_policy.confidence_multiplier,
            minimum_denominator_magnitude=0,
            denominator_sign="positive",
        )
    return correlated_ratio_of_means(
        numerator,
        denominator,
        policy=ratio_policy,
        stream_ids=ids,
        dependence_ids=ids,
        sampling_origin_id=prepared.scientific_id,
        deterministic=policy.deterministic_records,
    )


def _log_weights(
    applied_shifts: Array,
    measured_indices: Array,
    history_depth: int,
    dt: float,
    reference_energy: float,
) -> Array:
    """Row j measures state j+1, reached using applied_shifts[:,j]."""
    if history_depth == 0:
        return jnp.zeros(
            (applied_shifts.shape[0], measured_indices.shape[0]), dtype=jnp.float64
        )
    centered = applied_shifts - reference_energy
    prefix = jnp.concatenate(
        (
            jnp.zeros((centered.shape[0], 1), dtype=jnp.float64),
            jnp.cumsum(centered, axis=1),
        ),
        axis=1,
    )
    ends = measured_indices + 1
    return -dt * (prefix[:, ends] - prefix[:, ends - history_depth])


def _normalize_weights(
    log_weights: Array,
    policy: ProjectorEstimatorPolicy,
    excluded: int,
) -> tuple[Array, ProjectorWeightDiagnostics]:
    streams, draws = log_weights.shape
    if draws == 0 or streams == 0:
        status = jnp.asarray(
            int(ProjectorWeightStatus.INSUFFICIENT_HISTORY), dtype=jnp.int32
        )
        normalization = jnp.asarray(jnp.nan, dtype=jnp.float64)
        weights = jnp.zeros_like(log_weights)
        ess = jnp.zeros((streams,), dtype=jnp.float64)
        fraction = jnp.zeros_like(ess)
    else:
        maximum = jnp.max(log_weights)
        shifted = jnp.exp(log_weights - maximum)
        total = jnp.sum(shifted)
        # Common unit-mean normalization preserves relative replica/pair
        # weights and keeps denominator floors independent of E_ref.
        weights = shifted * ((streams * draws) / total)
        normalization = maximum + jnp.log(total / (streams * draws))
        sums = jnp.sum(weights, axis=1)
        squares = jnp.sum(weights * weights, axis=1)
        ess = sums * sums / squares
        fraction = ess / draws
        finite = (
            jnp.all(jnp.isfinite(log_weights))
            & jnp.all(jnp.isfinite(weights))
            & jnp.all(jnp.isfinite(ess))
        )
        in_range = jnp.all(weights > 0)
        concentrated = jnp.any(ess < policy.minimum_weight_ess) | jnp.any(
            fraction < policy.minimum_weight_fraction
        )
        status = (
            jnp.where(finite, 0, int(ProjectorWeightStatus.NONFINITE))
            | jnp.where(in_range, 0, int(ProjectorWeightStatus.NUMERICAL_RANGE))
            | jnp.where(concentrated, int(ProjectorWeightStatus.CONCENTRATED), 0)
        ).astype(jnp.int32)
    diagnostic = ProjectorWeightDiagnostics(
        log_normalization=normalization,
        effective_samples=ess,
        effective_fraction=fraction,
        status=status,
        successful=status == 0,
        retained_states=jnp.asarray(draws, dtype=jnp.int64),
        excluded_incomplete_windows=excluded,
    )
    return weights, diagnostic


def _reweighted(
    prepared: PreparedProjectorMonteCarlo,
    history: ProjectorMonteCarloHistory,
    policy: ProjectorEstimatorPolicy,
    indices: Array,
    count: int,
    depth: int,
) -> ProjectorReweightedEstimate:
    # Preserve pre-burn-in shifts; only measured states lacking a full window go.
    selected = indices[indices + 1 >= depth]
    excluded = indices.shape[0] - selected.shape[0]
    logs = _log_weights(
        history.applied_shifts[:, :count],
        selected,
        depth,
        prepared.problem.dt,
        policy.reference_energy,
    )
    weights, projected_diagnostic = _normalize_weights(logs, policy, excluded)
    pair_logs = (
        jnp.stack(tuple(logs[a] + logs[b] for a, b in prepared.pair_ids))
        if prepared.pair_ids
        else jnp.zeros((0, selected.shape[0]), dtype=jnp.float64)
    )
    pair_weights, pair_diagnostic = _normalize_weights(pair_logs, policy, excluded)
    complex_weights = weights.astype(jnp.complex128)
    complex_pair_weights = pair_weights.astype(jnp.complex128)
    projected = _ratio(
        complex_weights * history.projected_numerator[:, selected],
        complex_weights * history.projected_denominator[:, selected],
        prepared,
        policy,
        paired=False,
    )
    denominator = jnp.sum(
        complex_pair_weights * history.pair_denominators[:, selected],
        axis=0,
        keepdims=True,
    )
    numerators = jnp.sum(
        complex_pair_weights[:, :, None] * history.pair_numerators[:, selected, :], axis=0
    )
    replicas = tuple(
        _ratio(numerators[:, o][None, :], denominator, prepared, policy, paired=True)
        for o in range(numerators.shape[-1])
    )
    return ProjectorReweightedEstimate(
        history_depth=depth,
        history_horizon=depth * prepared.problem.dt,
        measured_state_numbers=selected + 1,
        projected=projected,
        replicas=replicas,
        projected_weights=projected_diagnostic,
        pair_weights=pair_diagnostic,
        projected_statistically_valid=projected.statistically_valid
        & projected_diagnostic.successful,
        replica_statistically_valid=jnp.stack(
            tuple(r.statistically_valid & pair_diagnostic.successful for r in replicas)
        ),
    )


def analyze_projector_monte_carlo(
    prepared: PreparedProjectorMonteCarlo,
    records: ProjectorMonteCarloResult | ProjectorMonteCarloHistory,
    *,
    policy: ProjectorEstimatorPolicy,
) -> ProjectorMonteCarloAnalysis:
    """Analyze raw aligned histories, never instantaneous or independent-pair ratios.

    This is an explicit host analysis boundary. Propagation refusal, statistical
    qualification, and finite-population/history assumptions remain separate.
    A raw history has no driver completion status: propagation_status is -1.
    """
    if not isinstance(policy, ProjectorEstimatorPolicy):
        raise TypeError("policy must be a ProjectorEstimatorPolicy.")
    history, propagation_status, count = _validated_history(prepared, records)
    indices = jnp.arange(policy.burn_in, count, policy.cadence, dtype=jnp.int64)
    shifts = _ratio(
        history.applied_shifts[:, indices],
        jnp.ones((prepared.plan.replicas, indices.shape[0]), dtype=jnp.float64),
        prepared,
        policy,
        paired=False,
        mean=True,
    )
    projected = _ratio(
        history.projected_numerator[:, indices],
        history.projected_denominator[:, indices],
        prepared,
        policy,
        paired=False,
    )
    denominator = jnp.sum(history.pair_denominators[:, indices], axis=0, keepdims=True)
    numerators = jnp.sum(history.pair_numerators[:, indices, :], axis=0)
    replicas = tuple(
        _ratio(numerators[:, o][None, :], denominator, prepared, policy, paired=True)
        for o in range(numerators.shape[-1])
    )
    reweighted = tuple(
        _reweighted(prepared, history, policy, indices, count, h)
        for h in policy.history_depths
    )
    systematic = ProjectorSystematicRecord(
        scientific_id=history.scientific_id,
        domain_id=history.domain_id,
        operator_id=history.operator_id,
        guide_id=history.guide_id,
        metric_id=history.metric_id,
        dt=prepared.problem.dt,
        assumptions=(
            "Stationarity and relevant physical relaxation must be established independently of small error bars.",
            "Population feedback, finite population, sign resolution, and hard support/resource limits require separate qualification.",
            "No initiator approximation; conditional per-step spawning and late compression are unbiased algorithm operations, not an unbiased stationary-estimator certificate.",
            "Weight ESS measures concentration; temporal and influence diagnostics remain in each correlated-ratio result.",
            "Encountered guide admissibility does not certify global provider support; the original physical domain and frozen guide/metric define the scope.",
        ),
        finite_history_claim="Finite exponential history weights do not exactly cancel the additive finite-step Euler projector and are not advertised unbiased.",
        asymptotic_claim="Any applicable bias-removal claim requires dt -> 0, h -> infinity, a horizon h*dt resolving physical relaxation, and population/sign/stationarity assumptions; no universal order-independent joint limit is asserted.",
    )
    return ProjectorMonteCarloAnalysis(
        shifts=shifts,
        projected=projected,
        replicas=replicas,
        reweighted=reweighted,
        measured_state_numbers=indices + 1,
        propagation_status=propagation_status,
        systematic=systematic,
        observable_ids=tuple(
            op.operator_id for op in (prepared.original_operator,) + prepared.observables
        ),
        observable_units=prepared.observable_units,
    )
