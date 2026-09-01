#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class DemographyStatus(IntEnum):
    SUCCESS = 0
    INVALID_EVENT_ORDER = 1
    INVALID_LINEAGE_COUNT = 2
    IMPOSSIBLE_SPECTRUM = 3
    POLARIZATION_MISMATCH = 4
    NONFINITE = 5


class PiecewiseConstantDemography(StrictModule):
    epoch_end_times: Array
    effective_population_size: Array
    migration_rate: Array
    population_count: int = eqx.field(static=True)

    def __init__(
        self,
        epoch_end_times: ArrayLike,
        effective_population_size: ArrayLike,
        /,
        *,
        migration_rate: ArrayLike | None = None,
    ):
        ends = jnp.asarray(epoch_end_times)
        sizes = jnp.asarray(effective_population_size)
        if ends.ndim != 1 or sizes.ndim != 2 or sizes.shape[0] != ends.shape[0]:
            raise ValueError(
                "effective_population_size must have shape (epochs, populations)."
            )
        if ends.shape[0] < 1 or sizes.shape[1] < 1:
            raise ValueError("Demographies require at least one epoch and population.")
        if not jnp.issubdtype(ends.dtype, jnp.inexact):
            ends = ends.astype(float)
        if not jnp.issubdtype(sizes.dtype, jnp.inexact):
            sizes = sizes.astype(float)
        population_count = int(sizes.shape[1])
        migration = (
            jnp.zeros(
                (ends.shape[0], population_count, population_count), dtype=sizes.dtype
            )
            if migration_rate is None
            else jnp.asarray(migration_rate, dtype=sizes.dtype)
        )
        if migration.shape != (ends.shape[0], population_count, population_count):
            raise ValueError(
                "migration_rate must have shape (epochs, populations, populations)."
            )
        host_ends = np.asarray(ends)
        host_sizes = np.asarray(sizes)
        host_migration = np.asarray(migration)
        if (
            np.any(np.isnan(host_ends))
            or np.any(host_ends <= 0.0)
            or np.any(np.diff(host_ends) <= 0.0)
        ):
            raise ValueError("epoch_end_times must be positive and strictly increasing.")
        if not np.isposinf(host_ends[-1]):
            raise ValueError("The final demographic epoch must end at positive infinity.")
        if not np.all(np.isfinite(host_sizes)) or np.any(host_sizes <= 0.0):
            raise ValueError("effective_population_size must be finite and positive.")
        if not np.all(np.isfinite(host_migration)) or np.any(host_migration < 0.0):
            raise ValueError("migration_rate must be finite and non-negative.")
        diagonal = np.diagonal(host_migration, axis1=-2, axis2=-1)
        if np.any(diagonal != 0.0):
            raise ValueError("migration_rate diagonal entries must be zero.")
        self.epoch_end_times = ends
        self.effective_population_size = sizes
        self.migration_rate = migration
        self.population_count = population_count


class PairwiseCoalescentResult(StrictModule):
    log_density: Array
    log_survival: Array
    cumulative_hazard: Array
    epoch_index: Array
    valid: Array
    status: Array
    evidence: Array
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(DemographyStatus.SUCCESS))


class CoalescentLikelihoodResult(StrictModule):
    per_event_log_likelihood: Array
    total_log_likelihood: Array
    integrated_hazard: Array
    valid: Array
    status: Array
    evidence: Array
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(DemographyStatus.SUCCESS))


class ExpectedSFSResult(StrictModule):
    expected_spectrum: Array
    sample_size: Array
    theta: Array
    valid: Array
    status: Array
    evidence: Array
    folded: bool = eqx.field(static=True)
    polarized: bool = eqx.field(static=True)
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(DemographyStatus.SUCCESS))


class SFSLikelihoodResult(StrictModule):
    per_bin_log_likelihood: Array
    total_log_likelihood: Array
    normalized_expected: Array
    observed_total: Array
    valid: Array
    status: Array
    evidence: Array
    likelihood: str = eqx.field(static=True)
    folded: bool = eqx.field(static=True)
    polarized: bool = eqx.field(static=True)
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(DemographyStatus.SUCCESS))


def _contract(
    method_name: str, /, *, exact: bool, output: OutputKind
) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        method_name,
        MethodKind.EXACT_MODEL if exact else MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.EXACT_AD,
        output,
        conditioning_statement=(
            "Conditioned on the declared piecewise-constant demographic history or "
            "expected frequency spectrum and its polarization convention."
        ),
        truncation_statement="No epoch, event, or frequency bin is truncated.",
        capacity_semantics="Array lengths are exact event/epoch/frequency capacities.",
        assumptions=("Kingman coalescent scaling uses diploid rate 1/(2Ne).",),
        nondifferentiable_outputs=("epoch_index", "status", "valid"),
    )


def _require_unmigrated(demography: PiecewiseConstantDemography, /) -> None:
    if np.any(np.asarray(demography.migration_rate) != 0.0):
        raise ValueError(
            "Single-population coalescent likelihoods require zero migration; "
            "structured migration must be represented by an explicit lineage-state model."
        )


def _integrated_pair_hazard(
    times: Array,
    demography: PiecewiseConstantDemography,
    population: int,
    /,
) -> tuple[Array, Array, Array]:
    ends = demography.epoch_end_times
    starts = jnp.concatenate((jnp.zeros((1,), dtype=ends.dtype), ends[:-1]))
    finite_end = jnp.minimum(times[..., None], ends)
    duration = jnp.maximum(finite_end - starts, 0.0)
    rate = 1.0 / (2.0 * demography.effective_population_size[:, population])
    cumulative = jnp.sum(duration * rate, axis=-1)
    epoch = jnp.sum(times[..., None] >= ends, axis=-1).astype(jnp.int32)
    epoch = jnp.minimum(epoch, ends.shape[0] - 1)
    instantaneous = rate[epoch]
    return cumulative, epoch, instantaneous


def pairwise_coalescent_log_density(
    times: ArrayLike,
    demography: PiecewiseConstantDemography,
    /,
    *,
    population: int = 0,
) -> PairwiseCoalescentResult:
    """Exact pairwise coalescent density through piecewise-constant epochs."""
    if not isinstance(demography, PiecewiseConstantDemography):
        raise TypeError("demography must be PiecewiseConstantDemography.")
    _require_unmigrated(demography)
    population_ = int(population)
    if population_ < 0 or population_ >= demography.population_count:
        raise IndexError("population is outside the demographic population axis.")
    times_ = jnp.asarray(times)
    if jnp.iscomplexobj(times_):
        raise TypeError("Coalescent times must be real-valued.")
    if not jnp.issubdtype(times_.dtype, jnp.inexact):
        times_ = times_.astype(float)
    cumulative, epoch, rate = _integrated_pair_hazard(times_, demography, population_)
    valid = jnp.isfinite(times_) & (times_ >= 0.0) & jnp.isfinite(cumulative)
    log_survival = -cumulative
    log_density = jnp.log(rate) + log_survival
    status = jnp.where(
        valid, int(DemographyStatus.SUCCESS), int(DemographyStatus.NONFINITE)
    ).astype(jnp.int32)
    evidence = jnp.stack((times_, cumulative), axis=-1)
    return PairwiseCoalescentResult(
        jnp.where(valid, log_density, -jnp.inf),
        jnp.where(valid, log_survival, -jnp.inf),
        cumulative,
        epoch,
        valid,
        status,
        evidence,
        _contract(
            "piecewise-pairwise-coalescent-density",
            exact=True,
            output=OutputKind.PROBABILISTIC,
        ),
    )


def coalescent_event_log_likelihood(
    event_times: ArrayLike,
    lineage_count: ArrayLike,
    demography: PiecewiseConstantDemography,
    /,
    *,
    population: int = 0,
    censor_time: float | ArrayLike | None = None,
) -> CoalescentLikelihoodResult:
    """Kingman event-time likelihood for a declared lineage-count trajectory."""
    if not isinstance(demography, PiecewiseConstantDemography):
        raise TypeError("demography must be PiecewiseConstantDemography.")
    _require_unmigrated(demography)
    times = jnp.asarray(event_times)
    lineages = jnp.asarray(lineage_count)
    if times.ndim != 1 or lineages.shape != times.shape:
        raise ValueError("event_times and lineage_count must be equal-length vectors.")
    if not jnp.issubdtype(lineages.dtype, jnp.integer):
        raise TypeError("lineage_count must contain integers.")
    if not jnp.issubdtype(times.dtype, jnp.inexact):
        times = times.astype(float)
    host_times = np.asarray(times)
    host_lineages = np.asarray(lineages)
    ordered = bool(
        np.all(np.isfinite(host_times))
        and np.all(host_times >= 0.0)
        and np.all(np.diff(host_times) > 0.0)
    )
    lineage_valid = bool(
        np.all(host_lineages >= 2)
        and (host_lineages.size < 2 or np.all(np.diff(host_lineages) == -1))
    )
    population_ = int(population)
    if population_ < 0 or population_ >= demography.population_count:
        raise IndexError("population is outside the demographic population axis.")
    previous = jnp.concatenate((jnp.zeros((1,), dtype=times.dtype), times[:-1]))
    cumulative_now, _, rate_now = _integrated_pair_hazard(times, demography, population_)
    cumulative_previous, _, _ = _integrated_pair_hazard(previous, demography, population_)
    pair_count = lineages.astype(times.dtype) * (lineages.astype(times.dtype) - 1.0) / 2.0
    integrated = pair_count * (cumulative_now - cumulative_previous)
    event_rate = pair_count * rate_now
    per_event = jnp.log(event_rate) - integrated
    censor_hazard = jnp.asarray(0.0, dtype=times.dtype)
    if censor_time is not None:
        censor = jnp.asarray(censor_time, dtype=times.dtype)
        if censor.shape != ():
            raise ValueError("censor_time must be scalar.")
        start = times[-1] if times.size else jnp.asarray(0.0, dtype=times.dtype)
        if float(np.asarray(censor)) < float(np.asarray(start)):
            raise ValueError("censor_time cannot precede the final coalescent event.")
        cumulative_censor, _, _ = _integrated_pair_hazard(censor, demography, population_)
        cumulative_start, _, _ = _integrated_pair_hazard(start, demography, population_)
        remaining = lineages[-1] - 1 if lineages.size else jnp.asarray(2, dtype=jnp.int32)
        remaining_pairs = (
            remaining.astype(times.dtype) * (remaining.astype(times.dtype) - 1.0) / 2.0
        )
        censor_hazard = remaining_pairs * (cumulative_censor - cumulative_start)
    finite = jnp.all(jnp.isfinite(per_event)) & jnp.isfinite(censor_hazard)
    valid = jnp.asarray(ordered & lineage_valid) & finite
    status = jnp.where(
        not ordered,
        int(DemographyStatus.INVALID_EVENT_ORDER),
        jnp.where(
            not lineage_valid,
            int(DemographyStatus.INVALID_LINEAGE_COUNT),
            jnp.where(
                finite, int(DemographyStatus.SUCCESS), int(DemographyStatus.NONFINITE)
            ),
        ),
    ).astype(jnp.int32)
    total = jnp.where(valid, jnp.sum(per_event) - censor_hazard, -jnp.inf)
    return CoalescentLikelihoodResult(
        per_event,
        total,
        jnp.sum(integrated) + censor_hazard,
        valid,
        status,
        jnp.stack((pair_count, integrated), axis=-1),
        _contract(
            "kingman-coalescent-event-likelihood", exact=True, output=OutputKind.SCALAR
        ),
    )


def standard_neutral_expected_sfs(
    sample_size: int,
    theta: float | ArrayLike,
    /,
    *,
    folded: bool = False,
) -> ExpectedSFSResult:
    """Exact Kingman infinite-sites expectation: E[xi_i] = theta / i."""
    size = int(sample_size)
    if size < 2:
        raise ValueError("sample_size must be at least two.")
    theta_ = jnp.asarray(theta)
    if theta_.shape != () or jnp.iscomplexobj(theta_):
        raise ValueError("theta must be a real scalar.")
    if not jnp.issubdtype(theta_.dtype, jnp.inexact):
        theta_ = theta_.astype(float)
    if not np.isfinite(float(np.asarray(theta_))) or float(np.asarray(theta_)) < 0.0:
        raise ValueError("theta must be finite and non-negative.")
    unfolded = jnp.zeros((size + 1,), dtype=theta_.dtype)
    indices = jnp.arange(1, size, dtype=theta_.dtype)
    unfolded = unfolded.at[1:size].set(theta_ / indices)
    if folded:
        expected = jnp.zeros_like(unfolded)
        for count in range(1, size):
            expected = expected.at[min(count, size - count)].add(unfolded[count])
    else:
        expected = unfolded
    valid = jnp.isfinite(theta_)
    return ExpectedSFSResult(
        expected,
        jnp.asarray(size, dtype=jnp.int32),
        theta_,
        valid,
        jnp.where(
            valid, int(DemographyStatus.SUCCESS), int(DemographyStatus.NONFINITE)
        ).astype(jnp.int32),
        jnp.asarray((size, jnp.sum(expected)), dtype=theta_.dtype),
        folded,
        not folded,
        _contract("standard-neutral-expected-sfs", exact=True, output=OutputKind.ARRAY),
    )


def sfs_log_likelihood(
    observed_spectrum: ArrayLike,
    expected_spectrum: ArrayLike,
    /,
    *,
    likelihood: Literal["poisson", "multinomial"] = "poisson",
    folded: bool = False,
    polarized: bool = True,
) -> SFSLikelihoodResult:
    """Stable independent-Poisson or conditional-multinomial SFS likelihood."""
    observed = jnp.asarray(observed_spectrum)
    expected = jnp.asarray(expected_spectrum)
    if observed.ndim != 1 or expected.shape != observed.shape:
        raise ValueError("observed_spectrum and expected_spectrum must be equal vectors.")
    if jnp.iscomplexobj(observed) or jnp.iscomplexobj(expected):
        raise TypeError("SFS values must be real-valued.")
    dtype = jnp.result_type(observed, expected, float)
    observed = observed.astype(dtype)
    expected = expected.astype(dtype)
    if likelihood not in ("poisson", "multinomial"):
        raise ValueError("likelihood must be 'poisson' or 'multinomial'.")
    if not isinstance(folded, bool) or not isinstance(polarized, bool):
        raise TypeError("folded and polarized must be bool values.")
    polarization_valid = folded or polarized
    nonnegative = jnp.all(observed >= 0.0) & jnp.all(expected >= 0.0)
    finite = jnp.all(jnp.isfinite(observed)) & jnp.all(jnp.isfinite(expected))
    impossible = jnp.any((observed > 0.0) & (expected <= 0.0))
    expected_total = jnp.sum(expected)
    observed_total = jnp.sum(observed)
    normalized = jnp.where(expected_total > 0.0, expected / expected_total, 0.0)
    if likelihood == "poisson":
        per_bin = (
            jsp.special.xlogy(observed, expected)
            - expected
            - jsp.special.gammaln(observed + 1.0)
        )
    else:
        per_bin = jsp.special.xlogy(observed, normalized)
    valid = (
        polarization_valid & nonnegative & finite & ~impossible & (expected_total > 0.0)
    )
    total = jnp.where(valid, jnp.sum(per_bin), -jnp.inf)
    if likelihood == "multinomial":
        total = (
            total
            + jsp.special.gammaln(observed_total + 1.0)
            - jnp.sum(jsp.special.gammaln(observed + 1.0))
        )
    status = jnp.where(
        not polarization_valid,
        int(DemographyStatus.POLARIZATION_MISMATCH),
        jnp.where(
            impossible | (expected_total <= 0.0) | ~nonnegative,
            int(DemographyStatus.IMPOSSIBLE_SPECTRUM),
            jnp.where(
                finite, int(DemographyStatus.SUCCESS), int(DemographyStatus.NONFINITE)
            ),
        ),
    ).astype(jnp.int32)
    return SFSLikelihoodResult(
        per_bin,
        total,
        normalized,
        observed_total,
        valid,
        status,
        jnp.asarray((observed_total, expected_total), dtype=dtype),
        likelihood,
        folded,
        polarized,
        _contract(f"{likelihood}-sfs-likelihood", exact=True, output=OutputKind.SCALAR),
    )


__all__ = [
    "CoalescentLikelihoodResult",
    "DemographyStatus",
    "ExpectedSFSResult",
    "PairwiseCoalescentResult",
    "PiecewiseConstantDemography",
    "SFSLikelihoodResult",
    "coalescent_event_log_likelihood",
    "pairwise_coalescent_log_density",
    "sfs_log_likelihood",
    "standard_neutral_expected_sfs",
]
