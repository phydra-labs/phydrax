#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, log, prod, sqrt
from pathlib import Path
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._array_archive import (
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from .._numerics import LogWeightedAccumulator
from .._strict import StrictModule
from ._estimates import IntegrationEstimate, IntegrationProvenance
from ._plans import MultilevelMonteCarloPlan
from ._status import IntegrationStatus
from ._targets import MultilevelTarget


_ACTIVE_STATUS = -1
_RESULT_FORMAT = "phydrax-multilevel-result"
_CHECKPOINT_KIND = "multilevel-monte-carlo"


def _digest(*parts: object) -> str:
    digest = hashlib.sha256(b"phydrax-multilevel\0")
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _plan_fingerprint(plan: MultilevelMonteCarloPlan, /) -> str:
    return _digest(
        plan.initial_samples,
        plan.samples_per_level,
        plan.target_rmse,
        plan.max_samples_per_level,
        plan.batch_size,
        plan.variance_fraction,
        plan.max_rounds,
    )


def _counts(
    value: int | tuple[int, ...],
    num_levels: int,
    name: str,
    /,
) -> tuple[int, ...]:
    resolved = (value,) * num_levels if isinstance(value, int) else tuple(value)
    if len(resolved) != num_levels:
        raise ValueError(f"{name} must contain one value per hierarchy level.")
    return tuple(int(item) for item in resolved)


def _finite_sample_mask(values: Array, /) -> Array:
    axes = tuple(range(1, values.ndim))
    finite = jnp.isfinite(values)
    return jnp.all(finite, axis=axes) if axes else finite


def _host_float(value: ArrayLike, /) -> float:
    return float(np.asarray(jax.device_get(value)))


def _norm(value: ArrayLike, /) -> float:
    array = jnp.asarray(value)
    return _host_float(jnp.max(jnp.abs(array)))


class MultilevelSampleBatch(StrictModule):
    """One prefix-addressed batch of fine/coarse samples for a hierarchy level."""

    fine_samples: Any
    coarse_samples: Any | None
    sample_indices: Array
    pair_ids: Array
    fine_valid: Array
    coarse_valid: Array
    costs: Array
    level_index: int = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        fine_samples: Any,
        coarse_samples: Any | None,
        sample_indices: ArrayLike,
        costs: ArrayLike,
        /,
        *,
        level_index: int,
        fine_valid: ArrayLike | None = None,
        coarse_valid: ArrayLike | None = None,
        pair_ids: ArrayLike | None = None,
        provenance: str,
    ):
        level = int(level_index)
        if level < 0:
            raise ValueError("level_index must be non-negative.")
        if level == 0 and coarse_samples is not None:
            raise ValueError("The base multilevel batch cannot contain coarse samples.")
        if level > 0 and coarse_samples is None:
            raise ValueError("Fine multilevel batches require paired coarse samples.")
        indices = jnp.asarray(sample_indices, dtype=jnp.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("sample_indices must be a non-empty vector.")
        if bool(jnp.any(indices < 0)) or bool(jnp.any(jnp.diff(indices) != 1)):
            raise ValueError("sample_indices must be consecutive and non-negative.")
        count = int(indices.size)
        cost_values = jnp.broadcast_to(jnp.asarray(costs, dtype=float), (count,))
        if bool(jnp.any(~jnp.isfinite(cost_values) | (cost_values <= 0.0))):
            raise ValueError("costs must be finite and strictly positive.")
        fine_mask = (
            jnp.ones((count,), dtype=bool)
            if fine_valid is None
            else jnp.asarray(fine_valid, dtype=bool)
        )
        coarse_mask = (
            jnp.ones((count,), dtype=bool)
            if coarse_valid is None
            else jnp.asarray(coarse_valid, dtype=bool)
        )
        if fine_mask.shape != (count,) or coarse_mask.shape != (count,):
            raise ValueError("Validity arrays must contain one entry per sample pair.")
        pairs = indices if pair_ids is None else jnp.asarray(pair_ids, dtype=jnp.int64)
        if pairs.shape != (count,):
            raise ValueError("pair_ids must contain one entry per sample pair.")
        identifier = str(provenance)
        if not identifier:
            raise ValueError("provenance must be non-empty.")
        self.fine_samples = fine_samples
        self.coarse_samples = coarse_samples
        self.sample_indices = indices
        self.pair_ids = pairs
        self.fine_valid = fine_mask
        self.coarse_valid = coarse_mask
        self.costs = cost_values
        self.level_index = level
        self.provenance = identifier

    @property
    def num_samples(self) -> int:
        return int(self.sample_indices.size)


class MultilevelRealization(StrictModule):
    """Reusable MLMC target/plan execution identity and prefix-stable root key."""

    target: MultilevelTarget
    plan: MultilevelMonteCarloPlan
    root_key: Array
    initial_samples: tuple[int, ...] = eqx.field(static=True)
    maximum_samples: tuple[int, ...] = eqx.field(static=True)
    fixed_samples: tuple[int, ...] | None = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    plan_fingerprint: str = eqx.field(static=True)


class MultilevelEstimatorState(StrictModule):
    """Mergeable correction moments and prefix indices for checkpointable MLMC."""

    accumulators: tuple[LogWeightedAccumulator | None, ...]
    attempted_counts: Array
    failed_counts: Array
    total_costs: Array
    next_indices: Array
    allocation_target: Array
    rounds: int = eqx.field(static=True)
    finished: bool = eqx.field(static=True)
    status: int = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)


class MultilevelDiagnostics(StrictModule):
    """Per-level moments, allocation, cost, failure, bias, and RMSE diagnostics."""

    correction_means: tuple[Array, ...]
    correction_variances: tuple[Array, ...]
    correction_standard_errors: tuple[Array, ...]
    sample_counts: Array
    attempted_counts: Array
    failed_counts: Array
    mean_costs: Array
    allocation_target: Array
    correction_variance_norms: Array
    sampling_standard_error: Array
    bias_estimate: Array
    rmse_estimate: Array
    weak_convergence_order: Array
    rounds: int = eqx.field(static=True)
    hierarchy_id: str = eqx.field(static=True)
    hierarchy_fingerprint: str = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)


@dataclass(frozen=True)
class MultilevelResultArchive:
    """Checksum-validated immutable arrays from a portable MLMC result archive."""

    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    def array(self, name: str, /) -> np.ndarray:
        if name not in self.arrays:
            raise KeyError(f"Multilevel result archive has no array {name!r}.")
        return self.arrays[name]


def materialize_multilevel(
    target: MultilevelTarget,
    plan: MultilevelMonteCarloPlan,
    key: Key[Array, ""],
    /,
) -> MultilevelRealization:
    """Validate and bind one hierarchy, allocation plan, and random root key."""
    if not isinstance(target, MultilevelTarget):
        raise TypeError("target must be a MultilevelTarget.")
    if not isinstance(plan, MultilevelMonteCarloPlan):
        raise TypeError("plan must be a MultilevelMonteCarloPlan.")
    key_data = jr.key_data(key)
    if key_data.shape != (2,):
        raise ValueError("MLMC materialization requires one scalar JAX key.")
    num_levels = target.hierarchy.num_levels
    initial = _counts(plan.initial_samples, num_levels, "initial_samples")
    maximum = _counts(
        plan.max_samples_per_level,
        num_levels,
        "max_samples_per_level",
    )
    fixed = (
        None
        if plan.samples_per_level is None
        else _counts(plan.samples_per_level, num_levels, "samples_per_level")
    )
    required = initial if fixed is None else fixed
    if any(limit < count for limit, count in zip(maximum, required, strict=True)):
        raise ValueError("max_samples_per_level cannot be smaller than required samples.")
    fingerprint = _plan_fingerprint(plan)
    key_tuple = tuple(int(value) for value in np.asarray(key_data).reshape((-1,)))
    realization_id = _digest(
        target.hierarchy.fingerprint,
        target.sampler_id,
        fingerprint,
        key_tuple,
    )
    return MultilevelRealization(
        target,
        plan,
        key,
        initial,
        maximum,
        fixed,
        realization_id,
        fingerprint,
    )


def initialize_multilevel(
    realization: MultilevelRealization, /
) -> MultilevelEstimatorState:
    """Create an empty prefix-stable estimator state."""
    if not isinstance(realization, MultilevelRealization):
        raise TypeError("realization must be a MultilevelRealization.")
    num_levels = realization.target.hierarchy.num_levels
    return MultilevelEstimatorState(
        (None,) * num_levels,
        jnp.zeros((num_levels,), dtype=jnp.int64),
        jnp.zeros((num_levels,), dtype=jnp.int64),
        jnp.zeros((num_levels,), dtype=float),
        jnp.zeros((num_levels,), dtype=jnp.int64),
        jnp.asarray(realization.initial_samples, dtype=jnp.int64),
        0,
        False,
        _ACTIVE_STATUS,
        realization.realization_id,
    )


def _replace_state(
    state: MultilevelEstimatorState,
    /,
    *,
    accumulators: tuple[LogWeightedAccumulator | None, ...] | None = None,
    attempted_counts: Array | None = None,
    failed_counts: Array | None = None,
    total_costs: Array | None = None,
    next_indices: Array | None = None,
    allocation_target: Array | None = None,
    rounds: int | None = None,
    finished: bool | None = None,
    status: int | None = None,
) -> MultilevelEstimatorState:
    return MultilevelEstimatorState(
        state.accumulators if accumulators is None else accumulators,
        state.attempted_counts if attempted_counts is None else attempted_counts,
        state.failed_counts if failed_counts is None else failed_counts,
        state.total_costs if total_costs is None else total_costs,
        state.next_indices if next_indices is None else next_indices,
        state.allocation_target if allocation_target is None else allocation_target,
        state.rounds if rounds is None else int(rounds),
        state.finished if finished is None else bool(finished),
        state.status if status is None else int(status),
        state.realization_id,
    )


def _sample_counts(state: MultilevelEstimatorState, /) -> tuple[int, ...]:
    return tuple(
        0 if accumulator is None else int(np.asarray(accumulator.count))
        for accumulator in state.accumulators
    )


def _statistics(
    state: MultilevelEstimatorState,
    /,
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[Array, ...]]:
    if any(accumulator is None for accumulator in state.accumulators):
        raise ValueError("Every level requires valid samples before statistics exist.")
    active = tuple(
        accumulator for accumulator in state.accumulators if accumulator is not None
    )
    means = tuple(accumulator.raw_mean for accumulator in active)
    standard_errors = tuple(accumulator.raw_standard_error for accumulator in active)
    variances = tuple(
        standard_error * standard_error * jnp.asarray(accumulator.count)
        for standard_error, accumulator in zip(standard_errors, active, strict=True)
    )
    return means, variances, standard_errors


def _bias_diagnostics(
    hierarchy: Any,
    means: tuple[Array, ...],
    /,
) -> tuple[float, float]:
    finest = _norm(means[-1])
    if len(means) < 3 or finest == 0.0:
        return finest, float("inf")
    previous = _norm(means[-2])
    coarse = hierarchy.levels[-2]
    fine = hierarchy.levels[-1]
    ratios = tuple(
        coarse_scale / fine_scale
        for coarse_scale, fine_scale in zip(
            coarse.resolutions, fine.resolutions, strict=True
        )
    )
    refinement_ratio = prod(ratios) ** (1.0 / len(ratios))
    if previous <= finest or refinement_ratio <= 1.0:
        return finest, 0.0
    order = log(previous / finest) / log(refinement_ratio)
    denominator = refinement_ratio**order - 1.0
    bias = finest if denominator <= 0.0 else finest / denominator
    return bias, order


def _allocation(
    realization: MultilevelRealization,
    state: MultilevelEstimatorState,
    /,
) -> tuple[tuple[int, ...], int | None]:
    counts = _sample_counts(state)
    if realization.fixed_samples is not None:
        if all(
            count >= target
            for count, target in zip(counts, realization.fixed_samples, strict=True)
        ):
            return realization.fixed_samples, int(IntegrationStatus.CONVERGED)
        return realization.fixed_samples, None
    if any(count < target for count, target in zip(counts, realization.initial_samples)):
        return realization.initial_samples, None
    means, variances, standard_errors = _statistics(state)
    variance_norms = tuple(max(_norm(variance), 1e-30) for variance in variances)
    mean_costs = tuple(
        max(
            _host_float(state.total_costs[index])
            / max(_host_float(state.attempted_counts[index]), 1.0),
            1e-30,
        )
        for index in range(len(counts))
    )
    sampling_error = sqrt(
        sum(
            variance / max(count, 1)
            for variance, count in zip(variance_norms, counts, strict=True)
        )
    )
    bias, _ = _bias_diagnostics(realization.target.hierarchy, means)
    rmse = sqrt(sampling_error * sampling_error + bias * bias)
    target_rmse = realization.plan.target_rmse
    assert target_rmse is not None
    if rmse <= target_rmse:
        return counts, int(IntegrationStatus.CONVERGED)
    variance_budget = realization.plan.variance_fraction * target_rmse * target_rmse
    normalization = sum(
        sqrt(variance * cost)
        for variance, cost in zip(variance_norms, mean_costs, strict=True)
    )
    desired = tuple(
        max(
            realization.initial_samples[index],
            int(ceil(normalization * sqrt(variance / cost) / variance_budget)),
        )
        for index, (variance, cost) in enumerate(
            zip(variance_norms, mean_costs, strict=True)
        )
    )
    desired = tuple(
        min(target, realization.maximum_samples[index])
        for index, target in enumerate(desired)
    )
    if all(target <= count for target, count in zip(desired, counts, strict=True)):
        bias_budget = sqrt(1.0 - realization.plan.variance_fraction) * target_rmse
        if bias > bias_budget:
            return desired, int(IntegrationStatus.REFINEMENT_STAGNATION)
    if all(
        _host_float(state.attempted_counts[index]) >= realization.maximum_samples[index]
        for index in range(len(counts))
    ):
        terminal = (
            IntegrationStatus.NO_VALID_SAMPLES
            if any(count < 2 for count in counts)
            else IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED
        )
        return desired, int(terminal)
    return desired, None


def _evaluate_batch(
    observable: Any,
    realization: MultilevelRealization,
    level_index: int,
    batch: MultilevelSampleBatch,
    /,
) -> tuple[Array, Array]:
    if not callable(observable):
        raise TypeError("The MLMC integrand must be a callable observable.")
    hierarchy = realization.target.hierarchy
    fine_level = hierarchy.levels[level_index]
    fine = jnp.asarray(observable(batch.fine_samples, fine_level))
    if fine.shape[:1] != (batch.num_samples,):
        raise ValueError("Fine observables must begin with the batch sample axis.")
    valid = batch.fine_valid & _finite_sample_mask(fine)
    if level_index == 0:
        return fine, valid
    coarse_level = hierarchy.levels[level_index - 1]
    coarse = jnp.asarray(observable(batch.coarse_samples, coarse_level))
    if coarse.shape != fine.shape:
        raise ValueError("Paired fine and coarse observables must have identical shapes.")
    valid = valid & batch.coarse_valid & _finite_sample_mask(coarse)
    return fine - coarse, valid


def _sample_level(
    observable: Any,
    realization: MultilevelRealization,
    state: MultilevelEstimatorState,
    level_index: int,
    count: int,
    /,
) -> MultilevelEstimatorState:
    start = int(np.asarray(state.next_indices[level_index]))
    indices = jnp.arange(start, start + count, dtype=jnp.int64)
    batch = realization.target.sampler(level_index, indices, realization.root_key)
    if not isinstance(batch, MultilevelSampleBatch):
        raise TypeError("A multilevel sampler must return MultilevelSampleBatch.")
    if batch.level_index != level_index:
        raise ValueError("Sampler batch level_index does not match the requested level.")
    if batch.provenance != realization.target.sampler_id:
        raise ValueError("Sampler batch provenance does not match target sampler_id.")
    if not bool(jnp.array_equal(batch.sample_indices, indices)):
        raise ValueError("Sampler changed the requested global sample indices.")
    if not bool(jnp.array_equal(batch.pair_ids, indices)):
        raise ValueError("Fine/coarse pair IDs must equal their global sample indices.")
    correction, valid = _evaluate_batch(
        observable,
        realization,
        level_index,
        batch,
    )
    chunk = LogWeightedAccumulator.from_values(
        correction,
        jnp.zeros((count,), dtype=float),
        mask=valid,
    )
    accumulators = list(state.accumulators)
    current = accumulators[level_index]
    accumulators[level_index] = chunk if current is None else current.merge(chunk)
    attempted = state.attempted_counts.at[level_index].add(count)
    failed = state.failed_counts.at[level_index].add(count - jnp.sum(valid))
    costs = state.total_costs.at[level_index].add(jnp.sum(batch.costs))
    next_indices = state.next_indices.at[level_index].set(start + count)
    return _replace_state(
        state,
        accumulators=tuple(accumulators),
        attempted_counts=attempted,
        failed_counts=failed,
        total_costs=costs,
        next_indices=next_indices,
    )


def advance_multilevel(
    observable: Any,
    realization: MultilevelRealization,
    state: MultilevelEstimatorState | None = None,
    /,
    *,
    num_rounds: int = 1,
) -> MultilevelEstimatorState:
    """Advance an MLMC allocation by bounded mergeable sampling rounds."""
    if not isinstance(realization, MultilevelRealization):
        raise TypeError("realization must be a MultilevelRealization.")
    current = initialize_multilevel(realization) if state is None else state
    if not isinstance(current, MultilevelEstimatorState):
        raise TypeError("state must be a MultilevelEstimatorState or None.")
    if current.realization_id != realization.realization_id:
        raise ValueError("Estimator state does not belong to this realization.")
    rounds = int(num_rounds)
    if rounds < 1:
        raise ValueError("num_rounds must be positive.")
    if current.finished:
        return current
    for _ in range(rounds):
        targets, terminal = _allocation(realization, current)
        current = _replace_state(
            current,
            allocation_target=jnp.asarray(targets, dtype=jnp.int64),
        )
        if terminal is not None:
            return _replace_state(current, finished=True, status=terminal)
        counts = _sample_counts(current)
        sampled = False
        for level_index, (available, target) in enumerate(
            zip(counts, targets, strict=True)
        ):
            deficit = max(target - available, 0)
            remaining = realization.maximum_samples[level_index] - int(
                np.asarray(current.attempted_counts[level_index])
            )
            request = min(deficit, realization.plan.batch_size, remaining)
            if request > 0:
                current = _sample_level(
                    observable,
                    realization,
                    current,
                    level_index,
                    request,
                )
                sampled = True
        current = _replace_state(current, rounds=current.rounds + 1)
        if not sampled:
            counts = _sample_counts(current)
            status = (
                IntegrationStatus.NO_VALID_SAMPLES
                if any(count < 2 for count in counts)
                else IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED
            )
            return _replace_state(current, finished=True, status=int(status))
    return current


def _final_diagnostics(
    realization: MultilevelRealization,
    state: MultilevelEstimatorState,
    /,
) -> MultilevelDiagnostics:
    means, variances, standard_errors = _statistics(state)
    counts = jnp.asarray(_sample_counts(state), dtype=jnp.int64)
    variance_norms = jnp.asarray(tuple(_norm(value) for value in variances))
    sampling_error = jnp.sqrt(jnp.sum(variance_norms / jnp.maximum(counts, 1)))
    bias, order = _bias_diagnostics(realization.target.hierarchy, means)
    rmse = jnp.sqrt(sampling_error * sampling_error + bias * bias)
    mean_costs = state.total_costs / jnp.maximum(state.attempted_counts, 1)
    return MultilevelDiagnostics(
        means,
        variances,
        standard_errors,
        counts,
        state.attempted_counts,
        state.failed_counts,
        mean_costs,
        state.allocation_target,
        variance_norms,
        sampling_error,
        jnp.asarray(bias),
        rmse,
        jnp.asarray(order),
        state.rounds,
        realization.target.hierarchy.hierarchy_id,
        realization.target.hierarchy.fingerprint,
        realization.target.sampler_id,
    )


def finalize_multilevel(
    realization: MultilevelRealization,
    state: MultilevelEstimatorState,
    /,
) -> IntegrationEstimate:
    """Finalize a sampled MLMC state into a canonical integration estimate."""
    if state.realization_id != realization.realization_id:
        raise ValueError("Estimator state does not belong to this realization.")
    counts = _sample_counts(state)
    if any(count < 2 for count in counts):
        raise ValueError("Every MLMC level requires at least two valid corrections.")
    diagnostics = _final_diagnostics(realization, state)
    value = sum(diagnostics.correction_means[1:], start=diagnostics.correction_means[0])
    status = (
        state.status if state.finished else int(IntegrationStatus.REFINEMENT_STAGNATION)
    )
    evaluations = state.attempted_counts[0] + 2 * jnp.sum(state.attempted_counts[1:])
    return IntegrationEstimate(
        value,
        status=status,
        num_evaluations=evaluations,
        error_estimate=diagnostics.rmse_estimate,
        error_kind="mlmc-rmse-estimate",
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "multilevel-monte-carlo",
            realization.target.hierarchy.hierarchy_id,
            realization.realization_id,
        ),
    )


def integrate_multilevel(
    observable: Any,
    realization: MultilevelRealization,
    /,
    *,
    state: MultilevelEstimatorState | None = None,
) -> IntegrationEstimate:
    """Run or resume MLMC through its plan's bounded adaptive allocation."""
    current = initialize_multilevel(realization) if state is None else state
    remaining = max(realization.plan.max_rounds - current.rounds, 1)
    current = advance_multilevel(
        observable,
        realization,
        current,
        num_rounds=remaining,
    )
    return finalize_multilevel(realization, current)


def _checkpoint_compatibility(realization: MultilevelRealization, /) -> dict[str, Any]:
    return {
        "realization_id": realization.realization_id,
        "hierarchy_fingerprint": realization.target.hierarchy.fingerprint,
        "plan_fingerprint": realization.plan_fingerprint,
        "sampler_id": realization.target.sampler_id,
    }


def write_multilevel_checkpoint(
    path: str | os.PathLike[str],
    realization: MultilevelRealization,
    state: MultilevelEstimatorState,
    /,
) -> Path:
    """Atomically write one pickle-free MLMC continuation state."""
    if state.realization_id != realization.realization_id:
        raise ValueError("Estimator state does not belong to this realization.")
    from ..uq._checkpoint import write_checkpoint_archive

    arrays: dict[str, Any] = {
        "attempted_counts": state.attempted_counts,
        "failed_counts": state.failed_counts,
        "total_costs": state.total_costs,
        "next_indices": state.next_indices,
        "allocation_target": state.allocation_target,
    }
    active: list[bool] = []
    field_names = (
        "log_scale",
        "weight_sum",
        "squared_weight_sum",
        "weighted_value_sum",
        "weighted_abs_square_sum",
        "squared_weight_value_sum",
        "squared_weight_abs_square_sum",
        "count",
    )
    for level_index, accumulator in enumerate(state.accumulators):
        active.append(accumulator is not None)
        if accumulator is not None:
            for name in field_names:
                arrays[f"level/{level_index}/{name}"] = vars(accumulator)[name]
    return write_checkpoint_archive(
        path,
        kind=_CHECKPOINT_KIND,
        compatibility=_checkpoint_compatibility(realization),
        state={
            "rounds": state.rounds,
            "finished": state.finished,
            "status": state.status,
            "active": active,
        },
        arrays=arrays,
    )


def read_multilevel_checkpoint(
    path: str | os.PathLike[str],
    realization: MultilevelRealization,
    /,
) -> MultilevelEstimatorState:
    """Restore and compatibility-check one MLMC continuation state."""
    from ..uq._checkpoint import read_checkpoint_archive

    manifest, arrays = read_checkpoint_archive(
        path,
        kind=_CHECKPOINT_KIND,
        compatibility=_checkpoint_compatibility(realization),
    )
    active = manifest.get("active")
    num_levels = realization.target.hierarchy.num_levels
    if not isinstance(active, list) or len(active) != num_levels:
        raise ValueError("MLMC checkpoint level inventory is invalid.")
    field_names = (
        "log_scale",
        "weight_sum",
        "squared_weight_sum",
        "weighted_value_sum",
        "weighted_abs_square_sum",
        "squared_weight_value_sum",
        "squared_weight_abs_square_sum",
        "count",
    )
    accumulators: list[LogWeightedAccumulator | None] = []
    for level_index, present in enumerate(active):
        if not present:
            accumulators.append(None)
        else:
            values = {name: arrays[f"level/{level_index}/{name}"] for name in field_names}
            accumulators.append(LogWeightedAccumulator(**values))
    return MultilevelEstimatorState(
        tuple(accumulators),
        arrays["attempted_counts"],
        arrays["failed_counts"],
        arrays["total_costs"],
        arrays["next_indices"],
        arrays["allocation_target"],
        int(manifest["rounds"]),
        bool(manifest["finished"]),
        int(manifest["status"]),
        realization.realization_id,
    )


def write_multilevel_result(
    path: str | os.PathLike[str],
    estimate: IntegrationEstimate,
    /,
) -> Path:
    """Write a checksum-validated portable MLMC result archive."""
    if not isinstance(estimate.diagnostics, MultilevelDiagnostics):
        raise TypeError("estimate must contain MultilevelDiagnostics.")

    diagnostics = estimate.diagnostics
    arrays: dict[str, Any] = {
        "value": estimate.value,
        "status": estimate.status,
        "num_evaluations": estimate.num_evaluations,
        "error_estimate": estimate.error_estimate,
        "sample_counts": diagnostics.sample_counts,
        "attempted_counts": diagnostics.attempted_counts,
        "failed_counts": diagnostics.failed_counts,
        "mean_costs": diagnostics.mean_costs,
        "allocation_target": diagnostics.allocation_target,
        "correction_variance_norms": diagnostics.correction_variance_norms,
        "sampling_standard_error": diagnostics.sampling_standard_error,
        "bias_estimate": diagnostics.bias_estimate,
        "rmse_estimate": diagnostics.rmse_estimate,
        "weak_convergence_order": diagnostics.weak_convergence_order,
    }
    for index, (mean, variance, standard_error) in enumerate(
        zip(
            diagnostics.correction_means,
            diagnostics.correction_variances,
            diagnostics.correction_standard_errors,
            strict=True,
        )
    ):
        arrays[f"level/{index}/mean"] = mean
        arrays[f"level/{index}/variance"] = variance
        arrays[f"level/{index}/standard_error"] = standard_error
    return write_array_archive(
        path,
        manifest={
            "format": _RESULT_FORMAT,
            "metadata": {
                "error_kind": estimate.error_kind,
                "method": estimate.provenance.method,
                "target": estimate.provenance.target,
                "realization": estimate.provenance.realization,
                "hierarchy_id": diagnostics.hierarchy_id,
                "hierarchy_fingerprint": diagnostics.hierarchy_fingerprint,
                "sampler_id": diagnostics.sampler_id,
                "rounds": diagnostics.rounds,
                "num_levels": len(diagnostics.correction_means),
            },
        },
        arrays=arrays,
    )


def read_multilevel_result(path: str | os.PathLike[str], /) -> MultilevelResultArchive:
    """Read and checksum-validate a portable MLMC result archive."""
    manifest, loaded = read_array_archive(path)

    if set(manifest) != {"format", "metadata", "arrays"}:
        raise ArrayArchiveCorruptionError(
            "Multilevel result manifest fields are invalid."
        )
    if manifest.get("format") != _RESULT_FORMAT:
        raise ArrayArchiveCorruptionError(
            "Archive is not a Phydrax multilevel result."
        )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ArrayArchiveCorruptionError(
            "Multilevel result metadata is invalid."
        )
    arrays: dict[str, np.ndarray] = {}
    for name, value in loaded.items():
        value.setflags(write=False)
        arrays[name] = value
    return MultilevelResultArchive(
        MappingProxyType(metadata),
        MappingProxyType(arrays),
    )


__all__ = [
    "MultilevelDiagnostics",
    "MultilevelEstimatorState",
    "MultilevelRealization",
    "MultilevelResultArchive",
    "MultilevelSampleBatch",
    "advance_multilevel",
    "finalize_multilevel",
    "initialize_multilevel",
    "integrate_multilevel",
    "materialize_multilevel",
    "read_multilevel_checkpoint",
    "read_multilevel_result",
    "write_multilevel_checkpoint",
    "write_multilevel_result",
]
