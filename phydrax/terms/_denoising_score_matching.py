#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from .._doc import DOC_KEY0
from .._fingerprint import canonical_fingerprint
from .._score_field import require_score_field
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ..integration._external import _canonical_weighted, materialize_weighted_target
from ..integration._targets import WeightedSampleTarget
from ..stochastic._gaussian_diffusion import AbstractGaussianDiffusion
from ._sample_statistics import effective_sample_size, normalized_log_weights
from ._time_sampling import UniformTimeSamplingPolicy


DenoisingScoreSamplingMode: TypeAlias = Literal["fixed", "resample"]
DenoisingScoreWeighting: TypeAlias = Literal[
    "unit", "conditional-variance", "diffusion-rate"
]
DenoisingScoreDataProvider: TypeAlias = Callable[
    [Key[Array, ""]], WeightedSampleTarget
]


def _canonical_target(
    target: WeightedSampleTarget,
    state_shape: tuple[int, ...],
    /,
) -> tuple[Array, Array, Array, bool, str]:
    if not isinstance(target, WeightedSampleTarget):
        raise TypeError("Denoising score data must be a WeightedSampleTarget.")
    if not target.normalized:
        raise ValueError("Denoising score matching requires a normalized data target.")
    batch = materialize_weighted_target(target)
    values, log_weights, included, _ = _canonical_weighted(batch.samples, batch)
    if log_weights.ndim != 1:
        raise ValueError(
            "Denoising score matching initially requires every weight axis to be sampled."
        )
    expected = (int(log_weights.shape[0]),) + state_shape
    if values.shape != expected:
        raise ValueError(
            f"Denoising score states must have shape {expected}; got {values.shape}."
        )
    valid = jnp.asarray(included, dtype=bool)
    if batch.support_valid is not None:
        support = jnp.asarray(batch.support_valid, dtype=bool)
        if support.shape != ():
            raise ValueError(
                "Denoising score support_valid must be scalar when no case axes remain."
            )
        valid = valid & support
    return (
        jnp.asarray(values),
        jnp.asarray(log_weights, dtype=float),
        valid,
        batch.independent,
        batch.provenance,
    )


class DenoisingScoreMatchingBatch(StrictModule):
    """One fixed Gaussian perturbation batch for denoising score matching."""

    clean_state: Array
    perturbed_state: Array
    time: Array
    noise: Array
    target_score: Array
    objective_weight: Array
    valid: Array
    log_weights: Array
    source_indices: Array
    num_samples: int = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    independent: bool = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    weighting: DenoisingScoreWeighting = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)
    data_provenance: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        clean_state: ArrayLike,
        perturbed_state: ArrayLike,
        time: ArrayLike,
        noise: ArrayLike,
        target_score: ArrayLike,
        objective_weight: ArrayLike,
        valid: ArrayLike,
        log_weights: ArrayLike,
        source_indices: ArrayLike,
        independent: bool,
        process_id: str,
        policy_id: str,
        weighting: DenoisingScoreWeighting,
        batch_id: str,
        data_provenance: str,
    ):
        clean = jnp.asarray(clean_state)
        perturbed = jnp.asarray(perturbed_state, dtype=clean.dtype)
        noise_array = jnp.asarray(noise, dtype=clean.dtype)
        target = jnp.asarray(target_score, dtype=clean.dtype)
        if clean.ndim != 2 or not (
            clean.shape == perturbed.shape == noise_array.shape == target.shape
        ):
            raise ValueError(
                "Denoising states, noise, and targets require one sample axis and "
                "one vector event axis."
            )
        count = int(clean.shape[0])
        expected = (count,)
        times = jnp.asarray(time, dtype=clean.dtype)
        weights = jnp.asarray(objective_weight, dtype=clean.dtype)
        validity = jnp.asarray(valid, dtype=bool)
        log_mass = jnp.asarray(log_weights, dtype=float)
        indices = jnp.asarray(source_indices, dtype=jnp.int32)
        if not (
            times.shape
            == weights.shape
            == validity.shape
            == log_mass.shape
            == indices.shape
            == expected
        ):
            raise ValueError("Denoising sample metadata must have shape (num_samples,).")
        if weighting not in ("unit", "conditional-variance", "diffusion-rate"):
            raise ValueError("Unknown denoising score weighting.")
        for owner, value in (
            ("process_id", process_id),
            ("policy_id", policy_id),
            ("batch_id", batch_id),
            ("data_provenance", data_provenance),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{owner} must be a non-empty string.")
        self.clean_state = clean
        self.perturbed_state = perturbed
        self.time = times
        self.noise = noise_array
        self.target_score = target
        self.objective_weight = weights
        self.valid = validity
        self.log_weights = log_mass
        self.source_indices = indices
        self.num_samples = count
        self.event_shape = tuple(clean.shape[1:])
        self.independent = bool(independent)
        self.process_id = process_id
        self.policy_id = policy_id
        self.weighting = weighting
        self.batch_id = batch_id
        self.data_provenance = data_provenance


class DenoisingScoreMatchingDiagnostics(StrictModule):
    objective: Array
    unweighted_score_rmse: Array
    weighted_score_rmse: Array
    mean_predicted_score_norm: Array
    mean_target_score_norm: Array
    valid_fraction: Array
    effective_sample_size: Array
    minimum_time: Array
    maximum_time: Array
    mean_time: Array
    minimum_perturbation_scale: Array
    maximum_perturbation_scale: Array
    minimum_objective_weight: Array
    maximum_objective_weight: Array
    objective_standard_error: Array
    finite: Array
    num_samples: int = eqx.field(static=True)
    event_size: int = eqx.field(static=True)
    independent: bool = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    weighting: DenoisingScoreWeighting = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return bool(self.finite) and bool(self.valid_fraction > 0.0)


class _DenoisingNodeEvaluation(StrictModule):
    squared_error: Array
    predicted_norm: Array
    target_norm: Array
    perturbation_scale: Array
    weights: Array
    valid: Array


class DenoisingScoreMatchingTerm(AbstractSamplingTerm):
    """Weighted denoising score objective for an exact Gaussian diffusion marginal."""

    fixed_target: WeightedSampleTarget | None
    target_provider: DenoisingScoreDataProvider | None
    process: AbstractGaussianDiffusion
    policy: UniformTimeSamplingPolicy
    scalar_weight: Array
    score_name: str = eqx.field(static=True)
    weighting: DenoisingScoreWeighting = eqx.field(static=True)
    sampling_mode: DenoisingScoreSamplingMode = eqx.field(static=True)
    state_label: str = eqx.field(static=True)
    time_label: str = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        score_name: str,
        data: WeightedSampleTarget | DenoisingScoreDataProvider,
        process: AbstractGaussianDiffusion,
        policy: UniformTimeSamplingPolicy,
        /,
        *,
        weighting: DenoisingScoreWeighting = "conditional-variance",
        sampling_mode: DenoisingScoreSamplingMode = "fixed",
        scalar_weight: ArrayLike = 1.0,
        state_label: str = "x",
        time_label: str = "t",
        label: str | None = None,
    ):
        if not isinstance(score_name, str) or not score_name:
            raise ValueError("score_name must be a non-empty string.")
        if not isinstance(process, AbstractGaussianDiffusion):
            raise TypeError("process must implement AbstractGaussianDiffusion.")
        if not isinstance(policy, UniformTimeSamplingPolicy):
            raise TypeError("policy must be a UniformTimeSamplingPolicy.")
        if not bool(policy.minimum_time > 0.0):
            raise ValueError("Denoising score minimum_time must be strictly positive.")
        if bool(policy.maximum_time > process.terminal_time):
            raise ValueError("Denoising score time policy exceeds the process interval.")
        if weighting not in ("unit", "conditional-variance", "diffusion-rate"):
            raise ValueError("Unknown denoising score weighting.")
        if sampling_mode not in ("fixed", "resample"):
            raise ValueError("sampling_mode must be 'fixed' or 'resample'.")
        if not state_label or not time_label or state_label == time_label:
            raise ValueError("state_label and time_label must be distinct and non-empty.")
        if sampling_mode == "fixed":
            if not isinstance(data, WeightedSampleTarget):
                raise TypeError("Fixed denoising score matching requires weighted data.")
            _canonical_target(data, process.state_shape)
            fixed = data
            provider = None
        else:
            if not callable(data):
                raise TypeError("Resampled denoising score matching requires a provider.")
            fixed = None
            provider = cast(DenoisingScoreDataProvider, data)
        weight = jnp.asarray(scalar_weight, dtype=float).reshape(())
        if not bool(jnp.isfinite(weight)) or float(weight) < 0.0:
            raise ValueError("scalar_weight must be finite and nonnegative.")
        self.fixed_target = fixed
        self.target_provider = provider
        self.process = process
        self.policy = policy
        self.scalar_weight = weight
        self.score_name = score_name
        self.weighting = weighting
        self.sampling_mode = sampling_mode
        self.state_label = str(state_label)
        self.time_label = str(time_label)
        self.label = None if label is None else str(label)

    def _objective_weight(self, time: Array, scale: Array, /) -> Array:
        if self.weighting == "unit":
            return jnp.ones_like(scale)
        if self.weighting == "conditional-variance":
            return scale**2
        return jax.vmap(self.process.diffusion_scale)(time) ** 2

    def sample(self, *, key: Key[Array, ""] = DOC_KEY0) -> DenoisingScoreMatchingBatch:
        data_key, time_key, noise_key = jr.split(key, 3)
        if self.sampling_mode == "fixed":
            if self.fixed_target is None:
                raise RuntimeError("Fixed denoising score data are unavailable.")
            target = self.fixed_target
        else:
            if self.target_provider is None:
                raise RuntimeError("Denoising score data provider is unavailable.")
            target = self.target_provider(data_key)
        clean, log_weights, valid, independent, provenance = _canonical_target(
            target, self.process.state_shape
        )
        count = int(clean.shape[0])
        time = self.policy.sample(time_key, (count,), dtype=clean.real.dtype)
        noise = jr.normal(noise_key, clean.shape, dtype=clean.dtype)
        mean_scale, scale = jax.vmap(
            lambda current: (
                self.process.transition_mean_scale(0.0, current),
                self.process.transition_scale(0.0, current),
            )
        )(time)
        factors = (count, 1)
        perturbed = mean_scale.reshape(factors) * clean + scale.reshape(factors) * noise
        target_score = -noise / scale.reshape(factors)
        objective_weight = self._objective_weight(time, scale)
        event_finite = jnp.all(
            jnp.isfinite(clean)
            & jnp.isfinite(perturbed)
            & jnp.isfinite(target_score),
            axis=-1,
        )
        valid = valid & event_finite & jnp.isfinite(log_weights)
        batch_id = self.label or canonical_fingerprint(
            {
                "kind": "denoising-score-batch",
                "process_id": self.process.process_id,
                "policy_id": self.policy.policy_id,
                "weighting": self.weighting,
                "data_provenance": provenance,
                "num_samples": count,
            }
        )
        return DenoisingScoreMatchingBatch(
            clean_state=clean,
            perturbed_state=perturbed,
            time=time,
            noise=noise,
            target_score=target_score,
            objective_weight=objective_weight,
            valid=valid,
            log_weights=log_weights,
            source_indices=jnp.arange(count, dtype=jnp.int32),
            independent=independent,
            process_id=self.process.process_id,
            policy_id=self.policy.policy_id,
            weighting=self.weighting,
            batch_id=batch_id,
            data_provenance=provenance,
        )

    def _evaluate_nodes(
        self,
        functions: Mapping[str, DomainFunction],
        batch: DenoisingScoreMatchingBatch,
        /,
        *,
        key: Key[Array, ""],
    ) -> _DenoisingNodeEvaluation:
        if not isinstance(batch, DenoisingScoreMatchingBatch):
            raise TypeError("batch must be a DenoisingScoreMatchingBatch.")
        score = require_score_field(
            functions,
            self.score_name,
            state_label=self.state_label,
            time_label=self.time_label,
        )
        count = batch.num_samples
        safe_state = jnp.where(batch.valid[:, None], batch.perturbed_state, 0.0)
        safe_target = jnp.where(batch.valid[:, None], batch.target_score, 0.0)
        safe_time = jnp.where(batch.valid, batch.time, self.policy.minimum_time)
        keys = jr.split(key, count)
        predicted = jax.vmap(
            lambda state, time, node_key: score(state, time, key=node_key)
        )(safe_state, safe_time, keys)
        if jnp.iscomplexobj(predicted):
            raise TypeError("Denoising score fields must be real-valued.")
        residual = predicted - safe_target
        squared_error = jnp.mean(residual**2, axis=-1)
        predicted_norm = jnp.sqrt(jnp.sum(predicted**2, axis=-1))
        target_norm = jnp.sqrt(jnp.sum(safe_target**2, axis=-1))
        weights = normalized_log_weights(batch.log_weights, batch.valid)
        scale = jax.vmap(lambda time: self.process.transition_scale(0.0, time))(
            batch.time
        )
        return _DenoisingNodeEvaluation(
            squared_error=squared_error,
            predicted_norm=predicted_norm,
            target_norm=target_norm,
            perturbation_scale=scale,
            weights=weights,
            valid=batch.valid,
        )

    def _diagnostics(
        self,
        functions: Mapping[str, DomainFunction],
        batch: DenoisingScoreMatchingBatch,
        /,
        *,
        key: Key[Array, ""],
    ) -> DenoisingScoreMatchingDiagnostics:
        evaluation = self._evaluate_nodes(functions, batch, key=key)
        weights = evaluation.weights
        weighted_loss = batch.objective_weight * evaluation.squared_error
        raw_objective = jnp.sum(weights * weighted_loss)
        objective = self.scalar_weight * raw_objective
        unweighted_rmse = jnp.sqrt(jnp.sum(weights * evaluation.squared_error))
        objective_mass = jnp.sum(weights * batch.objective_weight)
        weighted_rmse = jnp.sqrt(
            jnp.sum(weights * weighted_loss)
            / jnp.maximum(objective_mass, jnp.finfo(weights.dtype).tiny)
        )
        predicted_norm = jnp.sum(weights * evaluation.predicted_norm)
        target_norm = jnp.sum(weights * evaluation.target_norm)
        valid_fraction = jnp.mean(evaluation.valid.astype(float))
        effective = effective_sample_size(weights)
        minimum_time = jnp.min(jnp.where(evaluation.valid, batch.time, jnp.inf))
        maximum_time = jnp.max(jnp.where(evaluation.valid, batch.time, -jnp.inf))
        mean_time = jnp.sum(weights * batch.time)
        minimum_scale = jnp.min(
            jnp.where(evaluation.valid, evaluation.perturbation_scale, jnp.inf)
        )
        maximum_scale = jnp.max(
            jnp.where(evaluation.valid, evaluation.perturbation_scale, -jnp.inf)
        )
        minimum_weight = jnp.min(
            jnp.where(evaluation.valid, batch.objective_weight, jnp.inf)
        )
        maximum_weight = jnp.max(
            jnp.where(evaluation.valid, batch.objective_weight, -jnp.inf)
        )
        centered = weighted_loss - raw_objective
        variance = jnp.sum(weights * centered**2)
        standard_error = jnp.where(
            batch.independent & (effective > 1.0),
            self.scalar_weight * jnp.sqrt(variance / (effective - 1.0)),
            jnp.asarray(jnp.nan, dtype=objective.dtype),
        )
        finite = (
            jnp.isfinite(objective)
            & jnp.isfinite(unweighted_rmse)
            & jnp.isfinite(weighted_rmse)
            & jnp.isfinite(predicted_norm)
            & jnp.isfinite(target_norm)
            & jnp.isfinite(minimum_time)
            & jnp.isfinite(maximum_time)
            & jnp.isfinite(mean_time)
            & jnp.isfinite(minimum_scale)
            & jnp.isfinite(maximum_scale)
            & jnp.isfinite(minimum_weight)
            & jnp.isfinite(maximum_weight)
        )
        return DenoisingScoreMatchingDiagnostics(
            objective=objective,
            unweighted_score_rmse=unweighted_rmse,
            weighted_score_rmse=weighted_rmse,
            mean_predicted_score_norm=predicted_norm,
            mean_target_score_norm=target_norm,
            valid_fraction=valid_fraction,
            effective_sample_size=effective,
            minimum_time=minimum_time,
            maximum_time=maximum_time,
            mean_time=mean_time,
            minimum_perturbation_scale=minimum_scale,
            maximum_perturbation_scale=maximum_scale,
            minimum_objective_weight=minimum_weight,
            maximum_objective_weight=maximum_weight,
            objective_standard_error=standard_error,
            finite=finite,
            num_samples=batch.num_samples,
            event_size=batch.event_shape[0],
            independent=batch.independent,
            process_id=batch.process_id,
            policy_id=batch.policy_id,
            weighting=batch.weighting,
            batch_id=batch.batch_id,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        batch: DenoisingScoreMatchingBatch | None = None,
        **kwargs,
    ) -> Array:
        del iter_, kwargs
        resolved = self.sample(key=key) if batch is None else batch
        return self._diagnostics(functions, resolved, key=key).objective

    def diagnostics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: DenoisingScoreMatchingBatch | None = None,
    ) -> DenoisingScoreMatchingDiagnostics:
        resolved = self.sample(key=key) if batch is None else batch
        return self._diagnostics(functions, resolved, key=key)


__all__ = [
    "DenoisingScoreDataProvider",
    "DenoisingScoreMatchingBatch",
    "DenoisingScoreMatchingDiagnostics",
    "DenoisingScoreMatchingTerm",
    "DenoisingScoreSamplingMode",
    "DenoisingScoreWeighting",
]
