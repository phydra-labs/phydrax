#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
from math import prod
from typing import Any, cast, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from .._objective import AbstractSamplingObjectiveTerm
from .._strict import StrictModule
from ..operators.differential._stochastic_estimators import (
    stochastic_divergence_samples,
    StochasticTracePolicy,
)
from ..stochastic._state_time import (
    trajectory_state_time_samples,
    TrajectoryStateTimeSamples,
)
from ..stochastic._trajectory import StochasticTrajectory


ScoreMatchingMethod: TypeAlias = Literal["exact", "implicit", "sliced"]
ScoreMatchingSamplingMode: TypeAlias = Literal["fixed", "resample"]
ScoreSampleProvider: TypeAlias = Callable[
    [Key[Array, ""]], TrajectoryStateTimeSamples | StochasticTrajectory
]


def _policy_id(
    method: ScoreMatchingMethod,
    num_probes: int,
    distribution: Literal["rademacher", "normal"],
    /,
) -> str:
    digest = sha256(b"phydrax-score-matching-policy\0")
    digest.update(repr((method, num_probes, distribution)).encode("utf-8"))
    return digest.hexdigest()


class ScoreMatchingPolicy(StrictModule):
    """Derivative estimator policy for exact, implicit, or sliced score matching."""

    method: ScoreMatchingMethod = eqx.field(static=True)
    num_probes: int = eqx.field(static=True)
    distribution: Literal["rademacher", "normal"] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: ScoreMatchingMethod = "implicit",
        /,
        *,
        num_probes: int = 16,
        distribution: Literal["rademacher", "normal"] = "rademacher",
        policy_id: str | None = None,
    ):
        if method not in ("exact", "implicit", "sliced"):
            raise ValueError("method must be 'exact', 'implicit', or 'sliced'.")
        count = int(num_probes)
        if method == "exact":
            if count < 0:
                raise ValueError("num_probes must be nonnegative for exact matching.")
            count = 0
        elif count < 2:
            raise ValueError("Stochastic score matching requires at least two probes.")
        if distribution not in ("rademacher", "normal"):
            raise ValueError("distribution must be 'rademacher' or 'normal'.")
        resolved_id = (
            _policy_id(method, count, distribution)
            if policy_id is None
            else str(policy_id)
        )
        if not resolved_id:
            raise ValueError("policy_id must be non-empty.")
        self.method = method
        self.num_probes = count
        self.distribution = distribution
        self.policy_id = resolved_id


class ScoreMatchingBatch(StrictModule):
    samples: TrajectoryStateTimeSamples
    probe_key: Array
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        samples: TrajectoryStateTimeSamples,
        probe_key: Key[Array, ""],
        /,
        *,
        batch_id: str,
    ):
        if not isinstance(samples, TrajectoryStateTimeSamples):
            raise TypeError("samples must be TrajectoryStateTimeSamples.")
        if not batch_id:
            raise ValueError("batch_id must be non-empty.")
        self.samples = samples
        self.probe_key = probe_key
        self.batch_id = str(batch_id)


class ScoreMatchingDiagnostics(StrictModule):
    objective: Array
    mean_score_norm: Array
    mean_divergence: Array
    divergence_standard_error: Array
    path_standard_error: Array
    valid_fraction: Array
    effective_sample_size: Array
    time_coverage: Array
    finite: Array
    num_paths: int = eqx.field(static=True)
    num_times: int = eqx.field(static=True)
    num_probes: int = eqx.field(static=True)
    method: ScoreMatchingMethod = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return bool(self.finite) and bool(self.valid_fraction > 0.0)


class _ScoreNodeEvaluation(StrictModule):
    loss: Array
    score_norm: Array
    divergence: Array
    divergence_standard_error: Array
    weights: Array
    valid: Array
    path_indices: Array


def _as_samples(
    value: TrajectoryStateTimeSamples | StochasticTrajectory,
    /,
) -> TrajectoryStateTimeSamples:
    if isinstance(value, TrajectoryStateTimeSamples):
        return value
    if isinstance(value, StochasticTrajectory):
        return trajectory_state_time_samples(value)
    raise TypeError("Score sample providers must return trajectory or state-time samples.")


def _probes(
    key: Key[Array, ""],
    shape: tuple[int, ...],
    policy: ScoreMatchingPolicy,
    dtype,
    /,
) -> Array:
    full_shape = (policy.num_probes,) + shape
    if policy.distribution == "rademacher":
        return jr.rademacher(key, full_shape, dtype=dtype)
    return jr.normal(key, full_shape, dtype=dtype)


def _normalized_weights(
    log_weights: Array,
    valid: Array,
    /,
) -> Array:
    valid_count = jnp.sum(valid)
    valid_count = eqx.error_if(
        valid_count,
        valid_count <= 0,
        "Score-matching batch contains no valid state-time samples.",
    )
    safe_log_weights = jnp.where(valid, log_weights, -jnp.inf)
    reference = jnp.max(safe_log_weights) + 0.0 * valid_count
    unnormalized = jnp.where(valid, jnp.exp(log_weights - reference), 0.0)
    mass = jnp.sum(unnormalized)
    mass = eqx.error_if(
        mass,
        ~(jnp.isfinite(mass) & (mass > 0.0)),
        "Score-matching sample weights have zero finite mass.",
    )
    return unnormalized / mass


def _weighted_mean(values: Array, weights: Array, /) -> Array:
    return jnp.sum(
        weights.reshape(weights.shape + (1,) * (values.ndim - weights.ndim)) * values,
        axis=tuple(range(weights.ndim)),
    )


def _path_standard_error(
    node_loss: Array,
    weights: Array,
    path_indices: Array,
    num_paths: int,
    /,
) -> Array:
    path_weight = jax.ops.segment_sum(weights, path_indices, num_paths)
    path_total = jax.ops.segment_sum(weights * node_loss, path_indices, num_paths)
    active = path_weight > 0.0
    path_mean = jnp.where(active, path_total / jnp.maximum(path_weight, 1e-300), 0.0)
    count = jnp.sum(active)
    mean = jnp.sum(jnp.where(active, path_mean, 0.0)) / jnp.maximum(count, 1)
    squared = jnp.sum(jnp.where(active, (path_mean - mean) ** 2, 0.0))
    variance = jnp.where(count > 1, squared / (count - 1), jnp.nan)
    return jnp.sqrt(variance / jnp.maximum(count, 1))


class ScoreMatchingObjective(AbstractSamplingObjectiveTerm):
    """Particle-first score-field objective with path-cluster diagnostics."""

    fixed_samples: TrajectoryStateTimeSamples | None
    sample_provider: ScoreSampleProvider | None
    policy: ScoreMatchingPolicy
    scalar_weight: Array
    score_name: str = eqx.field(static=True)
    sampling_mode: ScoreMatchingSamplingMode = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        score_name: str,
        samples: TrajectoryStateTimeSamples
        | StochasticTrajectory
        | ScoreSampleProvider,
        /,
        *,
        policy: ScoreMatchingPolicy | None = None,
        sampling_mode: ScoreMatchingSamplingMode = "fixed",
        scalar_weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not isinstance(score_name, str) or not score_name:
            raise ValueError("score_name must be a non-empty string.")
        if sampling_mode not in ("fixed", "resample"):
            raise ValueError("sampling_mode must be 'fixed' or 'resample'.")
        resolved_policy = ScoreMatchingPolicy() if policy is None else policy
        if not isinstance(resolved_policy, ScoreMatchingPolicy):
            raise TypeError("policy must be a ScoreMatchingPolicy.")
        if sampling_mode == "resample":
            if not callable(samples):
                raise TypeError("Resampled score matching requires a sample provider.")
            fixed = None
            provider = cast(ScoreSampleProvider, samples)
        else:
            if callable(samples):
                raise TypeError("Fixed score matching requires materialized samples.")
            fixed = _as_samples(samples)
            provider = None
        weight = jnp.asarray(scalar_weight, dtype=float).reshape(())
        if bool(~jnp.isfinite(weight)) or float(weight) < 0.0:
            raise ValueError("scalar_weight must be finite and nonnegative.")
        self.fixed_samples = fixed
        self.sample_provider = provider
        self.policy = resolved_policy
        self.scalar_weight = weight
        self.score_name = score_name
        self.sampling_mode = sampling_mode
        self.label = label

    def sample(self, *, key: Key[Array, ""] = jr.key(0)) -> ScoreMatchingBatch:
        sample_key, probe_key = jr.split(key)
        if self.sampling_mode == "fixed":
            if self.fixed_samples is None:
                raise RuntimeError("Fixed score samples are unavailable.")
            samples = self.fixed_samples
        else:
            if self.sample_provider is None:
                raise RuntimeError("Score sample provider is unavailable.")
            samples = _as_samples(self.sample_provider(sample_key))
        return ScoreMatchingBatch(
            samples,
            probe_key,
            batch_id=self.label or self.policy.policy_id,
        )

    def _score_function(
        self,
        functions: Mapping[str, DomainFunction],
        samples: TrajectoryStateTimeSamples,
        /,
    ) -> DomainFunction:
        if self.score_name not in functions:
            raise KeyError(f"Missing score field {self.score_name!r}.")
        score = functions[self.score_name]
        if not isinstance(score, DomainFunction):
            raise TypeError("score field must be a DomainFunction.")
        allowed = {samples.state_label, samples.time_label}
        unknown = tuple(label for label in score.deps if label not in allowed)
        if unknown or samples.state_label not in score.deps:
            raise ValueError(
                "score field must depend on the state label and optionally the time label."
            )
        return score

    def _evaluate_nodes(
        self,
        functions: Mapping[str, DomainFunction],
        batch: ScoreMatchingBatch,
        /,
    ) -> _ScoreNodeEvaluation:
        if not isinstance(batch, ScoreMatchingBatch):
            raise TypeError("batch must be a ScoreMatchingBatch.")
        samples = batch.samples
        score = self._score_function(functions, samples)
        sample_rank = len(samples.leading_axes) + 1
        sample_shape = samples.log_weights.shape
        node_count = prod(sample_shape)
        state_shape = samples.states.shape[sample_rank:]
        states = jnp.asarray(samples.states.data).reshape((node_count,) + state_shape)
        times = jnp.asarray(samples.times.data).reshape((node_count,))
        valid = jnp.asarray(samples.valid.data, dtype=bool).reshape((node_count,))
        log_weights = jnp.asarray(samples.log_weights.data, dtype=float).reshape(
            (node_count,)
        )
        path_indices = jnp.asarray(samples.path_indices.data, dtype=jnp.int32).reshape(
            (node_count,)
        )
        safe_states = jnp.where(
            valid.reshape((node_count,) + (1,) * len(state_shape)),
            states,
            jnp.zeros_like(states),
        )
        safe_times = jnp.where(valid, times, 0.0)
        node_keys = jr.split(batch.probe_key, node_count)

        def score_at(state, time, key):
            arguments = tuple(
                state if dependency == samples.state_label else time
                for dependency in score.deps
            )
            return jnp.asarray(score.func(*arguments, key=key))

        def exact_node(state, time, key):
            value = score_at(state, time, key)
            jacobian = jax.jacrev(lambda current: score_at(current, time, key))(state)
            divergence = jnp.trace(
                jacobian.reshape((prod(state_shape), prod(state_shape)))
            )
            norm = jnp.sum(jnp.abs(value) ** 2)
            return 0.5 * norm + divergence, norm, divergence, jnp.asarray(0.0)

        trace_policy = StochasticTracePolicy(
            max(self.policy.num_probes, 2),
            distribution=self.policy.distribution,
        )

        def implicit_node(state, time, key):
            value = score_at(state, time, key)
            estimate = stochastic_divergence_samples(
                lambda current: score_at(current, time, key),
                state,
                key,
                policy=trace_policy,
            )
            norm = jnp.sum(jnp.abs(value) ** 2)
            return (
                0.5 * norm + estimate.mean,
                norm,
                estimate.mean,
                estimate.standard_error,
            )

        def sliced_node(state, time, key):
            value = score_at(state, time, key)
            probes = _probes(key, state_shape, self.policy, state.dtype)

            def one(probe):
                _, derivative = jax.jvp(
                    lambda current: score_at(current, time, key),
                    (state,),
                    (probe,),
                )
                projection = jnp.sum(probe * value)
                directional_divergence = jnp.sum(probe * derivative)
                return 0.5 * jnp.abs(projection) ** 2 + directional_divergence, directional_divergence

            losses, divergences = jax.vmap(one)(probes)
            divergence_mean = jnp.mean(divergences)
            divergence_error = jnp.std(divergences, ddof=1) / jnp.sqrt(
                float(self.policy.num_probes)
            )
            return (
                jnp.mean(losses),
                jnp.sum(jnp.abs(value) ** 2),
                divergence_mean,
                divergence_error,
            )

        if self.policy.method == "exact":
            node_loss, score_norm, divergence, divergence_error = jax.vmap(exact_node)(
                safe_states,
                safe_times,
                node_keys,
            )
        elif self.policy.method == "implicit":
            node_loss, score_norm, divergence, divergence_error = jax.vmap(
                implicit_node
            )(safe_states, safe_times, node_keys)
        else:
            node_loss, score_norm, divergence, divergence_error = jax.vmap(sliced_node)(
                safe_states,
                safe_times,
                node_keys,
            )
        if score_norm.shape != (node_count,):
            raise ValueError("score field output must have the same shape as each state.")
        normalized = _normalized_weights(log_weights, valid)
        return _ScoreNodeEvaluation(
            loss=node_loss,
            score_norm=score_norm,
            divergence=divergence,
            divergence_standard_error=divergence_error,
            weights=normalized,
            valid=valid,
            path_indices=path_indices,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        iter_: int | None = None,
        batch: ScoreMatchingBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        materialized = self.sample(key=key) if batch is None else batch
        evaluation = self._evaluate_nodes(functions, materialized)
        return self.scalar_weight * jnp.sum(evaluation.weights * evaluation.loss)

    def diagnostics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        batch: ScoreMatchingBatch | None = None,
    ) -> ScoreMatchingDiagnostics:
        materialized = self.sample(key=key) if batch is None else batch
        evaluation = self._evaluate_nodes(functions, materialized)
        samples = materialized.samples
        objective = self.scalar_weight * jnp.sum(
            evaluation.weights * evaluation.loss
        )
        score_norm = jnp.sqrt(
            jnp.sum(evaluation.weights * evaluation.score_norm)
        )
        divergence = jnp.sum(evaluation.weights * evaluation.divergence)
        divergence_error = jnp.sqrt(
            jnp.sum(
                evaluation.weights**2
                * evaluation.divergence_standard_error**2
            )
        )
        path_error = _path_standard_error(
            evaluation.loss,
            evaluation.weights,
            evaluation.path_indices,
            samples.num_paths,
        )
        valid_array = jnp.asarray(samples.valid.data, dtype=float)
        time_coverage = jnp.mean(
            valid_array,
            axis=tuple(range(valid_array.ndim - 1)),
        )
        ess = 1.0 / jnp.sum(evaluation.weights**2)
        finite = (
            jnp.isfinite(objective)
            & jnp.isfinite(score_norm)
            & jnp.isfinite(divergence)
        )
        return ScoreMatchingDiagnostics(
            objective=objective,
            mean_score_norm=score_norm,
            mean_divergence=divergence,
            divergence_standard_error=divergence_error,
            path_standard_error=path_error,
            valid_fraction=jnp.mean(evaluation.valid),
            effective_sample_size=ess,
            time_coverage=time_coverage,
            finite=finite,
            num_paths=samples.num_paths,
            num_times=samples.num_times,
            num_probes=self.policy.num_probes,
            method=self.policy.method,
            policy_id=self.policy.policy_id,
        )


__all__ = [
    "ScoreMatchingBatch",
    "ScoreMatchingDiagnostics",
    "ScoreMatchingMethod",
    "ScoreMatchingObjective",
    "ScoreMatchingPolicy",
    "ScoreMatchingSamplingMode",
    "ScoreSampleProvider",
]
