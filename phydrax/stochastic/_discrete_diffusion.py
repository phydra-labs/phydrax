#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


DiffusionPredictionKind: TypeAlias = Literal["epsilon", "clean", "score", "velocity"]
DiscreteTerminalRelationship: TypeAlias = Literal["exact", "approximate", "assumed"]


def _event_shape(value, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("event_shape must contain positive dimensions.")
    return shape


def _derived_schedule(beta: np.ndarray) -> dict[str, Array]:
    beta64 = np.asarray(beta, dtype=np.float64)
    alpha = 1.0 - beta64
    cumulative = np.cumprod(alpha)
    previous = np.concatenate((np.ones((1,), dtype=np.float64), cumulative[:-1]))
    one_minus = 1.0 - cumulative
    previous_one_minus = 1.0 - previous
    posterior_variance = beta64 * previous_one_minus / one_minus
    clipped_head = posterior_variance[min(1, posterior_variance.shape[0] - 1)]
    values = {
        "beta": beta64,
        "alpha": alpha,
        "cumulative_alpha": cumulative,
        "previous_cumulative_alpha": previous,
        "sqrt_cumulative_alpha": np.sqrt(cumulative),
        "sqrt_one_minus_cumulative_alpha": np.sqrt(one_minus),
        "posterior_variance": posterior_variance,
        "posterior_log_variance": np.log(
            np.concatenate(([clipped_head], posterior_variance[1:]))
        ),
        "posterior_mean_clean": beta64 * np.sqrt(previous) / one_minus,
        "posterior_mean_noisy": previous_one_minus * np.sqrt(alpha) / one_minus,
    }
    return {name: jnp.asarray(value) for name, value in values.items()}


class DiscreteGaussianDiffusionSchedule(StrictModule):
    """Finite Gaussian corruption schedule derived once in certification precision."""

    beta: Array
    alpha: Array
    cumulative_alpha: Array
    previous_cumulative_alpha: Array
    sqrt_cumulative_alpha: Array
    sqrt_one_minus_cumulative_alpha: Array
    posterior_variance: Array
    posterior_log_variance: Array
    posterior_mean_clean: Array
    posterior_mean_noisy: Array
    num_steps: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, beta: ArrayLike, /, *, schedule_id: str | None = None):
        host = np.asarray(beta, dtype=np.float64).reshape((-1,))
        if host.size <= 0 or np.any(~np.isfinite(host)) or np.any((host <= 0) | (host >= 1)):
            raise ValueError("beta must be a finite non-empty vector strictly inside (0, 1).")
        derived = _derived_schedule(host)
        if np.any(np.diff(np.asarray(derived["cumulative_alpha"])) >= 0):
            raise ValueError("cumulative signal must decrease strictly.")
        identifier = schedule_id or canonical_fingerprint(
            {
                "kind": "discrete-gaussian-diffusion-schedule",
                "beta": host.tolist(),
            }
        )
        if not identifier:
            raise ValueError("schedule_id must be non-empty or None.")
        for name, value in derived.items():
            setattr(self, name, value)
        self.num_steps = int(host.size)
        self.schedule_id = identifier

    @classmethod
    def linear(
        cls,
        num_steps: int,
        /,
        *,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> "DiscreteGaussianDiffusionSchedule":
        count = int(num_steps)
        if count <= 0:
            raise ValueError("num_steps must be positive.")
        return cls(np.linspace(beta_start, beta_end, count, dtype=np.float64))

    @classmethod
    def cosine(
        cls,
        num_steps: int,
        /,
        *,
        offset: float = 0.008,
        maximum_beta: float = 0.999,
    ) -> "DiscreteGaussianDiffusionSchedule":
        count = int(num_steps)
        if count <= 0:
            raise ValueError("num_steps must be positive.")
        points = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
        cumulative = np.cos((points + offset) / (1.0 + offset) * np.pi / 2.0) ** 2
        cumulative = cumulative / cumulative[0]
        beta = 1.0 - cumulative[1:] / cumulative[:-1]
        return cls(np.clip(beta, 1e-12, maximum_beta))

    def _extract(self, values: Array, timestep: ArrayLike, shape: tuple[int, ...]) -> Array:
        value = jnp.asarray(timestep)
        if not jnp.issubdtype(value.dtype, jnp.integer):
            raise TypeError("Diffusion timesteps must use an integer dtype.")
        time = value.astype(jnp.int32)
        time = eqx.error_if(
            time,
            jnp.any((time < 0) | (time >= self.num_steps)),
            "Diffusion timestep is outside the schedule.",
        )
        if time.shape == ():
            selected = values[time]
            return jnp.broadcast_to(selected, shape)
        if time.ndim >= len(shape) or tuple(shape[: time.ndim]) != tuple(time.shape):
            raise ValueError("Timestep arrays must match complete leading sample axes.")
        selected = values[time]
        return selected.reshape(time.shape + (1,) * (len(shape) - time.ndim))

    def corrupt(self, clean: ArrayLike, noise: ArrayLike, timestep: ArrayLike, /) -> Array:
        state = jnp.asarray(clean)
        if jnp.iscomplexobj(state) or not jnp.issubdtype(state.dtype, jnp.floating):
            raise TypeError("Discrete Gaussian corruption requires real floating states.")
        perturbation = jnp.asarray(noise, dtype=state.dtype)
        if state.shape != perturbation.shape:
            raise ValueError("clean and noise must have identical shapes.")
        signal = self._extract(self.sqrt_cumulative_alpha, timestep, state.shape)
        scale = self._extract(self.sqrt_one_minus_cumulative_alpha, timestep, state.shape)
        return signal * state + scale * perturbation

    def clean_from_epsilon(self, noisy, epsilon, timestep, /) -> Array:
        state = jnp.asarray(noisy)
        noise = jnp.asarray(epsilon, dtype=state.dtype)
        signal = self._extract(self.sqrt_cumulative_alpha, timestep, state.shape)
        scale = self._extract(self.sqrt_one_minus_cumulative_alpha, timestep, state.shape)
        return (state - scale * noise) / signal

    def epsilon_from_clean(self, noisy, clean, timestep, /) -> Array:
        state = jnp.asarray(noisy)
        signal = self._extract(self.sqrt_cumulative_alpha, timestep, state.shape)
        scale = self._extract(self.sqrt_one_minus_cumulative_alpha, timestep, state.shape)
        return (state - signal * jnp.asarray(clean, dtype=state.dtype)) / scale

    def score_from_epsilon(self, epsilon, timestep, shape, /) -> Array:
        noise = jnp.asarray(epsilon)
        scale = self._extract(self.sqrt_one_minus_cumulative_alpha, timestep, tuple(shape))
        return -noise / scale

    def epsilon_from_prediction(self, noisy, prediction, timestep, kind: DiffusionPredictionKind, /):
        state = jnp.asarray(noisy)
        value = jnp.asarray(prediction, dtype=state.dtype)
        if value.shape != state.shape:
            raise ValueError("Diffusion prediction must match the noisy state shape.")
        if kind == "epsilon":
            return value
        if kind == "clean":
            return self.epsilon_from_clean(state, value, timestep)
        scale = self._extract(self.sqrt_one_minus_cumulative_alpha, timestep, state.shape)
        signal = self._extract(self.sqrt_cumulative_alpha, timestep, state.shape)
        if kind == "score":
            return -scale * value
        if kind == "velocity":
            return scale * state + signal * value
        raise ValueError("Unknown diffusion prediction kind.")

    def posterior(self, clean, noisy, timestep, /):
        state = jnp.asarray(noisy)
        clean_state = jnp.asarray(clean, dtype=state.dtype)
        mean = self._extract(self.posterior_mean_clean, timestep, state.shape) * clean_state
        mean = mean + self._extract(self.posterior_mean_noisy, timestep, state.shape) * state
        variance = self._extract(self.posterior_variance, timestep, state.shape)
        log_variance = self._extract(self.posterior_log_variance, timestep, state.shape)
        return mean, variance, log_variance


class DiscreteDiffusionSample(StrictModule):
    final_state: Array
    trajectory: Array
    timesteps: Array
    valid: Array
    sampler_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    terminal_relationship: DiscreteTerminalRelationship = eqx.field(static=True)
    terminal_reference_id: str = eqx.field(static=True)


class AncestralGaussianDiffusion(StrictModule):
    """Ancestral reverse chain from a declared standard-Normal terminal reference."""

    schedule: DiscreteGaussianDiffusionSchedule
    predictor: Any
    event_shape: tuple[int, ...] = eqx.field(static=True)
    prediction_kind: DiffusionPredictionKind = eqx.field(static=True)
    terminal_relationship: DiscreteTerminalRelationship = eqx.field(static=True)
    terminal_reference_id: str = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule,
        predictor,
        event_shape,
        /,
        *,
        prediction_kind="epsilon",
        terminal_relationship: DiscreteTerminalRelationship = "approximate",
        terminal_reference_id: str = "standard-normal",
    ):
        if not isinstance(schedule, DiscreteGaussianDiffusionSchedule) or not callable(predictor):
            raise TypeError("Ancestral diffusion requires a schedule and predictor.")
        if prediction_kind not in ("epsilon", "clean", "score", "velocity"):
            raise ValueError("Unknown prediction kind.")
        if terminal_relationship not in ("exact", "approximate", "assumed"):
            raise ValueError("Unknown discrete terminal relationship.")
        if not terminal_reference_id:
            raise ValueError("terminal_reference_id must be non-empty.")
        events = _event_shape(event_shape)
        self.schedule = schedule
        self.predictor = predictor
        self.event_shape = events
        self.prediction_kind = prediction_kind
        self.terminal_relationship = terminal_relationship
        self.terminal_reference_id = terminal_reference_id
        self.sampler_id = canonical_fingerprint(
            {
                "kind": "ancestral-gaussian-diffusion",
                "schedule_id": schedule.schedule_id,
                "event_shape": list(events),
                "prediction_kind": prediction_kind,
                "terminal_reference_id": terminal_reference_id,
            }
        )

    def sample(self, key: Key[Array, ""], sample_shape: Sequence[int], /) -> DiscreteDiffusionSample:
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("sample_shape dimensions must be positive.")
        root, initial_key = jr.split(key)
        initial = jr.normal(initial_key, samples + self.event_shape)
        timesteps = jnp.arange(self.schedule.num_steps - 1, -1, -1, dtype=jnp.int32)

        def step(carry, timestep):
            state, current_key = carry
            current_key, model_key, noise_key = jr.split(current_key, 3)
            batch_time = jnp.full(samples, timestep, dtype=jnp.int32)
            prediction = self.predictor(state, batch_time, key=model_key)
            epsilon = self.schedule.epsilon_from_prediction(
                state, prediction, batch_time, self.prediction_kind
            )
            clean = self.schedule.clean_from_epsilon(state, epsilon, batch_time)
            mean, _, log_variance = self.schedule.posterior(clean, state, batch_time)
            noise = jr.normal(noise_key, state.shape, dtype=state.dtype)
            active = (timestep > 0).reshape((1,) * state.ndim)
            next_state = mean + active * jnp.exp(0.5 * log_variance) * noise
            return (next_state, current_key), next_state

        (final, _), trajectory = jax.lax.scan(step, (initial, root), timesteps)
        valid = jnp.all(jnp.isfinite(final), axis=tuple(range(len(samples), final.ndim)))
        return DiscreteDiffusionSample(
            final,
            trajectory,
            timesteps,
            valid,
            self.sampler_id,
            self.schedule.schedule_id,
            self.terminal_relationship,
            self.terminal_reference_id,
        )


class DDIMTransport(StrictModule):
    schedule: DiscreteGaussianDiffusionSchedule
    predictor: Any
    inference_timesteps: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)
    prediction_kind: DiffusionPredictionKind = eqx.field(static=True)
    eta: float = eqx.field(static=True)
    terminal_relationship: DiscreteTerminalRelationship = eqx.field(static=True)
    terminal_reference_id: str = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule,
        predictor,
        event_shape,
        /,
        *,
        num_inference_steps: int,
        eta: float = 0.0,
        prediction_kind: DiffusionPredictionKind = "epsilon",
        terminal_relationship: DiscreteTerminalRelationship = "approximate",
        terminal_reference_id: str = "standard-normal",
    ):
        if not isinstance(schedule, DiscreteGaussianDiffusionSchedule) or not callable(predictor):
            raise TypeError("DDIM requires a schedule and predictor.")
        count = int(num_inference_steps)
        if count <= 1 or count > schedule.num_steps:
            raise ValueError("num_inference_steps must lie in [2, schedule.num_steps].")
        if prediction_kind not in ("epsilon", "clean", "score", "velocity"):
            raise ValueError("Unknown prediction kind.")
        if terminal_relationship not in ("exact", "approximate", "assumed"):
            raise ValueError("Unknown discrete terminal relationship.")
        if not terminal_reference_id:
            raise ValueError("terminal_reference_id must be non-empty.")
        stochasticity = float(eta)
        if not np.isfinite(stochasticity) or stochasticity < 0.0:
            raise ValueError("eta must be finite and nonnegative.")
        indices = np.rint(np.linspace(schedule.num_steps - 1, 0, count)).astype(np.int32)
        if np.any(np.diff(indices) >= 0):
            raise ValueError("DDIM inference timesteps must be strictly decreasing.")
        self.schedule = schedule
        self.predictor = predictor
        self.inference_timesteps = jnp.asarray(indices)
        self.event_shape = _event_shape(event_shape)
        self.prediction_kind = prediction_kind
        self.eta = stochasticity
        self.terminal_relationship = terminal_relationship
        self.terminal_reference_id = terminal_reference_id
        self.sampler_id = canonical_fingerprint(
            {
                "kind": "ddim-transport",
                "schedule_id": schedule.schedule_id,
                "timesteps": indices.tolist(),
                "eta": stochasticity,
                "prediction_kind": prediction_kind,
                "terminal_reference_id": terminal_reference_id,
            }
        )

    def sample(self, key: Key[Array, ""], sample_shape: Sequence[int], /) -> DiscreteDiffusionSample:
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("sample_shape dimensions must be positive.")
        root, initial_key = jr.split(key)
        initial = jr.normal(initial_key, samples + self.event_shape)
        previous = jnp.concatenate((self.inference_timesteps[1:], jnp.asarray([-1], dtype=jnp.int32)))

        def step(carry, pair):
            state, current_key = carry
            timestep, previous_timestep = pair
            current_key, model_key, noise_key = jr.split(current_key, 3)
            batch_time = jnp.full(samples, timestep, dtype=jnp.int32)
            prediction = self.predictor(state, batch_time, key=model_key)
            epsilon = self.schedule.epsilon_from_prediction(
                state, prediction, batch_time, self.prediction_kind
            )
            clean = self.schedule.clean_from_epsilon(state, epsilon, batch_time)
            alpha = self.schedule.cumulative_alpha[timestep]
            alpha_previous = jnp.where(
                previous_timestep >= 0,
                self.schedule.cumulative_alpha[jnp.maximum(previous_timestep, 0)],
                1.0,
            )
            sigma = self.eta * jnp.sqrt(
                (1.0 - alpha_previous) / (1.0 - alpha) * (1.0 - alpha / alpha_previous)
            )
            direction = jnp.sqrt(jnp.maximum(1.0 - alpha_previous - sigma**2, 0.0)) * epsilon
            noise = jr.normal(noise_key, state.shape, dtype=state.dtype)
            next_state = jnp.sqrt(alpha_previous) * clean + direction + sigma * noise
            return (next_state, current_key), next_state

        (final, _), trajectory = jax.lax.scan(
            step,
            (initial, root),
            (self.inference_timesteps, previous),
        )
        valid = jnp.all(jnp.isfinite(final), axis=tuple(range(len(samples), final.ndim)))
        return DiscreteDiffusionSample(
            final,
            trajectory,
            self.inference_timesteps,
            valid,
            self.sampler_id,
            self.schedule.schedule_id,
            self.terminal_relationship,
            self.terminal_reference_id,
        )


def discrete_denoising_loss(
    predictor: Callable,
    schedule: DiscreteGaussianDiffusionSchedule,
    clean: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    prediction_kind: DiffusionPredictionKind = "epsilon",
) -> Array:
    state = jnp.asarray(clean)
    if state.ndim < 2:
        raise ValueError("clean data must have a leading sample axis and event axes.")
    time_key, noise_key, model_key = jr.split(key, 3)
    time = jr.randint(time_key, (state.shape[0],), 0, schedule.num_steps)
    noise = jr.normal(noise_key, state.shape, dtype=state.dtype)
    noisy = schedule.corrupt(state, noise, time)
    prediction = predictor(noisy, time, key=model_key)
    epsilon = schedule.epsilon_from_prediction(noisy, prediction, time, prediction_kind)
    return jnp.mean((epsilon - noise) ** 2)


__all__ = [
    "AncestralGaussianDiffusion",
    "DDIMTransport",
    "DiffusionPredictionKind",
    "DiscreteDiffusionSample",
    "DiscreteGaussianDiffusionSchedule",
    "DiscreteTerminalRelationship",
    "discrete_denoising_loss",
]
