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


CategoricalTerminalRelationship: TypeAlias = Literal["exact", "approximate", "assumed"]


def _normalize_rows(matrix: np.ndarray, /) -> np.ndarray:
    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError("Categorical kernels must have shape (steps, classes, classes).")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("Categorical kernels must be finite and nonnegative.")
    mass = np.sum(matrix, axis=-1, keepdims=True)
    if np.any(mass <= 0.0) or not np.allclose(mass, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Every categorical transition row must sum to one.")
    return matrix


class CategoricalDiffusionSchedule(StrictModule):
    """Finite categorical corruption kernels and their exact cumulative products."""

    transition: Array
    cumulative: Array
    num_steps: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, transition: ArrayLike, /, *, schedule_id: str | None = None):
        kernels = _normalize_rows(np.asarray(transition, dtype=np.float64))
        cumulative = []
        value = np.eye(kernels.shape[1], dtype=np.float64)
        for kernel in kernels:
            value = value @ kernel
            cumulative.append(value.copy())
        identifier = schedule_id or canonical_fingerprint(
            {
                "kind": "categorical-diffusion-schedule",
                "steps": int(kernels.shape[0]),
                "classes": int(kernels.shape[1]),
                "transition": kernels.tolist(),
            }
        )
        if not identifier:
            raise ValueError("schedule_id must be non-empty or None.")
        self.transition = jnp.asarray(kernels)
        self.cumulative = jnp.asarray(np.stack(cumulative))
        self.num_steps = int(kernels.shape[0])
        self.num_classes = int(kernels.shape[1])
        self.schedule_id = identifier

    @classmethod
    def uniform(
        cls,
        num_steps: int,
        num_classes: int,
        /,
        *,
        beta_start: float = 1e-3,
        beta_end: float = 0.1,
    ):
        steps = int(num_steps)
        classes = int(num_classes)
        if steps <= 0 or classes <= 1:
            raise ValueError("num_steps must be positive and num_classes exceed one.")
        kernels = []
        for beta in np.linspace(beta_start, beta_end, steps):
            kernels.append(
                (1.0 - beta) * np.eye(classes) + beta * np.ones((classes, classes)) / classes
            )
        return cls(np.stack(kernels))

    @classmethod
    def absorbing(
        cls,
        num_steps: int,
        num_classes: int,
        absorbing_class: int,
        /,
        *,
        beta_start: float = 1e-3,
        beta_end: float = 0.1,
    ):
        steps = int(num_steps)
        classes = int(num_classes)
        absorbing = int(absorbing_class)
        if absorbing < 0 or absorbing >= classes:
            raise ValueError("absorbing_class is out of range.")
        kernels = []
        for beta in np.linspace(beta_start, beta_end, steps):
            kernel = (1.0 - beta) * np.eye(classes)
            kernel[:, absorbing] += beta
            kernel[absorbing] = 0.0
            kernel[absorbing, absorbing] = 1.0
            kernels.append(kernel)
        return cls(np.stack(kernels))

    def _validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if not jnp.issubdtype(value.dtype, jnp.integer):
            raise TypeError("Categorical states must use an integer dtype.")
        return eqx.error_if(
            value,
            jnp.any((value < 0) | (value >= self.num_classes)),
            "Categorical state contains an out-of-range class.",
        )

    def _validate_timestep(self, timestep: ArrayLike, /) -> Array:
        value = jnp.asarray(timestep)
        if not jnp.issubdtype(value.dtype, jnp.integer):
            raise TypeError("Categorical timesteps must use an integer dtype.")
        time = value.astype(jnp.int32)
        return eqx.error_if(
            time,
            jnp.any((time < 0) | (time >= self.num_steps)),
            "Categorical timestep is outside the schedule.",
        )

    def marginal_probabilities(self, clean: ArrayLike, timestep: ArrayLike, /) -> Array:
        state = self._validate_state(clean)
        time = self._validate_timestep(timestep)
        if time.shape == ():
            return self.cumulative[time, state]
        if time.ndim > state.ndim or tuple(state.shape[: time.ndim]) != tuple(time.shape):
            raise ValueError("Timestep arrays must match complete leading sample axes.")
        flat = state.reshape((int(time.size), -1))
        probabilities = jax.vmap(lambda t, x: self.cumulative[t, x])(
            time.reshape((-1,)), flat
        )
        return probabilities.reshape(state.shape + (self.num_classes,))

    def corrupt(self, clean: ArrayLike, timestep: ArrayLike, key: Key[Array, ""], /) -> Array:
        probabilities = self.marginal_probabilities(clean, timestep)
        return jr.categorical(key, jnp.log(probabilities), axis=-1).astype(jnp.int32)

    def posterior_probabilities(
        self,
        clean: ArrayLike,
        noisy: ArrayLike,
        timestep: ArrayLike,
        /,
    ) -> Array:
        x0 = self._validate_state(clean)
        xt = self._validate_state(noisy)
        if x0.shape != xt.shape:
            raise ValueError("clean and noisy categorical states must match shapes.")
        time = self._validate_timestep(timestep)

        def one(clean_value, noisy_value, step):
            previous = jax.lax.cond(
                step > 0,
                lambda index: self.cumulative[index - 1, clean_value],
                lambda index: jax.nn.one_hot(clean_value, self.num_classes),
                step,
            )
            likelihood = self.transition[step, :, noisy_value]
            unnormalized = previous * likelihood
            normalizer = jnp.sum(unnormalized)
            unnormalized = eqx.error_if(
                unnormalized,
                normalizer <= 0.0,
                "Categorical posterior conditions on an impossible transition.",
            )
            return unnormalized / normalizer

        if time.shape == ():
            flat = jax.vmap(lambda a, b: one(a, b, time))(
                x0.reshape(-1), xt.reshape(-1)
            )
        else:
            if time.ndim > x0.ndim or tuple(x0.shape[: time.ndim]) != tuple(time.shape):
                raise ValueError("Timestep arrays must match complete leading sample axes.")
            flat_x0 = x0.reshape((int(time.size), -1))
            flat_xt = xt.reshape((int(time.size), -1))
            flat = jax.vmap(
                lambda row0, rowt, step: jax.vmap(lambda a, b: one(a, b, step))(
                    row0, rowt
                )
            )(flat_x0, flat_xt, time.reshape((-1,))).reshape((-1, self.num_classes))
        return flat.reshape(x0.shape + (self.num_classes,))

    def reverse_probabilities_from_clean_logits(
        self,
        noisy: ArrayLike,
        clean_logits: ArrayLike,
        timestep: ArrayLike,
        /,
    ) -> Array:
        """Convert predicted clean-state logits into one exact-kernel reverse step."""
        xt = self._validate_state(noisy)
        logits = jnp.asarray(clean_logits)
        if logits.shape != xt.shape + (self.num_classes,):
            raise ValueError("Clean-state logits must append one class axis to noisy state.")
        time = self._validate_timestep(timestep)

        def one(noisy_value, predicted_logits, step):
            previous = jax.lax.cond(
                step > 0,
                lambda index: self.cumulative[index - 1],
                lambda index: jnp.eye(self.num_classes, dtype=self.transition.dtype),
                step,
            )
            predicted_clean = jax.nn.softmax(predicted_logits)
            predicted_previous = predicted_clean @ previous
            unnormalized = predicted_previous * self.transition[step, :, noisy_value]
            normalizer = jnp.sum(unnormalized)
            unnormalized = eqx.error_if(
                unnormalized,
                normalizer <= 0.0,
                "Predicted categorical reverse transition has zero probability.",
            )
            return unnormalized / normalizer

        if time.shape == ():
            flat = jax.vmap(lambda value, prediction: one(value, prediction, time))(
                xt.reshape((-1,)), logits.reshape((-1, self.num_classes))
            )
        else:
            if time.ndim > xt.ndim or tuple(xt.shape[: time.ndim]) != tuple(time.shape):
                raise ValueError("Timestep arrays must match complete leading sample axes.")
            flat_xt = xt.reshape((int(time.size), -1))
            flat_logits = logits.reshape((int(time.size), -1, self.num_classes))
            flat = jax.vmap(
                lambda row, predictions, step: jax.vmap(
                    lambda value, prediction: one(value, prediction, step)
                )(row, predictions)
            )(flat_xt, flat_logits, time.reshape((-1,))).reshape(
                (-1, self.num_classes)
            )
        return flat.reshape(xt.shape + (self.num_classes,))


class CategoricalDiffusionSample(StrictModule):
    final_state: Array
    trajectory: Array
    timesteps: Array
    valid: Array
    sampler_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    terminal_relationship: CategoricalTerminalRelationship = eqx.field(static=True)
    terminal_reference_id: str = eqx.field(static=True)


class CategoricalReverseDiffusion(StrictModule):
    """Learned reverse categorical chain with an explicit terminal reference."""

    schedule: CategoricalDiffusionSchedule
    predictor: Any
    terminal_probabilities: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)
    terminal_relationship: CategoricalTerminalRelationship = eqx.field(static=True)
    terminal_reference_id: str = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule,
        predictor,
        event_shape,
        /,
        *,
        terminal_probabilities: ArrayLike | None = None,
        terminal_relationship: CategoricalTerminalRelationship = "assumed",
        terminal_reference_id: str | None = None,
    ):
        if not isinstance(schedule, CategoricalDiffusionSchedule) or not callable(predictor):
            raise TypeError("Categorical reverse diffusion requires schedule and predictor.")
        shape = tuple(int(size) for size in event_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("event_shape must contain positive dimensions.")
        if terminal_relationship not in ("exact", "approximate", "assumed"):
            raise ValueError("Unknown categorical terminal relationship.")
        terminal = (
            jnp.mean(schedule.cumulative[-1], axis=0)
            if terminal_probabilities is None
            else jnp.asarray(terminal_probabilities, dtype=schedule.transition.dtype)
        )
        if terminal.shape != (schedule.num_classes,):
            raise ValueError("terminal_probabilities must contain one value per class.")
        if bool(jnp.any(~jnp.isfinite(terminal) | (terminal < 0.0))):
            raise ValueError("terminal_probabilities must be finite and nonnegative.")
        total = jnp.sum(terminal)
        if not bool(jnp.isclose(total, 1.0, rtol=1e-10, atol=1e-12)):
            raise ValueError("terminal_probabilities must sum to one.")
        terminal = terminal / total
        reference_id = terminal_reference_id or canonical_fingerprint(
            {
                "kind": "categorical-terminal-reference",
                "schedule_id": schedule.schedule_id,
                "relationship": terminal_relationship,
                "probabilities": np.asarray(terminal).tolist(),
                "source": (
                    "uniform-clean-pushforward"
                    if terminal_probabilities is None
                    else "explicit"
                ),
            }
        )
        if not reference_id:
            raise ValueError("terminal_reference_id must be non-empty or None.")
        self.schedule = schedule
        self.predictor = predictor
        self.terminal_probabilities = terminal
        self.event_shape = shape
        self.terminal_relationship = terminal_relationship
        self.terminal_reference_id = reference_id
        self.sampler_id = canonical_fingerprint(
            {
                "kind": "categorical-reverse-diffusion",
                "schedule_id": schedule.schedule_id,
                "event_shape": list(shape),
                "terminal_reference_id": reference_id,
            }
        )

    def sample(self, key: Key[Array, ""], sample_shape: Sequence[int], /):
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("sample_shape dimensions must be positive.")
        root, initial_key = jr.split(key)
        logits = jnp.broadcast_to(
            jnp.log(self.terminal_probabilities),
            samples + self.event_shape + (self.schedule.num_classes,),
        )
        initial = jr.categorical(initial_key, logits, axis=-1).astype(jnp.int32)
        timesteps = jnp.arange(
            self.schedule.num_steps - 1, -1, -1, dtype=jnp.int32
        )

        def step(carry, timestep):
            state, current_key = carry
            current_key, model_key, sample_key = jr.split(current_key, 3)
            batch_time = jnp.full(samples, timestep, dtype=jnp.int32)
            clean_logits = jnp.asarray(
                self.predictor(state, batch_time, key=model_key)
            )
            expected = state.shape + (self.schedule.num_classes,)
            if clean_logits.shape != expected:
                raise ValueError("Categorical predictor must return clean-state logits.")
            reverse_probabilities = (
                self.schedule.reverse_probabilities_from_clean_logits(
                    state, clean_logits, batch_time
                )
            )
            next_state = jr.categorical(
                sample_key, jnp.log(reverse_probabilities), axis=-1
            ).astype(jnp.int32)
            return (next_state, current_key), next_state

        (final, _), trajectory = jax.lax.scan(step, (initial, root), timesteps)
        valid = jnp.all(
            (final >= 0) & (final < self.schedule.num_classes),
            axis=tuple(range(len(samples), final.ndim)),
        )
        return CategoricalDiffusionSample(
            final,
            trajectory,
            timesteps,
            valid,
            self.sampler_id,
            self.schedule.schedule_id,
            self.terminal_relationship,
            self.terminal_reference_id,
        )

def categorical_denoising_loss(
    predictor: Callable,
    schedule: CategoricalDiffusionSchedule,
    clean: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    mask: ArrayLike | None = None,
) -> Array:
    state = schedule._validate_state(clean)
    if state.ndim < 1:
        raise ValueError("clean categorical data require a leading sample axis.")
    time_key, noise_key, model_key = jr.split(key, 3)
    time = jr.randint(time_key, (state.shape[0],), 0, schedule.num_steps)
    noisy = schedule.corrupt(state, time, noise_key)
    logits = jnp.asarray(predictor(noisy, time, key=model_key))
    if logits.shape != state.shape + (schedule.num_classes,):
        raise ValueError("Categorical predictor must return one logit per class.")
    losses = -jnp.take_along_axis(
        jax.nn.log_softmax(logits), state[..., None], axis=-1
    )[..., 0]
    active = (
        jnp.ones(state.shape, dtype=bool)
        if mask is None
        else jnp.broadcast_to(jnp.asarray(mask, dtype=bool), state.shape)
    )
    count = jnp.sum(active)
    count = eqx.error_if(count, count <= 0, "Categorical denoising mask is empty.")
    return jnp.sum(jnp.where(active, losses, 0.0)) / count


__all__ = [
    "CategoricalDiffusionSample",
    "CategoricalTerminalRelationship",
    "CategoricalDiffusionSchedule",
    "CategoricalReverseDiffusion",
    "categorical_denoising_loss",
]
